"""
Response parsers for LLM outputs.

Uses JSON structured outputs via Pydantic models.
Also provides shared message-extraction helpers used across dataset loaders.
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from coeval.core.schema import AnswerResponse, MCQResponse, ParsedResponse

logger = logging.getLogger(__name__)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_LAST_THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_UNCLOSED_THINK_CONTENT_RE = re.compile(r"<think>(.*)", re.DOTALL)


def extract_think_content(text: str) -> str:
    """Extract the content inside the last ``<think>...</think>`` block.

    Returns an empty string when no think block is present.
    Also handles unclosed ``<think>`` (model hit max_tokens mid-reasoning).
    """
    match = _LAST_THINK_CONTENT_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _UNCLOSED_THINK_CONTENT_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks, keeping only the final answer.

    Think-model outputs (e.g. Tri-21B-Think via vLLM) include a full reasoning
    trace inside <think>…</think> before the actual response. Evaluation judges
    must only see the final answer — not the internal "mental simulation" that
    may contain fake citations or placeholder reasoning.

    Four cases handled:
    1. Normal: <think>…</think> followed by answer → strip block, keep answer.
    2. Think-only output: entire response is inside <think>…</think> with nothing
       after it (common for Tri-21B-Think on short-answer queries) → extract the
       think block's content as the answer rather than returning empty string.
    3. Unclosed <think>: model hit max_tokens mid-reasoning, no </think> emitted
       → extract whatever partial content was written inside the block rather than
       discarding it (applies to any model that hits the token limit mid-think).
    4. No think tags at all (Kimi, GPT, etc.) → returned unchanged.
    """
    after_think = _THINK_BLOCK_RE.sub("", text).strip()
    after_think = _UNCLOSED_THINK_RE.sub("", after_think).strip()

    if after_think:
        return after_think  # case 1 / case 4: answer exists outside think block

    # case 2: closed think block with nothing after it
    match = _LAST_THINK_CONTENT_RE.search(text)
    if match:
        return match.group(1).strip()

    # case 3: unclosed think block — recover partial content rather than return ""
    match = _UNCLOSED_THINK_CONTENT_RE.search(text)
    if match:
        return match.group(1).strip()

    return ""


def extract_system_prompt(messages: list[dict]) -> str | None:
    """Extract the first system message content."""
    for msg in messages:
        if msg.get("role") == "system" and msg.get("content"):
            return msg["content"].strip()
    return None


def extract_patient_prompt(messages: list[dict]) -> str:
    """Extract patient description from user messages."""
    parts = []
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            parts.append(msg["content"].strip())
    return "\n\n".join(parts)


def extract_assistant_json(messages: list[dict]) -> dict | None:
    """Parse the last assistant message as a JSON object.

    Returns None if the assistant message is not valid JSON or not a dict
    (e.g., N689 where assistant is just ``"5"``).
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").strip()
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return None
    return None


def load_rows(data_path: Path) -> list[dict]:
    """Load rows from JSONL or JSON file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.suffix == ".jsonl":
        with open(data_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]


JSON_BLOCK_PATTERN_STR = r"```(?:json)?\s*([\s\S]*?)```"
JSON_OBJECT_PATTERN_STR = r"\{[\s\S]*\}"
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning traces from model output."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


def parse_json_response[T: BaseModel](
    text: str,
    model: type[T],
    strict: bool = False,
) -> T | None:
    """
    Parse JSON response into a Pydantic model.

    Tries multiple extraction strategies:
    1. Direct JSON parse
    2. Extract from ```json``` code blocks
    3. Extract JSON object pattern

    Args:
        text: Raw LLM response text
        model: Pydantic model class to parse into
        strict: If True, raise on parse failure instead of returning None

    Returns:
        Parsed model instance or None if parsing fails
    """

    text = text.strip()

    strategies: list[tuple[str, Callable[[str], str | None]]] = [
        ("direct", lambda t: t),
        ("code_block", lambda t: _extract_pattern(t, JSON_BLOCK_PATTERN_STR, 1)),
        ("object_pattern", lambda t: _extract_pattern(t, JSON_OBJECT_PATTERN_STR, 0)),
    ]

    for strategy_name, extractor in strategies:
        try:
            extracted = extractor(text)
            if extracted:
                data = json.loads(extracted)
                result = model.model_validate(data)
                logger.debug(f"Parsed JSON using {strategy_name} strategy")
                return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Strategy {strategy_name} failed: {e}")
            continue

    if strict:
        raise ValueError(f"Failed to parse JSON response into {model.__name__}")
    return None


def _extract_pattern(text: str, pattern: str, group: int) -> str | None:
    """Extract text matching a regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(group).strip() if group > 0 else match.group(0)
    return None


def parse_json_text(text: str) -> dict | None:
    """Parse raw text into a JSON dict.

    Tries: direct parse → ``\\`\\`\\`json\\`\\`\\`` code block → bare ``{...}`` object.
    Returns ``None`` when no strategy succeeds.
    """
    text = text.strip()
    strategies: list[tuple[str, Callable[[str], str | None]]] = [
        ("direct", lambda t: t),
        ("code_block", lambda t: _extract_pattern(t, JSON_BLOCK_PATTERN_STR, 1)),
        ("object_pattern", lambda t: _extract_pattern(t, JSON_OBJECT_PATTERN_STR, 0)),
    ]
    for _, extractor in strategies:
        try:
            extracted = extractor(text)
            if extracted:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_answer(completion: str) -> str:
    """
    Parse answer from LLM completion (JSON format).

    Args:
        completion: Raw LLM completion text

    Returns:
        Extracted answer string
    """
    if not completion:
        return ""

    json_result = parse_json_response(completion, AnswerResponse)
    if json_result:
        return json_result.answer.strip()

    return completion.strip()


def parse_mcq_answer(completion: str) -> str:
    """
    Parse MCQ answer from LLM completion.

    Optimized for single-letter or short answers.

    Args:
        completion: Raw LLM completion text

    Returns:
        Extracted answer string (typically a letter)
    """
    if not completion:
        return ""

    json_result = parse_json_response(completion, MCQResponse)
    if json_result:
        return json_result.answer.strip()

    return parse_answer(completion)


def parse_structured_response[T: BaseModel](
    completion: str,
    response_model: type[T] | None = None,
) -> ParsedResponse:
    """
    Parse LLM completion into structured ParsedResponse.

    Args:
        completion: Raw LLM completion text
        response_model: Optional Pydantic model for structured parsing

    Returns:
        ParsedResponse with extracted fields
    """
    result = ParsedResponse(raw=completion)
    think_reasoning = extract_think_content(completion)
    completion = strip_think_tags(completion)

    if response_model:
        parsed = parse_json_response(completion, response_model)
        if parsed:
            result.answer = getattr(parsed, "answer", None)
            result.reasoning = think_reasoning or getattr(parsed, "reasoning", None)
            result.parse_method = "json_structured"
            return result

    json_result = parse_json_response(completion, AnswerResponse)
    if json_result:
        result.answer = json_result.answer
        result.reasoning = think_reasoning or json_result.reasoning
        result.parse_method = "json_auto"
        return result

    result.answer = completion.strip()
    result.reasoning = think_reasoning or None
    result.parse_method = "raw_text"
    return result


def extract_prompt_from_messages(messages: list[dict]) -> str:
    """
    Extract the prompt from conversation messages.

    Combines all system and user messages into a single prompt string.
    Excludes assistant messages (which contain the expected output).

    This is useful for datasets that store prompts as a messages array
    (system + user roles) even when the conversation is single-turn.

    Args:
        messages: List of message dicts with "role" and "content" keys.

    Returns:
        Combined prompt string from system and user messages.
    """
    prompt_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user") and content:
            prompt_parts.append(content.strip())

    return "\n\n".join(prompt_parts)


class JSONParser:
    """
    JSON Parser for structured LLM outputs.

    Primary parser for Pydantic-based structured outputs.

    Example:
        >>> parser = JSONParser(response_model=MCQResponse)
        >>> result = parser.parse('{"reasoning": "...", "answer": "B"}')
        >>> result.answer
        'B'
    """

    def __init__(
        self,
        response_model: type[BaseModel] = AnswerResponse,
        strict: bool = False,
    ) -> None:
        """
        Initialize JSON parser.

        Args:
            response_model: Pydantic model for response parsing
            strict: Whether to raise on parse failure
        """
        self.response_model = response_model
        self.strict = strict

    def parse(self, text: str) -> ParsedResponse:
        """
        Parse text into ParsedResponse.

        Args:
            text: Raw LLM response text

        Returns:
            ParsedResponse with extracted fields
        """
        return parse_structured_response(text, self.response_model)

    def parse_answer(self, text: str) -> str | None:
        """
        Extract just the answer field.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted answer string or None
        """
        result = self.parse(text)
        return result.answer
