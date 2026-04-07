"""
Multiple Choice Question (MCQ) accuracy metric.

Non-LLM deterministic evaluation using robust answer extraction.
"""

import json
import re
import unicodedata
from typing import Any

from coeval.core.schema import ClassificationResult
from coeval.core.types import LLMTestCase
from coeval.metrics.base import DeterministicMetric
from coeval.util.parsers import parse_mcq_answer

_OPTION_TERM = r"""(?=$|[\s\]\)\}\.,:;!?\"']|[^\x00-\x7f])"""

# Pattern to extract option text→letter mapping from input (e.g. "A. Yes\nB. No\nC. Maybe")
_OPTION_LINE_PATTERN = re.compile(
    r"^\s*\(?([A-Za-z0-9])\)?[\.\):\s]+(.+)$", re.MULTILINE
)

STRICT_SINGLE_OPTION_PATTERN = re.compile(
    r"""^\s*
    (?:\*\*|`|"|')?
    \(?\s*([A-Za-z0-9])\s*\)?
    (?:\*\*|`|"|')?
    [\).:]?\s*
    $""",
    re.VERBOSE,
)

TOKEN_PATTERN_STRICT = re.compile(
    rf"""(?:\*\*|`|"|')?
    \(?\s*([A-Za-z0-9])\s*\)?
    (?:\*\*|`|"|')?
    {_OPTION_TERM}""",
    re.VERBOSE,
)

LEADING_OPTION_PATTERN_STRICT = re.compile(
    r"""^\s*
    (?:\*\*|`|"|')?
    \(?\s*([A-Za-z0-9])\s*\)?
    (?:\*\*|`|"|')?
    (?:[\).:\s]|$)""",
    re.VERBOSE,
)

ANCHOR_PATTERN_STRICT = re.compile(
    rf"""
    (?:
        (?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s+(?:is|would\s+be|should\s+be)
        |(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*[:=]
        |final\s+(?:answer\s+)?(?:is|:|=)?
        |i\s+(?:would\s+)?(?:choose|select|pick|go\s+with)
        |(?:it(?:'s|\s+is)\s+)
        |(?:so\s+)
        |(?:therefore(?:,)?\s+)
    )
    \s*[:=\-]?\s*
    (?:\*\*|`|"|')?
    \(?\s*([A-Za-z0-9])\s*\)?
    (?:\*\*|`|"|')?
    {_OPTION_TERM}
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches markdown bold MCQ answers: **A:**, **B.**, **(C)**, **D**
BOLD_OPTION_PATTERN = re.compile(
    r"""\*\*\s*\(?\s*([A-Ja-j])\s*\)?\s*(?:[\.:\)]|\*\*)""",
    re.VERBOSE,
)

# Korean answer anchors: "정답은 X", "정답: X", "답은 X입니다"
KOREAN_ANCHOR_PATTERN = re.compile(
    rf"""
    정답\s*(?:은\s*)?[:=\-]?\s*
    (?:\*\*|`|"|')?
    \(?\s*([A-Za-z0-9])\s*\)?
    (?:\*\*|`|"|')?
    {_OPTION_TERM}
    """,
    re.VERBOSE,
)

# Anchor pattern that captures multi-word option TEXT (e.g., "the answer is yes")
ANCHOR_TEXT_PATTERN = re.compile(
    r"""
    (?:
        (?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s+(?:is|would\s+be|should\s+be)
        |(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*[:=]
        |final\s+(?:answer\s+)?(?:is|:|=)?
        |i\s+(?:would\s+)?(?:choose|select|pick|go\s+with)
    )
    \s*[:=\-]?\s*
    (?:\*\*|`|"|')?
    ([a-z][a-z\s]{0,30}?)
    (?:\*\*|`|"|')?
    (?=$|[\s\]\)\}\.,:;!?\"']|[^\x00-\x7f])
    """,
    re.IGNORECASE | re.VERBOSE,
)

NEGATION_WORDS = re.compile(
    r"\b(?:not|n't|cannot|can't|incorrect|wrong|false|eliminate|rule\s*out)\b",
    re.IGNORECASE,
)


def _nfkc_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _norm_letter(letter: str) -> str | None:
    if not letter or len(letter) != 1:
        return None
    c = letter.upper()
    if c.isalpha() or c.isdigit():
        return c
    return None


def _build_option_text_map(input_text: str | None) -> dict[str, str]:
    """
    Parse option text→letter mapping from input prompt.

    Given input like:
        A. Yes
        B. No
        C. Maybe
    Returns: {"yes": "A", "no": "B", "maybe": "C"}
    """
    if not input_text:
        return {}
    mapping: dict[str, str] = {}
    for m in _OPTION_LINE_PATTERN.finditer(input_text):
        letter = m.group(1).upper()
        text = m.group(2).strip().rstrip(".").strip()
        if text:
            mapping[_nfkc_casefold(text)] = letter
    return mapping


def _resolve_text_answer(
    raw_text: str,
    option_map: dict[str, str],
) -> tuple[str | None, str]:
    """
    Try to match anchored text answers (e.g., 'the answer is yes') against option_map.
    Returns (letter, method) or (None, '').
    """
    if not option_map:
        return None, ""
    norm = _nfkc_casefold(raw_text)
    matches = list(ANCHOR_TEXT_PATTERN.finditer(norm))
    for match in reversed(matches):
        captured = match.group(1).strip()
        if captured in option_map:
            return option_map[captured], "anchored_text"
    return None, ""


def _tail_region(text: str, max_tokens: int = 64) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[-max_tokens:])


def _try_parse_json_obj(text: str) -> dict[str, Any] | None:
    """
    Robustly attempt to parse a JSON object from a string that *may* contain
    surrounding text (some models violate strict JSON-only).
    """
    if not text:
        return None

    s = text.strip()
    decoder = json.JSONDecoder()

    if s.startswith("{"):
        try:
            obj, _ = decoder.raw_decode(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue

    return None


def _extract_option_from_answer_field(answer_field: Any) -> str | None:
    """
    Extract a single option letter/digit from JSON 'answer' value.
    Accepts strings like: "C", "C.", "(C)", "\"C\"", "[C]", "Option C".
    """
    if answer_field is None:
        return None

    ans = str(answer_field).strip()
    if not ans:
        return None

    m = STRICT_SINGLE_OPTION_PATTERN.match(ans)
    if m:
        return _norm_letter(m.group(1))

    tokens = list(TOKEN_PATTERN_STRICT.finditer(ans))
    if tokens:
        return _norm_letter(tokens[0].group(1))

    return None


def _negated_locally(
    text: str, letter_match: re.Match, lookback_chars: int = 32
) -> bool:
    """
    Local negation check: only inspect a small window *before* the matched letter,
    within the same clause. This prevents "answer is C, not B" from negating C.
    """
    i = letter_match.start(1)
    clause_start = max(
        text.rfind(".", 0, i),
        text.rfind("!", 0, i),
        text.rfind("?", 0, i),
        text.rfind("\n", 0, i),
    )
    clause_start = clause_start + 1
    window_start = max(clause_start, i - lookback_chars)
    prefix = text[window_start:i]

    tail_tokens = prefix.split()[-4:]
    tail = " ".join(tail_tokens)

    return bool(NEGATION_WORDS.search(tail))


def _extract_predicted_option(
    raw_text: str,
    option_map: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """
    Extract a SINGLE predicted option from the response.
    Returns (predicted_option, method). predicted_option is normalized (uppercase).

    Args:
        raw_text: The LLM response text.
        option_map: Optional mapping of option text → letter
                    (e.g., {"yes": "A", "no": "B", "maybe": "C"}).
                    Used to resolve text-based answers like "the answer is yes".
    """
    if not raw_text or not raw_text.strip():
        return None, "none"

    raw = raw_text.strip()

    obj = _try_parse_json_obj(raw)
    if isinstance(obj, dict) and "answer" in obj:
        answer_val = obj.get("answer")
        # Try text-based resolution first (e.g., "yes" → "A")
        if option_map and answer_val:
            ans_norm = _nfkc_casefold(str(answer_val).strip())
            if ans_norm in option_map:
                return option_map[ans_norm], "json_answer_text"
        predicted = _extract_option_from_answer_field(answer_val)
        if predicted:
            return predicted, "json_answer"

    try:
        parsed = parse_mcq_answer(raw)
    except Exception:
        parsed = raw

    if parsed and parsed.strip():
        p = parsed.strip()
        if _norm_letter(p) == p.upper() and len(p) == 1:
            return p.upper(), "direct_answer"

        # Try text-based resolution for direct parsed answer
        if option_map:
            p_norm = _nfkc_casefold(p)
            if p_norm in option_map:
                return option_map[p_norm], "direct_answer_text"

        m = LEADING_OPTION_PATTERN_STRICT.match(p)
        if m:
            predicted = _norm_letter(m.group(1))
            if predicted:
                return predicted, "leading_token"

    norm = _nfkc_casefold(raw)
    anchored = list(ANCHOR_PATTERN_STRICT.finditer(norm))
    for match in reversed(anchored):
        if _negated_locally(norm, match):
            continue
        predicted = _norm_letter(match.group(1))
        if predicted:
            return predicted, "anchored_token"

    # Text-based anchor matching (e.g., "the answer is yes" → "A")
    if option_map:
        text_pred, text_method = _resolve_text_answer(raw, option_map)
        if text_pred:
            return text_pred, text_method

    # Korean anchor: "정답은 X입니다", "정답: X"
    kr_anchored = list(KOREAN_ANCHOR_PATTERN.finditer(norm))
    for match in reversed(kr_anchored):
        predicted = _norm_letter(match.group(1))
        if predicted:
            return predicted, "korean_anchor"

    bold_matches = list(BOLD_OPTION_PATTERN.finditer(raw))
    for match in reversed(bold_matches):
        predicted = _norm_letter(match.group(1))
        if predicted:
            return predicted, "bold_token"

    tail = _tail_region(norm)
    tokens = list(TOKEN_PATTERN_STRICT.finditer(tail))
    for match in reversed(tokens):
        if _negated_locally(tail, match):
            continue
        predicted = _norm_letter(match.group(1))
        if predicted:
            return predicted, "last_token"

    return None, "no_match"


def multiple_choice_accuracy(
    llm_answer: str,
    answer_letter: str,
    answer_text: str | None = None,
    input_text: str | None = None,
) -> ClassificationResult:
    """
    Deterministic MCQ grading with robust extraction.

    Key features:
      - Extract ONE predicted option (prefer JSON 'answer'), then compare to ground truth.
      - No "search for correctness" leakage.
      - Boundary-safe tokenization prevents "because" => "B".
      - Local negation avoids suppressing correct choice in "C, not B".
      - Text-based option resolution: when options are text (e.g., Yes/No/Maybe),
        resolves "the answer is yes" → letter "A" using option map from input.

    Fallback: If no option letter is extracted but answer_text is provided,
    check if the full answer text appears in the response.
    """
    answer_norm = _norm_letter(answer_letter)
    if answer_norm is None:
        return ClassificationResult(
            is_correct=False,
            method="invalid_answer",
            predicted_label=None,
            expected_label=answer_letter,
        )

    option_map = _build_option_text_map(input_text)
    predicted, method = _extract_predicted_option(llm_answer, option_map)

    if predicted is not None:
        is_correct = predicted == answer_norm
        return ClassificationResult(
            is_correct=is_correct,
            method=method,
            predicted_label=predicted,
            expected_label=answer_norm,
        )

    if answer_text:
        llm_norm = _nfkc_casefold(llm_answer)
        answer_text_norm = _nfkc_casefold(answer_text)
        if answer_text_norm in llm_norm:
            return ClassificationResult(
                is_correct=True,
                method="answer_text",
                predicted_label=answer_text,
                expected_label=answer_norm,
            )

    return ClassificationResult(
        is_correct=False,
        method=method,
        predicted_label=None,
        expected_label=answer_norm,
    )


class MCQAccuracyMetric(DeterministicMetric):
    """
    Multiple Choice Question accuracy metric.

    Non-LLM deterministic evaluation that extracts predicted options
    from LLM responses using multiple strategies:
    - JSON answer field parsing
    - Leading token detection
    - Anchored phrase matching ("the answer is X")
    - Last token fallback

    Usage:
        metric = MCQAccuracyMetric()
        test_case = LLMTestCase(
            input="What is 2+2?",
            actual_output="The answer is C",
            expected_output="C",
        )
        metric.measure(test_case)
        print(metric.score, metric.success, metric.reason)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        strict_mode: bool = False,
        **kwargs: Any,
    ):
        """Initialize MCQ accuracy metric with threshold and strict mode."""
        super().__init__(
            threshold=threshold,
            strict_mode=strict_mode,
            **kwargs,
        )

    @property
    def __name__(self) -> str:
        return "MCQ Accuracy"

    def measure(
        self,
        test_case: LLMTestCase,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> float:
        """
        Evaluate MCQ accuracy.

        Uses:
        - test_case.actual_output: LLM's response
        - test_case.expected_output: Correct answer letter (e.g., "C")
        - test_case.context: Optional answer text for fallback matching

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        try:
            answer_text = test_case.context[0] if test_case.context else None

            result = multiple_choice_accuracy(
                llm_answer=test_case.actual_output or "",
                answer_letter=test_case.expected_output or "",
                answer_text=answer_text,
                input_text=test_case.input,
            )

            reason = (
                f"Predicted '{result.predicted_label}' via {result.method}, "
                f"expected '{result.expected_label}'"
                if result.predicted_label
                else f"Could not extract answer, expected '{result.expected_label}'"
            )

            return self._set_result(
                test_case,
                result.is_correct,
                reason,
                {
                    "is_correct": result.is_correct,
                    "method": result.method,
                    "predicted": result.predicted_label,
                    "expected": result.expected_label,
                },
            )

        except Exception as e:
            self.error = str(e)
            self.reason = f"Evaluation failed: {e}"
            raise
