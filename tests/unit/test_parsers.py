"""Tests for response parsers and message-extraction helpers."""

import json
from pathlib import Path

import pytest

from coeval.core.schema import AnswerResponse, MCQResponse, ParsedResponse
from coeval.util.parsers import (
    JSONParser,
    extract_assistant_json,
    extract_patient_prompt,
    extract_prompt_from_messages,
    extract_system_prompt,
    load_rows,
    parse_answer,
    parse_json_response,
    parse_json_text,
    parse_mcq_answer,
    parse_structured_response,
    strip_think_blocks,
)


class TestExtractSystemPrompt:
    def test_returns_system_content(self) -> None:
        msgs = [{"role": "system", "content": "  You are helpful  "}]
        assert extract_system_prompt(msgs) == "You are helpful"

    def test_returns_first_system(self) -> None:
        msgs = [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
        ]
        assert extract_system_prompt(msgs) == "First"

    def test_no_system_returns_none(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        assert extract_system_prompt(msgs) is None

    def test_empty_content_returns_none(self) -> None:
        msgs = [{"role": "system", "content": ""}]
        assert extract_system_prompt(msgs) is None

    def test_empty_list(self) -> None:
        assert extract_system_prompt([]) is None


class TestExtractPatientPrompt:
    def test_single_user_message(self) -> None:
        msgs = [{"role": "user", "content": "Patient data"}]
        assert extract_patient_prompt(msgs) == "Patient data"

    def test_multi_user_messages_joined(self) -> None:
        msgs = [
            {"role": "user", "content": "Part 1"},
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": "Part 2"},
        ]
        assert extract_patient_prompt(msgs) == "Part 1\n\nPart 2"

    def test_no_user_returns_empty(self) -> None:
        msgs = [{"role": "system", "content": "sys"}]
        assert extract_patient_prompt(msgs) == ""

    def test_empty_list(self) -> None:
        assert extract_patient_prompt([]) == ""

    def test_strips_whitespace(self) -> None:
        msgs = [{"role": "user", "content": "  trimmed  "}]
        assert extract_patient_prompt(msgs) == "trimmed"


class TestExtractAssistantJson:
    def test_valid_json_dict(self) -> None:
        msgs = [{"role": "assistant", "content": '{"key": "value"}'}]
        assert extract_assistant_json(msgs) == {"key": "value"}

    def test_non_dict_json_returns_none(self) -> None:
        msgs = [{"role": "assistant", "content": '"just a string"'}]
        assert extract_assistant_json(msgs) is None

    def test_invalid_json_returns_none(self) -> None:
        msgs = [{"role": "assistant", "content": "not json"}]
        assert extract_assistant_json(msgs) is None

    def test_no_assistant_returns_none(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        assert extract_assistant_json(msgs) is None

    def test_uses_last_assistant(self) -> None:
        msgs = [
            {"role": "assistant", "content": '{"a": 1}'},
            {"role": "assistant", "content": '{"b": 2}'},
        ]
        assert extract_assistant_json(msgs) == {"b": 2}

    def test_plain_number_returns_none(self) -> None:
        """N689-style where assistant is just '5'."""
        msgs = [{"role": "assistant", "content": "5"}]
        assert extract_assistant_json(msgs) is None


class TestLoadRows:
    def test_jsonl(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.jsonl"
        fp.write_text('{"a":1}\n{"b":2}\n')
        rows = load_rows(fp)
        assert len(rows) == 2
        assert rows[0] == {"a": 1}

    def test_jsonl_blank_lines_skipped(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.jsonl"
        fp.write_text('{"a":1}\n\n{"b":2}\n\n')
        rows = load_rows(fp)
        assert len(rows) == 2

    def test_json_array(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.json"
        fp.write_text(json.dumps([{"a": 1}, {"b": 2}]))
        rows = load_rows(fp)
        assert len(rows) == 2

    def test_json_single_object(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.json"
        fp.write_text(json.dumps({"a": 1}))
        rows = load_rows(fp)
        assert len(rows) == 1
        assert rows[0] == {"a": 1}

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_rows(Path("/nonexistent/path.jsonl"))


class TestParseJsonResponse:
    def test_direct_json(self) -> None:
        text = '{"reasoning": "think", "answer": "B"}'
        result = parse_json_response(text, MCQResponse)
        assert result is not None
        assert result.answer == "B"

    def test_code_block(self) -> None:
        text = 'Here is the answer:\n```json\n{"answer": "C"}\n```'
        result = parse_json_response(text, AnswerResponse)
        assert result is not None
        assert result.answer == "C"

    def test_object_pattern(self) -> None:
        text = 'Some preamble {"answer": "D"} trailing text'
        result = parse_json_response(text, AnswerResponse)
        assert result is not None
        assert result.answer == "D"

    def test_no_json_returns_none(self) -> None:
        result = parse_json_response("plain text", AnswerResponse)
        assert result is None

    def test_strict_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_json_response("not json", AnswerResponse, strict=True)

    def test_invalid_model_returns_none(self) -> None:
        text = '{"wrong_field": "value"}'
        result = parse_json_response(text, MCQResponse)
        assert result is None


class TestParseJsonText:
    def test_direct(self) -> None:
        assert parse_json_text('{"a": 1}') == {"a": 1}

    def test_code_block(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert parse_json_text(text) == {"a": 1}

    def test_embedded_object(self) -> None:
        text = 'prefix {"a": 1} suffix'
        assert parse_json_text(text) == {"a": 1}

    def test_non_dict_returns_none(self) -> None:
        assert parse_json_text("[1, 2, 3]") is None

    def test_no_json_returns_none(self) -> None:
        assert parse_json_text("plain text") is None

    def test_empty_returns_none(self) -> None:
        assert parse_json_text("") is None


class TestParseAnswer:
    def test_json_answer(self) -> None:
        text = '{"answer": "The answer is 42"}'
        assert parse_answer(text) == "The answer is 42"

    def test_plain_text_fallback(self) -> None:
        assert parse_answer("Just some text") == "Just some text"

    def test_empty_returns_empty(self) -> None:
        assert parse_answer("") == ""


class TestParseMcqAnswer:
    def test_json_mcq(self) -> None:
        text = '{"reasoning": "because", "answer": "B"}'
        assert parse_mcq_answer(text) == "B"

    def test_plain_letter_fallback(self) -> None:
        result = parse_mcq_answer("I think the answer is C")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_returns_empty(self) -> None:
        assert parse_mcq_answer("") == ""


class TestParseStructuredResponse:
    def test_json_auto(self) -> None:
        text = '{"reasoning": "step1", "answer": "yes"}'
        result = parse_structured_response(text)
        assert isinstance(result, ParsedResponse)
        assert result.answer == "yes"
        assert result.reasoning == "step1"
        assert result.parse_method == "json_auto"

    def test_raw_text_fallback(self) -> None:
        result = parse_structured_response("just plain text")
        assert result.answer == "just plain text"
        assert result.parse_method == "raw_text"

    def test_custom_model(self) -> None:
        text = '{"reasoning": "r", "answer": "A"}'
        result = parse_structured_response(text, MCQResponse)
        assert result.answer == "A"
        assert result.parse_method == "json_structured"

    def test_raw_preserved(self) -> None:
        text = '{"answer": "yes"}'
        result = parse_structured_response(text)
        assert result.raw == text


class TestExtractPromptFromMessages:
    def test_system_and_user(self) -> None:
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
        ]
        result = extract_prompt_from_messages(msgs)
        assert "System prompt" in result
        assert "User query" in result

    def test_excludes_assistant(self) -> None:
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = extract_prompt_from_messages(msgs)
        assert "A" not in result
        assert "Q" in result

    def test_empty_list(self) -> None:
        assert extract_prompt_from_messages([]) == ""

    def test_empty_content_skipped(self) -> None:
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real content"},
        ]
        assert extract_prompt_from_messages(msgs) == "Real content"


class TestJSONParser:
    def test_parse_returns_parsed_response(self) -> None:
        parser = JSONParser()
        result = parser.parse('{"answer": "yes"}')
        assert isinstance(result, ParsedResponse)
        assert result.answer == "yes"

    def test_parse_answer_returns_string(self) -> None:
        parser = JSONParser()
        assert parser.parse_answer('{"answer": "B"}') == "B"

    def test_parse_answer_plain_text(self) -> None:
        parser = JSONParser()
        result = parser.parse_answer("plain text")
        assert result == "plain text"

    def test_custom_model(self) -> None:
        parser = JSONParser(response_model=MCQResponse)
        result = parser.parse('{"reasoning": "r", "answer": "C"}')
        assert result.answer == "C"


class TestStripThinkBlocks:
    """Tests for strip_think_blocks() — four documented cases."""

    def test_case1_normal_think_then_answer(self) -> None:
        """Case 1: <think>…</think> followed by answer → keep only answer."""
        text = "<think>Let me reason about this.</think>The answer is 42."
        assert strip_think_blocks(text) == "The answer is 42."

    def test_case1_whitespace_around_answer(self) -> None:
        """Case 1: whitespace between </think> and answer is stripped."""
        text = "<think>reasoning</think>\n\nThe final answer."
        assert strip_think_blocks(text) == "The final answer."

    def test_case1_multiple_think_blocks(self) -> None:
        """Case 1: multiple think blocks are all stripped."""
        text = "<think>step 1</think> intermediate <think>step 2</think>Final."
        assert strip_think_blocks(text) == "intermediate Final."

    def test_case2_think_only_closed(self) -> None:
        """Case 2: entire output is inside <think>…</think> → extract content."""
        text = "<think>The answer is aspirin.</think>"
        assert strip_think_blocks(text) == "The answer is aspirin."

    def test_case2_think_only_with_whitespace(self) -> None:
        """Case 2: content inside think block is stripped of leading/trailing whitespace."""
        text = "<think>  answer here  </think>"
        assert strip_think_blocks(text) == "answer here"

    def test_case3_unclosed_think(self) -> None:
        """Case 3: model hit max_tokens, no </think> — recover partial content."""
        text = "<think>I started reasoning but ran out of tok"
        assert strip_think_blocks(text) == "I started reasoning but ran out of tok"

    def test_case3_unclosed_think_multiline(self) -> None:
        """Case 3: unclosed think with newlines — full partial content recovered."""
        text = "<think>\nLine one\nLine two\nLine thr"
        assert strip_think_blocks(text) == "Line one\nLine two\nLine thr"

    def test_case4_no_think_tags(self) -> None:
        """Case 4: no think tags → input returned unchanged."""
        text = "This is a normal model output with no think blocks."
        assert strip_think_blocks(text) == text

    def test_case4_empty_string(self) -> None:
        """Case 4: empty string → empty string returned."""
        assert strip_think_blocks("") == ""
