"""Tests for MCQAccuracyMetric."""

import pytest
from deepeval.test_case import LLMTestCase

from coeval.metrics.mcq_accuracy import MCQAccuracyMetric


@pytest.fixture
def metric() -> MCQAccuracyMetric:
    return MCQAccuracyMetric()


def make_test_case(
    actual_output: str,
    expected_output: str,
    answer_text: str | None = None,
) -> LLMTestCase:
    return LLMTestCase(
        input="Test question",
        actual_output=actual_output,
        expected_output=expected_output,
        context=[answer_text] if answer_text else None,
    )


class TestDirectAnswer:
    def test_correct(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("A", "A")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric.is_successful() is True
        assert metric._details["method"] == "direct_answer"
        assert metric._details["predicted"] == "A"

    def test_lowercase(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("b", "B")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "direct_answer"
        assert metric._details["predicted"] == "B"

    def test_wrong(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("A", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric.is_successful() is False
        assert metric._details["method"] == "direct_answer"
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == "B"


class TestLeadingToken:
    def test_correct(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("B. The mitochondria is the powerhouse", "B")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "leading_token"
        assert metric._details["predicted"] == "B"

    def test_wrong(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("C. This is wrong", "D")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "leading_token"
        assert metric._details["predicted"] == "C"
        assert metric._details["expected"] == "D"


class TestAnchoredToken:
    def test_correct(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("After consideration, the answer is C", "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "anchored_token"
        assert metric._details["predicted"] == "C"

    @pytest.mark.parametrize(
        "response",
        [
            "The correct answer is D",
            "Therefore, D",
            "So D",
            "The answer would be D",
        ],
    )
    def test_variants(self, metric: MCQAccuracyMetric, response: str) -> None:
        test_case = make_test_case(response, "D")
        metric.measure(test_case)

        assert metric.score == 1.0, f"Failed for: {response}"
        assert metric._details["predicted"] == "D"

    def test_wrong(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("The answer is A", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "anchored_token"
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == "B"


class TestLastToken:
    def test_correct(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("Let me think... considering all factors... A", "A")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "last_token"
        assert metric._details["predicted"] == "A"

    def test_wrong(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("After analysis, probably A", "C")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == "C"


class TestJsonAnswer:
    def test_correct(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case('{"answer": "C", "rationale": "..."}', "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "json_answer"
        assert metric._details["predicted"] == "C"

    def test_with_surrounding_text(self, metric: MCQAccuracyMetric) -> None:
        response = 'Here is my response: {"answer": "B"} based on analysis.'
        test_case = make_test_case(response, "B")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "json_answer"
        assert metric._details["predicted"] == "B"

    def test_wrong(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case('{"answer": "A"}', "C")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "json_answer"
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == "C"

    @pytest.mark.parametrize(
        "response",
        [
            '{"answer": "(D)"}',
            '{"answer": "D."}',
            '{"answer": "**D**"}',
        ],
    )
    def test_with_formatting(self, metric: MCQAccuracyMetric, response: str) -> None:
        test_case = make_test_case(response, "D")
        metric.measure(test_case)

        assert metric.score == 1.0, f"Failed for: {response}"
        assert metric._details["predicted"] == "D"


class TestAnswerTextFallback:
    def test_correct_when_no_letter(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case(
            "...패혈증...",
            "B",
            answer_text="패혈증",
        )
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "answer_text"
        assert metric._details["predicted"] == "패혈증"

    def test_not_used_when_letter_found(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case(
            "The answer is A. Sepsis is the diagnosis.",
            "B",
            answer_text="Sepsis",
        )
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "anchored_token"
        assert metric._details["predicted"] == "A"

    def test_no_match(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case(
            "...",
            "B",
            answer_text="Pneumonia",
        )
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "no_match"
        assert metric._details["predicted"] is None


class TestEdgeCases:
    def test_empty_response(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "none"
        assert metric._details["predicted"] is None
        assert metric._details["expected"] == "B"

    def test_whitespace_only_response(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("   \n\t  ", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "none"
        assert metric._details["predicted"] is None

    def test_irrelevant_letter_extraction(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("Sorry, unclear question", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["predicted"] is not None

    def test_invalid_answer_letter(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("A", "invalid")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "invalid_answer"
        assert metric._details["predicted"] is None


class TestNegation:
    def test_skips_nearby_negated_options(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("It's not C. The answer is D", "D")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "D"

    def test_eliminate_pattern(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("We can eliminate C. The answer is D", "D")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "D"

    def test_local_window_only(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("That's not A. The answer is B", "B")
        metric.measure(test_case)

        assert metric._details["predicted"] == "B"
        assert metric.score == 1.0


class TestFormatting:
    def test_markdown_bold(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("The answer is **B**", "B")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "B"

    def test_backtick(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("The answer is `C`", "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "C"

    def test_parenthesized(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("(A)", "A")
        metric.measure(test_case)

        assert metric.score == 1.0

    def test_with_period(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("A.", "A")
        metric.measure(test_case)

        assert metric.score == 1.0


class TestChainOfThought:
    def test_then_answer(self, metric: MCQAccuracyMetric) -> None:
        cot_response = """
        Let me analyze this step by step.
        First, option A suggests X which doesn't fit.
        Option B proposes Y which is closer.
        Option C indicates Z which aligns with the evidence.
        Therefore, C is the correct answer.
        """
        test_case = make_test_case(cot_response, "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "C"


class TestNumericOptions:
    def test_numeric(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("The answer is 3", "3")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "3"


class TestMockInference:
    @pytest.mark.parametrize(
        "correct_answer,expected_match",
        [("A", True), ("B", False), ("C", False), ("D", False)],
    )
    def test_always_returns_a(
        self,
        metric: MCQAccuracyMetric,
        correct_answer: str,
        expected_match: bool,
    ) -> None:
        test_case = make_test_case("A", correct_answer)
        metric.measure(test_case)

        assert metric.is_successful() is expected_match
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == correct_answer


class TestLeadingIExtraction:
    def test_extracts_i_not_d(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("I would choose D", "D")
        metric.measure(test_case)

        assert metric._details["predicted"] == "I"
        assert metric.score == 0.0


class TestKoreanAnchor:
    """Tests for Korean answer patterns: 정답은 X입니다, 정답: X, etc."""

    def test_correct_jeongdab(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("분석 결과... 정답은 C입니다.", "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "korean_anchor"
        assert metric._details["predicted"] == "C"

    def test_wrong_jeongdab(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("정답은 A입니다.", "B")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "korean_anchor"
        assert metric._details["predicted"] == "A"
        assert metric._details["expected"] == "B"

    def test_last_anchor_wins(self, metric: MCQAccuracyMetric) -> None:
        """When multiple 정답 mentions, the last one is used."""
        response = "정답은 A가 아닙니다.\n\n따라서 정답은 D입니다."
        test_case = make_test_case(response, "D")
        metric.measure(test_case)

        assert metric._details["predicted"] == "D"
        assert metric.score == 1.0

    def test_with_bold_formatting(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("정답은 **E**입니다.", "E")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "E"

    def test_with_colon(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("정답: B", "B")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "B"

    def test_not_confused_by_medical_abbreviations(
        self, metric: MCQAccuracyMetric
    ) -> None:
        """Should not pick 'T' from aPTT — should use 정답 anchor instead."""
        response = (
            "activated partial thromboplastin time(aPTT)입니다.\n\n정답은 E입니다."
        )
        test_case = make_test_case(response, "E")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "E"

    def test_lowercase_letter_in_korean(self, metric: MCQAccuracyMetric) -> None:
        """Model may output lowercase: 정답은 e입니다."""
        test_case = make_test_case("정답은 e입니다.", "E")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "E"

    def test_parenthesized_in_korean(self, metric: MCQAccuracyMetric) -> None:
        test_case = make_test_case("정답은 (C)입니다.", "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "C"


class TestNonAsciiBoundary:
    """Tests for _OPTION_TERM handling non-ASCII chars as word boundaries."""

    def test_letter_followed_by_korean(self, metric: MCQAccuracyMetric) -> None:
        """A letter followed by Korean char should be recognized as a valid token."""
        test_case = make_test_case("따라서 올바른 답은 A입니다", "A")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "A"

    def test_no_false_match_on_compound(self, metric: MCQAccuracyMetric) -> None:
        """'B형간염' should not cause 'B' to win over the 정답 anchor."""
        test_case = make_test_case("B형간염 환자에서 정답은 C입니다.", "C")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "C"


class TestAsyncMeasure:
    @pytest.mark.asyncio
    async def test_a_measure_delegates_to_measure(
        self, metric: MCQAccuracyMetric
    ) -> None:
        test_case = make_test_case("A", "A")
        score = await metric.a_measure(test_case)

        assert score == 1.0
        assert metric.is_successful() is True
        assert metric._details["predicted"] == "A"
