"""Tests for ClassificationMetric."""

import json

import pytest
from deepeval.test_case import LLMTestCase

from coeval.metrics.classification import (
    ClassificationMetric,
    compute_classification_result,
)


@pytest.fixture
def metric() -> ClassificationMetric:
    return ClassificationMetric(labels=["attributable", "not attributable"])


@pytest.fixture
def multiclass_metric() -> ClassificationMetric:
    return ClassificationMetric(labels=["positive", "negative", "neutral"])


@pytest.fixture
def ktas_metric() -> ClassificationMetric:
    return ClassificationMetric(labels=["1", "2", "3", "4", "5"])


def make_test_case(
    actual_output: str,
    expected_output: str,
    context: list[str] | None = None,
) -> LLMTestCase:
    return LLMTestCase(
        input="Test question",
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )


class TestJsonExtraction:
    def test_json_object(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case('{"answer": "attributable"}', "attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "json"
        assert metric._details["predicted"] == "attributable"

    def test_json_with_extra_fields(self, metric: ClassificationMetric) -> None:
        response = '{"answer": "not attributable", "confidence": 0.9}'
        test_case = make_test_case(response, "not attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "json"
        assert metric._details["predicted"] == "not attributable"

    def test_json_wrong_answer(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case('{"answer": "attributable"}', "not attributable")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "json"
        assert metric._details["predicted"] == "attributable"
        assert metric._details["expected"] == "not attributable"


class TestRegexExtraction:
    def test_json_answer_pattern(self, metric: ClassificationMetric) -> None:
        response = 'Here is my analysis: "answer": "attributable"'
        test_case = make_test_case(response, "attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "json_answer"

    def test_quoted_pattern(self, metric: ClassificationMetric) -> None:
        response = 'The classification is "not attributable".'
        test_case = make_test_case(response, "not attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "quoted"

    def test_final_answer_pattern(self) -> None:
        # Use labels that won't match via direct containment
        metric = ClassificationMetric(labels=["option_a", "option_b"])
        response = "Based on my analysis, the answer is option_a"
        test_case = make_test_case(response, "option_a")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "final_answer"

    @pytest.mark.parametrize(
        "response",
        [
            "The conclusion is attributable",
            "My verdict: attributable",
            "Answer: attributable",
        ],
    )
    def test_final_answer_variants(
        self, metric: ClassificationMetric, response: str
    ) -> None:
        test_case = make_test_case(response, "attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "attributable"


class TestExactMatch:
    def test_exact(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("attributable", "attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "exact"

    def test_case_insensitive(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("ATTRIBUTABLE", "attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "exact"

    def test_whitespace_normalized(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("  not   attributable  ", "not attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "exact"


class TestDirectMatch:
    def test_label_in_text(self, metric: ClassificationMetric) -> None:
        response = "This claim is clearly not attributable to the source."
        test_case = make_test_case(response, "not attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["method"] == "direct"

    def test_longest_match_wins(self, metric: ClassificationMetric) -> None:
        # "not attributable" should win over "attributable" (longer match)
        response = "The statement is not attributable"
        test_case = make_test_case(response, "not attributable")
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "not attributable"


class TestNoMatch:
    def test_no_match(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("I cannot determine this.", "attributable")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "no_match"
        assert metric._details["predicted"] is None

    def test_empty_response(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("", "attributable")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric._details["method"] == "empty"
        assert metric._details["predicted"] is None


class TestMulticlass:
    def test_positive(self, multiclass_metric: ClassificationMetric) -> None:
        test_case = make_test_case("positive", "positive")
        multiclass_metric.measure(test_case)

        assert multiclass_metric.score == 1.0
        assert multiclass_metric._details["predicted"] == "positive"

    def test_negative(self, multiclass_metric: ClassificationMetric) -> None:
        test_case = make_test_case('{"answer": "negative"}', "negative")
        multiclass_metric.measure(test_case)

        assert multiclass_metric.score == 1.0

    def test_neutral(self, multiclass_metric: ClassificationMetric) -> None:
        test_case = make_test_case("The sentiment is neutral.", "neutral")
        multiclass_metric.measure(test_case)

        assert multiclass_metric.score == 1.0

    def test_wrong_class(self, multiclass_metric: ClassificationMetric) -> None:
        test_case = make_test_case("positive", "negative")
        multiclass_metric.measure(test_case)

        assert multiclass_metric.score == 0.0
        assert multiclass_metric._details["predicted"] == "positive"
        assert multiclass_metric._details["expected"] == "negative"


class TestKtasJsonExtraction:
    """Tests for KTAS-specific JSON parsing (nested ktas.level)."""

    def test_ktas_nested_json(self, ktas_metric: ClassificationMetric) -> None:
        response = '{"ktas": {"level": 3, "rationale": "..."}}'
        test_case = make_test_case(response, "3")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "json"
        assert ktas_metric._details["predicted"] == "3"

    def test_ktas_nested_json_wrong_level(
        self, ktas_metric: ClassificationMetric
    ) -> None:
        response = '{"ktas": {"level": 2, "rationale": "..."}}'
        test_case = make_test_case(response, "4")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 0.0
        assert ktas_metric._details["method"] == "json"
        assert ktas_metric._details["predicted"] == "2"

    def test_ktas_with_extra_fields(self, ktas_metric: ClassificationMetric) -> None:
        response = json.dumps(
            {
                "ktas": {"level": 1, "rationale": "cardiac arrest"},
                "impression": "...",
                "differential_diagnosis": [],
            }
        )
        test_case = make_test_case(response, "1")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "json"

    def test_answer_takes_priority_over_ktas(
        self, ktas_metric: ClassificationMetric
    ) -> None:
        response = '{"answer": "3", "ktas": {"level": 5}}'
        test_case = make_test_case(response, "3")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["predicted"] == "3"


class TestKtasRegexExtraction:
    """Tests for KTAS-specific regex strategies (malformed/repeated JSON)."""

    def test_json_level_regex(self, ktas_metric: ClassificationMetric) -> None:
        # Malformed JSON — not parseable but regex can extract "level": 3
        response = '{"ktas": {"level": 3, "rationale": "..."'
        test_case = make_test_case(response, "3")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "json_level"

    def test_ktas_level_regex(self, ktas_metric: ClassificationMetric) -> None:
        response = "Based on the symptoms, KTAS Level 2 is appropriate."
        test_case = make_test_case(response, "2")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "ktas_level"

    def test_ktas_without_level_word(self, ktas_metric: ClassificationMetric) -> None:
        response = "I classify this as KTAS 4."
        test_case = make_test_case(response, "4")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "ktas_level"

    def test_level_number_regex(self, ktas_metric: ClassificationMetric) -> None:
        response = "The patient should be triaged as Level 5."
        test_case = make_test_case(response, "5")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["method"] == "level_number"


class TestWordBoundaryMatching:
    """Tests for word-boundary matching preventing false positives."""

    def test_no_false_substring_hit(self, ktas_metric: ClassificationMetric) -> None:
        # Label "1" should NOT match inside "196" or other numbers
        response = "The patient's heart rate is 196 and BP is 80/50."
        test_case = make_test_case(response, "1")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 0.0
        assert ktas_metric._details["method"] == "no_match"

    def test_standalone_digit_matches(self, ktas_metric: ClassificationMetric) -> None:
        response = "My assessment is 3"
        test_case = make_test_case(response, "3")
        ktas_metric.measure(test_case)

        assert ktas_metric.score == 1.0
        assert ktas_metric._details["predicted"] == "3"


class TestLabelsFromContext:
    def test_labels_from_context(self) -> None:
        metric = ClassificationMetric()  # No labels provided
        context = [json.dumps({"labels": ["yes", "no"]})]
        test_case = make_test_case("yes", "yes", context=context)
        metric.measure(test_case)

        assert metric.score == 1.0
        assert metric._details["predicted"] == "yes"

    def test_expected_added_to_labels(self) -> None:
        metric = ClassificationMetric(labels=["yes"])  # Missing "no"
        test_case = make_test_case("no", "no")
        metric.measure(test_case)

        # "no" should be auto-added to valid labels
        assert metric.score == 1.0
        assert metric._details["predicted"] == "no"

    def test_no_context(self) -> None:
        metric = ClassificationMetric()
        test_case = make_test_case("yes", "yes", context=None)
        metric.measure(test_case)

        # Falls back to using expected_label as valid label
        assert metric.score == 1.0
        assert metric._details["predicted"] == "yes"

    def test_invalid_json_in_context(self) -> None:
        metric = ClassificationMetric()
        context = ["not valid json", "{invalid}"]
        test_case = make_test_case("yes", "yes", context=context)
        metric.measure(test_case)

        # Falls back to using expected_label as valid label
        assert metric.score == 1.0
        assert metric._details["predicted"] == "yes"

    def test_json_without_labels_key(self) -> None:
        metric = ClassificationMetric()
        context = [json.dumps({"other_key": "value"})]
        test_case = make_test_case("yes", "yes", context=context)
        metric.measure(test_case)

        # Falls back to using expected_label as valid label
        assert metric.score == 1.0
        assert metric._details["predicted"] == "yes"


class TestComputeClassificationResult:
    def test_correct_prediction(self) -> None:
        result = compute_classification_result(
            response="attributable",
            expected_label="attributable",
            valid_labels=["attributable", "not attributable"],
        )

        assert result.is_correct is True
        assert result.predicted_label == "attributable"
        assert result.expected_label == "attributable"

    def test_incorrect_prediction(self) -> None:
        result = compute_classification_result(
            response="not attributable",
            expected_label="attributable",
            valid_labels=["attributable", "not attributable"],
        )

        assert result.is_correct is False
        assert result.predicted_label == "not attributable"
        assert result.expected_label == "attributable"

    def test_no_valid_labels_uses_expected(self) -> None:
        result = compute_classification_result(
            response="custom_label",
            expected_label="custom_label",
            valid_labels=None,
        )

        assert result.is_correct is True
        assert result.predicted_label == "custom_label"


class TestMetricMetadata:
    def test_stores_details_in_test_case(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("attributable", "attributable")
        metric.measure(test_case)

        assert test_case.additional_metadata is not None
        assert "Classification" in test_case.additional_metadata
        assert test_case.additional_metadata["Classification"]["is_correct"] is True

    def test_reason_on_success(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("attributable", "attributable")
        metric.measure(test_case)

        assert "Predicted 'attributable'" in metric.reason
        assert "expected 'attributable'" in metric.reason

    def test_reason_on_failure(self, metric: ClassificationMetric) -> None:
        test_case = make_test_case("unknown", "attributable")
        metric.measure(test_case)

        assert "Could not extract label" in metric.reason


class TestStrictMode:
    def test_strict_mode_zeroes_below_threshold(self) -> None:
        metric = ClassificationMetric(
            labels=["yes", "no"], threshold=1.0, strict_mode=True
        )
        test_case = make_test_case("yes", "no")
        metric.measure(test_case)

        assert metric.score == 0.0
        assert metric.success is False


class TestAsyncMeasure:
    @pytest.mark.asyncio
    async def test_a_measure_delegates_to_measure(
        self, metric: ClassificationMetric
    ) -> None:
        test_case = make_test_case("attributable", "attributable")
        score = await metric.a_measure(test_case)

        assert score == 1.0
        assert metric.is_successful() is True
        assert metric._details["predicted"] == "attributable"
