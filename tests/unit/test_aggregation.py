"""Tests for score aggregation functions."""

import pytest
from deepeval.test_case import LLMTestCase

from coeval.core.schema import EvalResult, MetricResult
from coeval.util.aggregation import (
    UNPARSEABLE_LABEL,
    _filter_results,
    avg_aggregator,
    clipped_avg_aggregator,
    composite_aggregator,
    f1_aggregator,
    ordinal_classification_aggregator,
    weighted_avg_aggregator,
)


def make_score_result(sample_id: int, metric_name: str, score: float) -> EvalResult:
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(input="test", actual_output=""),
        metrics=[MetricResult(name=metric_name, score=score, passed=score >= 0.5)],
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )


def make_eval_result(
    sample_id: int,
    metric_name: str,
    score: float,
    predicted: str | None,
    expected: str,
) -> EvalResult:
    """Create a mock EvalResult with classification details."""
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(
            input="test",
            actual_output=predicted or "",
            expected_output=expected,
        ),
        metrics=[
            MetricResult(
                name=metric_name,
                score=score,
                passed=score >= 0.5,
                reason="test",
                details={
                    "predicted": predicted,
                    "expected": expected,
                    "is_correct": predicted == expected,
                },
            )
        ],
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )


def make_multi_metric_result(
    sample_id: int,
    metrics: list[tuple[str, float, str | None, str]],
) -> EvalResult:
    """Create an EvalResult with multiple metrics.

    Args:
        metrics: List of (name, score, predicted, expected) tuples.
    """
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(
            input="test",
            actual_output=metrics[0][2] or "",
            expected_output=metrics[0][3],
        ),
        metrics=[
            MetricResult(
                name=name,
                score=score,
                passed=score >= 0.5,
                reason="test",
                details={
                    "predicted": predicted,
                    "expected": expected,
                    "is_correct": predicted == expected,
                },
            )
            for name, score, predicted, expected in metrics
        ],
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )


class TestAvgAggregator:
    def test_empty_results(self) -> None:
        result = avg_aggregator([])
        assert result.metric_scores == {}

    def test_single_result(self) -> None:
        results = [make_eval_result(0, "test", 1.0, "A", "A")]
        result = avg_aggregator(results)

        assert "test" in result.metric_scores
        assert result.metric_scores["test"].score == 1.0

    def test_multiple_results(self) -> None:
        results = [
            make_eval_result(0, "test", 1.0, "A", "A"),
            make_eval_result(1, "test", 0.0, "B", "A"),
            make_eval_result(2, "test", 1.0, "A", "A"),
        ]
        result = avg_aggregator(results)
        assert result.metric_scores["test"].score == pytest.approx(2 / 3)


class TestWeightedAvgAggregator:
    def test_empty_results(self) -> None:
        result = weighted_avg_aggregator([])
        assert result.metric_scores == {}

    def test_uniform_weights_match_avg(self) -> None:
        """With default weight=1.0, should match avg_aggregator."""
        results = [
            make_eval_result(0, "test", 1.0, "A", "A"),
            make_eval_result(1, "test", 0.0, "B", "A"),
        ]
        avg = avg_aggregator(results)
        weighted = weighted_avg_aggregator(results)
        assert weighted.metric_scores["test"].score == pytest.approx(
            avg.metric_scores["test"].score
        )

    def test_stores_numerator_denominator(self) -> None:
        results = [make_eval_result(0, "test", 0.8, "A", "A")]
        result = weighted_avg_aggregator(results)
        detail = result.metric_scores["test"]
        assert detail.numerator is not None
        assert detail.denominator is not None


class TestF1Aggregator:
    def test_empty_results(self) -> None:
        result = f1_aggregator([])
        assert result.metric_scores == {}

    def test_perfect_binary_classification(self) -> None:
        """All predictions correct -> F1 = 1.0."""
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
            make_eval_result(2, "Classification", 1.0, "A", "A"),
            make_eval_result(3, "Classification", 1.0, "B", "B"),
        ]
        result = f1_aggregator(results)

        assert result.metric_scores["Classification_f1"].score == pytest.approx(1.0)
        assert "Classification_accuracy" not in result.metric_scores

    def test_partial_classification_f1(self) -> None:
        """Mixed predictions — F1 between 0 and 1."""
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
            make_eval_result(2, "Classification", 0.0, "B", "A"),  # wrong
            make_eval_result(3, "Classification", 1.0, "A", "A"),
        ]
        result = f1_aggregator(results)

        f1 = result.metric_scores["Classification_f1"].score
        assert 0.0 < f1 < 1.0

    def test_breakdown_includes_classification_report(self) -> None:
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
        ]
        result = f1_aggregator(results)

        assert "Classification" in result.breakdown
        breakdown = result.breakdown["Classification"]
        assert "classification_report" in breakdown
        assert "n_unparseable" in breakdown
        report = breakdown["classification_report"]
        assert "A" in report
        assert "B" in report

    def test_no_details_skipped(self) -> None:
        """Results without classification details should be skipped."""
        results = [
            EvalResult(
                sample_id=0,
                test_case=LLMTestCase(input="test", actual_output="A"),
                metrics=[
                    MetricResult(
                        name="test",
                        score=1.0,
                        passed=True,
                        reason="test",
                        details={"some_key": "value"},  # no expected/predicted
                    )
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
        ]
        result = f1_aggregator(results)
        assert result.metric_scores == {}

    def test_unparseable_counted_as_incorrect(self) -> None:
        """Samples with predicted=None should count as wrong, not be dropped."""
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
            make_eval_result(2, "Classification", 0.0, None, "A"),  # unparseable
            make_eval_result(3, "Classification", 0.0, None, "B"),  # unparseable
        ]
        result = f1_aggregator(results)

        # F1 should be reduced (2 of 4 samples are unparseable)
        f1 = result.metric_scores["Classification_f1"].score
        assert 0.0 < f1 < 1.0

    def test_unparseable_not_phantom_class_in_f1(self) -> None:
        """UNPARSEABLE should not create a phantom class that dilutes macro F1."""
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
            make_eval_result(2, "Classification", 0.0, None, "A"),  # unparseable
        ]
        result = f1_aggregator(results)

        # Macro F1 should average over 2 classes (A, B), not 3
        report = result.breakdown["Classification"]["classification_report"]
        assert UNPARSEABLE_LABEL not in report
        assert "A" in report
        assert "B" in report

    def test_unparseable_reduces_recall(self) -> None:
        """UNPARSEABLE predictions reduce recall for the expected class."""
        # 2 samples expected "A": 1 correct, 1 unparseable
        # 1 sample expected "B": correct
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 0.0, None, "A"),  # unparseable
            make_eval_result(2, "Classification", 1.0, "B", "B"),
        ]
        result = f1_aggregator(results)

        report = result.breakdown["Classification"]["classification_report"]
        # Class A: recall = 1/2 (one unparseable), precision = 1/1
        assert report["A"]["recall"] == pytest.approx(0.5)
        assert report["A"]["precision"] == pytest.approx(1.0)
        # Class B: perfect
        assert report["B"]["recall"] == pytest.approx(1.0)
        assert report["B"]["precision"] == pytest.approx(1.0)

    def test_unparseable_count_in_breakdown(self) -> None:
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 0.0, None, "A"),
            make_eval_result(2, "Classification", 0.0, None, "B"),
        ]
        result = f1_aggregator(results)
        assert result.breakdown["Classification"]["n_unparseable"] == 2

    def test_all_unparseable(self) -> None:
        """All predictions unparseable -> F1 = 0."""
        results = [
            make_eval_result(0, "Classification", 0.0, None, "A"),
            make_eval_result(1, "Classification", 0.0, None, "B"),
        ]
        result = f1_aggregator(results)

        assert result.metric_scores["Classification_f1"].score == pytest.approx(0.0)

    def test_imbalanced_classes_lower_f1(self) -> None:
        """With imbalanced classes and errors, macro F1 should be low."""
        # 8 samples class A (6 correct, 2 misclassified as B), 2 samples class B (0 correct)
        results = [
            *[make_eval_result(i, "Cls", 1.0, "A", "A") for i in range(6)],
            *[make_eval_result(i, "Cls", 0.0, "B", "A") for i in range(6, 8)],
            *[make_eval_result(i, "Cls", 0.0, "A", "B") for i in range(8, 10)],
        ]
        result = f1_aggregator(results)

        f1 = result.metric_scores["Cls_f1"].score
        # Class B has 0 recall -> its F1 is 0, dragging macro F1 down
        assert f1 < 0.6


class TestOrdinalClassificationAggregator:
    def test_reports_all_metrics(self) -> None:
        """Should report accuracy, F1, kappa, and both weighted kappas."""
        results = [
            make_eval_result(0, "Classification", 1.0, "1", "1"),
            make_eval_result(1, "Classification", 1.0, "2", "2"),
            make_eval_result(2, "Classification", 0.0, "1", "2"),
            make_eval_result(3, "Classification", 1.0, "3", "3"),
        ]
        result = ordinal_classification_aggregator(results)

        assert "accuracy" in result.metric_scores
        assert "f1_macro" in result.metric_scores
        assert "cohens_kappa" in result.metric_scores
        assert "linear_weighted_kappa" in result.metric_scores
        assert "quadratic_weighted_kappa" in result.metric_scores

    def test_accuracy_calculation(self) -> None:
        """Accuracy should be correct / n with proper numerator/denominator."""
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 0.0, "B", "A"),
            make_eval_result(2, "Classification", 1.0, "A", "A"),
            make_eval_result(3, "Classification", 1.0, "B", "B"),
        ]
        result = ordinal_classification_aggregator(results)

        assert result.metric_scores["accuracy"].score == pytest.approx(0.75)
        assert result.metric_scores["accuracy"].numerator == 3.0
        assert result.metric_scores["accuracy"].denominator == 4.0

    def test_numerator_denominator_for_merging(self) -> None:
        """All metrics should include numerator/denominator for cross-dataset merging."""
        results = [
            make_eval_result(0, "Classification", 1.0, "1", "1"),
            make_eval_result(1, "Classification", 1.0, "2", "2"),
            make_eval_result(2, "Classification", 0.0, "1", "2"),
            make_eval_result(3, "Classification", 1.0, "3", "3"),
        ]
        result = ordinal_classification_aggregator(results)

        for key in [
            "accuracy",
            "f1_macro",
            "cohens_kappa",
            "linear_weighted_kappa",
            "quadratic_weighted_kappa",
        ]:
            detail = result.metric_scores[key]
            assert detail.numerator is not None, f"{key} missing numerator"
            assert detail.denominator is not None, f"{key} missing denominator"
            assert detail.denominator == 4.0, (
                f"{key} denominator should equal sample count"
            )

    def test_breakdown_includes_all_details(self) -> None:
        results = [
            make_eval_result(0, "Classification", 1.0, "A", "A"),
            make_eval_result(1, "Classification", 1.0, "B", "B"),
        ]
        result = ordinal_classification_aggregator(results)

        breakdown = result.breakdown["Classification"]
        assert "accuracy" in breakdown
        assert "f1_macro" in breakdown
        assert "cohens_kappa" in breakdown
        assert "linear_weighted_kappa" in breakdown
        assert "quadratic_weighted_kappa" in breakdown
        assert "classification_report" in breakdown

    def test_empty_results(self) -> None:
        result = ordinal_classification_aggregator([])
        assert result.metric_scores == {}

    def test_complete_output(self) -> None:
        """Should have all 6 metric types including adjacent_accuracy."""
        results = [
            make_eval_result(0, "KTAS", 1.0, "1", "1"),
            make_eval_result(1, "KTAS", 0.0, "2", "3"),
            make_eval_result(2, "KTAS", 1.0, "4", "4"),
        ]
        result = ordinal_classification_aggregator(results)

        assert len(result.metric_scores) == 6
        assert "accuracy" in result.metric_scores
        assert "adjacent_accuracy" in result.metric_scores

    def test_requires_at_least_2_samples(self) -> None:
        results = [make_eval_result(0, "Classification", 1.0, "A", "A")]
        result = ordinal_classification_aggregator(results)
        assert "accuracy" not in result.metric_scores


class TestMultipleMetrics:
    def test_f1_handles_multiple_metric_names(self) -> None:
        """f1_aggregator should compute separate F1 per metric name."""
        results = [
            make_multi_metric_result(
                0,
                [
                    ("metric1", 1.0, "A", "A"),
                    ("metric2", 0.0, "B", "A"),
                ],
            ),
            make_multi_metric_result(
                1,
                [
                    ("metric1", 1.0, "B", "B"),
                    ("metric2", 1.0, "B", "B"),
                ],
            ),
        ]
        result = f1_aggregator(results)

        assert "metric1_f1" in result.metric_scores
        assert "metric2_f1" in result.metric_scores

    def test_avg_handles_multiple_metric_names(self) -> None:
        results = [
            make_multi_metric_result(
                0,
                [
                    ("metric1", 1.0, "A", "A"),
                    ("metric2", 0.0, "B", "A"),
                ],
            ),
        ]
        result = avg_aggregator(results)
        assert result.metric_scores["metric1"].score == 1.0
        assert result.metric_scores["metric2"].score == 0.0


# ---------------------------------------------------------------------------
# classification_and_avg_aggregator
# ---------------------------------------------------------------------------


def _make_mixed_eval_result(
    sample_id: int,
    cls_predicted: str | None,
    cls_expected: str,
    geval_scores: list[tuple[str, float]],
) -> EvalResult:
    """Create EvalResult with both classification and GEval metrics."""
    metrics = [
        MetricResult(
            name="Classification",
            score=1.0 if cls_predicted == cls_expected else 0.0,
            passed=cls_predicted == cls_expected,
            reason="test",
            details={
                "predicted": cls_predicted,
                "expected": cls_expected,
                "is_correct": cls_predicted == cls_expected,
            },
        ),
    ]
    for name, score in geval_scores:
        metrics.append(
            MetricResult(name=name, score=score, passed=score > 0, reason="")
        )
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(
            input="test",
            actual_output=cls_predicted or "",
            expected_output=cls_expected,
        ),
        metrics=metrics,
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )


def _ordinal_with_labels(results):
    """Helper: ordinal_classification_aggregator with ADR label_order."""
    return ordinal_classification_aggregator(
        results, label_order=["Certain", "Probable/Likely", "Possible"]
    )


class TestCompositeAggregator:
    def test_empty(self) -> None:
        result = composite_aggregator([], pipelines=[])
        assert result.metric_scores == {}

    def test_classification_routed(self) -> None:
        """Classification metrics routed to ordinal, GEval to avg."""
        results = [
            _make_mixed_eval_result(0, "A", "A", [("GEval1", 0.8)]),
            _make_mixed_eval_result(1, "B", "B", [("GEval1", 0.6)]),
            _make_mixed_eval_result(2, "A", "B", [("GEval1", 0.4)]),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
                {"aggregator": avg_aggregator},
            ],
        )

        assert "accuracy" in result.metric_scores
        assert "f1_macro" in result.metric_scores
        assert "GEval1" in result.metric_scores

    def test_geval_averaged(self) -> None:
        """Non-classification metrics routed to avg, correctly averaged."""
        results = [
            _make_mixed_eval_result(
                0,
                "A",
                "A",
                [("Summary Completeness", 0.8), ("Summary Correctness", 0.6)],
            ),
            _make_mixed_eval_result(
                1,
                "B",
                "B",
                [("Summary Completeness", 0.4), ("Summary Correctness", 1.0)],
            ),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
                {"aggregator": avg_aggregator},
            ],
        )

        assert result.metric_scores["Summary Completeness"].score == pytest.approx(0.6)
        assert result.metric_scores["Summary Correctness"].score == pytest.approx(0.8)

    def test_label_order_passed_through(self) -> None:
        """label_order on sub-aggregator should affect adjacent_accuracy."""
        results = [
            _make_mixed_eval_result(0, "Certain", "Probable/Likely", []),
            _make_mixed_eval_result(1, "Possible", "Possible", []),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {"aggregator": _ordinal_with_labels, "metrics": ["Classification"]},
                {"aggregator": avg_aggregator},
            ],
        )

        # Certain vs Probable/Likely is distance=1, should count as adjacent
        assert result.metric_scores["adjacent_accuracy"].score == pytest.approx(1.0)

    def test_breakdown_present(self) -> None:
        """Breakdown keys should be present from classification aggregator."""
        results = [
            _make_mixed_eval_result(0, "A", "A", []),
            _make_mixed_eval_result(1, "B", "B", []),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
            ],
        )

        assert len(result.breakdown) > 0

    def test_catch_all_excludes_routed_metrics(self) -> None:
        """Catch-all aggregator should not see explicitly routed metrics."""
        results = [
            _make_mixed_eval_result(0, "A", "A", [("DDx F1", 0.8)]),
            _make_mixed_eval_result(1, "B", "B", [("DDx F1", 0.6)]),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
                {"aggregator": avg_aggregator},  # catch-all
            ],
        )

        # avg should only have DDx F1, not Classification
        assert "DDx F1" in result.metric_scores
        assert "Classification" not in result.metric_scores

    def test_only_geval_no_classification(self) -> None:
        """If no classification metrics, catch-all avg works alone."""
        results = [
            EvalResult(
                sample_id=0,
                test_case=LLMTestCase(input="test", actual_output="out"),
                metrics=[
                    MetricResult(name="DDx F1", score=0.8, passed=True, reason=""),
                    MetricResult(name="DDx Primary", score=1.0, passed=True, reason=""),
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
                {"aggregator": avg_aggregator},
            ],
        )

        assert result.metric_scores["DDx F1"].score == pytest.approx(0.8)
        assert result.metric_scores["DDx Primary"].score == pytest.approx(1.0)
        assert "accuracy" not in result.metric_scores

    def test_three_aggregators(self) -> None:
        """Three-way composition: ordinal + explicit bert routing + avg catch-all."""
        results = [
            EvalResult(
                sample_id=0,
                test_case=LLMTestCase(
                    input="t", actual_output="A", expected_output="A"
                ),
                metrics=[
                    MetricResult(
                        name="Classification",
                        score=1.0,
                        passed=True,
                        reason="",
                        details={"predicted": "A", "expected": "A"},
                    ),
                    MetricResult(name="BertScore", score=0.9, passed=True, reason=""),
                    MetricResult(name="GEval1", score=0.7, passed=True, reason=""),
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
            EvalResult(
                sample_id=1,
                test_case=LLMTestCase(
                    input="t", actual_output="B", expected_output="B"
                ),
                metrics=[
                    MetricResult(
                        name="Classification",
                        score=1.0,
                        passed=True,
                        reason="",
                        details={"predicted": "B", "expected": "B"},
                    ),
                    MetricResult(name="BertScore", score=0.8, passed=True, reason=""),
                    MetricResult(name="GEval1", score=0.5, passed=True, reason=""),
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],
                },
                {"aggregator": avg_aggregator, "metrics": ["BertScore"]},
                {"aggregator": avg_aggregator},  # catch-all for GEval1
            ],
        )

        assert "accuracy" in result.metric_scores
        assert result.metric_scores["BertScore"].score == pytest.approx(0.85)
        assert result.metric_scores["GEval1"].score == pytest.approx(0.6)

    def test_key_to_name_resolves_config_keys(self) -> None:
        """key_to_name maps YAML config keys to runtime metric __name__ values."""
        results = [
            _make_mixed_eval_result(0, "A", "A", [("GEval1", 0.8)]),
            _make_mixed_eval_result(1, "B", "B", [("GEval1", 0.6)]),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["classification"],  # config key, not __name__
                },
                {"aggregator": avg_aggregator},
            ],
            key_to_name={"classification": "Classification"},
        )

        # Classification routed to ordinal aggregator
        assert "accuracy" in result.metric_scores
        # GEval1 routed to catch-all avg
        assert "GEval1" in result.metric_scores

    def test_key_to_name_missing_key_raises(self) -> None:
        """key_to_name with unknown config key should raise KeyError."""
        results = [
            _make_mixed_eval_result(0, "A", "A", []),
        ]
        with pytest.raises(KeyError):
            composite_aggregator(
                results,
                pipelines=[
                    {
                        "aggregator": ordinal_classification_aggregator,
                        "metrics": ["nonexistent_key"],
                    },
                ],
                key_to_name={"classification": "Classification"},
            )

    def test_key_to_name_none_skips_resolution(self) -> None:
        """When key_to_name is None, metrics list is used as-is."""
        results = [
            _make_mixed_eval_result(0, "A", "A", [("GEval1", 0.8)]),
            _make_mixed_eval_result(1, "B", "B", [("GEval1", 0.6)]),
        ]
        result = composite_aggregator(
            results,
            pipelines=[
                {
                    "aggregator": ordinal_classification_aggregator,
                    "metrics": ["Classification"],  # raw __name__, no resolution
                },
                {"aggregator": avg_aggregator},
            ],
            key_to_name=None,
        )

        assert "accuracy" in result.metric_scores
        assert "GEval1" in result.metric_scores


class TestKeyToNameOnSimpleAggregators:
    """Simple aggregators accept key_to_name without error (passed by Hydra)."""

    def test_avg_accepts_key_to_name(self) -> None:
        results = [make_eval_result(0, "test", 1.0, "A", "A")]
        result = avg_aggregator(results, key_to_name={"x": "Y"})
        assert "test" in result.metric_scores

    def test_weighted_avg_accepts_key_to_name(self) -> None:
        results = [make_eval_result(0, "test", 1.0, "A", "A")]
        result = weighted_avg_aggregator(results, key_to_name={"x": "Y"})
        assert "test" in result.metric_scores

    def test_f1_accepts_key_to_name(self) -> None:
        results = [
            make_eval_result(0, "Cls", 1.0, "A", "A"),
            make_eval_result(1, "Cls", 1.0, "B", "B"),
        ]
        result = f1_aggregator(results, key_to_name={"x": "Y"})
        assert "Cls_f1" in result.metric_scores

    def test_ordinal_accepts_key_to_name(self) -> None:
        results = [
            make_eval_result(0, "Cls", 1.0, "A", "A"),
            make_eval_result(1, "Cls", 0.0, "B", "A"),
        ]
        result = ordinal_classification_aggregator(results, key_to_name={"x": "Y"})
        assert "accuracy" in result.metric_scores


class TestFilterResults:
    def test_include_filters_by_metric_name(self) -> None:
        results = [
            make_multi_metric_result(0, [("A", 1.0, "x", "x"), ("B", 0.5, "y", "y")]),
        ]
        filtered = _filter_results(results, include={"A"})

        assert len(filtered) == 1
        assert len(filtered[0].metrics) == 1
        assert filtered[0].metrics[0].name == "A"

    def test_exclude_filters_by_metric_name(self) -> None:
        results = [
            make_multi_metric_result(0, [("A", 1.0, "x", "x"), ("B", 0.5, "y", "y")]),
        ]
        filtered = _filter_results(results, exclude={"A"})

        assert len(filtered) == 1
        assert len(filtered[0].metrics) == 1
        assert filtered[0].metrics[0].name == "B"

    def test_drops_results_with_no_remaining_metrics(self) -> None:
        results = [
            make_eval_result(0, "A", 1.0, "x", "x"),
        ]
        filtered = _filter_results(results, include={"B"})
        assert len(filtered) == 0

    def test_no_filter_returns_all(self) -> None:
        results = [
            make_multi_metric_result(0, [("A", 1.0, "x", "x"), ("B", 0.5, "y", "y")]),
        ]
        filtered = _filter_results(results)
        assert len(filtered) == 1
        assert len(filtered[0].metrics) == 2


class TestOrdinalLabelOrder:
    def test_adjacent_accuracy_with_label_order(self) -> None:
        """Adjacent accuracy uses label_order for ordinal distance."""
        results = [
            make_eval_result(0, "Cls", 0.0, "Probable/Likely", "Certain"),
            make_eval_result(1, "Cls", 1.0, "Possible", "Possible"),
        ]
        result = ordinal_classification_aggregator(
            results, label_order=["Certain", "Probable/Likely", "Possible"]
        )

        # Certain→Probable/Likely is distance 1 (adjacent), Possible→Possible is distance 0
        assert result.metric_scores["adjacent_accuracy"].score == pytest.approx(1.0)

    def test_adjacent_accuracy_without_label_order_uses_int(self) -> None:
        """Without label_order, ordinal distance uses int conversion."""
        results = [
            make_eval_result(0, "Cls", 0.0, "2", "1"),  # distance 1 → adjacent
            make_eval_result(1, "Cls", 0.0, "3", "1"),  # distance 2 → not adjacent
        ]
        result = ordinal_classification_aggregator(results)

        # 1 of 2 is adjacent
        assert result.metric_scores["adjacent_accuracy"].score == pytest.approx(0.5)

    def test_non_adjacent_prediction_lowers_adjacent_accuracy(self) -> None:
        results = [
            make_eval_result(0, "Cls", 0.0, "Possible", "Certain"),  # distance 2
            make_eval_result(1, "Cls", 1.0, "Certain", "Certain"),  # distance 0
        ]
        result = ordinal_classification_aggregator(
            results, label_order=["Certain", "Probable/Likely", "Possible"]
        )

        assert result.metric_scores["adjacent_accuracy"].score == pytest.approx(0.5)


class TestClippedAvgAggregator:
    def test_negative_mean_clips_to_zero(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([-0.5, -0.3])]
        assert clipped_avg_aggregator(results).metric_scores["m"].score == 0.0

    def test_mean_above_one_clips_to_one(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([1.5, 1.2])]
        assert clipped_avg_aggregator(results).metric_scores["m"].score == 1.0

    def test_in_range_mean_passes_through(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([0.4, 0.6])]
        detail = clipped_avg_aggregator(results).metric_scores["m"]
        assert detail.score == pytest.approx(0.5)

    def test_numerator_rescaled_so_merge_cannot_undo_clip(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([-1.0, -0.5])]
        detail = clipped_avg_aggregator(results).metric_scores["m"]
        assert detail.score == 0.0
        assert detail.denominator == 2.0
        assert detail.numerator == 0.0

    def test_breakdown_keeps_raw_mean(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([-0.5, -0.3])]
        assert clipped_avg_aggregator(results).breakdown["m"]["mean"] == pytest.approx(
            -0.4
        )

    def test_empty_results(self) -> None:
        assert clipped_avg_aggregator([]).metric_scores == {}
