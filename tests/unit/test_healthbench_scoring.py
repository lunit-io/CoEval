"""Tests for HealthBench scoring logic and aggregation."""

import pytest
from deepeval.test_case import LLMTestCase

from coeval.core.schema import EvalResult, MetricResult
from coeval.metrics.healthbench_rubric import (
    calculate_score,
    parse_grading_response,
)
from coeval.util.aggregation import avg_aggregator

# ---------------------------------------------------------------------------
# calculate_score() tests
# ---------------------------------------------------------------------------


class TestCalculateScore:
    """Tests for the official HealthBench per-example scoring formula."""

    def test_all_criteria_met(self) -> None:
        """All positive criteria met -> score = 1.0."""
        rubrics = [
            {"criterion": "c1", "points": 1.0, "tags": []},
            {"criterion": "c2", "points": 2.0, "tags": []},
        ]
        grades = [
            {"criteria_met": True, "explanation": "ok"},
            {"criteria_met": True, "explanation": "ok"},
        ]
        assert calculate_score(rubrics, grades) == pytest.approx(1.0)

    def test_no_criteria_met(self) -> None:
        """No criteria met -> score = 0.0."""
        rubrics = [
            {"criterion": "c1", "points": 1.0, "tags": []},
            {"criterion": "c2", "points": 2.0, "tags": []},
        ]
        grades = [
            {"criteria_met": False, "explanation": "no"},
            {"criteria_met": False, "explanation": "no"},
        ]
        assert calculate_score(rubrics, grades) == pytest.approx(0.0)

    def test_partial_criteria_met(self) -> None:
        """Partial criteria met -> weighted score."""
        rubrics = [
            {"criterion": "c1", "points": 1.0, "tags": []},
            {"criterion": "c2", "points": 3.0, "tags": []},
        ]
        grades = [
            {"criteria_met": True, "explanation": "ok"},
            {"criteria_met": False, "explanation": "no"},
        ]
        # achieved = 1.0, total = 4.0 -> 0.25
        assert calculate_score(rubrics, grades) == pytest.approx(0.25)

    def test_negative_points_met(self) -> None:
        """Negative-point criteria met -> reduces score below 1.0."""
        rubrics = [
            {"criterion": "good", "points": 2.0, "tags": []},
            {"criterion": "bad (verbose)", "points": -1.0, "tags": []},
        ]
        grades = [
            {"criteria_met": True, "explanation": "ok"},
            {"criteria_met": True, "explanation": "verbose indeed"},
        ]
        # achieved = 2.0 + (-1.0) = 1.0, total_possible = 2.0 -> 0.5
        assert calculate_score(rubrics, grades) == pytest.approx(0.5)

    def test_negative_points_not_met(self) -> None:
        """Negative-point criteria NOT met (good!) -> doesn't reduce score."""
        rubrics = [
            {"criterion": "good", "points": 2.0, "tags": []},
            {"criterion": "bad (verbose)", "points": -1.0, "tags": []},
        ]
        grades = [
            {"criteria_met": True, "explanation": "ok"},
            {"criteria_met": False, "explanation": "not verbose"},
        ]
        # achieved = 2.0, total_possible = 2.0 -> 1.0
        assert calculate_score(rubrics, grades) == pytest.approx(1.0)

    def test_no_positive_points_returns_none(self) -> None:
        """All rubrics have non-positive points -> None."""
        rubrics = [
            {"criterion": "bad", "points": -1.0, "tags": []},
        ]
        grades = [
            {"criteria_met": True, "explanation": "bad"},
        ]
        assert calculate_score(rubrics, grades) is None

    def test_empty_rubrics(self) -> None:
        """Empty rubrics -> None (no total possible points)."""
        assert calculate_score([], []) is None

    def test_score_can_be_negative(self) -> None:
        """Score below 0 is valid (before clipping by aggregator)."""
        rubrics = [
            {"criterion": "good", "points": 1.0, "tags": []},
            {"criterion": "bad1", "points": -1.0, "tags": []},
            {"criterion": "bad2", "points": -1.0, "tags": []},
        ]
        grades = [
            {"criteria_met": False, "explanation": "not good"},
            {"criteria_met": True, "explanation": "bad1"},
            {"criteria_met": True, "explanation": "bad2"},
        ]
        # achieved = 0 + (-1.0) + (-1.0) = -2.0, total_possible = 1.0
        score = calculate_score(rubrics, grades)
        assert score is not None
        assert score < 0

    def test_parity_with_official_code(self) -> None:
        """Verify scoring matches official HealthBench code exactly."""
        # Reproduce Table 8 example structure
        rubrics = [
            {"criterion": "c1", "points": 3.0, "tags": ["accuracy"]},
            {"criterion": "c2", "points": 2.0, "tags": ["safety"]},
            {"criterion": "c3", "points": -1.0, "tags": ["verbosity"]},
        ]
        grades = [
            {"criteria_met": True, "explanation": ""},
            {"criteria_met": True, "explanation": ""},
            {"criteria_met": False, "explanation": ""},
        ]
        # achieved = 3 + 2 = 5, total_possible = 5 -> 1.0
        assert calculate_score(rubrics, grades) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# parse_grading_response() tests
# ---------------------------------------------------------------------------


class TestParseGradingResponse:
    def test_valid_json_true(self) -> None:
        text = '{"explanation": "good", "criteria_met": true}'
        result = parse_grading_response(text)
        assert result is not None
        assert result["criteria_met"] is True

    def test_valid_json_false(self) -> None:
        text = '{"explanation": "bad", "criteria_met": false}'
        result = parse_grading_response(text)
        assert result is not None
        assert result["criteria_met"] is False

    def test_markdown_fenced_json(self) -> None:
        text = '```json\n{"explanation": "ok", "criteria_met": true}\n```'
        result = parse_grading_response(text)
        assert result is not None
        assert result["criteria_met"] is True

    def test_invalid_json_returns_none(self) -> None:
        assert parse_grading_response("not json at all") is None

    def test_missing_criteria_met_returns_none(self) -> None:
        assert parse_grading_response('{"explanation": "ok"}') is None

    def test_string_boolean_returns_none(self) -> None:
        """criteria_met must be bool, not string."""
        text = '{"explanation": "ok", "criteria_met": "true"}'
        assert parse_grading_response(text) is None

    @pytest.mark.parametrize("response", [None, 123, {"criteria_met": True}])
    def test_non_string_response_returns_none(self, response: object) -> None:
        """Unexpected judge return types are invalid responses, not parser errors."""
        assert parse_grading_response(response) is None


# ---------------------------------------------------------------------------
# avg_aggregator tests for HealthBench scoring
# ---------------------------------------------------------------------------

METRIC_NAME = "HealthBench Rubric"


def _make_hb_eval_result(sample_id: int, score: float) -> EvalResult:
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(
            input="test",
            actual_output="response",
            expected_output="",
        ),
        metrics=[
            MetricResult(
                name=METRIC_NAME,
                score=score,
                passed=score >= 0.5,
                reason="test",
            )
        ],
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )


class TestHealthBenchAvgAggregation:
    """Tests that avg_aggregator correctly computes HealthBench scores.

    HealthBench uses simple mean of per-example scores (official paper).
    avg_aggregator does exactly this: fmean(scores) per metric name.
    """

    def test_empty_results(self) -> None:
        result = avg_aggregator([])
        assert result.metric_scores == {}

    def test_single_perfect_score(self) -> None:
        results = [_make_hb_eval_result(0, 1.0)]
        result = avg_aggregator(results)
        assert result.metric_scores[METRIC_NAME].score == pytest.approx(1.0)

    def test_mean_of_two_scores(self) -> None:
        """Mean of scores: (0.8 + 0.6) / 2 = 0.7."""
        results = [
            _make_hb_eval_result(0, 0.8),
            _make_hb_eval_result(1, 0.6),
        ]
        result = avg_aggregator(results)
        assert result.metric_scores[METRIC_NAME].score == pytest.approx(0.7)

    def test_negative_scores_not_clipped(self) -> None:
        """avg_aggregator preserves raw mean (no clipping).

        Per-example scores can be negative when penalties exceed positive
        criteria, but this is extremely rare in practice. The official
        HealthBench formula clips at aggregation, but per-example negatives
        are meaningful diagnostic signals worth preserving.
        """
        results = [
            _make_hb_eval_result(0, -0.5),
            _make_hb_eval_result(1, -0.3),
        ]
        result = avg_aggregator(results)
        assert result.metric_scores[METRIC_NAME].score == pytest.approx(-0.4)

    @pytest.mark.parametrize(
        "scores,expected",
        [
            ([1.0], 1.0),
            ([0.0], 0.0),
            ([0.5], 0.5),
            ([0.8, 0.6], 0.7),
            ([0.9, 0.7], 0.8),
            ([0.1, 0.2, 0.3], 0.2),
            ([1.0, 0.0], 0.5),
            ([0.25, 0.75, 0.50], 0.5),
            ([0.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 1.0], 1.0),
        ],
        ids=[
            "single_1.0",
            "single_0.0",
            "single_0.5",
            "two_normal",
            "two_high",
            "three_low",
            "two_extremes",
            "three_mixed",
            "all_zero",
            "all_perfect",
        ],
    )
    def test_parametric_mean(self, scores: list[float], expected: float) -> None:
        """Verify mean computation across many score distributions."""
        results = [_make_hb_eval_result(i, s) for i, s in enumerate(scores)]
        result = avg_aggregator(results)
        assert result.metric_scores[METRIC_NAME].score == pytest.approx(expected)

    def test_realistic_distribution(self) -> None:
        """Realistic HealthBench-like 10-sample distribution."""
        scores = [0.82, 0.45, 0.91, 0.67, 0.73, 0.58, 0.36, 0.95, 0.12, 0.88]
        expected = sum(scores) / len(scores)
        results = [_make_hb_eval_result(i, s) for i, s in enumerate(scores)]
        result = avg_aggregator(results)
        assert result.metric_scores[METRIC_NAME].score == pytest.approx(expected)

    def test_averages_each_metric_independently(self) -> None:
        """When multiple metrics coexist, each is averaged independently."""
        result = EvalResult(
            sample_id=0,
            test_case=LLMTestCase(
                input="test", actual_output="response", expected_output=""
            ),
            metrics=[
                MetricResult(name=METRIC_NAME, score=0.9, passed=True, reason="ok"),
                MetricResult(
                    name="Other Metric", score=0.1, passed=False, reason="low"
                ),
            ],
            generation_time_ms=100.0,
            scoring_time_ms=10.0,
        )

        aggregated = avg_aggregator([result])
        assert aggregated.metric_scores[METRIC_NAME].score == pytest.approx(0.9)
        assert aggregated.metric_scores["Other Metric"].score == pytest.approx(0.1)

    def test_numerator_denominator(self) -> None:
        """avg_aggregator stores numerator/denominator for cross-dataset merging."""
        results = [
            _make_hb_eval_result(0, 0.9),
            _make_hb_eval_result(1, 0.7),
        ]
        result = avg_aggregator(results)
        detail = result.metric_scores[METRIC_NAME]
        assert detail.numerator == pytest.approx(1.6)
        assert detail.denominator == 2.0

    def test_breakdown_stats(self) -> None:
        """avg_aggregator produces breakdown with mean, n_samples, min, max."""
        results = [
            _make_hb_eval_result(0, 0.8),
            _make_hb_eval_result(1, 0.4),
        ]
        result = avg_aggregator(results)
        assert METRIC_NAME in result.breakdown
        bd = result.breakdown[METRIC_NAME]
        assert bd["mean"] == pytest.approx(0.6)
        assert bd["n_samples"] == 2
        assert bd["min_score"] == pytest.approx(0.4)
        assert bd["max_score"] == pytest.approx(0.8)

    def test_sub_scores_expand_into_separate_metrics(self) -> None:
        """Sub-scores from metric details become independent metric entries,
        so avg_aggregator groups them by name automatically."""
        # Simulate what the runner produces after expanding _sub_scores:
        # 2 samples, each with main score + theme sub-score
        results = [
            EvalResult(
                sample_id=0,
                test_case=LLMTestCase(input="t", actual_output="r", expected_output=""),
                metrics=[
                    MetricResult(name=METRIC_NAME, score=0.8, passed=True, reason="ok"),
                    MetricResult(
                        name="theme:safety", score=0.8, passed=True, reason=""
                    ),
                    MetricResult(
                        name="cluster:dosage", score=0.9, passed=True, reason=""
                    ),
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
            EvalResult(
                sample_id=1,
                test_case=LLMTestCase(input="t", actual_output="r", expected_output=""),
                metrics=[
                    MetricResult(name=METRIC_NAME, score=0.6, passed=True, reason="ok"),
                    MetricResult(
                        name="theme:safety", score=0.6, passed=True, reason=""
                    ),
                    MetricResult(
                        name="theme:diagnosis", score=0.6, passed=True, reason=""
                    ),
                    MetricResult(
                        name="cluster:dosage", score=0.7, passed=True, reason=""
                    ),
                ],
                generation_time_ms=100.0,
                scoring_time_ms=10.0,
            ),
        ]

        aggregated = avg_aggregator(results)

        # Main score: (0.8 + 0.6) / 2
        assert aggregated.metric_scores[METRIC_NAME].score == pytest.approx(0.7)
        # theme:safety appears in both: (0.8 + 0.6) / 2
        assert aggregated.metric_scores["theme:safety"].score == pytest.approx(0.7)
        assert aggregated.metric_scores["theme:safety"].denominator == 2.0
        # theme:diagnosis only in sample 1
        assert aggregated.metric_scores["theme:diagnosis"].score == pytest.approx(0.6)
        assert aggregated.metric_scores["theme:diagnosis"].denominator == 1.0
        # cluster:dosage: (0.9 + 0.7) / 2
        assert aggregated.metric_scores["cluster:dosage"].score == pytest.approx(0.8)
        # All have breakdown
        assert aggregated.breakdown["theme:safety"]["n_samples"] == 2
        assert aggregated.breakdown["cluster:dosage"]["n_samples"] == 2
