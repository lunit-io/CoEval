"""Tests for the Conquer Health val/test splits and competition scoring rules."""

import json
from types import SimpleNamespace

import pytest

from coeval.core.runner import EvalRunner
from coeval.datasets.conquer import (
    ConquerHealthTestDataset,
    ConquerHealthValDataset,
)
from coeval.util.aggregation import avg_aggregator


def _val_ids() -> list[str]:
    from importlib import resources

    with resources.as_file(
        resources.files("coeval.data").joinpath("conquer_val_ids.json")
    ) as path:
        return json.loads(path.read_text())["prompt_ids"]


class TestValSplitShipsWithPackage:
    def test_val_ids_are_packaged_unique_and_nonempty(self) -> None:
        ids = _val_ids()
        assert len(ids) > 100
        assert len(set(ids)) == len(ids)

    def test_test_split_refuses_to_guess_an_id_list(self) -> None:
        """The test ids must never be defaulted: doing so would leak the holdout."""
        with pytest.raises(ValueError, match="requires ids_path"):
            ConquerHealthTestDataset()


class TestSplitLoading:
    @pytest.fixture(scope="class")
    def val(self) -> ConquerHealthValDataset:
        return ConquerHealthValDataset()

    def test_loads_every_requested_id(self, val: ConquerHealthValDataset) -> None:
        assert len(val.goldens) == len(_val_ids())

    def test_order_is_deterministic_across_instances(
        self, val: ConquerHealthValDataset
    ) -> None:
        """sample_id N must mean the same item for every team, on every run --
        cross-team comparison and the common-item intersection depend on it."""
        again = ConquerHealthValDataset()
        assert [g.scenario for g in again.goldens] == [g.scenario for g in val.goldens]

    def test_every_golden_carries_rubrics(self, val: ConquerHealthValDataset) -> None:
        assert all((g.additional_metadata or {}).get("rubrics") for g in val.goldens)

    def test_unknown_id_fails_loudly(self, tmp_path) -> None:
        """A stale id list must not silently yield a short, incomparable set."""
        bad = tmp_path / "ids.json"
        bad.write_text(json.dumps({"prompt_ids": ["not-a-real-prompt-id"]}))
        with pytest.raises(ValueError, match="not found in HealthBench Main"):
            ConquerHealthValDataset(ids_path=bad)

    def test_num_samples_truncates_for_smoke_tests(
        self, val: ConquerHealthValDataset
    ) -> None:
        small = ConquerHealthValDataset(num_samples=3)
        assert len(small.goldens) == 3
        assert [g.scenario for g in small.goldens] == [
            g.scenario for g in val.goldens[:3]
        ]


class _Metric:
    __name__ = "HealthBench Rubric"


def _failed_result_summary(*, as_zero: bool) -> tuple:
    """Build a summary over 1 good sample (score 1.0) and 1 inference failure."""
    runner = EvalRunner(
        client=SimpleNamespace(),
        score_inference_failures_as_zero=as_zero,
    )
    metrics = [_Metric()]
    test_cases = [
        SimpleNamespace(additional_metadata={}),
        SimpleNamespace(additional_metadata={}),
    ]
    metadata = [
        {
            "sample_id": 0,
            "rationale": "",
            "generation_time_ms": 1.0,
            "scoring_time_ms": 1.0,
            "inference_failed": False,
            "inference_error": None,
        },
        {
            "sample_id": 1,
            "rationale": "",
            "generation_time_ms": 1.0,
            "scoring_time_ms": 0.0,
            "inference_failed": True,
            "inference_error": "no content",
        },
    ]
    from coeval.core.evaluate import MetricScore

    lookup = {0: [MetricScore(name="HealthBench Rubric", score=1.0, success=True)]}
    results = runner._build_eval_results(test_cases, metadata, metrics, lookup)
    summary = runner._build_summary(results, "conquer", 1.0, avg_aggregator)
    return results, summary


class TestEndpointFailuresScoreZero:
    """Their failure scores 0; ours is excluded. The distinction decides fairness."""

    def test_competitive_mode_scores_failure_as_zero(self) -> None:
        results, summary = _failed_result_summary(as_zero=True)
        assert results[1].metrics[0].score == 0.0
        detail = summary.metric_scores["HealthBench Rubric"]
        # 1.0 and 0.0 over two samples -> 0.5, and the denominator keeps both.
        assert detail.score == pytest.approx(0.5)
        assert detail.denominator == pytest.approx(2.0)

    def test_research_mode_excludes_failure(self) -> None:
        results, summary = _failed_result_summary(as_zero=False)
        assert results[1].metrics[0].score is None
        detail = summary.metric_scores["HealthBench Rubric"]
        # The failure vanishes, so the single good sample sets the mean.
        assert detail.score == pytest.approx(1.0)
        assert detail.denominator == pytest.approx(1.0)

    def test_failure_count_is_still_reported_in_competitive_mode(self) -> None:
        """The zero must not hide the diagnostic -- a run absorbing many zeros
        needs to be visible as such, not just as a low score."""
        _, summary = _failed_result_summary(as_zero=True)
        assert summary.num_inference_failed == 1
        assert summary.num_samples == 2
