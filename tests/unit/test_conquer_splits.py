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


class TestSplitGeneratorSecrecy:
    """The generator is public, so the split must not be derivable from it.

    Without a salt the selection is a pure function of the (documented) sizes,
    which would mean the repository contains the holdout: clone, run with
    --n-val 300 --n-test 500, recover the exact test ids.
    """

    @staticmethod
    def _load_generator():
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "make_conquer_split.py"
        spec = importlib.util.spec_from_file_location("make_conquer_split", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["make_conquer_split"] = mod
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _rows(n: int = 400) -> list[dict]:
        return [
            {
                "prompt_id": f"id-{i:04d}",
                "example_tags": ["theme:hedging"],
                "rubrics": [{"criterion": "c", "points": 5, "tags": []}] * (2 + i % 9),
                "prompt": [{"role": "user", "content": "q"}],
            }
            for i in range(n)
        ]

    def test_different_salts_give_different_holdouts(self) -> None:
        gen = self._load_generator()
        rows = self._rows()
        _, test_a = gen.build_split(rows, 60, 100, "salt-aaaaaaaaaaaaaaaaaa")
        _, test_b = gen.build_split(rows, 60, 100, "salt-bbbbbbbbbbbbbbbbbb")
        assert set(test_a) != set(test_b)
        # Overlap should sit near chance for a 160-of-400 draw, not near total.
        assert len(set(test_a) & set(test_b)) < 0.8 * len(test_a)

    def test_same_salt_is_reproducible(self) -> None:
        gen = self._load_generator()
        rows = self._rows()
        first = gen.build_split(rows, 60, 100, "salt-cccccccccccccccccc")
        second = gen.build_split(rows, 60, 100, "salt-cccccccccccccccccc")
        assert first == second

    def test_splits_stay_disjoint_under_salting(self) -> None:
        gen = self._load_generator()
        val, test = gen.build_split(self._rows(), 60, 100, "salt-dddddddddddddddddd")
        assert not set(val) & set(test)

    def test_fingerprint_does_not_reveal_the_salt(self) -> None:
        gen = self._load_generator()
        salt = "salt-eeeeeeeeeeeeeeeeee"
        fp = gen.salt_fingerprint(salt)
        assert salt not in fp
        assert fp != gen.salt_fingerprint(salt + "x")

    def test_shipped_val_file_carries_no_salt(self) -> None:
        """The val list is published; a salt inside it would hand over the test set."""
        from importlib import resources

        with resources.as_file(
            resources.files("coeval.data").joinpath("conquer_val_ids.json")
        ) as path:
            payload = json.loads(path.read_text())
        assert "salt" not in payload
        assert "salt_fingerprint" in payload
