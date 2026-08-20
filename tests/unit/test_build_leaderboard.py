"""Tests for the leaderboard ranking script.

The tie test is the part worth guarding: a test that always answers "tied" is
worse than none, because it hides real differences behind a veneer of rigour.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_leaderboard", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules["build_leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


bl = _load_module()


def write_run(
    root: Path,
    team: str,
    scores: dict[int, float | None],
    *,
    prompt_prefix: str = "q",
    inference_failed: set[int] | None = None,
) -> None:
    """Emit a minimal CoEval results file for one submission."""
    inference_failed = inference_failed or set()
    rows = []
    for sid, score in scores.items():
        metrics = []
        if score is not None:
            metrics.append(
                {"name": "HealthBench Rubric", "score": score, "passed": score >= 0.5}
            )
            metrics.append({"name": "axis:accuracy", "score": score, "passed": True})
            metrics.append({"name": "theme:hedging", "score": score, "passed": True})
        rows.append(
            {
                "sample_id": sid,
                "input": f"{prompt_prefix}{sid}",
                "actual_output": "a" * 100,
                "passed": score is not None and score >= 0.5,
                "inference_failed": sid in inference_failed,
                "inference_error": "boom" if sid in inference_failed else None,
                "scoring_failed": score is None and sid not in inference_failed,
                "generation_time_ms": 1000.0,
                "scoring_time_ms": 10.0,
                "metrics": metrics,
            }
        )
    d = root / team
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_ds.json").write_text(json.dumps(rows))


def run_script(runs: Path, out: Path, *extra: str) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--runs",
        str(runs),
        "--dataset",
        "ds",
        "--out",
        str(out),
        "--iterations",
        "2000",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


class TestTieDetection:
    def test_identical_teams_are_tied(self, tmp_path: Path) -> None:
        scores = {i: (i % 10) / 10 for i in range(200)}
        write_run(tmp_path / "runs", "alpha", scores)
        write_run(tmp_path / "runs", "beta", dict(scores))
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        assert payload["entries"][0]["tied_with_ranks"] == [2]

    def test_a_real_gap_is_not_called_a_tie(self, tmp_path: Path) -> None:
        """0.9 against 0.2 across 200 items must not be dismissed as noise."""
        strong = dict.fromkeys(range(200), 0.9)
        weak = dict.fromkeys(range(200), 0.2)
        write_run(tmp_path / "runs", "strong", strong)
        write_run(tmp_path / "runs", "weak", weak)
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        assert [e["team"] for e in payload["entries"]] == ["strong", "weak"]
        assert payload["entries"][0]["tied_with_ranks"] == []
        assert payload["entries"][1]["tied_with_ranks"] == []

    def test_gap_inside_measured_noise_is_a_tie(self, tmp_path: Path) -> None:
        """A ~1pp separation is smaller than run-to-run noise and must tie."""
        base = {i: (i % 10) / 10 for i in range(200)}
        nudged = {i: min(1.0, v + 0.01) for i, v in base.items()}
        write_run(tmp_path / "runs", "aa", base)
        write_run(tmp_path / "runs", "bb", nudged)
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        assert payload["entries"][0]["tied_with_ranks"] == [2]


class TestCommonItemIntersection:
    def test_items_ungradeable_for_one_team_are_dropped_for_all(
        self, tmp_path: Path
    ) -> None:
        a = dict.fromkeys(range(10), 1.0)
        b = dict.fromkeys(range(10), 1.0)
        b[3] = None  # judge failed on this item for team b only
        write_run(tmp_path / "runs", "aa", a)
        write_run(tmp_path / "runs", "bb", b)
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        assert payload["dataset"]["n_scored"] == 9
        assert payload["dataset"]["dropped_sample_ids"] == [3]
        # Both teams scored over the same 9 items, so both are 1.0.
        assert all(e["n_scored"] == 9 for e in payload["entries"])

    def test_no_common_items_is_an_error(self, tmp_path: Path) -> None:
        write_run(tmp_path / "runs", "aa", {0: 1.0, 1: None})
        write_run(tmp_path / "runs", "bb", {0: None, 1: 1.0})
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "lb.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "no item was successfully graded" in proc.stderr


class TestIntegrityGuards:
    def test_different_prompt_text_refuses_to_rank(self, tmp_path: Path) -> None:
        """Same sample_id must mean the same question, or the ranking is meaningless."""
        write_run(tmp_path / "runs", "aa", {0: 1.0, 1: 1.0}, prompt_prefix="q")
        write_run(tmp_path / "runs", "bb", {0: 1.0, 1: 1.0}, prompt_prefix="DIFFERENT")
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "lb.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "different prompt text" in proc.stderr

    def test_different_item_counts_refuse_to_rank(self, tmp_path: Path) -> None:
        write_run(tmp_path / "runs", "aa", {0: 1.0, 1: 1.0, 2: 1.0})
        write_run(tmp_path / "runs", "bb", {0: 1.0, 1: 1.0})
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "lb.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "not comparable" in proc.stderr


class TestReporting:
    def test_endpoint_failures_are_surfaced_not_buried(self, tmp_path: Path) -> None:
        scores = dict.fromkeys(range(100), 0.5)
        write_run(tmp_path / "runs", "aa", scores)
        # 10 items answered but flagged as endpoint failures -> scored, and flagged.
        write_run(
            tmp_path / "runs", "bb", dict(scores), inference_failed=set(range(10))
        )
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        bb = next(e for e in payload["entries"] if e["team"] == "bb")
        assert bb["n_inference_failed"] == 10
        assert "endpoint_failures_scored_zero" in bb["flags"]

    def test_runoff_list_is_capped_and_truncation_is_declared(
        self, tmp_path: Path
    ) -> None:
        scores = {i: (i % 10) / 10 for i in range(150)}
        for name in ("t1", "t2", "t3", "t4", "t5", "t6"):
            write_run(tmp_path / "runs", name, dict(scores))
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--runoff-max", "3"
        )
        assert len(payload["runoff_candidates"]) == 3
        assert payload["runoff_excluded_count"] == 3
        assert any("cost-bounded subset" in n for n in payload["notes"])

    def test_axis_and_theme_breakdowns_are_present(self, tmp_path: Path) -> None:
        scores = dict.fromkeys(range(20), 0.4)
        write_run(tmp_path / "runs", "aa", scores)
        write_run(tmp_path / "runs", "bb", dict(scores))
        payload = run_script(tmp_path / "runs", tmp_path / "lb.json")
        e = payload["entries"][0]
        assert e["axis_scores"]["accuracy"] == pytest.approx(0.4)
        assert e["theme_scores"]["hedging"] == pytest.approx(0.4)
        assert e["mean_response_chars"] == 100


class TestNoiseModel:
    def test_run_noise_shrinks_with_set_size(self) -> None:
        assert bl.run_noise_sd(200) == pytest.approx(0.0174)
        assert bl.run_noise_sd(500) == pytest.approx(0.0174 * (200 / 500) ** 0.5)
        assert bl.run_noise_sd(800) < bl.run_noise_sd(500)

    def test_clipped_mean_matches_official_aggregation(self) -> None:
        """Penalty criteria can drive a per-example score negative; the reported
        figure clips into [0, 1]."""
        assert bl.clipped_mean([-0.5, -0.5]) == 0.0
        assert bl.clipped_mean([1.5, 1.5]) == 1.0
        assert bl.clipped_mean([0.25, 0.75]) == pytest.approx(0.5)


class TestLiveBoardScoring:
    """The live board is a development signal, so it must not let one team's
    judge failure move another team's number."""

    def test_provisional_does_not_intersect_across_teams(self, tmp_path: Path) -> None:
        clean = dict.fromkeys(range(100), 0.5)
        with_gap = dict(clean)
        with_gap[7] = None  # judge failed on this item, for this team only
        write_run(tmp_path / "runs", "clean", clean)
        write_run(tmp_path / "runs", "gappy", with_gap)
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        by_team = {e["team"]: e for e in payload["entries"]}
        assert by_team["clean"]["n_scored"] == 100
        assert by_team["clean"]["n_ungraded"] == 0
        assert by_team["gappy"]["n_scored"] == 99
        assert by_team["gappy"]["n_ungraded"] == 1
        assert payload["scoring"]["item_pairing"] == "per-team"

    def test_official_still_intersects(self, tmp_path: Path) -> None:
        clean = dict.fromkeys(range(100), 0.5)
        with_gap = dict(clean)
        with_gap[7] = None
        write_run(tmp_path / "runs", "clean", clean)
        write_run(tmp_path / "runs", "gappy", with_gap)
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "official"
        )
        assert payload["scoring"]["item_pairing"] == "intersection"
        assert all(e["n_scored"] == 99 for e in payload["entries"])

    def test_pairing_can_be_forced_either_way(self, tmp_path: Path) -> None:
        scores = dict.fromkeys(range(60), 0.5)
        write_run(tmp_path / "runs", "aa", scores)
        write_run(tmp_path / "runs", "bb", dict(scores))
        forced_off = run_script(
            tmp_path / "runs",
            tmp_path / "a.json",
            "--stage",
            "official",
            "--pair-items",
            "off",
        )
        assert forced_off["scoring"]["item_pairing"] == "per-team"
        forced_on = run_script(
            tmp_path / "runs",
            tmp_path / "b.json",
            "--stage",
            "provisional",
            "--pair-items",
            "on",
        )
        assert forced_on["scoring"]["item_pairing"] == "intersection"

    def test_rank_is_positional_and_ties_never_merge_ranks(
        self, tmp_path: Path
    ) -> None:
        """Ranking is on raw score even when the gap is inside noise; the tie flag
        is information, not a ranking rule."""
        scores = {i: (i % 10) / 10 for i in range(200)}
        nudged = {i: min(1.0, v + 0.005) for i, v in scores.items()}
        write_run(tmp_path / "runs", "lower", scores)
        write_run(tmp_path / "runs", "higher", nudged)
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        assert [e["rank"] for e in payload["entries"]] == [1, 2]
        assert payload["entries"][0]["team"] == "higher"
        assert payload["entries"][0]["tied_with_ranks"] == [2]


class TestResubmissionDelta:
    """A raw delta without a significance verdict invites reporting noise as
    progress: at n~300 a change under ~4 points is not measurable."""

    @staticmethod
    def _first_run(tmp_path: Path, score: float) -> Path:
        write_run(tmp_path / "runs", "solo", dict.fromkeys(range(300), score))
        write_run(tmp_path / "runs", "other", dict.fromkeys(range(300), 0.5))
        out = tmp_path / "prev.json"
        run_script(tmp_path / "runs", out, "--stage", "provisional")
        return out

    def _resubmit(self, tmp_path: Path, new_score: float, prev: Path) -> dict:
        write_run(tmp_path / "runs", "solo", dict.fromkeys(range(300), new_score))
        payload = run_script(
            tmp_path / "runs",
            tmp_path / "now.json",
            "--stage",
            "provisional",
            "--previous",
            str(prev),
        )
        return next(e for e in payload["entries"] if e["team"] == "solo")

    def test_large_gain_is_reported_as_improvement(self, tmp_path: Path) -> None:
        prev = self._first_run(tmp_path, 0.40)
        entry = self._resubmit(tmp_path, 0.60, prev)
        assert entry["delta_verdict"] == "improved"
        assert entry["delta_significant"] is True
        assert entry["previous_score"] == pytest.approx(0.40, abs=1e-3)

    def test_small_gain_is_not_reported_as_improvement(self, tmp_path: Path) -> None:
        prev = self._first_run(tmp_path, 0.40)
        entry = self._resubmit(tmp_path, 0.41, prev)
        assert entry["delta_verdict"] == "no significant change"
        assert entry["delta_significant"] is False

    def test_large_loss_is_reported_as_regression(self, tmp_path: Path) -> None:
        prev = self._first_run(tmp_path, 0.60)
        entry = self._resubmit(tmp_path, 0.40, prev)
        assert entry["delta_verdict"] == "regressed"

    def test_absent_previous_run_yields_no_delta_fields(self, tmp_path: Path) -> None:
        scores = dict.fromkeys(range(80), 0.5)
        write_run(tmp_path / "runs", "aa", scores)
        write_run(tmp_path / "runs", "bb", dict(scores))
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        assert "delta" not in payload["entries"][0]
        # The threshold is still published so the UI can explain the rule.
        assert payload["scoring"]["delta_threshold"] > 0


class TestUnusableSubmissions:
    """One team's broken run must not take the whole live board down with it,
    and must not quietly vanish from the official one either."""

    @staticmethod
    def _corrupt(root: Path, team: str, dataset: str = "ds") -> None:
        d = root / team
        d.mkdir(parents=True, exist_ok=True)
        (d / f"results_{dataset}.json").write_text('[{"sample_id": 0, "inp')

    def test_provisional_skips_and_reports_a_corrupt_file(self, tmp_path: Path) -> None:
        write_run(tmp_path / "runs", "good", dict.fromkeys(range(50), 0.5))
        write_run(tmp_path / "runs", "also-good", dict.fromkeys(range(50), 0.4))
        self._corrupt(tmp_path / "runs", "truncated")
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        assert [e["team"] for e in payload["entries"]] == ["good", "also-good"]
        assert [u["team"] for u in payload["unusable_submissions"]] == ["truncated"]
        assert "not valid JSON" in payload["unusable_submissions"][0]["reason"]
        assert any("could not be scored" in n for n in payload["notes"])

    def test_provisional_skips_a_run_with_nothing_gradeable(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path / "runs", "good", dict.fromkeys(range(40), 0.5))
        write_run(tmp_path / "runs", "good2", dict.fromkeys(range(40), 0.5))
        write_run(tmp_path / "runs", "allfail", dict.fromkeys(range(40), None))
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        assert {e["team"] for e in payload["entries"]} == {"good", "good2"}
        assert payload["unusable_submissions"][0]["team"] == "allfail"
        assert "no item was gradeable" in payload["unusable_submissions"][0]["reason"]

    def test_provisional_skips_a_missing_results_file(self, tmp_path: Path) -> None:
        write_run(tmp_path / "runs", "good", dict.fromkeys(range(30), 0.5))
        write_run(tmp_path / "runs", "good2", dict.fromkeys(range(30), 0.5))
        (tmp_path / "runs" / "never-ran").mkdir(parents=True)
        payload = run_script(
            tmp_path / "runs", tmp_path / "lb.json", "--stage", "provisional"
        )
        assert payload["unusable_submissions"][0]["team"] == "never-ran"
        assert "missing" in payload["unusable_submissions"][0]["reason"]

    def test_official_refuses_rather_than_dropping_a_team(self, tmp_path: Path) -> None:
        """Money depends on the official table, so a human looks before it ships."""
        write_run(tmp_path / "runs", "good", dict.fromkeys(range(40), 0.5))
        self._corrupt(tmp_path / "runs", "truncated")
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "lb.json"),
                "--stage",
                "official",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "truncated" in proc.stderr

    def test_on_unusable_overrides_the_stage_default(self, tmp_path: Path) -> None:
        write_run(tmp_path / "runs", "good", dict.fromkeys(range(40), 0.5))
        write_run(tmp_path / "runs", "good2", dict.fromkeys(range(40), 0.5))
        self._corrupt(tmp_path / "runs", "truncated")
        forced_skip = run_script(
            tmp_path / "runs",
            tmp_path / "a.json",
            "--stage",
            "official",
            "--on-unusable",
            "skip",
        )
        assert len(forced_skip["unusable_submissions"]) == 1
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "b.json"),
                "--stage",
                "provisional",
                "--on-unusable",
                "fail",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0

    def test_every_submission_unusable_is_always_an_error(self, tmp_path: Path) -> None:
        self._corrupt(tmp_path / "runs", "a")
        self._corrupt(tmp_path / "runs", "b")
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--runs",
                str(tmp_path / "runs"),
                "--dataset",
                "ds",
                "--out",
                str(tmp_path / "lb.json"),
                "--stage",
                "provisional",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "no scorable submission" in proc.stderr
