import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

import coeval.main as main_module
import coeval.util.summary as summary_module
from coeval.core.schema import EvalSummary


def _summary(
    dataset: str,
    *,
    num_samples: int = 2,
    num_passed: int = 0,
    num_inference_failed: int = 0,
    num_scoring_failed: int = 0,
) -> EvalSummary:
    return EvalSummary(
        dataset=dataset,
        num_samples=num_samples,
        num_passed=num_passed,
        num_inference_failed=num_inference_failed,
        num_scoring_failed=num_scoring_failed,
        total_time_s=1.0,
        avg_generation_ms=1.0,
        avg_scoring_ms=1.0,
        metric_scores={},
    )


@pytest.mark.parametrize(
    ("num_inference_failed", "num_scoring_failed"),
    [(2, 0), (0, 2)],
)
def test_main_exits_nonzero_when_no_samples_are_evaluated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    num_inference_failed: int,
    num_scoring_failed: int,
) -> None:
    summaries = [
        _summary(
            "failed",
            num_inference_failed=num_inference_failed,
            num_scoring_failed=num_scoring_failed,
        )
    ]
    monkeypatch.setattr(
        main_module, "run_evaluation", AsyncMock(return_value=summaries)
    )
    monkeypatch.setattr(main_module, "_build_display", lambda _cfg, value: value)
    monkeypatch.setattr(main_module, "console", MagicMock())
    monkeypatch.setattr(
        summary_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(run=SimpleNamespace(dir=str(tmp_path))),
    )
    cfg = OmegaConf.create({"datasets": {}, "runner": {"output_dir": "enabled"}})

    with pytest.raises(SystemExit) as exc_info:
        main_module.main.__wrapped__(cfg)

    assert exc_info.value.code == 1
    assert (tmp_path / "summary_combined.json").exists()


def test_main_keeps_zero_exit_when_any_sample_is_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries = [
        _summary(
            "partial",
            num_inference_failed=1,
        )
    ]
    monkeypatch.setattr(
        main_module, "run_evaluation", AsyncMock(return_value=summaries)
    )
    monkeypatch.setattr(main_module, "_build_display", lambda _cfg, value: value)
    monkeypatch.setattr(main_module, "console", MagicMock())
    cfg = OmegaConf.create({"datasets": {}, "runner": {"output_dir": None}})

    assert main_module.main.__wrapped__(cfg) is None


def test_main_exits_nonzero_when_no_dataset_has_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "run_evaluation", AsyncMock(return_value=[]))
    monkeypatch.setattr(main_module, "_build_display", lambda _cfg, value: value)
    monkeypatch.setattr(main_module, "console", MagicMock())
    cfg = OmegaConf.create({"datasets": {}, "runner": {"output_dir": None}})

    with pytest.raises(SystemExit) as exc_info:
        main_module.main.__wrapped__(cfg)

    assert exc_info.value.code == 1


def test_combined_summary_includes_failure_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        summary_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(run=SimpleNamespace(dir=str(tmp_path))),
    )
    summaries = [
        _summary("a", num_inference_failed=1),
        _summary("b", num_scoring_failed=2),
    ]

    output_path = summary_module.save_combined_summary(summaries)

    assert output_path is not None
    combined = json.loads(output_path.read_text())
    assert combined["total_inference_failed"] == 1
    assert combined["total_scoring_failed"] == 2
    assert combined["per_dataset"]["a"]["num_inference_failed"] == 1
    assert combined["per_dataset"]["a"]["num_scoring_failed"] == 0
    assert combined["per_dataset"]["b"]["num_inference_failed"] == 0
    assert combined["per_dataset"]["b"]["num_scoring_failed"] == 2
