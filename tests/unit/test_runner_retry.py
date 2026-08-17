"""Tests for candidate-inference retries and failure accounting."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from deepeval.test_case import LLMTestCase

from coeval.core.evaluate import MetricScore
from coeval.core.runner import EvalRunner
from coeval.core.schema import EvalResult, EvalSummary, MetricResult
from coeval.util.aggregation import avg_aggregator, merge_summaries
from coeval.util.console import EvalConsole


class _SequencedClient:
    """Inference fake that returns or raises each supplied outcome in order."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def generate(self, _messages: list[dict]) -> str:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        name="retry_dataset",
        goldens=[object()],
        get_generation_input=lambda _golden: [{"role": "user", "content": "q"}],
    )


@pytest.mark.asyncio
async def test_two_transient_failures_then_success_yields_successful_sample() -> None:
    """A third successful attempt must not be recorded as an inference failure."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), "recovered answer"]
    )
    runner = EvalRunner(client=client, concurrent_limit=1)

    predictions, metadata = await runner._generate_predictions(_dataset())

    assert client.calls == 3
    assert predictions == ["recovered answer"]
    assert metadata[0]["inference_failed"] is False
    assert metadata[0]["inference_error"] is None


@pytest.mark.asyncio
async def test_exhausted_attempts_record_final_exception() -> None:
    """Failure accounting must retain the exception from the final attempt."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), RuntimeError("final")]
    )
    runner = EvalRunner(client=client, concurrent_limit=1)

    predictions, metadata = await runner._generate_predictions(_dataset())

    assert client.calls == 3
    assert predictions == [""]
    assert metadata[0]["inference_failed"] is True
    assert metadata[0]["inference_error"] == "final"


@pytest.mark.asyncio
async def test_custom_attempt_count_retries_without_sleep_when_delay_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-delay retry must use the configured attempt count without sleeping."""
    client = _SequencedClient([RuntimeError("first"), "recovered answer"])
    runner = EvalRunner(
        client=client,
        concurrent_limit=1,
        inference_max_attempts=2,
        inference_retry_delay_s=0,
    )

    async def sleep_must_not_run(_delay: float) -> None:
        raise AssertionError("zero retry delay must not sleep")

    monkeypatch.setattr("coeval.core.runner.asyncio.sleep", sleep_must_not_run)

    assert (
        await runner._generate_with_retry([{"role": "user", "content": "q"}], 7)
        == "recovered answer"
    )
    assert client.calls == 2


@pytest.mark.asyncio
async def test_retry_delay_doubles_after_each_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries must wait longer after each consecutive transient failure."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), "recovered answer"]
    )
    runner = EvalRunner(
        client=client,
        concurrent_limit=1,
        inference_max_attempts=3,
        inference_retry_delay_s=0.25,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("coeval.core.runner.asyncio.sleep", sleep)

    result = await runner._generate_with_retry([{"role": "user", "content": "q"}], 7)

    assert result == "recovered answer"
    assert client.calls == 3
    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"inference_max_attempts": 0}, "inference_max_attempts"),
        ({"inference_retry_delay_s": -0.1}, "inference_retry_delay_s"),
    ],
)
def test_invalid_retry_configuration_is_rejected(
    kwargs: dict[str, float | int], message: str
) -> None:
    """Invalid retry bounds must fail before any inference is attempted."""
    with pytest.raises(ValueError, match=message):
        EvalRunner(client=_SequencedClient(["unused"]), **kwargs)


@pytest.mark.parametrize(
    ("metadata", "eval_lookup", "reason"),
    [
        (
            {
                "sample_id": 1,
                "inference_failed": True,
                "inference_error": "final",
            },
            {},
            "Skipped: inference failed",
        ),
        (
            {
                "sample_id": 1,
                "inference_failed": False,
                "inference_error": None,
            },
            {},
            "Missing eval result",
        ),
    ],
)
def test_unavailable_scores_are_none(
    metadata: dict, eval_lookup: dict, reason: str
) -> None:
    """Infrastructure failures must be excluded from score aggregation."""

    class Metric:
        __name__ = "metric"

    runner = EvalRunner(client=_SequencedClient(["unused"]))
    scores = runner._score_test_case(
        SimpleNamespace(additional_metadata={}), metadata, [Metric], eval_lookup
    )

    assert len(scores) == 1
    assert scores[0].score is None
    assert scores[0].passed is False
    assert scores[0].reason == reason


class _Metric:
    __name__ = "metric"


def _run_dataset(num_samples: int) -> SimpleNamespace:
    def build_test_cases(predictions: list[str]) -> list[LLMTestCase]:
        return [
            LLMTestCase(
                input=f"question {sample_id}",
                actual_output=prediction,
                additional_metadata={"_sample_id": sample_id},
            )
            for sample_id, prediction in enumerate(predictions)
        ]

    return SimpleNamespace(
        name="retry_dataset",
        goldens=list(range(num_samples)),
        get_generation_input=lambda golden: [{"role": "user", "content": str(golden)}],
        build_test_cases=build_test_cases,
    )


class _SelectiveClient:
    def __init__(self, failed_sample_ids: set[int]) -> None:
        self.failed_sample_ids = failed_sample_ids

    async def generate(self, messages: list[dict]) -> str:
        sample_id = int(messages[-1]["content"])
        if sample_id in self.failed_sample_ids:
            raise RuntimeError(f"inference failed for {sample_id}")
        return f"answer {sample_id}"


def _capture_built_results(
    runner: EvalRunner, monkeypatch: pytest.MonkeyPatch
) -> list[EvalResult]:
    captured: list[EvalResult] = []
    build_results = runner._build_eval_results

    def capture(*args, **kwargs) -> list[EvalResult]:
        results = build_results(*args, **kwargs)
        captured.extend(results)
        return results

    monkeypatch.setattr(runner, "_build_eval_results", capture)
    return captured


@pytest.mark.asyncio
async def test_run_skips_failed_inference_cases_during_judge_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed candidate remains in results but never reaches the Judge."""
    runner = EvalRunner(
        client=_SelectiveClient({0}),
        concurrent_limit=1,
        inference_max_attempts=1,
    )
    captured = _capture_built_results(runner, monkeypatch)
    evaluated_ids: list[int] = []

    def evaluate_scorable(test_cases, **_kwargs):
        evaluated_ids.extend(tc.additional_metadata["_sample_id"] for tc in test_cases)
        return {
            sample_id: [
                MetricScore(name="metric", score=1.0, success=True, reason="ok")
            ]
            for sample_id in evaluated_ids
        }

    monkeypatch.setattr("coeval.core.runner.evaluate", evaluate_scorable)

    summary = await runner.run(_run_dataset(2), [_Metric], avg_aggregator)

    assert evaluated_ids == [1]
    assert [result.sample_id for result in captured] == [0, 1]
    assert captured[0].inference_failed is True
    assert captured[0].metrics[0].score is None
    assert captured[1].passed is True
    assert summary.num_inference_failed == 1


@pytest.mark.asyncio
async def test_run_does_not_call_judge_when_all_inferences_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-failed candidate batch still produces one failure result per sample."""
    runner = EvalRunner(
        client=_SelectiveClient({0, 1}),
        concurrent_limit=1,
        inference_max_attempts=1,
    )
    captured = _capture_built_results(runner, monkeypatch)

    def evaluate_must_not_run(*_args, **_kwargs):
        raise AssertionError("Judge evaluation must be skipped")

    monkeypatch.setattr("coeval.core.runner.evaluate", evaluate_must_not_run)

    summary = await runner.run(_run_dataset(2), [_Metric], avg_aggregator)

    assert [result.sample_id for result in captured] == [0, 1]
    assert all(result.inference_failed for result in captured)
    assert all(result.metrics[0].score is None for result in captured)
    assert summary.num_inference_failed == 2
    assert summary.num_evaluated == 0


def _metadata(sample_id: int, *, inference_failed: bool = False) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "rationale": "",
        "generation_time_ms": 1.0,
        "inference_failed": inference_failed,
        "inference_error": "candidate failure" if inference_failed else None,
    }


def _test_case(sample_id: int) -> LLMTestCase:
    return LLMTestCase(
        input=f"question {sample_id}",
        actual_output="answer",
        additional_metadata={"_sample_id": sample_id},
    )


def test_build_eval_results_marks_only_judge_failures_as_scoring_failed() -> None:
    """Only a non-inference row with a missing top-level score is scoring-failed."""
    runner = EvalRunner(client=_SequencedClient(["unused"]))
    results = runner._build_eval_results(
        [_test_case(10), _test_case(20), _test_case(30)],
        [_metadata(10), _metadata(20, inference_failed=True), _metadata(30)],
        [_Metric],
        {
            10: [MetricScore(name="metric", score=1.0, success=True)],
            30: [MetricScore(name="metric", score=None, success=False)],
        },
    )

    assert [result.scoring_failed for result in results] == [False, False, True]


def _eval_result(
    sample_id: int,
    *,
    score: float | None,
    inference_failed: bool = False,
    scoring_failed: bool = False,
) -> EvalResult:
    return EvalResult(
        sample_id=sample_id,
        test_case=_test_case(sample_id),
        metrics=[
            MetricResult(
                name="metric",
                score=score,
                passed=score is not None and score >= 0.5,
            )
        ],
        generation_time_ms=1.0,
        scoring_time_ms=1.0,
        inference_failed=inference_failed,
        scoring_failed=scoring_failed,
    )


def test_summary_and_schema_exclude_scoring_failures_from_pass_rate() -> None:
    """Pass rate counts only rows with completed inference and scoring."""
    runner = EvalRunner(client=_SequencedClient(["unused"]))
    results = [
        _eval_result(0, score=1.0),
        _eval_result(1, score=None, inference_failed=True),
        _eval_result(2, score=None, scoring_failed=True),
    ]

    summary = runner._build_summary(results, "mixed", 3.0, avg_aggregator)

    assert results[2].passed is False
    assert results[2].to_dict()["scoring_failed"] is True
    assert summary.num_samples == 3
    assert summary.num_evaluated == 1
    assert summary.num_passed == 1
    assert summary.num_inference_failed == 1
    assert summary.num_scoring_failed == 1
    assert summary.pass_rate == 1.0
    assert summary.scoring_failure_rate == pytest.approx(1 / 3)
    assert summary.to_dict()["num_scoring_failed"] == 1


def _summary(
    dataset: str,
    *,
    num_samples: int,
    num_passed: int,
    num_inference_failed: int,
    num_scoring_failed: int,
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


def test_merge_summaries_sums_scoring_failures() -> None:
    """Merged summary denominators retain scoring failures from every shard."""
    merged = merge_summaries(
        [
            _summary(
                "a",
                num_samples=2,
                num_passed=1,
                num_inference_failed=1,
                num_scoring_failed=0,
            ),
            _summary(
                "b",
                num_samples=2,
                num_passed=0,
                num_inference_failed=0,
                num_scoring_failed=2,
            ),
        ],
        "merged",
    )

    assert merged.num_scoring_failed == 2
    assert merged.num_evaluated == 1
    assert merged.pass_rate == 1.0


def test_console_reports_scoring_failures_in_denominator_and_annotation() -> None:
    """Completion output uses the fully scored denominator and labels both failures."""
    eval_console = EvalConsole()
    eval_console._console = MagicMock()

    eval_console.end_eval(
        1,
        3,
        1.0,
        2.0,
        num_inference_failed=1,
        num_scoring_failed=1,
    )

    message = eval_console._console.print.call_args.args[0]
    assert "1/1" in message
    assert "1 inference failures" in message
    assert "1 scoring failures" in message
