"""Tests for CoEval's per-sample metric dispatch."""

import asyncio

import pytest

from coeval.core.evaluate import a_evaluate, a_evaluate_one, gather_or_cancel
from coeval.core.types import BaseMetric, LLMTestCase
from coeval.metrics.base import DeterministicMetric


class _FakeJudge(BaseMetric):
    """LLM-style metric: scores from metadata after an await, copy_metrics-safe."""

    def __init__(self, delay: float = 0.0, fail: bool = False, _semaphore=None) -> None:
        self.delay = delay
        self.fail = fail
        self._semaphore = _semaphore
        self.threshold = 0.5
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None

    @property
    def __name__(self) -> str:
        return "fake judge"

    def measure(self, test_case, *args, **kwargs) -> float:
        raise NotImplementedError

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        _SEEN_SEMAPHORES.append(self._semaphore)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("judge exploded")
        self.score = test_case.additional_metadata["want"]
        self.reason = f"scored {self.score}"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return bool(self.success)


class _FakeDeterministic(DeterministicMetric):
    @property
    def __name__(self) -> str:
        return "fake deterministic"

    def measure(self, test_case, *args, **kwargs) -> float:
        return self._set_result(test_case, is_correct=True, reason="inline", details={})


_SEEN_SEMAPHORES: list[object] = []


def _case(sample_id: int, want: float) -> LLMTestCase:
    return LLMTestCase(
        input=f"q{sample_id}",
        actual_output=f"a{sample_id}",
        additional_metadata={"_sample_id": sample_id, "want": want},
    )


@pytest.mark.asyncio
async def test_llm_metric_state_is_isolated_per_sample() -> None:
    """Concurrent samples must not clobber each other's score.

    Sample 0 is slower, so sample 1 finishes mid-flight — the interleave that
    would corrupt scores if the metric instance were shared.
    """
    template = _FakeJudge(delay=0.02)
    lookup = await a_evaluate(
        [_case(0, 0.25), _case(1, 0.75)], [template], max_concurrent=2
    )

    assert [s.score for s in lookup[0]] == [0.25]
    assert [s.score for s in lookup[1]] == [0.75]
    assert lookup[0][0].success is False
    assert lookup[1][0].success is True
    assert template.score is None, "template metric must stay unmeasured"


@pytest.mark.asyncio
async def test_metric_owned_semaphore_survives_the_copy() -> None:
    """A metric holding its own semaphore stays globally rate-limited.

    HealthBenchRubricMetric relies on this to cap judge calls across samples.
    """
    _SEEN_SEMAPHORES.clear()
    shared = asyncio.Semaphore(1)
    template = _FakeJudge(_semaphore=shared)

    await a_evaluate([_case(0, 1.0), _case(1, 1.0)], [template], max_concurrent=2)

    assert len(_SEEN_SEMAPHORES) == 2
    assert all(sem is shared for sem in _SEEN_SEMAPHORES)


@pytest.mark.asyncio
async def test_failing_metric_is_recorded_not_raised() -> None:
    scores = await a_evaluate_one(
        _case(0, 1.0), [_FakeJudge(fail=True)], asyncio.Semaphore(1)
    )

    assert len(scores) == 1
    assert scores[0].score is None
    assert scores[0].success is False
    assert "judge exploded" in scores[0].error


@pytest.mark.asyncio
async def test_failing_metric_raises_when_errors_not_ignored() -> None:
    with pytest.raises(RuntimeError, match="judge exploded"):
        await a_evaluate_one(
            _case(0, 1.0),
            [_FakeJudge(fail=True)],
            asyncio.Semaphore(1),
            ignore_errors=False,
        )


@pytest.mark.asyncio
async def test_deterministic_and_llm_metrics_both_reported() -> None:
    scores = await a_evaluate_one(
        _case(0, 1.0),
        [_FakeDeterministic(), _FakeJudge()],
        asyncio.Semaphore(1),
    )

    assert {s.name: s.score for s in scores} == {
        "fake deterministic": 1.0,
        "fake judge": 1.0,
    }


@pytest.mark.asyncio
async def test_hung_metric_times_out_instead_of_stalling_the_run() -> None:
    """A metric that never returns must not hold its semaphore slot forever."""

    class _Hangs(_FakeJudge):
        async def a_measure(self, test_case, *args, **kwargs) -> float:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    scores = await asyncio.wait_for(
        a_evaluate_one(
            _case(0, 1.0), [_Hangs()], asyncio.Semaphore(1), metric_timeout_s=0.05
        ),
        timeout=5,
    )

    assert scores[0].score is None
    assert "TimeoutError" in scores[0].error


@pytest.mark.asyncio
async def test_failing_task_cancels_its_siblings() -> None:
    """gather_or_cancel must stop the other samples, not leave them spending calls."""
    finished: list[str] = []

    async def slow() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            finished.append("cancelled")
            raise
        finished.append("completed")
        return "slow"

    async def boom() -> str:
        await asyncio.sleep(0.01)
        raise RuntimeError("first failure")

    with pytest.raises(RuntimeError, match="first failure"):
        await gather_or_cancel([slow(), boom()])

    assert finished == ["cancelled"]


@pytest.mark.asyncio
async def test_multiple_llm_metrics_run_concurrently() -> None:
    """Two LLM metrics on one test case must overlap, not serialise."""

    started = asyncio.get_running_loop().time()
    scores = await a_evaluate_one(
        _case(0, 1.0),
        [_FakeJudge(delay=0.2), _FakeJudge(delay=0.2)],
        asyncio.Semaphore(1),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert len(scores) == 2
    assert elapsed < 0.35, f"serialised: {elapsed:.2f}s for 2 x 0.2s metrics"
