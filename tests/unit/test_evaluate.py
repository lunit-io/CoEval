"""Tests for CoEval's per-sample metric dispatch."""

import asyncio

import pytest

from coeval.core.evaluate import a_evaluate, a_evaluate_one
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
