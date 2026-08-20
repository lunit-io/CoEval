"""Metric evaluation — deterministic metrics run inline, LLM metrics await the judge."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass

from deepeval.metrics.utils import copy_metrics

from coeval.core.types import BaseMetric, TestCase
from coeval.metrics.base import DeterministicMetric

logger = logging.getLogger(__name__)

EvalLookup = dict[int, list["MetricScore"]]

# Ceiling on a single LLM metric call. deepeval's dispatcher applied a per-test-case
# deadline; without one, a metric that never returns holds its semaphore slot forever
# and the whole run stalls (observed once: a stuck judge call hung a run for an hour).
DEFAULT_METRIC_TIMEOUT_S = 900.0


def _get_sample_id(obj: object) -> int:
    metadata = getattr(obj, "additional_metadata", None) or {}
    sample_id = metadata.get("_sample_id")
    if sample_id is None:
        raise KeyError(
            f"_sample_id missing from additional_metadata on {type(obj).__name__}. "
            "Ensure build_test_case() injects _sample_id."
        )
    return int(sample_id)


@dataclass
class MetricScore:
    name: str
    score: float | None
    success: bool
    reason: str | None = None
    error: str | None = None

    @classmethod
    def from_metric(cls, metric: BaseMetric) -> MetricScore:
        return cls(
            name=metric.__name__,
            score=metric.score,
            success=metric.is_successful(),
            reason=metric.reason,
            error=metric.error,
        )

    @classmethod
    def from_error(cls, metric: BaseMetric, error: str) -> MetricScore:
        return cls(name=metric.__name__, score=None, success=False, error=error)


async def a_evaluate_one(
    test_case: TestCase,
    metrics: Sequence[BaseMetric],
    semaphore: asyncio.Semaphore,
    ignore_errors: bool = True,
    metric_timeout_s: float | None = DEFAULT_METRIC_TIMEOUT_S,
) -> list[MetricScore]:
    """Score one test case; ``semaphore`` caps concurrent test cases.

    LLM metrics run on per-test-case copies because score/reason live on the
    instance. ``copy_metrics`` forwards ``__init__`` args, so a metric holding its
    own semaphore (``HealthBenchRubricMetric``) keeps sharing it and stays globally
    rate-limited.
    """
    scores: list[MetricScore] = []
    llm_metrics: list[BaseMetric] = []

    # ponytail: deterministic metrics reuse the shared instance — measure() has no
    # await, so concurrent samples can't interleave between it and reading .score.
    for metric in metrics:
        if not isinstance(metric, DeterministicMetric):
            llm_metrics.append(metric)
            continue
        try:
            metric.measure(test_case)
        except Exception as e:
            if not ignore_errors:
                raise
            scores.append(MetricScore.from_error(metric, str(e)))
            continue
        scores.append(MetricScore.from_metric(metric))

    if not llm_metrics:
        return scores

    async def measure(metric: BaseMetric) -> MetricScore:
        coro = metric.a_measure(test_case, _show_indicator=False)
        try:
            if metric_timeout_s is None:
                await coro
            else:
                await asyncio.wait_for(coro, timeout=metric_timeout_s)
        except Exception as e:
            if not ignore_errors:
                raise
            logger.error("Metric %s failed: %s", metric.__name__, e)
            return MetricScore.from_error(metric, f"{type(e).__name__}: {e}")
        return MetricScore.from_metric(metric)

    async with semaphore:
        copied = copy_metrics(llm_metrics)
        if len(copied) == 1:
            scores.append(await measure(copied[0]))
        else:
            scores.extend(await asyncio.gather(*[measure(m) for m in copied]))

    return scores


async def a_evaluate(
    test_cases: Sequence[TestCase],
    metrics: Sequence[BaseMetric],
    max_concurrent: int = 10,
    ignore_errors: bool = True,
    metric_timeout_s: float | None = DEFAULT_METRIC_TIMEOUT_S,
) -> EvalLookup:
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await gather_or_cancel(
        [
            a_evaluate_one(tc, metrics, semaphore, ignore_errors, metric_timeout_s)
            for tc in test_cases
        ]
    )
    return dict(zip((_get_sample_id(tc) for tc in test_cases), results, strict=True))


async def gather_or_cancel(coros: list[Coroutine]) -> list:
    """``asyncio.gather`` that cancels and drains siblings when one task fails.

    Plain ``gather`` re-raises immediately but leaves the others running, so a
    failure mid-run can keep spending judge calls after the caller gave up.
    """
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
