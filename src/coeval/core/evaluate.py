"""Metric evaluation — deterministic metrics run inline, LLM metrics await the judge."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from deepeval.metrics.utils import copy_metrics

from coeval.core.types import BaseMetric, TestCase
from coeval.metrics.base import DeterministicMetric

logger = logging.getLogger(__name__)

EvalLookup = dict[int, list["MetricScore"]]


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

    # ponytail: sequential — no dataset configures more than one LLM metric today.
    async with semaphore:
        for metric in copy_metrics(llm_metrics):
            try:
                await metric.a_measure(test_case, _show_indicator=False)
            except Exception as e:
                if not ignore_errors:
                    raise
                logger.error("Metric %s failed: %s", metric.__name__, e)
                scores.append(MetricScore.from_error(metric, str(e)))
                continue
            scores.append(MetricScore.from_metric(metric))

    return scores


async def a_evaluate(
    test_cases: Sequence[TestCase],
    metrics: Sequence[BaseMetric],
    max_concurrent: int = 10,
    ignore_errors: bool = True,
) -> EvalLookup:
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await asyncio.gather(
        *[a_evaluate_one(tc, metrics, semaphore, ignore_errors) for tc in test_cases]
    )
    return dict(zip((_get_sample_id(tc) for tc in test_cases), results, strict=True))
