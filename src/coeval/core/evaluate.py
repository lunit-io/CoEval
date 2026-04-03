"""Metric evaluation — deterministic metrics run directly, LLM metrics via deepeval."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from deepeval.evaluate import evaluate as _deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

from coeval.core.types import BaseMetric, TestCase
from coeval.metrics.base import DeterministicMetric

logger = logging.getLogger(__name__)
logging.getLogger("deepeval.evaluate.execute").setLevel(logging.WARNING)

EvalLookup = dict[int, list["MetricScore"]]


def _get_sample_id(obj: object) -> int:
    """Extract _sample_id from a test case or test result's additional_metadata."""
    metadata = getattr(obj, "additional_metadata", None) or {}
    sample_id = metadata.get("_sample_id")
    if sample_id is None:
        raise KeyError(
            f"_sample_id missing from additional_metadata on {type(obj).__name__}. "
            "Ensure build_test_cases() injects _sample_id."
        )
    return int(sample_id)


@dataclass
class MetricScore:
    """Single metric result for one sample."""

    name: str
    score: float | None
    success: bool
    reason: str | None = None
    error: str | None = None

    @classmethod
    def from_metric_data(cls, md: object) -> MetricScore:
        return cls(
            name=md.name,
            score=md.score,
            success=md.success,
            reason=md.reason,
            error=md.error,
        )


def _evaluate_deterministic(
    test_cases: list[TestCase],
    metrics: list[DeterministicMetric],
    ignore_errors: bool,
) -> EvalLookup:
    """Run deterministic metrics directly — no deepeval overhead."""
    lookup: EvalLookup = defaultdict(list)
    for tc in test_cases:
        sid = _get_sample_id(tc)
        for metric in metrics:
            try:
                metric.measure(tc)
            except Exception as e:
                if not ignore_errors:
                    raise
                lookup[sid].append(
                    MetricScore(
                        name=metric.__name__, score=None, success=False, error=str(e)
                    )
                )
                continue
            lookup[sid].append(
                MetricScore(
                    name=metric.__name__,
                    score=metric.score,
                    success=metric.is_successful(),
                    reason=metric.reason,
                )
            )
    return lookup


def _evaluate_nondeterministic(
    test_cases: list[TestCase],
    metrics: list[BaseMetric],
    max_concurrent: int,
    ignore_errors: bool,
) -> EvalLookup:
    """Run non-deterministic metrics via deepeval.evaluate()."""
    result = _deepeval_evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=True, max_concurrent=max_concurrent),
        display_config=DisplayConfig(
            show_indicator=True, print_results=False, verbose_mode=False
        ),
        error_config=ErrorConfig(ignore_errors=ignore_errors),
    )
    return {
        _get_sample_id(tr): [MetricScore.from_metric_data(md) for md in tr.metrics_data]
        for tr in result.test_results
    }


def evaluate(
    test_cases: list[TestCase],
    metrics: Sequence[BaseMetric],
    max_concurrent: int = 10,
    ignore_errors: bool = True,
) -> EvalLookup:
    """Evaluate metrics and return per-sample scores.

    Deterministic metrics run directly (fast, no deepeval overhead).
    LLM-based metrics run through deepeval.evaluate().
    """
    deterministic = [m for m in metrics if isinstance(m, DeterministicMetric)]
    nondeterministic = [m for m in metrics if not isinstance(m, DeterministicMetric)]

    lookup: EvalLookup = defaultdict(list)

    if deterministic:
        for sid, scores in _evaluate_deterministic(
            test_cases, deterministic, ignore_errors
        ).items():
            lookup[sid].extend(scores)

    if nondeterministic:
        for sid, scores in _evaluate_nondeterministic(
            test_cases, nondeterministic, max_concurrent, ignore_errors
        ).items():
            lookup[sid].extend(scores)

    return dict(lookup)
