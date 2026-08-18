"""
Score Aggregation Functions for Evaluation.

Provides pluggable aggregation strategies for computing metric scores
across evaluation results. Configured at the dataset level in YAML.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from statistics import fmean
from typing import Any

from hydra.utils import instantiate
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

from coeval.core.schema import (
    EvalResult,
    EvalSummary,  # noqa: E402
    MetricScoreDetail,
)


@dataclass
class AggregationResult:
    """Result from score aggregation."""

    metric_scores: dict[str, MetricScoreDetail] = field(default_factory=dict)
    breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)


ScoreAggregatorFn = Callable[[list[EvalResult]], AggregationResult]

EMPTY_RESULT = AggregationResult()
UNPARSEABLE_LABEL = "UNPARSEABLE"


def avg_aggregator(
    results: list[EvalResult],
    key_to_name: dict[str, str] | None = None,  # noqa: ARG001
) -> AggregationResult:
    """Simple average: Σ(score) / n for each metric.

    Stores ``numerator`` / ``denominator`` for cross-dataset merging and
    produces a ``breakdown`` with per-group statistics.
    """
    if not results:
        return EMPTY_RESULT

    scores: dict[str, list[float]] = defaultdict(list)
    for m in chain.from_iterable(r.metrics for r in results):
        if (
            m.score is not None
        ):  # None = judge infrastructure failure, excluded from aggregate
            scores[m.name].append(m.score)

    metric_scores: dict[str, MetricScoreDetail] = {}
    breakdown: dict[str, dict[str, Any]] = {}

    for name, s in scores.items():
        n = len(s)
        mean = fmean(s)
        metric_scores[name] = MetricScoreDetail(
            score=mean,
            numerator=sum(s),
            denominator=float(n),
        )
        breakdown[name] = {
            "mean": mean,
            "n_samples": n,
            "min_score": min(s),
            "max_score": max(s),
        }

    return AggregationResult(metric_scores=metric_scores, breakdown=breakdown)


def clipped_avg_aggregator(
    results: list[EvalResult],
    key_to_name: dict[str, str] | None = None,  # noqa: ARG001
) -> AggregationResult:
    """Simple average with the reported mean clipped to [0, 1].

    The official HealthBench reporting metric, for subsets whose penalty criteria
    can drive a per-example score (and so the mean) net-negative.

    Wraps :func:`avg_aggregator` and clips each metric's ``score``.
    ``numerator``/``denominator`` are rescaled to match it
    (``numerator == score * denominator``) so a :func:`weighted_merge`
    recomputing ``sum(num) / sum(den)`` cannot undo the clip.  ``breakdown`` and
    the per-sample scores stay raw as diagnostics.
    """
    base = avg_aggregator(results)
    clipped = AggregationResult(metric_scores={}, breakdown=dict(base.breakdown))
    for name, detail in base.metric_scores.items():
        score = min(1.0, max(0.0, detail.score))
        den = detail.denominator
        clipped.metric_scores[name] = MetricScoreDetail(
            score=score,
            numerator=(score * den) if den is not None else None,
            denominator=den,
        )
    return clipped


def weighted_avg_aggregator(
    results: list[EvalResult],
    key_to_name: dict[str, str] | None = None,  # noqa: ARG001
) -> AggregationResult:
    """
    Weighted average: Σ(score * weight) / Σ(weight) for each metric.

    Reads weight from metric.details["weight"] (defaults to 1.0).
    Stores numerator/denominator for proper cross-dataset merging.

    Use case: HealthBench themes with varying sample importance.
    """
    if not results:
        return EMPTY_RESULT

    numerators: dict[str, float] = defaultdict(float)
    denominators: dict[str, float] = defaultdict(float)

    for m in chain.from_iterable(r.metrics for r in results):
        if (
            m.score is None
        ):  # judge infrastructure failure — exclude, same as avg_aggregator
            continue
        weight = m.details.get("weight", 1.0) if m.details else 1.0
        numerators[m.name] += m.score * weight
        denominators[m.name] += weight

    return AggregationResult(
        metric_scores={
            name: MetricScoreDetail(
                score=numerators[name] / denominators[name]
                if denominators[name]
                else 0.0,
                numerator=numerators[name],
                denominator=denominators[name],
            )
            for name in numerators
        }
    )


def f1_aggregator(
    results: list[EvalResult],
    key_to_name: dict[str, str] | None = None,  # noqa: ARG001
) -> AggregationResult:
    """Macro-averaged F1 for classification with detailed breakdown."""
    if not results:
        return EMPTY_RESULT

    pairs = _extract_classification_pairs(results)
    if not pairs:
        return EMPTY_RESULT

    metric_scores: dict[str, MetricScoreDetail] = {}
    breakdown: dict[str, dict[str, Any]] = {}

    for name, (y_true, y_pred) in pairs.items():
        true_labels = sorted(set(y_true))
        metric_scores[f"{name}_f1"] = MetricScoreDetail(
            score=float(
                f1_score(
                    y_true, y_pred, labels=true_labels, average="macro", zero_division=0
                )
            )
        )
        breakdown[name] = {
            "classification_report": classification_report(
                y_true, y_pred, labels=true_labels, output_dict=True, zero_division=0
            ),
            "n_unparseable": sum(1 for p in y_pred if p == UNPARSEABLE_LABEL),
        }

    return AggregationResult(metric_scores=metric_scores, breakdown=breakdown)


def _extract_classification_pairs(
    results: list[EvalResult],
) -> dict[str, tuple[list[str], list[str]]]:
    """
    Extract (y_true, y_pred) pairs from classification results.

    Returns:
        Dict mapping metric name to (y_true, y_pred) lists.
    """
    y_true: dict[str, list[str]] = defaultdict(list)
    y_pred: dict[str, list[str]] = defaultdict(list)

    for m in chain.from_iterable(r.metrics for r in results):
        if not m.details:
            continue
        expected = m.details.get("expected")
        if expected is None:
            continue
        predicted = m.details.get("predicted")
        y_true[m.name].append(str(expected))
        y_pred[m.name].append(
            str(predicted) if predicted is not None else UNPARSEABLE_LABEL
        )

    return {name: (y_true[name], y_pred[name]) for name in y_true}


def ordinal_classification_aggregator(
    results: list[EvalResult],
    label_order: list[str] | None = None,
    key_to_name: dict[str, str] | None = None,  # noqa: ARG001
) -> AggregationResult:
    """Comprehensive classification aggregator with ordinal metrics.

    Args:
        results: Evaluation results with classification details.
        label_order: Explicit ordinal ordering of labels.
            When ``None``, labels are assumed to be numeric strings.
    """
    if not results:
        return EMPTY_RESULT

    pairs = _extract_classification_pairs(results)
    if not pairs:
        return EMPTY_RESULT

    rank = {label: i for i, label in enumerate(label_order)} if label_order else None

    metric_scores: dict[str, MetricScoreDetail] = {}
    breakdown: dict[str, dict[str, Any]] = {}

    for name, (y_true, y_pred) in pairs.items():
        if len(y_true) < 2:
            continue

        n = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
        accuracy = correct / n

        # Adjacent accuracy (±1 ordinal distance)
        adjacent_correct = 0
        for t, p in zip(y_true, y_pred, strict=True):
            try:
                dist = abs(rank[t] - rank[p]) if rank else abs(int(t) - int(p))
            except (KeyError, ValueError):
                continue
            if dist <= 1:
                adjacent_correct += 1
        adjacent_accuracy = adjacent_correct / n

        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        linear_kappa = float(cohen_kappa_score(y_true, y_pred, weights="linear"))
        quadratic_kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

        # Accuracy: exact merging via correct_count / n
        metric_scores["accuracy"] = MetricScoreDetail(
            score=accuracy,
            numerator=float(correct),
            denominator=float(n),
        )

        metric_scores["adjacent_accuracy"] = MetricScoreDetail(
            score=adjacent_accuracy,
            numerator=float(adjacent_correct),
            denominator=float(n),
        )

        # F1 & kappa: sample-weighted approximation for merging
        for key, val in [
            ("f1_macro", f1),
            ("cohens_kappa", kappa),
            ("linear_weighted_kappa", linear_kappa),
            ("quadratic_weighted_kappa", quadratic_kappa),
        ]:
            metric_scores[key] = MetricScoreDetail(
                score=val,
                numerator=val * n,
                denominator=float(n),
            )

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        breakdown[name] = {
            "accuracy": accuracy,
            "adjacent_accuracy": adjacent_accuracy,
            "f1_macro": f1,
            "cohens_kappa": kappa,
            "linear_weighted_kappa": linear_kappa,
            "quadratic_weighted_kappa": quadratic_kappa,
            "n_samples": n,
            "classification_report": report,
        }

    return AggregationResult(metric_scores=metric_scores, breakdown=breakdown)


def _filter_results(
    results: list[EvalResult],
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[EvalResult]:
    """Filter EvalResult list to include/exclude specific metric names.

    Creates new EvalResult instances with filtered metric lists.
    Results with no remaining metrics are dropped.
    """
    filtered: list[EvalResult] = []
    for r in results:
        metrics = [
            m
            for m in r.metrics
            if (include is None or m.name in include)
            and (exclude is None or m.name not in exclude)
        ]
        if metrics:
            filtered.append(r.model_copy(update={"metrics": metrics}))
    return filtered


def composite_aggregator(
    results: list[EvalResult],
    pipelines: Sequence,
    key_to_name: dict[str, str] | None = None,
) -> AggregationResult:
    """Compose multiple aggregators with metric routing.

    Each pipeline entry is an aggregator config with an optional ``metrics``
    list for routing.  Entries without ``metrics`` act as catch-alls,
    receiving all metrics not explicitly routed to other entries.

    Supports two pipeline entry formats:
        - **nested** (test-friendly): ``{"aggregator": <callable>, "metrics": [...]}``
        - **flat** (Hydra YAML with ``_recursive_: false``): config dict with
          ``_target_`` + ``_partial_: true`` + optional ``metrics``.

    Args:
        results: Evaluation results.
        pipelines: List of pipeline entries.
        key_to_name: Mapping from YAML config keys to runtime metric
            ``__name__`` values (e.g. ``{"classification": "Classification"}``).
    """
    if not results:
        return EMPTY_RESULT

    parsed: list[tuple[ScoreAggregatorFn, list[str] | None]] = []
    for entry in pipelines:
        raw = entry.get("metrics")
        metrics = [key_to_name[m] for m in raw] if raw and key_to_name else raw
        agg_fn = entry.get("aggregator") or instantiate(
            {k: v for k, v in entry.items() if k != "metrics"}
        )
        parsed.append((agg_fn, metrics))

    # Collect explicitly routed metric names
    routed: set[str] = {n for _, names in parsed if names for n in names}
    merged = AggregationResult(metric_scores={}, breakdown={})

    for agg_fn, metric_names in parsed:
        is_routed = metric_names is not None
        is_catch_all = not is_routed and len(routed) > 0

        if is_routed:
            subset = _filter_results(results, include=set(metric_names))
        elif is_catch_all:
            subset = _filter_results(results, exclude=routed)
        else:
            subset = results

        sub = agg_fn(subset)
        merged.metric_scores.update(sub.metric_scores)
        merged.breakdown.update(sub.breakdown)

    return merged


default_aggregator = avg_aggregator


def weighted_merge(details: list[MetricScoreDetail]) -> MetricScoreDetail:
    """Merge using numerator/denominator when available, else simple mean."""
    if all(d.numerator is not None and d.denominator is not None for d in details):
        total_num = sum(d.numerator for d in details)  # type: ignore[arg-type]
        total_den = sum(d.denominator for d in details)  # type: ignore[arg-type]
        return MetricScoreDetail(
            score=total_num / total_den if total_den else 0.0,
            numerator=total_num,
            denominator=total_den,
        )
    return MetricScoreDetail(score=fmean(d.score for d in details))


def simple_merge(details: list[MetricScoreDetail]) -> MetricScoreDetail:
    """Merge using simple (unweighted) mean of scores."""
    return MetricScoreDetail(score=fmean(d.score for d in details))


def merge_summaries(
    summaries: list[EvalSummary],
    group_name: str,
    merge_metric: Callable[[list[MetricScoreDetail]], MetricScoreDetail] = simple_merge,
    **kwargs: object,  # noqa: ARG001 — absorbed by Hydra
) -> EvalSummary:
    """Merge multiple EvalSummary objects into one."""
    if len(summaries) == 1:
        return summaries[0].model_copy(update={"dataset": group_name})

    all_names = list(
        dict.fromkeys(chain.from_iterable(s.metric_scores for s in summaries))
    )

    merged = {
        name: merge_metric(
            [s.metric_scores[name] for s in summaries if name in s.metric_scores]
        )
        for name in all_names
    }

    total_samples = sum(s.num_samples for s in summaries)
    return EvalSummary(
        dataset=group_name,
        num_samples=total_samples,
        num_passed=sum(s.num_passed for s in summaries),
        total_time_s=sum(s.total_time_s for s in summaries),
        avg_generation_ms=(
            sum(s.avg_generation_ms * s.num_samples for s in summaries) / total_samples
            if total_samples
            else 0
        ),
        avg_scoring_ms=(
            sum(s.avg_scoring_ms * s.num_samples for s in summaries) / total_samples
            if total_samples
            else 0
        ),
        metric_scores=merged,
        num_inference_failed=sum(s.num_inference_failed for s in summaries),
        num_scoring_failed=sum(s.num_scoring_failed for s in summaries),
    )
