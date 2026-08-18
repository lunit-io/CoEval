"""Evaluation orchestration and summary utilities."""

import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from statistics import fmean

from hydra.core.hydra_config import HydraConfig

from coeval.core.schema import EvalSummary
from coeval.util import console


def save_combined_summary(summaries: list[EvalSummary]) -> Path | None:
    """Build and save a combined summary from multiple dataset summaries."""
    if not summaries:
        return None

    output_dir = Path(HydraConfig.get().run.dir)
    combined_file = output_dir / "summary_combined.json"

    _HEALTHBENCH_SUBSET_PREFIXES = (
        "theme:",
        "physician_agreed_category:",
        "cluster:",
    )

    all_metrics: defaultdict[str, list[float]] = defaultdict(list)
    for name, detail in chain.from_iterable(s.metric_scores.items() for s in summaries):
        if not name.startswith(_HEALTHBENCH_SUBSET_PREFIXES):
            all_metrics[name].append(detail.score)

    combined = {
        "num_datasets": len(summaries),
        "total_samples": sum(s.num_samples for s in summaries),
        "total_passed": sum(s.num_passed for s in summaries),
        "total_inference_failed": sum(s.num_inference_failed for s in summaries),
        "total_scoring_failed": sum(s.num_scoring_failed for s in summaries),
        "total_time_s": round(sum(s.total_time_s for s in summaries), 2),
        "metric_scores": {name: fmean(scores) for name, scores in all_metrics.items()},
        "per_dataset": {
            s.dataset: {
                "num_samples": s.num_samples,
                "num_passed": s.num_passed,
                "num_inference_failed": s.num_inference_failed,
                "num_scoring_failed": s.num_scoring_failed,
                "total_time_s": round(s.total_time_s, 2),
                "metric_scores": {k: v.score for k, v in s.metric_scores.items()},
            }
            for s in summaries
        },
    }

    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    console.print(f"\n💾 Combined summary saved: {combined_file}")
    return combined_file
