#!/usr/bin/env python3
"""
CLI for CoEval Evaluation Framework using Hydra.

Usage:
    mise run eval                                    # Run default dataset (pubmedqa)
    mise run eval -- datasets=all                    # Run all datasets
    mise run eval -- datasets=medmcqa num_samples=100
"""

import asyncio
import logging
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import hydra  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from coeval.core import EvalRunner, EvalSummary  # noqa: E402
from coeval.util import console  # noqa: E402
from coeval.util.summary import save_combined_summary  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).parent / "conf")


def _flatten_datasets(cfg: DictConfig) -> list[tuple[str, DictConfig, str | None]]:
    """Flatten nested dataset groups into (name, cfg, group_name) triples."""
    entries: list[tuple[str, DictConfig, str | None]] = []
    for name, ds_cfg in cfg.datasets.items():
        if "_target_" in ds_cfg:
            entries.append((name, ds_cfg, None))
        else:
            for child_name, child_cfg in ds_cfg.items():
                entries.append((child_name, child_cfg, name))
    return entries


async def run_evaluation(cfg: DictConfig) -> list[EvalSummary]:
    """Run evaluation for all datasets (runner handles saving)."""
    runner: EvalRunner = instantiate(cfg.runner)
    flat = _flatten_datasets(cfg)

    summaries: list[EvalSummary] = []
    for idx, (name, ds_cfg, _group) in enumerate(flat, 1):
        dataset = instantiate(ds_cfg)
        metric_dict = instantiate(cfg.metrics[name])
        metrics = list(metric_dict.values())
        aggregator = instantiate(
            cfg.score_aggregators[name],
            key_to_name={k: m.__name__ for k, m in metric_dict.items()},
        )

        if len(dataset) == 0:
            logger.warning(
                f"Dataset '{name}' has 0 samples — skipping. "
                "Run the generation script first to populate the dataset."
            )
            continue

        console.dataset_header(name, idx, len(flat), len(dataset))
        summary = await runner.run(dataset, metrics, aggregator, dataset_name=name)

        console.metric_scores(
            summary.dataset,
            {k: v.score for k, v in summary.metric_scores.items()},
            breakdown=summary.breakdown or None,
        )
        summaries.append(summary)

    return summaries


def _build_display(cfg: DictConfig, summaries: list[EvalSummary]) -> list[EvalSummary]:
    """Merge grouped subsets into single display entries."""
    flat = _flatten_datasets(cfg)
    groups: dict[str, list[str]] = {}
    for name, _, group_name in flat:
        if group_name is not None:
            groups.setdefault(group_name, []).append(name)

    if not groups:
        return summaries

    by_name = {s.dataset: s for s in summaries}
    grouped_names = {n for members in groups.values() for n in members}
    display = [s for s in summaries if s.dataset not in grouped_names]

    for grp, members in groups.items():
        member_summaries = [by_name[m] for m in members if m in by_name]
        if member_summaries:
            agg = instantiate(cfg.score_aggregators[grp])
            display.append(agg(member_summaries, grp))

    return display


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    console.banner()
    console.print()
    console.set_config(cfg)

    try:
        summaries = asyncio.run(run_evaluation(cfg))
        display = _build_display(cfg, summaries)

        if display:
            console.leaderboard(
                [
                    {
                        "dataset": s.dataset,
                        "metric_scores": {
                            k: v.score for k, v in s.metric_scores.items()
                        },
                        "samples": s.num_samples,
                        "time": s.total_time_s,
                    }
                    for s in display
                ]
            )
            if cfg.runner.output_dir:
                save_combined_summary(summaries)

        if not summaries or sum(s.num_evaluated for s in summaries) == 0:
            console.error(
                "Evaluation failed: no samples completed inference and scoring"
            )
            sys.exit(1)

    except Exception as e:
        console.error(str(e))
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
