#!/usr/bin/env python3
"""Fabricate submission directories for testing the leaderboard end of the pipeline.

Exercises the dashboard's ingestion, sorting, tie display, delta rendering and
failure surfacing without a GPU, a judge, or a real team. Runs in seconds, so it
can be re-run freely while building the UI.

Team specs are `name:kind` where kind is one of:

    strong        clearly ahead of the field
    mid           mid-table
    weak          clearly behind
    tie           deliberately within noise of `mid` -- checks the tie path
    flaky         2% of items fail inference (scored 0, and flagged)
    broken        every item fails the judge   -> unusable, must be skipped, not zeroed
    truncated     results file cut mid-write   -> unusable, must be skipped, not crash
    empty         no results file at all       -> unusable
    verbose       mid-table but 3x the response length -- checks the length column

Example:
    python scripts/make_fixture_submissions.py --out /tmp/subs --n-items 300 \\
        --teams alpha:strong beta:mid gamma:tie delta:flaky eps:broken zeta:truncated
    python scripts/build_leaderboard.py --runs /tmp/subs --dataset conquer_val \\
        --stage provisional --out /tmp/lb.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

AXES = [
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
]
THEMES = [
    "global_health",
    "hedging",
    "communication",
    "context_seeking",
    "emergency_referrals",
    "health_data_tasks",
    "complex_responses",
]

# base score, per-item sd, inference-failure rate, judge-failure rate, length factor
KINDS: dict[str, tuple[float, float, float, float, float]] = {
    "strong": (0.62, 0.26, 0.0, 0.004, 1.0),
    "mid": (0.50, 0.28, 0.0, 0.004, 1.0),
    "tie": (0.502, 0.28, 0.0, 0.004, 1.0),
    "weak": (0.38, 0.28, 0.0, 0.004, 1.0),
    "flaky": (0.50, 0.28, 0.06, 0.004, 1.0),
    "verbose": (0.50, 0.28, 0.0, 0.004, 3.0),
    "broken": (0.50, 0.28, 0.0, 1.0, 1.0),
    "truncated": (0.50, 0.28, 0.0, 0.004, 1.0),
    "empty": (0.50, 0.28, 0.0, 0.004, 1.0),
}


def build_rows(kind: str, n_items: int, rng: random.Random) -> list[dict]:
    base, sd, inf_rate, judge_rate, length_factor = KINDS[kind]
    rows = []
    for sid in range(n_items):
        inference_failed = rng.random() < inf_rate
        judge_failed = rng.random() < judge_rate
        score = None
        if not inference_failed and not judge_failed:
            score = max(0.0, min(1.0, rng.gauss(base, sd)))

        metrics: list[dict] = []
        if score is not None:
            metrics.append(
                {"name": "HealthBench Rubric", "score": score, "passed": score >= 0.5}
            )
            for axis in AXES:
                metrics.append(
                    {
                        "name": f"axis:{axis}",
                        "score": max(0.0, min(1.0, rng.gauss(base, 0.12))),
                        "passed": True,
                    }
                )
            metrics.append(
                {
                    "name": f"theme:{THEMES[sid % len(THEMES)]}",
                    "score": score,
                    "passed": score >= 0.5,
                }
            )
        elif inference_failed:
            # Endpoint produced nothing: scored 0 by the runner, kept in the
            # denominator. Represented here as a present-but-failed row.
            metrics.append(
                {
                    "name": "HealthBench Rubric",
                    "score": 0.0,
                    "passed": False,
                    "reason": "Inference failed: scored 0",
                }
            )

        rows.append(
            {
                "sample_id": sid,
                "input": f"conquer-item-{sid}",
                "actual_output": "x" * int(rng.gauss(3100, 400) * length_factor),
                "passed": bool(score and score >= 0.5),
                "inference_failed": inference_failed,
                "inference_error": "endpoint returned no content"
                if inference_failed
                else None,
                "scoring_failed": judge_failed,
                "generation_time_ms": max(
                    500.0, rng.gauss(19000, 4000) * length_factor
                ),
                "scoring_time_ms": rng.gauss(400, 60),
                "metrics": metrics,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="conquer_val")
    ap.add_argument("--n-items", type=int, default=301)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--teams",
        nargs="+",
        required=True,
        help="name:kind pairs, e.g. alpha:strong beta:mid gamma:broken",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for spec in args.teams:
        name, _, kind = spec.partition(":")
        kind = kind or "mid"
        if kind not in KINDS:
            raise SystemExit(f"unknown kind {kind!r}; pick from {sorted(KINDS)}")
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"results_{args.dataset}.json"

        if kind == "empty":
            target.unlink(missing_ok=True)
        elif kind == "truncated":
            body = json.dumps(build_rows(kind, args.n_items, rng))
            target.write_text(body[: len(body) // 3])  # cut mid-write
        else:
            target.write_text(json.dumps(build_rows(kind, args.n_items, rng)))

        (d / "submission.json").write_text(
            json.dumps(
                {
                    "repo": f"github.com/conquer-health/{name}",
                    "commit": f"{rng.getrandbits(40):010x}",
                    "submitted_at": "2026-08-22T10:14:00Z",
                    "models_used": "L1 only",
                    "rag_sources": ["clinical_guidelines"],
                },
                indent=2,
            )
        )
        made.append(f"{name}:{kind}")

    print(f"wrote {len(made)} submission dirs to {args.out}")
    print("  " + "  ".join(made))


if __name__ == "__main__":
    main()
