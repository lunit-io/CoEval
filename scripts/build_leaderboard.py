#!/usr/bin/env python3
"""Build the Conquer Health leaderboard from per-submission CoEval artifacts.

Consumes one directory per submission, each holding a CoEval run's
``results_<dataset>.json``, and emits the leaderboard JSON contract consumed by
the hackathon dashboard.

Three things this does that a plain sort of ``summary.metric_scores`` does not:

1. **Common-item intersection, at test time only.** A judge infrastructure
   failure excludes that example for that team alone, so two teams can end up
   averaged over different item sets, and comparing those means is not sound.
   For the official run every team is rescored over the items graded for *all*
   of them, and the discarded count is reported rather than hidden.

   The live board deliberately does not do this. Intersecting there would mean
   one team's judge failure silently moves every other team's displayed score,
   so a team's number could change without them resubmitting -- which reads as a
   broken leaderboard. The live board is a development signal, so each team is
   scored on its own graded items and told how many of its own were ungradeable.

2. **Paired bootstrap.** Item difficulty is a nuisance term shared by every
   team. Resampling the item indices once per iteration and applying that same
   resample to all teams keeps comparisons paired, so the interval on a
   *difference* is much tighter than two independent intervals would suggest.

   Ranking is by raw score in both regimes: entries are sorted on the score and
   ``rank`` is strictly positional. ``tied_with_ranks`` is reported alongside as
   information and never collapses two ranks into one. On the live board it is
   best rendered as a footnote; for the official result it is the basis of the
   pre-declared tiebreak.

3. **Honest uncertainty.** The bootstrap captures item-sampling variance only.
   Repeat runs of an identical configuration measured sd 0.0174 at n=200 --
   judge nondeterminism (the judge runs at temperature 1.0 because it
   degenerates at greedy decode) plus vLLM batch nondeterminism. That component
   is invisible to a bootstrap over items, so reporting the bootstrap interval
   alone would overstate confidence. Both are reported, and the tie test uses
   the combination.

Usage:
    python scripts/build_leaderboard.py \
        --runs /path/to/submissions --dataset conquer_test \
        --stage official --out leaderboard.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
PRIMARY_METRIC = "HealthBench Rubric"

# Run-to-run noise of an identical configuration, measured over 6 repeat runs of
# HealthBench Main at n=200 (scores 0.4429 0.4169 0.4578 0.4596 0.4568 0.4628).
# Scaled to other set sizes as sd/sqrt(n) -- it averages down with items like any
# per-item noise, it simply is not caused by *which* items were drawn.
RUN_NOISE_SD_AT = (0.0174, 200)


def run_noise_sd(n_items: int) -> float:
    sd, ref_n = RUN_NOISE_SD_AT
    return sd * math.sqrt(ref_n / max(n_items, 1))


@dataclass
class Submission:
    team: str
    path: Path
    item_scores: dict[int, float] = field(default_factory=dict)
    axis: dict[int, dict[str, float]] = field(default_factory=dict)
    theme: dict[int, dict[str, float]] = field(default_factory=dict)
    response_chars: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    n_items: int = 0
    n_inference_failed: int = 0
    n_scoring_failed: int = 0
    input_fingerprint: dict[int, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class UnusableSubmission(Exception):
    """This submission cannot be scored. Whether that halts the build or merely
    removes one row is the caller's decision, not the loader's."""


def load_submission(team_dir: Path, dataset: str) -> Submission:
    results_path = team_dir / f"results_{dataset}.json"
    if not results_path.exists():
        raise UnusableSubmission(f"missing {results_path.name}")
    try:
        rows = json.loads(results_path.read_text())
    except json.JSONDecodeError as exc:
        # A run killed mid-write leaves a truncated file. On a rolling board that
        # must not take every other team's score down with it.
        raise UnusableSubmission(
            f"{results_path.name} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise UnusableSubmission(f"{results_path.name} holds no results")

    sub = Submission(team=team_dir.name, path=team_dir)
    meta_path = team_dir / "submission.json"
    if meta_path.exists():
        sub.meta = json.loads(meta_path.read_text())

    for row in rows:
        sid = int(row["sample_id"])
        sub.n_items += 1
        sub.input_fingerprint[sid] = hashlib.sha256(
            (row.get("input") or "").encode()
        ).hexdigest()[:16]
        if row.get("inference_failed"):
            sub.n_inference_failed += 1
        if row.get("scoring_failed"):
            sub.n_scoring_failed += 1
        sub.response_chars.append(len(row.get("actual_output") or ""))
        if row.get("generation_time_ms") is not None:
            sub.latencies_ms.append(float(row["generation_time_ms"]))

        axis: dict[str, float] = {}
        theme: dict[str, float] = {}
        for m in row.get("metrics", []):
            name, score = m.get("name"), m.get("score")
            if score is None:
                continue
            if name == PRIMARY_METRIC:
                sub.item_scores[sid] = float(score)
            elif name.startswith("axis:"):
                axis[name[len("axis:") :]] = float(score)
            elif name.startswith("theme:"):
                theme[name[len("theme:") :]] = float(score)
        sub.axis[sid] = axis
        sub.theme[sid] = theme

    if not sub.item_scores:
        raise UnusableSubmission(
            f"no item was gradeable ({sub.n_inference_failed} inference failures, "
            f"{sub.n_scoring_failed} judge failures across {sub.n_items} items)"
        )
    return sub


def check_same_items(subs: list[Submission]) -> None:
    """Every team must have been asked the same questions, in the same order."""
    ref = subs[0]
    for sub in subs[1:]:
        if sub.input_fingerprint.keys() != ref.input_fingerprint.keys():
            raise SystemExit(
                f"{sub.team} has sample_ids {sorted(set(sub.input_fingerprint) ^ set(ref.input_fingerprint))[:5]} "
                f"that {ref.team} does not (or vice versa) — these runs are not comparable."
            )
        mismatched = [
            sid
            for sid, fp in sub.input_fingerprint.items()
            if fp != ref.input_fingerprint[sid]
        ]
        if mismatched:
            raise SystemExit(
                f"{sub.team} sample_id {mismatched[:5]} carries different prompt text than "
                f"{ref.team}. One of them ran a different item list; refusing to rank."
            )


def mean_axis(sub: Submission, common: list[int]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for sid in common:
        for k, v in sub.axis.get(sid, {}).items():
            buckets.setdefault(k, []).append(v)
    return {k: round(st.mean(v), 4) for k, v in sorted(buckets.items())}


def mean_theme(sub: Submission, common: list[int]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for sid in common:
        for k, v in sub.theme.get(sid, {}).items():
            buckets.setdefault(k, []).append(v)
    return {k: round(st.mean(v), 4) for k, v in sorted(buckets.items())}


def clipped_mean(values: list[float]) -> float:
    """Official HealthBench aggregation: penalty criteria can push a per-example
    score negative, so the mean is clipped into [0, 1]."""
    return min(1.0, max(0.0, st.mean(values))) if values else 0.0


def bootstrap(
    subs: list[Submission],
    items_for: dict[str, list[int]],
    iterations: int,
    seed: int,
    *,
    paired: bool,
) -> dict[str, list[float]]:
    """Resample items to get a sampling distribution of each team's score.

    ``paired`` draws one resample per iteration and applies it to every team, so
    item difficulty -- a nuisance term they share -- cancels in any comparison
    between them. That needs a common item set, so it is only available when the
    caller intersected. Unpaired, each team is resampled over its own items and
    comparisons carry both teams' variance.
    """
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {s.team: [] for s in subs}
    for _ in range(iterations):
        shared = None
        if paired:
            pool = items_for[subs[0].team]
            shared = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
        for sub in subs:
            pool = items_for[sub.team]
            idx = shared or [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
            draws[sub.team].append(clipped_mean([sub.item_scores[i] for i in idx]))
    return draws


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="dir of per-team subdirs")
    ap.add_argument("--dataset", default="conquer_test")
    ap.add_argument("--stage", choices=["provisional", "official"], default="official")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge", default="", help="judge model id, for provenance")
    ap.add_argument(
        "--runoff-top",
        type=int,
        default=3,
        help="how many leading teams to flag for a multi-vote runoff",
    )
    ap.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "a prior leaderboard JSON. Emits per-entry delta plus a significance "
            "verdict, so the dashboard decides whether a resubmission counts as "
            "an improvement from a boolean rather than picking a threshold itself."
        ),
    )
    ap.add_argument(
        "--on-unusable",
        choices=["auto", "fail", "skip"],
        default="auto",
        help=(
            "what to do with a submission that cannot be scored at all. 'auto' "
            "fails for --stage official, where a human should look before money "
            "is awarded, and skips for provisional, where one broken run at 3am "
            "must not freeze the board for everyone else."
        ),
    )
    ap.add_argument(
        "--pair-items",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "score every team over the intersection of gradeable items. 'auto' "
            "means on for --stage official and off for provisional."
        ),
    )
    ap.add_argument(
        "--runoff-max",
        type=int,
        default=4,
        help=(
            "hard cap on runoff candidates. A 3-vote re-grade costs 3x the judge "
            "calls, which is ~17 min for 3 teams at n=500 but ~114 min for 20, so "
            "the list has to be bounded. Truncation is reported, never silent."
        ),
    )
    args = ap.parse_args()

    team_dirs = sorted(p for p in args.runs.iterdir() if p.is_dir())
    if not team_dirs:
        raise SystemExit(f"no submission directories under {args.runs}")
    skip_unusable = (
        args.stage != "official"
        if args.on_unusable == "auto"
        else args.on_unusable == "skip"
    )
    subs: list[Submission] = []
    unusable: list[dict[str, str]] = []
    for d in team_dirs:
        try:
            subs.append(load_submission(d, args.dataset))
        except UnusableSubmission as exc:
            if not skip_unusable:
                raise SystemExit(f"{d.name}: {exc}") from exc
            unusable.append({"team": d.name, "reason": str(exc)})
            print(f"  SKIPPED {d.name}: {exc}")
    if not subs:
        raise SystemExit(
            "no scorable submission found"
            + (f" ({len(unusable)} unusable)" if unusable else "")
        )
    check_same_items(subs)

    previous: dict[str, float] = {}
    if args.previous and args.previous.exists():
        prior = json.loads(args.previous.read_text())
        previous = {e["team"]: e["score"] for e in prior.get("entries", [])}

    all_ids = set(subs[0].input_fingerprint)
    pair_items = (
        args.stage == "official"
        if args.pair_items == "auto"
        else args.pair_items == "on"
    )

    if pair_items:
        common = sorted(set.intersection(*(set(s.item_scores) for s in subs)))
        if not common:
            raise SystemExit("no item was successfully graded for every team")
        items_for = dict.fromkeys((s.team for s in subs), common)
        dropped = sorted(all_ids - set(common))
    else:
        items_for = {s.team: sorted(s.item_scores) for s in subs}
        if not all(items_for.values()):
            empty = [t for t, v in items_for.items() if not v]
            raise SystemExit(f"no item was gradeable at all for: {', '.join(empty)}")
        common = sorted(set.union(*(set(v) for v in items_for.values())))
        dropped = []

    draws = bootstrap(subs, items_for, args.iterations, args.seed, paired=pair_items)
    noise_sd = run_noise_sd(min(len(v) for v in items_for.values()))
    z = 1.959964 if abs(args.confidence - 0.95) < 1e-9 else 1.959964
    lo_q, hi_q = (1 - args.confidence) / 2, 1 - (1 - args.confidence) / 2

    scored = []
    for sub in subs:
        vals = [sub.item_scores[i] for i in items_for[sub.team]]
        score = clipped_mean(vals)
        boot = draws[sub.team]
        boot_sd = st.pstdev(boot) if len(boot) > 1 else 0.0
        total_sd = math.hypot(boot_sd, noise_sd)
        scored.append(
            {
                "sub": sub,
                "score": score,
                "boot_sd": boot_sd,
                "total_sd": total_sd,
                "ci_item_low": pct(boot, lo_q),
                "ci_item_high": pct(boot, hi_q),
                "ci_low": max(0.0, score - z * total_sd),
                "ci_high": min(1.0, score + z * total_sd),
            }
        )
    scored.sort(key=lambda e: e["score"], reverse=True)

    # Tie test on the paired difference. Two teams tie when the interval on
    # (a - b) straddles zero once run-to-run noise from two separate runs is
    # folded in alongside the item-sampling term.
    def tied(a: dict, b: dict) -> bool:
        diffs = [
            x - y
            for x, y in zip(draws[a["sub"].team], draws[b["sub"].team], strict=True)
        ]
        diff_sd = st.pstdev(diffs) if len(diffs) > 1 else 0.0
        sd = math.sqrt(diff_sd**2 + 2 * noise_sd**2)
        delta = a["score"] - b["score"]
        return abs(delta) <= z * sd

    entries = []
    for i, e in enumerate(scored):
        sub = e["sub"]
        ties = [j + 1 for j, o in enumerate(scored) if j != i and tied(e, o)]
        flags = []
        fail_rate = (sub.n_inference_failed + sub.n_scoring_failed) / max(
            sub.n_items, 1
        )
        if fail_rate > 0.05:
            flags.append("high_failure_rate")
        if sub.n_inference_failed > 0.02 * sub.n_items:
            flags.append("endpoint_failures_scored_zero")
        # Whether a resubmission actually moved. Both runs carry run-to-run noise,
        # so the difference has sd = sqrt(2) * noise_sd. Reporting a raw delta
        # without this verdict invites "you improved by 1 point" when the honest
        # statement is "no measurable change".
        delta_threshold = z * math.sqrt(2) * noise_sd
        delta_block: dict[str, object] = {}
        if sub.team in previous:
            delta = e["score"] - previous[sub.team]
            delta_block = {
                "previous_score": round(previous[sub.team], 4),
                "delta": round(delta, 4),
                "delta_threshold": round(delta_threshold, 4),
                "delta_significant": abs(delta) > delta_threshold,
                "delta_verdict": (
                    "improved"
                    if delta > delta_threshold
                    else "regressed"
                    if delta < -delta_threshold
                    else "no significant change"
                ),
            }

        entries.append(
            {
                "rank": i + 1,
                "team": sub.team,
                "score": round(e["score"], 4),
                "ci_low": round(e["ci_low"], 4),
                "ci_high": round(e["ci_high"], 4),
                "ci_item_sampling": [
                    round(e["ci_item_low"], 4),
                    round(e["ci_item_high"], 4),
                ],
                "sd_item_sampling": round(e["boot_sd"], 4),
                "sd_total": round(e["total_sd"], 4),
                "tied_with_ranks": ties,
                "n_scored": len(items_for[sub.team]),
                "n_ungraded": sub.n_items - len(items_for[sub.team]),
                "n_items": sub.n_items,
                "n_inference_failed": sub.n_inference_failed,
                "n_scoring_failed": sub.n_scoring_failed,
                "axis_scores": mean_axis(sub, items_for[sub.team]),
                "theme_scores": mean_theme(sub, items_for[sub.team]),
                "mean_response_chars": round(st.mean(sub.response_chars))
                if sub.response_chars
                else None,
                "mean_latency_ms": round(st.mean(sub.latencies_ms))
                if sub.latencies_ms
                else None,
                "flags": flags,
                **delta_block,
                "submission": sub.meta,
            }
        )

    # Everyone tied with the leader is a genuine contender, plus the top N. But a
    # multi-vote runoff costs 3x the judge calls, so the list is capped by rank
    # and the cut is stated -- a silently truncated list reads as "these are all
    # the contenders" when it is not.
    leader = entries[0]
    contenders = [leader["team"]] + [
        entries[r - 1]["team"] for r in leader["tied_with_ranks"]
    ]
    for e in entries[: args.runoff_top]:
        if e["team"] not in contenders:
            contenders.append(e["team"])
    ranked = [e["team"] for e in entries]
    contenders.sort(key=ranked.index)
    runoff = contenders[: args.runoff_max]
    runoff_truncated = len(contenders) - len(runoff)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": args.stage,
        "dataset": {
            "name": args.dataset,
            "n_items": subs[0].n_items,
            "n_scored": len(common),
            "n_dropped": len(dropped),
            "dropped_sample_ids": dropped,
            "judge": args.judge,
        },
        "scoring": {
            "primary_metric": PRIMARY_METRIC,
            "aggregation": "clipped mean over the common item set",
            "bootstrap_iterations": args.iterations,
            "confidence": args.confidence,
            "item_pairing": "intersection" if pair_items else "per-team",
            "ranking": "raw score, descending; rank is positional",
            "tie_reporting": (
                "tied_with_ranks is informational and never merges two ranks"
            ),
            "run_to_run_sd": round(noise_sd, 4),
            "delta_threshold": round(z * math.sqrt(2) * noise_sd, 4),
            "run_to_run_basis": (
                f"measured sd {RUN_NOISE_SD_AT[0]} over 6 identical runs at "
                f"n={RUN_NOISE_SD_AT[1]}, scaled by 1/sqrt(n)"
            ),
            "ci_note": (
                "ci_item_sampling is the paired bootstrap over items. ci_low/ci_high "
                "additionally fold in run-to-run noise, which a bootstrap over items "
                "cannot see, and are the interval to quote."
            ),
            "tie_rule": (
                "two teams tie when the interval on their paired score difference "
                "contains zero after run-to-run noise is included"
            ),
        },
        "entries": entries,
        "unusable_submissions": unusable,
        "runoff_candidates": runoff,
        "runoff_excluded_count": runoff_truncated,
        "notes": (
            [
                f"{len(dropped)} of {subs[0].n_items} items were not gradeable for "
                "every team and are excluded from all scores so the comparison stays "
                "paired."
            ]
            if pair_items
            else [
                "Each team is scored on its own gradeable items (see n_ungraded per "
                "entry). Scores are not strictly paired across teams, which is the "
                "right trade for a development signal: intersecting would let one "
                "team's judge failure move everyone else's displayed score."
            ]
        )
        + (
            [
                f"{len(unusable)} submission(s) could not be scored and are absent "
                "from the table rather than shown as zero: "
                + ", ".join(u["team"] for u in unusable)
            ]
            if unusable
            else []
        )
        + (
            [
                f"{runoff_truncated} further team(s) are statistically tied with the "
                f"leader but fall outside the runoff cap of {args.runoff_max}; the "
                "runoff list is a cost-bounded subset, not the full set of contenders."
            ]
            if runoff_truncated
            else []
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"stage={args.stage}  scored {len(common)}/{subs[0].n_items} items")
    print(
        f"{'rank':>4} {'team':<22} {'score':>7} {'95% CI':>17} {'tied':>10} {'flags'}"
    )
    for e in entries:
        ci = f"[{e['ci_low']:.3f},{e['ci_high']:.3f}]"
        print(
            f"{e['rank']:>4} {e['team']:<22} {e['score']:>7.4f} {ci:>17} "
            f"{str(e['tied_with_ranks']):>10} {','.join(e['flags'])}"
        )
    print(f"\nrunoff candidates: {', '.join(payload['runoff_candidates'])}")
    if runoff_truncated:
        print(
            f"WARNING: {runoff_truncated} more team(s) tie with the leader but exceed "
            f"--runoff-max={args.runoff_max}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
