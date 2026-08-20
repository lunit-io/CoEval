#!/usr/bin/env python3
"""Generate the Conquer Health validation / test split from HealthBench Main.

Design constraints this script enforces:

1. **Disjoint.** No prompt_id appears in both splits.
2. **Theme-stratified.** Each ``theme:*`` stratum contributes to both splits in
   proportion to its share of the pool, so neither split over-represents a theme.
3. **Balanced on rubric structure.** Within a theme, candidates are drawn by
   systematic sampling over a criteria-count-sorted order, then dealt
   alternately to val and test. Random slicing would leave the two splits with
   different criteria-count distributions, and an example's score granularity is
   1/n_criteria -- a 2-criterion example is far noisier than a 30-criterion one,
   so an imbalance here makes val a biased proxy for test.

The val id list is published (participants iterate against it). The test id list
is written separately and must live OUTSIDE the repository -- every HealthBench
prompt and rubric is public, so the only thing protecting the test set is which
items were chosen.

Usage:
    python scripts/make_conquer_split.py --n-val 300 --n-test 500 \
        --val-out src/coeval/data/conquer_val_ids.json \
        --test-out /mnt/vast/lunit/coe_evaluation/conquer/conquer_test_ids.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coeval.datasets.healthbench import MAIN_URL, _HealthBenchDatasetBase  # noqa: E402

SCHEMA_VERSION = 1


def theme_of(row: dict) -> str:
    for tag in row.get("example_tags", []):
        if tag.startswith("theme:"):
            return tag
    return "theme:unknown"


def systematic(items: list, k: int) -> list:
    """Pick k evenly spaced items, preserving the ordering's distribution."""
    n = len(items)
    if k >= n:
        return list(items)
    return [items[round(i * n / k)] for i in range(k)]


def build_split(
    rows: list[dict], n_val: int, n_test: int
) -> tuple[list[str], list[str]]:
    by_theme: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_theme[theme_of(row)].append(row)

    total = sum(len(v) for v in by_theme.values())
    val_ids: list[str] = []
    test_ids: list[str] = []

    for theme in sorted(by_theme):
        group = by_theme[theme]
        share = len(group) / total
        # Proportional allocation; every theme present in the pool must appear
        # in both splits, hence the floor of 1.
        k_val = max(1, round(n_val * share))
        k_test = max(1, round(n_test * share))
        need = k_val + k_test
        if need > len(group):
            raise SystemExit(
                f"{theme}: need {need} examples but stratum holds only {len(group)}"
            )

        # Sort by rubric structure, then deal alternately -> matched distributions.
        ordered = sorted(
            group, key=lambda r: (len(r.get("rubrics", [])), r["prompt_id"])
        )
        pool = systematic(ordered, need)

        # Deal proportionally: walk the pool and assign to whichever split is
        # furthest behind its quota. Keeps both splits spread across the whole
        # criteria-count range instead of splitting it in half.
        v: list[dict] = []
        t: list[dict] = []
        for row in pool:
            want_val = (len(v) + 1) / k_val if k_val else 2.0
            want_test = (len(t) + 1) / k_test if k_test else 2.0
            if len(v) < k_val and (len(t) >= k_test or want_val <= want_test):
                v.append(row)
            else:
                t.append(row)
        val_ids += [r["prompt_id"] for r in v]
        test_ids += [r["prompt_id"] for r in t]

    return val_ids, test_ids


def describe(name: str, rows: list[dict]) -> dict:
    ncrit = [len(r.get("rubrics", [])) for r in rows]
    negrate = [
        sum(1 for x in r["rubrics"] if x["points"] < 0) / len(r["rubrics"])
        for r in rows
        if r.get("rubrics")
    ]
    posmass = [
        sum(x["points"] for x in r["rubrics"] if x["points"] > 0)
        for r in rows
        if r.get("rubrics")
    ]
    themes = Counter(theme_of(r) for r in rows)
    stats = {
        "n": len(rows),
        "judge_calls": sum(ncrit),
        "criteria_per_example": round(st.mean(ncrit), 3),
        "criteria_sd": round(st.pstdev(ncrit), 3),
        "neg_rate": round(st.mean(negrate), 4),
        "pos_point_mass": round(st.mean(posmass), 2),
        "themes": {k: round(v / len(rows), 4) for k, v in sorted(themes.items())},
    }
    print(f"\n[{name}] n={stats['n']}  judge_calls={stats['judge_calls']}")
    print(
        f"  criteria/example {stats['criteria_per_example']} (sd {stats['criteria_sd']})"
        f"   neg_rate {stats['neg_rate']}   pos_mass {stats['pos_point_mass']}"
    )
    for k, v in stats["themes"].items():
        print(f"    {v * 100:5.1f}%  {k}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--test-out", type=Path, required=True)
    args = ap.parse_args()

    raw = _HealthBenchDatasetBase._read_or_download(MAIN_URL)
    rows = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    rows = [
        r for r in rows if r.get("prompt") and r.get("rubrics") and r.get("prompt_id")
    ]
    print(f"pool: {len(rows)} scorable HealthBench Main examples")

    val_ids, test_ids = build_split(rows, args.n_val, args.n_test)

    overlap = set(val_ids) & set(test_ids)
    if overlap:
        raise SystemExit(f"BUG: {len(overlap)} ids in both splits")
    if len(set(val_ids)) != len(val_ids) or len(set(test_ids)) != len(test_ids):
        raise SystemExit("BUG: duplicate ids within a split")

    by_id = {r["prompt_id"]: r for r in rows}
    describe("pool", rows)
    val_stats = describe("val", [by_id[i] for i in val_ids])
    test_stats = describe("test", [by_id[i] for i in test_ids])

    for out, ids, stats, split in (
        (args.val_out, val_ids, val_stats, "val"),
        (args.test_out, test_ids, test_stats, "test"),
    ):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "source": MAIN_URL,
                    "selection": "theme-stratified, systematic over criteria-count order",
                    "stats": stats,
                    "prompt_ids": sorted(ids),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {out}  ({len(ids)} ids)")

    print("\n=== proxy check: val vs test ===")
    for key in ("criteria_per_example", "neg_rate", "pos_point_mass"):
        v, t = val_stats[key], test_stats[key]
        drift = abs(v - t) / max(abs(t), 1e-9) * 100
        flag = "OK " if drift < 5 else "DRIFT"
        print(f"  {flag} {key:22} val={v:<10} test={t:<10} drift={drift:.2f}%")
    worst = max(
        abs(val_stats["themes"].get(k, 0) - test_stats["themes"].get(k, 0))
        for k in set(val_stats["themes"]) | set(test_stats["themes"])
    )
    print(
        f"  {'OK ' if worst < 0.02 else 'DRIFT'} max theme-share gap  {worst * 100:.2f}pp"
    )
    print(f"\npool coverage: {len(val_ids) + len(test_ids)}/{len(rows)} examples used")


if __name__ == "__main__":
    main()
