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

That is also why --salt is mandatory. This script is itself public, so a purely
deterministic selection would mean the repository *contains* the holdout: anyone
could re-run it with the documented sizes and recover the exact test ids in
seconds. Salting the per-theme ordering breaks that. Publishing the val ids stays
safe because recovering the salt from them is a 2^128 search.

Keep the salt with the test id list, on storage outside this repository. Losing
it costs reproducibility of this split; leaking it costs the holdout.

Usage:
    SALT=$(openssl rand -hex 32)
    python scripts/make_conquer_split.py --n-val 300 --n-test 500 --salt "$SALT" \
        --val-out src/coeval/data/conquer_val_ids.json \
        --test-out <held-out-dir>/conquer_test_ids.json
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coeval.datasets.healthbench import MAIN_URL, _HealthBenchDatasetBase  # noqa: E402

SCHEMA_VERSION = 2
MIN_SALT_CHARS = 16


def theme_of(row: dict) -> str:
    for tag in row.get("example_tags", []):
        if tag.startswith("theme:"):
            return tag
    return "theme:unknown"


def salted_key(salt: str, prompt_id: str) -> str:
    """Keyed digest of a prompt id: deterministic given the salt, unpredictable
    without it. Ordering by this is what keeps the split out of the public repo."""
    return hmac.new(salt.encode(), prompt_id.encode(), hashlib.sha256).hexdigest()


def salt_fingerprint(salt: str) -> str:
    """Published so two artifacts can be checked for a common salt without
    revealing it."""
    return hashlib.sha256(b"conquer-split-v2|" + salt.encode()).hexdigest()[:16]


def balanced_order(items: list[dict], salt: str) -> list[dict]:
    """One ordering per theme, such that any contiguous window of it is
    representative of the theme's rubric-size distribution.

    Items are bucketed by criteria count, ordered inside each bucket by the
    salted digest, then merged on relative position within their bucket. The
    merged sequence therefore cycles through the whole size range repeatedly
    instead of walking it once, so a prefix -- or the slice just after it -- is a
    fair sample rather than the small-rubric end of the theme.

    Two properties follow, and both matter operationally:

    - val takes the prefix and test the slice after it, so **val does not move
      when n_test changes**. It can be frozen and published while the test size
      is still undecided.
    - test is a growing window, so test(500) is a prefix of test(1000). The test
      size can be settled as late as the scoring run, against measured capacity,
      without regenerating anything.

    The ordering stays secret because the within-bucket order is salt-keyed;
    bucket membership alone is public, and is not enough to reconstruct it.
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in items:
        buckets[len(row.get("rubrics", []))].append(row)

    placed: list[tuple[float, str, dict]] = []
    for size, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: salted_key(salt, r["prompt_id"]))
        # Offset each bucket's positions by a salted amount rather than centring
        # them. A fixed +0.5 pins every small bucket to the middle of the merged
        # sequence, so rare rubric sizes systematically miss the prefix and pile
        # into the slice after it -- which showed up as val/test drift on mean
        # criteria count growing with the test size. A per-bucket offset
        # decorrelates bucket size from position.
        offset = int(salted_key(salt, f"bucket:{size}")[:8], 16) / 0x100000000
        n = len(bucket)
        for i, row in enumerate(bucket):
            placed.append(
                (((i + offset) / n) % 1.0, salted_key(salt, row["prompt_id"]), row)
            )
    placed.sort(key=lambda t: (t[0], t[1]))
    return [row for _pos, _key, row in placed]


def build_split(
    rows: list[dict], n_val: int, n_test: int, salt: str
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

        # One balanced, salt-secret ordering per theme. val is the prefix, test
        # the slice immediately after it. k_val depends only on n_val, so val is
        # invariant to the test size, and growing k_test only extends the test
        # window rather than reshuffling either split.
        order = balanced_order(group, salt)
        val_ids += [r["prompt_id"] for r in order[:k_val]]
        test_ids += [r["prompt_id"] for r in order[k_val : k_val + k_test]]

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
    ap.add_argument(
        "--salt",
        required=True,
        help=(
            "high-entropy secret keying the selection. Generate with "
            "`openssl rand -hex 32` and store it beside the test id list, "
            "never in the repository."
        ),
    )
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--test-out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.salt) < MIN_SALT_CHARS:
        raise SystemExit(
            f"--salt must be at least {MIN_SALT_CHARS} characters; a guessable salt "
            "is the same as no salt, because this script is public. "
            "Try: openssl rand -hex 32"
        )

    raw = _HealthBenchDatasetBase._read_or_download(MAIN_URL)
    rows = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    rows = [
        r for r in rows if r.get("prompt") and r.get("rubrics") and r.get("prompt_id")
    ]
    print(f"pool: {len(rows)} scorable HealthBench Main examples")

    val_ids, test_ids = build_split(rows, args.n_val, args.n_test, args.salt)

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
                    "selection": (
                        "theme-stratified; salt-keyed candidate pool, then dealt over "
                        "criteria-count order"
                    ),
                    # Fingerprint only. The salt itself never enters an artifact --
                    # the val file is published, and a file carrying the salt would
                    # hand over the holdout.
                    "salt_fingerprint": salt_fingerprint(args.salt),
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
