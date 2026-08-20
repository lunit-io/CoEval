# Conquer Health leaderboard JSON contract

Produced by `scripts/build_leaderboard.py`, consumed by the hackathon dashboard.
A worked example with every field populated is in
[`leaderboard.example.json`](./leaderboard.example.json) — build against that.

```bash
python scripts/build_leaderboard.py \
    --runs  /path/to/submissions \      # one subdirectory per team
    --dataset conquer_test \            # or conquer_val for the live board
    --stage  official \                 # or provisional
    --judge  deepseek-ai/DeepSeek-V4-Pro \
    --out    leaderboard.json
```

Each team subdirectory holds that submission's CoEval `results_<dataset>.json`,
plus an optional `submission.json` whose contents are passed through verbatim
under `entries[].submission`.

## Minimal render

Three fields are enough for the page as scoped: `entries[].rank`,
`entries[].team`, `entries[].score`. Everything else is optional enrichment and
can be ignored without breaking anything.

Two are worth adding cheaply, because leaving them out changes what the page
*means*:

- **`stage`** — `provisional` is a self-reported-cadence live board;
  `official` is the frozen ranking. Rendering an official-looking table over
  provisional numbers invites a dispute at the freeze.
- **`entries[].tied_with_ranks`** — non-empty means the gap to those ranks is
  inside measurement noise. Showing rank 2 above rank 3 when they are tied
  presents a coin flip as a result. A shared-position marker or a "tied" chip
  is enough.

## Top level

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | `1`. Bumped on any breaking change. |
| `generated_at` | ISO-8601 | UTC, second precision. |
| `stage` | `"provisional"` \| `"official"` | See above. |
| `dataset.name` | string | `conquer_val` or `conquer_test`. |
| `dataset.n_items` | int | Items in the split. |
| `dataset.n_scored` | int | Items actually used — see common-item rule below. |
| `dataset.n_dropped` | int | `n_items - n_scored`. |
| `dataset.dropped_sample_ids` | int[] | Which ones, for audit. |
| `dataset.judge` | string | Judge model id, for provenance. |
| `scoring.*` | object | Method, bootstrap settings, noise basis, tie rule. Renderable as a methodology footnote. |
| `entries` | array | Sorted by `score` descending; `rank` is 1-based. |
| `runoff_candidates` | string[] | Teams warranting a multi-vote re-grade. |
| `runoff_excluded_count` | int | Contenders cut by the cost cap. If `> 0`, the runoff list is **not** the full set of contenders. |
| `notes` | string[] | Human-readable caveats for this run. Safe to render verbatim. |

## Entry

| Field | Type | Notes |
|---|---|---|
| `rank`, `team`, `score` | int, string, float | `score` is the clipped mean in `[0, 1]`. |
| `ci_low`, `ci_high` | float | **The interval to quote.** Item sampling *and* run-to-run noise. |
| `ci_item_sampling` | [float, float] | Bootstrap only. Narrower, and incomplete — do not display this as *the* CI. |
| `sd_item_sampling`, `sd_total` | float | The two components. |
| `tied_with_ranks` | int[] | Ranks statistically indistinguishable from this one. |
| `n_scored`, `n_items` | int | Identical for all entries by construction. |
| `n_inference_failed` | int | Their endpoint erred, timed out, or returned nothing. Scored 0, not dropped. |
| `n_scoring_failed` | int | Our judge failed. Excluded from everyone's score. |
| `axis_scores` | object | 5 keys: `accuracy`, `completeness`, `context_awareness`, `communication_quality`, `instruction_following`. The interesting breakdown for a health chatbot. |
| `theme_scores` | object | 7 HealthBench themes. |
| `mean_response_chars` | int | Worth showing: rubric scoring rewards length, so displaying it is a mild deterrent. |
| `mean_latency_ms` | int | Mean generation time per item. |
| `flags` | string[] | `high_failure_rate` (>5% of items), `endpoint_failures_scored_zero` (>2%). |
| `submission` | object | Verbatim `submission.json`: repo, commit, models used, RAG sources. |

## Guarantees

- **Every entry is scored over the identical item set.** A judge failure
  excludes that item for *all* teams, not just the affected one, so the
  comparison stays paired. `dataset.n_dropped` reports the cost.
- **The script refuses to rank incomparable runs.** Differing `sample_id` sets,
  or the same `sample_id` carrying different prompt text, is a hard error rather
  than a silently skewed table.
- **Ranks are stable for a given input**; the bootstrap is seeded (`--seed`).
- **No silent truncation.** A capped runoff list always sets
  `runoff_excluded_count` and appends a note.

## Why two intervals

`ci_item_sampling` is a paired bootstrap over items and answers "would a
different draw of 500 questions have changed this?". It cannot see the other
noise source: six repeat runs of an identical configuration on the same items
measured sd **0.0174** at n=200, from judge nondeterminism (the judge runs at
temperature above zero because it degenerates into repetition loops at greedy
decode) plus vLLM batch nondeterminism. That component is real, scales as
`1/sqrt(n)` (≈0.011 at n=500), and is folded into `ci_low`/`ci_high`.

The practical consequence for the page: **at n=500, teams within roughly 3
points are tied.** `tied_with_ranks` encodes exactly that, so the UI does not
need to recompute it.
