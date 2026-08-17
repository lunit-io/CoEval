# HealthBench Main port — design

**Date:** 2026-08-17
**Status:** approved, ready for implementation planning
**Source:** `chain-of-evidence` @ `origin/develop` (`77b2a31`)

## Goal

Bring the `healthbench_main` benchmark — the headline HealthBench eval (`oss_eval`,
5,000 examples) — into CoEval, together with the judge configuration needed to run
it. CoEval currently ships only the Consensus subset.

## Non-goals

Explicitly out of scope. Each was considered and rejected:

| Excluded | Reason |
|---|---|
| Upstream's `_HealthBenchBase` loader refactor (`SUBSET_NAME`/`REMOTE_URL`/`LOCAL_PATH`/`TOTAL_SAMPLES`) | Would rewrite the working Consensus loader and its tests, drop CoEval's XDG cache logic, and bake the Lunit-internal `/mnt/vast/...` path into a public Apache-2.0 repo. |
| `hard` and `professional` subsets | Not requested. |
| `metrics/_rubric_grading.py` split and `length_penalty_per_char` / `length_baseline_chars` | Those exist upstream only to serve HealthBench Professional. |
| `runner.max_workers` rename | Explicitly excluded by the requester. CoEval keeps `concurrent_limit: 30`. |
| Adding `healthbench_main` to `conf/datasets/all.yaml` | 5,000 examples x many rubric criteria each would make `datasets=all` smoke runs ruinously slow and expensive. Upstream omits it too. |
| Upstream's `sample_ratio: 0.1` default on Consensus | Changing an existing dataset's default breaks comparison against prior CoEval results. |
| Upstream's internal endpoint `shared-cluster-vm-026:9411` / model `debug-phase2` in `passthrough.yaml` | Internal hostnames must not land in a public repo. |

## Spike findings

A live probe of `codex exec` (v0.146.0) on this machine produced three facts that
constrain the design:

1. **`codex exec` exits 0 even on total failure.** With no valid auth it emitted ten
   `401 Unauthorized` errors, wrote nothing to the `-o` file, and still returned
   exit code 0. **Return code is not a reliable success signal.** An empty or
   unparseable output file is the authoritative failure indicator.
2. **stderr carries benign noise.** `warning: Codex could not find bubblewrap on PATH`
   appears on every run; Codex falls back to a bundled copy and works fine. stderr
   content must not be treated as a failure signal.
3. **`--output-schema` is available** and matches what the HealthBench grader needs
   exactly: `GRADER_TEMPLATE` asks for `{"explanation": str, "criteria_met": bool}`.

**Prerequisite (environment, not code):** `codex` must be authenticated once via
`codex login --api-key "$OPENAI_API_KEY"`. Setting `OPENAI_API_KEY` alone is not
enough, and `-c preferred_auth_method="apikey"` did not pick it up.

## Components

### 1. `HealthBenchMainDataset`

Added to `src/coeval/datasets/healthbench.py` on top of the existing
`_HealthBenchDatasetBase` (`_URL` / `_LABEL` + XDG download cache). The existing
`HealthBenchConsensusDataset` and the shared base are untouched.

```python
MAIN_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

@register_dataset("healthbench_main")
class HealthBenchMainDataset(_HealthBenchDatasetBase):
    _URL = MAIN_URL
    _LABEL = "Main"

    @property
    def name(self) -> str:
        return "HealthBenchMain"
```

The module docstring's "Available subsets" list gains Main. Stratified sampling
(`sample_ratio`) and `num_samples` come free from the base class.

First run downloads a ~60 MB blob to `~/.cache/coeval/`; subsequent runs read the
cache.

### 2. `clipped_avg_aggregator`

Added to `src/coeval/util/aggregation.py`, ported from upstream.

```python
def clipped_avg_aggregator(results, key_to_name=None) -> AggregationResult:
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
```

The `numerator = score * denominator` rescale is load-bearing: without it a
downstream `weighted_merge` recomputing `sum(num) / sum(den)` would undo the clip.
`breakdown` and per-sample scores stay raw as diagnostics.

**Why Main needs it:** Main's rubrics mix positive and negative (penalty) criteria,
so a per-example score — and therefore the mean — can go net-negative. Clipping the
reported mean to `[0, 1]` is the official HealthBench formula.

**Consensus also switches to it.** All 8,053 Consensus criteria carry a uniform +5
with no penalties, so per-example scores are always in `[0, 1]` and the clip is a
**no-op safeguard**, not load-bearing. It is adopted so both subsets share one
aggregation path. The config comment must say exactly this — describing it as
load-bearing on Consensus would be wrong.

### 3. `CodexExecJudge`

New file `src/coeval/clients/codex_exec_judge.py`. A `DeepEvalBaseLLM`
implementation where one grading call is one `codex exec` process.

```python
class CodexExecJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        sandbox: str = "read-only",
        timeout: float = 300.0,
        output_schema: dict | None = None,
        cwd: str | None = None,
        system_prompt: str = "",
    ) -> None: ...
```

**Command assembled per call:**

```
codex exec --model {model} --ephemeral --skip-git-repo-check --color never \
           -s {sandbox} -C {cwd} [--output-schema {schema_path}] -o {out_path} -
```

with the prompt written to stdin.

**Design decisions:**

- **Working directory.** Defaults to an instance-owned temp directory, so Codex
  cannot read the CoEval repo and contaminate a grading verdict with repo context.
- **System prompt.** `codex exec` has no system-prompt flag, so `system_prompt` is
  prepended to the user prompt as a labelled block. This is a semantic difference
  from the HTTP `PassthroughJudge` and must be stated in the docstring.
- **Failure detection.** Raise `RuntimeError` when the process exits non-zero **or**
  the output file is missing / empty / whitespace-only. Per spike finding 1, the
  empty-file check is the one that actually fires. Per finding 2, stderr is logged
  at debug level and never treated as failure.
- **Timeout.** `asyncio.wait_for` around process completion; kill the process on
  expiry, then raise.
- **No internal retry.** `grade_with_retry` in `healthbench_rubric.py` already
  catches `Exception` and retries `MAX_RETRIES` (3) times, then applies
  `on_failure="false"` to preserve official HealthBench scoring semantics. Adding a
  second retry layer inside the judge would multiply attempts silently.
- **Concurrency.** Bounded by the metric's existing
  `HealthBenchRubricMetric(concurrent_limit=...)` semaphore. No new mechanism.
  Because each call is a process rather than an HTTP request, README will recommend
  `concurrent_limit=4` when grading with Codex.
- **Temp file hygiene.** Output file is created per call via `mkstemp` inside the
  instance temp dir (unique per concurrent call) and removed in a `finally`. The
  schema file is written once per instance.
- **Hydra conversion.** The judge YAML sets `_convert_: all` so `output_schema`
  arrives as a plain `dict`, not a `DictConfig`. This keeps `omegaconf` out of the
  client module.

Exported from `src/coeval/clients/__init__.py`.

### 4. Judge configs

`src/coeval/conf/datasets/metrics/judge/` ends up holding exactly two entries.

| File | Change |
|---|---|
| `gpt-4.1.yaml` | Modified: `api_base` becomes `${oc.env:OPENAI_API_BASE,https://api.openai.com/v1}` so it can point at a compatible proxy. Everything else unchanged. |
| `gpt-5.6-sol.yaml` | New: `_target_: coeval.clients.CodexExecJudge`, `_convert_: all`, `model: gpt-5.6-sol`, and the `{explanation, criteria_met}` JSON Schema inlined under `output_schema`. |

Upstream's judge YAMLs nest a `config:` block targeting `coe_common.llm.config.LLMConfig`.
CoEval's `PassthroughJudge` takes flat kwargs, so the port is a mechanical flattening
— which CoEval's existing `gpt-4.1.yaml` already does correctly.

### 5. `passthrough.yaml` refresh

Generation parameters only; the endpoint stays public-safe.

| Key | Before | After | Reason |
|---|---|---|---|
| `max_tokens` | 4096 | 32768 | HealthBench answers are long-form open text; 4096 truncates them and silently depresses rubric scores. |
| `timeout` | 120.0 | 360.0 | Follows from the larger generation budget. |
| `additional_kwargs` | absent | `top_p: 1.0` | Matches upstream. |
| `api_base`, `model`, `system_prompt` | unchanged | unchanged | Upstream's values are internal infrastructure (`shared-cluster-vm-026`, `debug-phase2`) or depend on `coe_common.prompt.factory`, which CoEval does not have. |

### 6. Remove `gpt-oss-120b`

Delete `src/coeval/conf/client/gpt-oss-120b.yaml`. A repo-wide grep found no
references outside the file itself, so this is a clean removal.

## File inventory (15)

**New (4)**

- `src/coeval/clients/codex_exec_judge.py`
- `src/coeval/conf/datasets/metrics/judge/gpt-5.6-sol.yaml`
- `src/coeval/conf/datasets/healthbench_main.yaml`
- `tests/unit/test_codex_exec_judge.py`

**Modified (10)**

- `src/coeval/datasets/healthbench.py` — `MAIN_URL`, `HealthBenchMainDataset`, docstring
- `src/coeval/datasets/__init__.py` — import + `__all__`
- `src/coeval/util/aggregation.py` — `clipped_avg_aggregator`
- `src/coeval/clients/__init__.py` — export `CodexExecJudge`
- `src/coeval/conf/client/passthrough.yaml` — generation params
- `src/coeval/conf/datasets/metrics/judge/gpt-4.1.yaml` — env-overridable `api_base`
- `src/coeval/conf/datasets/healthbench_consensus.yaml` — switch to `clipped_avg_aggregator`
- `tests/unit/test_healthbench_dataset.py` — Main coverage
- `tests/unit/test_aggregation.py` — clipped aggregator coverage
- `README.md` — dataset row, judge swap examples, Codex prerequisite

**Deleted (1)**

- `src/coeval/conf/client/gpt-oss-120b.yaml`

**Deliberately untouched:** `conf/datasets/all.yaml`, `conf/runner/default.yaml`,
`metrics/healthbench_rubric.py`, the `_HealthBenchDatasetBase` loader.

## Configuration

`conf/datasets/healthbench_main.yaml` follows the shape of the existing
`healthbench_consensus.yaml` — dataset, metric, judge default, aggregator — with
`_target_` paths rewritten from `coe_evaluation.*` to `coeval.*` and the aggregator
inlined rather than pulled from the `metrics/aggregator` config group (matching both
upstream and `healthbench_consensus.yaml`).

Upstream's file references judge presets `gpt-oss-120b` and `qwen-3.6` in its usage
comments. Neither exists in CoEval, so the comments are rewritten to reference only
`gpt-4.1` and `gpt-5.6-sol`.

Usage:

```bash
mise run eval -- datasets=healthbench_main num_samples=5
mise run eval -- datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-5.6-sol
mise run eval -- datasets=healthbench_main '++datasets.healthbench_main.sample_ratio=0.1'
```

## Testing

All unit tests run offline — no network, no `codex` subprocess.

**`clipped_avg_aggregator`**
- mean below 0 clips to 0.0
- mean above 1 clips to 1.0
- mean already in range passes through unchanged
- `numerator == score * denominator` after clipping, so `weighted_merge` across
  summaries cannot undo the clip
- `breakdown` retains the raw (unclipped) mean

**`HealthBenchMainDataset`**
- `_URL` points at the `oss_eval` blob; `name == "HealthBenchMain"`
- negative-point (penalty) rubrics survive the load verbatim
- registered as `healthbench_main`; exported from `coeval.datasets`

Upstream's fixture patches `os.path.exists`, which CoEval's loader does not call.
Tests must instead patch `coeval.datasets.healthbench._CACHE_DIR` to a `tmp_path`,
as the existing Consensus tests already do — otherwise the real user cache is read
or written during a test run.

**`CodexExecJudge`** (patching `asyncio.create_subprocess_exec`)
- valid JSON in the output file is returned verbatim
- **exit code 0 with an empty output file raises** — the exact failure mode the
  spike observed
- missing output file raises
- timeout kills the process and raises
- a bubblewrap warning on stderr with valid output still succeeds
- assembled argv contains `--ephemeral`, `--skip-git-repo-check`, `-s`, `-o`, and
  `--output-schema` only when `output_schema` is set
- `system_prompt` appears in the stdin payload

## Verification

1. `mise run format && mise run lint && mise run test`
2. `uv run coeval datasets=healthbench_main num_samples=2 --cfg job` — confirms
   `datasets`, `metrics`, and `score_aggregators` all resolve under the
   `healthbench_main` key. Compose only, no execution.
3. Same with `datasets/metrics/judge@healthbench_judge=gpt-5.6-sol`.
4. `uv run coeval datasets=healthbench_consensus --cfg job` — confirms the
   aggregator switch did not break the existing config.
5. Smoke run, requires `codex login` first and incurs real cost plus a 60 MB
   download: `mise run eval -- datasets=healthbench_main num_samples=2`. To be
   confirmed with the requester before running.

## Risks

| Risk | Mitigation |
|---|---|
| `codex` unauthenticated in CI or on another machine | Judge raises a clear error; `grade_with_retry` degrades to `criteria_met=False` rather than crashing the run. README documents the `codex login` prerequisite. |
| Process-per-grading is slow at `concurrent_limit=10` | README recommends `concurrent_limit=4` for the Codex judge. Not enforced in code. |
| Full Main run cost (5,000 examples x many criteria) | Config comments and README lead with `num_samples` and `sample_ratio` smoke-test recipes. |
| `codex exec` CLI flags change across versions | Pinned behaviour verified against v0.146.0; recorded here and in the module docstring. |
