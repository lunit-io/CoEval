---
name: coeval
description: "Operate the CoEval medical LLM evaluation repository: set up the toolchain, compose Hydra configs, run or monitor evaluations, configure OpenAI-compatible candidate and judge models, and audit or compare evaluation artifacts. Use when working in this repository or when asked about CoEval runs, HealthBench, passthrough clients, retries, Hydra overrides, or evaluation_outputs. Do not use for generic evaluation work outside CoEval."
---

# CoEval

Run CoEval evaluations reproducibly and report enough evidence for another person to verify the result. Prefer the repository's current configuration and code over remembered commands.

## Establish scope

1. Resolve the repository root with `git rev-parse --show-toplevel` and work from it.
2. Read `git status --short --branch`. Preserve unrelated tracked and untracked changes.
3. Classify the request before acting:
   - For explanation, review, diagnosis, comparison, or status, stay read-only.
   - For an explicit run request, allow only the requested evaluation outputs and caches.
   - For a code or config change, edit only the requested scope and run proportionate tests.
   - Commit, push, delete, stop processes, or change services only when explicitly requested.
4. Never expose credentials in commands, logs, artifacts, or replies. Do not write secrets into tracked `mise.toml`.

## Inspect the live contract

Read only the files needed for the requested operation:

- `README.md` and `mise.toml` for setup and task entry points.
- `src/coeval/conf/config.yaml` for root defaults.
- `src/coeval/conf/client/*.yaml` for the candidate client shape.
- `src/coeval/conf/datasets/<dataset>.yaml` for dataset, metric, aggregator, sampling, and judge composition.
- `src/coeval/conf/datasets/metrics/judge/*.yaml` for judge authentication and options.
- `src/coeval/core/schema.py` and the relevant aggregator when interpreting artifacts.

Treat the current YAML and implementation as the source of truth if prose examples disagree. Do not reuse an endpoint, model, credential mode, concurrency, sample count, or output path from an earlier run without verifying it.

## Choose the operation

### Set up

Use the repository tasks rather than inventing installation commands:

```bash
mise trust
mise run sync
```

Run `mise run test`, `mise run lint`, and `uv run ruff format --check src/ tests/` when validating code changes. Do not run the mutating formatter across the repository for an operational evaluation request.

### Compose an evaluation

Build overrides from the current Hydra object paths. The candidate fields currently live below `client.llm.config`:

```bash
mise run eval -- \
  datasets=medqa \
  num_samples=5 \
  client.llm.config.api_base=http://host:port/v1 \
  client.llm.config.model=model-name
```

Quote Hydra values containing spaces or shell-significant characters. Use `+key=value` only to add a missing key and `++key=value` when Hydra must add or override it; use a normal override for declared keys.

For HealthBench, name the subset and judge explicitly:

```bash
mise run eval -- \
  datasets=healthbench_consensus \
  '++datasets.healthbench_consensus.sample_ratio=0.1' \
  datasets/metrics/judge@healthbench_judge=gpt-4.1
```

Use a positive `num_samples=N` for a small capped smoke test. Use `sample_ratio` only when a proportional HealthBench subset is intended, require `0 < sample_ratio <= 1`, and report the actual loaded count and sampling seed. Do not silently combine them: `num_samples` takes priority when both are set.

Configure retry policies on both sides when requested. `max_attempts` includes the initial call, and retry delays use exponential backoff:

```text
runner.inference_max_attempts=<N>
runner.inference_retry_delay_s=<seconds>
metrics.<dataset>.healthbench_rubric.max_attempts=<N>
metrics.<dataset>.healthbench_rubric.retry_delay_s=<seconds>
```

### Preflight a live run

Before starting a paid, long, or remote evaluation:

1. Show the exact command with secret values omitted.
2. Resolve and record the dataset, requested sampling, candidate endpoint/model, judge, reasoning effort, both concurrency limits, both retry policies, and output directory. If the user did not name a candidate, surface the resolved default before calling it.
3. Probe the OpenAI-compatible candidate endpoint and model catalog without printing its key. Confirm the configured model is actually served.
4. For `gpt-4.1`, confirm `OPENAI_API_KEY` is set and verify `OPENAI_API_BASE` when overridden.
5. For a Codex-backed judge, verify the required CLI version and `codex login status`; keep process concurrency modest unless the user specifies otherwise.
6. Check for a duplicate active evaluation targeting the same scope. Do not stop it without authorization.
7. Estimate candidate calls, judge rubric calls, cost-sensitive scope, and expected duration when the data makes that possible.
8. If the user did not specify scope, start with a small smoke test. For a new endpoint, new judge, or costly run, smoke-test first unless the user explicitly asks to skip it; then continue to the explicitly requested full or ratio scope without requiring another confirmation.

Use Hydra's config display options when useful to validate composition before generation. A composed config is not proof that endpoints, authentication, or the evaluation itself work.

### Monitor

Identify a run by its process and Hydra output directory. Report current progress, elapsed time, retry warnings, terminal inference failures, terminal judge failures, and whether generation or scoring is active. Distinguish a still-running, interrupted, or stopped run from a completed run.

Do not call a transient warning a terminal failure when a later retry succeeds. Do not infer completion from the presence of a directory or partial result file. The repository has no resume or `COMPLETE` marker; label interrupted runs partial.

### Audit artifacts

Inspect the run's `.hydra/config.yaml`, `.hydra/overrides.yaml`, `results_<dataset>.json`, `summary_<dataset>.json`, and `summary_combined.json` when present.

Verify at minimum:

1. The resolved candidate, judge, dataset, sampling, retry, and concurrency settings match the request.
2. The results array length equals `num_samples`, with the expected unique sample IDs.
3. `num_evaluated = num_samples - num_inference_failed - num_scoring_failed`.
4. Each failure flag agrees with its error and metric payload; judge infrastructure failures must not be interpreted as negative rubric labels.
5. The benchmark score comes from `metric_scores`; report `pass_rate` separately because it is thresholded and can differ materially.
6. Metric numerators and denominators reconcile with the aggregate. For HealthBench, inspect rubric-grade counts and theme/category/cluster denominators as applicable.
7. Every expected flattened dataset has an individual result/summary pair. The combined summary includes exactly those datasets, but never use it alone as proof of completeness: it omits per-dataset failure detail, skipped datasets require logs/config inspection, and same-named metrics may be combined without sample weighting.
8. Timing fields are supported by the implementation that produced the run before using them for efficiency conclusions. If the artifact lacks its commit SHA, state that provenance gap. Do not treat a zero timing field as measured latency without checking its producer.

For comparisons, require matching dataset versions, sample IDs or sampling policy, system prompt, candidate generation settings, judge configuration, retry/failure policy, and aggregation. Label any unmatched comparison as non-equivalent.

## Report the outcome

For every completed run, including a completed smoke test, lead with a Markdown
table using one row per dataset and primary metric:

| Status | Dataset / scope | Samples | Evaluated | Primary score | Pass rate | Inference failures | Scoring failures | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete | `<dataset / subset>` | `<n>` | `<n>` | `<score>` | `<rate>` | `<n>` | `<n>` | `<time>` |

Use `N/A` for unavailable values rather than fabricating zero. Label completed
smoke tests as `Smoke complete`, and never label partial, interrupted, or
failure-contaminated runs as complete.

After the table, include:

- Dataset/subset, actual sample count, and sampling method.
- Candidate endpoint host and model, with credentials omitted.
- Judge model/provider and relevant reasoning effort.
- Metric name, score, numerator/denominator when available, and separate pass rate.
- Inference and scoring failure counts, including whether retries recovered.
- Total elapsed time and only trustworthy latency fields.
- Absolute artifact directory and exact files inspected.
- Original command with secrets redacted when it was preserved; otherwise label an equivalent command reconstructed from Hydra overrides. Include the repository commit when known, process exit state, and completion status.
- Any mismatch, incomplete state, or limitation that affects comparability.

Never present a smoke test, partial artifact, or failure-contaminated score as a completed benchmark.
