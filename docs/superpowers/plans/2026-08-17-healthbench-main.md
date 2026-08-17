# HealthBench Main Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `healthbench_main` benchmark (HealthBench `oss_eval`, 5,000 examples) to CoEval, with a `codex exec`-backed judge and the clipped aggregator its penalty rubrics require.

**Architecture:** Three code additions on top of existing CoEval structure — a dataset subclass reusing the current `_HealthBenchDatasetBase`, a `clipped_avg_aggregator` wrapping `avg_aggregator`, and a new `CodexExecJudge` (`DeepEvalBaseLLM`) that grades by spawning one headless `codex exec` process per call. Everything else is Hydra YAML wiring.

**Tech Stack:** Python 3.12, Hydra, DeepEval, pytest (`asyncio_mode = "auto"`), ruff, uv/mise, codex-cli >= 0.146.0

**Spec:** `docs/superpowers/specs/2026-08-17-healthbench-main-port-design.md`

## Global Constraints

- Branch: `feature/healthbench-main`. Already created; do not branch again.
- **Commit messages must NOT contain a `Co-Authored-By` trailer.** Explicit requester instruction.
- One atomic commit per task. Do not batch tasks into a single commit.
- ruff: `line-length = 88`, `target-version = "py312"`. Run `mise run format` before committing.
- Never introduce internal Lunit hostnames (`shared-cluster-vm-026`) or paths (`/mnt/vast/...`) — CoEval is a public Apache-2.0 repo.
- Do not modify: `src/coeval/conf/datasets/all.yaml`, `src/coeval/conf/runner/default.yaml`, `src/coeval/metrics/healthbench_rubric.py`, `_HealthBenchDatasetBase`, `HealthBenchConsensusDataset`.
- Test commands: `mise run test` (unit), `uv run pytest tests/unit/<file> -v` (single file).

---

## Task 0: Verify the environment builds — ALREADY DONE

The repo's `.venv` was empty and set to Python 3.13 (pyproject requires 3.12), and `mise.toml` was untrusted. `uv run` on 3.13 fails building `outlines-core` from source because Rust is absent.

This was resolved while writing the plan: `mise trust && mise run sync` completed (exit 0) and `mise run test` reported **304 passed** as the baseline. Re-run the steps below only if the environment looks broken.

**Files:** none (environment only)

- [x] **Step 1: Trust mise and sync**

```bash
mise trust
mise run sync
```

Expected: Python 3.12 provisioned, `.venv` populated. This pulls `sglang[all]` (torch etc.) and can take several minutes.

- [x] **Step 2: Confirm the existing suite is green before changing anything**

```bash
mise run test
```

Expected: PASS — 304 tests at baseline. If it fails here, stop and report; the baseline is broken and nothing below can be trusted.

- [x] **Step 3: No commit**

Environment setup produces no tracked changes.

---

## Task 1: `clipped_avg_aggregator`

The official HealthBench reporting metric. Main mixes positive and negative (penalty) rubric criteria, so a per-example score — and the mean — can go net-negative.

**Files:**
- Modify: `src/coeval/util/aggregation.py` (insert after `avg_aggregator`, before `weighted_avg_aggregator`)
- Test: `tests/unit/test_aggregation.py`

**Interfaces:**
- Consumes: `avg_aggregator`, `AggregationResult`, `MetricScoreDetail` — all already in the module.
- Produces: `clipped_avg_aggregator(results: list[EvalResult], key_to_name: dict[str, str] | None = None) -> AggregationResult`. Task 5 and Task 6 reference it as `coeval.util.aggregation.clipped_avg_aggregator`.

- [ ] **Step 1: Write the failing tests**

Add this helper near the other factories at the top of `tests/unit/test_aggregation.py`. The existing `make_eval_result` forces classification details, which these tests do not want.

```python
def make_score_result(sample_id: int, metric_name: str, score: float) -> EvalResult:
    """Create an EvalResult carrying a single bare score, no classification details."""
    return EvalResult(
        sample_id=sample_id,
        test_case=LLMTestCase(input="test", actual_output=""),
        metrics=[MetricResult(name=metric_name, score=score, passed=score >= 0.5)],
        generation_time_ms=100.0,
        scoring_time_ms=10.0,
    )
```

Add `clipped_avg_aggregator` to the existing `from coeval.util.aggregation import (...)` block, and append this test class at the end of the file:

```python
class TestClippedAvgAggregator:
    """clip(mean(per-example scores), 0, 1) — the official HealthBench formula."""

    def test_negative_mean_clips_to_zero(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([-0.5, -0.3])]
        assert clipped_avg_aggregator(results).metric_scores["m"].score == 0.0

    def test_mean_above_one_clips_to_one(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([1.5, 1.2])]
        assert clipped_avg_aggregator(results).metric_scores["m"].score == 1.0

    def test_in_range_mean_passes_through(self) -> None:
        results = [make_score_result(i, "m", s) for i, s in enumerate([0.4, 0.6])]
        detail = clipped_avg_aggregator(results).metric_scores["m"]
        assert detail.score == pytest.approx(0.5)

    def test_numerator_rescaled_so_merge_cannot_undo_clip(self) -> None:
        """weighted_merge recomputes sum(num)/sum(den); it must not resurrect -0.75."""
        results = [make_score_result(i, "m", s) for i, s in enumerate([-1.0, -0.5])]
        detail = clipped_avg_aggregator(results).metric_scores["m"]
        assert detail.score == 0.0
        assert detail.denominator == 2.0
        assert detail.numerator == 0.0

    def test_breakdown_keeps_raw_mean(self) -> None:
        """Raw statistics survive as diagnostics even when the score is clipped."""
        results = [make_score_result(i, "m", s) for i, s in enumerate([-0.5, -0.3])]
        assert clipped_avg_aggregator(results).breakdown["m"]["mean"] == pytest.approx(-0.4)

    def test_empty_results(self) -> None:
        assert clipped_avg_aggregator([]).metric_scores == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_aggregation.py::TestClippedAvgAggregator -v`
Expected: FAIL at import — `ImportError: cannot import name 'clipped_avg_aggregator'`

- [ ] **Step 3: Write the implementation**

In `src/coeval/util/aggregation.py`, directly after `avg_aggregator`:

```python
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
    recomputing ``sum(num) / sum(den)`` cannot undo the clip. ``breakdown`` and
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_aggregation.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Format, lint, commit**

```bash
mise run format && mise run lint
git add src/coeval/util/aggregation.py tests/unit/test_aggregation.py
git commit -m "feat(aggregation): add clipped_avg_aggregator

Clips the reported mean to [0, 1] per the official HealthBench formula,
for subsets whose penalty criteria can drive a per-example score net-negative.

Rescales numerator to score * denominator so a downstream weighted_merge
recomputing sum(num)/sum(den) cannot undo the clip. Raw statistics stay in
breakdown as diagnostics."
```

---

## Task 2: `HealthBenchMainDataset`

**Files:**
- Modify: `src/coeval/datasets/healthbench.py`
- Modify: `src/coeval/datasets/__init__.py`
- Test: `tests/unit/test_healthbench_dataset.py`

**Interfaces:**
- Consumes: `_HealthBenchDatasetBase` (provides `_load`, `_read_or_download`, `_build_golden`, `_stratified_sample`, `_get_theme`), `register_dataset`.
- Produces: `coeval.datasets.healthbench.MAIN_URL` and `HealthBenchMainDataset`, registered as `"healthbench_main"`, exported from `coeval.datasets`. Task 5's YAML targets `coeval.datasets.healthbench.HealthBenchMainDataset`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_healthbench_dataset.py`, change the module docstring to `"""Tests for HealthBench dataset loaders."""` and extend the import to:

```python
from coeval.datasets.healthbench import (
    HealthBenchConsensusDataset,
    HealthBenchMainDataset,
)
```

Append this helper and these two test classes at the end of the file:

```python
def _mock_urlopen(mock_urlopen: MagicMock, samples: list[dict]) -> None:
    """Wire a patched urlopen to return the given samples as a JSONL response."""
    mock_response = MagicMock()
    mock_response.read.return_value = _mock_jsonl_response(samples)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response


class TestHealthBenchMainDataset:
    @pytest.fixture(autouse=True)
    def no_cache(self, tmp_path):
        """Point cache dir at an empty temp directory so the real cache is untouched."""
        with patch(
            "coeval.datasets.healthbench._CACHE_DIR",
            tmp_path / "coeval_cache",
        ):
            yield

    def test_points_at_oss_eval_blob(self) -> None:
        assert HealthBenchMainDataset._URL.endswith(
            "2025-05-07-06-14-12_oss_eval.jsonl"
        )

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_loads_conversational_goldens(self, mock_urlopen: MagicMock) -> None:
        _mock_urlopen(mock_urlopen, [_make_sample(f"id-{i}") for i in range(4)])
        dataset = HealthBenchMainDataset()
        assert len(dataset.goldens) == 4
        assert isinstance(dataset.goldens[0], ConversationalGolden)
        assert dataset.name == "HealthBenchMain"

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_preserves_penalty_rubrics(self, mock_urlopen: MagicMock) -> None:
        """Negative-point criteria survive the load verbatim — the clip depends on them."""
        sample = _make_sample(
            prompt_id="penalty-001",
            rubrics=[
                {"criterion": "Accurate", "points": 5.0, "tags": ["accuracy"]},
                {
                    "criterion": "Overly verbose",
                    "points": -2.0,
                    "tags": ["cluster:concision"],
                },
            ],
        )
        _mock_urlopen(mock_urlopen, [sample])
        rubrics = HealthBenchMainDataset().goldens[0].additional_metadata["rubrics"]
        assert [r["points"] for r in rubrics] == [5.0, -2.0]


class TestHealthBenchSubsetWiring:
    def test_registry_names(self) -> None:
        assert HealthBenchMainDataset._registry_name == "healthbench_main"
        assert HealthBenchConsensusDataset._registry_name == "healthbench_consensus"

    def test_exported_from_package(self) -> None:
        import coeval.datasets as datasets_pkg

        for name in ("HealthBenchMainDataset", "HealthBenchConsensusDataset"):
            assert name in datasets_pkg.__all__
            assert hasattr(datasets_pkg, name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_healthbench_dataset.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'HealthBenchMainDataset'`

- [ ] **Step 3: Add the dataset class**

In `src/coeval/datasets/healthbench.py`, replace the `Available subsets:` block of the module docstring with:

```
Available subsets:
    - Main: 5,000 examples (``oss_eval``) — the headline HealthBench eval
    - Consensus: 3,671 examples with 34 consensus criteria
```

Add next to `CONSENSUS_URL`:

```python
MAIN_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
```

Insert this class immediately after `_HealthBenchDatasetBase` and before `HealthBenchConsensusDataset`:

```python
@register_dataset("healthbench_main")
class HealthBenchMainDataset(_HealthBenchDatasetBase):
    """HealthBench main subset (``oss_eval``, 5,000 examples).

    The headline HealthBench eval. Consensus and Hard are both slices of it, but
    they overlap on 586 prompts and cover only 4,085/5,000 together, so running
    those is not a substitute for this.

    Rubrics mix positive and negative (penalty) point criteria, so a per-example
    score can go net-negative; the reported metric clips the mean to [0, 1]
    (see :func:`coeval.util.aggregation.clipped_avg_aggregator`).

    Usage:
        dataset = HealthBenchMainDataset(num_samples=5)
        print(len(dataset.goldens))
    """

    _URL = MAIN_URL
    _LABEL = "Main"

    @property
    def name(self) -> str:
        return "HealthBenchMain"
```

- [ ] **Step 4: Export it**

In `src/coeval/datasets/__init__.py`, change the healthbench import to:

```python
from coeval.datasets.healthbench import (
    HealthBenchConsensusDataset,
    HealthBenchMainDataset,
)
```

and add `"HealthBenchMainDataset",` to `__all__` immediately after `"HealthBenchConsensusDataset",`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_healthbench_dataset.py -v`
Expected: PASS, including the pre-existing Consensus tests.

- [ ] **Step 6: Format, lint, commit**

```bash
mise run format && mise run lint
git add src/coeval/datasets/healthbench.py src/coeval/datasets/__init__.py tests/unit/test_healthbench_dataset.py
git commit -m "feat(datasets): add HealthBench Main (oss_eval) subset

The headline HealthBench eval, 5,000 examples. Consensus and Hard are
curated slices that overlap on 586 prompts and cover only 4,085/5,000
together, so neither substitutes for it.

Built on the existing _HealthBenchDatasetBase, so stratified sample_ratio
and the XDG download cache come for free and the Consensus loader is
untouched."
```

---

## Task 3: `CodexExecJudge`

A `DeepEvalBaseLLM` that grades by spawning one headless `codex exec` process per call.

**Files:**
- Create: `src/coeval/clients/codex_exec_judge.py`
- Modify: `src/coeval/clients/__init__.py`
- Test: `tests/unit/test_codex_exec_judge.py`

**Interfaces:**
- Consumes: `deepeval.models.DeepEvalBaseLLM`.
- Produces: `coeval.clients.CodexExecJudge` with
  `__init__(model: str = "gpt-5.6-sol", sandbox: str = "read-only", timeout: float = 300.0, output_schema: dict[str, Any] | None = None, cwd: str | None = None, system_prompt: str = "")`,
  `async a_generate(prompt: str, *, system_prompt: str | None = None, **_kwargs) -> str`,
  `generate(prompt: str, *, system_prompt: str | None = None, **_kwargs) -> str`.
  Task 4's YAML targets `coeval.clients.CodexExecJudge`.

**Critical base-class behaviour:** `DeepEvalBaseLLM.__init__` runs
`self.name = parse_model_name(model)` and then `self.model = self.load_model()`.
It **overwrites** `self.model`. Store the CLI model id in `self._model_id`, set it
*before* calling `super().__init__()` (because `load_model()` runs inside it), and
use `self._model_id` — never `self.model` — when building argv.

**Behaviours the spike established (codex-cli 0.146.0):**
- `codex exec` exits 0 even when every request 401s, leaving the `-o` file empty. The output file, not the return code, is the authoritative success signal.
- `warning: Codex could not find bubblewrap on PATH` is written to stderr on most systems and is harmless.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_codex_exec_judge.py`:

```python
"""Tests for the codex exec-backed judge.

No real subprocess is spawned: asyncio.create_subprocess_exec is patched with a
fake that writes the output file the way codex exec would.
"""

import asyncio
import json
from pathlib import Path

import pytest

from coeval.clients.codex_exec_judge import CodexExecJudge

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
    "additionalProperties": False,
}

VALID_OUTPUT = '{"explanation": "meets the criterion", "criteria_met": true}'


class _FakeProc:
    """Stand-in for a codex exec process.

    output=None simulates codex deleting/never producing the file; output=""
    simulates the observed failure mode where it exits 0 having written nothing.
    """

    def __init__(self, argv, returncode=0, stderr=b"", output=VALID_OUTPUT, hang=False):
        self._argv = argv
        self.returncode = returncode
        self._stderr = stderr
        self._output = output
        self._hang = hang
        self.killed = False
        self.stdin_payload = None

    async def communicate(self, payload=None):
        self.stdin_payload = payload
        if self._hang:
            await asyncio.sleep(3600)
        out_path = Path(self._argv[self._argv.index("-o") + 1])
        if self._output is None:
            out_path.unlink(missing_ok=True)
        else:
            out_path.write_text(self._output)
        return b"", self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


@pytest.fixture
def spawn(monkeypatch):
    """Install a fake subprocess spawner; returns a dict recording argv and proc."""

    def _install(**proc_kwargs):
        state: dict = {}

        async def _create(*argv, **_kwargs):
            proc = _FakeProc(list(argv), **proc_kwargs)
            state["argv"] = list(argv)
            state["proc"] = proc
            return proc

        monkeypatch.setattr(
            "coeval.clients.codex_exec_judge.asyncio.create_subprocess_exec",
            _create,
        )
        return state

    return _install


async def test_returns_output_file_contents(spawn) -> None:
    spawn(output=VALID_OUTPUT)
    result = await CodexExecJudge(model="gpt-5.6-sol").a_generate("grade this")
    assert json.loads(result)["criteria_met"] is True


async def test_exit_zero_with_empty_output_raises(spawn) -> None:
    """codex exec exits 0 even when every request 401s, leaving -o empty."""
    spawn(output="")
    with pytest.raises(RuntimeError, match="no output"):
        await CodexExecJudge().a_generate("grade this")


async def test_missing_output_file_raises(spawn) -> None:
    spawn(output=None)
    with pytest.raises(RuntimeError, match="no output"):
        await CodexExecJudge().a_generate("grade this")


async def test_whitespace_only_output_raises(spawn) -> None:
    spawn(output="   \n  ")
    with pytest.raises(RuntimeError, match="no output"):
        await CodexExecJudge().a_generate("grade this")


async def test_nonzero_exit_raises(spawn) -> None:
    spawn(returncode=2, output=VALID_OUTPUT)
    with pytest.raises(RuntimeError, match="exit code 2"):
        await CodexExecJudge().a_generate("grade this")


async def test_timeout_kills_process_and_raises(spawn) -> None:
    state = spawn(hang=True)
    with pytest.raises(RuntimeError, match="timed out"):
        await CodexExecJudge(timeout=0.05).a_generate("grade this")
    assert state["proc"].killed is True


async def test_bubblewrap_warning_on_stderr_still_succeeds(spawn) -> None:
    """stderr noise is not a failure signal."""
    spawn(
        stderr=b"warning: Codex could not find bubblewrap on PATH",
        output=VALID_OUTPUT,
    )
    assert await CodexExecJudge().a_generate("grade this") == VALID_OUTPUT


async def test_argv_contains_headless_flags(spawn) -> None:
    state = spawn(output=VALID_OUTPUT)
    await CodexExecJudge(model="gpt-5.6-sol", sandbox="read-only").a_generate("x")
    argv = state["argv"]
    assert argv[:2] == ["codex", "exec"]
    for flag in ("--ephemeral", "--skip-git-repo-check", "-o", "-C"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "gpt-5.6-sol"
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[-1] == "-"


async def test_output_schema_flag_only_when_schema_given(spawn) -> None:
    state = spawn(output=VALID_OUTPUT)
    await CodexExecJudge().a_generate("x")
    assert "--output-schema" not in state["argv"]

    # Keep a reference: the schema file lives in a TemporaryDirectory owned by
    # the judge, which is wiped when the instance is garbage collected.
    state = spawn(output=VALID_OUTPUT)
    judge = CodexExecJudge(output_schema=GRADING_SCHEMA)
    await judge.a_generate("x")
    schema_path = state["argv"][state["argv"].index("--output-schema") + 1]
    assert json.loads(Path(schema_path).read_text()) == GRADING_SCHEMA


async def test_system_prompt_prepended_to_stdin(spawn) -> None:
    """codex exec has no system-prompt flag, so it rides in the prompt body."""
    state = spawn(output=VALID_OUTPUT)
    await CodexExecJudge().a_generate("grade this", system_prompt="be terse")
    payload = state["proc"].stdin_payload.decode()
    assert "be terse" in payload
    assert "grade this" in payload


async def test_no_system_prompt_sends_bare_prompt(spawn) -> None:
    state = spawn(output=VALID_OUTPUT)
    await CodexExecJudge().a_generate("grade this")
    assert state["proc"].stdin_payload.decode() == "grade this"


def test_model_id_survives_base_class_init() -> None:
    """DeepEvalBaseLLM.__init__ overwrites self.model with load_model()."""
    judge = CodexExecJudge(model="gpt-5.6-sol")
    assert judge._model_id == "gpt-5.6-sol"
    assert judge.get_model_name() == judge.name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_codex_exec_judge.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'coeval.clients.codex_exec_judge'`

- [ ] **Step 3: Write the implementation**

Create `src/coeval/clients/codex_exec_judge.py`:

```python
"""DeepEvalBaseLLM adapter that grades via the ``codex exec`` CLI.

One judge call is one headless ``codex exec`` process. Unlike
:class:`~coeval.clients.passthrough_judge.PassthroughJudge` there is no HTTP
endpoint: Codex uses its own persisted credentials, so the machine must be
authenticated once with::

    codex login --api-key "$OPENAI_API_KEY"

Setting ``OPENAI_API_KEY`` alone is not sufficient.

Verified against codex-cli 0.146.0. Two CLI behaviours drive the error handling
here:

- ``codex exec`` exits 0 even when every upstream request fails; on failure it
  simply leaves the ``-o`` output file empty. The output file, not the return
  code, is the authoritative success signal.
- A ``bubblewrap not found`` warning is written to stderr on most systems and is
  harmless — Codex falls back to a bundled copy. stderr is never treated as a
  failure signal.

This class does not retry. ``grade_with_retry`` in
:mod:`coeval.metrics.healthbench_rubric` already catches exceptions, retries
``MAX_RETRIES`` times, and then applies HealthBench's official
``on_failure="false"`` semantics.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)


class CodexExecJudge(DeepEvalBaseLLM):
    """Judge backed by headless ``codex exec`` invocations.

    Args:
        model: Model id passed to ``codex exec --model``.
        sandbox: Codex sandbox policy (``read-only``, ``workspace-write``,
            ``danger-full-access``). Grading needs no writes.
        timeout: Seconds to wait for one process before killing it.
        output_schema: JSON Schema forced on the model's final message via
            ``--output-schema``. ``None`` leaves the response free-form.
        cwd: Working root for Codex. Defaults to a private temp directory so
            Codex cannot read the surrounding repository and let its contents
            leak into a grading verdict.
        system_prompt: ``codex exec`` has no system-prompt flag, so this is
            prepended to the prompt as a labelled block rather than sent as a
            separate system message. This differs from ``PassthroughJudge``.
    """

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        sandbox: str = "read-only",
        timeout: float = 300.0,
        output_schema: dict[str, Any] | None = None,
        cwd: str | None = None,
        system_prompt: str = "",
    ) -> None:
        # Set before super().__init__ — it calls load_model(), and it also
        # overwrites self.model with that return value.
        self._model_id = model
        self.sandbox = sandbox
        self.timeout = timeout
        self._system_prompt = system_prompt
        self._tmpdir = tempfile.TemporaryDirectory(prefix="coeval-codex-")
        self._cwd = cwd or self._tmpdir.name

        self._schema_path: str | None = None
        if output_schema is not None:
            schema_file = Path(self._tmpdir.name) / "output_schema.json"
            schema_file.write_text(json.dumps(dict(output_schema)))
            self._schema_path = str(schema_file)

        super().__init__(model=model)

    def load_model(self) -> str:
        """Return the model id — there is no in-process model object to load."""
        return self._model_id

    def get_model_name(self) -> str:
        """Return the judge model name."""
        return self.name

    def _build_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """Fold the system prompt into the prompt body; codex exec has no flag for it."""
        sp = system_prompt if system_prompt is not None else self._system_prompt
        if not sp:
            return prompt
        return f"# System\n{sp}\n\n# Task\n{prompt}"

    def _build_argv(self, out_path: str) -> list[str]:
        """Assemble the codex exec command line for one grading call."""
        argv = [
            "codex",
            "exec",
            "--model",
            self._model_id,
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-s",
            self.sandbox,
            "-C",
            self._cwd,
        ]
        if self._schema_path is not None:
            argv += ["--output-schema", self._schema_path]
        argv += ["-o", out_path, "-"]
        return argv

    async def a_generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        """Run one codex exec process and return its final message.

        Raises:
            RuntimeError: On timeout, non-zero exit, or an empty/absent output
                file. Callers rely on this to trigger their own retry logic.
        """
        fd, out_path = tempfile.mkstemp(suffix=".json", dir=self._tmpdir.name)
        os.close(fd)
        argv = self._build_argv(out_path)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            payload = self._build_prompt(prompt, system_prompt).encode()
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(payload), timeout=self.timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"codex exec timed out after {self.timeout}s"
                ) from None

            if stderr:
                logger.debug(
                    "codex exec stderr: %s", stderr.decode(errors="replace").strip()
                )
            if proc.returncode != 0:
                raise RuntimeError(f"codex exec failed with exit code {proc.returncode}")

            # codex exec exits 0 even when every request fails, leaving this file
            # empty — so the file, not the return code, decides.
            output = (
                Path(out_path).read_text().strip()
                if os.path.exists(out_path)
                else ""
            )
            if not output:
                raise RuntimeError(
                    "codex exec produced no output; check `codex login status`"
                )
            return output
        finally:
            Path(out_path).unlink(missing_ok=True)

    def generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        """Synchronously run one codex exec process."""
        return asyncio.run(self.a_generate(prompt, system_prompt=system_prompt))
```

- [ ] **Step 4: Export it**

In `src/coeval/clients/__init__.py`, add to the docstring's client list
`- CodexExecJudge: DeepEvalBaseLLM judge backed by the codex exec CLI`, add the
import `from coeval.clients.codex_exec_judge import CodexExecJudge`, and add
`"CodexExecJudge",` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_codex_exec_judge.py -v`
Expected: PASS (12 tests)

- [ ] **Step 6: Format, lint, commit**

```bash
mise run format && mise run lint
git add src/coeval/clients/codex_exec_judge.py src/coeval/clients/__init__.py tests/unit/test_codex_exec_judge.py
git commit -m "feat(clients): add CodexExecJudge for headless codex exec grading

One judge call is one codex exec process. Codex uses its own persisted
credentials, so the host needs codex login --api-key once.

Verified against codex-cli 0.146.0, which exits 0 even when every upstream
request 401s and simply leaves the -o file empty — so success is decided by
the output file, not the return code. Benign bubblewrap warnings on stderr
are logged at debug and never treated as failure.

No internal retry: grade_with_retry already retries and applies HealthBench's
on_failure=false semantics."
```

---

## Task 4: Judge configs

**Files:**
- Modify: `src/coeval/conf/datasets/metrics/judge/gpt-4.1.yaml`
- Create: `src/coeval/conf/datasets/metrics/judge/gpt-5.6-sol.yaml`

**Interfaces:**
- Consumes: `coeval.clients.PassthroughJudge` (existing), `coeval.clients.CodexExecJudge` (Task 3).
- Produces: two entries in the `datasets/metrics/judge` config group, selectable as `datasets/metrics/judge@healthbench_judge=<name>`.

- [ ] **Step 1: Make gpt-4.1's endpoint overridable**

Replace `src/coeval/conf/datasets/metrics/judge/gpt-4.1.yaml` with:

```yaml
# Judge: gpt-4.1 via OpenAI API, or any OpenAI-compatible endpoint via OPENAI_API_BASE
_target_: coeval.clients.PassthroughJudge
api_base: ${oc.env:OPENAI_API_BASE,https://api.openai.com/v1}
model: gpt-4.1
api_key: ${oc.env:OPENAI_API_KEY}
temperature: 0.0
max_tokens: 2048
```

- [ ] **Step 2: Add the codex judge config**

Create `src/coeval/conf/datasets/metrics/judge/gpt-5.6-sol.yaml`:

```yaml
# Judge: gpt-5.6-sol via headless `codex exec` (codex-cli >= 0.146.0).
#
# Codex uses its own persisted credentials, not OPENAI_API_KEY directly.
# Authenticate the host once:
#   codex login --api-key "$OPENAI_API_KEY"
#
# One grading call is one process, so keep the metric's concurrency modest:
#   mise run eval -- datasets=healthbench_main \
#     datasets/metrics/judge@healthbench_judge=gpt-5.6-sol \
#     metrics.healthbench_main.healthbench_rubric.concurrent_limit=4
#
# `_convert_: all` makes Hydra hand `output_schema` over as a plain dict rather
# than a DictConfig, so the client can json.dumps it without importing omegaconf.
# The schema matches what GRADER_TEMPLATE asks the judge to return.
_target_: coeval.clients.CodexExecJudge
_convert_: all
model: gpt-5.6-sol
sandbox: read-only
timeout: 300.0
output_schema:
  type: object
  properties:
    explanation:
      type: string
    criteria_met:
      type: boolean
  required:
    - explanation
    - criteria_met
  additionalProperties: false
```

- [ ] **Step 3: Verify both compose**

`healthbench_main.yaml` does not exist yet, so check against the existing Consensus config:

```bash
uv run coeval datasets=healthbench_consensus --cfg job | head -40
uv run coeval datasets=healthbench_consensus datasets/metrics/judge@healthbench_judge=gpt-5.6-sol --cfg job | head -40
```

Expected: both print a config; the second shows `_target_: coeval.clients.CodexExecJudge` under `healthbench_judge` with the `output_schema` mapping intact. No Hydra composition errors.

- [ ] **Step 4: Commit**

```bash
git add src/coeval/conf/datasets/metrics/judge/
git commit -m "feat(conf): add gpt-5.6-sol codex judge, make gpt-4.1 endpoint overridable

gpt-5.6-sol grades through headless codex exec and pins the grader's reply
shape with --output-schema, matching what GRADER_TEMPLATE asks for.

gpt-4.1 now reads api_base from OPENAI_API_BASE so it can target a
compatible proxy without editing the file."
```

---

## Task 5: `healthbench_main` dataset config

**Files:**
- Create: `src/coeval/conf/datasets/healthbench_main.yaml`

**Interfaces:**
- Consumes: `HealthBenchMainDataset` (Task 2), `clipped_avg_aggregator` (Task 1), the `gpt-4.1` judge (Task 4), and the existing `coeval.metrics.healthbench_rubric.HealthBenchRubricMetric`.
- Produces: the `datasets=healthbench_main` Hydra selection.

- [ ] **Step 1: Write the config**

Create `src/coeval/conf/datasets/healthbench_main.yaml`:

```yaml
# @package _global_
#
# Standalone HealthBench Main config — the full `oss_eval` subset (5,000 examples),
# the headline HealthBench benchmark. Consensus and Hard are curated slices of it.
#
# Scoring: clipped_avg_aggregator = clip(mean(per-example scores), 0, 1) — the
# official HealthBench formula. Main mixes positive and negative (penalty) criteria,
# so a per-example score — and the mean — can go net-negative; the clip is
# load-bearing here, unlike on Consensus where it is a no-op safeguard.
#
# Cost: 5,000 examples × many rubric criteria each = tens of thousands of judge
# calls. Smoke-test first:
#   mise run eval -- datasets=healthbench_main num_samples=5
# Stratified theme subset (preserves theme distribution):
#   mise run eval -- datasets=healthbench_main '++datasets.healthbench_main.sample_ratio=0.1'
#
# Judge model swap (shared config group: conf/datasets/metrics/judge/):
#   mise run eval -- datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-4.1
#   mise run eval -- datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-5.6-sol

defaults:
  - metrics/judge@healthbench_judge: gpt-4.1

datasets:
  healthbench_main:
    _target_: coeval.datasets.healthbench.HealthBenchMainDataset
    num_samples: ${num_samples}
    system_prompt: ${system_prompt}
    sample_ratio: null

metrics:
  healthbench_main:
    healthbench_rubric:
      _target_: coeval.metrics.healthbench_rubric.HealthBenchRubricMetric
      concurrent_limit: 10
      judge: ${healthbench_judge}

score_aggregators:
  healthbench_main:
    _target_: coeval.util.aggregation.clipped_avg_aggregator
    _partial_: true
```

- [ ] **Step 2: Verify it composes**

```bash
uv run coeval datasets=healthbench_main num_samples=2 --cfg job
```

Expected: prints a config where `datasets.healthbench_main`, `metrics.healthbench_main`, and `score_aggregators.healthbench_main` are all present, with `_target_: coeval.util.aggregation.clipped_avg_aggregator`. No execution, no network.

- [ ] **Step 3: Verify the judge override composes**

```bash
uv run coeval datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-5.6-sol --cfg job | head -40
```

Expected: `healthbench_judge._target_` is `coeval.clients.CodexExecJudge`.

- [ ] **Step 4: Commit**

```bash
git add src/coeval/conf/datasets/healthbench_main.yaml
git commit -m "feat(conf): add healthbench_main dataset config

Wires HealthBenchMainDataset to HealthBenchRubricMetric and
clipped_avg_aggregator, defaulting to the gpt-4.1 judge.

Deliberately not added to datasets/all.yaml: 5,000 examples times many
rubric criteria each would make datasets=all smoke runs ruinous."
```

---

## Task 6: Switch Consensus to the clipped aggregator

Requested so both HealthBench subsets share one aggregation path. On Consensus the clip is a **no-op safeguard**, not load-bearing — the comment must say so.

**Files:**
- Modify: `src/coeval/conf/datasets/healthbench_consensus.yaml`

- [ ] **Step 1: Rewrite the scoring comment and the aggregator**

Replace the scoring comment block (currently lines beginning `# Scoring: avg_aggregator ...` through `# scores are always in [0, 1].`) with:

```yaml
# Scoring: clipped_avg_aggregator = clip(mean(per-example scores), 0, 1) — the
# official HealthBench formula. On Consensus the clip is a no-op safeguard rather
# than load-bearing: all 8,053 rubric criteria in this subset carry a uniform +5
# with no negative penalties, so per-example scores are always in [0, 1]. It is
# used anyway so Consensus and Main share one aggregation path.
```

Also update the judge-swap example comments to reference only judges that exist
in this repo (`gpt-4.1`, `gpt-5.6-sol`) — drop `gpt-5-mini` and `kimi-25`.

Then change the aggregator target:

```yaml
score_aggregators:
  healthbench_consensus:
    _target_: coeval.util.aggregation.clipped_avg_aggregator
    _partial_: true
```

- [ ] **Step 2: Verify it composes and the suite still passes**

```bash
uv run coeval datasets=healthbench_consensus --cfg job | head -40
mise run test
```

Expected: `score_aggregators.healthbench_consensus._target_` is
`coeval.util.aggregation.clipped_avg_aggregator`; all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/coeval/conf/datasets/healthbench_consensus.yaml
git commit -m "refactor(conf): use clipped_avg_aggregator for healthbench_consensus

Gives Consensus and Main a single aggregation path. On Consensus the clip
is a no-op safeguard, not load-bearing — every criterion in the subset
carries a uniform +5 with no penalties, so per-example scores never leave
[0, 1]. The comment now says exactly that.

Also drops judge-swap examples naming presets this repo does not ship."
```

---

## Task 7: Raise the passthrough generation budget

HealthBench answers are long-form open text. At `max_tokens: 4096` they get truncated, which silently depresses rubric scores — the judge grades a cut-off answer without any error surfacing.

**Files:**
- Modify: `src/coeval/conf/client/passthrough.yaml`

- [ ] **Step 1: Update the generation parameters**

Change only these keys. `api_base`, `model`, and `system_prompt` stay as they are — upstream's values are internal infrastructure or depend on `coe_common`, which CoEval does not have.

```yaml
  temperature: 0.0
  # Open-ended benchmarks (HealthBench) produce long-form answers; a small budget
  # truncates them and silently depresses rubric scores.
  max_tokens: 32768
  timeout: 360.0
  api_key: ${oc.env:OPENAI_API_KEY,null}
  additional_kwargs:
    top_p: 1.0
```

- [ ] **Step 2: Verify it composes**

```bash
uv run coeval datasets=healthbench_main --cfg job | head -30
```

Expected: `client.max_tokens: 32768`, `client.timeout: 360.0`, and
`client.additional_kwargs.top_p: 1.0`. `PassthroughClient.__init__` already
accepts `additional_kwargs`, so no code change is needed.

- [ ] **Step 3: Commit**

```bash
git add src/coeval/conf/client/passthrough.yaml
git commit -m "chore(conf): raise passthrough generation budget for long-form answers

HealthBench responses are open-ended prose. At max_tokens 4096 they were
truncated and graded as-is, silently depressing rubric scores with no error.

Raises the budget to 32768 with a matching 360s timeout and pins top_p=1.0,
matching the upstream chain-of-evidence client. Endpoint and model stay on
the public localhost defaults."
```

---

## Task 8: Remove the gpt-oss-120b client config

**Files:**
- Delete: `src/coeval/conf/client/gpt-oss-120b.yaml`

- [ ] **Step 1: Confirm nothing references it**

```bash
grep -rn "gpt-oss" README.md src tests mise.toml scripts
```

Expected: matches only inside `src/coeval/conf/client/gpt-oss-120b.yaml` itself. If
anything else matches, stop and report rather than deleting.

- [ ] **Step 2: Delete and verify**

```bash
git rm src/coeval/conf/client/gpt-oss-120b.yaml
mise run test
uv run coeval datasets=healthbench_main --cfg job | head -5
```

Expected: tests PASS, config still composes on the default `client: passthrough`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(conf): remove gpt-oss-120b client config

Unreferenced anywhere else in the repo and no longer part of the supported
client set."
```

---

## Task 9: Document HealthBench Main and the codex judge

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the dataset row**

In the datasets table (around line 157), add a row directly above the existing
HealthBench Consensus row:

```markdown
| [HealthBench](https://huggingface.co/datasets/openai/HealthBench) | `healthbench_main` | OpenAI | Open-ended multi-turn | LLM-as-judge (rubric) |
```

Change the existing HealthBench row's subset cell to make the distinction explicit:
`healthbench_consensus` stays as-is, but append to the table (or the paragraph
beneath it) one sentence: *"`healthbench_main` is the headline 5,000-example
benchmark; `healthbench_consensus` is a physician-validated slice of it."*

- [ ] **Step 2: Add usage and the codex prerequisite**

In the judge-swap section (around line 218), replace the single example with:

````markdown
```bash
# Swap judge model for HealthBench
mise run eval -- datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-4.1

# Grade with gpt-5.6-sol through headless `codex exec` (codex-cli >= 0.146.0).
# Authenticate the host once, then keep concurrency modest — each grading call
# is a separate process, not an HTTP request:
#   codex login --api-key "$OPENAI_API_KEY"
mise run eval -- datasets=healthbench_main \
  datasets/metrics/judge@healthbench_judge=gpt-5.6-sol \
  metrics.healthbench_main.healthbench_rubric.concurrent_limit=4
```
````

In the HealthBench quick-start example (around line 118), add below it:

````markdown
```bash
# Full HealthBench (5,000 examples) — smoke-test with num_samples first
mise run eval -- datasets=healthbench_main num_samples=5
```
````

- [ ] **Step 3: Verify the links and commands are accurate**

```bash
grep -n "healthbench_main\|gpt-5.6-sol\|codex login" README.md
```

Expected: every command shown corresponds to a config that now exists. Cross-check
each `datasets/metrics/judge@healthbench_judge=<name>` against
`ls src/coeval/conf/datasets/metrics/judge/`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document healthbench_main and the codex exec judge

Adds the headline 5,000-example subset to the dataset table, a smoke-test
recipe, and the codex login prerequisite plus the lower concurrent_limit
that process-per-call grading wants."
```

---

## Final verification

- [ ] **Step 1: Full suite, lint, format check**

```bash
mise run format && mise run lint && mise run test
```

Expected: all PASS, no diff left by `format`.

- [ ] **Step 2: Compose checks**

```bash
uv run coeval datasets=healthbench_main num_samples=2 --cfg job
uv run coeval datasets=healthbench_main datasets/metrics/judge@healthbench_judge=gpt-5.6-sol --cfg job
uv run coeval datasets=healthbench_consensus --cfg job
uv run coeval datasets=all num_samples=2 --cfg job
```

Expected: all four compose. The last one confirms `all.yaml` is unaffected and
still excludes `healthbench_main`.

- [ ] **Step 3: Review the commit series**

```bash
git log --oneline main..HEAD
git log main..HEAD --format='%b' | grep -i "co-authored" && echo "FAIL: co-author trailer present" || echo "OK: no co-author trailers"
```

Expected: eleven commits — the spec doc, this plan doc, and the nine feature
commits from Tasks 1-9 (Task 0 produces none) — and no co-author trailers.

- [ ] **Step 4: Smoke run — ASK FIRST**

Do **not** run this without explicit confirmation. It downloads a ~60 MB blob and
bills real judge calls.

```bash
mise run eval -- datasets=healthbench_main num_samples=2
```

If grading with `gpt-5.6-sol`, `codex login status` must report a logged-in state
first; otherwise every criterion silently grades as `criteria_met: false` after
three retries.
