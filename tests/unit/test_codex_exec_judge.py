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

    ``output=None`` simulates codex never producing the file; ``output=""``
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
