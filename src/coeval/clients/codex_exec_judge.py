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
        """Fold the system prompt into the body; codex exec has no flag for it."""
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
                raise RuntimeError(
                    f"codex exec failed with exit code {proc.returncode}"
                )

            # codex exec exits 0 even when every request fails, leaving this file
            # empty — so the file, not the return code, decides.
            output = (
                Path(out_path).read_text().strip() if os.path.exists(out_path) else ""
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
