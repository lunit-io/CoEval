import asyncio
import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)

REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


class CodexExecJudge(DeepEvalBaseLLM):
    """Judge backed by headless ``codex exec`` invocations.

    Args:
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
        reasoning_effort: str | None = "high",
        timeout: float = 300.0,
        output_schema: dict[str, Any] | None = None,
        cwd: str | None = None,
        system_prompt: str = "",
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"Unknown reasoning_effort {reasoning_effort!r}. "
                f"Expected one of {sorted(REASONING_EFFORTS)}, or None to omit the flag."
            )

        # Set before super().__init__ — it calls load_model(), and it also
        # overwrites self.model with that return value.
        self._model_id = model
        self.sandbox = sandbox
        self.reasoning_effort = reasoning_effort
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
        return self.name

    def _build_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """Fold the system prompt into the body; codex exec has no flag for it."""
        sp = system_prompt if system_prompt is not None else self._system_prompt
        if not sp:
            return prompt
        return f"# System\n{sp}\n\n# Task\n{prompt}"

    def _build_argv(self, out_path: str) -> list[str]:
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
        if self.reasoning_effort is not None:
            argv += ["-c", f"model_reasoning_effort={self.reasoning_effort}"]
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
            except asyncio.CancelledError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                raise
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
        return asyncio.run(self.a_generate(prompt, system_prompt=system_prompt))
