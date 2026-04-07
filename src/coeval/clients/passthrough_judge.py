"""DeepEvalBaseLLM adapter for PassthroughClient.

Wraps PassthroughClient so it can serve as a judge for both:
- deepeval built-in metrics (Faithfulness, GEval, etc.) via generate/a_generate
- custom metrics like HealthBench rubric via a_generate(prompt, system_prompt=...)

This eliminates the need for separate GPTModel and PassthroughClient configs
when the same LLM endpoint is used for both.
"""

import asyncio
from typing import Any

from deepeval.models import DeepEvalBaseLLM

from coeval.clients.passthrough import PassthroughClient


class PassthroughJudge(DeepEvalBaseLLM):
    """DeepEvalBaseLLM backed by PassthroughClient.

    Args:
        api_base: LLM server URL (any OpenAI-compatible API).
        model: Model name.
        system_prompt: Default system prompt for judge calls. Empty by default
            so deepeval metrics receive no system message interference.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        api_key: API key for authentication.
        additional_kwargs: Extra OpenAI parameters (top_p, etc.).
    """

    def __init__(
        self,
        api_base: str,
        model: str = "gpt-4.1",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        api_key: str | None = None,
        additional_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = PassthroughClient(
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
        )
        self._system_prompt = system_prompt
        super().__init__(model=model)

    def load_model(self) -> PassthroughClient:
        return self._client

    def get_model_name(self) -> str:
        return self.name

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        sp = system_prompt if system_prompt is not None else self._system_prompt
        messages: list[dict[str, Any]] = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        return asyncio.run(
            self._client.generate(self._build_messages(prompt, system_prompt))
        )

    async def a_generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        return await self._client.generate(self._build_messages(prompt, system_prompt))
