import asyncio
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai_like import OpenAILike

from coeval.llm.config import LLMConfig
from coeval.llm.factory import create_llm_client
from coeval.llm.utils import to_chat_messages


class PassthroughJudge(DeepEvalBaseLLM):
    """OpenAI-compatible DeepEval judge with per-call system-prompt overrides."""

    def __init__(
        self,
        config: LLMConfig,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self._llm = create_llm_client(config)
        self._system_prompt = system_prompt
        super().__init__(model=config.model)

    def load_model(self) -> OpenAILike:
        return self._llm

    def get_model_name(self) -> str:
        """Return the judge model name."""
        return self.name

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[ChatMessage]:
        """Build chat messages from system prompt and user content."""
        sp = system_prompt if system_prompt is not None else self._system_prompt
        messages: list[dict[str, Any]] = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt})
        return to_chat_messages(messages)

    def generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        """Synchronously generate a judge response."""
        return asyncio.run(self.a_generate(prompt, system_prompt=system_prompt))

    async def a_generate(
        self, prompt: str, *, system_prompt: str | None = None, **_kwargs: Any
    ) -> str:
        """Asynchronously generate a judge response."""
        response = await self._llm.achat(self._build_messages(prompt, system_prompt))
        return str(response.message.content or "")
