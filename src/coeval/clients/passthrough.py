"""Passthrough client for baseline evaluation — direct LLM inference, no retrieval.

Composes an ``OpenAILike`` (as ``self.llm`` via ``EvaluationLLMClient``) that Hydra
builds through :func:`coeval.llm.factory.create_llm_client`. Works with any
OpenAI-compatible API — vLLM, SGLang, OpenAI, Azure OpenAI.

Example YAML:
    client:
      _target_: coeval.clients.PassthroughClient
      llm:
        _target_: coeval.llm.factory.create_llm_client
        config:
          _target_: coeval.llm.config.LLMConfig
          api_base: http://shared-cluster-vm-026:9006/v1
          model: default
"""

from typing import Any

from coeval.clients.base import EvaluationLLMClient
from coeval.llm.utils import to_chat_messages


class PassthroughClient(EvaluationLLMClient):
    """Direct LLM inference without retrieval (baseline).

    System prompts are composed by the dataset's ``get_generation_input()`` —
    this client does not inject its own.
    """

    async def generate(self, messages: list[dict[str, Any]]) -> str:
        """Forward a complete message list to the LLM and return its reply.

        Args:
            messages: Full message list from ``dataset.get_generation_input()``.

        Returns:
            Raw LLM response content.

        Raises:
            RuntimeError: If the reply carries no content.
        """
        response = await self.llm.achat(to_chat_messages(messages))
        content = response.message.content
        if content is None:
            raise RuntimeError("Inference response contained no content")
        return str(content)
