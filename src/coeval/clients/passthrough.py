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
