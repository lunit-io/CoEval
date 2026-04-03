"""Passthrough client for baseline evaluation.

Extends LLMClient for OpenAI-compatible API calls.
No retrieval - pure LLM inference for baseline comparison.
"""

import logging
from typing import Any

from coeval.llm.client import LLMClient
from coeval.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class PassthroughClient(LLMClient):
    """Passthrough client - direct LLM inference without retrieval.

    Extends LLMClient with generate() that forwards a complete message list
    to the underlying LLM. System prompts are composed by the dataset's
    ``get_generation_input()`` — this client does not inject its own.

    Works with any OpenAI-compatible API:
    - OpenAI: https://api.openai.com/v1
    - vLLM/SGLang: http://localhost:8000/v1
    - Azure OpenAI: https://{resource}.openai.azure.com/...

    Example:
        >>> client = PassthroughClient(
        ...     api_base="http://localhost:8000/v1",
        ...     model="learning-unit/your-model",
        ... )
        >>> messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        >>> response = await client.generate(messages)
    """

    def __init__(
        self,
        api_base: str,
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        api_key: str | None = None,
        additional_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize PassthroughClient.

        Args:
            api_base: LLM server URL (any OpenAI-compatible API).
            model: Model name.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            api_key: API key for authentication.
            additional_kwargs: Extra OpenAI parameters (top_p, presence_penalty, etc.).
        """
        config = LLMConfig(
            api_base=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            is_function_calling_model=False,
            additional_kwargs=additional_kwargs,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        """Client name for logging/identification."""
        return self.__class__.__name__

    async def generate(self, messages: list[dict[str, Any]]) -> str:
        """Generate response from a chat message list.

        Args:
            messages: Full message list from ``dataset.get_generation_input()``.

        Returns:
            Raw LLM response content.

        Raises:
            LLMError: If API call fails.
        """
        return await self.achat(messages)
