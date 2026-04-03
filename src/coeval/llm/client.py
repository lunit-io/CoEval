"""Async LLM client using llama-index OpenAILike."""

from typing import Any, NoReturn

import httpx
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike

from coeval.llm.config import LLMConfig
from coeval.llm.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


def normalize_api_key(api_key: str | None) -> str | None:
    """Normalize an API key by stripping whitespace and handling empty/null values."""
    if api_key is None or api_key.strip().lower() in ["null", "none", "empty", ""]:
        return None
    return api_key.strip()


class LLMClient:
    """Async client for OpenAI-compatible LLM APIs.

    Wraps llama-index's OpenAILike client with error handling.

    Example:
        config = LLMConfig(api_base="http://localhost:8000/v1", model="your-model")
        client = LLMClient(config)
        response = await client.achat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.llm = OpenAILike(
            model=config.model,
            api_base=config.api_base,
            api_key=normalize_api_key(config.api_key),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            is_chat_model=True,
            is_function_calling_model=config.is_function_calling_model,
            context_window=config.context_window or 4096,
            additional_kwargs=config.get_additional_kwargs_dict(),
        )

    async def achat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The assistant's response content.
        """
        if kwargs:
            raise ValueError(
                f"Call-time parameter overrides not supported: {list(kwargs.keys())}. "
                "Configure temperature/max_tokens in LLMConfig instead."
            )

        chat_messages = self._to_chat_messages(messages)

        try:
            response = await self.llm.achat(chat_messages)
            content = response.message.content

            if content is None:
                raise LLMResponseError("Empty response: message content is None")

            return str(content)

        except ValueError:
            raise
        except (httpx.ConnectError, ConnectionError) as e:
            raise LLMConnectionError(f"Failed to connect to LLM service: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"LLM request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            raise LLMResponseError(
                f"HTTP error (status={e.response.status_code}): {e}"
            ) from e
        except (
            LLMConnectionError,
            LLMTimeoutError,
            LLMResponseError,
            LLMRateLimitError,
        ):
            raise
        except Exception as e:
            self._raise_from_exception(e)

    @staticmethod
    def _to_chat_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
        """Convert dict messages to llama-index ChatMessage objects."""
        role_map = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
            "function": MessageRole.FUNCTION,
        }

        special_fields = {"role", "content"}

        chat_messages: list[ChatMessage] = []
        for msg in messages:
            role_str = msg.get("role", "user")
            if role_str not in role_map:
                raise ValueError(
                    f"Unknown message role: '{role_str}'. "
                    f"Expected one of: {list(role_map.keys())}"
                )
            role = role_map[role_str]
            content = msg.get("content", "")

            additional_kwargs: dict[str, Any] = {}
            for key, value in msg.items():
                if key not in special_fields and value is not None:
                    additional_kwargs[key] = value

            if additional_kwargs:
                chat_messages.append(
                    ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            else:
                chat_messages.append(ChatMessage(role=role, content=content))

        return chat_messages

    @staticmethod
    def _raise_from_exception(e: Exception) -> NoReturn:
        """Convert generic exception to appropriate LLM exception."""
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
        raise LLMError(f"Unexpected LLM error: {e}") from e

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        pass

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
