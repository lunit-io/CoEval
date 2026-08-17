"""Message conversion helpers for OpenAI-compatible LLM clients."""

from typing import Any

from llama_index.core.base.llms.types import ChatMessage


def to_chat_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert OpenAI-style role/content dicts to llama-index ChatMessages.

    Fields beyond ``role``/``content`` (``name``, ``tool_call_id``,
    ``tool_calls``, ...) are preserved in ``additional_kwargs``. An unknown role
    is rejected by ``ChatMessage`` itself.
    """
    return [
        ChatMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            additional_kwargs={
                k: v
                for k, v in msg.items()
                if k not in ("role", "content") and v is not None
            },
        )
        for msg in messages
    ]
