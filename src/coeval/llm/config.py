"""Configuration dataclass for LLM clients."""

from dataclasses import dataclass
from typing import Any

# Type alias for hashable kwargs storage
_KwargsTuple = tuple[tuple[str, Any], ...] | None


@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration for LLM client.

    Attributes:
        api_base: Base URL for the LLM API (e.g., http://shared-cluster-vm-026:9006/v1).
        model: Model name/identifier to use.
        api_key: API key for authentication. None disables api key authentication.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        is_function_calling_model: Whether model supports function/tool calling.
        context_window: Model context window size (for token management).
        additional_kwargs: Extra OpenAI parameters (top_p, presence_penalty, etc.).
    """

    api_base: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: float = 60.0
    max_retries: int = 3
    is_function_calling_model: bool = True
    context_window: int | None = None
    additional_kwargs: dict[str, Any] | _KwargsTuple | None = None

    def __post_init__(self) -> None:
        """Convert dict additional_kwargs to tuple for hashability."""
        if isinstance(self.additional_kwargs, dict):
            object.__setattr__(
                self,
                "additional_kwargs",
                tuple(sorted(self.additional_kwargs.items())),
            )

    def get_additional_kwargs_dict(self) -> dict[str, Any]:
        """Get additional_kwargs as a dict (for passing to OpenAILike)."""
        if self.additional_kwargs is None:
            return {}
        return dict(self.additional_kwargs)
