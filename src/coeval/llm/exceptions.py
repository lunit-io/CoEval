"""Exception hierarchy for LLM operations."""


class LLMError(Exception):
    """Base exception for LLM operations."""

    pass


class LLMConnectionError(LLMError):
    """Connection to LLM service failed."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    pass


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM service."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    pass
