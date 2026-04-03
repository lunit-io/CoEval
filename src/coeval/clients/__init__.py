"""Inference clients for evaluation.

Available clients:
- PassthroughClient: Direct LLM calls via LLMClient (baseline - no retrieval)
"""

from coeval.clients.passthrough import PassthroughClient

__all__ = [
    "PassthroughClient",
]
