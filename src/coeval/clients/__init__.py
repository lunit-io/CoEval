"""Inference clients for evaluation.

Available clients:
- PassthroughClient: Direct LLM calls via LLMClient (baseline - no retrieval)
"""

from coeval.clients.passthrough import PassthroughClient
from coeval.clients.passthrough_judge import PassthroughJudge

__all__ = [
    "PassthroughClient",
    "PassthroughJudge",
]
