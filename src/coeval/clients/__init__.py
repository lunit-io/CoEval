"""Inference clients for evaluation.

Available clients:
- EvaluationLLMClient: base for online LLM-only clients; wraps an OpenAILike as self.llm
- PassthroughClient: direct LLM calls, no retrieval (baseline)
- PassthroughJudge: DeepEvalBaseLLM judge backed by an OpenAILike
"""

from coeval.clients.base import EvaluationLLMClient
from coeval.clients.passthrough import PassthroughClient
from coeval.clients.passthrough_judge import PassthroughJudge

__all__ = [
    "EvaluationLLMClient",
    "PassthroughClient",
    "PassthroughJudge",
]
