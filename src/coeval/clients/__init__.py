"""Inference clients for evaluation.

Available clients:
- PassthroughClient: Direct LLM calls via LLMClient (baseline - no retrieval)
- PassthroughJudge: DeepEvalBaseLLM judge backed by PassthroughClient (HTTP)
- CodexExecJudge: DeepEvalBaseLLM judge backed by the `codex exec` CLI
"""

from coeval.clients.codex_exec_judge import CodexExecJudge
from coeval.clients.passthrough import PassthroughClient
from coeval.clients.passthrough_judge import PassthroughJudge

__all__ = [
    "CodexExecJudge",
    "PassthroughClient",
    "PassthroughJudge",
]
