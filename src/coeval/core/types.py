"""Shared type aliases for the CoEval Evaluation Framework.

Centralizes deepeval re-exports and union types used across core modules
(evaluate, runner, schema) and dataset/metric classes to avoid scattered
``from deepeval.*`` imports throughout the codebase.
"""

from typing import Any, Protocol, runtime_checkable

from deepeval.dataset import ConversationalGolden, EvaluationDataset, Golden
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn

# ---------------------------------------------------------------------------
# Union aliases
# ---------------------------------------------------------------------------

TestCase = LLMTestCase | ConversationalTestCase

AnyGolden = Golden | ConversationalGolden

ChatMessages = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceClient(Protocol):
    """Protocol for async inference clients (PassthroughClient, etc.)."""

    @property
    def name(self) -> str: ...

    async def generate(self, messages: ChatMessages) -> str: ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # deepeval re-exports — datasets
    "ConversationalGolden",
    "EvaluationDataset",
    "Golden",
    # deepeval re-exports — test cases
    "ConversationalTestCase",
    "LLMTestCase",
    "Turn",
    # deepeval re-exports — metrics
    "BaseConversationalMetric",
    "BaseMetric",
    # Union aliases
    "AnyGolden",
    "ChatMessages",
    "TestCase",
    # Protocols
    "InferenceClient",
]
