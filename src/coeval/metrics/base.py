"""
Base metric classes using DeepEval's BaseMetric.

- DeterministicMetric: Rule-based, no LLM required (e.g., MCQ accuracy, ROUGE)
- For LLM-as-judge metrics, use DeepEval metrics directly with GPTModel
"""

from abc import abstractmethod
from typing import Any

from coeval.core.types import BaseMetric, LLMTestCase


class DeterministicMetric(BaseMetric):
    """
    Base class for deterministic (non-LLM) evaluation metrics.

    Use for rule-based evaluations where:
    - Same input always produces same output
    - No LLM inference required
    - Fast, reproducible results

    Examples: MCQ accuracy, ROUGE score, exact match, BLEU

    Subclasses must implement:
        - __name__: Metric identifier
        - measure(): Synchronous evaluation logic
    """

    def __init__(
        self,
        threshold: float = 0.5,
        strict_mode: bool = False,
        **kwargs: Any,
    ):
        self.threshold = threshold
        self.strict_mode = strict_mode
        self._kwargs = kwargs

        # DeepEval state attributes
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None

    @property
    @abstractmethod
    def __name__(self) -> str:
        """Metric name identifier."""
        ...

    @abstractmethod
    def measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Synchronous evaluation.

        Must set: self.score, self.success
        Optionally set: self.reason

        Args:
            test_case: DeepEval test case with input/output

        Returns:
            Evaluation score
        """
        ...

    def _set_result(
        self,
        test_case: LLMTestCase,
        is_correct: bool,
        reason: str,
        details: dict[str, Any],
    ) -> float:
        """Set score, success, reason, and attach details to test_case metadata."""
        self.score = 1.0 if is_correct else 0.0
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0
        self.success = self.score >= self.threshold
        self.reason = reason
        self._details = details
        if test_case.additional_metadata is None:
            test_case.additional_metadata = {}
        test_case.additional_metadata[self.__name__] = self._details
        return self.score

    def _fail(self, reason: str) -> float:
        """Return 0.0 with a failure reason (no extraction / parse error)."""
        self.score = 0.0
        self.success = False
        self.reason = reason
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Async evaluation - delegates to sync measure.

        Deterministic metrics have no I/O, so async just wraps sync.
        """
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Check if the metric evaluation was successful."""
        if self.error is not None:
            self.success = False
        return self.success or False
