"""
Schema definitions for the CoEval Evaluation Framework.

All models in one place for improved visibility.
"""

from typing import Any

from pydantic import BaseModel, Field, computed_field

from coeval.core.types import ConversationalTestCase

# NOTE: EvalResult.test_case uses ``Any`` instead of ``TestCase`` to avoid
# Pydantic v2 union-validation errors (ConversationalTestCase has no .get()).

__all__ = [
    "AnswerResponse",
    "MCQResponse",
    "MetricResult",
    "MetricScoreDetail",
    "EvalResult",
    "EvalSummary",
    "ParsedResponse",
    "ClassificationResult",
]


class AnswerResponse(BaseModel):
    """
    Standard structured response for LLM outputs.

    Used with vLLM structured_outputs to enforce JSON format.
    """

    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning process before arriving at the answer.",
    )
    answer: str = Field(
        description="The final answer to the question.",
    )


class MCQResponse(BaseModel):
    """
    Structured response for multiple-choice questions.

    Guides the LLM to output a single letter answer with reasoning.
    """

    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning process for selecting the answer.",
    )
    answer: str = Field(
        description="The answer letter (e.g., 'A', 'B', 'C', 'D', or 'yes', 'no', 'maybe').",
    )


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    name: str
    score: float | None  # None = judge infrastructure failure, excluded from aggregate
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result for a single evaluation sample."""

    model_config = {"arbitrary_types_allowed": True}

    sample_id: int
    test_case: Any
    rationale: str = ""
    metrics: list[MetricResult]
    generation_time_ms: float
    scoring_time_ms: float
    inference_failed: bool = False
    inference_error: str | None = None
    scoring_failed: bool = False

    @computed_field
    @property
    def passed(self) -> bool:
        """Whether all metrics passed (False if inference or scoring failed)."""
        if self.inference_failed or self.scoring_failed:
            return False
        return all(m.passed for m in self.metrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        tc = self.test_case
        if isinstance(tc, ConversationalTestCase):
            input_text = "\n\n".join(f"{t.role}: {t.content}" for t in tc.turns[:-1])
            actual_output = tc.turns[-1].content if tc.turns else ""
            expected_output = None
        else:
            input_text = tc.input
            actual_output = tc.actual_output
            expected_output = tc.expected_output

        result = {
            "sample_id": self.sample_id,
            "input": input_text,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "rationale": self.rationale,
            "passed": self.passed,
            "inference_failed": self.inference_failed,
            "inference_error": self.inference_error,
            "scoring_failed": self.scoring_failed,
            "generation_time_ms": self.generation_time_ms,
            "scoring_time_ms": self.scoring_time_ms,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.passed,
                    "reason": m.reason,
                    **({"details": m.details} if m.details else {}),
                }
                for m in self.metrics
            ],
        }
        return result


class MetricScoreDetail(BaseModel):
    """
    Detailed metric score with optional numerator/denominator for proper aggregation.

    For weighted metrics (e.g., rubric-based), stores raw values to enable
    correct cross-dataset merging: overall = sum(numerator) / sum(denominator)

    For simple average metrics, only score is set.
    """

    score: float
    numerator: float | None = None
    denominator: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"score": self.score}
        if self.numerator is not None:
            result["numerator"] = self.numerator
        if self.denominator is not None:
            result["denominator"] = self.denominator
        return result


class EvalSummary(BaseModel):
    """Summary statistics for an evaluation run."""

    dataset: str
    num_samples: int = Field(gt=0)
    num_passed: int
    total_time_s: float
    avg_generation_ms: float
    avg_scoring_ms: float
    metric_scores: dict[str, MetricScoreDetail]
    num_inference_failed: int = 0
    num_scoring_failed: int = 0
    breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @computed_field
    @property
    def num_evaluated(self) -> int:
        """Number of samples with completed inference and scoring."""
        return self.num_samples - self.num_inference_failed - self.num_scoring_failed

    @computed_field
    @property
    def pass_rate(self) -> float:
        """Pass rate among successfully evaluated samples."""
        return self.num_passed / self.num_evaluated if self.num_evaluated > 0 else 0.0

    @computed_field
    @property
    def inference_failure_rate(self) -> float:
        """Rate of inference failures."""
        return (
            self.num_inference_failed / self.num_samples
            if self.num_samples > 0
            else 0.0
        )

    @computed_field
    @property
    def scoring_failure_rate(self) -> float:
        """Rate of scoring infrastructure failures."""
        return (
            self.num_scoring_failed / self.num_samples if self.num_samples > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "num_evaluated": self.num_evaluated,
            "num_passed": self.num_passed,
            "num_inference_failed": self.num_inference_failed,
            "num_scoring_failed": self.num_scoring_failed,
            "pass_rate": self.pass_rate,
            "inference_failure_rate": self.inference_failure_rate,
            "scoring_failure_rate": self.scoring_failure_rate,
            "total_time_s": self.total_time_s,
            "avg_generation_ms": self.avg_generation_ms,
            "avg_scoring_ms": self.avg_scoring_ms,
            "metric_scores": {k: v.to_dict() for k, v in self.metric_scores.items()},
            "breakdown": self.breakdown,
        }


class ParsedResponse(BaseModel):
    """Parsed LLM response with optional reasoning and answer fields."""

    answer: str | None = None
    reasoning: str | None = None
    raw: str = ""
    parse_method: str = "unknown"
    error: str | None = None


class ClassificationResult(BaseModel):
    """
    Unified classification result for MCQ and general classification tasks.

    Works for binary, multi-class, and MCQ (which is just multi-class with letter options).
    """

    is_correct: bool
    method: str
    predicted_label: str | None = None
    expected_label: str | None = None
