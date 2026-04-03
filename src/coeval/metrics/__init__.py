"""
Metrics module for evaluation.

Provides:
- DeterministicMetric: Base class for rule-based metrics (MCQ accuracy, etc.)
- HealthBenchRubricMetric: LLM-as-judge rubric scoring
- DeepEval wrappers: Faithfulness, AnswerRelevancy, ContextualPrecision, ContextualRecall
"""

from coeval.metrics.answer_relevancy import AnswerRelevancyMetric
from coeval.metrics.base import DeterministicMetric
from coeval.metrics.classification import ClassificationMetric
from coeval.metrics.contextual_precision import ContextualPrecisionMetric
from coeval.metrics.contextual_recall import ContextualRecallMetric
from coeval.metrics.faithfulness import FaithfulnessMetric
from coeval.metrics.healthbench_rubric import HealthBenchRubricMetric
from coeval.metrics.mcq_accuracy import MCQAccuracyMetric
from coeval.metrics.numeric_accuracy import NumericAccuracyMetric

__all__ = [
    "AnswerRelevancyMetric",
    "ClassificationMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "DeterministicMetric",
    "FaithfulnessMetric",
    "HealthBenchRubricMetric",
    "MCQAccuracyMetric",
    "NumericAccuracyMetric",
]
