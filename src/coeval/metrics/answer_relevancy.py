"""
Answer Relevancy metric - re-exported from DeepEval for convenience.

This module provides a simple re-export of DeepEval's AnswerRelevancyMetric.

For direct usage:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models import GPTModel

    judge = GPTModel(model="gpt-4.1", base_url="https://api.openai.com/v1")
    metric = AnswerRelevancyMetric(model=judge, threshold=0.7)
"""

# Re-export for backwards compatibility and convenience
from deepeval.metrics import AnswerRelevancyMetric

__all__ = ["AnswerRelevancyMetric"]
