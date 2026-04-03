"""
Contextual Precision metric - re-exported from DeepEval for convenience.

This module provides a simple re-export of DeepEval's ContextualPrecisionMetric.

For direct usage:
    from deepeval.metrics import ContextualPrecisionMetric
    from deepeval.models import GPTModel

    judge = GPTModel(model="gpt-4.1", base_url="https://api.openai.com/v1")
    metric = ContextualPrecisionMetric(model=judge, threshold=0.7)
"""

# Re-export for backwards compatibility and convenience
from deepeval.metrics import ContextualPrecisionMetric

__all__ = ["ContextualPrecisionMetric"]
