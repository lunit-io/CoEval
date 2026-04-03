"""Core evaluation components."""

from coeval.core.registry import (
    clear_registries,
    get_dataset,
    get_dataset_class,
    get_metric,
    get_metric_class,
    list_datasets,
    list_metrics,
    register_dataset,
    register_metric,
)
from coeval.core.runner import EvalRunner
from coeval.core.schema import (
    AnswerResponse,
    EvalResult,
    EvalSummary,
    MCQResponse,
    MetricResult,
    ParsedResponse,
)
from coeval.core.types import (
    AnyGolden,
    BaseConversationalMetric,
    BaseMetric,
    ChatMessages,
    ConversationalGolden,
    ConversationalTestCase,
    EvaluationDataset,
    Golden,
    InferenceClient,
    LLMTestCase,
    TestCase,
    Turn,
)

__all__ = [
    # deepeval re-exports (via types)
    "ConversationalGolden",
    "ConversationalTestCase",
    "EvaluationDataset",
    "Golden",
    "LLMTestCase",
    "Turn",
    # Type aliases
    "AnyGolden",
    "ChatMessages",
    "TestCase",
    # Client protocols
    "InferenceClient",
    # Metric bases
    "BaseConversationalMetric",
    "BaseMetric",
    # Schema
    "AnswerResponse",
    "EvalResult",
    "EvalRunner",
    "EvalSummary",
    "MCQResponse",
    "MetricResult",
    "ParsedResponse",
    # Registry
    "clear_registries",
    "get_dataset",
    "get_dataset_class",
    "get_metric",
    "get_metric_class",
    "list_datasets",
    "list_metrics",
    "register_dataset",
    "register_metric",
]
