"""
CoEval — Medical LLM Evaluation Framework

A simple, extensible evaluation framework for medical LLM benchmarks.

Usage (CLI):
    mise run eval
    mise run eval -- datasets=all num_samples=5

Usage (Python):
    from coeval import EvalRunner
    from coeval.clients import PassthroughClient

    client = PassthroughClient(api_base="http://localhost:8000/v1", model="your-model")
    runner = EvalRunner(client=client)
    summary = await runner.run(dataset, metrics, aggregator)
"""

__version__ = "0.1.0"

from coeval.core import (
    AnswerResponse,
    EvalResult,
    EvalRunner,
    EvalSummary,
    MCQResponse,
    MetricResult,
    ParsedResponse,
)
from coeval.util import (
    JSONParser,
    console,
    parse_answer,
    parse_json_response,
    parse_mcq_answer,
)

__all__ = [
    "__version__",
    "AnswerResponse",
    "EvalResult",
    "EvalRunner",
    "EvalSummary",
    "JSONParser",
    "MCQResponse",
    "MetricResult",
    "ParsedResponse",
    "console",
    "parse_answer",
    "parse_json_response",
    "parse_mcq_answer",
]
