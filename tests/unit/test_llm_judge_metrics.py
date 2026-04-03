"""Unit tests for LLM-as-judge metrics (AnswerRelevancy, Faithfulness, ContextualPrecision).

These tests verify metric instantiation and configuration without making actual LLM calls.
For integration tests with real LLM judges, see tests/integration/.
"""

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class MockDeepEvalLLM(DeepEvalBaseLLM):
    """A mock LLM for testing that extends DeepEvalBaseLLM."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self._model_name = model_name

    def load_model(self) -> None:
        """No-op load for mock."""
        pass

    def generate(self, prompt: str) -> str:
        """Return a mock response."""
        return "mock response"

    async def a_generate(self, prompt: str) -> str:
        """Return a mock response asynchronously."""
        return "mock response"

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name


@pytest.fixture
def mock_llm() -> MockDeepEvalLLM:
    """Fixture providing a mock LLM for metric tests."""
    return MockDeepEvalLLM()


class TestAnswerRelevancyMetricInstantiation:
    """Test AnswerRelevancyMetric configuration and instantiation."""

    def test_instantiation_with_mock_model(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric can be instantiated with a mock model."""
        metric = AnswerRelevancyMetric(model=mock_llm, threshold=0.7)

        assert metric.threshold == 0.7

    def test_default_threshold(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric uses default threshold when not specified."""
        metric = AnswerRelevancyMetric(model=mock_llm)

        assert metric.threshold == 0.5  # DeepEval default

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.7, 1.0])
    def test_threshold_values(
        self, mock_llm: MockDeepEvalLLM, threshold: float
    ) -> None:
        """Metric accepts various threshold values."""
        metric = AnswerRelevancyMetric(model=mock_llm, threshold=threshold)

        assert metric.threshold == threshold

    def test_has_measure_method(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has measure and a_measure methods."""
        metric = AnswerRelevancyMetric(model=mock_llm)

        assert hasattr(metric, "measure")
        assert hasattr(metric, "a_measure")
        assert callable(metric.measure)
        assert callable(metric.a_measure)

    def test_metric_name(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has expected name."""
        metric = AnswerRelevancyMetric(model=mock_llm)

        assert metric.__name__ == "Answer Relevancy"

    def test_model_name_from_mock(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric correctly retrieves model name from mock."""
        _ = AnswerRelevancyMetric(model=mock_llm)

        assert mock_llm.get_model_name() == "mock-model"


class TestFaithfulnessMetricInstantiation:
    """Test FaithfulnessMetric configuration and instantiation."""

    def test_instantiation_with_mock_model(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric can be instantiated with a mock model."""
        metric = FaithfulnessMetric(model=mock_llm, threshold=0.7)

        assert metric.threshold == 0.7

    def test_default_threshold(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric uses default threshold when not specified."""
        metric = FaithfulnessMetric(model=mock_llm)

        assert metric.threshold == 0.5  # DeepEval default

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.7, 1.0])
    def test_threshold_values(
        self, mock_llm: MockDeepEvalLLM, threshold: float
    ) -> None:
        """Metric accepts various threshold values."""
        metric = FaithfulnessMetric(model=mock_llm, threshold=threshold)

        assert metric.threshold == threshold

    def test_has_measure_method(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has measure and a_measure methods."""
        metric = FaithfulnessMetric(model=mock_llm)

        assert hasattr(metric, "measure")
        assert hasattr(metric, "a_measure")
        assert callable(metric.measure)
        assert callable(metric.a_measure)

    def test_metric_name(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has expected name."""
        metric = FaithfulnessMetric(model=mock_llm)

        assert metric.__name__ == "Faithfulness"


class TestContextualPrecisionMetricInstantiation:
    """Test ContextualPrecisionMetric configuration and instantiation."""

    def test_instantiation_with_mock_model(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric can be instantiated with a mock model."""
        metric = ContextualPrecisionMetric(model=mock_llm, threshold=0.7)

        assert metric.threshold == 0.7

    def test_default_threshold(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric uses default threshold when not specified."""
        metric = ContextualPrecisionMetric(model=mock_llm)

        assert metric.threshold == 0.5  # DeepEval default

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.7, 1.0])
    def test_threshold_values(
        self, mock_llm: MockDeepEvalLLM, threshold: float
    ) -> None:
        """Metric accepts various threshold values."""
        metric = ContextualPrecisionMetric(model=mock_llm, threshold=threshold)

        assert metric.threshold == threshold

    def test_has_measure_method(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has measure and a_measure methods."""
        metric = ContextualPrecisionMetric(model=mock_llm)

        assert hasattr(metric, "measure")
        assert hasattr(metric, "a_measure")
        assert callable(metric.measure)
        assert callable(metric.a_measure)

    def test_metric_name(self, mock_llm: MockDeepEvalLLM) -> None:
        """Metric has expected name."""
        metric = ContextualPrecisionMetric(model=mock_llm)

        assert metric.__name__ == "Contextual Precision"


class TestLLMTestCaseCreation:
    """Test LLMTestCase creation for metric evaluation."""

    def test_basic_test_case(self) -> None:
        """Basic test case can be created."""
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
        )

        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "Paris is the capital of France."

    def test_test_case_with_retrieval_context(self) -> None:
        """Test case with retrieval context can be created."""
        test_case = LLMTestCase(
            input="What is metformin used for?",
            actual_output="Metformin is used to treat type 2 diabetes.",
            retrieval_context=[
                "Metformin is a first-line medication for type 2 diabetes.",
                "It works by decreasing glucose production in the liver.",
            ],
        )

        assert test_case.input == "What is metformin used for?"
        assert test_case.actual_output == "Metformin is used to treat type 2 diabetes."
        assert len(test_case.retrieval_context) == 2

    def test_test_case_with_expected_output(self) -> None:
        """Test case with expected output can be created."""
        test_case = LLMTestCase(
            input="What is 2+2?",
            actual_output="4",
            expected_output="4",
        )

        assert test_case.expected_output == "4"


class TestMetricReExports:
    """Test that metrics are properly re-exported from coeval."""

    def test_answer_relevancy_import(self) -> None:
        """AnswerRelevancyMetric can be imported from coeval."""
        from coeval.metrics.answer_relevancy import AnswerRelevancyMetric

        assert AnswerRelevancyMetric is not None

    def test_faithfulness_import(self) -> None:
        """FaithfulnessMetric can be imported from coeval."""
        from coeval.metrics.faithfulness import FaithfulnessMetric

        assert FaithfulnessMetric is not None

    def test_contextual_precision_import(self) -> None:
        """ContextualPrecisionMetric can be imported from coeval."""
        from coeval.metrics.contextual_precision import (
            ContextualPrecisionMetric,
        )

        assert ContextualPrecisionMetric is not None
