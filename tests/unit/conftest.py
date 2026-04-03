"""Pytest configuration for coeval tests."""

from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring LLM API calls (deselect with '-m \"not integration\"')",
    )


@pytest.fixture(autouse=True)
def _mock_deepeval_model():
    """Prevent GEval from requiring an OpenAI API key in unit tests."""
    with patch(
        "deepeval.metrics.g_eval.g_eval.initialize_model",
        return_value=(MagicMock(), True),
    ):
        yield
