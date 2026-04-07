"""
Numeric accuracy metric for clinical calculation benchmarks.

Extracts numeric value from LLM response and checks if it falls within
the tolerance range [lower_limit, upper_limit].
"""

from __future__ import annotations

import re
from typing import Any

from coeval.core.types import LLMTestCase
from coeval.metrics.base import DeterministicMetric

# Patterns to extract numeric values from LLM output
_NUMERIC_PATTERNS = [
    # "The answer is 25.24" or "= 25.24"
    re.compile(r"(?:answer|result|value)\s*(?:is|=|:)\s*(-?\d+\.?\d*)", re.IGNORECASE),
    # "25.24 mL/min" or "25.24 kg/m^2"
    re.compile(
        r"(-?\d+\.?\d*)\s*(?:mL|mg|kg|mmol|mm|cm|%|g/dL|IU|beats|breaths|mmHg|mEq)",
        re.IGNORECASE,
    ),
    # "CrCl = 25.24" or "BMI = 18.5"
    re.compile(r"=\s*(-?\d+\.?\d*)\s*(?:mL|mg|kg|$|\s)", re.IGNORECASE),
    # Last standalone number in the response
    re.compile(r"(-?\d+\.?\d*)\s*$", re.MULTILINE),
]


def extract_numeric_answer(text: str) -> float | None:
    """Extract the most likely numeric answer from LLM output."""
    if not text or not text.strip():
        return None

    # Try each pattern in priority order
    for pattern in _NUMERIC_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            # Take the last match (most likely the final answer)
            try:
                return float(matches[-1].group(1))
            except (ValueError, IndexError):
                continue

    # Fallback: find any number in the text
    all_numbers = re.findall(r"-?\d+\.?\d*", text)
    if all_numbers:
        try:
            return float(all_numbers[-1])
        except ValueError:
            pass

    return None


class NumericAccuracyMetric(DeterministicMetric):
    """
    Numeric accuracy metric for clinical calculations.

    Checks if the extracted numeric answer falls within the tolerance range
    [lower_limit, upper_limit] from the dataset metadata.

    If no range is provided, uses exact match with relative tolerance.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        default_tolerance: float = 0.05,
        strict_mode: bool = False,
        **kwargs: Any,
    ):
        """Initialize with default relative tolerance for numeric comparison."""
        super().__init__(
            threshold=threshold,
            strict_mode=strict_mode,
            **kwargs,
        )
        self.default_tolerance = default_tolerance

    # Date patterns: MM/DD/YYYY, YYYY-MM-DD, etc.
    _DATE_PATTERN = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b")

    @property
    def __name__(self) -> str:
        return "Numeric Accuracy"

    def _measure_date(self, test_case: LLMTestCase) -> float:
        """Exact date string match for date-type outputs."""
        expected = (test_case.expected_output or "").strip()
        actual = test_case.actual_output or ""

        matches = self._DATE_PATTERN.findall(actual)
        predicted = matches[-1] if matches else None
        is_correct = predicted == expected if predicted else False

        return self._set_result(
            test_case,
            is_correct,
            f"Predicted '{predicted}', expected '{expected}' (date)",
            {"is_correct": is_correct, "predicted": predicted, "expected": expected},
        )

    def _check_range(
        self, predicted: float, expected: float, metadata: dict[str, Any]
    ) -> tuple[bool, str]:
        """Check if predicted value is within range or relative tolerance."""
        lower = metadata.get("lower_limit")
        upper = metadata.get("upper_limit")

        if lower is not None and upper is not None:
            try:
                lower_f, upper_f = float(lower), float(upper)
                return (
                    lower_f <= predicted <= upper_f,
                    f"Predicted {predicted:.4f}, range [{lower_f:.4f}, {upper_f:.4f}], expected {expected:.4f}",
                )
            except (ValueError, TypeError):
                pass

        rel_diff = abs(predicted - expected) / max(abs(expected), 1e-10)
        return (
            rel_diff <= self.default_tolerance,
            f"Predicted {predicted:.4f}, expected {expected:.4f}, rel_diff {rel_diff:.4f}",
        )

    def measure(
        self,
        test_case: LLMTestCase,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> float:
        try:
            metadata = test_case.additional_metadata or {}

            if metadata.get("output_type", "decimal") == "date":
                return self._measure_date(test_case)

            predicted = extract_numeric_answer(test_case.actual_output or "")
            expected = extract_numeric_answer(test_case.expected_output or "")

            if predicted is None:
                return self._fail("Could not extract numeric value from response")
            if expected is None:
                return self._fail("Could not parse expected answer")

            is_correct, reason = self._check_range(predicted, expected, metadata)

            return self._set_result(
                test_case,
                is_correct,
                reason,
                {
                    "is_correct": is_correct,
                    "predicted": predicted,
                    "expected": expected,
                },
            )

        except Exception as e:
            self.error = str(e)
            self.reason = f"Evaluation failed: {e}"
            raise
