"""
Classification metric for binary/multi-class tasks.

Per-sample score is 1.0 (correct) or 0.0 (incorrect).
Aggregate score (F1, accuracy, etc.) is determined by score_aggregator in dataset config.

For AttributionBench (F1): use f1_aggregator
For simple accuracy: use avg_aggregator (default)
"""

from __future__ import annotations

import json
import re
from typing import Any

from coeval.core.schema import ClassificationResult
from coeval.core.types import LLMTestCase
from coeval.metrics.base import DeterministicMetric

# Regex strategies applied in order. First match against valid labels wins.
# Generic patterns (model-agnostic)
_RE_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"([^"]+)"', re.IGNORECASE)
_RE_QUOTED = re.compile(r'"([^"]+)"')
_RE_FINAL_ANSWER = re.compile(
    r'(?:answer|conclusion|verdict)\s*(?:is|:)\s*["\']?([^"\'\n.]+)',
    re.IGNORECASE,
)
# Domain-specific patterns (ordinal / triage levels)
_RE_JSON_LEVEL = re.compile(r'"level"\s*:\s*(\d+)')
_RE_KTAS_LEVEL = re.compile(r"KTAS\s*(?:Level\s*)?(\d)", re.IGNORECASE)
_RE_LEVEL_NUMBER = re.compile(r"Level\s+(\d)\b", re.IGNORECASE)

# ADR causality patterns
_RE_JSON_CATEGORY = re.compile(r'"category"\s*:\s*"([^"]+)"')

EXTRACTION_STRATEGIES: dict[str, re.Pattern[str]] = {
    "json_answer": _RE_JSON_ANSWER,
    "json_level": _RE_JSON_LEVEL,
    "json_category": _RE_JSON_CATEGORY,
    "ktas_level": _RE_KTAS_LEVEL,
    "level_number": _RE_LEVEL_NUMBER,
    "quoted": _RE_QUOTED,
    "final_answer": _RE_FINAL_ANSWER,
}


def _normalize_label(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _try_parse_json(text: str) -> str | None:
    """Extract a label from well-formed JSON responses.

    Supports:
      - ``{"answer": "..."}``  (generic)
      - ``{"ktas": {"level": 3, ...}}``  (KTAS triage)
    """
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    if "answer" in data:
        return str(data["answer"]).strip()

    ktas = data.get("ktas")
    if isinstance(ktas, dict) and "level" in ktas:
        return str(ktas["level"]).strip()

    # ADR causality: {"causality_grading": {"category": "Possible", ...}}
    causality = data.get("causality_grading")
    if isinstance(causality, dict) and "category" in causality:
        return str(causality["category"]).strip()

    return None


def _extract_label(response: str, valid_labels: list[str]) -> tuple[str | None, str]:
    """Extract a classification label from LLM output using multiple strategies."""
    if not response:
        return None, "empty"

    text = response.strip()
    normalized_labels = {_normalize_label(lbl): lbl for lbl in valid_labels}

    json_label = _try_parse_json(text)
    if json_label:
        norm = _normalize_label(json_label)
        if norm in normalized_labels:
            return normalized_labels[norm], "json"

    for method, pattern in EXTRACTION_STRATEGIES.items():
        match = pattern.search(text)
        if match:
            candidate = _normalize_label(match.group(1))
            if candidate in normalized_labels:
                return normalized_labels[candidate], method

    text_normalized = _normalize_label(text)

    if text_normalized in normalized_labels:
        return normalized_labels[text_normalized], "exact"

    # Word-boundary search — pick the longest match to avoid
    # false positives (e.g. label "1" matching inside "196").
    boundary_hits = sorted(
        (
            (norm, orig)
            for norm, orig in normalized_labels.items()
            if re.search(r"\b" + re.escape(norm) + r"\b", text_normalized)
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    if boundary_hits:
        return boundary_hits[0][1], "direct"

    return None, "no_match"


def compute_classification_result(
    response: str,
    expected_label: str,
    valid_labels: list[str] | None = None,
) -> ClassificationResult:
    """
    Compute classification result from LLM response.

    Args:
        response: The LLM's response text.
        expected_label: The ground truth label.
        valid_labels: List of valid labels. If None, uses [expected_label].

    Returns:
        ClassificationResult with is_correct, method, predicted_label, expected_label.
    """
    labels = valid_labels or [expected_label]
    if expected_label not in labels:
        labels = [*labels, expected_label]

    predicted, method = _extract_label(response, labels)
    is_correct = predicted is not None and _normalize_label(
        predicted
    ) == _normalize_label(expected_label)

    return ClassificationResult(
        is_correct=is_correct,
        method=method,
        predicted_label=predicted,
        expected_label=expected_label,
    )


class ClassificationMetric(DeterministicMetric):
    """
    Classification metric for binary/multi-class tasks.

    Per-sample score is 1.0 (correct) or 0.0 (incorrect).
    Aggregate score (F1, accuracy, etc.) is determined by score_aggregator in dataset config.
    """

    def __init__(
        self,
        labels: list[str] | None = None,
        threshold: float = 1.0,
        strict_mode: bool = False,
        **kwargs: Any,
    ):
        """Initialize with an optional set of valid classification labels."""
        super().__init__(threshold=threshold, strict_mode=strict_mode, **kwargs)
        self.labels = labels or []

    @property
    def __name__(self) -> str:
        return "Classification"

    def measure(
        self,
        test_case: LLMTestCase,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> float:
        try:
            expected = test_case.expected_output or ""
            response = test_case.actual_output or ""
            valid_labels = self.labels or self._extract_labels_from_context(test_case)

            result = compute_classification_result(
                response=response,
                expected_label=expected,
                valid_labels=valid_labels or None,
            )

            reason = (
                f"Predicted '{result.predicted_label}' via {result.method}, "
                f"expected '{result.expected_label}'"
                if result.predicted_label
                else f"Could not extract label, expected '{result.expected_label}'"
            )

            return self._set_result(
                test_case,
                result.is_correct,
                reason,
                {
                    "is_correct": result.is_correct,
                    "method": result.method,
                    "predicted": result.predicted_label,
                    "expected": result.expected_label,
                },
            )

        except Exception as e:
            self.error = str(e)
            self.reason = f"Evaluation failed: {e}"
            raise

    def _extract_labels_from_context(self, test_case: LLMTestCase) -> list[str]:
        """Extract valid labels from test case context JSON."""
        if not test_case.context:
            return []

        for item in test_case.context:
            try:
                data = json.loads(item)
                if isinstance(data, dict) and "labels" in data:
                    return list(data["labels"])
            except (json.JSONDecodeError, TypeError):
                continue
        return []
