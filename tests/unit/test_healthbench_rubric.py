"""Tests for HealthBench rubric grading metric."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepeval.test_case import ConversationalTestCase, Turn

from coeval.metrics.healthbench_rubric import (
    GRADER_TEMPLATE,
    HealthBenchRubricMetric,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_judge() -> MagicMock:
    """Create a mock judge (DeepEvalBaseLLM-compatible)."""
    client = MagicMock()
    client.a_generate = AsyncMock()
    return client


def _make_test_case(
    response: str = "You should rest and take ibuprofen.",
    rubrics: list[dict] | None = None,
    user_input: str = "I have a headache. What should I do?",
) -> ConversationalTestCase:
    """Create a ConversationalTestCase with HealthBench metadata."""
    if rubrics is None:
        rubrics = [
            {"criterion": "Recommends rest", "points": 1.0, "tags": ["advice"]},
            {"criterion": "Mentions medication", "points": 2.0, "tags": ["treatment"]},
        ]

    return ConversationalTestCase(
        turns=[
            Turn(role="user", content=user_input),
            Turn(role="assistant", content=response),
        ],
        additional_metadata={
            "rubrics": rubrics,
            "example_tags": ["general_health"],
            "prompt_id": "test-001",
        },
    )


# ---------------------------------------------------------------------------
# Grader template tests
# ---------------------------------------------------------------------------


class TestGraderTemplate:
    def test_template_has_placeholders(self) -> None:
        assert "<<conversation>>" in GRADER_TEMPLATE
        assert "<<rubric_item>>" in GRADER_TEMPLATE

    def test_template_mentions_criteria_met(self) -> None:
        assert "criteria_met" in GRADER_TEMPLATE

    def test_template_mentions_explanation(self) -> None:
        assert "explanation" in GRADER_TEMPLATE

    def test_template_matches_official(self) -> None:
        """Template should contain key phrases from official HealthBench eval."""
        assert "last turn" in GRADER_TEMPLATE
        assert "rubric item" in GRADER_TEMPLATE
        assert "negative" in GRADER_TEMPLATE.lower()


# ---------------------------------------------------------------------------
# Metric initialization tests
# ---------------------------------------------------------------------------


class TestHealthBenchRubricMetricInit:
    def test_metric_name(self) -> None:
        client = _make_judge()
        metric = HealthBenchRubricMetric(judge=client)
        assert metric.__name__ == "HealthBench Rubric"

    def test_default_concurrent_limit(self) -> None:
        client = _make_judge()
        metric = HealthBenchRubricMetric(judge=client)
        assert metric.concurrent_limit == 10

    def test_custom_concurrent_limit(self) -> None:
        client = _make_judge()
        metric = HealthBenchRubricMetric(judge=client, concurrent_limit=5)
        assert metric.concurrent_limit == 5


# ---------------------------------------------------------------------------
# Async measure tests
# ---------------------------------------------------------------------------


class TestHealthBenchRubricMeasure:
    @pytest.mark.asyncio
    async def test_perfect_score(self) -> None:
        """All criteria met -> score = 1.0."""
        client = _make_judge()
        client.a_generate.return_value = json.dumps(
            {"explanation": "ok", "criteria_met": True}
        )
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case()

        score = await metric.a_measure(test_case)

        assert score == pytest.approx(1.0)
        assert metric.success is True
        assert metric._details["num_criteria_met"] == 2
        assert metric._details["num_criteria_total"] == 2

    @pytest.mark.asyncio
    async def test_zero_score(self) -> None:
        """No criteria met -> score = 0.0."""
        client = _make_judge()
        client.a_generate.return_value = json.dumps(
            {"explanation": "bad", "criteria_met": False}
        )
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case()

        score = await metric.a_measure(test_case)

        assert score == pytest.approx(0.0)
        assert metric.success is False
        assert metric._details["num_criteria_met"] == 0

    @pytest.mark.asyncio
    async def test_partial_score(self) -> None:
        """Some criteria met -> partial score."""
        client = _make_judge()
        # First rubric met (1pt), second not met (2pt)
        responses = [
            json.dumps({"explanation": "yes", "criteria_met": True}),
            json.dumps({"explanation": "no", "criteria_met": False}),
        ]
        client.a_generate.side_effect = responses
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case()

        score = await metric.a_measure(test_case)

        # achieved=1.0, total=3.0 -> 1/3
        assert score == pytest.approx(1 / 3)
        assert metric._details["achieved_points"] == 1.0
        assert metric._details["total_possible_points"] == 3.0

    @pytest.mark.asyncio
    async def test_no_rubrics_returns_zero(self) -> None:
        """No rubrics in metadata -> score = 0.0."""
        client = _make_judge()
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(rubrics=[])

        score = await metric.a_measure(test_case)

        assert score == 0.0
        assert metric.success is False
        assert metric._details.get("error") == "no_rubrics"

    @pytest.mark.asyncio
    async def test_negative_points_rubric(self) -> None:
        """Negative points for undesirable criteria."""
        client = _make_judge()
        # Good criterion met, bad criterion NOT met (good!)
        responses = [
            json.dumps({"explanation": "accurate", "criteria_met": True}),
            json.dumps({"explanation": "not verbose", "criteria_met": False}),
        ]
        client.a_generate.side_effect = responses
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(
            rubrics=[
                {"criterion": "Accurate info", "points": 2.0, "tags": []},
                {"criterion": "Is verbose", "points": -1.0, "tags": []},
            ]
        )

        score = await metric.a_measure(test_case)
        # achieved=2.0, total=2.0 -> 1.0
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_retry_on_bad_json(self) -> None:
        """Should retry when grader returns invalid JSON."""
        client = _make_judge()
        client.a_generate.side_effect = [
            "not valid json",
            json.dumps({"explanation": "ok", "criteria_met": True}),
            json.dumps({"explanation": "ok", "criteria_met": True}),
        ]
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(
            rubrics=[{"criterion": "c1", "points": 1.0, "tags": []}]
        )

        score = await metric.a_measure(test_case)
        assert score == pytest.approx(1.0)
        # Called twice for one rubric item (1 retry + 1 success)
        assert client.a_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_after_max_retries(self) -> None:
        """After max retries, criteria_met defaults to False."""
        client = _make_judge()
        client.a_generate.return_value = "always bad json"
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(
            rubrics=[{"criterion": "c1", "points": 1.0, "tags": []}]
        )

        score = await metric.a_measure(test_case)
        # Falls back to criteria_met=False -> 0/1 = 0.0
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_details_contain_rubric_grades(self) -> None:
        """_details should contain per-rubric grading info."""
        client = _make_judge()
        client.a_generate.return_value = json.dumps(
            {"explanation": "good explanation", "criteria_met": True}
        )
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case()

        await metric.a_measure(test_case)

        assert "rubric_grades" in metric._details
        grades = metric._details["rubric_grades"]
        assert len(grades) == 2
        assert grades[0]["criterion"] == "Recommends rest"
        assert grades[0]["criteria_met"] is True
        assert grades[0]["explanation"] == "good explanation"

    @pytest.mark.asyncio
    async def test_grader_receives_conversation_with_response(self) -> None:
        """Grader prompt should include the conversation + model response."""
        client = _make_judge()
        client.a_generate.return_value = json.dumps(
            {"explanation": "ok", "criteria_met": True}
        )
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(
            response="Take some rest.",
            rubrics=[{"criterion": "c1", "points": 1.0, "tags": []}],
            user_input="I have a headache",
        )

        await metric.a_measure(test_case)

        # Check that the grader was called with the conversation in the prompt
        call_args = client.a_generate.call_args
        grader_prompt = call_args[0][0]  # first positional arg (string)
        assert "I have a headache" in grader_prompt
        assert "Take some rest." in grader_prompt
        assert "assistant:" in grader_prompt
        assert call_args[1]["system_prompt"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_example_tags_added_to_details(self) -> None:
        """example_tags should be extracted from metadata to details."""
        client = _make_judge()
        client.a_generate.return_value = json.dumps(
            {"explanation": "ok", "criteria_met": True}
        )
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case()

        await metric.a_measure(test_case)

        assert "example_tags" in metric._details
        assert metric._details["example_tags"] == ["general_health"]

    @pytest.mark.asyncio
    async def test_cluster_scores_calculated(self) -> None:
        """Cluster tags should be used to calculate localized subset scores."""
        client = _make_judge()
        # Item 1: passed (1 point)
        # Item 2: failed (2 points) -> Total subset score: 1/3
        responses = [
            json.dumps({"explanation": "yes", "criteria_met": True}),
            json.dumps({"explanation": "no", "criteria_met": False}),
        ]
        client.a_generate.side_effect = responses
        metric = HealthBenchRubricMetric(judge=client)
        test_case = _make_test_case(
            rubrics=[
                {"criterion": "A", "points": 1.0, "tags": ["cluster:test_cluster"]},
                {"criterion": "B", "points": 2.0, "tags": ["cluster:test_cluster"]},
            ]
        )

        await metric.a_measure(test_case)

        assert "cluster_scores" in metric._details
        assert "cluster:test_cluster" in metric._details["cluster_scores"]
        assert metric._details["cluster_scores"][
            "cluster:test_cluster"
        ] == pytest.approx(1 / 3)
