"""
HealthBench Rubric Grading Metric (LLM-as-Judge).

Grades model responses against rubric criteria using a grader LLM.
Each criterion is independently judged, then scores are aggregated
per the official HealthBench scoring: Σ(met × points) / Σ(max(0, points)).

Reference:
    - https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
"""

import asyncio
import json
import logging
import re
from typing import Any, Literal

from deepeval.models import DeepEvalBaseLLM

from coeval.core.types import BaseConversationalMetric, ConversationalTestCase

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3


def parse_grading_response(text: str) -> dict | None:
    """Parse grader JSON response, stripping markdown fences if present."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict) and "criteria_met" in result:
            label = result["criteria_met"]
            if label is True or label is False:
                return result
    except json.JSONDecodeError:
        pass
    return None


def calculate_rubric_score(rubric: list[dict], grading: list[dict]) -> float | None:
    """Score = Σ(met × points) / Σ(max(0, points)).

    Returns None if no positive-point claims exist (empty or all-penalty rubric).

    Note: `achieved` can be negative when negative-point (penalty) claims are met.
    This is intentional — it mirrors the HealthBench scoring formula where a model
    that asserts penalised claims ends up with a net-negative score.
    """
    total_possible = sum(r["points"] for r in rubric if r["points"] > 0)
    if total_possible == 0:
        return None
    achieved = sum(
        r["points"] for r, g in zip(rubric, grading, strict=True) if g["criteria_met"]
    )
    return achieved / total_possible


# Backward-compatible alias used by tests / callers.
calculate_score = calculate_rubric_score


async def grade_with_retry(
    model: DeepEvalBaseLLM,
    semaphore: asyncio.Semaphore,
    prompt: str,
    context_label: str,
    on_failure: Literal["raise", "false"],
    system_prompt: str = "You are a helpful evaluation assistant.",
) -> dict:
    """Retry loop for rubric grading.

    Args:
        model: DeepEval judge LLM.
        semaphore: Concurrency limiter.
        prompt: Full grading prompt.
        context_label: Short label for log messages (e.g. claim text or criterion).
        on_failure: What to do when retries are exhausted.
            "raise"  — raise RuntimeError so the caller can exclude the sample
                       from aggregates (AnswerCorrectnessRubric semantics).
            "false"  — return {"criteria_met": False, "explanation": "..."}
                       to preserve HealthBench official scoring semantics.
        system_prompt: System prompt passed to the judge.

    Returns:
        Parsed grading dict with at least "criteria_met" (bool) and "explanation" (str).

    Raises:
        RuntimeError: Only when on_failure="raise" and all retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await model.a_generate(prompt, system_prompt=system_prompt)
        except Exception:
            logger.warning(
                "Grader request failed (attempt %d/%d)",
                attempt + 1,
                MAX_RETRIES,
                exc_info=True,
            )
            continue
        result = parse_grading_response(response)
        if result is not None:
            return result
        logger.warning("Grading parse failed (attempt %d/%d)", attempt + 1, MAX_RETRIES)

    logger.error(
        "Grading failed after %d retries for: %s", MAX_RETRIES, context_label[:80]
    )
    if on_failure == "raise":
        raise RuntimeError(
            f"Judge failed to grade claim after {MAX_RETRIES} retries: "
            f"{context_label[:80]}"
        )
    return {
        "explanation": "Failed to parse grading response after retries",
        "criteria_met": False,
    }


__all__ = [
    "HealthBenchRubricMetric",
    "GRADER_TEMPLATE",
    "MAX_RETRIES",
    "grade_with_retry",
    "parse_grading_response",
    "calculate_score",
    "calculate_rubric_score",
]

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class HealthBenchRubricMetric(BaseConversationalMetric):
    """LLM-as-judge rubric grading for HealthBench.

    Grades each rubric criterion independently using a grader LLM,
    then computes the weighted score per HealthBench's official formula.

    Args:
        judge: DeepEvalBaseLLM instance for grading API calls.
        concurrent_limit: Max concurrent grading requests across all examples.
    """

    def __init__(
        self,
        judge: DeepEvalBaseLLM,
        concurrent_limit: int = 10,
        threshold: float = 0.5,
        **_kwargs: Any,
    ) -> None:
        self.judge = judge
        self.concurrent_limit = concurrent_limit
        self.threshold = threshold
        self._semaphore = asyncio.Semaphore(concurrent_limit)

        # DeepEval state
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None
        self._details: dict[str, Any] = {}

    @property
    def __name__(self) -> str:
        return "HealthBench Rubric"

    def measure(
        self,
        test_case: ConversationalTestCase,
        *_args: Any,
        **_kwargs: Any,
    ) -> float:
        """Synchronous measure - delegates to async."""
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        *_args: Any,
        **_kwargs: Any,
    ) -> float:
        """Async rubric grading for a single test case."""
        try:
            metadata = test_case.additional_metadata or {}
            rubrics = metadata.get("rubrics", [])

            if not rubrics:
                self.score = 0.0
                self.success = False
                self.reason = "No rubrics found in metadata"
                self._details = {"error": "no_rubrics"}
                return 0.0

            # Build conversation string from system_prompt + turns
            convo_with_response: list[dict[str, str]] = []
            system_prompt = metadata.get("system_prompt")
            if system_prompt:
                convo_with_response.append({"role": "system", "content": system_prompt})
            for turn in test_case.turns:
                convo_with_response.append({"role": turn.role, "content": turn.content})
            convo_str = "\n\n".join(
                f"{m['role']}: {m['content']}" for m in convo_with_response
            )

            # Grade each rubric criterion concurrently
            grading_responses = await asyncio.gather(
                *[self._grade_rubric_item(convo_str, rubric) for rubric in rubrics]
            )

            # Calculate score
            score = calculate_score(rubrics, grading_responses)
            if score is None:
                self.score = 0.0
                self.success = False
                self.reason = "No positive-point rubrics found"
                self._details = {"error": "no_positive_rubrics"}
                return 0.0

            self.score = score
            self.success = score >= self.threshold
            self.reason = (
                f"Score {score:.4f} from {len(rubrics)} rubric criteria "
                f"({sum(1 for g in grading_responses if g['criteria_met'])}/{len(rubrics)} met)"
            )
            example_tags = metadata.get("example_tags", [])
            cluster_scores = self._calculate_cluster_scores(rubrics, grading_responses)

            # Build sub-scores for tag/cluster-level aggregation.
            # The runner promotes these into separate MetricResult entries
            # so avg_aggregator can group them by name automatically.
            sub_scores: list[dict[str, object]] = []
            for tag in example_tags:
                if tag.startswith(("theme:", "physician_agreed_category:")):
                    sub_scores.append({"name": tag, "score": score})
            for tag, cs in cluster_scores.items():
                sub_scores.append({"name": tag, "score": cs})

            self._details = {
                "example_tags": example_tags,
                "cluster_scores": cluster_scores,
                "_sub_scores": sub_scores,
                "rubric_grades": [
                    {
                        "criterion": r["criterion"],
                        "points": r["points"],
                        "tags": r.get("tags", []),
                        "criteria_met": g["criteria_met"],
                        "explanation": g.get("explanation", ""),
                    }
                    for r, g in zip(rubrics, grading_responses, strict=True)
                ],
                "achieved_points": sum(
                    r["points"]
                    for r, g in zip(rubrics, grading_responses, strict=True)
                    if g["criteria_met"]
                ),
                "total_possible_points": sum(
                    r["points"] for r in rubrics if r["points"] > 0
                ),
                "num_criteria_met": sum(
                    1 for g in grading_responses if g["criteria_met"]
                ),
                "num_criteria_total": len(rubrics),
            }
            # Store in additional_metadata so the runner's _score_test_case
            # can retrieve details via test_case.additional_metadata[metric_name]
            if test_case.additional_metadata is None:
                test_case.additional_metadata = {}
            test_case.additional_metadata[self.__name__] = self._details

            return self.score

        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            self.reason = f"Grading failed: {e}"
            logger.exception("HealthBench rubric grading failed")
            raise

    async def _grade_rubric_item(
        self,
        convo_str: str,
        rubric: dict,
    ) -> dict:
        """Grade a single rubric item with retry logic."""
        rubric_str = f"[{rubric['points']}] {rubric['criterion']}"
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
            "<<rubric_item>>", rubric_str
        )
        return await grade_with_retry(
            model=self.judge,
            semaphore=self._semaphore,
            prompt=grader_prompt,
            context_label=rubric["criterion"],
            on_failure="false",
            system_prompt="You are a helpful assistant.",
        )

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success or False

    def _calculate_cluster_scores(
        self, rubrics: list[dict], grading_responses: list[dict]
    ) -> dict[str, float]:
        """Calculate subset scores for each unique `cluster:*` tag."""
        # Find all unique cluster tags
        cluster_tags = set()
        for r in rubrics:
            for tag in r.get("tags", []):
                if tag.startswith("cluster:"):
                    cluster_tags.add(tag)

        cluster_scores = {}
        for tag in cluster_tags:
            # Filter rubrics and their corresponding responses that have this tag
            filtered_pairs = [
                (r, g)
                for r, g in zip(rubrics, grading_responses, strict=True)
                if tag in r.get("tags", [])
            ]
            if not filtered_pairs:
                continue

            f_rubrics, f_responses = zip(*filtered_pairs, strict=True)
            score = calculate_score(list(f_rubrics), list(f_responses))
            if score is not None:
                cluster_scores[tag] = score

        return cluster_scores
