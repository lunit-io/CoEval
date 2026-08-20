"""
Evaluation Runner - orchestrates the full evaluation pipeline.

Design:
    - Pluggable inference clients
    - Generation and judging overlap per sample (no phase barrier)
    - a_evaluate_one() handles metric dispatch (deterministic vs LLM judge)
    - Runner focuses on generation, scoring, and reporting
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from coeval.core.evaluate import EvalLookup, MetricScore, a_evaluate_one
from coeval.core.schema import (
    EvalResult,
    EvalSummary,
    MetricResult,
)
from coeval.core.types import BaseMetric, InferenceClient, TestCase
from coeval.datasets.base import GoldenDatasetBase
from coeval.util.aggregation import ScoreAggregatorFn
from coeval.util.console import console
from coeval.util.parsers import parse_structured_response

logger = logging.getLogger(__name__)

# (test_case, metadata, metric scores) — scores is None when inference failed.
_PipelineResult = tuple[TestCase, dict, list[MetricScore] | None]


class EvalRunner:
    """Evaluation runner.

    Args:
        client: Inference client (PassthroughClient, etc.)
        concurrent_limit: Max concurrent evaluations
        output_dir: Directory to write evaluation results to
        score_inference_failures_as_zero: How to treat a sample whose inference
            failed -- an endpoint error, a timeout, or a reply carrying no
            content.

            ``False`` (default) drops it from the aggregate. That is the right
            reading for research: report quality on what the model actually
            answered.

            ``True`` scores it 0 and keeps it in the denominator. Use this for
            competitive or comparative runs, where dropping is unsound in three
            ways. It rewards failure -- returning an error on the questions you
            expect to answer badly raises your mean, while returning a bad
            answer lowers it. It desynchronises denominators, so two systems get
            averaged over different item sets and stop being comparable. And it
            is simply wrong on the merits: failing to answer is not the same as
            not being asked.

            Judge infrastructure failures are unaffected either way. Those are
            our fault, not the endpoint's, and stay excluded via
            ``scoring_failed``.

            Note that sub-scores (``theme:*``, ``axis:*``) still omit
            inference-failed samples; only the headline metric carries the 0.
            Compare ``metric_scores[...].denominator`` against ``num_samples``
            to see how many zeros a run absorbed.
    """

    def __init__(
        self,
        client: InferenceClient,
        concurrent_limit: int = 10,
        output_dir: str | None = None,
        inference_max_attempts: int = 3,
        inference_retry_delay_s: float = 1.0,
        score_inference_failures_as_zero: bool = False,
    ):
        if inference_max_attempts < 1:
            raise ValueError("inference_max_attempts must be at least 1")
        if inference_retry_delay_s < 0:
            raise ValueError("inference_retry_delay_s must be non-negative")

        self.client = client
        self.concurrent_limit = concurrent_limit
        self.output_dir = output_dir
        self.score_inference_failures_as_zero = score_inference_failures_as_zero
        self.inference_max_attempts = inference_max_attempts
        self.inference_retry_delay_s = inference_retry_delay_s
        self.semaphore = asyncio.Semaphore(concurrent_limit)

    def _enable_json_mode(self, metrics: Sequence[BaseMetric]) -> None:
        """Enable JSON mode for judge models."""
        for metric in metrics:
            if hasattr(metric, "model") and hasattr(metric.model, "model_data"):
                metric.model.model_data.supports_json = True

    async def _generate_with_retry(self, messages: list[dict], sample_id: int) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.inference_max_attempts + 1):
            try:
                async with self.semaphore:
                    result = await self.client.generate(messages)
                return str(result) if result else ""
            except Exception as error:
                last_error = error
                if attempt == self.inference_max_attempts:
                    break
                logger.warning(
                    "Inference attempt %s/%s failed for sample %s: %s",
                    attempt,
                    self.inference_max_attempts,
                    sample_id,
                    error,
                )
                if self.inference_retry_delay_s:
                    await asyncio.sleep(
                        self.inference_retry_delay_s * 2 ** (attempt - 1)
                    )

        assert last_error is not None
        raise last_error

    async def _generate_one(
        self,
        dataset: GoldenDatasetBase,
        idx: int,
        golden: object,
    ) -> tuple[str, dict]:
        """Returns (actual_output, metadata)."""
        gen_start = time.perf_counter()
        inference_error: str | None = None

        try:
            query = dataset.get_generation_input(golden)
            completion = await self._generate_with_retry(query, idx)
        except Exception as e:
            logger.error(f"Inference failed for sample {idx}: {e}")
            completion, inference_error = "", str(e)

        parsed = parse_structured_response(completion)
        metadata = {
            "sample_id": idx,
            "rationale": parsed.reasoning or "",
            "generation_time_ms": (time.perf_counter() - gen_start) * 1000,
            "scoring_time_ms": 0.0,
            "inference_failed": inference_error is not None,
            "inference_error": inference_error,
        }
        return parsed.answer or completion, metadata

    def _score_test_case(
        self,
        test_case: TestCase,
        metadata: dict,
        metrics: Sequence[BaseMetric],
        eval_lookup: EvalLookup,
    ) -> list[MetricResult]:
        """Score a single test case from eval_lookup."""
        if metadata["inference_failed"]:
            as_zero = self.score_inference_failures_as_zero
            return [
                MetricResult(
                    name=m.__name__,
                    score=0.0 if as_zero else None,
                    passed=False,
                    reason=(
                        "Inference failed: scored 0 (endpoint produced no usable answer)"
                        if as_zero
                        else "Skipped: inference failed"
                    ),
                )
                for m in metrics
            ]

        sample_id = metadata["sample_id"]
        scores = eval_lookup.get(sample_id)
        if scores is None:
            logger.error(f"No eval result for sample_id={sample_id}")
            return [
                MetricResult(
                    name=m.__name__,
                    score=None,
                    passed=False,
                    reason="Missing eval result",
                )
                for m in metrics
            ]

        extra = getattr(test_case, "additional_metadata", None) or {}
        results: list[MetricResult] = []
        for ms in scores:
            details = extra.get(ms.name, {})
            sub_scores = details.pop("_sub_scores", [])
            results.append(
                MetricResult(
                    name=ms.name,
                    score=ms.score,
                    passed=ms.success and ms.score is not None,
                    reason=ms.error or ms.reason or "",
                    details=details,
                )
            )
            for sub in sub_scores:
                results.append(
                    MetricResult(
                        name=sub["name"],
                        score=sub["score"],
                        passed=ms.success,
                        reason="",
                    )
                )
        return results

    def _build_eval_results(
        self,
        test_cases: list[TestCase],
        metadata_list: list[dict],
        metrics: Sequence[BaseMetric],
        eval_lookup: EvalLookup,
    ) -> list[EvalResult]:
        """Build EvalResult list from eval_lookup."""
        results: list[EvalResult] = []
        for tc, meta in zip(test_cases, metadata_list, strict=True):
            metric_results = self._score_test_case(tc, meta, metrics, eval_lookup)
            top_level_scores = eval_lookup.get(meta["sample_id"])
            expected_metric_names = sorted(metric.__name__ for metric in metrics)
            returned_metric_names = sorted(
                score.name for score in (top_level_scores or [])
            )
            metric_set_mismatch = (
                top_level_scores is not None
                and returned_metric_names != expected_metric_names
            )
            if not meta["inference_failed"] and metric_set_mismatch:
                logger.error(
                    "Incomplete eval result for sample_id=%s: expected metrics=%s, "
                    "returned metrics=%s",
                    meta["sample_id"],
                    expected_metric_names,
                    returned_metric_names,
                )
            scoring_failed = (
                not meta["inference_failed"]
                and bool(metrics)
                and (
                    top_level_scores is None
                    or metric_set_mismatch
                    or any(score.score is None for score in top_level_scores)
                )
            )
            results.append(
                EvalResult(
                    sample_id=meta["sample_id"],
                    test_case=tc,
                    rationale=meta["rationale"],
                    metrics=metric_results,
                    generation_time_ms=meta["generation_time_ms"],
                    scoring_time_ms=meta["scoring_time_ms"],
                    inference_failed=meta["inference_failed"],
                    inference_error=meta["inference_error"],
                    scoring_failed=scoring_failed,
                )
            )
        return results

    def _build_summary(
        self,
        results: list[EvalResult],
        dataset_name: str,
        total_time: float,
        score_aggregator: ScoreAggregatorFn,
    ) -> EvalSummary:
        """Build summary statistics from evaluation results."""
        # Judge failures are always excluded -- they are our infrastructure, not the
        # endpoint's answer. Inference failures are excluded only in research mode;
        # in competitive mode they carry a 0 into the denominator.
        fully_scored = [
            r
            for r in results
            if not r.scoring_failed
            and (self.score_inference_failures_as_zero or not r.inference_failed)
        ]
        n = len(results)

        aggregation_result = score_aggregator(fully_scored)

        return EvalSummary(
            dataset=dataset_name,
            num_samples=n,
            num_passed=sum(r.passed for r in fully_scored),
            num_inference_failed=sum(r.inference_failed for r in results),
            num_scoring_failed=sum(r.scoring_failed for r in results),
            total_time_s=total_time,
            avg_generation_ms=sum(r.generation_time_ms for r in results) / n
            if n
            else 0,
            avg_scoring_ms=sum(r.scoring_time_ms for r in results) / n if n else 0,
            metric_scores=aggregation_result.metric_scores,
            breakdown=aggregation_result.breakdown,
        )

    async def run(
        self,
        dataset: GoldenDatasetBase,
        metrics: Sequence[BaseMetric],
        score_aggregator: ScoreAggregatorFn,
        dataset_name: str | None = None,
    ) -> EvalSummary:
        """Run evaluation pipeline.

        Args:
            dataset: The dataset to evaluate
            metrics: List of metric instances to use for this evaluation
            score_aggregator: Function to aggregate metric scores.
            dataset_name: Override for dataset.name (used for file naming).
        """
        if not metrics:
            raise ValueError("Evaluation requires at least one metric")

        name = dataset_name or dataset.name
        self._enable_json_mode(metrics)
        console.start_eval(len(dataset.goldens), name)
        start_time = time.perf_counter()

        # Generation and judging overlap: each sample goes straight to the judge
        # once its own generation lands, instead of waiting for the whole pass.
        # The two semaphores cap each stage independently and provide backpressure.
        judge_semaphore = asyncio.Semaphore(self.concurrent_limit)

        async def generate_then_score(idx: int, golden: object) -> _PipelineResult:
            prediction, metadata = await self._generate_one(dataset, idx, golden)
            progress.advance(gen_task)
            test_case = dataset.build_test_case(idx, golden, prediction)

            scores = None
            if not metadata["inference_failed"]:
                score_start = time.perf_counter()
                scores = await a_evaluate_one(test_case, metrics, judge_semaphore)
                metadata["scoring_time_ms"] = (time.perf_counter() - score_start) * 1000

            progress.advance(judge_task)
            return test_case, metadata, scores

        with console.progress() as progress:
            gen_task = progress.add_task(
                f"[cyan]Generating responses for {name}...",
                total=len(dataset.goldens),
            )
            judge_task = progress.add_task(
                f"[magenta]Judging {name}...",
                total=len(dataset.goldens),
            )
            pipelined = await asyncio.gather(
                *[
                    generate_then_score(idx, golden)
                    for idx, golden in enumerate(dataset.goldens)
                ]
            )

        test_cases = [tc for tc, _, _ in pipelined]
        metadata_list = [meta for _, meta, _ in pipelined]
        eval_lookup = {
            meta["sample_id"]: scores
            for _, meta, scores in pipelined
            if scores is not None
        }

        results = self._build_eval_results(
            test_cases, metadata_list, metrics, eval_lookup
        )
        total_time = time.perf_counter() - start_time
        summary = self._build_summary(results, name, total_time, score_aggregator)

        console.end_eval(
            summary.num_passed,
            len(dataset.goldens),
            summary.pass_rate,
            total_time,
            summary.num_inference_failed,
            summary.num_scoring_failed,
        )

        if self.output_dir:
            self._save_results(results, summary)

        return summary

    @staticmethod
    def _save_results(
        results: list[EvalResult], summary: EvalSummary, pretty: bool = True
    ) -> Path:
        """Save results to disk."""
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.run.dir)

        results_file = output_dir / f"results_{summary.dataset}.json"
        summary_file = output_dir / f"summary_{summary.dataset}.json"

        with open(results_file, "w") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2 if pretty else None,
                ensure_ascii=False,
            )
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

        console.saved(str(results_file))
        return results_file
