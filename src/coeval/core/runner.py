"""
Evaluation Runner - orchestrates the full evaluation pipeline.

Design:
    - Pluggable inference clients
    - Custom evaluate() handles metric dispatch (det vs deepeval)
    - Runner focuses on generation, scoring, and reporting
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from coeval.core.evaluate import EvalLookup, evaluate
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


class EvalRunner:
    """Evaluation runner.

    Args:
        client: Inference client (PassthroughClient, etc.)
        concurrent_limit: Max concurrent evaluations
        output_dir: Directory to write evaluation results to
    """

    def __init__(
        self,
        client: InferenceClient,
        concurrent_limit: int = 10,
        output_dir: str | None = None,
        inference_max_attempts: int = 3,
        inference_retry_delay_s: float = 1.0,
    ):
        if inference_max_attempts < 1:
            raise ValueError("inference_max_attempts must be at least 1")
        if inference_retry_delay_s < 0:
            raise ValueError("inference_retry_delay_s must be non-negative")

        self.client = client
        self.concurrent_limit = concurrent_limit
        self.output_dir = output_dir
        self.inference_max_attempts = inference_max_attempts
        self.inference_retry_delay_s = inference_retry_delay_s
        self.semaphore = asyncio.Semaphore(concurrent_limit)

    def _enable_json_mode(self, metrics: Sequence[BaseMetric]) -> None:
        """Enable JSON mode for judge models."""
        for metric in metrics:
            if hasattr(metric, "model") and hasattr(metric.model, "model_data"):
                metric.model.model_data.supports_json = True

    async def _generate_with_retry(self, messages: list[dict], sample_id: int) -> str:
        """Generate one candidate response, retrying transient inference failures."""
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

    async def _generate_predictions(
        self,
        dataset: GoldenDatasetBase,
    ) -> tuple[list[str], list[dict]]:
        """Generate predictions for all goldens.

        Returns:
            Tuple of (predictions, metadata_list) in same order as goldens.
        """
        goldens = dataset.goldens
        dataset_name = dataset.name

        async def generate_one(idx: int, golden: object) -> tuple[str, dict]:
            gen_start = time.perf_counter()
            inference_error: str | None = None

            try:
                query = dataset.get_generation_input(golden)
                completion = await self._generate_with_retry(query, idx)
            except Exception as e:
                logger.error(f"Inference failed for sample {idx}: {e}")
                completion, inference_error = "", str(e)

            gen_ms = (time.perf_counter() - gen_start) * 1000

            parsed = parse_structured_response(completion)
            actual_output = parsed.answer or completion
            rationale = parsed.reasoning or ""

            progress.advance(task)
            metadata = {
                "sample_id": idx,
                "rationale": rationale,
                "generation_time_ms": gen_ms,
                "inference_failed": inference_error is not None,
                "inference_error": inference_error,
            }
            return actual_output, metadata

        with console.progress() as progress:
            task = progress.add_task(
                f"[cyan]Generating responses for {dataset_name}...",
                total=len(goldens),
            )
            results = await asyncio.gather(
                *[generate_one(i, g) for i, g in enumerate(goldens)]
            )

        predictions = [pred for pred, _ in results]
        metadata_list = [meta for _, meta in results]
        return predictions, metadata_list

    def _score_test_case(
        self,
        test_case: TestCase,
        metadata: dict,
        metrics: Sequence[BaseMetric],
        eval_lookup: EvalLookup,
    ) -> list[MetricResult]:
        """Score a single test case from eval_lookup."""
        if metadata["inference_failed"]:
            return [
                MetricResult(
                    name=m.__name__,
                    score=None,
                    passed=False,
                    reason="Skipped: inference failed",
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
        return [
            EvalResult(
                sample_id=meta["sample_id"],
                test_case=tc,
                rationale=meta["rationale"],
                metrics=self._score_test_case(tc, meta, metrics, eval_lookup),
                generation_time_ms=meta["generation_time_ms"],
                scoring_time_ms=0.0,
                inference_failed=meta["inference_failed"],
                inference_error=meta["inference_error"],
            )
            for tc, meta in zip(test_cases, metadata_list, strict=True)
        ]

    def _build_summary(
        self,
        results: list[EvalResult],
        dataset_name: str,
        total_time: float,
        score_aggregator: ScoreAggregatorFn,
    ) -> EvalSummary:
        """Build summary statistics from evaluation results."""
        successful = [r for r in results if not r.inference_failed]
        n = len(results)

        aggregation_result = score_aggregator(results)

        return EvalSummary(
            dataset=dataset_name,
            num_samples=n,
            num_passed=sum(r.passed for r in successful),
            num_inference_failed=n - len(successful),
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
        name = dataset_name or dataset.name
        self._enable_json_mode(metrics)
        console.start_eval(len(dataset.goldens), name)
        start_time = time.perf_counter()

        predictions, metadata_list = await self._generate_predictions(dataset)
        test_cases = dataset.build_test_cases(predictions)

        eval_lookup = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            max_concurrent=self.concurrent_limit,
        )

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
