# 📊 Adding a New Metric

CoEval metrics fall into two categories: **deterministic** (rule-based, no LLM) and **LLM-as-judge** (uses a judge model via DeepEval).

---

## Quick Overview

```
src/coeval/
├── metrics/
│   ├── base.py                  # DeterministicMetric base class
│   ├── mcq_accuracy.py          # MCQ exact match
│   ├── classification.py        # Label classification
│   ├── numeric_accuracy.py      # Numeric comparison
│   ├── healthbench_rubric.py    # LLM-as-judge rubric scoring
│   ├── faithfulness.py          # DeepEval FaithfulnessMetric
│   ├── answer_relevancy.py      # DeepEval AnswerRelevancyMetric
│   ├── contextual_precision.py  # DeepEval ContextualPrecisionMetric
│   └── contextual_recall.py     # DeepEval ContextualRecallMetric
└── conf/datasets/metrics/
    ├── mcq_accuracy.yaml        # One YAML per metric
    ├── classification.yaml
    └── judge/                   # Judge model configs
        └── gpt-4.1.yaml
```

---

## Adding a Deterministic Metric

Deterministic metrics are fast, reproducible, and require no LLM. They run synchronously via `.measure()`.

### 1. Create the metric class

```python
# src/coeval/metrics/my_metric.py
from typing import Any

from coeval.core.types import LLMTestCase
from coeval.metrics.base import DeterministicMetric


class MyMetric(DeterministicMetric):
    @property
    def __name__(self) -> str:
        return "my_metric"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        predicted = test_case.actual_output.strip().lower()
        expected = test_case.expected_output.strip().lower()
        is_correct = predicted == expected

        return self._set_result(
            test_case,
            is_correct=is_correct,
            reason="Exact match" if is_correct else f"Expected '{expected}', got '{predicted}'",
            details={"predicted": predicted, "expected": expected},
        )
```

> **Key contract:**
> Set `self.score` (0.0 or 1.0) and `self.success` (bool) — `_set_result()` handles this
> Use `_fail(reason)` for cases where extraction fails (returns 0.0)
> `_set_result()` also attaches `details` to `test_case.additional_metadata[metric_name]` for breakdown reporting

### 2. Create the YAML config

```yaml
# src/coeval/conf/datasets/metrics/my_metric.yaml
my_metric:
  _target_: coeval.metrics.my_metric.MyMetric
  threshold: 1.0
```

### 3. Reference from a dataset config

```yaml
# src/coeval/conf/datasets/my_dataset.yaml
defaults:
  - metrics@metrics.my_dataset: my_metric
  - metrics/aggregator@score_aggregators.my_dataset: default
```

✅ Done! Your metric is now wired into the evaluation pipeline.

---

## DeterministicMetric API

The `DeterministicMetric` base class provides:

| Method | Description |
|--------|-------------|
| `measure(test_case)` | **Required.** Score a single test case. Must set `self.score` and `self.success`. |
| `_set_result(test_case, is_correct, reason, details)` | Helper that sets score/success/reason and attaches details to metadata. Returns score. |
| `_fail(reason)` | Helper for failed extraction — sets score=0.0, success=False. Returns 0.0. |
| `a_measure(test_case)` | Async wrapper — just calls `measure()` (deterministic metrics have no I/O). |

> ⚠️ **Important:** `use_deepeval = False` is set as a class variable. This tells the evaluation dispatcher to run the metric directly via `.measure()` instead of through DeepEval's async pipeline.

---

## Adding an LLM-as-Judge Metric

For metrics that require an LLM to evaluate (e.g., rubric grading, faithfulness):

### Option A: Use DeepEval's built-in metrics (simplest)

```python
# src/coeval/metrics/my_judge_metric.py
from deepeval.metrics import FaithfulnessMetric  # or any DeepEval metric

# Re-export — the YAML config handles instantiation
```

```yaml
# src/coeval/conf/datasets/metrics/my_judge_metric.yaml
defaults:
  - judge@my_judge_metric.model: gpt-4.1

my_judge_metric:
  _target_: deepeval.metrics.FaithfulnessMetric
  threshold: 0.7
  include_reason: true
  async_mode: true
```

### Option B: Custom LLM-as-judge (e.g., HealthBench rubric)

See `healthbench_rubric.py` for a full example of a custom `BaseConversationalMetric` implementation that:
- Sends per-criterion prompts to a judge LLM
- Parses structured JSON responses
- Computes weighted scores from rubric criteria

---

## Aggregation Strategies

Each dataset config specifies how metric scores are aggregated across samples:

| Aggregator | YAML Key | Formula | Use case |
|-----------|----------|---------|----------|
| Simple Average | `default` / `simple_avg` | `mean(scores)` | Most MCQ datasets |
| Weighted Average | `weighted_avg` | `weighted_mean(scores, weights)` | Grouped sub-datasets |
| Macro F1 | `f1` | `mean(per_class_f1)` | Classification tasks |

Reference in dataset config:

```yaml
defaults:
  - metrics/aggregator@score_aggregators.my_dataset: default   # or f1, weighted_avg
```
