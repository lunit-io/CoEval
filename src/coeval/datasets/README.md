# 🧰 Adding a New Dataset


CoEval datasets are simple: a **Python loader** that returns `Golden` objects + a **YAML config** that wires it into Hydra.

---

## Quick Overview

```
src/coeval/
├── datasets/
│   ├── base.py              # GoldenDatasetBase, MultiTurnDatasetBase
│   ├── medqa.py             # 17 dataset loaders (one file each)
│   ├── healthbench.py
│   └── ...
└── conf/datasets/
    ├── medqa.yaml           # One YAML per dataset
    ├── all.yaml             # Composes all datasets
    └── metrics/             # Shared metric configs
```

---

## Step-by-Step: Adding an MCQ Dataset

### 1. Create the dataset loader

Create a new file in `src/coeval/datasets/`.

```python
# src/coeval/datasets/my_dataset.py
from datasets import load_dataset

from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


class MyDataset(GoldenDatasetBase):
    """My custom medical MCQ dataset."""

    TOTAL_SAMPLES = 500  # Total in the test split (used for progress display)

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MyDataset"

    def _load(self, num_samples: int | None) -> list[Golden]:
        dataset = load_dataset("your-hf-org/your-dataset", split="test")

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            options = row["options"]  # e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
            options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

            goldens.append(
                Golden(
                    input=MCQ_PROMPT_TEMPLATE.format(
                        question=row["question"],
                        options=options_text,
                    ),
                    expected_output=row["answer"],  # e.g. "A"
                    additional_metadata={"source": "my_dataset"},
                )
            )
        return goldens
```

> **Key points:**
> Inherit from `GoldenDatasetBase` (single-turn) or `MultiTurnDatasetBase` (multi-turn)
> Accept `num_samples: int | None` and `**kwargs` (Hydra passes `system_prompt` via kwargs)
> Return a list of `Golden` objects with `input` (formatted prompt) and `expected_output` (answer letter)
> Use `MCQ_PROMPT_TEMPLATE` or `MCQ_WITH_CONTEXT_TEMPLATE` from `datasets.base`

### 2. Create the Hydra config

```yaml
# src/coeval/conf/datasets/my_dataset.yaml
# @package _global_
defaults:
  - metrics@metrics.my_dataset: mcq_accuracy
  - metrics/aggregator@score_aggregators.my_dataset: default

datasets:
  my_dataset:
    _target_: coeval.datasets.my_dataset.MyDataset
    num_samples: ${num_samples}
    system_prompt: ${system_prompt}
```

> **The `defaults` entries:**
> `metrics@metrics.<name>`: Which metric to use (see `conf/datasets/metrics/`)
> `metrics/aggregator@score_aggregators.<name>`: How to aggregate scores
>
> **Available metrics:** `mcq_accuracy`, `classification`, `numeric_accuracy`, `healthbench_rubric`
> **Available aggregators:** `default` (average), `weighted_avg`, `f1`, `simple_avg`

### 3. Add to `all.yaml`

```yaml
# src/coeval/conf/datasets/all.yaml
defaults:
  - ...existing datasets...
  - my_dataset        # ← Add this line
```

### 4. Test it

```bash
mise run eval -- datasets=my_dataset num_samples=5
```

✅ Done! Your dataset is now available via CLI.

---

## Base Classes

### `GoldenDatasetBase` — Single-turn datasets

Most medical MCQ datasets use this. The base class:
- Inherits from DeepEval's `EvaluationDataset`
- Accepts `system_prompt` (injected from Hydra config)
- Provides `build_test_cases(predictions)` to create `LLMTestCase` objects
- Provides `get_generation_input(golden)` that returns `[system_msg, user_msg]`

### `MultiTurnDatasetBase` — Multi-turn chat datasets

For datasets with conversation history (e.g., HealthBench). Differences:
- Uses `ConversationalGolden` with `turns: list[Turn]` instead of `Golden` with `input: str`
- System prompts stored per-example in `additional_metadata["system_prompt"]`
- `build_test_cases()` returns `ConversationalTestCase` with the model prediction appended as the final turn

---

## Prompt Templates

Two built-in templates in `datasets/base.py`:

```python
MCQ_PROMPT_TEMPLATE = """
Question: {question}
{options}
Answer: """

MCQ_WITH_CONTEXT_TEMPLATE = """
Context: {context}

Question: {question}
{options}
Answer: """
```

For custom prompts (e.g., MedCalc, MedHallu), define your own template in the dataset file.

---

## Tips

- **Option formatting**: Always format as `A. option_text\nB. option_text\n...`
- **Answer normalization**: `expected_output` should be a single uppercase letter (A/B/C/D)
- **Metadata**: Use `additional_metadata` to pass extra info (difficulty level, source, etc.) for breakdown reporting
- **Grouped datasets**: For datasets with subsets (e.g., KorMedMCQA has Doctor/Nurse/Pharm/Dentist), create one YAML with nested entries under a group key. See `kormedmcqa.yaml` for an example.
