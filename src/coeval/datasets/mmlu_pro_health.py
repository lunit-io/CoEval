"""
MMLU Pro (Health) Dataset Loader.

Health subset of MMLU Pro benchmark.
Returns list of Golden objects for use with EvaluationDataset.
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


@register_dataset("mmlu_pro_health")
class MMLUProHealthDataset(GoldenDatasetBase):
    """
    MMLU Pro Health subset dataset loader.

    Loads the health category from MMLU-Pro benchmark.
    Reference: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

    Note: MMLU-Pro can have up to 10 options (A-J).
    """

    HUGGINGFACE_PATH = "TIGER-Lab/MMLU-Pro"
    SPLIT = "test"
    CATEGORY = "health"
    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    TOTAL_SAMPLES = 818

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MMLU Pro Health"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MMLU Pro Health samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

        # Filter to health category
        dataset = dataset.filter(lambda x: x.get("category") == self.CATEGORY)

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()
            opts = row.get("options", [])

            answer_letter = (row.get("answer") or "").strip().upper()

            if not question_text or not opts:
                continue

            # Convert list options to dict with letter keys
            if isinstance(opts, list):
                options = {}
                for i, v in enumerate(opts):
                    if v and v.strip().upper() != "N/A":
                        letter = chr(ord("A") + i)
                        options[letter] = v.strip()
            else:
                options = opts

            if answer_letter not in self.OPTION_LETTERS or answer_letter not in options:
                continue

            options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

            question = MCQ_PROMPT_TEMPLATE.format(
                question=question_text,
                options=options_text,
            )

            goldens.append(
                Golden(
                    input=question,
                    expected_output=answer_letter,
                    context=[options.get(answer_letter, "")],
                    additional_metadata={
                        "source": "mmlu_pro_health",
                        "category": row.get("category"),
                        "options": options,
                    },
                )
            )

        return goldens
