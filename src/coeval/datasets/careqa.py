"""
CareQA Dataset Loader.

Spanish medical exam questions translated to English (CareQA_en).
Returns list of Golden objects for use with EvaluationDataset.
"""

import logging

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase

logger = logging.getLogger(__name__)


@register_dataset("careqa")
class CareQADataset(GoldenDatasetBase):
    """
    CareQA dataset loader (English version).

    Loads medical exam questions from HuggingFace.
    Reference: https://huggingface.co/datasets/HPAI-BSC/CareQA
    """

    HUGGINGFACE_PATH = "HPAI-BSC/CareQA"
    SUBSET = "CareQA_en"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D"]
    TOTAL_SAMPLES = 5621

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "CareQA"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load CareQA samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, self.SUBSET, split=self.SPLIT)

        goldens = []
        for idx, row in enumerate(dataset):
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = row.get("question", "")

            # Parse options: op1, op2, op3, op4
            options = {
                "A": row.get("op1", ""),
                "B": row.get("op2", ""),
                "C": row.get("op3", ""),
                "D": row.get("op4", ""),
            }

            # Answer is 1-indexed (cop)
            cop = row.get("cop", 1)
            if isinstance(cop, int) and 1 <= cop <= 4:
                answer_letter = self.OPTION_LETTERS[cop - 1]
            else:
                logger.debug(f"Skipping row {idx}: invalid answer index cop={cop}")
                continue

            # Format options text
            options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

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
                        "source": "careqa",
                        "options": options,
                    },
                )
            )

        return goldens
