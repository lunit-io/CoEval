"""
MedMCQA Dataset Loader.

Medical multiple-choice questions covering various medical subjects.
Returns list of Golden objects for use with EvaluationDataset.
"""

import logging

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase

logger = logging.getLogger(__name__)


@register_dataset("medmcqa")
class MedMCQADataset(GoldenDatasetBase):
    """
    MedMCQA dataset loader.

    Loads medical MCQ questions from HuggingFace.
    Reference: https://huggingface.co/datasets/lighteval/med_mcqa
    """

    HUGGINGFACE_PATH = "lighteval/med_mcqa"
    SPLIT = "validation"
    OPTION_LETTERS = ["A", "B", "C", "D"]
    TOTAL_SAMPLES = 4183

    def __init__(self, num_samples: int | None = None, **kwargs):
        """Initialize dataset, loading up to ``num_samples`` examples."""
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MedMCQA"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedMCQA samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

        goldens = []
        for idx, row in enumerate(dataset):
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()

            # Parse options: opa, opb, opc, opd
            options = {
                "A": (row.get("opa") or "").strip(),
                "B": (row.get("opb") or "").strip(),
                "C": (row.get("opc") or "").strip(),
                "D": (row.get("opd") or "").strip(),
            }

            # Answer is 1-indexed (cop)
            cop = row.get("cop", -1)
            if not isinstance(cop, int) or cop not in [1, 2, 3, 4]:
                logger.warning("Skipping row %d: invalid answer index", idx)
                continue

            if not question_text or not any(options.values()):
                logger.warning("Skipping row %d: missing question or options", idx)
                continue

            answer_letter = self.OPTION_LETTERS[cop - 1]

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
                        "source": "medmcqa",
                        "subject": row.get("subject_name"),
                        "topic": row.get("topic_name"),
                        "options": options,
                    },
                )
            )

        return goldens
