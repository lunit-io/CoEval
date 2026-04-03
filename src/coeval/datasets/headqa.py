"""
HEAD-QA v2 Dataset Loader.

Healthcare professional examination questions in English.
Returns list of Golden objects for use with EvaluationDataset.
"""

import logging

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase

logger = logging.getLogger(__name__)


@register_dataset("headqa")
class HeadQADataset(GoldenDatasetBase):
    """
    HEAD-QA v2 dataset loader.

    Loads healthcare examination questions from HuggingFace.
    Reference: https://huggingface.co/datasets/alesi12/head_qa_v2

    Note: HEAD-QA uses numeric options (1, 2, 3, 4, 5) instead of letters.
    We convert these to A, B, C, D, E for consistency with MCQ accuracy.
    """

    HUGGINGFACE_PATH = "alesi12/head_qa_v2"
    SUBSET = "en"
    SPLIT = "train"
    TOTAL_SAMPLES = 12751

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "HEAD-QA v2"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load HEAD-QA v2 samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, self.SUBSET, split=self.SPLIT)

        goldens = []
        for idx, row in enumerate(dataset):
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("qtext") or "").strip()
            answers = row.get("answers", [])
            correct_answer = row.get("ra", -1)

            # Validate
            if not question_text or not answers:
                logger.debug(f"Skipping row {idx}: missing question or answers")
                continue
            if not isinstance(correct_answer, int) or not (
                1 <= correct_answer <= len(answers)
            ):
                logger.debug(
                    f"Skipping row {idx}: invalid answer index ra={correct_answer}"
                )
                continue

            # Convert numeric options to letter options
            option_letters = [chr(ord("A") + i) for i in range(len(answers))]
            options = {}
            for i, ans in enumerate(answers):
                letter = option_letters[i]
                options[letter] = ans.get("atext", "").strip()

            # Convert 1-indexed answer to letter
            answer_letter = option_letters[correct_answer - 1]

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
                        "source": "headqa",
                        "options": options,
                    },
                )
            )

        return goldens
