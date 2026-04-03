"""
MedXpertQA Dataset Loader.

Expert-level medical question answering benchmark.
Returns list of Golden objects for use with EvaluationDataset.
"""

from enum import StrEnum

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


class MedXpertQuestionType(StrEnum):
    """Question type for MedXpertQA."""

    REASONING = "reasoning"
    UNDERSTANDING = "understanding"
    ALL = "all"


class _MedXpertQABase(GoldenDatasetBase):
    """Base class for MedXpertQA datasets."""

    HUGGINGFACE_PATH = "TsinghuaC3I/MedXpertQA"
    SUBSET = "Text"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    TOTAL_SAMPLES = 2450

    QUESTION_TYPE: str = "all"

    def __init__(self, num_samples: int | None = None, **kwargs):
        self.question_type = MedXpertQuestionType(self.QUESTION_TYPE)
        super().__init__(goldens=self._load(num_samples), **kwargs)

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedXpertQA samples as Golden objects."""
        full_dataset = load_dataset(self.HUGGINGFACE_PATH, self.SUBSET)
        dataset = full_dataset[self.SPLIT]

        if self.question_type != MedXpertQuestionType.ALL:
            dataset = dataset.filter(
                lambda x: (
                    str(x.get("question_type", "")).strip().lower()
                    == self.question_type.value
                )
            )

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()
            raw_options = row.get("options") or {}

            answer_letter = str(row.get("label", "")).strip().upper()

            if not question_text or not raw_options:
                continue

            if isinstance(raw_options, dict):
                options = {k.upper(): v for k, v in raw_options.items() if v}
            else:
                continue

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
                        "source": "medxpertqa",
                        "question_type": row.get("question_type"),
                        "options": options,
                    },
                )
            )

        return goldens


@register_dataset("medxpertqa_reasoning")
class MedXpertQAReasoningDataset(_MedXpertQABase):
    """MedXpertQA reasoning questions."""

    QUESTION_TYPE = "reasoning"
    TOTAL_SAMPLES = 1861

    @property
    def name(self) -> str:
        return "MedXpertQA Reasoning"


@register_dataset("medxpertqa_understanding")
class MedXpertQAUnderstandingDataset(_MedXpertQABase):
    """MedXpertQA understanding questions."""

    QUESTION_TYPE = "understanding"
    TOTAL_SAMPLES = 589

    @property
    def name(self) -> str:
        return "MedXpertQA Understanding"
