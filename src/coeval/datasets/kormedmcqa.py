"""
KorMedMCQA Dataset Loader.

Korean medical licensing exam MCQ questions across four professional domains:
doctor, nurse, pharmacist (pharm), and dentist.
Returns list of Golden objects for use with EvaluationDataset.
"""

import logging

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase

logger = logging.getLogger(__name__)


class _KorMedMCQABase(GoldenDatasetBase):
    """
    Base class for KorMedMCQA dataset loaders.

    Loads Korean medical licensing exam MCQs from HuggingFace.
    Reference: https://huggingface.co/datasets/sean0042/KorMedMCQA
    """

    HUGGINGFACE_PATH = "sean0042/KorMedMCQA"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D", "E"]
    # Subclasses must set this to a specific subset name.
    SUBSET: str

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load KorMedMCQA samples as Golden objects."""
        dataset = load_dataset(
            self.HUGGINGFACE_PATH, name=self.SUBSET, split=self.SPLIT
        )

        goldens = []
        for idx, row in enumerate(dataset):
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()

            options = {
                "A": (row.get("A") or "").strip(),
                "B": (row.get("B") or "").strip(),
                "C": (row.get("C") or "").strip(),
                "D": (row.get("D") or "").strip(),
                "E": (row.get("E") or "").strip(),
            }

            answer_int = row.get("answer", -1)
            if not isinstance(answer_int, int) or answer_int not in range(1, 6):
                logger.debug(f"Skipping row {idx}: invalid answer={answer_int}")
                continue

            if not question_text or not any(options.values()):
                logger.debug(f"Skipping row {idx}: missing question or options")
                continue

            answer_letter = self.OPTION_LETTERS[answer_int - 1]
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
                        "source": "kormedmcqa",
                        "subject": row.get("subject"),
                        "year": row.get("year"),
                        "options": options,
                    },
                )
            )

        return goldens


@register_dataset("kormedmcqa_doctor")
class KorMedMCQADoctorDataset(_KorMedMCQABase):
    """KorMedMCQA — Korean Medical Licensing Exam (435 test samples)."""

    SUBSET = "doctor"
    TOTAL_SAMPLES = 435

    @property
    def name(self) -> str:
        return "KorMedMCQA (Doctor)"


@register_dataset("kormedmcqa_nurse")
class KorMedMCQANurseDataset(_KorMedMCQABase):
    """KorMedMCQA — Korean Nursing Licensing Exam (878 test samples)."""

    SUBSET = "nurse"
    TOTAL_SAMPLES = 878

    @property
    def name(self) -> str:
        return "KorMedMCQA (Nurse)"


@register_dataset("kormedmcqa_pharm")
class KorMedMCQAPharmDataset(_KorMedMCQABase):
    """KorMedMCQA — Korean Pharmacy Licensing Exam (885 test samples)."""

    SUBSET = "pharm"
    TOTAL_SAMPLES = 885

    @property
    def name(self) -> str:
        return "KorMedMCQA (Pharm)"


@register_dataset("kormedmcqa_dentist")
class KorMedMCQADentistDataset(_KorMedMCQABase):
    """KorMedMCQA — Korean Dental Licensing Exam (811 test samples)."""

    SUBSET = "dentist"
    TOTAL_SAMPLES = 811

    @property
    def name(self) -> str:
        return "KorMedMCQA (Dentist)"
