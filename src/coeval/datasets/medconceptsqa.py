"""
MedConceptsQA Dataset Loader.

Medical concepts question-answering based on medical coding vocabularies.
Returns list of Golden objects for use with EvaluationDataset.
"""

from enum import StrEnum

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


class MedConceptsVocab(StrEnum):
    """Vocabulary type for MedConceptsQA."""

    ATC = "atc"
    ICD10CM = "icd10cm"
    ICD10PROC = "icd10proc"
    ICD9CM = "icd9cm"
    ICD9PROC = "icd9proc"
    ICD10CM_SAMPLE = "icd10cm_sample"
    ALL = "all"


class MedConceptsDifficulty(StrEnum):
    """Difficulty level for MedConceptsQA."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class _MedConceptsQABase(GoldenDatasetBase):
    """Base class for MedConceptsQA datasets."""

    HUGGINGFACE_PATH = "ofir408/MedConceptsQA"
    SAMPLE_PATH = "sameedkhan/medconceptsqa-sample_medarc_2k"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D"]

    DIFFICULTY: str = "easy"
    VOCAB: str = "icd10cm_sample"

    def __init__(self, num_samples: int | None = None, **kwargs):
        self.vocab = MedConceptsVocab(self.VOCAB)
        self.difficulty = MedConceptsDifficulty(self.DIFFICULTY)
        super().__init__(goldens=self._load(num_samples), **kwargs)

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedConceptsQA samples as Golden objects."""
        subset = f"{self.vocab.value}_{self.difficulty.value}"

        if self.vocab == MedConceptsVocab.ICD10CM_SAMPLE:
            subset_name = subset.replace("_sample", "")
            ds = load_dataset(self.SAMPLE_PATH, subset_name)
            dataset = ds[self.SPLIT]
        else:
            ds = load_dataset(self.HUGGINGFACE_PATH, subset)
            dataset = ds[self.SPLIT]

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()

            # Parse options: option1, option2, option3, option4
            options = {}
            for i, letter in enumerate(self.OPTION_LETTERS, start=1):
                opt_value = row.get(f"option{i}", "")
                if opt_value:
                    options[letter] = opt_value.strip()

            # Clean question text - remove embedded options if present
            question_lines = []
            for line in question_text.splitlines():
                stripped = line.strip()
                is_option_line = any(
                    stripped.startswith(f"{letter}{sep}")
                    for letter in self.OPTION_LETTERS
                    for sep in (".", ")", ":", "-")
                )
                if not is_option_line:
                    question_lines.append(line)
            question_stem = "\n".join(question_lines).strip()

            # Answer is a letter (answer_id)
            answer_letter = (row.get("answer_id") or "").strip().upper()

            if not question_stem or not options:
                continue

            if answer_letter not in options:
                continue

            # Format options text
            options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

            question = MCQ_PROMPT_TEMPLATE.format(
                question=question_stem,
                options=options_text,
            )

            goldens.append(
                Golden(
                    input=question,
                    expected_output=answer_letter,
                    context=[options.get(answer_letter, "")],
                    additional_metadata={
                        "source": "medconceptsqa",
                        "vocab": row.get("vocab"),
                        "level": row.get("level"),
                        "options": options,
                    },
                )
            )

        return goldens


@register_dataset("medconceptsqa_easy")
class MedConceptsQAEasyDataset(_MedConceptsQABase):
    """MedConceptsQA easy difficulty (icd10cm_sample)."""

    DIFFICULTY = "easy"
    VOCAB = "icd10cm_sample"
    TOTAL_SAMPLES = 2000

    @property
    def name(self) -> str:
        return "MedConceptsQA Easy"


@register_dataset("medconceptsqa_medium")
class MedConceptsQAMediumDataset(_MedConceptsQABase):
    """MedConceptsQA medium difficulty (icd10cm_sample)."""

    DIFFICULTY = "medium"
    VOCAB = "icd10cm_sample"
    TOTAL_SAMPLES = 2000

    @property
    def name(self) -> str:
        return "MedConceptsQA Medium"


@register_dataset("medconceptsqa_hard")
class MedConceptsQAHardDataset(_MedConceptsQABase):
    """MedConceptsQA hard difficulty (icd10cm_sample)."""

    DIFFICULTY = "hard"
    VOCAB = "icd10cm_sample"
    TOTAL_SAMPLES = 2000

    @property
    def name(self) -> str:
        return "MedConceptsQA Hard"


@register_dataset("medconceptsqa_atc_easy")
class MedConceptsQAATCEasyDataset(_MedConceptsQABase):
    """MedConceptsQA ATC easy difficulty."""

    DIFFICULTY = "easy"
    VOCAB = "atc"
    TOTAL_SAMPLES = 6436

    @property
    def name(self) -> str:
        return "MedConceptsQA ATC Easy"


@register_dataset("medconceptsqa_atc_medium")
class MedConceptsQAATCMediumDataset(_MedConceptsQABase):
    """MedConceptsQA ATC medium difficulty."""

    DIFFICULTY = "medium"
    VOCAB = "atc"
    TOTAL_SAMPLES = 6436

    @property
    def name(self) -> str:
        return "MedConceptsQA ATC Medium"


@register_dataset("medconceptsqa_atc_hard")
class MedConceptsQAATCHardDataset(_MedConceptsQABase):
    """MedConceptsQA ATC hard difficulty."""

    DIFFICULTY = "hard"
    VOCAB = "atc"
    TOTAL_SAMPLES = 5934

    @property
    def name(self) -> str:
        return "MedConceptsQA ATC Hard"
