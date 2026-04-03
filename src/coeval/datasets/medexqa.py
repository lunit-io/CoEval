"""
MedExQA Dataset Loader.

Medical explanation question answering across multiple specialties.
Returns list of Golden objects for use with EvaluationDataset.
"""

import urllib.request
from enum import StrEnum
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
from datasets import Dataset, concatenate_datasets

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


class MedExQASpecialty(StrEnum):
    """Specialty for MedExQA."""

    BIOMEDICAL_ENGINEER = "biomedical_engineer"
    CLINICAL_LABORATORY_SCIENTIST = "clinical_laboratory_scientist"
    CLINICAL_PSYCHOLOGIST = "clinical_psychologist"
    OCCUPATIONAL_THERAPIST = "occupational_therapist"
    SPEECH_PATHOLOGIST = "speech_pathologist"
    ALL = "all"


@register_dataset("medexqa")
class MedExQADataset(GoldenDatasetBase):
    """
    MedExQA dataset loader.

    Loads medical QA with explanations from HuggingFace.
    Reference: https://huggingface.co/datasets/bluesky333/MedExQA

    Args:
        specialty: Filter by specialty (default: all)
    """

    BASE_URL = "https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test"
    OPTION_LETTERS = ["A", "B", "C", "D"]
    TOTAL_SAMPLES = 940

    def __init__(
        self,
        num_samples: int | None = None,
        specialty: str = "all",
        **kwargs,
    ):
        self.specialty = MedExQASpecialty(specialty)
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MedExQA"

    def _download_tsv(self, specialty_name: str) -> Dataset | None:
        """Download and parse TSV file for a specialty."""
        url = f"{self.BASE_URL}/{specialty_name}_test.tsv"
        cache_dir = Path(gettempdir()) / "coeval" / "medexqa"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{specialty_name}_test.tsv"

        if not cache_path.exists():
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read().decode("utf-8")
                    cache_path.write_text(content)
            except Exception as e:
                print(f"Warning: Could not download {specialty_name}: {e}")
                return None

        try:
            df = pd.read_csv(
                cache_path,
                sep="\t",
                header=None,
                names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"],
            )
            df["specialty"] = specialty_name
            return Dataset.from_pandas(df, preserve_index=False)
        except Exception as e:
            print(f"Warning: Could not parse {specialty_name}: {e}")
            return None

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedExQA samples as Golden objects."""
        if self.specialty == MedExQASpecialty.ALL:
            specialties = [s for s in MedExQASpecialty if s != MedExQASpecialty.ALL]
        else:
            specialties = [self.specialty]

        datasets = []
        for sp in specialties:
            ds = self._download_tsv(sp.value)
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            return []

        combined = concatenate_datasets(datasets)

        goldens = []
        for row in combined:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()

            options = {
                "A": row.get("A", ""),
                "B": row.get("B", ""),
                "C": row.get("C", ""),
                "D": row.get("D", ""),
            }

            answer_letter = (row.get("answer") or "").strip().upper()

            if not question_text or answer_letter not in self.OPTION_LETTERS:
                continue

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
                        "source": "medexqa",
                        "specialty": row.get("specialty"),
                        "exp0": row.get("exp0"),
                        "exp1": row.get("exp1"),
                        "options": options,
                    },
                )
            )

        return goldens
