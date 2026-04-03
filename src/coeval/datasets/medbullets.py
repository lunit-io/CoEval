"""
Medbullets Dataset Loader.

USMLE-style step preparation questions.
Returns list of Golden objects for use with EvaluationDataset.
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


class _MedbulletsBase(GoldenDatasetBase):
    """Base class for Medbullets datasets."""

    HUGGINGFACE_PATH = "mkieffer/Medbullets"
    NUM_OPTIONS: int = 4
    OPTION_LETTERS = ["A", "B", "C", "D"]

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load Medbullets samples as Golden objects."""
        split = f"op{self.NUM_OPTIONS}_test"
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=split)

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = (row.get("question") or "").strip()
            opts = row.get("options", {})

            # Answer is a letter string
            answer_letter = (row.get("answer") or "").strip().upper()

            if not question_text:
                continue

            # Options is a dict with letter keys
            if isinstance(opts, dict):
                options = {
                    k: v for k, v in opts.items() if k in self.OPTION_LETTERS and v
                }
            else:
                continue

            # Validate answer
            if answer_letter not in self.OPTION_LETTERS or answer_letter not in options:
                continue

            # Format options text
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
                        "source": "medbullets",
                        "num_options": self.NUM_OPTIONS,
                        "options": options,
                    },
                )
            )

        return goldens


@register_dataset("medbullets_4options")
class Medbullets4OptionsDataset(_MedbulletsBase):
    """Medbullets with 4 answer options."""

    NUM_OPTIONS = 4
    TOTAL_SAMPLES = 308

    @property
    def name(self) -> str:
        return "Medbullets 4 Options"


@register_dataset("medbullets_5options")
class Medbullets5OptionsDataset(_MedbulletsBase):
    """Medbullets with 5 answer options."""

    NUM_OPTIONS = 5
    OPTION_LETTERS = ["A", "B", "C", "D", "E"]
    TOTAL_SAMPLES = 308

    @property
    def name(self) -> str:
        return "Medbullets 5 Options"
