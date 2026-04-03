"""
M-ARC Dataset Loader.

Medical Adaptation of ARC (AI2 Reasoning Challenge) questions.
Returns list of Golden objects for use with EvaluationDataset.
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


@register_dataset("m_arc")
class MARCDataset(GoldenDatasetBase):
    """
    M-ARC dataset loader.

    Loads medical adaptation of ARC benchmark from HuggingFace.
    Reference: https://huggingface.co/datasets/mkieffer/M-ARC
    """

    HUGGINGFACE_PATH = "mkieffer/M-ARC"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    TOTAL_SAMPLES = 100

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "M-ARC"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load M-ARC samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

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

            # Options should be a dict with letter keys
            if isinstance(opts, dict):
                options = {k: v for k, v in opts.items() if v and v.strip()}
            elif isinstance(opts, list):
                options = {}
                for i, v in enumerate(opts):
                    if v and v.strip():
                        letter = chr(ord("A") + i)
                        options[letter] = v.strip()
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
                        "source": "m_arc",
                        "options": options,
                    },
                )
            )

        return goldens
