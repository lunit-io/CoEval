"""
MedQA Dataset Loader.

USMLE-style medical licensing exam questions.
Returns list of Golden objects for use with EvaluationDataset.
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


@register_dataset("medqa")
class MedQADataset(GoldenDatasetBase):
    """
    MedQA dataset loader.

    Loads USMLE-style medical exam questions from HuggingFace.
    Returns list of Golden objects.
    """

    HUGGINGFACE_PATH = "GBaker/MedQA-USMLE-4-options"
    SPLIT = "test"
    OPTION_LETTERS = ["A", "B", "C", "D"]
    TOTAL_SAMPLES = 1273

    def __init__(self, num_samples: int | None = None, **kwargs):
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MedQA"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedQA samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = row.get("question", "")

            # Parse options from dict
            opts = row.get("options", {})
            if isinstance(opts, dict):
                options = {k.upper(): v for k, v in opts.items()}
            elif isinstance(opts, list):
                options = dict(zip(self.OPTION_LETTERS, opts, strict=False))
            else:
                options = {}

            # Get correct answer - may be int index, letter, or full answer text
            answer = row.get("answer", row.get("answer_idx", 0))
            if isinstance(answer, int):
                answer_letter = self.OPTION_LETTERS[answer]
            elif (
                len(str(answer).strip()) == 1
                and str(answer).strip().upper() in self.OPTION_LETTERS
            ):
                answer_letter = str(answer).strip().upper()
            else:
                # Full answer text - find matching option letter
                answer_text_normalized = str(answer).strip().lower()
                answer_letter = None
                for letter, opt_text in options.items():
                    if opt_text.strip().lower() == answer_text_normalized:
                        answer_letter = letter
                        break
                if answer_letter is None:
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
                        "source": "medqa",
                        "options": options,
                    },
                )
            )

        return goldens
