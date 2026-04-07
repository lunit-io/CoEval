"""
MetaMedQA Dataset Loader.

Meta-dataset combining multiple medical QA sources.
Returns list of Golden objects for use with EvaluationDataset.
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_PROMPT_TEMPLATE, GoldenDatasetBase


@register_dataset("metamedqa")
class MetaMedQADataset(GoldenDatasetBase):
    """
    MetaMedQA dataset loader.

    Loads medical QA meta-dataset from HuggingFace.
    Reference: https://huggingface.co/datasets/maximegmd/MetaMedQA

    Note: Answer is provided as text, needs to be matched to option letter.
    """

    HUGGINGFACE_PATH = "maximegmd/MetaMedQA"
    SPLIT = "test"
    TOTAL_SAMPLES = 1373

    def __init__(self, num_samples: int | None = None, **kwargs):
        """Initialize dataset, loading up to ``num_samples`` examples."""
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "MetaMedQA"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MetaMedQA samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

        goldens = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            question_text = row.get("question", "")
            opts = row.get("options", {})
            gold_text = row.get("answer", "")

            if not question_text or not opts:
                continue

            if isinstance(opts, dict):
                options = {k.upper(): v for k, v in opts.items() if v}
            else:
                continue

            # Find correct answer letter by matching gold_text to option value
            answer_letter = None
            for letter, opt_text in options.items():
                if (opt_text or "").strip().lower() == (
                    gold_text or ""
                ).strip().lower():
                    answer_letter = letter
                    break

            if answer_letter is None:
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
                        "source": "metamedqa",
                        "options": options,
                    },
                )
            )

        return goldens
