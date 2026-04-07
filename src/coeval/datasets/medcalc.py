"""
MedCalc-Bench Dataset Loader.

Clinical calculation benchmark — tests ability to extract patient parameters
and apply medical formulas (Cockcroft-Gault, APACHE II, etc.).

Evaluation: numeric answer within tolerance range [Lower Limit, Upper Limit].

Reference:
    - https://huggingface.co/datasets/ncbi/MedCalc-Bench
    - https://arxiv.org/abs/2406.12036
"""

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import GoldenDatasetBase

MEDCALC_PROMPT_TEMPLATE = """{patient_note}

Question: {question}
Answer: """


@register_dataset("medcalc")
class MedCalcDataset(GoldenDatasetBase):
    """
    MedCalc-Bench dataset loader.

    Loads clinical calculation problems from HuggingFace (test split = 1,100 samples).
    Each sample has a patient note, question, ground truth numeric answer,
    and tolerance range for evaluation.
    """

    HUGGINGFACE_PATH = "ncbi/MedCalc-Bench"
    SPLIT = "test"
    TOTAL_SAMPLES = 1100

    def __init__(self, num_samples: int | None = None, **kwargs):
        """Initialize dataset, loading up to ``num_samples`` examples."""
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "medcalc"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load MedCalc-Bench samples as Golden objects."""
        dataset = load_dataset(self.HUGGINGFACE_PATH, split=self.SPLIT)

        goldens: list[Golden] = []
        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            patient_note = row.get("Patient Note", "")
            question = row.get("Question", "")
            ground_truth = row.get("Ground Truth Answer", "")
            lower_limit = row.get("Lower Limit", "")
            upper_limit = row.get("Upper Limit", "")
            calculator_name = row.get("Calculator Name", "")
            output_type = row.get("Output Type", "")
            explanation = row.get("Ground Truth Explanation", "")

            if not question or not ground_truth:
                continue

            prompt = MEDCALC_PROMPT_TEMPLATE.format(
                patient_note=patient_note,
                question=question,
            )

            goldens.append(
                Golden(
                    input=prompt,
                    expected_output=str(ground_truth),
                    context=[explanation] if explanation else [],
                    additional_metadata={
                        "calculator_name": calculator_name,
                        "output_type": output_type,
                        "lower_limit": str(lower_limit),
                        "upper_limit": str(upper_limit),
                    },
                )
            )

        return goldens
