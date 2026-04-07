"""
PubMedQA Dataset Loader.

Biomedical yes/no/maybe classification task.
"""

import json
import urllib.request

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import MCQ_WITH_CONTEXT_TEMPLATE, GoldenDatasetBase


@register_dataset("pubmedqa")
class PubMedQADataset(GoldenDatasetBase):
    """
    PubMedQA dataset loader.

    Loads the official 500-sample test set from HuggingFace.

    Usage:
        dataset = PubMedQADataset(num_samples=100)

        # Goldens available via inherited property
        print(len(dataset.goldens))

        # After predictions, build test cases:
        dataset.build_test_cases(predictions)
        evaluate(test_cases=dataset.test_cases, metrics=metrics)
    """

    HUGGINGFACE_PATH = "qiaojin/PubMedQA"
    SUBSET = "pqa_labeled"
    SPLIT = "train"
    TEST_IDS_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
    OPTIONS = {"A": "Yes", "B": "No", "C": "Maybe"}
    TOTAL_SAMPLES = 500

    def __init__(self, num_samples: int | None = None, **kwargs):
        """Initialize dataset, loading up to ``num_samples`` examples."""
        super().__init__(goldens=self._load(num_samples), **kwargs)

    @property
    def name(self) -> str:
        return "PubMedQA"

    def _load(self, num_samples: int | None) -> list[Golden]:
        """Load PubMedQA samples as Golden objects."""
        dataset = load_dataset(
            self.HUGGINGFACE_PATH,
            name=self.SUBSET,
            split=self.SPLIT,
        )

        with urllib.request.urlopen(self.TEST_IDS_URL, timeout=30) as response:
            test_ids = set(json.loads(response.read().decode()))
        dataset = dataset.filter(lambda x: str(x["pubid"]) in test_ids)

        decision_to_letter = {"yes": "A", "no": "B", "maybe": "C"}
        goldens = []

        for row in dataset:
            if num_samples is not None and len(goldens) >= num_samples:
                break

            context_dict = row.get("context", {})
            labels = context_dict.get("labels", [])
            contexts = context_dict.get("contexts", [])
            formatted_context = "\n".join(
                f"{label}. {ctx}" for label, ctx in zip(labels, contexts, strict=False)
            )

            options_text = "\n".join(f"{k}. {v}" for k, v in self.OPTIONS.items())
            decision = row.get("final_decision", "").lower()
            answer_letter = decision_to_letter.get(decision, "A")

            question = MCQ_WITH_CONTEXT_TEMPLATE.format(
                context=formatted_context,
                question=row.get("question", ""),
                options=options_text,
            )

            goldens.append(
                Golden(
                    input=question,
                    expected_output=answer_letter,
                    context=[self.OPTIONS.get(answer_letter, "")],
                    retrieval_context=[formatted_context],
                    additional_metadata={"pubid": row.get("pubid")},
                )
            )

        return goldens
