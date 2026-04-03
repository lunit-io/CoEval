"""
MedHallu Dataset Loader.

Binary classification benchmark for detecting medical hallucinations in LLM responses.
Follows the official evaluation protocol: randomly select ground_truth or hallucinated_answer
per row, producing one test case per sample.

Reference:
    - https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu
    - https://arxiv.org/abs/2502.14302
    - https://github.com/MedHallu/MedHallu
"""

import random
from typing import Literal

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import (
    MCQ_PROMPT_TEMPLATE,
    MCQ_WITH_CONTEXT_TEMPLATE,
    GoldenDatasetBase,
)


@register_dataset("medhallu")
class MedHalluDataset(GoldenDatasetBase):
    """
    MedHallu dataset loader.

    Binary classification task: determine if a medical answer is hallucinated.
    Labels: "0" (factual) / "1" (hallucinated)

    Following the official evaluation protocol:
        - Each row randomly selects either ground_truth or hallucinated_answer
        - One test case per row (balanced by random selection)
        - Overall F1 as primary metric
        - Difficulty-level F1 reported in metadata

    Subsets:
        - pqa_labeled: 1,000 human-annotated samples (primary evaluation set)
        - pqa_artificial: 9,000 AI-generated samples
    """

    HUGGINGFACE_PATH = "UTAustin-AIHealth/MedHallu"

    def __init__(
        self,
        subset: Literal["pqa_labeled", "pqa_artificial"] = "pqa_labeled",
        split: str = "train",
        use_knowledge: bool = True,
        seed: int = 42,
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        self.subset = subset
        self.split = split
        self.use_knowledge = use_knowledge
        self.seed = seed
        self._num_samples = num_samples
        super().__init__(goldens=self._load(), **kwargs)

    @property
    def name(self) -> str:
        return "medhallu"

    def _load(self) -> list[Golden]:
        dataset = load_dataset(
            self.HUGGINGFACE_PATH,
            name=self.subset,
            split=self.split,
        )

        rng = random.Random(self.seed)

        goldens: list[Golden] = []
        for row in dataset:
            if self._num_samples is not None and len(goldens) >= self._num_samples:
                break

            golden = self._build_golden(row, rng)
            if golden is not None:
                goldens.append(golden)

        return goldens

    def _build_golden(self, row: dict, rng: random.Random) -> Golden | None:
        question = row.get("Question", "")
        knowledge_raw = row.get("Knowledge", [])
        ground_truth = row.get("Ground Truth", "")
        hallucinated = row.get("Hallucinated Answer", "")
        difficulty = row.get("Difficulty Level", "")
        category = row.get("Category of Hallucination", "")

        if not question or not ground_truth or not hallucinated:
            return None

        knowledge_text = (
            "\n".join(knowledge_raw)
            if isinstance(knowledge_raw, list)
            else str(knowledge_raw)
        )

        # Official protocol: randomly select ground_truth or hallucinated_answer
        use_hallucinated = rng.randint(0, 1)
        if use_hallucinated:
            answer = hallucinated
            label = "1"  # hallucinated
        else:
            answer = ground_truth
            label = "0"  # factual

        # Format as MCQ: "Is the following answer factual or hallucinated?"
        mcq_question = (
            f"{question}\n\n"
            f"Answer to evaluate: {answer}\n\n"
            f"Is the above answer factual or does it contain hallucinated information?"
        )
        options = "A. Factual (0)\nB. Hallucinated (1)"

        if self.use_knowledge:
            prompt = MCQ_WITH_CONTEXT_TEMPLATE.format(
                context=knowledge_text,
                question=mcq_question,
                options=options,
            )
        else:
            prompt = MCQ_PROMPT_TEMPLATE.format(
                question=mcq_question,
                options=options,
            )

        # Map label to MCQ answer: "0" -> "A", "1" -> "B"
        expected = "A" if label == "0" else "B"

        return Golden(
            input=prompt,
            expected_output=expected,
            context=[knowledge_text],
            additional_metadata={
                "difficulty_level": difficulty,
                "category_of_hallucination": category,
                "answer_type": "hallucinated" if use_hallucinated else "ground_truth",
                "eval_type": "classification",
                "labels": ["A", "B"],
            },
        )
