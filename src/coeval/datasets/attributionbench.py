"""
AttributionBench Dataset Loader.

Binary classification benchmark for evaluating LLM attribution/citation quality.

Reference:
    - https://huggingface.co/datasets/osunlp/AttributionBench
    - https://arxiv.org/abs/2402.15089
    - https://github.com/OSU-NLP-Group/AttributionBench
"""

from typing import Literal

from datasets import load_dataset

from coeval.core.registry import register_dataset
from coeval.core.types import Golden
from coeval.datasets.base import GoldenDatasetBase

# Attribution label mapping
ATTRIBUTION_LABELS: dict[str, str] = {
    "attributable": "attributable",
    "not attributable": "not attributable",
    "unattributable": "not attributable",
}

# 4 In-Distribution (ID) source datasets for ID-Avg F1
ID_SOURCE_DATASETS: list[str] = [
    "ExpertQA",
    "Stanford-GenSearch",
    "AttributedQA",
    "LFQA",
]

ATTRIBUTION_PROMPT_TEMPLATE = """### Instruction:
Please verify whether the reference supports the claim. Options: 'attributable' or 'not attributable'.

###Input:
Claim: {claim}

References: {references}

Is this claim attributable to the references? Answer with either "attributable" or "not attributable".

### Output:"""

# Per-source sample counts (ID test set)
SOURCE_SAMPLE_COUNTS: dict[str, int] = {
    "ExpertQA": 612,
    "Stanford-GenSearch": 600,
    "AttributedQA": 230,
    "LFQA": 168,
}


@register_dataset("attributionbench")
class AttributionBenchDataset(GoldenDatasetBase):
    """
    AttributionBench dataset loader.

    Binary classification task: determine if a claim is attributable to references.
    Labels: "attributable" / "not attributable"

    Subsets:
        - overall_balanced: Balanced train/test split across all sources
        - subset_balanced: Per-source balanced splits

    Source datasets (ID - for ID-Avg F1):
        - ExpertQA (612 samples)
        - Stanford-GenSearch (600 samples)
        - AttributedQA (230 samples)
        - LFQA (168 samples)

    Source datasets (OOD):
        - BEGIN, HAGRID

    Usage:
        dataset = AttributionBenchDataset(num_samples=100)
        print(len(dataset.goldens))

        dataset.build_test_cases(predictions)
        evaluate(test_cases=dataset.test_cases, metrics=metrics)
    """

    HUGGINGFACE_PATH = "osunlp/AttributionBench"
    TOTAL_SAMPLES = 1610

    def __init__(
        self,
        subset: Literal["overall_balanced", "subset_balanced"] = "overall_balanced",
        split: Literal["test", "test_ood"] = "test",
        source: str | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize AttributionBench dataset.

        Args:
            subset: Dataset configuration ('overall_balanced' or 'subset_balanced').
            split: Data split ('test' for ID, 'test_ood' for OOD).
            source: Filter by source dataset (e.g., 'ExpertQA'). None for all.
            num_samples: Limit samples (None = all).
        """
        self.subset = subset
        self.split = split
        self.source = source
        self._num_samples = num_samples
        super().__init__(goldens=self._load(), **kwargs)

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        base = "attributionbench"
        suffix = "_ood" if self.split == "test_ood" else ""
        source_suffix = (
            f"_{self.source.lower().replace('-', '_')}" if self.source else ""
        )
        return f"{base}{suffix}{source_suffix}"

    def _load(self) -> list[Golden]:
        """Load AttributionBench samples as Golden objects."""
        dataset = load_dataset(
            self.HUGGINGFACE_PATH,
            name=self.subset,
            split=self.split,
        )

        if self.source:
            dataset = dataset.filter(lambda x: x["src_dataset"] == self.source)

        goldens: list[Golden] = []
        for row in dataset:
            if self._num_samples is not None and len(goldens) >= self._num_samples:
                break

            golden = self._build_golden(row)
            if golden is not None:
                goldens.append(golden)

        return goldens

    def _build_golden(self, row: dict) -> Golden | None:
        """Build Golden from AttributionBench row."""
        label_raw = row.get("attribution_label", "")
        label = ATTRIBUTION_LABELS.get(label_raw.lower().strip())
        if not label:
            return None

        question = row.get("question", "")
        claim = row.get("claim", row.get("claim_raw_string", ""))
        references = row.get("references", [])
        if not claim or not references:
            return None

        references_text = "\n".join(references)

        prompt = ATTRIBUTION_PROMPT_TEMPLATE.format(
            claim=claim,
            references=references_text,
        )

        return Golden(
            input=prompt,
            expected_output=label,
            context=[claim],
            retrieval_context=references,
            additional_metadata={
                "source_dataset": row.get("src_dataset", ""),
                "prompt_id": row.get("id", ""),
                "question_raw": question,
                "claim_raw": claim,
                "citation_links": row.get("citation_links", []),
                "eval_type": "classification",
                "labels": ["attributable", "not attributable"],
            },
        )


@register_dataset("attributionbench_expertqa")
class AttributionBenchExpertQADataset(AttributionBenchDataset):
    """AttributionBench ExpertQA subset (612 samples)."""

    TOTAL_SAMPLES = SOURCE_SAMPLE_COUNTS["ExpertQA"]

    def __init__(self, num_samples: int | None = None, **kwargs) -> None:
        super().__init__(source="ExpertQA", num_samples=num_samples, **kwargs)

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return "attributionbench_expertqa"


@register_dataset("attributionbench_stanford_gensearch")
class AttributionBenchStanfordGenSearchDataset(AttributionBenchDataset):
    """AttributionBench Stanford-GenSearch subset (600 samples)."""

    TOTAL_SAMPLES = SOURCE_SAMPLE_COUNTS["Stanford-GenSearch"]

    def __init__(self, num_samples: int | None = None, **kwargs) -> None:
        super().__init__(source="Stanford-GenSearch", num_samples=num_samples, **kwargs)

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return "attributionbench_stanford_gensearch"


@register_dataset("attributionbench_attributedqa")
class AttributionBenchAttributedQADataset(AttributionBenchDataset):
    """AttributionBench AttributedQA subset (230 samples)."""

    TOTAL_SAMPLES = SOURCE_SAMPLE_COUNTS["AttributedQA"]

    def __init__(self, num_samples: int | None = None, **kwargs) -> None:
        super().__init__(source="AttributedQA", num_samples=num_samples, **kwargs)

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return "attributionbench_attributedqa"


@register_dataset("attributionbench_lfqa")
class AttributionBenchLFQADataset(AttributionBenchDataset):
    """AttributionBench LFQA subset (168 samples)."""

    TOTAL_SAMPLES = SOURCE_SAMPLE_COUNTS["LFQA"]

    def __init__(self, num_samples: int | None = None, **kwargs) -> None:
        super().__init__(source="LFQA", num_samples=num_samples, **kwargs)

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return "attributionbench_lfqa"
