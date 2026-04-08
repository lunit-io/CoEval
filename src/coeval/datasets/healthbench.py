"""
HealthBench Dataset Loaders.

Loads HealthBench subsets from OpenAI's public Azure blob storage.
Each example contains multi-turn conversation prompts with rubric items
for LLM-as-judge grading.

Available subsets:
    - Full: 5,000 examples with diverse rubric criteria
    - Consensus: 3,671 examples with 34 consensus criteria

Reference:
    - https://github.com/openai/simple-evals
    - https://openai.com/index/healthbench/
"""

import hashlib
import json
import logging
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from coeval.core.registry import register_dataset
from coeval.core.types import ConversationalGolden, Turn
from coeval.datasets.base import MultiTurnDatasetBase

logger = logging.getLogger(__name__)

FULL_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
CONSENSUS_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "coeval"


class _HealthBenchDatasetBase(MultiTurnDatasetBase):
    """Shared loader logic for all HealthBench subsets.

    Downloads JSONL from Azure blob and maps each example to a ConversationalGolden:
    - scenario: prompt_id (unique identifier)
    - turns: user/assistant turns from conversation (Turn only supports these roles)
    - additional_metadata: rubrics, example_tags, system_prompt (if present)

    build_test_cases() produces ConversationalTestCase objects via
    MultiTurnDatasetBase, appending the model prediction as the final turn.
    """

    _URL: str  # override in subclass
    _LABEL: str  # human-readable label for log messages

    def __init__(
        self,
        num_samples: int | None = None,
        sample_ratio: float | None = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(goldens=self._load(num_samples, sample_ratio, seed), **kwargs)

    def _load(
        self,
        num_samples: int | None,
        sample_ratio: float | None,
        seed: int,
    ) -> list[ConversationalGolden]:
        """Download and parse HealthBench JSONL.

        Args:
            num_samples: Hard cap — take first N samples (simple truncation).
                Takes priority over ``sample_ratio`` when both are set.
            sample_ratio: Stratified sampling ratio (0.0–1.0) by ``theme:*``
                tag, preserving theme distribution.
            seed: RNG seed for reproducible stratified sampling.
        """
        lines = self._read_or_download(self._URL).strip().split("\n")

        all_goldens: list[ConversationalGolden] = []
        for line in lines:
            row = json.loads(line)
            golden = self._build_golden(row)
            if golden is not None:
                all_goldens.append(golden)

        if num_samples is not None:
            goldens = all_goldens[:num_samples]
        elif sample_ratio is not None:
            goldens = self._stratified_sample(all_goldens, sample_ratio, seed)
        else:
            goldens = all_goldens

        logger.info(f"Loaded {len(goldens)} HealthBench {self._LABEL} samples")
        return goldens

    @staticmethod
    def _read_or_download(url: str) -> str:
        """Return file contents, downloading and caching if needed."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        filename = url.rsplit("/", 1)[-1] or url_hash
        cache_path = _CACHE_DIR / f"{filename}.{url_hash}"

        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return cache_path.read_text()

        logger.info(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read().decode("utf-8")

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(data)
        logger.info(f"Cached to {cache_path}")
        return data

    @staticmethod
    def _get_theme(golden: ConversationalGolden) -> str:
        """Extract ``theme:*`` tag from a golden's metadata."""
        tags = (golden.additional_metadata or {}).get("example_tags", [])
        for tag in tags:
            if tag.startswith("theme:"):
                return tag
        return "theme:unknown"

    @classmethod
    def _stratified_sample(
        cls,
        goldens: list[ConversationalGolden],
        ratio: float,
        seed: int,
    ) -> list[ConversationalGolden]:
        """Stratified sampling by theme, preserving distribution."""
        by_theme: dict[str, list[ConversationalGolden]] = defaultdict(list)
        for g in goldens:
            by_theme[cls._get_theme(g)].append(g)

        rng = random.Random(seed)
        sampled: list[ConversationalGolden] = []
        for theme in sorted(by_theme):
            group = by_theme[theme]
            n = max(1, round(len(group) * ratio))
            sampled.extend(rng.sample(group, min(n, len(group))))
        return sampled

    @staticmethod
    def _build_golden(row: dict) -> ConversationalGolden | None:
        """Build a ConversationalGolden from a HealthBench JSONL row.

        Each row has:
            - prompt: list of message dicts (conversation history)
            - rubrics: list of {criterion, points, tags}
            - example_tags: list of str
            - prompt_id: unique identifier
        """
        prompt_messages = row.get("prompt", [])
        rubrics = row.get("rubrics", [])
        example_tags = row.get("example_tags", [])
        prompt_id = row.get("prompt_id", "")

        if not prompt_messages:
            return None

        # Turn only supports "user" | "assistant" roles
        turns = [
            Turn(role=msg["role"], content=msg.get("content", ""))
            for msg in prompt_messages
            if msg.get("role") in ("user", "assistant")
        ]

        # Extract system prompt if present
        system_prompt = next(
            (
                msg.get("content", "")
                for msg in prompt_messages
                if msg.get("role") == "system"
            ),
            None,
        )

        return ConversationalGolden(
            scenario=prompt_id or "healthbench",
            turns=turns,
            additional_metadata={
                "system_prompt": system_prompt,
                "rubrics": rubrics,
                "example_tags": example_tags,
                "prompt_id": prompt_id,
            },
        )


@register_dataset("healthbench_consensus")
class HealthBenchConsensusDataset(_HealthBenchDatasetBase):
    """HealthBench Consensus subset (3,671 examples, 34 consensus criteria).

    Usage:
        dataset = HealthBenchConsensusDataset(num_samples=5)
        print(len(dataset.goldens))
    """

    TOTAL_SAMPLES = 3671
    _URL = CONSENSUS_URL
    _LABEL = "Consensus"

    @property
    def name(self) -> str:
        return "HealthBenchConsensus"


@register_dataset("healthbench_full")
class HealthBenchFullDataset(_HealthBenchDatasetBase):
    """HealthBench Full dataset (5,000 examples with diverse rubric criteria).

    Usage:
        dataset = HealthBenchFullDataset(num_samples=5)
        print(len(dataset.goldens))
    """

    TOTAL_SAMPLES = 5000
    _URL = FULL_URL
    _LABEL = "Full"

    @property
    def name(self) -> str:
        return "HealthBenchFull"
