"""
Base dataset classes for evaluation.

Extends DeepEval's EvaluationDataset with additional functionality.
"""

from abc import abstractmethod
from typing import Any

from coeval.core.types import (
    AnyGolden,
    ChatMessages,
    ConversationalTestCase,
    EvaluationDataset,
    LLMTestCase,
    Turn,
)

# Default system prompt for single-turn datasets without their own.
DEFAULT_SYSTEM_PROMPT = ""

# Default prompt templates
MCQ_PROMPT_TEMPLATE = """Select the best answer.

Question: {question}
{options}
Answer: """

MCQ_WITH_CONTEXT_TEMPLATE = """Select the best answer.

Context: {context}

Question: {question}
{options}
Answer: """


class GoldenDatasetBase(EvaluationDataset):
    """
    Base class extending EvaluationDataset with dataset name and test case building.

    Inherits from DeepEval's EvaluationDataset:
        - goldens: List[Golden] property
        - test_cases: List[LLMTestCase] property
        - add_test_case(): Add test case to dataset

    Subclasses must implement:
        - name: Dataset identifier

    Usage:
        dataset = MyDataset(num_samples=100, system_prompt="You are a helpful assistant.")

        # Goldens are available via inherited property
        print(dataset.goldens)

        # After generating predictions, build test cases:
        dataset.build_test_cases(predictions)

        # Test cases now available:
        evaluate(test_cases=dataset.test_cases, metrics=metrics)
    """

    def __init__(self, *, system_prompt: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        ...

    @property
    def is_multi_turn(self) -> bool:
        """Whether this dataset provides multi-turn chat inputs."""
        return False

    def __len__(self) -> int:
        """Return the number of golden samples in this dataset."""
        return len(self.goldens)

    def get_generation_input(self, golden: AnyGolden) -> ChatMessages:
        """Return chat messages to send to the inference client.

        Default: ``[system, user]`` using ``self.system_prompt`` and
        ``golden.input``.  Subclasses with custom prompts (ADR, ED, etc.)
        should override this method.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": golden.input})  # type: ignore[union-attr]
        return messages

    def build_test_cases(self, predictions: list[str]) -> list[LLMTestCase]:
        """
        Build LLMTestCase objects from predictions and add to dataset.

        Args:
            predictions: List of actual_output strings, same order as goldens

        Returns:
            List of LLMTestCase objects (also accessible via self.test_cases)
        """
        if len(predictions) != len(self.goldens):
            raise ValueError(
                f"Predictions length ({len(predictions)}) != goldens length ({len(self.goldens)})"
            )

        built: list[LLMTestCase] = []
        for idx, (golden, pred) in enumerate(
            zip(self.goldens, predictions, strict=True)
        ):
            metadata = dict(golden.additional_metadata or {})
            metadata["_sample_id"] = idx
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=pred,
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=golden.retrieval_context,
                additional_metadata=metadata,
            )
            self.add_test_case(test_case)
            built.append(test_case)
        return built


class MultiTurnDatasetBase(GoldenDatasetBase):
    """Base class for multi-turn chat evaluation datasets.

    Subclasses provide ``ConversationalGolden`` objects with native ``turns``
    (user/assistant only). System prompts are stored as a string in
    ``additional_metadata["system_prompt"]`` when present.

    ``get_generation_input()`` composes messages from ``system_prompt`` +
    ``turns``.  ``build_test_cases()`` returns ``ConversationalTestCase``
    objects with the model prediction appended as the final turn.
    """

    @property
    def is_multi_turn(self) -> bool:
        return True

    def get_generation_input(self, golden: AnyGolden) -> ChatMessages:
        """Compose messages from system_prompt (metadata) + turns.

        Prepends system message from metadata, falling back to
        ``DEFAULT_SYSTEM_PROMPT`` when absent, then appends each turn.
        """
        metadata = golden.additional_metadata or {}
        system_prompt = metadata.get("system_prompt")
        if system_prompt is None:
            system_prompt = self.system_prompt
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for turn in golden.turns or []:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def build_test_cases(self, predictions: list[str]) -> list[ConversationalTestCase]:
        """Build ConversationalTestCase objects from predictions.

        Args:
            predictions: Model responses, same order as goldens.

        Returns:
            List of ConversationalTestCase with full conversation turns.
        """
        if len(predictions) != len(self.goldens):
            raise ValueError(
                f"Predictions length ({len(predictions)}) "
                f"!= goldens length ({len(self.goldens)})"
            )

        built: list[ConversationalTestCase] = []
        for idx, (golden, pred) in enumerate(
            zip(self.goldens, predictions, strict=True)
        ):
            metadata = dict(golden.additional_metadata or {})
            if metadata.get("system_prompt") is None:
                metadata["system_prompt"] = self.system_prompt
            metadata["_sample_id"] = idx

            turns = list(golden.turns or [])
            turns.append(Turn(role="assistant", content=pred))

            test_case = ConversationalTestCase(
                turns=turns,
                additional_metadata=metadata,
            )
            built.append(test_case)
        return built
