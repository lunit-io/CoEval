from abc import ABC, abstractmethod
from typing import Any

from llama_index.llms.openai_like import OpenAILike


class EvaluationLLMClient(ABC):
    """Online LLM-only eval client: ``messages -> answer``."""

    def __init__(self, llm: OpenAILike) -> None:
        self.llm = llm

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]]) -> str:
        """Run inference for one sample; return its answer."""
        ...
