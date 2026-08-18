from llama_index.llms.openai_like import OpenAILike

from coeval.llm.config import LLMConfig


def normalize_api_key(api_key: str | None) -> str | None:
    """Map empty API keys, including Hydra's string ``"null"``, to ``None``."""
    if api_key is None or api_key.strip().lower() in ("null", "none", "empty", ""):
        return None
    return api_key.strip()


def create_llm_client(config: LLMConfig) -> OpenAILike:
    return OpenAILike(
        model=config.model,
        api_base=config.api_base,
        api_key=normalize_api_key(config.api_key),
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=config.max_retries,
        is_chat_model=True,
        is_function_calling_model=config.is_function_calling_model,
        context_window=config.context_window or 4096,
        additional_kwargs=config.get_additional_kwargs_dict(),
    )
