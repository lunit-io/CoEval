"""Tests for the OpenAILike factory, message conversion, and PassthroughClient.

These cover the surface that replaced the old LLMClient wrapper. No network:
the factory is asserted on the constructed OpenAILike, and PassthroughClient is
driven through a stub llm.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import MessageRole

from coeval.clients.passthrough import PassthroughClient
from coeval.llm.config import LLMConfig
from coeval.llm.factory import create_llm_client, normalize_api_key
from coeval.llm.utils import to_chat_messages


class TestNormalizeApiKey:
    @pytest.mark.parametrize(
        "value", [None, "", "   ", "null", "NULL", "none", "empty"]
    )
    def test_placeholders_become_none(self, value: str | None) -> None:
        """Hydra resolves an unset ${oc.env:...,null} to the string 'null'."""
        assert normalize_api_key(value) is None

    def test_real_key_is_stripped(self) -> None:
        assert normalize_api_key("  sk-abc123  ") == "sk-abc123"


class TestCreateLLMClient:
    def test_maps_config_onto_openai_like(self) -> None:
        llm = create_llm_client(
            LLMConfig(
                api_base="http://shared-cluster-vm-026:9006/v1",
                model="my-model",
                api_key="sk-test",
                temperature=0.3,
                max_tokens=1234,
            )
        )
        assert llm.model == "my-model"
        assert llm.api_base == "http://shared-cluster-vm-026:9006/v1"
        assert llm.temperature == 0.3
        assert llm.max_tokens == 1234

    def test_context_window_defaults_when_unset(self) -> None:
        llm = create_llm_client(LLMConfig(api_base="http://x/v1", model="m"))
        assert llm.context_window == 4096

    def test_additional_kwargs_survive_the_hashable_round_trip(self) -> None:
        """LLMConfig stores additional_kwargs as a tuple; the factory must re-dict it."""
        llm = create_llm_client(
            LLMConfig(
                api_base="http://x/v1", model="m", additional_kwargs={"top_p": 1.0}
            )
        )
        assert llm.additional_kwargs["top_p"] == 1.0


class TestToChatMessages:
    def test_maps_roles(self) -> None:
        messages = to_chat_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert [m.role for m in messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert messages[1].content == "hi"

    def test_extra_fields_go_to_additional_kwargs(self) -> None:
        messages = to_chat_messages(
            [{"role": "tool", "content": "out", "tool_call_id": "call-1"}]
        )
        assert messages[0].additional_kwargs == {"tool_call_id": "call-1"}

    def test_none_valued_extras_are_dropped(self) -> None:
        messages = to_chat_messages([{"role": "user", "content": "hi", "name": None}])
        assert messages[0].additional_kwargs == {}

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            to_chat_messages([{"role": "wizard", "content": "hi"}])


class TestPassthroughClient:
    async def test_generate_returns_message_content(self) -> None:
        llm = SimpleNamespace(
            achat=AsyncMock(
                return_value=SimpleNamespace(
                    message=SimpleNamespace(content="the answer")
                )
            )
        )
        client = PassthroughClient(llm=llm)
        assert await client.generate([{"role": "user", "content": "q"}]) == "the answer"

    async def test_generate_forwards_full_message_list(self) -> None:
        """The client injects no system prompt of its own."""
        llm = SimpleNamespace(
            achat=AsyncMock(
                return_value=SimpleNamespace(message=SimpleNamespace(content="ok"))
            )
        )
        await PassthroughClient(llm=llm).generate(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "q"},
            ]
        )
        sent = llm.achat.await_args.args[0]
        assert [m.role for m in sent] == [MessageRole.SYSTEM, MessageRole.USER]

    async def test_none_content_becomes_empty_string(self) -> None:
        llm = SimpleNamespace(
            achat=AsyncMock(
                return_value=SimpleNamespace(message=SimpleNamespace(content=None))
            )
        )
        assert await PassthroughClient(llm=llm).generate([]) == ""

    def test_name_is_class_name(self) -> None:
        assert PassthroughClient(llm=SimpleNamespace()).name == "PassthroughClient"
