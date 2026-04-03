"""Tests for generate() dispatch in runner — datasets always return list[dict]."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from deepeval.dataset import Golden

from coeval.core.runner import EvalRunner
from coeval.datasets.base import DEFAULT_SYSTEM_PROMPT


def _dataset_with(golden: Golden, get_generation_input=None) -> SimpleNamespace:
    """Build a minimal dataset-like object.

    If get_generation_input is provided, use it; otherwise use the base class
    default: [system(DEFAULT), user(golden.input)].
    """
    if get_generation_input is None:

        def get_generation_input(g: Golden) -> list[dict]:
            return [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": g.input},
            ]

    return SimpleNamespace(
        name="test_dataset",
        goldens=[golden],
        get_generation_input=get_generation_input,
    )


@pytest.mark.asyncio
async def test_generate_receives_custom_messages() -> None:
    """When dataset overrides get_generation_input, runner forwards the result."""
    custom_messages = [
        {"role": "system", "content": "You are a doctor."},
        {"role": "user", "content": "I have a fever."},
    ]
    golden = Golden(input="unused", expected_output="")
    client = SimpleNamespace(
        generate=AsyncMock(return_value='{"answer": "chat_mode"}'),
    )

    def custom_input(_g: Golden) -> list[dict]:
        return custom_messages

    runner = EvalRunner(client=client, concurrent_limit=1)
    predictions, _ = await runner._generate_predictions(
        _dataset_with(golden, get_generation_input=custom_input)
    )

    assert predictions == ["chat_mode"]
    client.generate.assert_awaited_once_with(custom_messages)


@pytest.mark.asyncio
async def test_generate_receives_composed_messages_for_single_turn() -> None:
    """Default get_generation_input composes [system, user] messages."""
    golden = Golden(
        input="What is diabetes?",
        expected_output="",
    )
    client = SimpleNamespace(
        generate=AsyncMock(return_value='{"answer": "response"}'),
    )

    runner = EvalRunner(client=client, concurrent_limit=1)
    predictions, _ = await runner._generate_predictions(_dataset_with(golden))

    assert predictions == ["response"]
    client.generate.assert_awaited_once_with(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "What is diabetes?"},
        ]
    )


@pytest.mark.asyncio
async def test_generate_uses_default_system_prompt() -> None:
    """Default get_generation_input uses DEFAULT_SYSTEM_PROMPT."""
    golden = Golden(
        input="What is aspirin?",
        expected_output="",
    )
    client = SimpleNamespace(
        generate=AsyncMock(return_value='{"answer": "aspirin_info"}'),
    )

    runner = EvalRunner(client=client, concurrent_limit=1)
    predictions, _ = await runner._generate_predictions(_dataset_with(golden))

    assert predictions == ["aspirin_info"]
    client.generate.assert_awaited_once_with(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "What is aspirin?"},
        ]
    )
