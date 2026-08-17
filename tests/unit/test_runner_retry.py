"""Tests for candidate-inference retries and failure accounting."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from coeval.core.runner import EvalRunner


class _SequencedClient:
    """Inference fake that returns or raises each supplied outcome in order."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def generate(self, _messages: list[dict]) -> str:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        name="retry_dataset",
        goldens=[object()],
        get_generation_input=lambda _golden: [{"role": "user", "content": "q"}],
    )


@pytest.mark.asyncio
async def test_two_transient_failures_then_success_yields_successful_sample() -> None:
    """A third successful attempt must not be recorded as an inference failure."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), "recovered answer"]
    )
    runner = EvalRunner(client=client, concurrent_limit=1)

    predictions, metadata = await runner._generate_predictions(_dataset())

    assert client.calls == 3
    assert predictions == ["recovered answer"]
    assert metadata[0]["inference_failed"] is False
    assert metadata[0]["inference_error"] is None


@pytest.mark.asyncio
async def test_exhausted_attempts_record_final_exception() -> None:
    """Failure accounting must retain the exception from the final attempt."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), RuntimeError("final")]
    )
    runner = EvalRunner(client=client, concurrent_limit=1)

    predictions, metadata = await runner._generate_predictions(_dataset())

    assert client.calls == 3
    assert predictions == [""]
    assert metadata[0]["inference_failed"] is True
    assert metadata[0]["inference_error"] == "final"


@pytest.mark.asyncio
async def test_custom_attempt_count_retries_without_sleep_when_delay_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-delay retry must use the configured attempt count without sleeping."""
    client = _SequencedClient([RuntimeError("first"), "recovered answer"])
    runner = EvalRunner(
        client=client,
        concurrent_limit=1,
        inference_max_attempts=2,
        inference_retry_delay_s=0,
    )

    async def sleep_must_not_run(_delay: float) -> None:
        raise AssertionError("zero retry delay must not sleep")

    monkeypatch.setattr("coeval.core.runner.asyncio.sleep", sleep_must_not_run)

    assert (
        await runner._generate_with_retry([{"role": "user", "content": "q"}], 7)
        == "recovered answer"
    )
    assert client.calls == 2


@pytest.mark.asyncio
async def test_retry_delay_doubles_after_each_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries must wait longer after each consecutive transient failure."""
    client = _SequencedClient(
        [RuntimeError("first"), RuntimeError("second"), "recovered answer"]
    )
    runner = EvalRunner(
        client=client,
        concurrent_limit=1,
        inference_max_attempts=3,
        inference_retry_delay_s=0.25,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("coeval.core.runner.asyncio.sleep", sleep)

    result = await runner._generate_with_retry([{"role": "user", "content": "q"}], 7)

    assert result == "recovered answer"
    assert client.calls == 3
    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"inference_max_attempts": 0}, "inference_max_attempts"),
        ({"inference_retry_delay_s": -0.1}, "inference_retry_delay_s"),
    ],
)
def test_invalid_retry_configuration_is_rejected(
    kwargs: dict[str, float | int], message: str
) -> None:
    """Invalid retry bounds must fail before any inference is attempted."""
    with pytest.raises(ValueError, match=message):
        EvalRunner(client=_SequencedClient(["unused"]), **kwargs)


@pytest.mark.parametrize(
    ("metadata", "eval_lookup", "reason"),
    [
        (
            {
                "sample_id": 1,
                "inference_failed": True,
                "inference_error": "final",
            },
            {},
            "Skipped: inference failed",
        ),
        (
            {
                "sample_id": 1,
                "inference_failed": False,
                "inference_error": None,
            },
            {},
            "Missing eval result",
        ),
    ],
)
def test_unavailable_scores_are_none(
    metadata: dict, eval_lookup: dict, reason: str
) -> None:
    """Infrastructure failures must be excluded from score aggregation."""

    class Metric:
        __name__ = "metric"

    runner = EvalRunner(client=_SequencedClient(["unused"]))
    scores = runner._score_test_case(
        SimpleNamespace(additional_metadata={}), metadata, [Metric], eval_lookup
    )

    assert len(scores) == 1
    assert scores[0].score is None
    assert scores[0].passed is False
    assert scores[0].reason == reason
