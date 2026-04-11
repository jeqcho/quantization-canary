"""Unit tests for src/probe.py.

Uses a mocked Anthropic client so no real API calls are made. Tests cover
the happy path, API call parameters, cost computation, wall-clock
timestamps, and retry behavior.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import anthropic
import pytest

from src.probe import ProbeResult, call_opus, call_opus_parallel


# ---------- helpers ----------


def _mock_response(text="hello", input_tokens=5, output_tokens=10):
    """Build a fake anthropic Message response object for tests."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class _FakeRateLimit(anthropic.RateLimitError):
    """A RateLimitError instance bypassing the real SDK's __init__ requirements."""

    def __init__(self):
        pass


class _FakeBadRequest(anthropic.BadRequestError):
    def __init__(self):
        pass


def _nop_sleep(_seconds):
    """Sleep replacement so retry tests don't actually wait."""
    pass


# ---------- happy path ----------


def test_call_opus_returns_probe_result_with_expected_fields():
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        text="Paris", input_tokens=12, output_tokens=3
    )

    result = call_opus(
        "What is the capital of France?",
        max_tokens=50,
        client=client,
        sleep_fn=_nop_sleep,
    )

    assert isinstance(result, ProbeResult)
    assert result.text == "Paris"
    assert result.input_tokens == 12
    assert result.output_tokens == 3


def test_call_opus_invokes_api_with_opus_4_6_at_temperature_zero():
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    call_opus("hi", max_tokens=50, client=client, sleep_fn=_nop_sleep)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-6"
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_call_opus_computes_cost_at_opus_rates():
    client = MagicMock()
    # 1M input + 1M output at $15 + $75 per M = $90 total
    client.messages.create.return_value = _mock_response(
        input_tokens=1_000_000, output_tokens=1_000_000
    )

    result = call_opus("hi", max_tokens=50, client=client, sleep_fn=_nop_sleep)

    assert result.cost_usd == pytest.approx(90.0)


def test_call_opus_records_wall_clock_timestamps_in_utc():
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    before = datetime.now(timezone.utc)
    result = call_opus("hi", max_tokens=50, client=client, sleep_fn=_nop_sleep)
    after = datetime.now(timezone.utc)

    assert before <= result.started_at <= result.finished_at <= after
    assert result.started_at.tzinfo is timezone.utc
    assert result.finished_at.tzinfo is timezone.utc


# ---------- retry behavior ----------


def test_call_opus_retries_on_rate_limit_then_succeeds():
    client = MagicMock()
    client.messages.create.side_effect = [
        _FakeRateLimit(),
        _mock_response(text="recovered"),
    ]

    result = call_opus("hi", max_tokens=50, client=client, sleep_fn=_nop_sleep)

    assert result.text == "recovered"
    assert client.messages.create.call_count == 2


def test_call_opus_gives_up_after_max_retries():
    client = MagicMock()
    client.messages.create.side_effect = [_FakeRateLimit()] * 10

    with pytest.raises(anthropic.RateLimitError):
        call_opus(
            "hi",
            max_tokens=50,
            client=client,
            sleep_fn=_nop_sleep,
            max_retries=3,
        )

    assert client.messages.create.call_count == 3


def test_call_opus_does_not_retry_on_bad_request():
    client = MagicMock()
    client.messages.create.side_effect = [_FakeBadRequest()]

    with pytest.raises(anthropic.BadRequestError):
        call_opus("hi", max_tokens=50, client=client, sleep_fn=_nop_sleep)

    # 4xx errors should fail fast, not retry
    assert client.messages.create.call_count == 1


# ---------- call_opus_parallel ----------


def test_call_opus_parallel_returns_results_in_input_order():
    client = MagicMock()
    # Each call returns a response whose text encodes its prompt so we
    # can verify ordering.
    def make_response(prompt):
        return _mock_response(text=f"resp-{prompt}", input_tokens=5, output_tokens=10)

    client.messages.create.side_effect = lambda **kwargs: make_response(
        kwargs["messages"][0]["content"]
    )

    requests = [("prompt-A", 100), ("prompt-B", 100), ("prompt-C", 100)]
    results = call_opus_parallel(
        requests,
        client=client,
        max_workers=4,
        sleep_fn=_nop_sleep,
    )

    assert len(results) == 3
    assert [r.text for r in results] == ["resp-prompt-A", "resp-prompt-B", "resp-prompt-C"]


def test_call_opus_parallel_calls_on_progress_for_each_completion():
    client = MagicMock()
    client.messages.create.return_value = _mock_response()
    progress_calls = []

    def on_progress(completed, total, result):
        progress_calls.append((completed, total))

    requests = [("a", 100), ("b", 100), ("c", 100)]
    call_opus_parallel(
        requests,
        client=client,
        max_workers=4,
        sleep_fn=_nop_sleep,
        on_progress=on_progress,
    )

    assert len(progress_calls) == 3
    # Final call should have completed == total
    assert progress_calls[-1][0] == 3
    assert progress_calls[-1][1] == 3


def test_call_opus_parallel_propagates_exceptions():
    client = MagicMock()
    client.messages.create.side_effect = _FakeBadRequest()

    with pytest.raises(anthropic.BadRequestError):
        call_opus_parallel(
            [("a", 100)],
            client=client,
            max_workers=4,
            sleep_fn=_nop_sleep,
        )


def test_call_opus_parallel_with_empty_requests_returns_empty_list():
    client = MagicMock()
    results = call_opus_parallel([], client=client, max_workers=4, sleep_fn=_nop_sleep)
    assert results == []
