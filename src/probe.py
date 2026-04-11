"""Anthropic API wrapper: single probe call with retries and wall-clock timestamps.

Every call returns a ProbeResult with input/output token counts, USD cost
computed at Opus 4.6 rates, and actual wall-clock start/finish timestamps
(not cron-scheduled time, since GitHub Actions cron can drift several
minutes under load).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import anthropic

OPUS_MODEL_ID = "claude-opus-4-6"

# Opus 4.6 pricing per million tokens (USD). Update if Anthropic changes rates.
OPUS_INPUT_COST_PER_M = 15.0
OPUS_OUTPUT_COST_PER_M = 75.0

# Errors that represent transient failures worth retrying. 4xx errors
# other than rate-limit are not retryable - they will just fail again.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


@dataclass(frozen=True)
class ProbeResult:
    text: str
    started_at: datetime
    finished_at: datetime
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    """USD cost for an Opus 4.6 call given input/output token counts."""
    return (
        input_tokens * OPUS_INPUT_COST_PER_M / 1_000_000
        + output_tokens * OPUS_OUTPUT_COST_PER_M / 1_000_000
    )


def call_opus(
    prompt: str,
    max_tokens: int,
    *,
    client: anthropic.Anthropic | None = None,
    max_retries: int = 5,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ProbeResult:
    """Single Opus 4.6 call at temperature=0 with exponential-backoff retries.

    Retries on rate-limit, connection, timeout, and 5xx errors (exponential
    backoff capped at 60 s). Does not retry on 4xx client errors. Records
    actual wall-clock start/finish time and computes USD cost from the
    returned usage stats.
    """
    if client is None:
        client = anthropic.Anthropic()

    started_at = datetime.now(timezone.utc)
    response = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=OPUS_MODEL_ID,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except _RETRYABLE_ERRORS:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff capped at 60 s.
            sleep_fn(min(2**attempt, 60))

    if response is None:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    finished_at = datetime.now(timezone.utc)
    return ProbeResult(
        text=response.content[0].text,
        started_at=started_at,
        finished_at=finished_at,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=_compute_cost(
            response.usage.input_tokens, response.usage.output_tokens
        ),
    )


def call_opus_parallel(
    requests: list[tuple[str, int]],
    *,
    client: anthropic.Anthropic | None = None,
    max_workers: int = 50,
    max_retries: int = 5,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_progress: Optional[Callable[[int, int, ProbeResult], None]] = None,
) -> list[ProbeResult]:
    """Call call_opus on N (prompt, max_tokens) pairs concurrently.

    Returns the ProbeResult list in the SAME ORDER as the input requests
    (not in completion order). Uses a single shared Anthropic client.
    Propagates exceptions from any individual call.

    on_progress(completed, total, result) is invoked from the main
    thread once per completed call (in completion order, not input
    order). Useful for live progress logging.
    """
    if not requests:
        return []
    if client is None:
        client = anthropic.Anthropic()

    results: list[ProbeResult | None] = [None] * len(requests)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                call_opus,
                prompt,
                max_tokens,
                client=client,
                max_retries=max_retries,
                sleep_fn=sleep_fn,
            ): idx
            for idx, (prompt, max_tokens) in enumerate(requests)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()  # propagates exceptions
            results[idx] = result
            completed += 1
            if on_progress is not None:
                on_progress(completed, len(requests), result)

    # Type-narrow for the type checker - all positions are filled by now.
    return [r for r in results if r is not None]
