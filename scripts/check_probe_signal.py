"""One-shot probe-set signal check: test Opus 4.6 vs Sonnet 4.6.

For each prompt in probes/probe_set_v1.jsonl, calls both Opus 4.6 and
Sonnet 4.6 at temperature=0, then compares the outputs. Prompts where
both models produce byte-identical output have zero fingerprint signal
(they are in a deterministic top-1 basin that no realistic quantization
would perturb) and must be rewritten or replaced.

Parallelism: all 2*N calls are dispatched concurrently to a 50-worker
thread pool. At Anthropic API usage tier 5 this finishes in roughly
the time of a single slowest call.

Reports hash equality as the primary filter and normalized character
edit distance as secondary context for borderline cases. Writes a
full JSON artifact with raw outputs so prompts can be inspected.

Run from the repo root:

    uv run python scripts/check_probe_signal.py

Budget note: respects each prompt's configured max_tokens. At 30
prompts with max_tokens=500, worst-case cost is well under $2
(Opus ~$1.1 + Sonnet ~$0.2). Actual cost is usually lower since
real outputs rarely hit the cap.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Allow `uv run python scripts/check_probe_signal.py` from the repo root
# by ensuring the repo root is on sys.path before importing src.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import anthropic  # noqa: E402

from src.env import load_env  # noqa: E402
from src.metrics import normalized_char_edit_distance  # noqa: E402

# Load .env BEFORE anthropic.Anthropic() reads ANTHROPIC_API_KEY from os.environ.
load_env()

OPUS_MODEL_ID = "claude-opus-4-6"
SONNET_MODEL_ID = "claude-sonnet-4-6"

# Pricing per million tokens (USD).
OPUS_INPUT_COST_PER_M = 15.0
OPUS_OUTPUT_COST_PER_M = 75.0
SONNET_INPUT_COST_PER_M = 3.0
SONNET_OUTPUT_COST_PER_M = 15.0

WORKERS = 50
MAX_RETRIES = 4

REPO_ROOT = _REPO_ROOT
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.jsonl"
LOG_DIR = REPO_ROOT / "logs"

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def _call_model(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Single temperature=0 call with exponential-backoff retries on transient errors.

    Returns (text, input_tokens, output_tokens).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return (
                response.content[0].text,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        except _RETRYABLE:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable: MAX_RETRIES must be >= 1")


def _sha256_hex12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _load_probes() -> list[dict]:
    with PROBE_SET_PATH.open() as f:
        return [json.loads(line) for line in f]


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"probe-signal-check-{timestamp}.log"
    results_path = LOG_DIR / f"probe-signal-check-{timestamp}.json"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    probes = _load_probes()
    client = anthropic.Anthropic()

    log(f"Probe-set signal check started at {datetime.now(timezone.utc).isoformat()}")
    log(f"Loaded {len(probes)} prompts from {PROBE_SET_PATH}")
    log(
        f"Comparing {OPUS_MODEL_ID} vs {SONNET_MODEL_ID} "
        f"at temperature=0 with {WORKERS}-worker pool"
    )
    log("=" * 72)

    # Build the full task list: (probe_dict, model_name, model_id)
    tasks: list[tuple[dict, str, str]] = []
    for p in probes:
        tasks.append((p, "opus", OPUS_MODEL_ID))
        tasks.append((p, "sonnet", SONNET_MODEL_ID))

    raw_results: dict[tuple[str, str], tuple[str, int, int]] = {}
    errors: list[tuple[str, str, Exception]] = []

    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {
            executor.submit(
                _call_model, client, model_id, p["prompt"], p["max_tokens"]
            ): (p, model_name)
            for p, model_name, model_id in tasks
        }

        log(f"Dispatched {len(future_to_task)} concurrent calls. Waiting...")
        log("")

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            p, model_name = future_to_task[future]
            elapsed = time.monotonic() - start_time
            try:
                text, in_tok, out_tok = future.result()
                raw_results[(p["id"], model_name)] = (text, in_tok, out_tok)
                log(
                    f"[{completed:3d}/{len(future_to_task)}] "
                    f"{p['id']} {model_name:<6s} ok "
                    f"({out_tok:3d} tok)  (t={elapsed:5.1f}s)"
                )
            except Exception as e:
                errors.append((p["id"], model_name, e))
                log(
                    f"[{completed:3d}/{len(future_to_task)}] "
                    f"{p['id']} {model_name:<6s} "
                    f"ERROR: {type(e).__name__}: {e}"
                )

    elapsed_total = time.monotonic() - start_time
    log("")
    log(f"All calls finished in {elapsed_total:.1f}s wall time.")
    log("")

    # Per-prompt comparison from the collected results.
    log("PER-PROMPT COMPARISON")
    log("=" * 72)

    results: list[dict] = []
    for p in probes:
        opus_data = raw_results.get((p["id"], "opus"))
        sonnet_data = raw_results.get((p["id"], "sonnet"))
        if not opus_data or not sonnet_data:
            log(f"  {p['id']} skipped (missing result from one or both models)")
            continue

        opus_text, opus_in, opus_out = opus_data
        sonnet_text, sonnet_in, sonnet_out = sonnet_data

        identical = opus_text == sonnet_text
        edit_dist = normalized_char_edit_distance(opus_text, sonnet_text)
        opus_hash = _sha256_hex12(opus_text)
        sonnet_hash = _sha256_hex12(sonnet_text)

        if identical:
            marker = "ZERO SIGNAL (byte-identical)"
        elif edit_dist < 0.05:
            marker = f"low signal, dist={edit_dist:.4f}"
        else:
            marker = f"dist={edit_dist:.4f}"

        log(
            f"  {p['id']} {p['category']:<11s} "
            f"opus={opus_hash} sonnet={sonnet_hash}  {marker}"
        )

        results.append(
            {
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "max_tokens": p["max_tokens"],
                "identical": identical,
                "edit_distance": edit_dist,
                "opus_hash": opus_hash,
                "sonnet_hash": sonnet_hash,
                "opus_output": opus_text,
                "sonnet_output": sonnet_text,
                "opus_input_tokens": opus_in,
                "opus_output_tokens": opus_out,
                "sonnet_input_tokens": sonnet_in,
                "sonnet_output_tokens": sonnet_out,
            }
        )

    # Cost calculation.
    opus_in_total = sum(r["opus_input_tokens"] for r in results)
    opus_out_total = sum(r["opus_output_tokens"] for r in results)
    sonnet_in_total = sum(r["sonnet_input_tokens"] for r in results)
    sonnet_out_total = sum(r["sonnet_output_tokens"] for r in results)
    opus_cost = (
        opus_in_total * OPUS_INPUT_COST_PER_M
        + opus_out_total * OPUS_OUTPUT_COST_PER_M
    ) / 1_000_000
    sonnet_cost = (
        sonnet_in_total * SONNET_INPUT_COST_PER_M
        + sonnet_out_total * SONNET_OUTPUT_COST_PER_M
    ) / 1_000_000

    zero_signal = [r for r in results if r["identical"]]
    low_signal = [
        r for r in results if not r["identical"] and r["edit_distance"] < 0.05
    ]

    log("")
    log("=" * 72)
    log("SUMMARY")
    log(f"Tested: {len(results)}/{len(probes)} prompts")
    log(f"Zero-signal (byte-identical): {len(zero_signal)}")
    log(f"Low-signal (edit dist < 0.05): {len(low_signal)}")
    if results:
        mean_dist = sum(r["edit_distance"] for r in results) / len(results)
        log(f"Mean edit distance: {mean_dist:.4f}")
    log(f"Errors: {len(errors)}")
    log(f"Wall time: {elapsed_total:.1f}s")
    log(
        f"Cost: Opus ${opus_cost:.4f} + Sonnet ${sonnet_cost:.4f} "
        f"= ${opus_cost + sonnet_cost:.4f}"
    )

    if zero_signal:
        log("")
        log("ZERO-SIGNAL PROMPTS (must rewrite or replace):")
        for r in zero_signal:
            log(
                f"  - {r['id']} ({r['category']}): "
                f"same output between Opus and Sonnet"
            )

    if low_signal:
        log("")
        log("LOW-SIGNAL PROMPTS (consider rewriting):")
        for r in low_signal:
            log(
                f"  - {r['id']} ({r['category']}): "
                f"edit_distance={r['edit_distance']:.4f}"
            )

    with results_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("")
    log(f"Log:      {log_path}")
    log(f"Results:  {results_path}")

    return 1 if zero_signal else 0


if __name__ == "__main__":
    sys.exit(main())
