"""Daily tick entrypoint.

Runs each probe in probes/probe_set_v<N>.jsonl n_samples times at T=0,
computes per-prompt distance-to-baseline and z-score, aggregates to D and
Z, replays the CUSUM state from historical results, computes the verdict,
and writes results/<date>.json. Does NOT commit - the GitHub Actions
workflow handles `git add`/`git commit`/`git push`.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.env import load_env
from src.metrics import (
    cusum_step,
    normalized_char_edit_distance,
    per_prompt_zscore,
)
from src.probe import OPUS_MODEL_ID, ProbeResult, call_opus_parallel

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = REPO_ROOT / "probes"
BASELINE_DIR = REPO_ROOT / "baseline"
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"

BASELINE_VERSION = "v1"
PROBE_SET_VERSION = "v1"


# ---------- core logic ----------


def compute_verdict(
    Z: float,
    yellow: float,
    red: float,
    cusum_alarmed: bool,
    cusum_alarm_streak: int,
) -> str:
    """Map aggregate Z and CUSUM state to a green/yellow/red verdict.

    Rules (from the plan):
      - red   if Z >= red threshold OR CUSUM alarmed >= 3 consecutive days
      - yellow if Z >= yellow threshold OR CUSUM alarmed 1 or 2 consecutive days
      - green otherwise
    """
    if Z >= red or (cusum_alarmed and cusum_alarm_streak >= 3):
        return "red"
    if Z >= yellow or (cusum_alarmed and cusum_alarm_streak >= 1):
        return "yellow"
    return "green"


def run_daily_tick(
    probes: list[dict],
    baseline: dict,
    call_fn: Callable[[str, int], ProbeResult],
    n_samples: int = 3,
    cusum_state: float = 0.0,
    cusum_alarm_streak: int = 0,
) -> dict:
    """Run one daily tick. Returns the result dict; does not persist it.

    cusum_state and cusum_alarm_streak are the values carried over from
    yesterday's result (or 0 if there is no yesterday).
    """
    started_at = datetime.now(timezone.utc)

    per_prompt_entries = []
    d_values = []
    z_values = []

    for probe in probes:
        pid = probe["id"]
        baseline_entry = baseline["prompts"][pid]
        reference = baseline_entry["reference"]
        mu = baseline_entry["mu"]
        sigma = baseline_entry["sigma"]

        samples: list[str] = []
        sample_distances: list[float] = []
        for _ in range(n_samples):
            r = call_fn(probe["prompt"], probe["max_tokens"])
            samples.append(r.text)
            sample_distances.append(
                normalized_char_edit_distance(r.text, reference)
            )

        # Median-over-samples is the per-prompt distance for today, robust
        # against a single weird draw.
        d_i = statistics.median(sample_distances)
        z_i = per_prompt_zscore(d_i, mu, sigma)

        d_values.append(d_i)
        z_values.append(z_i)
        per_prompt_entries.append(
            {
                "id": pid,
                "category": probe.get("category", "unknown"),
                "d": d_i,
                "z": z_i,
                "samples": samples,
                "sample_distances": sample_distances,
            }
        )

    D = statistics.mean(d_values)
    Z = statistics.mean(z_values)

    # CUSUM: deviation = D(today) - target_D; step; track streak.
    target_D = baseline["cusum"]["target_D"]
    k = baseline["cusum"]["k"]
    h = baseline["cusum"]["h"]
    deviation = D - target_D
    new_cusum_state, alarmed = cusum_step(cusum_state, deviation, k, h)

    if alarmed:
        new_alarm_streak = cusum_alarm_streak + 1
    else:
        new_alarm_streak = 0

    verdict = compute_verdict(
        Z=Z,
        yellow=baseline["thresholds"]["yellow_z"],
        red=baseline["thresholds"]["red_z"],
        cusum_alarmed=alarmed,
        cusum_alarm_streak=new_alarm_streak,
    )

    finished_at = datetime.now(timezone.utc)

    return {
        "date": started_at.strftime("%Y-%m-%d"),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "model": baseline["model"],
        "baseline_version": baseline["version"],
        "probe_set_version": baseline["probe_set_version"],
        "n_samples": n_samples,
        "D": D,
        "Z": Z,
        "cusum_state": new_cusum_state,
        "cusum_alarmed": alarmed,
        "cusum_alarm_streak": new_alarm_streak,
        "verdict": verdict,
        "per_prompt": per_prompt_entries,
    }


def replay_cusum_from_history(
    baseline: dict, results_dir: Path
) -> tuple[float, int]:
    """Replay CUSUM state from all existing result files in chronological order.

    Returns (cusum_state, cusum_alarm_streak) suitable for passing as the
    starting state of the next tick. This is deliberately stateless: every
    day the worker recomputes CUSUM from scratch, so manual edits to past
    result files propagate naturally and there is no hidden state file.
    """
    if not results_dir.exists():
        return 0.0, 0
    result_files = sorted(results_dir.glob("*.json"))
    state = 0.0
    streak = 0
    target_D = baseline["cusum"]["target_D"]
    k = baseline["cusum"]["k"]
    h = baseline["cusum"]["h"]
    for path in result_files:
        with path.open() as f:
            prev = json.load(f)
        deviation = prev["D"] - target_D
        state, alarmed = cusum_step(state, deviation, k, h)
        streak = streak + 1 if alarmed else 0
    return state, streak


def save_result(result: dict, results_dir: Path) -> Path:
    """Write a daily-tick result to results/<date>.json and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result['date']}.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


# ---------- CLI entry point ----------


def _make_lookup_call_fn(
    probes: list[dict],
    prefetched: list[ProbeResult],
    n_samples: int,
) -> Callable[[str, int], ProbeResult]:
    """Lookup-style fake call_fn over prefetched parallel results.

    Mirrors the helper in calibrate.py - the prefetched list is in
    probe-major order so we can slice cleanly per probe.
    """
    counters: dict[str, int] = {p["prompt"]: 0 for p in probes}
    pools: dict[str, list[ProbeResult]] = {}
    for probe_idx, probe in enumerate(probes):
        start = probe_idx * n_samples
        pools[probe["prompt"]] = prefetched[start : start + n_samples]

    def lookup(prompt: str, max_tokens: int) -> ProbeResult:
        idx = counters[prompt]
        counters[prompt] += 1
        return pools[prompt][idx]

    return lookup


def main() -> int:
    load_env()

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"worker-{timestamp}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    # Load probes + baseline.
    probe_path = PROBES_DIR / f"probe_set_{PROBE_SET_VERSION}.jsonl"
    with probe_path.open() as f:
        probes = [json.loads(line) for line in f if line.strip()]

    baseline_path = BASELINE_DIR / f"baseline_{BASELINE_VERSION}.json"
    with baseline_path.open() as f:
        baseline = json.load(f)

    log(f"Worker tick started at {datetime.now(timezone.utc).isoformat()}")
    log(f"Probes: {len(probes)}  Baseline: {baseline_path.name}")

    # Replay CUSUM from history so we don't need any external state file.
    cusum_state, alarm_streak = replay_cusum_from_history(baseline, RESULTS_DIR)
    log(
        f"Replayed CUSUM from history: state={cusum_state:.6f}, "
        f"alarm_streak={alarm_streak}"
    )

    n_samples = 3
    total = len(probes) * n_samples

    # Build the request list in probe-major order so the lookup_call_fn
    # slices cleanly.
    requests: list[tuple[str, int]] = []
    for probe in probes:
        for _ in range(n_samples):
            requests.append((probe["prompt"], probe["max_tokens"]))

    log(
        f"Dispatching {total} calls to a 50-worker pool ({len(probes)} probes "
        f"x {n_samples} samples)"
    )

    start_time = datetime.now(timezone.utc)

    def on_progress(completed: int, total: int, result: ProbeResult) -> None:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        log(
            f"  [{completed:3d}/{total}] ok ({result.output_tokens:3d} tok)  "
            f"(t={elapsed:5.1f}s)"
        )

    prefetched = call_opus_parallel(
        requests,
        max_workers=50,
        on_progress=on_progress,
    )

    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    log(f"All {total} calls finished in {elapsed_total:.1f}s wall time.")

    lookup_call = _make_lookup_call_fn(probes, prefetched, n_samples)

    result = run_daily_tick(
        probes,
        baseline,
        call_fn=lookup_call,
        n_samples=n_samples,
        cusum_state=cusum_state,
        cusum_alarm_streak=alarm_streak,
    )

    path = save_result(result, RESULTS_DIR)
    log("")
    log(f"Result written to {path}")
    log(
        f"D={result['D']:.4f}  Z={result['Z']:.4f}  "
        f"verdict={result['verdict']}  "
        f"cusum_state={result['cusum_state']:.6f}"
    )

    # Cost summary for this tick.
    total_cost = sum(r.cost_usd for r in prefetched)
    log(f"Tick cost: ${total_cost:.4f}")

    # Refresh docs/data.json rollup for the static site.
    from src.render import refresh_rollup  # local import to avoid cycles

    refresh_rollup()
    log("Rollup refreshed (docs/data.json)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
