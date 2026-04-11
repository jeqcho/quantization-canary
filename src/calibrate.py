"""One-time baseline builder.

Runs each prompt in probes/probe_set_v<N>.jsonl n_samples times at T=0,
picks the medoid sample as the reference output, measures the per-prompt
noise floor (mu, sigma) from the sample-to-medoid edit-distance
distribution, and derives empirical yellow/red Z thresholds and CUSUM
k/h parameters from the same calibration window.

Writes baseline/baseline_v<N>.json. Invoked manually, not by cron.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.env import load_env
from src.metrics import normalized_char_edit_distance, per_prompt_zscore
from src.probe import OPUS_MODEL_ID, ProbeResult, call_opus_parallel

# Note: the plan's original phrasing of the noise floor said "pairwise"
# distances among the 20 samples. After a second look, computing
# (mu_i, sigma_i) from the distance-to-medoid distribution is the
# statistically consistent choice: daily-tick measurements are also
# distance-to-medoid, so the Z-score is the deviation of today's distance
# from the baseline distance-to-medoid distribution. Using pairwise mu
# would bias Z by a factor of ~2 since pairwise expected distance is
# roughly twice the expected distance to the central sample.

BASELINE_VERSION = "v1"
PROBE_SET_VERSION = "v1"
REPO_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = REPO_ROOT / "probes"
BASELINE_DIR = REPO_ROOT / "baseline"
LOG_DIR = REPO_ROOT / "logs"


# ---------- helpers ----------


def find_medoid(texts: list[str]) -> int:
    """Return the index of the sample with minimum mean edit distance to all others.

    Ties are broken by the lowest index (stable, deterministic).
    """
    n = len(texts)
    if n == 0:
        raise ValueError("Cannot find medoid of empty list")
    if n == 1:
        return 0

    best_idx = 0
    best_mean_dist = float("inf")
    for i, t in enumerate(texts):
        mean_dist = sum(
            normalized_char_edit_distance(t, other)
            for j, other in enumerate(texts)
            if j != i
        ) / (n - 1)
        if mean_dist < best_mean_dist:
            best_mean_dist = mean_dist
            best_idx = i
    return best_idx


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. pct in [0, 100]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _std(values: list[float]) -> float:
    """Population std that is safe for n<2 (returns 0.0)."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


# ---------- core logic ----------


def build_baseline(
    probes: list[dict],
    call_fn: Callable[[str, int], ProbeResult],
    n_samples: int = 20,
) -> dict:
    """Compute a baseline dict from live model calls.

    call_fn(prompt, max_tokens) is invoked n_samples times per probe.
    Returns a pure dict with no file I/O; use save_baseline() to persist.
    """
    # Collect all samples up front.
    samples_by_probe: dict[str, list[str]] = {}
    probe_results_by_probe: dict[str, list[ProbeResult]] = {}
    for probe in probes:
        pid = probe["id"]
        results = [
            call_fn(probe["prompt"], probe["max_tokens"]) for _ in range(n_samples)
        ]
        probe_results_by_probe[pid] = results
        samples_by_probe[pid] = [r.text for r in results]

    # Per-prompt reference, noise floor, and per-sample distances.
    per_prompt: dict[str, dict] = {}
    per_sample_distances: dict[str, list[float]] = {}

    for probe in probes:
        pid = probe["id"]
        texts = samples_by_probe[pid]
        medoid_idx = find_medoid(texts)
        reference = texts[medoid_idx]
        distances = [normalized_char_edit_distance(t, reference) for t in texts]
        mu = statistics.mean(distances)
        sigma = _std(distances)

        per_prompt[pid] = {
            "reference": reference,
            "mu": mu,
            "sigma": sigma,
            "reference_output_tokens": probe_results_by_probe[pid][
                medoid_idx
            ].output_tokens,
            "n_samples": n_samples,
        }
        per_sample_distances[pid] = distances

    # Simulate n_samples "stable days" to derive the calibration Z and D
    # distributions. For each sample index, D_sample = mean over prompts of
    # the sample's distance to its prompt's medoid; Z_sample = mean over
    # prompts of its per-prompt z-score.
    z_values: list[float] = []
    d_values: list[float] = []
    for sample_idx in range(n_samples):
        per_prompt_zs = []
        per_prompt_ds = []
        for probe in probes:
            pid = probe["id"]
            d = per_sample_distances[pid][sample_idx]
            mu = per_prompt[pid]["mu"]
            sigma = per_prompt[pid]["sigma"]
            per_prompt_zs.append(per_prompt_zscore(d, mu, sigma))
            per_prompt_ds.append(d)
        z_values.append(statistics.mean(per_prompt_zs))
        d_values.append(statistics.mean(per_prompt_ds))

    # Empirical Z thresholds: 95th and 99.5th percentile of calibration Z.
    z_sorted = sorted(z_values)
    yellow_z = _percentile(z_sorted, 95.0)
    red_z = _percentile(z_sorted, 99.5)

    # CUSUM parameters calibrated against the D distribution.
    d_mean = statistics.mean(d_values)
    d_std = _std(d_values)
    # Standard SPC defaults scaled by sigma_D; floor at tiny epsilons so
    # degenerate (perfectly stable) calibrations still produce usable
    # k, h rather than zeros.
    cusum_k = max(0.5 * d_std, 1e-6)
    cusum_h = max(5.0 * d_std, 1e-5)

    return {
        "version": BASELINE_VERSION,
        "probe_set_version": PROBE_SET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": OPUS_MODEL_ID,
        "n_samples": n_samples,
        "prompts": per_prompt,
        "thresholds": {
            "yellow_z": yellow_z,
            "red_z": red_z,
        },
        "cusum": {
            "target_D": d_mean,
            "k": cusum_k,
            "h": cusum_h,
        },
        "calibration_distributions": {
            "z_values": z_values,
            "d_values": d_values,
            "d_mean": d_mean,
            "d_std": d_std,
        },
    }


def save_baseline(baseline: dict, path: Path) -> None:
    """Write baseline dict to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)


def load_probes(probe_set_version: str = PROBE_SET_VERSION) -> list[dict]:
    """Load the prompt list from probes/probe_set_<version>.jsonl."""
    path = PROBES_DIR / f"probe_set_{probe_set_version}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- CLI entry point ----------


def _make_lookup_call_fn(
    probes: list[dict],
    prefetched: list[ProbeResult],
    n_samples: int,
) -> Callable[[str, int], ProbeResult]:
    """Build a fake call_fn that returns prefetched ProbeResults in order.

    The prefetched list MUST be ordered as: probe[0] sample[0..n-1],
    then probe[1] sample[0..n-1], etc. (which is exactly what
    call_opus_parallel returns when called with that request shape).

    This lets the parallel-prefetched results flow back into the
    pure, synchronous build_baseline() without changing its signature
    or its tests.
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
    log_path = LOG_DIR / f"calibration-{timestamp}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    probes = load_probes(PROBE_SET_VERSION)
    n_samples = 20
    total = len(probes) * n_samples

    log(f"Calibration started at {datetime.now(timezone.utc).isoformat()}")
    log(f"Loaded {len(probes)} probes from probe_set_{PROBE_SET_VERSION}.jsonl")
    log(
        f"Running {n_samples} samples per probe against {OPUS_MODEL_ID} "
        f"in a 50-worker pool ({total} total calls)"
    )
    log("")

    # Build the request list: probe-major order so the lookup_call_fn
    # can slice it cleanly.
    requests: list[tuple[str, int]] = []
    for probe in probes:
        for _ in range(n_samples):
            requests.append((probe["prompt"], probe["max_tokens"]))

    start_time = datetime.now(timezone.utc)

    def on_progress(completed: int, total: int, result: ProbeResult) -> None:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        log(
            f"  [{completed:4d}/{total}] ok ({result.output_tokens:3d} tok)  "
            f"(t={elapsed:6.1f}s)"
        )

    prefetched = call_opus_parallel(
        requests,
        max_workers=50,
        on_progress=on_progress,
    )

    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    log("")
    log(f"All {total} calls finished in {elapsed_total:.1f}s wall time.")

    # Compute baseline from the prefetched results.
    lookup_call = _make_lookup_call_fn(probes, prefetched, n_samples)
    baseline = build_baseline(probes, call_fn=lookup_call, n_samples=n_samples)

    baseline_path = BASELINE_DIR / f"baseline_{BASELINE_VERSION}.json"
    save_baseline(baseline, baseline_path)

    log("")
    log(f"Baseline written to {baseline_path}")
    log(
        f"Thresholds: yellow_z={baseline['thresholds']['yellow_z']:.4f}, "
        f"red_z={baseline['thresholds']['red_z']:.4f}"
    )
    log(
        f"CUSUM: target_D={baseline['cusum']['target_D']:.4f}, "
        f"k={baseline['cusum']['k']:.6f}, h={baseline['cusum']['h']:.6f}"
    )

    # Per-prompt noise floor summary.
    log("")
    log("Per-prompt noise floor:")
    for pid, p in baseline["prompts"].items():
        log(f"  {pid}: mu={p['mu']:.4f}  sigma={p['sigma']:.4f}  ref_tokens={p['reference_output_tokens']}")

    # Stability rule check.
    unstable = [
        (pid, p["mu"])
        for pid, p in baseline["prompts"].items()
        if p["mu"] > 0.05
    ]
    if unstable:
        log("")
        log("WARNING: prompts with mu > 0.05 (stability rule violated):")
        for pid, mu in unstable:
            log(f"  - {pid}: mu={mu:.4f}")
    else:
        log("")
        log("All 30 prompts passed the stability rule (mu <= 0.05).")

    # Cost summary.
    total_in = sum(r.input_tokens for r in prefetched)
    total_out = sum(r.output_tokens for r in prefetched)
    total_cost = sum(r.cost_usd for r in prefetched)
    log("")
    log(
        f"Total tokens: {total_in} in, {total_out} out  Total cost: ${total_cost:.4f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
