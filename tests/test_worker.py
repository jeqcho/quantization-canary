"""Tests for src/worker.py.

Uses a fake call_fn and a synthetic baseline so no real API calls are
made. Focus: the daily-tick aggregation logic and the verdict mapping.
"""

from datetime import datetime, timezone

import pytest

from src.probe import ProbeResult
from src.worker import compute_verdict, run_daily_tick


# ---------- helpers ----------


def _fake_result(text: str) -> ProbeResult:
    return ProbeResult(
        text=text,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        input_tokens=10,
        output_tokens=len(text) // 4,
        cost_usd=0.001,
    )


def _minimal_baseline(prompts: dict, target_D: float = 0.0) -> dict:
    """Construct a minimal baseline dict for tests."""
    return {
        "version": "v1",
        "probe_set_version": "v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model": "claude-opus-4-6",
        "n_samples": 20,
        "prompts": prompts,
        "thresholds": {"yellow_z": 2.0, "red_z": 3.0},
        "cusum": {"target_D": target_D, "k": 0.001, "h": 0.05},
        "calibration_distributions": {
            "z_values": [],
            "d_values": [],
            "d_mean": target_D,
            "d_std": 0.01,
        },
    }


# ---------- run_daily_tick ----------


def test_run_daily_tick_calls_model_n_samples_times_per_probe():
    probes = [
        {"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100},
        {"id": "p2", "category": "test", "prompt": "bye", "max_tokens": 100},
    ]
    baseline = _minimal_baseline(
        {
            "p1": {"reference": "ref1", "mu": 0.0, "sigma": 0.01, "reference_output_tokens": 10, "n_samples": 20},
            "p2": {"reference": "ref2", "mu": 0.0, "sigma": 0.01, "reference_output_tokens": 10, "n_samples": 20},
        }
    )
    call_count = 0

    def fake_call(prompt, max_tokens):
        nonlocal call_count
        call_count += 1
        return _fake_result(f"ref{prompt[0]}")

    run_daily_tick(
        probes, baseline, call_fn=fake_call, n_samples=3, cusum_state=0.0
    )

    assert call_count == 6  # 2 probes * 3 samples


def test_run_daily_tick_identical_to_baseline_yields_zero_distance():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _minimal_baseline(
        {
            "p1": {
                "reference": "identical baseline text",
                "mu": 0.0,
                "sigma": 0.01,
                "reference_output_tokens": 10,
                "n_samples": 20,
            }
        }
    )

    def fake_call(prompt, max_tokens):
        return _fake_result("identical baseline text")

    result = run_daily_tick(
        probes, baseline, call_fn=fake_call, n_samples=3, cusum_state=0.0
    )

    assert result["D"] == 0.0
    assert result["Z"] == 0.0
    assert result["per_prompt"][0]["d"] == 0.0
    assert result["per_prompt"][0]["z"] == 0.0


def test_run_daily_tick_stable_output_yields_green_verdict():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _minimal_baseline(
        {
            "p1": {
                "reference": "the baseline reference",
                "mu": 0.0,
                "sigma": 0.01,
                "reference_output_tokens": 10,
                "n_samples": 20,
            }
        }
    )

    def fake_call(prompt, max_tokens):
        return _fake_result("the baseline reference")

    result = run_daily_tick(
        probes, baseline, call_fn=fake_call, n_samples=3, cusum_state=0.0,
        cusum_alarm_streak=0,
    )

    assert result["verdict"] == "green"


def test_run_daily_tick_includes_samples_for_drilldown():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _minimal_baseline(
        {
            "p1": {
                "reference": "baseline",
                "mu": 0.0,
                "sigma": 0.01,
                "reference_output_tokens": 10,
                "n_samples": 20,
            }
        }
    )
    texts = iter(["sample_a", "sample_b", "sample_c"])

    def fake_call(prompt, max_tokens):
        return _fake_result(next(texts))

    result = run_daily_tick(
        probes, baseline, call_fn=fake_call, n_samples=3, cusum_state=0.0
    )

    assert result["per_prompt"][0]["samples"] == [
        "sample_a",
        "sample_b",
        "sample_c",
    ]


# ---------- compute_verdict ----------


def test_verdict_green_when_Z_low_and_cusum_quiet():
    assert compute_verdict(Z=0.5, yellow=2.0, red=3.0, cusum_alarmed=False, cusum_alarm_streak=0) == "green"


def test_verdict_yellow_when_Z_in_mid_band():
    assert compute_verdict(Z=2.5, yellow=2.0, red=3.0, cusum_alarmed=False, cusum_alarm_streak=0) == "yellow"


def test_verdict_red_when_Z_above_red_threshold():
    assert compute_verdict(Z=3.5, yellow=2.0, red=3.0, cusum_alarmed=False, cusum_alarm_streak=0) == "red"


def test_verdict_yellow_on_first_cusum_alarm():
    assert compute_verdict(Z=0.5, yellow=2.0, red=3.0, cusum_alarmed=True, cusum_alarm_streak=1) == "yellow"


def test_verdict_red_on_third_consecutive_cusum_alarm():
    assert compute_verdict(Z=0.5, yellow=2.0, red=3.0, cusum_alarmed=True, cusum_alarm_streak=3) == "red"
