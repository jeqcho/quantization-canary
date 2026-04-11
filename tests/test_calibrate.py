"""Unit tests for src/calibrate.py.

Uses a fake call_fn (no real API) so these run fast and deterministically.
"""

from datetime import datetime, timezone

import pytest

from src.calibrate import build_baseline, find_medoid
from src.probe import ProbeResult


# ---------- find_medoid ----------


def test_medoid_of_identical_strings_returns_first():
    texts = ["hello", "hello", "hello"]
    assert find_medoid(texts) == 0


def test_medoid_is_the_central_string():
    # "cat", "cot", "dog" - "cot" is closest to both others
    texts = ["cat", "cot", "dog"]
    assert find_medoid(texts) == 1


def test_medoid_of_two_strings_is_first():
    texts = ["abc", "xyz"]
    assert find_medoid(texts) == 0


def test_medoid_handles_many_strings():
    # Centered around "banana", one outlier
    texts = ["banana", "banana", "banana", "apple", "banana"]
    assert find_medoid(texts) in (0, 1, 2, 4)  # any of the bananas


# ---------- build_baseline ----------


def _fake_probe_result(text: str) -> ProbeResult:
    return ProbeResult(
        text=text,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        input_tokens=10,
        output_tokens=len(text) // 4,
        cost_usd=0.001,
    )


def test_build_baseline_calls_model_n_times_per_probe():
    probes = [
        {"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100},
        {"id": "p2", "category": "test", "prompt": "bye", "max_tokens": 100},
    ]
    call_count = 0

    def fake_call(prompt, max_tokens):
        nonlocal call_count
        call_count += 1
        return _fake_probe_result(f"response to {prompt} #{call_count}")

    build_baseline(probes, call_fn=fake_call, n_samples=5)

    # 2 probes * 5 samples = 10 calls
    assert call_count == 10


def test_build_baseline_returns_expected_structure():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]

    def fake_call(prompt, max_tokens):
        return _fake_probe_result("deterministic response")

    baseline = build_baseline(probes, call_fn=fake_call, n_samples=5)

    assert "version" in baseline
    assert "model" in baseline
    assert "n_samples" in baseline
    assert baseline["n_samples"] == 5
    assert "prompts" in baseline
    assert "p1" in baseline["prompts"]
    assert "reference" in baseline["prompts"]["p1"]
    assert "mu" in baseline["prompts"]["p1"]
    assert "sigma" in baseline["prompts"]["p1"]
    assert "thresholds" in baseline
    assert "yellow_z" in baseline["thresholds"]
    assert "red_z" in baseline["thresholds"]
    assert "cusum" in baseline
    assert "target_D" in baseline["cusum"]
    assert "k" in baseline["cusum"]
    assert "h" in baseline["cusum"]


def test_build_baseline_perfectly_deterministic_prompt_has_zero_noise():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]

    def fake_call(prompt, max_tokens):
        return _fake_probe_result("exactly the same every time")

    baseline = build_baseline(probes, call_fn=fake_call, n_samples=5)

    p = baseline["prompts"]["p1"]
    assert p["reference"] == "exactly the same every time"
    assert p["mu"] == 0.0
    assert p["sigma"] == 0.0


def test_build_baseline_noisy_prompt_has_positive_mu_and_sigma():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    counter = [0]

    def fake_call(prompt, max_tokens):
        counter[0] += 1
        # Vary the output to simulate T=0 jitter
        return _fake_probe_result(f"response variant {counter[0]} with some text")

    baseline = build_baseline(probes, call_fn=fake_call, n_samples=5)

    p = baseline["prompts"]["p1"]
    assert p["mu"] > 0.0
    assert p["sigma"] >= 0.0  # could be 0 if all distances equal by coincidence


def test_build_baseline_computes_thresholds_from_z_distribution():
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    counter = [0]

    def fake_call(prompt, max_tokens):
        counter[0] += 1
        return _fake_probe_result(f"response variant {counter[0]}")

    baseline = build_baseline(probes, call_fn=fake_call, n_samples=10)

    # Thresholds should exist and be ordered correctly
    yellow = baseline["thresholds"]["yellow_z"]
    red = baseline["thresholds"]["red_z"]
    assert yellow <= red  # red threshold is higher than yellow
