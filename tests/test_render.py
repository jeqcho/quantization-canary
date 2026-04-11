"""Tests for src/render.py - the static-site rollup builder."""

import json
from pathlib import Path

import pytest

from src.render import build_rollup


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _baseline_for(probes: list[dict]) -> dict:
    return {
        "version": "v1",
        "probe_set_version": "v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model": "claude-opus-4-6",
        "n_samples": 20,
        "prompts": {
            p["id"]: {
                "reference": f"baseline_ref_for_{p['id']}",
                "mu": 0.02,
                "sigma": 0.005,
                "reference_output_tokens": 100,
                "n_samples": 20,
            }
            for p in probes
        },
        "thresholds": {"yellow_z": 2.0, "red_z": 3.0},
        "cusum": {"target_D": 0.02, "k": 0.001, "h": 0.05},
        "calibration_distributions": {
            "z_values": [0.1, 0.2, 0.3],
            "d_values": [0.02, 0.021, 0.022],
            "d_mean": 0.021,
            "d_std": 0.001,
        },
    }


def _result(date: str, D: float, Z: float, verdict: str, per_prompt: list) -> dict:
    return {
        "date": date,
        "started_at": f"{date}T00:00:00+00:00",
        "finished_at": f"{date}T00:01:00+00:00",
        "model": "claude-opus-4-6",
        "baseline_version": "v1",
        "probe_set_version": "v1",
        "n_samples": 3,
        "D": D,
        "Z": Z,
        "cusum_state": 0.0,
        "cusum_alarmed": False,
        "cusum_alarm_streak": 0,
        "verdict": verdict,
        "per_prompt": per_prompt,
    }


# ---------- build_rollup ----------


def test_build_rollup_with_no_results_has_empty_history(tmp_path):
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)
    (tmp_path / "results").mkdir()

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    assert rollup["history"] == []
    assert rollup["latest"] is None
    assert "thresholds" in rollup


def test_build_rollup_with_single_result_has_latest_and_one_history_entry(tmp_path):
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)

    result = _result(
        "2026-04-12",
        D=0.025,
        Z=0.4,
        verdict="green",
        per_prompt=[
            {
                "id": "p1",
                "category": "test",
                "d": 0.025,
                "z": 0.4,
                "samples": ["s1", "s2", "s3"],
                "sample_distances": [0.02, 0.025, 0.03],
            }
        ],
    )
    _write_json(tmp_path / "results" / "2026-04-12.json", result)

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    assert len(rollup["history"]) == 1
    assert rollup["history"][0]["date"] == "2026-04-12"
    assert rollup["history"][0]["D"] == 0.025
    assert rollup["latest"]["date"] == "2026-04-12"
    assert rollup["latest"]["verdict"] == "green"


def test_build_rollup_history_sorted_chronologically(tmp_path):
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)

    pp = [
        {
            "id": "p1",
            "category": "test",
            "d": 0.02,
            "z": 0.0,
            "samples": ["s"],
            "sample_distances": [0.02],
        }
    ]
    _write_json(
        tmp_path / "results" / "2026-04-12.json",
        _result("2026-04-12", D=0.02, Z=0.0, verdict="green", per_prompt=pp),
    )
    _write_json(
        tmp_path / "results" / "2026-04-10.json",
        _result("2026-04-10", D=0.02, Z=0.0, verdict="green", per_prompt=pp),
    )
    _write_json(
        tmp_path / "results" / "2026-04-11.json",
        _result("2026-04-11", D=0.02, Z=0.0, verdict="green", per_prompt=pp),
    )

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    dates = [h["date"] for h in rollup["history"]]
    assert dates == ["2026-04-10", "2026-04-11", "2026-04-12"]
    # Latest is the most recent
    assert rollup["latest"]["date"] == "2026-04-12"


def test_build_rollup_latest_per_prompt_sorted_by_abs_z_descending(tmp_path):
    probes = [
        {"id": "p1", "category": "test", "prompt": "first", "max_tokens": 100},
        {"id": "p2", "category": "test", "prompt": "second", "max_tokens": 100},
        {"id": "p3", "category": "test", "prompt": "third", "max_tokens": 100},
    ]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)

    per_prompt = [
        {"id": "p1", "category": "test", "d": 0.02, "z": 0.5, "samples": ["a"], "sample_distances": [0.02]},
        {"id": "p2", "category": "test", "d": 0.05, "z": 3.5, "samples": ["b"], "sample_distances": [0.05]},
        {"id": "p3", "category": "test", "d": 0.03, "z": 1.5, "samples": ["c"], "sample_distances": [0.03]},
    ]
    _write_json(
        tmp_path / "results" / "2026-04-12.json",
        _result("2026-04-12", D=0.033, Z=1.83, verdict="red", per_prompt=per_prompt),
    )

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    ids = [pp["id"] for pp in rollup["latest"]["per_prompt"]]
    assert ids == ["p2", "p3", "p1"]


def test_build_rollup_latest_per_prompt_includes_baseline_reference_and_prompt_text(tmp_path):
    probes = [
        {
            "id": "p1",
            "category": "test",
            "prompt": "Original prompt text",
            "max_tokens": 100,
        }
    ]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)

    _write_json(
        tmp_path / "results" / "2026-04-12.json",
        _result(
            "2026-04-12",
            D=0.02,
            Z=0.0,
            verdict="green",
            per_prompt=[
                {
                    "id": "p1",
                    "category": "test",
                    "d": 0.02,
                    "z": 0.0,
                    "samples": ["s1"],
                    "sample_distances": [0.02],
                }
            ],
        ),
    )

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    pp = rollup["latest"]["per_prompt"][0]
    assert pp["prompt"] == "Original prompt text"
    assert pp["baseline_reference"] == "baseline_ref_for_p1"
    assert pp["baseline_mu"] == 0.02
    assert pp["baseline_sigma"] == 0.005


def test_build_rollup_includes_thresholds_and_meta(tmp_path):
    probes = [{"id": "p1", "category": "test", "prompt": "hi", "max_tokens": 100}]
    baseline = _baseline_for(probes)
    _write_jsonl(tmp_path / "probes" / "probe_set_v1.jsonl", probes)
    _write_json(tmp_path / "baseline" / "baseline_v1.json", baseline)
    (tmp_path / "results").mkdir()

    rollup = build_rollup(
        results_dir=tmp_path / "results",
        baseline_path=tmp_path / "baseline" / "baseline_v1.json",
        probes_path=tmp_path / "probes" / "probe_set_v1.jsonl",
    )

    assert rollup["thresholds"]["yellow_z"] == 2.0
    assert rollup["thresholds"]["red_z"] == 3.0
    assert rollup["model"] == "claude-opus-4-6"
    assert rollup["baseline_version"] == "v1"
    assert rollup["probe_set_version"] == "v1"
    assert "generated_at" in rollup
