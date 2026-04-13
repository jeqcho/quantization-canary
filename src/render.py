"""Rollup renderer.

Reads results/*.json + baseline + probes and produces docs/data.json,
the single artifact the static site loads on page open. The site never
has to fetch per-day result files - the rollup carries:

- top-level meta (model, baseline version, probe set version, thresholds)
- a chronological history list (date, D, Z, verdict, cusum_state) for chart
- a `latest` block with the most recent tick's per-prompt drill-down,
  joined against baseline references and prompt text so the diff view
  has everything it needs in one fetch
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = REPO_ROOT / "probes"
BASELINE_DIR = REPO_ROOT / "baseline"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

PROBE_SET_VERSION = "v1"
BASELINE_VERSION = "v1"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _enrich_per_prompt(
    raw_per_prompt: list[dict],
    probes_by_id: dict[str, dict],
    baseline_prompts: dict[str, dict],
) -> list[dict]:
    """Join today's per-prompt entries with baseline reference and prompt text.

    Returns a new list sorted by absolute z descending so the most-drifted
    prompts surface first in the drill-down view.
    """
    enriched = []
    for pp in raw_per_prompt:
        pid = pp["id"]
        probe = probes_by_id.get(pid, {})
        baseline_entry = baseline_prompts.get(pid, {})
        enriched.append(
            {
                "id": pid,
                "category": pp.get("category", probe.get("category", "unknown")),
                "prompt": probe.get("prompt", ""),
                "max_tokens": probe.get("max_tokens"),
                "d": pp["d"],
                "z": pp["z"],
                "samples": pp["samples"],
                "sample_distances": pp.get("sample_distances", []),
                "baseline_reference": baseline_entry.get("reference", ""),
                "baseline_mu": baseline_entry.get("mu"),
                "baseline_sigma": baseline_entry.get("sigma"),
            }
        )
    enriched.sort(key=lambda e: abs(e["z"]), reverse=True)
    return enriched


def build_rollup(
    results_dir: Path,
    baseline_path: Path,
    probes_path: Path,
) -> dict:
    """Build the docs/data.json rollup from disk artifacts. Pure: no I/O writes."""
    baseline = _load_json(baseline_path)
    probes = _load_jsonl(probes_path)
    probes_by_id = {p["id"]: p for p in probes}

    # Load and sort all daily results chronologically by date.
    result_files = sorted(results_dir.glob("*.json"))
    results = [_load_json(p) for p in result_files]
    results.sort(key=lambda r: r["date"])

    history = [
        {
            "date": r["date"],
            "D": r["D"],
            "Z": r["Z"],
            "verdict": r["verdict"],
            "cusum_state": r.get("cusum_state", 0.0),
            "cusum_alarmed": r.get("cusum_alarmed", False),
        }
        for r in results
    ]

    latest: dict | None = None
    if results:
        most_recent = results[-1]
        latest = {
            "date": most_recent["date"],
            "started_at": most_recent.get("started_at"),
            "finished_at": most_recent.get("finished_at"),
            "D": most_recent["D"],
            "Z": most_recent["Z"],
            "verdict": most_recent["verdict"],
            "cusum_state": most_recent.get("cusum_state", 0.0),
            "cusum_alarmed": most_recent.get("cusum_alarmed", False),
            "cusum_alarm_streak": most_recent.get("cusum_alarm_streak", 0),
            "n_samples": most_recent.get("n_samples", 0),
            "per_prompt": _enrich_per_prompt(
                most_recent.get("per_prompt", []),
                probes_by_id,
                baseline.get("prompts", {}),
            ),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": baseline.get("model", ""),
        "baseline_version": baseline.get("version", ""),
        "probe_set_version": baseline.get("probe_set_version", ""),
        "baseline_created_at": baseline.get("created_at", ""),
        "n_probes": len(probes),
        "thresholds": baseline.get("thresholds", {}),
        "cusum": baseline.get("cusum", {}),
        "noise_band": {
            "center": baseline.get("cusum", {}).get("target_D", 0),
            "std": baseline.get("calibration_distributions", {}).get("d_std", 0),
        },
        "latest": latest,
        "history": history,
    }


def save_rollup(rollup: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rollup, f, indent=2, ensure_ascii=False)


def refresh_rollup() -> Path:
    """Convenience: build the rollup using default repo paths and write it.

    Used by worker.py at the end of every daily tick.
    """
    rollup = build_rollup(
        results_dir=RESULTS_DIR,
        baseline_path=BASELINE_DIR / f"baseline_{BASELINE_VERSION}.json",
        probes_path=PROBES_DIR / f"probe_set_{PROBE_SET_VERSION}.jsonl",
    )
    out = DOCS_DIR / "data.json"
    save_rollup(rollup, out)
    return out
