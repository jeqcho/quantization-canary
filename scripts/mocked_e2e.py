"""Mocked end-to-end pipeline dry run.

Builds a synthetic baseline and 10 days of synthetic daily-tick results
without making a single real Anthropic API call. Then writes a fresh
docs/data.json so the static site has data to render. Useful for
verifying the full pipeline (calibrate -> worker -> render -> site)
before spending the real ~$13 calibration budget.

Day 1-7 simulate a stable model. Day 8-10 simulate a sudden checkpoint
swap, so you can watch the verdict transition green -> red and the
CUSUM accumulate.

All synthetic artifacts are written under mock_run/ (kept out of git
via .gitignore) so production paths under baseline/, results/ stay
clean. Only docs/data.json is overwritten so the site has something
to display.

Run:
    uv run python scripts/mocked_e2e.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.calibrate import build_baseline, save_baseline  # noqa: E402
from src.probe import ProbeResult  # noqa: E402
from src.render import build_rollup, save_rollup  # noqa: E402
from src.worker import (  # noqa: E402
    replay_cusum_from_history,
    run_daily_tick,
    save_result,
)

REPO_ROOT = _REPO_ROOT
MOCK_DIR = REPO_ROOT / "mock_run"
DOCS_DATA = REPO_ROOT / "docs" / "data.json"


# Six representative synthetic probes - small enough to run instantly,
# big enough to populate the home chart and the evidence drill-down.
SYNTHETIC_PROBES = [
    {"id": "mp1", "category": "exposition", "prompt": "Explain photosynthesis in 200 words.", "max_tokens": 500},
    {"id": "mp2", "category": "reasoning", "prompt": "If a train leaves at noon...", "max_tokens": 500},
    {"id": "mp3", "category": "code", "prompt": "Write a function that reverses a string.", "max_tokens": 500},
    {"id": "mp4", "category": "creative", "prompt": "Write a haiku about debugging.", "max_tokens": 500},
    {"id": "mp5", "category": "structured", "prompt": "List 5 causes of inflation.", "max_tokens": 500},
    {"id": "mp6", "category": "echo", "prompt": "Write 200 words using these 5 rare words: ...", "max_tokens": 500},
]


def _fake_result(text: str) -> ProbeResult:
    now = datetime.now(timezone.utc)
    return ProbeResult(
        text=text,
        started_at=now,
        finished_at=now,
        input_tokens=20,
        output_tokens=len(text) // 4,
        cost_usd=0.0,
    )


def make_stable_call(probes: list[dict]):
    """Build a fake call_fn that returns slightly-varying outputs per prompt.

    Each prompt has a pool of 30 candidate outputs that share most of
    their text but differ in one of three injected "noise words". When
    the worker requests samples, we cycle through the pool — so the
    medoid found in calibration is the most common output, and daily
    ticks drawing from the same pool produce similar samples (small mu,
    nonzero sigma).
    """
    pool: dict[str, list[str]] = {}
    counters: dict[str, int] = {}
    noise_words = ["calmly", "softly", "quietly"]
    for p in probes:
        base = (
            f"Stable synthetic output for probe {p['id']}: this is a "
            f"longer paragraph that simulates a 200-token response from "
            f"the model, with some filler text repeated several times to "
            f"make the character count realistic. The model considered "
            f"the prompt {{NOISE}} before producing this measured answer. "
            f"More filler follows to bring the total length up to a few "
            f"hundred characters of text, which is roughly what real Opus "
            f"4.6 outputs at temperature zero look like for these probes."
        )
        pool[p["prompt"]] = [
            base.replace("{NOISE}", noise_words[i % 3]) for i in range(30)
        ]
        counters[p["prompt"]] = 0

    def fake_call(prompt: str, max_tokens: int) -> ProbeResult:
        idx = counters[prompt] % len(pool[prompt])
        counters[prompt] += 1
        return _fake_result(pool[prompt][idx])

    return fake_call


def make_drifted_call(probes: list[dict]):
    """Returns a fake call_fn whose outputs are radically different from
    the stable baseline - simulates a checkpoint swap or heavy quantization.
    """
    drifted: dict[str, str] = {}
    for p in probes:
        drifted[p["prompt"]] = (
            f"DRIFTED OUTPUT FOR {p['id']}: this is a completely different "
            f"response that bears no resemblance whatsoever to the stable "
            f"baseline. Imagine the model has been silently swapped for a "
            f"different checkpoint, or aggressively quantized, or fine-tuned "
            f"on a totally different objective. None of the stylistic, "
            f"structural, or content choices match the original baseline."
        )

    def fake_call(prompt: str, max_tokens: int) -> ProbeResult:
        return _fake_result(drifted[prompt])

    return fake_call


def main() -> int:
    if MOCK_DIR.exists():
        shutil.rmtree(MOCK_DIR)
    MOCK_DIR.mkdir()
    mock_results = MOCK_DIR / "results"
    mock_results.mkdir()

    # Persist the synthetic probe set so build_rollup() can read it back.
    mock_probes_path = MOCK_DIR / "probes.jsonl"
    with mock_probes_path.open("w") as f:
        for p in SYNTHETIC_PROBES:
            f.write(json.dumps(p) + "\n")

    print(f"Mock E2E starting in {MOCK_DIR}/")
    print(f"  {len(SYNTHETIC_PROBES)} synthetic probes")
    print()

    # ── 1. calibration ──
    print("Step 1/3: synthetic calibration (n_samples=20)")
    stable_call = make_stable_call(SYNTHETIC_PROBES)
    baseline = build_baseline(SYNTHETIC_PROBES, call_fn=stable_call, n_samples=20)
    mock_baseline_path = MOCK_DIR / "baseline.json"
    save_baseline(baseline, mock_baseline_path)
    print(
        f"  baseline thresholds: yellow_z={baseline['thresholds']['yellow_z']:.4f}  "
        f"red_z={baseline['thresholds']['red_z']:.4f}"
    )
    print(
        f"  CUSUM target_D={baseline['cusum']['target_D']:.4f}  "
        f"k={baseline['cusum']['k']:.6f}  h={baseline['cusum']['h']:.6f}"
    )

    # Show per-prompt noise floor
    print("  per-prompt noise floor:")
    for pid, p in baseline["prompts"].items():
        print(f"    {pid}: mu={p['mu']:.4f}  sigma={p['sigma']:.4f}")

    # ── 2. simulate 10 days of ticks ──
    print()
    print("Step 2/3: simulating 10 daily ticks (7 stable, 3 drifted)")

    # Use a fresh stable_call that resumes from a different starting offset
    # so the daily-tick samples are not always the same as calibration.
    stable_call_for_ticks = make_stable_call(SYNTHETIC_PROBES)
    drifted_call = make_drifted_call(SYNTHETIC_PROBES)

    base_date = datetime.now(timezone.utc).date() - timedelta(days=9)
    for day_offset in range(10):
        date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        is_drifted = day_offset >= 7
        call_fn = drifted_call if is_drifted else stable_call_for_ticks

        cusum_state, alarm_streak = replay_cusum_from_history(baseline, mock_results)
        result = run_daily_tick(
            SYNTHETIC_PROBES,
            baseline,
            call_fn=call_fn,
            n_samples=3,
            cusum_state=cusum_state,
            cusum_alarm_streak=alarm_streak,
        )
        # Override the auto-generated date so all 10 ticks land on
        # consecutive synthetic dates.
        result["date"] = date
        save_result(result, mock_results)

        marker = "DRIFTED" if is_drifted else "stable "
        print(
            f"  day {day_offset + 1:2d} ({date}) {marker}  "
            f"D={result['D']:.4f}  Z={result['Z']:7.3f}  "
            f"verdict={result['verdict']:<6}  "
            f"cusum={result['cusum_state']:.4f} "
            f"streak={result['cusum_alarm_streak']}"
        )

    # ── 3. build rollup and write to docs/data.json ──
    print()
    print("Step 3/3: building rollup -> docs/data.json")
    rollup = build_rollup(
        results_dir=mock_results,
        baseline_path=mock_baseline_path,
        probes_path=mock_probes_path,
    )
    save_rollup(rollup, DOCS_DATA)
    print(f"  Wrote {DOCS_DATA}")
    print(
        f"  Latest verdict: {rollup['latest']['verdict']} "
        f"(D={rollup['latest']['D']:.4f}, Z={rollup['latest']['Z']:.3f})"
    )
    print(f"  History points: {len(rollup['history'])}")
    print(
        f"  Per-prompt drill-down entries: "
        f"{len(rollup['latest']['per_prompt'])}"
    )

    print()
    print("Done. Inspect the site by serving docs/ locally:")
    print("  cd docs && python -m http.server 8000")
    print("  open http://localhost:8000")
    print()
    print(f"Mock artifacts under {MOCK_DIR}/ (gitignored).")
    print(
        "Production paths (baseline/, results/) are untouched. "
        "Only docs/data.json was overwritten."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
