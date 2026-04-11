# quantization-canary

> **🔗 Live site: https://chojeq.com/quantization-canary/**

A public, continuously-running watchdog for silent changes to **Claude Opus 4.6**.

A GitHub Actions cron queries the Anthropic API once a day, runs a fixed set of 30 prompts at `temperature=0`, and measures how much the output has drifted from a one-time pinned baseline. The site publishes a green/yellow/red verdict, a time-series chart, and a per-prompt drill-down so any visitor can answer one question at a glance: **has Opus 4.6 changed?**

The whole project is built around the value that *the dataset is the provenance*. Every probe, every calibration sample, every daily result, and every intermediate artifact lives in this repo as a git-tracked JSON file. Anyone can clone, fork, audit, or replay the full history.

---

## What you'll see on the site

| Page | What it shows |
|---|---|
| **Home** | The current verdict (🟢/🟡/🔴), the daily fingerprint-distance time series with the calibration noise band, and links to evidence + methodology. |
| **Evidence** | The 30 latest per-prompt distances, sorted by absolute z-score. Click any row to see the literal baseline-vs-today character-level diff and a "copy replay snippet" button so you can reproduce the API call yourself. |
| **Methodology** | Plain-language explainer of what the metric is, what the verdict means, what it does **not** mean, and the project's known limits. |

---

## What this site does — and does not — claim

**It claims:** to detect *dramatic* changes to Opus 4.6 served via the Anthropic API. Things like a checkpoint swap (a different model under the same ID), heavy requantization (e.g. FP16 → INT4), large fine-tuning rounds, or server-side moderation rewrites that change refusal patterns. Specifically, anything that adds roughly `≥0.10` to the aggregate normalized character-edit distance from the pinned baseline.

**It does not claim:** to detect *subtle* changes. The first calibration measured Opus 4.6's intrinsic `temperature=0` nondeterminism on long-form prose at roughly 20% of output characters per call, sample-to-sample. That's the noise floor we live with — it comes from Anthropic's serving stack (mixed-precision arithmetic, variable batching, possibly speculative decoding) and we have no way to lower it from outside. The kind of quantization update that could be slipped past a casual user is also subtle enough to hide inside that noise. We are honest about this limit instead of pretending we have superhuman drift sensitivity.

The full writeup of the noise-floor finding, with raw numbers and a verification recipe, is at [`reports/2026-04-11-calibration-noise-floor.md`](reports/2026-04-11-calibration-noise-floor.md).

We also do not try to attribute *which kind* of change occurred. From black-box API access, "quantization change," "weight swap," and "server-side moderation rewrite" are not cleanly separable. The site reports a single "something changed" verdict and shows the diff evidence — visitors decide what it means.

---

## Architecture

```
┌────────────────────────────────┐
│  GitHub Actions cron (daily)   │
│  runs src/worker.py,           │
│  then commits new files        │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│  results/YYYY-MM-DD.json       │◄── baseline/baseline_v1.json
│  docs/data.json (rollup)       │    probes/probe_set_v1.jsonl
└──────────────┬─────────────────┘    (all in-repo, git-tracked)
               │
               ▼
┌────────────────────────────────┐
│  GitHub Pages (static site)    │
│  reads docs/data.json          │
└────────────────────────────────┘
```

- **Worker** is Python + `uv` + the Anthropic SDK. All API calls go through a 50-worker thread pool, so the daily tick (90 calls) finishes in ~25 seconds and the one-time calibration (600 calls) finishes in ~2 minutes.
- **Storage** is plain JSON files committed to this repo. No database, no object store.
- **Site** is static HTML + vanilla JS + Chart.js (CDN) + jsdiff (CDN). Three pages, no framework, no build step.
- **Hosting** is GitHub Pages, served from `/docs`. The custom domain CNAME points to `chojeq.com`.
- **Secrets:** `ANTHROPIC_API_KEY` lives in GitHub Actions Secrets only. Never committed.
- **Cost ceiling:** ~$2.50/day of API spend on the daily tick. Calibration is a ~$17 one-time cost.

The single-verdict design and the honest scope are intentional, not accidents. See the [design plan](https://github.com/jeqcho/quantization-canary/blob/main/reports/2026-04-11-calibration-noise-floor.md) and the methodology page on the site for the reasoning.

---

## How to read the verdict

The methodology details are on the site, but the short version:

1. During calibration, every probe is called 20× at `temperature=0`. We pick the **medoid** sample as the reference text and measure each prompt's intrinsic noise floor `(mu_i, sigma_i)` from the distribution of distances from the other 19 samples to the medoid.
2. From those 20 simulated "stable days," we derive empirical `yellow_z` and `red_z` thresholds at the 95th and 99.5th percentiles, plus CUSUM `k` and `h` parameters scaled to the calibration `D` distribution. Thresholds are NOT hardcoded — they come from whatever the noise actually was.
3. Every daily tick calls each probe 3× at `temperature=0`, computes the median character-level Levenshtein distance to the pinned reference, then aggregates a daily `D(t)` and `Z(t)`.
4. **Two detectors run in parallel:** an *abrupt* check on `Z(today)` against the empirical thresholds, and a *slow-drift* CUSUM control chart on the `D(t)` series.
5. Verdict is the worse of the two:
   - 🟢 **Stable** — `Z` below yellow AND CUSUM not alarmed
   - 🟡 **Possible drift** — `Z` between yellow and red, OR CUSUM alarmed 1–2 consecutive days
   - 🔴 **Change detected** — `Z` above red, OR CUSUM alarmed ≥3 consecutive days

The site does **not** auto-reset its baseline. When the verdict goes and stays red, evidence stays up. A human (anyone with repo access) decides whether to run a new calibration — for example, after Anthropic publishes a model update — or leave the alarm in place as an unannounced-change flag.

---

## Repository layout

```
quantization-canary/
├── .github/workflows/
│   └── daily.yml                # Actions cron + workflow_dispatch
├── src/
│   ├── env.py                   # .env loader (override=True for local dev)
│   ├── metrics.py               # normalized char-Levenshtein, z-score, CUSUM
│   ├── probe.py                 # Anthropic API wrapper (single + 50-worker batch)
│   ├── calibrate.py             # one-time baseline builder
│   ├── worker.py                # daily tick entrypoint
│   └── render.py                # builds docs/data.json rollup
├── tests/                       # 54 unit tests
│   ├── test_metrics.py
│   ├── test_probe.py
│   ├── test_calibrate.py
│   ├── test_worker.py
│   └── test_render.py
├── probes/probe_set_v1.jsonl    # 30 hand-curated prompts
├── baseline/baseline_v1.json    # pinned reference outputs + noise floor + thresholds
├── results/                     # one JSON per daily tick (committed by Actions bot)
├── docs/                        # GitHub Pages root
│   ├── index.html               # single-page site, hash-routed sections
│   ├── style.css
│   ├── app.js                   # uses Chart.js + jsdiff (CDN)
│   └── data.json                # rollup the static site reads on page load
├── scripts/
│   ├── check_probe_signal.py    # cross-model signal check (Opus vs Sonnet)
│   └── mocked_e2e.py            # full pipeline dry run, no API spend
├── reports/
│   └── 2026-04-11-calibration-noise-floor.md
├── pyproject.toml
└── uv.lock
```

---

## Development

You'll need [`uv`](https://github.com/astral-sh/uv) and an Anthropic API key in `.env`:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
uv sync
uv run pytest                          # 54 tests, ~0.4 seconds
```

### Useful entry points

| Command | What it does | API spend |
|---|---|---|
| `uv run pytest` | Run the unit test suite | $0 |
| `uv run python scripts/mocked_e2e.py` | Synthetic end-to-end pipeline run, populates `mock_run/` and writes a fake `docs/data.json` so you can preview the site offline | $0 |
| `uv run python scripts/check_probe_signal.py` | Run all 30 probes against both Opus 4.6 and Sonnet 4.6 in parallel, flag any prompts producing byte-identical output (zero signal) | ~$1 |
| `uv run python -m src.calibrate` | Build the baseline (30 probes × 20 samples at T=0, parallelized to 50 workers) | ~$17 one-time |
| `uv run python -m src.worker` | Run one daily tick locally (30 probes × 3 samples), write `results/<today>.json`, refresh `docs/data.json` | ~$2.50 |

To preview the site locally:

```bash
cd docs && python -m http.server 8000
open http://localhost:8000
```

### `.env` is the source of truth

`src/env.py` calls `dotenv.load_dotenv(override=True)`, so the key in `.env` always wins over any `ANTHROPIC_API_KEY` exported in your shell dotfiles. This matters because it's easy to have a stale key in `~/.zshrc` and a working one in `.env` — without override, the stale one would silently win.

In GitHub Actions, no `.env` file exists. The workflow injects `ANTHROPIC_API_KEY` from a repo secret directly into the worker's environment, and `load_dotenv` is a no-op.

---

## Cost ledger

| Item | Cost |
|---|---|
| Probe-set cross-model signal check (one-time, sanity) | $1.05 |
| First real calibration (30 × 20 samples) | $17.30 |
| Local + GHA verification daily ticks (one-time, shipping) | ~$5 |
| **Total spent to ship** | **~$23** |
| Daily cost going forward | ~$2.50/day |

---

## Reproducing the methodology in another language

The metric is deliberately simple: **normalized character-level Levenshtein distance**, divided by `max(len_a, len_b)`. No tokenization step, no embeddings, no library version to pin. Any language with a Levenshtein implementation will give bit-identical results, which is the entire point of choosing character-level over a token-level proxy.

Reference: see `src/metrics.py` for the canonical implementation (one function, ~5 lines).

---

## Honest project status

The project ships, runs daily on its own, and produces real time-series data. The main caveat — and the reason you should not call it a "subtle quantization detector" — is the noise floor finding documented in `reports/2026-04-11-calibration-noise-floor.md`. We chose to ship the honest version of the project and let real production data tell us where to invest next, rather than tune the methodology to a more impressive-sounding claim that the underlying API can't actually support.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

Built with the Anthropic API and a lot of patience, by [@jeqcho](https://github.com/jeqcho) and Claude Opus 4.6 (which is, hilariously, also the model being watched).
