# quantization-canary

Public fingerprint-based watchdog for silent changes to **Claude Opus 4.6**.

A GitHub Actions cron queries Opus 4.6 daily via the Anthropic API, runs a
deterministic fingerprint over a curated prompt set, and publishes a
time-series dashboard so visitors can answer a single question at a glance:
**has Opus 4.6 changed?**

Design plan: see the approved spec in the project's plan file.

## Quick start (development)

```bash
uv sync
uv run pytest
```

## Status

Scaffolding in progress. No real API calls yet.
