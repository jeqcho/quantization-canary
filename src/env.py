"""Environment loading helper.

All entry points (worker, calibrate, scripts) should call `load_env()`
before instantiating an Anthropic client. This ensures the repo's local
`.env` is the source of truth for local development, even when the user's
shell already has an ANTHROPIC_API_KEY exported from their dotfiles
(which may be a different account with no credits).

In GitHub Actions, no `.env` file exists; `load_dotenv` is a no-op, and
the secret-provided env var is used as-is.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the repo's .env into os.environ, overriding any shell-set values.

    Idempotent: safe to call multiple times. No-op in CI where .env doesn't exist.
    """
    env_path = _REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
