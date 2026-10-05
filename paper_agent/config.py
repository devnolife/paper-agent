"""Environment and defaults.

Settings come from the environment (optionally loaded from ``.env`` files in the current
working directory and its parents, nearest last so it wins). Nothing here is cached so tests
can monkeypatch ``os.environ``.

Env:
  OLLAMA_URL                http://127.0.0.1:11434
  COPILOT_GITHUB_TOKEN      fine-grained PAT with the "Copilot Requests" permission (else CLI login)
  COPILOT_CONFIG_DIR        CLI config dir (default ~/.paper-agent/copilot)
  COPILOT_MAX_CONCURRENCY   parallel Copilot requests (default 2)
  BROWSER_AGENT_URL / BROWSER_AGENT_API_KEY   optional studio-revisi browser-agent for Scopus/S2 search
  CROSSREF_EMAIL            polite-pool contact for Crossref/OpenAlex
  PAPER_AGENT_ROLES         JSON overriding model roles, e.g. {"draft": ["copilot", "gpt-5-mini"]}
  PAPER_AGENT_CACHE_DIR     HTTP cache dir (default ~/.paper-agent/cache)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

# role -> (engine, model). Roles are chosen for the account tier most users have (see README):
# the draft model writes, a *different* model post-edits, a local model back-translates for QA.
DEFAULT_ROLES: Dict[str, Tuple[str, str]] = {
    "draft": ("copilot", "gpt-5-mini"),
    "review": ("copilot", "claude-haiku-4.5"),
    "qa": ("ollama", "gemma3:27b"),
    "fallback": ("ollama", "gemma3:27b"),
    "critic": ("copilot", "claude-haiku-4.5"),
    "critic2": ("ollama", "gemma3:27b"),
    "planner": ("copilot", "gpt-5-mini"),
}


def load_env(start: Path | None = None) -> None:
    """Load ``.env`` files from ``start`` up to the filesystem root (nearest wins), then ``~/.paper-agent/.env``."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    here = (start or Path.cwd()).resolve()
    chain = [p / ".env" for p in [here, *here.parents]]
    for env in reversed(chain):            # farthest first so the nearest overrides
        if env.exists():
            load_dotenv(env, override=True)
    home_env = Path.home() / ".paper-agent" / ".env"
    if home_env.exists():
        load_dotenv(home_env, override=False)


def ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def cache_dir() -> Path:
    d = Path(os.getenv("PAPER_AGENT_CACHE_DIR") or Path.home() / ".paper-agent" / "cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def roles() -> Dict[str, Tuple[str, str]]:
    r = dict(DEFAULT_ROLES)
    raw = os.getenv("PAPER_AGENT_ROLES", "").strip()
    if raw:
        for k, v in json.loads(raw).items():
            r[k] = (str(v[0]), str(v[1]))
    return r
