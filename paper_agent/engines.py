"""Chat engines. Every call is logged with the model that *actually* answered.

Two engines:
  ollama   local server (``OLLAMA_URL``), JSON mode, temperature 0.2
  copilot  GitHub Copilot SDK (``github-copilot-sdk``). The Copilot CLI accepts any model name and,
           when the account is not entitled to it, silently serves another model; only the
           ``assistant.usage`` event carries the real one. ``copilot_chat`` reads that event and
           counts it in ``SERVED`` so reports can name the serving model, not the requested one.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from .config import ollama_url

SERVED: Dict[str, int] = {}          # "<engine>:<served model>" -> calls
REQUESTED: Dict[str, int] = {}       # "<engine>:<requested model>" -> calls
_SERVED_LOCK = threading.Lock()
CALL_LOG: list = []                  # dicts: engine, requested, served, seconds, chars_in, chars_out
MAX_CALL_LOG = 2000
_WARNED: set = set()
_LAST = threading.local()


def last_served() -> Optional[str]:
    """``<engine>:<served model>`` of the most recent call on this thread."""
    return getattr(_LAST, "served", None)


def _record(engine: str, requested: str, served: str, dt: float, n_in: int, n_out: int) -> None:
    _LAST.served = f"{engine}:{served}"
    with _SERVED_LOCK:
        SERVED[f"{engine}:{served}"] = SERVED.get(f"{engine}:{served}", 0) + 1
        REQUESTED[f"{engine}:{requested}"] = REQUESTED.get(f"{engine}:{requested}", 0) + 1
        if len(CALL_LOG) < MAX_CALL_LOG:
            CALL_LOG.append({"engine": engine, "requested": requested, "served": served,
                             "seconds": round(dt, 2), "chars_in": n_in, "chars_out": n_out})
        if served != requested and engine == "copilot" and (requested, served) not in _WARNED:
            _WARNED.add((requested, served))
            print(f"[paper-agent] copilot served '{served}' for requested '{requested}' "
                  f"(account not entitled?) — provenance uses the served model", file=sys.stderr)


def served_summary() -> Dict[str, int]:
    return dict(SERVED)


# ----------------------------------------------------------------------------- ollama
def ollama_chat(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    t0 = time.time()
    body: dict = {
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{ollama_url()}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["message"]["content"]
    _record("ollama", model, data.get("model") or model, time.time() - t0, len(system) + len(user), len(text))
    return text


def ollama_models() -> list:
    try:
        r = requests.get(f"{ollama_url()}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except requests.RequestException:
        return []


# ----------------------------------------------------------------------------- copilot
JSON_INSTRUCTION = "Reply ONLY with valid JSON, no other text and no code fences."
NO_TOOLS = ["__none__"]              # a tool name that never exists -> session without tools


def _deny(_request: Any, _invocation: Any) -> dict:
    return {"kind": "denied-by-rules"}


class CopilotRuntime:
    """One Copilot CLI per process, driven by an event loop on a background thread (callers are sync)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Any = None
        self._slots: Optional[threading.BoundedSemaphore] = None

    def available(self) -> bool:
        try:
            import importlib.util
            if importlib.util.find_spec("copilot") is None:
                return False
        except (ImportError, ValueError):
            return False
        return os.getenv("COPILOT_DISABLED", "").lower() not in ("1", "true", "yes")

    def _ensure_started(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            from copilot import CopilotClient
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="copilot-sdk", daemon=True).start()
            cfg = Path(os.getenv("COPILOT_CONFIG_DIR") or Path.home() / ".paper-agent" / "copilot").expanduser()
            cfg.mkdir(parents=True, exist_ok=True)
            options: dict = {"log_level": "error", "auto_start": True, "auto_restart": True,
                             "use_logged_in_user": True, "cli_args": ["--config-dir", str(cfg)]}
            if token := os.getenv("COPILOT_GITHUB_TOKEN", "").strip():
                options["github_token"] = token
                options["use_logged_in_user"] = False
            if cli_path := os.getenv("COPILOT_CLI_PATH", "").strip():
                options["cli_path"] = cli_path
            client = CopilotClient(options)
            asyncio.run_coroutine_threadsafe(client.start(), loop).result(timeout=60)
            self._loop, self._client = loop, client
            self._slots = threading.BoundedSemaphore(max(1, int(os.getenv("COPILOT_MAX_CONCURRENCY") or 2)))
            return client

    def _submit(self, coro) -> Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        with self._lock:
            client, loop = self._client, self._loop
            self._client = self._loop = None
        if client is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(client.stop(), loop).result(timeout=10)
        except Exception:  # pragma: no cover
            pass
        loop.call_soon_threadsafe(loop.stop)

    def status(self) -> dict:
        if not self.available():
            return {"available": False, "authenticated": False}
        try:
            client = self._ensure_started()
            resp = self._submit(client.get_auth_status()).result(timeout=30)
        except Exception as exc:  # noqa: BLE001
            return {"available": True, "authenticated": False, "error": str(exc)[:200]}
        return {"available": True, "authenticated": bool(getattr(resp, "isAuthenticated", False)),
                "login": getattr(resp, "login", None),
                "token_source": "env" if os.getenv("COPILOT_GITHUB_TOKEN", "").strip() else "login"}

    def generate(self, model: str, system: str, prompt: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
        client = self._ensure_started()
        assert self._slots is not None
        with self._slots:
            return self._submit(self._generate(client, model, system, prompt, timeout)).result(timeout=timeout + 15)

    @staticmethod
    async def _generate(client: Any, model: str, system: str, prompt: str, timeout: float):
        config: dict = {"model": model, "available_tools": NO_TOOLS, "on_permission_request": _deny,
                        "streaming": False}
        if system:
            config["system_message"] = {"mode": "replace", "content": system}
        session = await client.create_session(config)
        served: dict = {}

        def _usage(event: Any) -> None:
            kind = getattr(getattr(event, "type", None), "value", getattr(event, "type", ""))
            if str(kind) == "assistant.usage":
                served["model"] = getattr(getattr(event, "data", None), "model", None)

        if callable(getattr(session, "on", None)):
            session.on(_usage)
        try:
            event = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
            content = getattr(getattr(event, "data", None), "content", None) if event else None
            return (content if isinstance(content, str) else None), served.get("model")
        finally:
            try:
                await session.destroy()
            except Exception:  # pragma: no cover
                pass


copilot_runtime = CopilotRuntime()
atexit.register(copilot_runtime.stop)


def copilot_chat(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    if not copilot_runtime.available():
        raise RuntimeError("Copilot SDK not installed or COPILOT_DISABLED (pip install 'paper-agent[copilot]')")
    t0 = time.time()
    sys_msg = (system + "\n\n" + JSON_INSTRUCTION) if json_mode else system
    text, served = copilot_runtime.generate(model, sys_msg, user, timeout)
    if text is None:
        raise ValueError("copilot returned no text")
    text = text.strip()
    if json_mode:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    _record("copilot", model, served or model, time.time() - t0, len(sys_msg) + len(user), len(text))
    return text


ENGINES: Dict[str, Callable[..., str]] = {"ollama": ollama_chat, "copilot": copilot_chat}


def chat(engine: str, model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    return ENGINES[engine](model, system, user, timeout=timeout, json_mode=json_mode)


def chat_json(engine: str, model: str, system: str, user: str, retries: int = 2, timeout: float = 600) -> Any:
    """Parsed JSON or None after ``retries`` failed attempts (network, non-JSON, schema)."""
    for attempt in range(retries + 1):
        try:
            raw = chat(engine, model, system, user, timeout=timeout, json_mode=True)
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"    [{engine}:{model}] attempt {attempt + 1} failed: {str(exc)[:120]}", file=sys.stderr)
    return None


def engine_status() -> dict:
    return {"ollama": {"url": ollama_url(), "models": ollama_models()},
            "copilot": copilot_runtime.status()}
