"""Chat engines. Every call is logged with the model that *actually* answered.

Two engines:
  ollama   local server (``OLLAMA_URL``), JSON mode, temperature 0.2
  copilot  GitHub Copilot SDK (``github-copilot-sdk``). The Copilot CLI accepts any model name and,
           when the account is not entitled to it, silently serves another model; only the
           ``assistant.usage`` event carries the real one. ``copilot_chat`` reads that event and
           counts it in ``SERVED`` so reports can name the serving model, not the requested one.

``chat``/``chat_json`` return text/parsed JSON as before. ``chat_reply`` returns a :class:`Reply` with the same
text plus what a benchmark needs: served model, reasoning effort, token counts, Copilot AI units (nano-AIU from the
usage event), time to first token, and Ollama's own timing counters.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from .config import ollama_url

SERVED: Dict[str, int] = {}          # "<engine>:<served model>" -> calls
REQUESTED: Dict[str, int] = {}       # "<engine>:<requested model>" -> calls
_SERVED_LOCK = threading.Lock()
CALL_LOG: list = []                  # dicts: engine, requested, served, seconds, chars_in, chars_out (+ usage)
MAX_CALL_LOG = 2000
_WARNED: set = set()
_LAST = threading.local()


@dataclass
class Usage:
    """Token and cost counters of one call; ``None`` when the engine does not report the field."""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    nano_aiu: Optional[float] = None         # Copilot AI units x 1e9 (copilotUsage.totalNanoAiu)
    cost: Optional[float] = None             # Copilot usage-event "cost" (request multiplier)
    ttft_s: Optional[float] = None           # time to first token
    api_calls: int = 0                       # model requests behind this call (usage events / 1 for ollama)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def aiu(self) -> Optional[float]:
        return None if self.nano_aiu is None else self.nano_aiu / 1e9


@dataclass
class Reply:
    text: str
    engine: str
    requested: str
    served: str
    effort: Optional[str]
    seconds: float
    usage: Usage
    thinking: Optional[str] = None           # reasoning text when the engine returns it separately

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"]["aiu"] = self.usage.aiu
        return d


def last_served() -> Optional[str]:
    """``<engine>:<served model>`` of the most recent call on this thread."""
    return getattr(_LAST, "served", None)


def _record(engine: str, requested: str, served: str, dt: float, n_in: int, n_out: int,
            usage: Optional[Usage] = None, effort: Optional[str] = None) -> None:
    _LAST.served = f"{engine}:{served}"
    with _SERVED_LOCK:
        SERVED[f"{engine}:{served}"] = SERVED.get(f"{engine}:{served}", 0) + 1
        REQUESTED[f"{engine}:{requested}"] = REQUESTED.get(f"{engine}:{requested}", 0) + 1
        if len(CALL_LOG) < MAX_CALL_LOG:
            entry = {"engine": engine, "requested": requested, "served": served,
                     "seconds": round(dt, 2), "chars_in": n_in, "chars_out": n_out}
            if effort:
                entry["effort"] = effort
            if usage is not None:
                entry.update({k: v for k, v in (("input_tokens", usage.input_tokens),
                                                ("output_tokens", usage.output_tokens),
                                                ("aiu", usage.aiu)) if v is not None})
            CALL_LOG.append(entry)
        if served != requested and engine == "copilot" and (requested, served) not in _WARNED:
            _WARNED.add((requested, served))
            print(f"[paper-agent] copilot served '{served}' for requested '{requested}' "
                  f"(account not entitled?) — provenance uses the served model", file=sys.stderr)


def served_summary() -> Dict[str, int]:
    return dict(SERVED)


# ----------------------------------------------------------------------------- ollama
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
# keys of ``options`` that are request fields or local settings, not Ollama sampling options
_OLLAMA_REQUEST_KEYS = ("keep_alive", "think", "num_ctx_max")


def split_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Strip a leading ``<think>…</think>`` block (models that reason inline); returns (answer, reasoning)."""
    m = _THINK_RE.match(text or "")
    if not m:
        return text, None
    return text[m.end():], m.group(1).strip()


def ollama_reply(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True,
                 effort: Optional[str] = None, options: Optional[dict] = None) -> Reply:
    """One Ollama chat call. ``options`` may hold Ollama sampling options (``num_predict``, ``temperature``, …) and
    ``keep_alive`` (else ``OLLAMA_KEEP_ALIVE``), ``think`` (bool or "low"/"medium"/"high"), ``num_ctx_max`` (cap for
    the context window that is sized to the input). ``effort`` is recorded only; map it to ``think`` if wanted."""
    t0 = time.time()
    opts = dict(options or {})
    req = {k: opts.pop(k) for k in _OLLAMA_REQUEST_KEYS if k in opts}
    num_predict = int(opts.get("num_predict", 4096))
    # Ollama truncates prompts to num_ctx (dropping the system prompt first); size the window to the input + output
    # unless the caller fixes it (a different num_ctx makes Ollama reload the model)
    want = (len(system) + len(user)) // 3 + num_predict + 512
    num_ctx = int(opts.pop("num_ctx", 0)) or max(8192, min(int(req.get("num_ctx_max") or 32768), want))
    body: dict = {
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx, "num_predict": num_predict, **opts},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        body["format"] = "json"
    keep_alive = req.get("keep_alive", os.getenv("OLLAMA_KEEP_ALIVE") or None)
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if "think" in req:
        body["think"] = req["think"]
    r = requests.post(f"{ollama_url()}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    text, inline = split_thinking(msg.get("content") or "")
    thinking = msg.get("thinking") or inline
    ns = 1e9
    usage = Usage(input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"), api_calls=1,
                  extra={"num_ctx": num_ctx, "done_reason": data.get("done_reason"),
                         **{k: round(data[k] / ns, 3) for k in ("total_duration", "load_duration",
                                                               "prompt_eval_duration", "eval_duration") if data.get(k)}})
    served = data.get("model") or model
    dt = time.time() - t0
    _record("ollama", model, served, dt, len(system) + len(user), len(text), usage, effort)
    return Reply(text, "ollama", model, served, effort, round(dt, 3), usage, thinking)


def ollama_chat(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    return ollama_reply(model, system, user, timeout=timeout, json_mode=json_mode).text


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


def _deny(_request: Any, _invocation: Any = None) -> dict:
    return {"kind": "denied-by-rules"}


def _event_kind(event: Any) -> str:
    kind = getattr(event, "type", "")
    return str(getattr(kind, "value", kind))


def _deny_v1(_request: Any, _invocation: Any = None) -> Any:
    try:
        from copilot.generated.rpc import PermissionDecisionReject
    except ImportError:  # pragma: no cover
        return {"kind": "denied-by-rules"}
    return PermissionDecisionReject(feedback="paper-agent sessions have no tools")


def _seconds(v: Any) -> Optional[float]:
    """SDK >= 1.0 reports durations as ``timedelta``; other representations have no documented unit -> None."""
    return float(v.total_seconds()) if hasattr(v, "total_seconds") else None


def _add(a: Optional[float], b: Any) -> Optional[float]:
    if b is None:
        return a
    try:
        return (a or 0) + float(b)
    except (TypeError, ValueError):
        return a


def usage_from_events(events: list) -> Tuple[Usage, Optional[str]]:
    """Sum the ``assistant.usage`` event data of one turn (several model requests -> several events).
    Returns the usage and the model named by the *last* event (the one that produced the answer)."""
    u, served = Usage(), None
    for data in events:
        u.api_calls += 1
        served = getattr(data, "model", None) or served
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens"):
            v = _add(getattr(u, name), getattr(data, name, None))
            setattr(u, name, None if v is None else int(v))
        cu = getattr(data, "copilot_usage", None)
        u.nano_aiu = _add(u.nano_aiu, getattr(cu, "total_nano_aiu", None) if cu is not None else None)
        u.cost = _add(u.cost, getattr(data, "cost", None))
        if u.ttft_s is None:
            u.ttft_s = _seconds(getattr(data, "time_to_first_token", None))
        eff = getattr(data, "reasoning_effort", None)
        if eff:
            u.extra["served_effort"] = str(eff)
        fr = getattr(data, "finish_reason", None)
        if fr:
            u.extra["finish_reason"] = str(fr)
    return u, served


class CopilotRuntime:
    """One Copilot client per process, driven by an event loop on a background thread (callers are sync).

    SDK 0.1.x (options dicts) spawns its own CLI. SDK >= 1.0 (keyword API) either connects to an already running
    runtime (``COPILOT_RUNTIME_URL`` + ``COPILOT_CONNECTION_TOKEN``) or spawns one; sessions use ``mode="empty"``
    (no environment context, custom instructions, skills, memory or telemetry) so a prompt is exactly what we send.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Any = None
        self._slots: Optional[threading.BoundedSemaphore] = None
        self._v1 = False

    def available(self) -> bool:
        try:
            import importlib.util
            if importlib.util.find_spec("copilot") is None:
                return False
        except (ImportError, ValueError):
            return False
        return os.getenv("COPILOT_DISABLED", "").lower() not in ("1", "true", "yes")

    @staticmethod
    def _scratch() -> Path:
        d = Path(os.getenv("COPILOT_CONFIG_DIR") or Path.home() / ".paper-agent" / "copilot").expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_started(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            import copilot
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="copilot-sdk", daemon=True).start()
            cfg = self._scratch()
            token = os.getenv("COPILOT_GITHUB_TOKEN", "").strip()
            cli_path = os.getenv("COPILOT_CLI_PATH", "").strip()
            self._v1 = hasattr(copilot, "RuntimeConnection")
            if self._v1:
                from copilot import CopilotClient, RuntimeConnection
                url = os.getenv("COPILOT_RUNTIME_URL", "").strip()
                if url:
                    conn = RuntimeConnection.for_uri(url, connection_token=os.getenv("COPILOT_CONNECTION_TOKEN") or None)
                    client = CopilotClient(connection=conn, mode="empty", log_level="error")
                else:
                    kw: dict = {"connection": RuntimeConnection.for_stdio(path=cli_path or None), "mode": "empty",
                                "base_directory": str(cfg), "log_level": "error"}
                    if token:
                        kw.update(github_token=token, use_logged_in_user=False)
                    client = CopilotClient(**kw)
            else:
                from copilot import CopilotClient
                options: dict = {"log_level": "error", "auto_start": True, "auto_restart": True,
                                 "use_logged_in_user": True, "cli_args": ["--config-dir", str(cfg)]}
                if token:
                    options["github_token"] = token
                    options["use_logged_in_user"] = False
                if cli_path:
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
        source = ("runtime" if os.getenv("COPILOT_RUNTIME_URL", "").strip()
                  else "env" if os.getenv("COPILOT_GITHUB_TOKEN", "").strip() else "login")
        return {"available": True, "authenticated": bool(getattr(resp, "isAuthenticated", getattr(resp, "is_authenticated", False))),
                "login": getattr(resp, "login", None), "token_source": source}

    def generate(self, model: str, system: str, prompt: str, timeout: float,
                 effort: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Usage]:
        client = self._ensure_started()
        assert self._slots is not None
        with self._slots:
            coro = (self._generate_v1 if self._v1 else self._generate)(client, model, system, prompt, timeout, effort)
            return self._submit(coro).result(timeout=timeout + 15)

    @staticmethod
    async def _generate(client: Any, model: str, system: str, prompt: str, timeout: float, effort: Optional[str] = None):
        config: dict = {"model": model, "available_tools": NO_TOOLS, "on_permission_request": _deny,
                        "streaming": False}
        if system:
            config["system_message"] = {"mode": "replace", "content": system}
        if effort:
            config["reasoning_effort"] = effort
        session = await client.create_session(config)
        events: list = []

        def _usage(event: Any) -> None:
            if _event_kind(event) == "assistant.usage":
                events.append(getattr(event, "data", None))

        if callable(getattr(session, "on", None)):
            session.on(_usage)
        try:
            event = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
            content = getattr(getattr(event, "data", None), "content", None) if event else None
            usage, served = usage_from_events([e for e in events if e is not None])
            return (content if isinstance(content, str) else None), served, usage
        finally:
            try:
                await session.destroy()
            except Exception:  # pragma: no cover
                pass

    async def _generate_v1(self, client: Any, model: str, system: str, prompt: str, timeout: float,
                           effort: Optional[str] = None):
        kw: dict = {"model": model, "available_tools": [], "on_permission_request": _deny_v1, "streaming": False,
                    "client_name": "paper-agent", "working_directory": str(self._scratch())}
        if system:
            kw["system_message"] = {"mode": "replace", "content": system}
        if effort:
            kw["reasoning_effort"] = effort
        session = await client.create_session(**kw)
        events: list = []
        session.on(lambda ev: events.append(getattr(ev, "data", None)) if _event_kind(ev) == "assistant.usage" else None)
        try:
            event = await session.send_and_wait(prompt, timeout=timeout)
            content = getattr(getattr(event, "data", None), "content", None) if event else None
            usage, served = usage_from_events([e for e in events if e is not None])
            return (content if isinstance(content, str) else None), served, usage
        finally:
            try:
                await session.disconnect()
            except Exception:  # pragma: no cover
                pass


copilot_runtime = CopilotRuntime()
atexit.register(copilot_runtime.stop)


def copilot_reply(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True,
                  effort: Optional[str] = None, options: Optional[dict] = None) -> Reply:
    if not copilot_runtime.available():
        raise RuntimeError("Copilot SDK not installed or COPILOT_DISABLED (pip install 'paper-agent[copilot]')")
    t0 = time.time()
    sys_msg = (system + "\n\n" + JSON_INSTRUCTION) if json_mode else system
    text, served, usage = copilot_runtime.generate(model, sys_msg, user, timeout, effort)
    if text is None:
        raise ValueError("copilot returned no text")
    text = text.strip()
    if json_mode:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    dt = time.time() - t0
    _record("copilot", model, served or model, dt, len(sys_msg) + len(user), len(text), usage, effort)
    return Reply(text, "copilot", model, served or model, effort, round(dt, 3), usage)


def copilot_chat(model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    return copilot_reply(model, system, user, timeout=timeout, json_mode=json_mode).text


ENGINES: Dict[str, Callable[..., str]] = {"ollama": ollama_chat, "copilot": copilot_chat}
REPLY_ENGINES: Dict[str, Callable[..., Reply]] = {"ollama": ollama_reply, "copilot": copilot_reply}


def chat(engine: str, model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True) -> str:
    return ENGINES[engine](model, system, user, timeout=timeout, json_mode=json_mode)


def chat_reply(engine: str, model: str, system: str, user: str, timeout: float = 600, json_mode: bool = True,
               effort: Optional[str] = None, options: Optional[dict] = None) -> Reply:
    """Like :func:`chat` but returns a :class:`Reply` (served model, effort, tokens, AI units, timings)."""
    return REPLY_ENGINES[engine](model, system, user, timeout=timeout, json_mode=json_mode, effort=effort,
                                 options=options)


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
