"""Run models over benchmark items: resumable, cost-gated, one local model at a time, everything logged.

Records go to ``<run_dir>/raw/<model-slug>.jsonl`` (one line per call: item, served model, effort, tokens, AI units,
seconds, measured GPU energy for local models, the answer text, follow-up turns and sandbox results). A record is
reused when an error-free one exists for the same item, repetition and prompt hash, so an interrupted run resumes
and a changed prompt is run again. API models run concurrently (``workers``); local models run one after another on
the shared Ollama server, each unloaded (short ``keep_alive``) before the next one is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from .. import engines
from ..config import ollama_url
from . import power
from .cost import Prices
from .models import ModelInfo
from .schema import Item
from .tasks import Context, Task, registry

DEFAULT_AIU_PER_CALL = {"economy": 0.5, "standard": 2.0, "premium": 6.0}
KEEP_ALIVE = "60s"


class GateError(RuntimeError):
    """High-cost models selected without explicit approval."""


@dataclass
class RunConfig:
    run_dir: Path
    stage: str = "main"
    approve_high_cost: bool = False
    workers: int = 4
    limit_per_task: Optional[int] = None
    reps: int = 1
    timeout: float = 900
    retries: int = 2
    seed: int = 7
    local_workers: int = 1        # parallel requests per local model (energy is then measured per run segment)
    shared_models: tuple = ()      # local models other users depend on: always one request at a time
    budget_usd: Optional[float] = None   # API runs: start no new call once this run has spent the budget


def prompt_sha(system: str, user: str, json_mode: bool) -> str:
    return hashlib.sha256(f"{int(json_mode)}\x00{system}\x00{user}".encode("utf-8")).hexdigest()[:16]


def select_items(items: List[Item], limit_per_task: Optional[int], seed: int = 7) -> List[Item]:
    """All items, or a fixed random subset of ``limit_per_task`` per task (same subset for every model)."""
    if not limit_per_task:
        return list(items)
    out = []
    for task in sorted({i.task for i in items}):
        group = sorted((i for i in items if i.task == task), key=lambda i: i.id)
        random.Random(f"{seed}:{task}").shuffle(group)
        out += sorted(group[:limit_per_task], key=lambda i: i.id)
    return out


# ----------------------------------------------------------------------------- record store
class Store:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "raw"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._glock = threading.Lock()

    def path(self, model: ModelInfo) -> Path:
        return self.dir / f"{model.spec.slug}.jsonl"

    def records(self, model: ModelInfo) -> List[dict]:
        p = self.path(model)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a line cut by a crash
        return out

    def done(self, model: ModelInfo) -> Dict[Tuple[str, int, str], dict]:
        return {(r["item"], r.get("rep", 0), r.get("prompt_sha", "")): r for r in self.records(model) if not r.get("error")}

    def append(self, model: ModelInfo, rec: dict) -> None:
        with self._glock:
            lock = self._locks.setdefault(model.spec.slug, threading.Lock())
        with lock, self.path(model).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def latest_records(run_dir: Path) -> Dict[str, Dict[Tuple[str, int], dict]]:
    """{model label: {(item, rep): last record}} over all raw files (error records only when nothing better)."""
    out: Dict[str, Dict[Tuple[str, int], dict]] = {}
    for p in sorted((Path(run_dir) / "raw").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r["item"], r.get("rep", 0))
            cur = out.setdefault(r["model"], {}).get(k)
            if cur is None or not r.get("error") or cur.get("error"):
                out[r["model"]][k] = r
    return out


# ----------------------------------------------------------------------------- planning & gate
def history_aiu(run_dir: Path) -> Dict[str, List[float]]:
    """AI units per call observed so far, per model label and per tier (keys ``tier:<tier>``)."""
    hist: Dict[str, List[float]] = {}
    for model, recs in latest_records(run_dir).items():
        for r in recs.values():
            aiu = (r.get("usage") or {}).get("aiu")
            if aiu is None or r.get("error"):
                continue
            tot = aiu + sum((f.get("usage") or {}).get("aiu") or 0 for f in r.get("followups") or [])
            hist.setdefault(model, []).append(tot)
            hist.setdefault(f"tier:{r.get('tier')}", []).append(tot)
    return hist


def plan(items: List[Item], models: List[ModelInfo], ctx: Context, cfg: RunConfig) -> dict:
    tasks = registry()
    store = Store(cfg.run_dir)
    hist = history_aiu(cfg.run_dir)
    prices = Prices.from_env()
    sel = select_items(items, cfg.limit_per_task, cfg.seed)
    rows, total_aiu = [], 0.0
    for m in models:
        done = store.done(m)
        todo = 0
        for it in sel:
            try:
                system, user = tasks[it.task].prompt(it, ctx)
            except Exception:  # noqa: BLE001 - e.g. a source not fetched yet; count as to-do
                todo += cfg.reps
                continue
            sha = prompt_sha(system, user, tasks[it.task].json_mode)
            todo += sum(1 for rep in range(cfg.reps) if (it.id, rep, sha) not in done)
        per_call = None
        if m.spec.engine == "copilot":
            obs = hist.get(m.label) or hist.get(f"tier:{m.tier}")
            per_call = sum(obs) / len(obs) if obs else DEFAULT_AIU_PER_CALL.get(m.tier, 2.0)
            if m.spec.effort in ("high", "xhigh", "max"):
                per_call *= 3.0 if not hist.get(m.label) else 1.0
        est = None if per_call is None else per_call * todo
        total_aiu += est or 0.0
        rows.append({"model": m.label, "tier": m.tier, "family": m.family, "high_cost": m.high_cost, "calls": todo,
                     "aiu_per_call": None if per_call is None else round(per_call, 3),
                     "est_aiu": None if est is None else round(est, 1),
                     "est_usd": None if est is None else round(prices.api_usd(est), 2)})
    return {"items": len(sel), "models": len(models), "rows": rows, "est_aiu": round(total_aiu, 1),
            "est_usd": round(prices.api_usd(total_aiu), 2), "prices": prices.describe(),
            "needs_approval": [r["model"] for r in rows if r["high_cost"] and r["calls"]]}


def check_gate(p: dict, cfg: RunConfig) -> None:
    if p["needs_approval"] and not cfg.approve_high_cost:
        raise GateError(f"high-cost models selected ({', '.join(p['needs_approval'])}); estimated "
                        f"{p['est_aiu']} AIU ≈ US${p['est_usd']}. Re-run with --approve-high-cost after approval.")


# ----------------------------------------------------------------------------- ollama helpers
def ollama_ps() -> List[dict]:
    try:
        r = requests.get(f"{ollama_url()}/api/ps", timeout=10)
        r.raise_for_status()
        return r.json().get("models", []) or []
    except (requests.RequestException, ValueError):
        return []


def wait_unloaded(model: str, timeout: float = 180, log: Callable = print) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(m.get("name") == model or m.get("model") == model for m in ollama_ps()):
            return True
        time.sleep(5)
    log(f"  {model} still loaded after {timeout:.0f} s (another user may be using it)")
    return False


# ----------------------------------------------------------------------------- one call
def _call(model: ModelInfo, task: Task, system: str, user: str, cfg: RunConfig) -> engines.Reply:
    last: Optional[Exception] = None
    for attempt in range(cfg.retries + 1):
        try:
            return engines.chat_reply(model.spec.engine, model.spec.model, system, user, timeout=cfg.timeout,
                                      json_mode=task.json_mode, effort=model.spec.effort,
                                      options={**model.engine_options(task.max_tokens),
                                               **({"keep_alive": KEEP_ALIVE} if model.spec.engine == "ollama" else {})})
        except Exception as exc:  # noqa: BLE001 - network, rate limit, empty answer
            last = exc
            time.sleep(min(60, 5 * 3 ** attempt))
    raise last  # type: ignore[misc]


def record_aiu(rec: dict) -> float:
    """AI units of a record including its follow-up turns (0 for local models and failed calls)."""
    turns = [rec] + list(rec.get("followups") or [])
    return float(sum((t.get("usage") or {}).get("aiu") or 0.0 for t in turns))


def _reply_fields(rep: engines.Reply) -> dict:
    return {"requested": rep.requested, "served": rep.served, "seconds": rep.seconds,
            "usage": {**{k: v for k, v in rep.usage.__dict__.items() if k != "extra"}, "aiu": rep.usage.aiu,
                      "extra": rep.usage.extra},
            "thinking_chars": len(rep.thinking or "")}


def run_one(model: ModelInfo, it: Item, rep: int, task: Task, ctx: Context, cfg: RunConfig,
            energy: Optional[Callable[[float, float], dict]] = None) -> dict:
    system, user = task.prompt(it, ctx)
    rec: dict = {"item": it.id, "task": it.task, "rep": rep, "model": model.label, "engine": model.spec.engine,
                 "effort": model.spec.effort, "tier": model.tier, "family": model.family, "stage": cfg.stage,
                 "prompt_sha": prompt_sha(system, user, task.json_mode), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    try:
        r = _call(model, task, system, user, cfg)
        rec.update(_reply_fields(r), text=r.text, error=None)
        follow, info = task.followup(it, r.text, ctx)
        rec.update(info)
        if follow:
            r2 = _call(model, task, system, follow, cfg)
            rec["followups"] = [{**_reply_fields(r2), "prompt_sha": prompt_sha(system, follow, task.json_mode),
                                 "text": r2.text}]
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {exc}"[:500], text=rec.get("text", ""))
    t1 = time.time()
    rec["wall_s"] = round(t1 - t0, 3)
    if energy is not None:
        rec["energy"] = energy(t0, t1)
    return rec


# ----------------------------------------------------------------------------- runs
def _todo(model: ModelInfo, items: List[Item], tasks: Dict[str, Task], ctx: Context, cfg: RunConfig,
          store: Store, log: Callable) -> List[Tuple[Item, int]]:
    done = store.done(model)
    jobs = []
    for it in items:
        try:
            system, user = tasks[it.task].prompt(it, ctx)
        except Exception as exc:  # noqa: BLE001
            log(f"  skip {it.id}: prompt unavailable ({exc})")
            continue
        sha = prompt_sha(system, user, tasks[it.task].json_mode)
        jobs += [(it, rep) for rep in range(cfg.reps) if (it.id, rep, sha) not in done]
    return jobs


def run_api(models: List[ModelInfo], items: List[Item], ctx: Context, cfg: RunConfig, store: Store,
            log: Callable) -> dict:
    tasks = registry()
    os.environ["COPILOT_MAX_CONCURRENCY"] = str(max(int(os.getenv("COPILOT_MAX_CONCURRENCY") or 0), cfg.workers))
    queues = {m.label: [(m, it, rep) for it, rep in _todo(m, items, tasks, ctx, cfg, store, log)] for m in models}
    jobs = []
    while any(queues.values()):                 # round-robin across models spreads load over vendors
        for q in queues.values():
            if q:
                jobs.append(q.pop(0))
    budget_aiu = None if cfg.budget_usd is None else cfg.budget_usd / Prices.from_env().usd_per_aiu
    log(f"API: {len(jobs)} calls over {len(models)} models, {cfg.workers} workers"
        + ("" if budget_aiu is None else f", budget US${cfg.budget_usd:.2f} ({budget_aiu:.0f} AIU)"))
    n_ok = n_err = n_skip = 0
    aiu = 0.0
    stop = threading.Event()
    spent_lock = threading.Lock()
    spent = [0.0]

    def guarded(m: ModelInfo, it: Item, rep: int) -> Optional[dict]:
        # spend is counted in the worker, so the next call sees the cap at once; calls already running when it is
        # reached finish (overshoot <= workers - 1 calls)
        if stop.is_set():
            return None
        rec = run_one(m, it, rep, tasks[it.task], ctx, cfg)
        if budget_aiu is not None:
            with spent_lock:
                spent[0] += record_aiu(rec)
                if spent[0] >= budget_aiu:
                    stop.set()
        return rec

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {pool.submit(guarded, m, it, rep): (m, it) for m, it, rep in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            m, it = futs[fut]
            rec = fut.result()
            if rec is None:
                n_skip += 1
                continue
            store.append(m, rec)
            if rec.get("error"):
                n_err += 1
                log(f"  [{i}/{len(jobs)}] {m.label} {it.id} ERROR {rec['error'][:120]}")
            else:
                n_ok += 1
                aiu += record_aiu(rec)
                if i % 25 == 0 or i == len(jobs):
                    log(f"  [{i}/{len(jobs)}] ok={n_ok} err={n_err} aiu={aiu:.2f}")
    if stop.is_set():
        log(f"  budget reached: {spent[0]:.1f} AIU >= {budget_aiu:.0f} AIU; {n_skip} calls not started")
    return {"calls": len(jobs), "ok": n_ok, "errors": n_err, "skipped": n_skip, "aiu": round(aiu, 3),
            "stopped_by_budget": stop.is_set()}


def run_local(models: List[ModelInfo], items: List[Item], ctx: Context, cfg: RunConfig, store: Store,
              log: Callable) -> dict:
    tasks = registry()
    sampler = power.GpuSampler()
    sampling = sampler.start()
    summary = {"calls": 0, "ok": 0, "errors": 0, "models": {}}
    try:
        for m in models:
            jobs = _todo(m, items, tasks, ctx, cfg, store, log)
            if not jobs:
                log(f"{m.label}: nothing to do")
                continue
            log(f"{m.label}: {len(jobs)} calls")
            wait_unloaded(m.spec.model, timeout=30, log=log)
            before = power.snapshot()
            gpus_all = sorted(before)
            baseline = {g: sampler.baseline(g, 6.0) for g in gpus_all} if sampling else {}
            util = {g: sampler.utilisation(g) for g in gpus_all} if sampling else {}
            t_load = time.time()
            try:   # load the model with a tiny request; its GPU is where memory grew
                engines.chat_reply("ollama", m.spec.model, "", "ok", timeout=600, json_mode=False,
                                   options={**m.engine_options(8), "keep_alive": KEEP_ALIVE, "num_predict": 8})
            except Exception as exc:  # noqa: BLE001
                log(f"  load failed: {exc}")
            load_s = time.time() - t_load
            after = power.snapshot()
            gpus = power.model_gpus(before, after)
            if not gpus:
                ps = next((x for x in ollama_ps() if x.get("name") == m.spec.model or x.get("model") == m.spec.model), {})
                gpus = power.ollama_gpus(ps.get("size_vram"))
            log(f"  loaded in {load_s:.1f} s on GPU {gpus or '?'}; idle baseline W {baseline}; util% {util}")
            meta = {"gpus": gpus, "baseline_w": {str(g): baseline.get(g) for g in gpus_all}, "load_s": round(load_s, 1),
                    "util_before": {str(g): util.get(g) for g in gpus_all}}

            def energy(t0: float, t1: float) -> dict:
                if not sampling:
                    return {**meta, "gross_j": None, "marginal_j": None}
                # every GPU's window is kept: on a shared server the model may sit on a GPU other than the one
                # detected at load time, so the attribution can be revisited in the analysis
                per_gpu = {str(g): sampler.window(g, t0, t1) for g in gpus_all}
                if not gpus:
                    return {**meta, "gross_j": None, "marginal_j": None, "per_gpu": per_gpu}
                gross, marg = sampler.energy_j(gpus, t0, t1, {g: baseline.get(g) or 0.0 for g in gpus})
                return {**meta, "gross_j": None if gross is None else round(gross, 1),
                        "marginal_j": None if marg is None else round(marg, 1), "per_gpu": per_gpu}

            ok = err = 0
            workers = 1 if m.spec.model in cfg.shared_models else max(1, cfg.local_workers)
            t_seg = time.time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # one request at a time: energy per call; in parallel, calls overlap and energy is per segment
                futs = {pool.submit(run_one, m, it, rep, tasks[it.task], ctx, cfg, energy if workers == 1 else None):
                        it for it, rep in jobs}
                for i, fut in enumerate(as_completed(futs), 1):
                    it = futs[fut]
                    rec = fut.result()
                    store.append(m, rec)
                    if rec.get("error"):
                        err += 1
                        log(f"  [{i}/{len(jobs)}] {it.id} ERROR {rec['error'][:120]}")
                    else:
                        ok += 1
                    if i % 20 == 0 or i == len(jobs):
                        log(f"  [{i}/{len(jobs)}] ok={ok} err={err}")
            t_end = time.time()
            seg = {"model": m.label, "t0": round(t_seg, 3), "t1": round(t_end, 3), "calls": len(jobs), "ok": ok,
                   "workers": workers, "stage": cfg.stage, **meta}
            if sampling:
                seg["per_gpu"] = {str(g): sampler.window(g, t_seg, t_end) for g in gpus_all}
                if gpus:
                    gross, marg = sampler.energy_j(gpus, t_seg, t_end, {g: baseline.get(g) or 0.0 for g in gpus})
                    seg.update(gross_j=None if gross is None else round(gross, 1),
                               marginal_j=None if marg is None else round(marg, 1))
            edir = Path(cfg.run_dir) / "energy"
            edir.mkdir(parents=True, exist_ok=True)
            with (edir / f"{m.spec.slug}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            summary["calls"] += len(jobs)
            summary["ok"] += ok
            summary["errors"] += err
            summary["models"][m.label] = {"ok": ok, "errors": err, **meta}
            wait_unloaded(m.spec.model, timeout=240, log=log)
    finally:
        sampler.stop()
    return summary


def run(items: List[Item], models: List[ModelInfo], ctx: Context, cfg: RunConfig, log: Callable = print) -> dict:
    p = plan(items, models, ctx, cfg)
    check_gate(p, cfg)
    sel = select_items(items, cfg.limit_per_task, cfg.seed)
    store = Store(cfg.run_dir)
    api = [m for m in models if m.spec.engine == "copilot"]
    local = [m for m in models if m.spec.engine == "ollama"]
    out = {"plan": p}
    if api:
        out["api"] = run_api(api, sel, ctx, cfg, store, log)
    if local:
        out["local"] = run_local(local, sel, ctx, cfg, store, log)
    return out
