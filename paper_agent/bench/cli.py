"""``paper-agent bench <command>``: plan, run, score, silver, judge, human-sheet, human-ingest, report, manifest, check.

Paths: public items ``<repo>/bench/items``; private items, silver/gold labels and source snapshots in
``--private-dir`` (or ``BENCH_PRIVATE_DIR``); raw runs in ``--run-dir`` (or ``BENCH_RUN_DIR``, default
``<private-dir>/runs``). Model-written code runs with ``--python`` (or ``BENCH_PYTHON``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List, Optional

from . import TASKS


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "bench" / "items").is_dir() and (p / "paper_agent").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


def add_parser(sub) -> None:
    b = sub.add_parser("bench", help="AcademicBench: academic-writing and research-coding benchmark")
    bs = b.add_subparsers(dest="bench_cmd", required=True)

    def common(p, models: bool = True):
        p.add_argument("--private-dir", default=os.getenv("BENCH_PRIVATE_DIR"), help="private items/labels dir")
        p.add_argument("--run-dir", default=os.getenv("BENCH_RUN_DIR"), help="raw runs dir (default <private>/runs)")
        p.add_argument("--tasks", nargs="*", choices=TASKS, help="tasks (default: all)")
        if models:
            p.add_argument("--roster", default=None, help="roster YAML (default bench/models.yaml)")
            p.add_argument("--models", nargs="*", default=[], help="model specs engine:model[@effort]")
            p.add_argument("--tiers", nargs="*", default=[], choices=["local", "economy", "standard", "premium"])
            p.add_argument("--effort", default=None, help="override reasoning effort of API models that accept one")
            p.add_argument("--limit-per-task", type=int, default=0, help="same random N items per task (pilot)")
            p.add_argument("--stage", default="main")
            p.add_argument("--reps", type=int, default=1)
        p.add_argument("--python", default=os.getenv("BENCH_PYTHON"), help="interpreter for model-written code")
        p.add_argument("--offline", action="store_true")

    for name, helptext in (("plan", "count calls and estimate cost (no model calls)"),
                           ("run", "run models over items (resumable; cost-gated)")):
        p = bs.add_parser(name, help=helptext)
        common(p)
        if name == "run":
            p.add_argument("--approve-high-cost", action="store_true",
                           help="allow premium models and high reasoning effort (after reviewing `bench plan`)")
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--local-workers", type=int, default=1,
                           help="parallel requests per local model (energy then measured per run segment)")
            p.add_argument("--shared-models", nargs="*", default=[],
                           help="local models other users depend on: never more than one request at a time")
            p.add_argument("--timeout", type=float, default=900)
            p.add_argument("--budget-usd", type=float, default=None,
                           help="API models: start no new call once this run has spent this many US$ (AI units)")
    p = bs.add_parser("score", help="deterministic scoring of the latest records")
    common(p, models=False)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=4)
    p = bs.add_parser("silver", help="extraction silver labels from a cross-vendor model panel")
    common(p, models=False)
    p.add_argument("--panel", nargs="+", required=True)
    p = bs.add_parser("judge", help="cross-vendor LLM judges for prose tasks (leniency-calibrated)")
    common(p, models=False)
    p.add_argument("--judges", nargs="*", default=[])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=4)
    p = bs.add_parser("human-sheet", help="blind rating sheets for two human raters")
    common(p, models=False)
    p.add_argument("--n-prose", type=int, default=30)
    p.add_argument("--n-extraction", type=int, default=20)
    p.add_argument("--seed", type=int, default=5)
    p = bs.add_parser("human-ingest", help="read filled sheets: agreement, judge-human correlation, gold labels")
    common(p, models=False)
    p = bs.add_parser("report", help="leaderboard, summary.json, per-item CSV")
    common(p, models=False)
    p.add_argument("--out", default="experiments/bench")
    p.add_argument("--roster", default=None)
    p = bs.add_parser("manifest", help="SHA-256 manifest of public and private items")
    common(p, models=False)
    p.add_argument("--write", action="store_true")
    p = bs.add_parser("check", help="validate items, manifest, word lists and the sandbox")
    common(p, models=False)


def _dirs(args):
    from .schema import item_dirs, load_items
    root = repo_root()
    priv = Path(args.private_dir).expanduser() if args.private_dir else None
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else ((priv / "runs") if priv else root / "bench-runs")
    items = load_items(item_dirs(root / "bench" / "items", priv), args.tasks)
    return root, priv, run_dir, items


def _ctx(root: Path, priv: Optional[Path], args):
    from .tasks import Context
    return Context(repo=root, private_dir=priv, python=args.python, offline=getattr(args, "offline", False))


def _models(root: Path, args) -> List:
    from .models import load_roster, select
    roster = load_roster(Path(args.roster) if args.roster else root / "bench" / "models.yaml")
    models = select(roster, args.models, args.tiers, args.effort)
    if not models:
        raise SystemExit("no models selected: use --tiers and/or --models")
    return models


def dispatch(args, log: Callable = print) -> dict:
    root, priv, run_dir, items = _dirs(args)
    ctx = _ctx(root, priv, args)
    cmd = args.bench_cmd
    if cmd in ("plan", "run"):
        from .runner import RunConfig, plan, run
        models = _models(root, args)
        cfg = RunConfig(run_dir=run_dir, stage=args.stage, limit_per_task=args.limit_per_task or None, reps=args.reps,
                        approve_high_cost=getattr(args, "approve_high_cost", False),
                        workers=getattr(args, "workers", 4), timeout=getattr(args, "timeout", 900),
                        local_workers=getattr(args, "local_workers", 1),
                        budget_usd=getattr(args, "budget_usd", None),
                        shared_models=tuple(getattr(args, "shared_models", []) or ()))
        if cmd == "plan":
            p = plan(items, models, ctx, cfg)
            for r in p["rows"]:
                log(f"  {r['model']:44} {r['tier']:9} calls={r['calls']:4}  est AIU={r['est_aiu']}  US$={r['est_usd']}"
                    + ("  [needs --approve-high-cost]" if r["high_cost"] and r["calls"] else ""))
            log(f"total: {p['items']} items x {p['models']} models; est {p['est_aiu']} AIU ≈ US${p['est_usd']}")
            return p
        return run(items, models, ctx, cfg, log=log)
    if cmd == "score":
        from .score import score_runs
        return score_runs(items, ctx, run_dir, args.models or None, args.tasks, args.workers, log=log)
    if cmd == "silver":
        from .score import build_silver
        if priv is None:
            raise SystemExit("--private-dir is required")
        return build_silver(items, run_dir, args.panel, priv, log=log)
    if cmd == "judge":
        from .judge import run_judges
        return run_judges(items, ctx, run_dir, args.judges or None, args.models or None, args.workers, log=log)
    if cmd == "human-sheet":
        from .human import make_sheets
        return make_sheets(items, ctx, run_dir, args.n_prose, args.n_extraction, args.seed, log=log)
    if cmd == "human-ingest":
        from .human import ingest
        return ingest(items, ctx, run_dir, log=log)
    if cmd == "report":
        from .models import load_roster
        from .report import write_report
        roster = load_roster(Path(args.roster) if args.roster else root / "bench" / "models.yaml")
        names = {}
        for m in roster:
            names[m.label] = m.name
            names[f"{m.spec.engine}:{m.spec.model}"] = m.name
            for eff in ("low", "medium", "high", "xhigh", "max"):
                names.setdefault(f"{m.spec.engine}:{m.spec.model}@{eff}", f"{m.name} ({eff})" if eff != "low" else m.name)
        out = Path(args.out)
        return write_report(items, run_dir, out if out.is_absolute() else root / out, names, log=log)
    if cmd == "manifest":
        from .schema import build_manifest, check_manifest
        path = root / "bench" / "MANIFEST.json"
        if args.write:
            m = build_manifest(items, note="public items in bench/items; private items are hashes only")
            path.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            log(f"wrote {path}: {m['n_items']} items")
            return m
        problems = check_manifest(json.loads(path.read_text(encoding="utf-8")), items) if path.exists() else ["no manifest"]
        for p in problems:
            log(f"  {p}")
        return {"ok": not problems, "problems": problems}
    if cmd == "check":
        from . import sandbox
        from .tasks import registry
        reg = registry()
        problems = []
        for it in items:
            try:
                reg[it.task].prompt(it, ctx)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{it.id}: {exc}")
        counts = {}
        for it in items:
            counts[f"{it.task}/{it.privacy}"] = counts.get(f"{it.task}/{it.privacy}", 0) + 1
        res = {"items": counts, "sandbox": sandbox.available(), "scope": sandbox.scope_ok(), "problems": problems[:50]}
        log(json.dumps(res, ensure_ascii=False, indent=1))
        return res
    raise SystemExit(f"unknown bench command {cmd}")


def build_standalone() -> argparse.ArgumentParser:   # pragma: no cover - convenience for python -m
    ap = argparse.ArgumentParser(prog="paper-agent-bench")
    add_parser(ap.add_subparsers(dest="cmd", required=True))
    return ap
