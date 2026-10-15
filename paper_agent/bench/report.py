"""Aggregate scores, judge ratings, costs and provenance into the leaderboard (``summary.json`` + Markdown + CSV).

Item score: deterministic score, or for judged tasks 0.5 x auto + 0.5 x mean judge rating (normalised to 0–1,
judge leniency removed); a
model error counts as 0; items the scorer could not grade (e.g. no silver label) are left out. Task score = mean
over items; composites = unweighted mean of task scores (academic: six tasks; coding: code and LaTeX; overall: all
eight). CIs: percentile bootstrap over items, stratified by task for composites. Cost: AI units -> USD for API
models; measured GPU energy and GPU time (-> rental-equivalent USD) for local models.
"""
from __future__ import annotations

import csv
import io
import json
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import ACADEMIC_TASKS, CODING_TASKS, TASKS
from . import stats as S
from .cost import Prices, joules_to_kwh
from .judge import leniency_offsets, load_judgements
from .runner import latest_records
from .schema import Item
from .score import answer_hash, load_scores

JUDGE_WEIGHT = 0.5


def item_scores(items: List[Item], run_dir: Path, use_judges: bool = True) -> Dict[str, Dict[str, dict]]:
    """{model: {item_id: {"score", "auto", "judge", "task"}}} averaged over repetitions."""
    by_id = {it.id: it for it in items}
    scores = load_scores(run_dir)
    judg = load_judgements(run_dir) if use_judges else {}
    offsets = leniency_offsets([r for rows in judg.values() for r in rows])
    recs = latest_records(run_dir)
    out: Dict[str, Dict[str, dict]] = {}
    for model, rows in scores.items():
        acc: Dict[str, List[dict]] = {}
        for (iid, rep), row in rows.items():
            it = by_id.get(iid)
            if it is None:
                continue
            if row.get("error") and row.get("score") is None and not row.get("answer_hash"):
                auto: Optional[float] = 0.0                     # the model failed to answer
            else:
                auto = row.get("score")
            if auto is None:
                continue
            judge = None
            rec = recs.get(model, {}).get((iid, rep))
            if use_judges and rec is not None and not rec.get("error"):
                h = answer_hash(rec)
                js = [min(1.0, max(0.0, j["score"] - offsets.get(j["judge"], 0.0)))
                      for j in judg.get((model, iid, rep), []) if j.get("answer_hash") == h and j.get("score") is not None]
                judge = sum(js) / len(js) if js else None
            final = auto if judge is None else (1 - JUDGE_WEIGHT) * auto + JUDGE_WEIGHT * judge
            acc.setdefault(iid, []).append({"score": final, "auto": auto, "judge": judge, "task": it.task})
        out[model] = {iid: {"task": v[0]["task"], "score": statistics.mean(x["score"] for x in v),
                            "auto": statistics.mean(x["auto"] for x in v),
                            "judge": (statistics.mean(x["judge"] for x in v if x["judge"] is not None)
                                      if any(x["judge"] is not None for x in v) else None), "n_reps": len(v)}
                      for iid, v in acc.items()}
    return out


def composite(task_means: Dict[str, Optional[float]], tasks) -> Optional[float]:
    vals = [task_means.get(t) for t in tasks]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def stratified_ci(per_task: Dict[str, List[float]], tasks, n_boot: int = 2000, seed: int = 0
                  ) -> Tuple[Optional[float], Optional[float]]:
    if any(not per_task.get(t) for t in tasks):
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        means = []
        for t in tasks:
            xs = per_task[t]
            means.append(sum(xs[rng.randrange(len(xs))] for _ in range(len(xs))) / len(xs))
        vals.append(sum(means) / len(means))
    vals.sort()
    return S._percentile(vals, 0.025), S._percentile(vals, 0.975)


def energy_segments(run_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for p in sorted((Path(run_dir) / "energy").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                sg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sg.get("calls"):
                out.setdefault(sg["model"], []).append(sg)
    return out


def costs(run_dir: Path, prices: Prices) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for model, recs in latest_records(run_dir).items():
        c = {"calls": 0, "errors": 0, "aiu": 0.0, "in_tokens": 0, "out_tokens": 0, "seconds": [], "gross_j": 0.0,
             "marginal_j": 0.0, "energy_calls": 0, "wall_s": 0.0, "timed_calls": 0, "served": {}, "substituted": 0,
             "engine": None,
             "tier": None, "family": None}
        for r in recs.values():
            c["engine"], c["tier"], c["family"] = r.get("engine"), r.get("tier"), r.get("family")
            c["calls"] += 1
            if r.get("error"):
                c["errors"] += 1
                continue
            turns = [r] + list(r.get("followups") or [])
            for t in turns:
                u = t.get("usage") or {}
                c["aiu"] += u.get("aiu") or 0.0
                c["in_tokens"] += u.get("input_tokens") or 0
                c["out_tokens"] += u.get("output_tokens") or 0
                served = t.get("served") or r.get("requested")
                c["served"][served] = c["served"].get(served, 0) + 1
                if served and t.get("requested") and served != t.get("requested"):
                    c["substituted"] += 1
            c["seconds"].append(r.get("seconds") or 0.0)
            if "energy" in r:                    # sequential local call: its own window is GPU time
                c["wall_s"] += r.get("wall_s") or 0.0
                c["timed_calls"] += 1
            e = r.get("energy") or {}
            if e.get("gross_j") is not None:
                c["gross_j"] += e["gross_j"]
                c["marginal_j"] += e.get("marginal_j") or 0.0
                c["energy_calls"] += 1
        ok = c["calls"] - c["errors"]
        local = c["engine"] == "ollama"
        # parallel local runs have no per-call energy: GPU time and energy come from their run segments
        for sg in energy_segments(run_dir).get(model, []) if local else []:
            if sg.get("workers", 1) <= 1:
                continue                         # sequential segment: already counted call by call
            c["wall_s"] += sg["t1"] - sg["t0"]
            c["timed_calls"] += sg["calls"]
            if sg.get("gross_j") is not None:
                c["gross_j"] += sg["gross_j"]
                c["marginal_j"] += sg.get("marginal_j") or 0.0
                c["energy_calls"] += sg["calls"]
        kwh_g, kwh_m = joules_to_kwh(c["gross_j"]), joules_to_kwh(c["marginal_j"])
        out[model] = {"engine": c["engine"], "tier": c["tier"], "family": c["family"], "calls": c["calls"],
                      "errors": c["errors"], "aiu": round(c["aiu"], 3), "usd": round(prices.api_usd(c["aiu"]), 4),
                      "usd_per_100": round(prices.api_usd(c["aiu"]) * 100 / ok, 4) if ok and not local else None,
                      "in_tokens": c["in_tokens"], "out_tokens": c["out_tokens"],
                      "median_s": round(statistics.median(c["seconds"]), 2) if c["seconds"] else None,
                      "served": c["served"], "substituted_turns": c["substituted"]}
        if local:
            gpu_h = c["wall_s"] / 3600
            rental = prices.rental_usd(c["wall_s"])
            out[model].update({"kwh_gross": round(kwh_g, 5) if c["energy_calls"] else None,
                               "kwh_marginal": round(kwh_m, 5) if c["energy_calls"] else None,
                               "kwh_per_100": round(kwh_g * 100 / c["energy_calls"], 5) if c["energy_calls"] else None,
                               "energy_cost": prices.energy_cost(kwh_g) if c["energy_calls"] else None,
                               "gpu_hours": round(gpu_h, 4), "rental_usd": None if rental is None else round(rental, 4),
                               "timed_calls": c["timed_calls"], "energy_calls": c["energy_calls"],
                               # per 100 calls whose GPU time was measured (a run killed mid-way leaves calls untimed)
                               "usd_per_100": (round(rental * 100 / c["timed_calls"], 4)
                                               if rental is not None and c["timed_calls"] else None)})
    return out


def build_summary(items: List[Item], run_dir: Path, names: Optional[Dict[str, str]] = None, top_k: int = 6,
                  use_judges: bool = True) -> dict:
    prices = Prices.from_env()
    per = item_scores(items, run_dir, use_judges)
    cst = costs(run_dir, prices)
    models = {}
    for model, rows in per.items():
        per_task: Dict[str, List[float]] = {}
        auto_task: Dict[str, List[float]] = {}
        judge_task: Dict[str, List[float]] = {}
        for r in rows.values():
            per_task.setdefault(r["task"], []).append(r["score"])
            auto_task.setdefault(r["task"], []).append(r["auto"])
            if r["judge"] is not None:
                judge_task.setdefault(r["task"], []).append(r["judge"])
        tasks = {}
        for t in TASKS:
            xs = per_task.get(t) or []
            lo, hi = S.bootstrap_ci(xs, seed=1) if xs else (None, None)
            tasks[t] = {"n": len(xs), "mean": None if not xs else round(sum(xs) / len(xs), 4),
                        "ci": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
                        "auto": None if not auto_task.get(t) else round(statistics.mean(auto_task[t]), 4),
                        "judge": None if not judge_task.get(t) else round(statistics.mean(judge_task[t]), 4),
                        "n_judged": len(judge_task.get(t) or [])}
        means = {t: tasks[t]["mean"] for t in TASKS}
        comp = {}
        for name, group in (("academic", ACADEMIC_TASKS), ("coding", CODING_TASKS), ("overall", TASKS)):
            v = composite(means, group)
            lo, hi = stratified_ci(per_task, group, seed=2) if v is not None else (None, None)
            comp[name] = {"mean": None if v is None else round(v, 4),
                          "ci": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)]}
        info = cst.get(model, {})
        models[model] = {"name": (names or {}).get(model, model), "tier": info.get("tier"), "family": info.get("family"),
                         "engine": info.get("engine"), "tasks": tasks, "composite": comp, "cost": info}
    ranked = sorted((m for m in models if models[m]["composite"]["overall"]["mean"] is not None),
                    key=lambda m: -models[m]["composite"]["overall"]["mean"])
    pairwise = []
    top = ranked[:top_k]
    raw = []
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            common = sorted(set(per[a]) & set(per[b]))
            xa, xb = [per[a][k]["score"] for k in common], [per[b][k]["score"] for k in common]
            w, sg = S.wilcoxon(xa, xb), S.sign_test(xa, xb)
            raw.append(w["p"])
            pairwise.append({"a": a, "b": b, "n": len(common), "mean_diff": round((sum(xa) - sum(xb)) / len(common), 4)
                             if common else None, "wilcoxon_p": round(w["p"], 5), "sign_p": round(sg["p"], 5)})
    for row, adj in zip(pairwise, S.holm(raw) if raw else []):
        row["wilcoxon_p_holm"] = round(adj, 5)
    pts = []
    for m in ranked:
        c = models[m]["cost"].get("usd_per_100")
        if c is not None:
            pts.append((c, models[m]["composite"]["overall"]["mean"], m))
    counts: Dict[str, Dict[str, int]] = {}
    for it in items:
        counts.setdefault(it.task, {"public": 0, "private": 0})[it.privacy] += 1
    return {"benchmark": "SkripsiBench", "created": time.strftime("%Y-%m-%d %H:%M"), "items": counts,
            "judge_weight": JUDGE_WEIGHT if use_judges else 0.0, "prices": prices.describe(), "ranking": ranked,
            "models": models, "pairwise_top": pairwise, "pareto_usd_per_100_vs_overall": S.pareto_front(pts)}


def _f(v, d: int = 3) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def leaderboard_md(summary: dict) -> str:
    m = summary["models"]
    lines = [f"# SkripsiBench leaderboard ({summary['created']})", "",
             "Scores in [0, 1]; 95 % bootstrap CI in brackets. Cost: API models in US$ per 100 items "
             f"(1 AIU = US${summary['prices']['usd_per_aiu']}); local models in kWh per 100 items (GPU board energy) "
             "and rental-equivalent US$ when a GPU price is set.", "",
             "| # | Model | Tier | Academic | Coding | Overall [95 % CI] | US$/100 | kWh/100 | Median s |",
             "|---:|---|---|---:|---:|---|---:|---:|---:|"]
    for i, k in enumerate(summary["ranking"], 1):
        r = m[k]
        c = r["cost"]
        ov = r["composite"]["overall"]
        lines.append(f"| {i} | {r['name']} | {r['tier']} | {_f(r['composite']['academic']['mean'])} | "
                     f"{_f(r['composite']['coding']['mean'])} | {_f(ov['mean'])} [{_f(ov['ci'][0], 2)}, {_f(ov['ci'][1], 2)}] | "
                     f"{_f(c.get('usd_per_100'), 3)} | {_f(c.get('kwh_per_100'), 4)} | {_f(c.get('median_s'), 1)} |")
    lines += ["", "## Per task", "", "| Model | " + " | ".join(TASKS) + " |", "|---|" + "---:|" * len(TASKS)]
    for k in summary["ranking"]:
        lines.append(f"| {m[k]['name']} | " + " | ".join(_f(m[k]['tasks'][t]['mean']) for t in TASKS) + " |")
    if summary.get("pairwise_top"):
        lines += ["", "## Top models compared item by item", "",
                  "| A | B | n | mean diff | Wilcoxon p (Holm) | sign-test p |", "|---|---|---:|---:|---:|---:|"]
        for p in summary["pairwise_top"]:
            lines.append(f"| {m[p['a']]['name']} | {m[p['b']]['name']} | {p['n']} | {_f(p['mean_diff'])} | "
                         f"{_f(p.get('wilcoxon_p_holm'), 4)} | {_f(p['sign_p'], 4)} |")
    subs = [(m[k]["name"], m[k]["cost"].get("served"), m[k]["cost"].get("substituted_turns"))
            for k in summary["ranking"] if m[k]["cost"].get("substituted_turns")]
    lines += ["", "## Served models", ""]
    lines += ([f"- {n}: {s} turns answered by another model: {sv}" for n, sv, s in subs] or
              ["- every call was answered by the requested model"])
    return "\n".join(lines) + "\n"


def per_item_csv(items: List[Item], run_dir: Path, use_judges: bool = True) -> str:
    per = item_scores(items, run_dir, use_judges)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "item", "task", "score", "auto", "judge"])
    for model in sorted(per):
        for iid in sorted(per[model]):
            r = per[model][iid]
            w.writerow([model, iid, r["task"], round(r["score"], 6), round(r["auto"], 6),
                        "" if r["judge"] is None else round(r["judge"], 6)])
    return buf.getvalue()


def write_report(items: List[Item], run_dir: Path, out_dir: Path, names: Optional[Dict[str, str]] = None,
                 log: Callable = print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(items, run_dir, names)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "leaderboard.md").write_text(leaderboard_md(summary), encoding="utf-8")
    (out_dir / "per_item.csv").write_text(per_item_csv(items, run_dir), encoding="utf-8")
    log(f"report: {len(summary['ranking'])} ranked models -> {out_dir}")
    return summary
