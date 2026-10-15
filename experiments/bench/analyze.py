"""Analyses for the SkripsiBench paper beyond the leaderboard, from a run directory -> ``analysis.json`` (aggregates
only: no item text, no answers).

  BENCH_RUN_DIR=<run dir> BENCH_PRIVATE_DIR=<private dir> python experiments/bench/analyze.py

Computes: study cost and volume; per-model diagnostics (reference fabrication, DOI accuracy, abstention; coding
pass@1 and after one repair; generic results chapters; verbatim evidence; unsupported numbers; non-standard words);
task discrimination (ceiling effects); local vs API gaps; LLM-judge reliability (leniency, Krippendorff's alpha
between judges, agreement with the deterministic scores); agreement of the extraction silver panel; ranking
robustness (stratified bootstrap of the overall rank and of the difference to the leader); sensitivity of the ranking
to the judges (prose scored by the automatic checks only or by the judges only) and to the reference verification
source (references that ``audit_refs.py`` finds outside Crossref credited as the scorer would).
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_agent.bench import ACADEMIC_TASKS, CODING_TASKS, TASKS  # noqa: E402
from paper_agent.bench import stats as S  # noqa: E402
from paper_agent.bench.judge import CRITERIA, leniency_offsets  # noqa: E402
from paper_agent.bench.report import item_scores  # noqa: E402
from paper_agent.bench.runner import latest_records, record_aiu  # noqa: E402
from paper_agent.bench.schema import item_dirs, load_items  # noqa: E402
from paper_agent.bench.score import load_scores  # noqa: E402
from paper_agent.bench.tasks import extraction as X  # noqa: E402
from paper_agent.bench.tasks import parse_json_loose  # noqa: E402

OUT = ROOT / "experiments" / "bench" / "analysis.json"
PANEL = ["copilot:gpt-6.1-sol@low", "copilot:claude-sonnet-5@low", "copilot:gemini-4-argon"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def r4(x):
    return None if x is None else round(float(x), 4)


def study_cost(run_dir: Path, private: Path) -> dict:
    """All API spending of the study (every record, retries included) by stage, plus judges and cost probes."""
    by_stage, calls, tokens = Counter(), Counter(), Counter()
    for p in (run_dir / "raw").glob("copilot__*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = r.get("stage") or "?"
            by_stage[st] += record_aiu(r)
            calls[st] += 1 + len(r.get("followups") or [])
            for t in [r] + list(r.get("followups") or []):
                u = t.get("usage") or {}
                tokens["in"] += u.get("input_tokens") or 0
                tokens["out"] += u.get("output_tokens") or 0
    jc = 0
    for p in (run_dir / "judges").glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_stage["judges"] += j.get("aiu") or 0.0
            jc += 1
    calls["judges"] = jc
    probe = private / "runs" / "taksir" / "taksir-premium.json"
    if probe.exists():
        pr = json.loads(probe.read_text(encoding="utf-8"))
        by_stage["cost-probes"] += pr.get("probe_spent_aiu") or 0.0
        calls["cost-probes"] = sum(len(v) for v in pr.get("probes", {}).values())
    local_calls = 0
    for p in (run_dir / "raw").glob("ollama__*.jsonl"):
        local_calls += sum(1 for _ in p.read_text(encoding="utf-8").splitlines())
    total = sum(by_stage.values())
    return {"aiu_by_stage": {k: round(v, 1) for k, v in sorted(by_stage.items())}, "aiu_total": round(total, 1),
            "usd_total": round(total * 0.01, 2), "api_calls_by_stage": dict(sorted(calls.items())),
            "api_calls_total": sum(calls.values()), "api_tokens": dict(tokens), "local_calls": local_calls}


def diagnostics(scores: dict, recs: dict, items: list) -> dict:
    kind = {it.id: it.input.get("kind") for it in items if it.task == "latex"}
    out = {}
    for model, rows in scores.items():
        by = defaultdict(list)
        for (_iid, _rep), r in rows.items():
            by[r["task"]].append(r)
        d: dict = {}
        ref = by["references"]
        returned = sum(r.get("returned") or 0 for r in ref)
        found = sum(r.get("found") or 0 for r in ref)
        d["references"] = {"returned_mean": r4(mean([r.get("returned") or 0 for r in ref])),
                           "hallucination_rate": r4((returned - found) / returned) if returned else None,
                           "doi_correct_rate": r4(sum(r.get("doi_correct") or 0 for r in ref) / returned) if returned else None,
                           "doi_unresolvable": sum(r.get("doi_unresolvable") or 0 for r in ref),
                           "abstained_items": sum(1 for r in ref if (r.get("returned") or 0) < 5), "returned": returned}
        cod = by["coding"]
        d["coding"] = {"pass1": r4(mean([1.0 if r.get("pass1") else 0.0 for r in cod])),
                       "pass_repair": r4(mean([1.0 if r.get("pass_repair") else 0.0 for r in cod])),
                       "ran_first": r4(mean([1.0 if r.get("ran_first") else 0.0 for r in cod])),
                       "repaired_items": sum(1 for r in cod if r.get("repaired"))}
        st = by["structure"]
        d["structure"] = {"generic_ch4_rate": r4(mean([1.0 if r.get("generic_ch4") else 0.0 for r in st])),
                          "ch3_f1": r4(mean([(r.get("ch3") or {}).get("f1") for r in st])),
                          "ch4_f1": r4(mean([(r.get("ch4") or {}).get("f1") for r in st])),
                          "ch3_recall": r4(mean([(r.get("ch3") or {}).get("recall") for r in st])),
                          "ch3_precision": r4(mean([(r.get("ch3") or {}).get("precision") for r in st]))}
        ex = by["extraction"]
        fields = defaultdict(list)
        for r in ex:
            for f, v in (r.get("fields") or {}).items():
                fields[f].append(v)
        d["extraction"] = {"evidence_verbatim": r4(mean([r.get("evidence_verbatim") for r in ex])),
                           "fields": {f: r4(mean(v)) for f, v in sorted(fields.items())}}
        gr = by["grounded"]
        d["grounded"] = {"unsupported_per_answer": r4(mean([r.get("unsupported_numbers") or 0 for r in gr])),
                         "answers_with_unsupported": r4(mean([1.0 if (r.get("unsupported_numbers") or 0) else 0.0
                                                              for r in gr])),
                         "foreign_citation_answers": r4(mean([1.0 if (r.get("foreign_citations") or 0) else 0.0
                                                              for r in gr])),
                         "coverage": r4(mean([(r.get("parts") or {}).get("coverage") for r in gr])),
                         "cited_sentences": r4(mean([(r.get("parts") or {}).get("cited_sentences") for r in gr])),
                         "words": r4(mean([r.get("words") for r in gr]))}
        la = by["language"]
        d["language"] = {"nonstandard_per_answer": r4(mean([r.get("nonstandard_count") or 0 for r in la])),
                         "english_ratio": r4(mean([r.get("english_ratio") for r in la])),
                         "numbers_kept": r4(mean([(r.get("parts") or {}).get("numbers_kept") for r in la])),
                         "answers_with_new_numbers": r4(mean([1.0 if r.get("numbers_fabricated") else 0.0 for r in la]))}
        lt = by["latex"]
        fix = [r for r in lt if kind.get(r["item"]) == "fix"]
        tab = [r for r in lt if kind.get(r["item"]) == "table"]
        bib = [r for r in lt if kind.get(r["item"]) == "bibtex"]
        d["latex"] = {"fix_compiled": r4(mean([1.0 if r.get("compiled") else 0.0 for r in fix])),
                      "fix_score": r4(mean([r.get("score") for r in fix])),
                      "table_cells": r4(mean([r.get("cell_accuracy") or 0.0 for r in tab])),
                      "table_score": r4(mean([r.get("score") for r in tab])),
                      "bibtex": r4(mean([r.get("score") for r in bib]))}
        me = by["methodology"]
        d["methodology"] = {"accuracy": r4(mean([r.get("score") for r in me])),
                            "unparsed": sum(1 for r in me if not r.get("parsed"))}
        rr = list(recs.get(model, {}).values())
        secs = [r.get("seconds") for r in rr if r.get("seconds")]
        outs = [(r.get("usage") or {}).get("output_tokens") or 0 for r in rr]
        reas = [(r.get("usage") or {}).get("reasoning_tokens") or 0 for r in rr]
        d["latency"] = {"median_s": r4(statistics.median(secs)) if secs else None,
                        "p90_s": r4(sorted(secs)[int(0.9 * (len(secs) - 1))]) if secs else None,
                        "out_tokens_mean": r4(mean(outs)), "reasoning_tokens_mean": r4(mean(reas)),
                        "thinking_chars_mean": r4(mean([r.get("thinking_chars") or 0 for r in rr]))}
        out[model] = d
    return out


def discrimination(summary: dict) -> dict:
    res = {}
    for t in TASKS:
        api = [m["tasks"][t]["mean"] for m in summary["models"].values() if m["engine"] == "copilot"]
        loc = [m["tasks"][t]["mean"] for m in summary["models"].values() if m["engine"] == "ollama"]
        res[t] = {"api_min": r4(min(api)), "api_max": r4(max(api)), "api_sd": r4(statistics.pstdev(api)),
                  "api_at_ceiling": sum(1 for v in api if v >= 0.95), "api_n": len(api),
                  "local_min": r4(min(loc)), "local_max": r4(max(loc)), "local_mean": r4(mean(loc)),
                  "api_mean": r4(mean(api)), "gap": r4(mean(api) - mean(loc))}
    return res


def judge_reliability(run_dir: Path, recs: dict) -> dict:
    rows = []
    for p in sorted((run_dir / "judges").glob("*.jsonl")):
        latest = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ratings"):
                latest[(r["model"], r["item"], r.get("rep", 0), r["answer_hash"])] = r
        rows += list(latest.values())
    judges = sorted({r["judge"] for r in rows})
    offsets = leniency_offsets(rows)
    by_answer = defaultdict(dict)
    for r in rows:
        by_answer[(r["model"], r["item"], r.get("rep", 0), r["answer_hash"])][r["judge"]] = r
    common = [v for v in by_answer.values() if set(v) == set(judges)]
    alpha_mean = S.krippendorff_alpha([[v[j]["score"] for j in judges] for v in common], "interval")
    per_crit = {}
    for task, crits in CRITERIA.items():
        sub = [v for v in common if next(iter(v.values()))["task"] == task]
        for c in crits:
            units = [[v[j]["ratings"].get(c) for j in judges] for v in sub]
            a = S.krippendorff_alpha(units, "ordinal")
            per_crit[f"{task}:{c}"] = r4(a)
    pairs = {}
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            xs = [(v[a]["score"], v[b]["score"]) for v in by_answer.values() if a in v and b in v]
            sp = S.spearman([x for x, _ in xs], [y for _, y in xs])
            pairs[f"{a} vs {b}"] = {"n": sp["n"], "rho": r4(sp["rho"])}
    return {"judges": judges, "ratings": len(rows), "answers": len(by_answer), "answers_all_judges": len(common),
            "mean_by_judge": {j: r4(mean([r["score"] for r in rows if r["judge"] == j])) for j in judges},
            "leniency_offset": {j: r4(v) for j, v in offsets.items()},
            "alpha_interval_all_judges": r4(alpha_mean), "alpha_ordinal_by_criterion": per_crit,
            "spearman_pairs": pairs}


def auto_vs_judge(per_item: dict) -> dict:
    res = {}
    for task in ("grounded", "language"):
        a, j = [], []
        ma, mj = defaultdict(list), defaultdict(list)
        for model, rows in per_item.items():
            for r in rows.values():
                if r["task"] == task and r["judge"] is not None:
                    a.append(r["auto"])
                    j.append(r["judge"])
                    ma[model].append(r["auto"])
                    mj[model].append(r["judge"])
        sp = S.spearman(a, j)
        models = sorted(ma)
        spm = S.spearman([mean(ma[m]) for m in models], [mean(mj[m]) for m in models])
        res[task] = {"answers": sp["n"], "rho_answers": r4(sp["rho"]), "models": spm["n"], "rho_model_means": r4(spm["rho"])}
    return res


def silver_agreement(recs: dict, items: list) -> dict:
    ext = [it.id for it in items if it.task == "extraction"]
    counts = {f: Counter() for f in X.FIELDS}
    for iid in ext:
        got = []
        for lab in PANEL:
            r = recs.get(lab, {}).get((iid, 0))
            if r and not r.get("error"):
                got.append(X.normalize(parse_json_loose(r.get("text", ""))))
        if len(got) < 3:
            continue
        for f in X.FIELDS:
            vals = Counter(json.dumps(g.get(f), sort_keys=True) for g in got)
            top = vals.most_common(1)[0][1]
            counts[f]["unanimous" if top == 3 else "majority" if top == 2 else "none"] += 1
    return {f: dict(c) for f, c in counts.items()}


def overall_score(rows: dict, pick=lambda r: r["score"]) -> float:
    """Mean of the task means (the leaderboard's overall score); ``pick`` chooses the per-item score."""
    by_task = defaultdict(list)
    for r in rows.values():
        by_task[r["task"]].append(pick(r))
    return mean([mean(by_task[t]) for t in TASKS if by_task.get(t)])


def judge_sensitivity(per_item: dict) -> dict:
    """Overall ranking when the prose tasks are scored by the automatic checks only or by the judges only."""
    main = {m: overall_score(rows) for m, rows in per_item.items()}
    order = sorted(main, key=lambda m: -main[m])
    res = {}
    for name, pick in (("auto_only", lambda r: r["auto"]),
                       ("judge_only", lambda r: r["score"] if r["judge"] is None else r["judge"])):
        alt = {m: overall_score(rows, pick) for m, rows in per_item.items()}
        alt_order = sorted(alt, key=lambda m: -alt[m])
        sp = S.spearman([main[m] for m in order], [alt[m] for m in order])
        res[name] = {"rho": r4(sp["rho"]), "top4": alt_order[:4], "top4_same_set": set(alt_order[:4]) == set(order[:4]),
                     "max_rank_shift": max(abs(alt_order.index(m) - order.index(m)) for m in order),
                     "overall": {m: r4(alt[m]) for m in alt_order}}
    return res


def verification_sensitivity(per_item: dict, audit: Optional[dict], items: list) -> Optional[dict]:
    """The ranking when the references that another database confirms (``refs_audit.json``) get the scorer's credit."""
    if not audit or not audit.get("corrections"):
        return None
    n_req = {it.id: int(it.input.get("n", 5)) for it in items if it.task == "references"}
    delta = defaultdict(float)
    for c in audit["corrections"]:
        rows = per_item.get(c["model"], {})
        if c["item"] in rows:
            delta[(c["model"], c["item"])] += c["credit"] / n_req[c["item"]] / rows[c["item"]].get("n_reps", 1)
    alt = {m: {iid: dict(r, score=min(1.0, r["score"] + delta.get((m, iid), 0.0))) for iid, r in rows.items()}
           for m, rows in per_item.items()}

    def ref_mean(rows):
        return mean([r["score"] for r in rows.values() if r["task"] == "references"])
    main = {m: overall_score(rows) for m, rows in per_item.items()}
    new = {m: overall_score(rows) for m, rows in alt.items()}
    ref_main = {m: ref_mean(rows) for m, rows in per_item.items()}
    ref_new = {m: ref_mean(rows) for m, rows in alt.items()}
    order, new_order = sorted(main, key=lambda m: -main[m]), sorted(new, key=lambda m: -new[m])
    ref_order = sorted(ref_main, key=lambda m: -ref_main[m])
    return {"complete": bool(audit.get("complete")), "corrected_refs": len(audit["corrections"]),
            "rho_overall": r4(S.spearman([main[m] for m in order], [new[m] for m in order])["rho"]),
            "rho_references": r4(S.spearman([ref_main[m] for m in ref_order], [ref_new[m] for m in ref_order])["rho"]),
            "max_rank_shift": max(abs(new_order.index(m) - order.index(m)) for m in order),
            "leader_same": new_order[0] == order[0], "top4_same_set": set(new_order[:4]) == set(order[:4]),
            "max_references_gain": r4(max(ref_new[m] - ref_main[m] for m in ref_main)),
            "max_overall_gain": r4(max(new[m] - main[m] for m in main)),
            "references": {m: [r4(ref_main[m]), r4(ref_new[m])] for m in ref_order},
            "overall": {m: [r4(main[m]), r4(new[m])] for m in new_order}}


def rank_bootstrap(per_item: dict, n_boot: int = 1000, seed: int = 3) -> dict:
    """Task-stratified bootstrap over items: P(rank 1), rank CI, and the paired difference to the observed leader."""
    models = sorted(per_item)
    tasks = list(TASKS)
    ids = {t: sorted({iid for m in models for iid, r in per_item[m].items() if r["task"] == t}) for t in tasks}
    rng = random.Random(seed)
    obs = {m: overall_score(per_item[m]) for m in models}
    leader = max(models, key=lambda m: obs[m])
    top1, ranks, diffs = Counter(), defaultdict(list), defaultdict(list)
    for _ in range(n_boot):
        sample = {t: [ids[t][rng.randrange(len(ids[t]))] for _ in ids[t]] for t in tasks}
        overall = {}
        for m in models:
            means = []
            for t in tasks:
                vals = [per_item[m][i]["score"] for i in sample[t] if i in per_item[m]]
                means.append(sum(vals) / len(vals))
            overall[m] = sum(means) / len(means)
        order = sorted(models, key=lambda m: -overall[m])
        top1[order[0]] += 1
        for k, m in enumerate(order, 1):
            ranks[m].append(k)
            diffs[m].append(overall[leader] - overall[m])
    out = {}
    for m in models:
        rs, ds = sorted(ranks[m]), sorted(diffs[m])
        out[m] = {"p_top1": r4(top1[m] / n_boot), "rank_median": statistics.median(rs),
                  "rank_ci": [rs[int(0.025 * (len(rs) - 1))], rs[int(0.975 * (len(rs) - 1))]],
                  "diff_vs_leader": r4(obs[leader] - obs[m]),
                  "diff_ci": [r4(ds[int(0.025 * (len(ds) - 1))]), r4(ds[int(0.975 * (len(ds) - 1))])]}
    return out


def main() -> None:
    run_dir = Path(os.environ["BENCH_RUN_DIR"]).expanduser()
    private = Path(os.environ["BENCH_PRIVATE_DIR"]).expanduser()
    items = load_items(item_dirs(ROOT / "bench" / "items", private))
    summary = json.loads((ROOT / "experiments" / "bench" / "summary.json").read_text(encoding="utf-8"))
    recs = latest_records(run_dir)
    scores = load_scores(run_dir)
    per_item = item_scores(items, run_dir)
    audit_path = OUT.parent / "refs_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    res = {"cost": study_cost(run_dir, private), "diagnostics": diagnostics(scores, recs, items),
           "discrimination": discrimination(summary), "judges": judge_reliability(run_dir, recs),
           "auto_vs_judge": auto_vs_judge(per_item), "silver_agreement": silver_agreement(recs, items),
           "rank_bootstrap": rank_bootstrap(per_item), "judge_sensitivity": judge_sensitivity(per_item),
           "verification_sensitivity": verification_sensitivity(per_item, audit, items),
           "groups": {g: {"academic": ACADEMIC_TASKS, "coding": CODING_TASKS}[g] for g in ("academic", "coding")}}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"analysis -> {OUT}")


if __name__ == "__main__":
    main()
