"""All AcademicBench results as Markdown tables, generated from the released data (no number typed by hand).

  python experiments/bench/make_results_md.py            # print the tables
  python experiments/bench/make_results_md.py --write    # replace the block between the markers in README.md
  python experiments/bench/make_results_md.py --check    # exit 1 when README.md is out of date (CI)

Reads ``summary.json``, ``analysis.json`` and ``refs_audit.json``; writes the block between
``<!-- results:begin -->`` and ``<!-- results:end -->``. A README without the markers is left alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "bench"
README = ROOT / "README.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
TASKS = ["structure", "methodology", "extraction", "grounded", "references", "language", "coding", "latex"]
TASK_HEAD = ["Outline", "Method.", "Extract.", "Grounded", "Refs", "Formal ID", "Code", "LaTeX"]
JUDGE_CRIT = {"kesetiaan_sumber": "faithfulness to the sources", "kesetiaan_isi": "faithfulness to the summary",
              "cakupan": "coverage", "kelengkapan": "completeness", "keruntutan": "coherence",
              "bahasa_baku": "formal Indonesian", "kepatuhan": "instruction compliance"}
SILVER_FIELD = {"pendekatan": "approach", "data_jenis": "data type", "teknik_pengumpulan": "collection technique",
                "analisis": "analysis technique", "alat": "software", "n_sampel": "sample size",
                "hipotesis": "hypothesis"}
STAGE = {"main": "answers (main run)", "pilot": "answers (pilot)", "judges": "LLM judges",
         "cost-probes": "premium cost probes"}


def f(v, d: int = 3) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def pct(v, d: int = 0) -> str:
    return "–" if v is None else f"{100 * v:.{d}f}"


def sgn(v, d: int = 3) -> str:
    return "–" if v is None else (f"−{abs(v):.{d}f}" if v < 0 else f"{v:.{d}f}")


def thousands(n) -> str:
    return f"{int(round(n)):,}"


def table(head: list, rows: list, align: str) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join({"l": "---", "r": "---:", "c": ":---:"}[a] for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def best(values: list, v, higher: bool = True) -> bool:
    vals = [x for x in values if x is not None]
    return v is not None and bool(vals) and v == (max(vals) if higher else min(vals))


def results(summary: dict, analysis: dict, audit: dict) -> str:
    ms, ranked = summary["models"], summary["ranking"]
    D, rb = analysis["diagnostics"], analysis["rank_bootstrap"]
    name = {k: ms[k]["name"].replace(" (Q8_0)", "") for k in ranked}
    leader = ranked[0]
    out = []

    # 1. leaderboard
    rows = []
    for i, k in enumerate(ranked, 1):
        m, c = ms[k], ms[k].get("cost") or {}
        ov = m["composite"]["overall"]
        cell = f"{ov['mean']:.3f} [{ov['ci'][0]:.2f}, {ov['ci'][1]:.2f}]"
        rows.append([i, f"**{name[k]}**" if i <= 3 else name[k], m["tier"], f"**{cell}**" if i <= 3 else cell,
                     f(m["composite"]["academic"]["mean"]), f(m["composite"]["coding"]["mean"]),
                     f(c.get("usd_per_100")), f(c.get("kwh_per_100"), 4) if c.get("kwh_per_100") else "–",
                     f(D[k]["latency"]["median_s"], 1)])
    out += ["### Leaderboard", "",
            "Overall score = unweighted mean of the eight task scores, with its 95 % task-stratified bootstrap "
            "interval. Academic: the six writing and research tasks; Code: data-analysis code and LaTeX. "
            "US$ per 100 items: API models in the vendor's AI units at US$0.01, local models as GPU time at "
            f"US${summary['prices']['gpu_usd_per_hour']:.2f}/h (on-demand L40S). kWh: GPU board energy of the local "
            "models (shared server, upper bound).", "",
            table(["#", "Model", "Tier", "Overall [95 % CI]", "Academic", "Code", "US$ / 100", "kWh / 100",
                   "Median s / call"], rows, "rllrrrrrr"), ""]

    # 2. per-task scores
    cols = {t: [ms[k]["tasks"][t]["mean"] for k in ranked] for t in TASKS}
    rows = []
    for k in ranked:
        rows.append([name[k]] + [f"**{f(ms[k]['tasks'][t]['mean'], 2)}**" if best(cols[t], ms[k]["tasks"][t]["mean"])
                                 else f(ms[k]["tasks"][t]["mean"], 2) for t in TASKS])
    n_items = {t: summary["items"][t]["public"] + summary["items"][t]["private"] for t in TASKS}
    out += ["### Scores per task", "",
            "Mean item score in [0, 1]; best per column in bold. Items per task: "
            + ", ".join(f"{h.lower()} {n_items[t]}" for h, t in zip(TASK_HEAD, TASKS)) + ".", "",
            table(["Model"] + TASK_HEAD, rows, "l" + "r" * len(TASKS)), ""]

    # 3. robustness of the ranking
    rows = []
    for k in ranked:
        r = rb[k]
        lo, hi = r["diff_ci"]
        sig = "–" if k == leader else ("yes" if lo > 0 else "no")
        rows.append([name[k], pct(r["p_top1"], 1), r["rank_median"], f"{r['rank_ci'][0]}–{r['rank_ci'][1]}",
                     "–" if k == leader else f"{sgn(r['diff_vs_leader'])} [{sgn(lo)}, {sgn(hi)}]", sig])
    out += ["### How stable is the ranking?", "",
            "1,000 task-stratified bootstrap samples of the items (seed 3): how often each model ranks first, its "
            f"median rank and 95 % rank interval, and its paired difference to the leader ({name[leader]}). "
            "*Significant*: the difference interval excludes zero (no multiplicity adjustment).", "",
            table(["Model", "P(rank 1) %", "Median rank", "Rank 95 % interval", "Δ to leader [95 % CI]",
                   "Significant"], rows, "lrrrrc"), ""]

    js, po, vs = analysis["judge_sensitivity"], analysis["public_only"], analysis["verification_sensitivity"]
    rows = [["prose scored by the automatic checks only (no LLM judges)", f(js["auto_only"]["rho"]),
             js["auto_only"]["max_rank_shift"], "yes" if js["auto_only"]["top4"][0] == leader else "no",
             "yes" if js["auto_only"]["top4_same_set"] else "no"],
            ["prose scored by the LLM judges only", f(js["judge_only"]["rho"]), js["judge_only"]["max_rank_shift"],
             "yes" if js["judge_only"]["top4"][0] == leader else "no",
             "yes" if js["judge_only"]["top4_same_set"] else "no"],
            ["without the private extraction task (public items only)", f(po["rho"]), po["max_rank_shift"],
             "yes" if po["leader_same"] else "no", "yes" if po["top4_same_set"] else "no"],
            [f"references found by the second-source audit credited ({vs['corrected_refs']} references)",
             f(vs["rho_overall"]), vs["max_rank_shift"], "yes" if vs["leader_same"] else "no",
             "yes" if vs["top4_same_set"] else "no"]]
    out += ["### Sensitivity of the ranking", "",
            "The overall ranking recomputed under alternative scoring choices, against the published ranking.", "",
            table(["Alternative", "Spearman ρ", "Largest rank shift", "Same leader", "Same top four"], rows, "lrrcc"),
            ""]

    # 4. diagnostics
    nowhere = {k: v.get("no_match_rate") for k, v in audit["models"].items()}
    rows = []
    for k in ranked:
        d = D[k]
        r, c, g = d["references"], d["coding"], d["grounded"]
        rows.append([name[k], pct(r["hallucination_rate"]), pct(nowhere.get(k)), pct(r["doi_correct_rate"]),
                     f(r["returned_mean"], 1), f(c["pass1"], 2), f(c["pass_repair"], 2), f(d["latex"]["fix_compiled"], 2),
                     f(g["unsupported_per_answer"], 2), pct(g["foreign_citation_answers"]),
                     f(d["extraction"]["evidence_verbatim"], 2), pct(d["structure"]["generic_ch4_rate"])])
    out += ["### What separates the models", "",
            "References: share of listed references not found in Crossref, found in none of Crossref, doi.org, "
            "Scopus and OpenAlex, and listed with the correct DOI; references listed per topic (5 requested). Code: "
            "share of tasks solved by the first program and after one repair turn. LaTeX: broken documents that "
            "compile after the fix. Grounded: numbers per review found in no source; reviews with author–year "
            "citations (forbidden). Extraction: quoted evidence found verbatim. Outline: chapter IV built around "
            "statistical tests where the programme's theses have none.", "",
            table(["Model", "Refs not in Crossref %", "Refs found nowhere %", "Correct DOI %", "Refs / topic",
                   "Code pass@1", "+ repair", "LaTeX fix", "Unsupported numbers", "Author–year %",
                   "Verbatim evidence", "Generic ch. IV %"], rows, "l" + "r" * 11), ""]

    # 5. reference audit
    rows = []
    for g, label in (("api", "API models"), ("local", "local models")):
        a = audit[g]
        sh = a["share_of_not_found"]
        rows.append([label, thousands(a["returned"]), f"{thousands(a['not_found'])} ({pct(a['not_found_rate'])} %)",
                     f"{a['exact']} ({pct(sh['exact'])} %)", f"{a['title']} ({pct(sh['title'])} %)",
                     f"{a['none'] + a['unchecked']} ({pct((sh['none'] or 0) + (sh['unchecked'] or 0))} %)",
                     ", ".join(f"{k} {v}" for k, v in sorted(a["via"].items())) or "–"])
    out += ["### Reference audit", "",
            "Every reference that Crossref does not know, looked up again with the same matching rule (title "
            "similarity ≥ 0.90, year ± 1, first author): a given DOI at doi.org (all registration agencies), the "
            "title in Scopus and OpenAlex. Shares are of the references not found in Crossref.", "",
            table(["Group", "Listed", "Not in Crossref", "Found elsewhere, matching", "Title only (wrong year / author)",
                   "Found nowhere", "Found via"], rows, "lrrrrrl"), ""]

    # 6. task discrimination
    disc = analysis["discrimination"]
    rows = [[h, f"{f(disc[t]['api_min'], 2)}–{f(disc[t]['api_max'], 2)}", f(disc[t]["api_sd"]),
             f"{disc[t]['api_at_ceiling']} / {disc[t]['api_n']}", f(disc[t]["local_mean"], 2),
             f"{f(disc[t]['local_min'], 2)}–{f(disc[t]['local_max'], 2)}", f(disc[t]["gap"], 2)]
            for h, t in zip(TASK_HEAD, TASKS)]
    out += ["### Which tasks discriminate?", "",
            "Spread of the task scores among API models (ceiling: score ≥ 0.95) and the gap between the API and the "
            "local group means.", "",
            table(["Task", "API range", "API SD", "API at ceiling", "Local mean", "Local range", "API − local gap"],
                  rows, "lrrrrrr"), ""]

    # 7. judges
    J, avj = analysis["judges"], analysis["auto_vs_judge"]
    crit = sorted(J["alpha_ordinal_by_criterion"].items())
    rows = [[t.split(":")[0], JUDGE_CRIT.get(t.split(":")[1], t.split(":")[1]), f(v, 2)] for t, v in crit]
    judge_rows = [[k.split(":", 1)[1].rsplit("@", 1)[0], f(J["mean_by_judge"][k]), sgn(J["leniency_offset"][k])]
                  for k in sorted(J["leniency_offset"])]
    out += ["### LLM judges", "",
            f"Three judges of three vendors rated {thousands(J['answers'])} prose answers ({thousands(J['ratings'])} "
            f"ratings); a judge never rates models of its own vendor. Agreement on the {J['answers_all_judges']} "
            f"answers that all three rated: Krippendorff's α = {f(J['alpha_interval_all_judges'], 2)} (interval, mean "
            "rating). Judges against the deterministic checks (Spearman ρ): "
            + "; ".join(f"{t} {f(avj[t]['rho_answers'], 2)} across {avj[t]['answers']} answers, "
                        f"{f(avj[t]['rho_model_means'], 2)} across {avj[t]['models']} model means" for t in sorted(avj))
            + ".", "",
            table(["Judge", "Mean rating (0–1)", "Leniency offset removed"], judge_rows, "lrr"), "",
            table(["Task", "Criterion", "Krippendorff's α (ordinal)"], rows, "llr"), ""]

    # 8. extraction silver labels
    sa = analysis["silver_agreement"]
    rows = [[SILVER_FIELD.get(k, k), v.get("unanimous", 0), v.get("majority", 0), v.get("none", 0)]
            for k, v in sa.items()]
    out += ["### Extraction silver labels", "",
            "Agreement of the three-model panel per field over the 30 private excerpts; fields without a majority "
            "are not scored.", "",
            table(["Field", "Unanimous", "Majority", "No majority (not scored)"], rows, "lrrr"), ""]

    # 9. cost and volume
    cost = analysis["cost"]
    rows = [[STAGE.get(st, st), thousands(cost["api_calls_by_stage"].get(st, 0)), thousands(aiu),
             f"{aiu * summary['prices']['usd_per_aiu']:.2f}"] for st, aiu in sorted(cost["aiu_by_stage"].items(),
                                                                                   key=lambda x: -x[1])]
    rows.append(["**total**", f"**{thousands(cost['api_calls_total'])}**", f"**{thousands(cost['aiu_total'])}**",
                 f"**{cost['usd_total']:.2f}**"])
    loc = [k for k in ranked if ms[k]["engine"] == "ollama"]
    gpu_h = sum((ms[k].get("cost") or {}).get("gpu_hours") or 0 for k in loc)
    kwh = sum((ms[k].get("cost") or {}).get("kwh_gross") or 0 for k in loc)
    out += ["### Cost of the study", "",
            f"API tokens: {cost['api_tokens']['in'] / 1e6:.1f} M in, {cost['api_tokens']['out'] / 1e6:.1f} M out. "
            f"Local models: {thousands(cost['local_calls'])} calls, {gpu_h:.1f} GPU-hours, {kwh:.2f} kWh gross GPU "
            "board energy.", "",
            table(["Stage", "API calls", "AI units", "US$"], rows, "lrrr")]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="update the block in README.md")
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    args = ap.parse_args()
    load = lambda n: json.loads((BENCH / n).read_text(encoding="utf-8"))  # noqa: E731
    block = results(load("summary.json"), load("analysis.json"), load("refs_audit.json"))
    if not (args.write or args.check):
        print(block, end="")
        return 0
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("README.md has no results block; nothing to do")
        return 0
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{block}{END}{tail}"
    if args.check:
        if new != text:
            print("README.md results are out of date: run python experiments/bench/make_results_md.py --write")
            return 1
        print("README.md results are up to date")
        return 0
    README.write_text(new, encoding="utf-8")
    print(f"results -> {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
