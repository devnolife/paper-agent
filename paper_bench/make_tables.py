"""LaTeX tables and number macros for the SkripsiBench paper, from ``experiments/bench/{summary,analysis,refs_audit}.json``.

  python paper_bench/make_tables.py      # -> paper_bench/generated/{leaderboard,diagnostics,numbers}.tex

Every number in the results text is a macro defined in ``numbers.tex``, never typed by hand, so a new run (for
instance with the premium tier) updates the paper by regenerating this file and recompiling; statements that depend
on the ranking still have to be re-read.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "bench" / "summary.json"
ANALYSIS = ROOT / "experiments" / "bench" / "analysis.json"
AUDIT = ROOT / "experiments" / "bench" / "refs_audit.json"
OUT = ROOT / "paper_bench" / "generated"
TASKS = ["structure", "methodology", "extraction", "grounded", "references", "language", "coding", "latex"]
TASK_ABBR = {"structure": "Str", "methodology": "Met", "extraction": "Ext", "grounded": "Grd", "references": "Ref",
             "language": "Lng", "coding": "Cod", "latex": "TeX"}
TIER_ABBR = {"economy": "E", "standard": "S", "premium": "P", "local": "L"}


def tex(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#"))


def f(v, d: int = 2) -> str:
    return "--" if v is None else f"{v:.{d}f}"


def signed(v, d: int = 3) -> str:
    """A number for running text: a negative value gets a math minus, not a hyphen."""
    return "--" if v is None else (f"$-${abs(v):.{d}f}" if v < 0 else f"{v:.{d}f}")


def pct(v, d: int = 0) -> str:
    return "--" if v is None else f"{100 * v:.{d}f}"


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def best_in(values: list, v, higher: bool = True) -> bool:
    vals = [x for x in values if x is not None]
    return v is not None and bool(vals) and v == (max(vals) if higher else min(vals))


CAPTIONS = {
    "en": {"leaderboard": "Leaderboard (one run per model and item; API models at low reasoning effort where the model "
                          "accepts one). Per-task scores, composites (academic: the first six tasks; code: code and LaTeX) "
                          "and the overall score with its 95\\,\\% bootstrap CI; cost per 100 items: API models in AI "
                          "units at US\\$0.01, local models as L40S rental equivalent; kWh: GPU board energy of local "
                          "models. T: tier (E economy, S standard, L local). Best value per column in bold.",
           "diagnostics": "Diagnostics behind the scores. Ref: share of listed references not found in Crossref (not "
                          "CR), share found in none of Crossref, doi.org, Scopus and OpenAlex (none), and share with the "
                          "correct DOI; refs/item: references listed per item (5 requested). Code: share of tasks solved by the first program and after one repair turn. "
                          "TeX fix: broken documents that compile after the model's fix. Grounded: numbers per review "
                          "found in no source; share of reviews with author--year citations (forbidden). Ext.\\ evid.: "
                          "share of quoted evidence found verbatim in the excerpt. Generic IV: share of outlines with a "
                          "test-centred results chapter where the programme's theses have none.",
           "head": ["Model", "Acad.", "Code", "Overall [95\\,\\% CI]", "References", "not CR\\,\\%", "refs/item",
                    "+repair", "fix", "Grounded", "unsup.", "auth.--yr\\,\\%", "evid.", "Generic", "s/call", "Median",
                    "none\\,\\%"]},
    "id": {"leaderboard": "Papan peringkat (satu run per model dan butir; model API dengan reasoning effort low bila "
                          "model menerimanya). Skor per tugas, skor gabungan (akademik: enam tugas pertama; kode: kode dan "
                          "LaTeX) dan skor keseluruhan dengan IK 95\\,\\% bootstrap; biaya per 100 butir: model API dalam "
                          "AI units dengan US\\$0,01, model lokal sebagai setara sewa L40S; kWh: energi papan GPU model "
                          "lokal. T: tier (E hemat, S menengah, L lokal). Nilai terbaik per kolom dicetak tebal.",
           "diagnostics": "Diagnostik di balik skor. Ref: porsi rujukan yang tidak ditemukan di Crossref (tdk di CR), "
                          "porsi yang tidak ditemukan di Crossref, doi.org, Scopus maupun OpenAlex (nihil), dan porsi "
                          "dengan DOI benar; ruj./butir: rujukan yang dicantumkan per butir "
                          "(diminta 5). Kode: porsi tugas yang diselesaikan program pertama dan setelah satu putaran "
                          "perbaikan. TeX fix: dokumen rusak yang dapat dikompilasi setelah diperbaiki model. Grounded: "
                          "angka per tinjauan yang tidak ada di sumber; porsi tinjauan dengan sitasi penulis--tahun "
                          "(dilarang). Ekst.\\ bukti: porsi kutipan bukti yang ditemukan persis di cuplikan. BAB IV "
                          "generik: porsi kerangka dengan bab hasil berpusat pada uji statistik padahal skripsi prodi "
                          "tersebut tidak memilikinya.",
           "head": ["Model", "Akad.", "Kode", "Keseluruhan [IK 95\\,\\%]", "Rujukan", "tdk di CR\\,\\%", "ruj./butir",
                    "+perbaikan", "fix", "Grounded", "tdk did.", "pen.--th\\,\\%", "bukti", "BAB IV gen.", "dtk/pgl",
                    "Median", "nihil\\,\\%"]},
}


def leaderboard_table(summary: dict, lang: str = "en") -> str:
    ms = summary["models"]
    ranked = summary["ranking"]
    overall = [ms[k]["composite"]["overall"]["mean"] for k in ranked]
    h = CAPTIONS[lang]["head"]
    lines = ["\\begin{table*}[t]", "  \\caption{" + CAPTIONS[lang]["leaderboard"] + "}",
             "  \\label{tab:leaderboard}", "  \\centering", "  \\scriptsize", "  \\setlength{\\tabcolsep}{2.6pt}",
             "  \\begin{tabular}{@{}r l c " + "r" * len(TASKS) + " r r l r r@{}}", "    \\toprule",
             "    \\# & " + h[0] + " & T & " + " & ".join(TASK_ABBR[t] for t in TASKS) +
             f" & {h[1]} & {h[2]} & {h[3]} & US\\$/100 & kWh/100 \\\\", "    \\midrule"]
    cols = {t: [ms[k]["tasks"][t]["mean"] for k in ranked] for t in TASKS}
    acad = [ms[k]["composite"]["academic"]["mean"] for k in ranked]
    code = [ms[k]["composite"]["coding"]["mean"] for k in ranked]
    bold = "\\textbf{%s}"
    for i, k in enumerate(ranked, 1):
        m = ms[k]
        cells = [bold % f(m["tasks"][t]["mean"]) if best_in(cols[t], m["tasks"][t]["mean"]) else f(m["tasks"][t]["mean"])
                 for t in TASKS]
        a, c = m["composite"]["academic"]["mean"], m["composite"]["coding"]["mean"]
        ov = m["composite"]["overall"]
        cost = m.get("cost") or {}
        row = [str(i), tex(m["name"]), TIER_ABBR.get(m["tier"], "?")] + cells + [
            bold % f(a) if best_in(acad, a) else f(a), bold % f(c) if best_in(code, c) else f(c),
            (bold % f(ov["mean"]) if best_in(overall, ov["mean"]) else f(ov["mean"])) + f" [{f(ov['ci'][0])}, {f(ov['ci'][1])}]",
            f(cost.get("usd_per_100"), 3), f(cost.get("kwh_per_100"), 3)]
        lines.append("    " + " & ".join(row) + " \\\\")
        if i < len(ranked) and m["engine"] != ms[ranked[i]]["engine"]:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def diagnostics_table(summary: dict, analysis: dict, audit: Optional[dict] = None, lang: str = "en") -> str:
    ms, D = summary["models"], analysis["diagnostics"]
    nowhere = {k: v.get("no_match_rate") for k, v in ((audit or {}).get("models") or {}).items()}
    h = CAPTIONS[lang]["head"]
    lines = ["\\begin{table*}[t]", "  \\caption{" + CAPTIONS[lang]["diagnostics"] + "}",
             "  \\label{tab:diagnostics}", "  \\centering", "  \\scriptsize", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\begin{tabular}{@{}l c r r r r r r r r r r r r@{}}", "    \\toprule",
             f"    & & \\multicolumn{{4}}{{c}}{{{h[4]}}} & \\multicolumn{{2}}{{c}}{{{h[2]}}} & TeX & "
             f"\\multicolumn{{2}}{{c}}{{{h[9]}}} & Ext. & {h[13]} & {h[15]} \\\\",
             "    \\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\\cmidrule(lr){10-11}",
             f"    {h[0]} & T & {h[5]} & {h[16]} & DOI\\,\\% & {h[6]} & pass@1 & {h[7]} & {h[8]} & {h[10]} & {h[11]} & "
             f"{h[12]} & IV\\,\\% & {h[14]} \\\\", "    \\midrule"]
    ranked = summary["ranking"]
    for i, k in enumerate(ranked):
        m, d = ms[k], D[k]
        r, c, g = d["references"], d["coding"], d["grounded"]
        row = [tex(m["name"]), TIER_ABBR.get(m["tier"], "?"), pct(r["hallucination_rate"]), pct(nowhere.get(k)),
               pct(r["doi_correct_rate"]),
               f(r["returned_mean"], 1), f(c["pass1"]), f(c["pass_repair"]), f(d["latex"]["fix_compiled"]),
               f(g["unsupported_per_answer"]), pct(g["foreign_citation_answers"]), f(d["extraction"]["evidence_verbatim"]),
               pct(d["structure"]["generic_ch4_rate"]), f(d["latency"]["median_s"], 1)]
        lines.append("    " + " & ".join(row) + " \\\\")
        if i + 1 < len(ranked) and m["engine"] != ms[ranked[i + 1]]["engine"]:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def audit_numbers(audit: Optional[dict], ver: Optional[dict], api: list, loc: list) -> dict:
    """Macros of the second-source reference audit and of the ranking's sensitivity to it ('--' while absent)."""
    names = ["AuditApiNF", "AuditApiExact", "AuditApiTitle", "AuditApiNone", "AuditLocNF", "AuditLocExact",
             "AuditLocTitle", "AuditLocNone", "AuditUnchecked", "NoneApiMin", "NoneApiMax", "NoneLocMin", "NoneLocMax",
             "NoneApiNonzero", "NoneApiZero",
             "VerRho", "VerRefRho", "VerMaxShift", "VerRefMaxGain", "VerOverallMaxGain", "VerLeaderSame"]
    n = {k: "--" for k in names}
    if audit and all("share_of_not_found" in (audit.get(g) or {}) for g in ("api", "local")):
        for g, pre in (("api", "AuditApi"), ("local", "AuditLoc")):
            sh = audit[g]["share_of_not_found"]
            n.update({f"{pre}NF": thousands(audit[g]["not_found"]), f"{pre}Exact": pct(sh["exact"]),
                      f"{pre}Title": pct(sh["title"]), f"{pre}None": pct((sh["none"] or 0) + (sh["unchecked"] or 0))})
        n["AuditUnchecked"] = audit.get("unchecked", 0)
        for ks, pre in ((api, "NoneApi"), (loc, "NoneLoc")):
            vals = [audit["models"][k]["no_match_rate"] for k in ks if k in audit["models"]]
            if vals:
                n.update({f"{pre}Min": pct(min(vals)), f"{pre}Max": pct(max(vals))})
        api_none = [audit["models"][k]["no_match_rate"] for k in api if k in audit["models"]]
        n.update({"NoneApiNonzero": sum(1 for v in api_none if v), "NoneApiZero": sum(1 for v in api_none if not v)})
    if ver:
        n.update({"VerRho": f(ver["rho_overall"], 3), "VerRefRho": f(ver["rho_references"], 3),
                  "VerMaxShift": ver["max_rank_shift"], "VerRefMaxGain": f(ver["max_references_gain"], 2),
                  "VerOverallMaxGain": f(ver["max_overall_gain"], 3), "VerLeaderSame": "yes" if ver["leader_same"] else "no"})
    return n


def numbers(summary: dict, analysis: dict, audit: Optional[dict] = None) -> dict:
    ms, A, D = summary["models"], analysis, analysis["diagnostics"]
    ranked = summary["ranking"]
    api = [k for k in ranked if ms[k]["engine"] == "copilot"]
    loc = [k for k in ranked if ms[k]["engine"] == "ollama"]
    ov = {k: ms[k]["composite"]["overall"]["mean"] for k in ranked}
    usd = {k: (ms[k].get("cost") or {}).get("usd_per_100") for k in ranked}
    name = {k: ms[k]["name"] for k in ranked}

    def key(n: str) -> str:
        return next(k for k in ranked if ms[k]["name"] == n)

    top4 = ranked[:4]
    pw = summary.get("pairwise_top", [])
    p_top4 = [p.get("wilcoxon_p_holm") for p in pw if p["a"] in top4 and p["b"] in top4]
    rb = A["rank_bootstrap"]
    disc = A["discrimination"]
    refh_api = [D[k]["references"]["hallucination_rate"] for k in api]
    refh_loc = [D[k]["references"]["hallucination_rate"] for k in loc]
    doi_api = [D[k]["references"]["doi_correct_rate"] for k in api]
    cheapest = min(api, key=lambda k: usd[k])
    within = [k for k in api if ov[k] >= ov[ranked[0]] - 0.05]
    cheapest_close = min(within, key=lambda k: usd[k])
    loc_usd = [usd[k] for k in loc if usd[k] is not None]
    loc_kwh = [(ms[k].get("cost") or {}).get("kwh_per_100") for k in loc]
    loc_kwh = [x for x in loc_kwh if x]
    gpu_h = sum((ms[k].get("cost") or {}).get("gpu_hours") or 0 for k in loc)
    cost = A["cost"]
    st = cost["aiu_by_stage"]
    J = A["judges"]
    crit = [v for v in J["alpha_ordinal_by_criterion"].values() if v is not None]
    leni = J["leniency_offset"].values()
    pass1_loc = [D[k]["coding"]["pass1"] for k in loc]
    rep_loc = [D[k]["coding"]["pass_repair"] for k in loc]
    pass1_api = [D[k]["coding"]["pass1"] for k in api]
    sa = A["silver_agreement"]
    fields = sum(sum(v.values()) for v in sa.values())
    unanimous = sum(v.get("unanimous", 0) for v in sa.values())
    best_local = loc[0]
    worst_api = api[-1]
    sah = key("Gemma2 9B CPT SahabatAI (Q8_0)")
    g4 = key("Gemma 4 8B")
    n = {
        "NModels": len(ranked), "NApi": len(api), "NLocal": len(loc), "NItems": sum(
            v["public"] + v["private"] for v in summary["items"].values()),
        "NApiCalls": thousands(cost["api_calls_total"]), "NBenchCalls": thousands(cost["api_calls_by_stage"].get("main", 0)
                                                                                   + cost["api_calls_by_stage"].get("pilot", 0)),
        "NJudgeCalls": thousands(cost["api_calls_by_stage"].get("judges", 0)), "NLocalCalls": thousands(cost["local_calls"]),
        "StudyUSD": f"{cost['usd_total']:.2f}", "JudgeUSD": f"{st.get('judges', 0) / 100:.2f}",
        "AnswerUSD": f"{(st.get('main', 0) + st.get('pilot', 0)) / 100:.2f}", "ProbeUSD": f"{st.get('cost-probes', 0) / 100:.2f}",
        "TokensInM": f"{cost['api_tokens']['in'] / 1e6:.1f}", "TokensOutM": f"{cost['api_tokens']['out'] / 1e6:.1f}",
        "GpuHours": f"{gpu_h:.1f}", "GpuUSD": f"{gpu_h * (summary['prices'].get('gpu_usd_per_hour') or 0):.2f}",
        "TopName": tex(name[ranked[0]]), "TopScore": f(ov[ranked[0]], 3), "SecondName": tex(name[ranked[1]]),
        "ThirdName": tex(name[ranked[2]]), "FourthName": tex(name[ranked[3]]), "FourthScore": f(ov[ranked[3]], 3),
        "SecondScore": f(ov[ranked[1]], 3), "ThirdScore": f(ov[ranked[2]], 3),
        **{f"Diff{c}{part}": signed(v) for c, k in zip("BCD", ranked[1:4]) for part, v in
           (("", rb[k]["diff_vs_leader"]), ("Lo", rb[k]["diff_ci"][0]), ("Hi", rb[k]["diff_ci"][1]))},
        "NoSigN": sum(1 for k in ranked[1:] if rb[k]["diff_ci"][0] <= 0),
        "JudgeAutoRho": f(A["judge_sensitivity"]["auto_only"]["rho"], 3),
        "JudgeOnlyRho": f(A["judge_sensitivity"]["judge_only"]["rho"], 3),
        "JudgeAutoShift": A["judge_sensitivity"]["auto_only"]["max_rank_shift"],
        "JudgeOnlyShift": A["judge_sensitivity"]["judge_only"]["max_rank_shift"],
        "PubRho": f(A["public_only"]["rho"], 3) if A.get("public_only") else "--",
        "PubMaxShift": A["public_only"]["max_rank_shift"] if A.get("public_only") else "--",
        "TopFourMinP": f(min(p_top4), 2) if p_top4 else "--",
        "PTopA": pct(rb[ranked[0]]["p_top1"]), "PTopB": pct(rb[ranked[1]]["p_top1"]), "PTopC": pct(rb[ranked[2]]["p_top1"]),
        "TopUSD": f(usd[ranked[0]], 2), "SecondUSD": f(usd[ranked[1]], 2), "ThirdUSD": f(usd[ranked[2]], 2),
        "FourthUSD": f(usd[ranked[3]], 2), "TopMedS": f(D[ranked[0]]["latency"]["median_s"], 1),
        "ThirdMedS": f(D[ranked[2]]["latency"]["median_s"], 1),
        "CheapName": tex(name[cheapest]), "CheapUSD": f(usd[cheapest], 3), "CheapScore": f(ov[cheapest], 3),
        "CheapCloseName": tex(name[cheapest_close]), "CheapCloseUSD": f(usd[cheapest_close], 3),
        "CostRatio": f"{usd[ranked[0]] / usd[cheapest]:.0f}", "CheapGap": f(ov[ranked[0]] - ov[cheapest], 2),
        "ParetoN": len(summary.get("pareto_usd_per_100_vs_overall", [])),
        "ParetoNames": ", ".join(tex(name[k]) for k in summary.get("pareto_usd_per_100_vs_overall", [])),
        "MethCeil": disc["methodology"]["api_at_ceiling"], "LangCeil": disc["language"]["api_at_ceiling"],
        "MethApiMin": f(disc["methodology"]["api_min"]), "LangApiMin": f(disc["language"]["api_min"]),
        "RefApiMin": f(disc["references"]["api_min"]), "RefApiMax": f(disc["references"]["api_max"]),
        "RefApiSD": f(disc["references"]["api_sd"]), "StructApiMin": f(disc["structure"]["api_min"]),
        "StructApiMax": f(disc["structure"]["api_max"]), "LatexApiMin": f(disc["latex"]["api_min"]),
        "ExtApiMin": f(disc["extraction"]["api_min"]), "ExtApiMax": f(disc["extraction"]["api_max"]),
        "HalApiMin": pct(min(refh_api)), "HalApiMax": pct(max(refh_api)), "HalLocMin": pct(min(refh_loc)),
        "HalLocMax": pct(max(refh_loc)), "DoiApiMin": pct(min(doi_api)), "DoiApiMax": pct(max(doi_api)),
        "BestLocalName": tex(name[best_local]), "BestLocalScore": f(ov[best_local], 3), "WorstApiName": tex(name[worst_api]),
        "WorstApiScore": f(ov[worst_api], 3), "GapRef": f(disc["references"]["gap"]), "GapCode": f(disc["coding"]["gap"]),
        "GapLang": f(disc["language"]["gap"]), "GapMeth": f(disc["methodology"]["gap"]),
        "SahLang": f(ms[sah]["tasks"]["language"]["mean"], 3), "GemmaFourLang": f(ms[g4]["tasks"]["language"]["mean"], 3),
        "SahCode": f(ms[sah]["tasks"]["coding"]["mean"]), "SahOverall": f(ov[sah], 3), "GemmaFourOverall": f(ov[g4], 3),
        "LocUSDMin": f(min(loc_usd), 2), "LocUSDMax": f(max(loc_usd), 2), "LocKwhMin": f(min(loc_kwh), 3),
        "LocKwhMax": f(max(loc_kwh), 3), "LocPassOneMin": f(min(pass1_loc)), "LocPassOneMax": f(max(pass1_loc)),
        "LocPassOneMean": f(statistics.mean(pass1_loc)), "LocRepairMean": f(statistics.mean(rep_loc)),
        "ApiPassOneMin": f(min(pass1_api)), "ApiPerfectCode": sum(1 for x in pass1_api if x == 1.0),
        "AlphaJudges": f(J["alpha_interval_all_judges"]), "NAllJudges": J["answers_all_judges"],
        "NJudgedAnswers": thousands(J["answers"]), "AlphaCritMin": f(min(crit)), "AlphaCritMax": f(max(crit)),
        "LeniencyRange": f(max(leni) - min(leni)),
        "RhoModelGrd": f(A["auto_vs_judge"]["grounded"]["rho_model_means"]),
        "RhoModelLng": f(A["auto_vs_judge"]["language"]["rho_model_means"]),
        "RhoAnsGrd": f(A["auto_vs_judge"]["grounded"]["rho_answers"]),
        "RhoAnsLng": f(A["auto_vs_judge"]["language"]["rho_answers"]),
        "SilverUnanimous": pct(unanimous / fields) if fields else "--",
        "SilverAnalysisNone": sa.get("analisis", {}).get("none", 0),
        "Substitutions": "no",
        "GenericApiMin": pct(min(D[k]["structure"]["generic_ch4_rate"] for k in api)),
        "GenericApiMax": pct(max(D[k]["structure"]["generic_ch4_rate"] for k in api)),
        "FixApiMin": f(min(D[k]["latex"]["fix_compiled"] for k in api)),
        "ForeignLocMax": pct(max(D[k]["grounded"]["foreign_citation_answers"] for k in loc)),
        "ForeignLocMin": pct(min(D[k]["grounded"]["foreign_citation_answers"] for k in loc)),
        "LocMethMax": f(max(ms[k]["tasks"]["methodology"]["mean"] for k in loc)),
        "LocMethMin": f(min(ms[k]["tasks"]["methodology"]["mean"] for k in loc)),
        "LocLangMin": f(min(ms[k]["tasks"]["language"]["mean"] for k in loc)),
        "LocLangMax": f(max(ms[k]["tasks"]["language"]["mean"] for k in loc)),
        "ForeignApiMax": pct(max(D[k]["grounded"]["foreign_citation_answers"] for k in api)),
        "UnsupApiMax": f(max(D[k]["grounded"]["unsupported_per_answer"] for k in api)),
        "NPremium": 9,
    }
    n.update(audit_numbers(audit, A.get("verification_sensitivity"), api, loc))
    n.update(per_model_numbers(summary, analysis, audit))
    return n


DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def macro_name(label: str) -> str:
    """``copilot:gpt-6.1-sol@low`` -> ``GptSixOneSol``; ``ollama:qwen2.5-coder:32b`` -> ``QwenTwoFiveCoderThreeTwoB``."""
    model = label.split(":", 1)[1].rsplit("@", 1)[0] if "@" in label else label.split(":", 1)[1]
    if "sahabatai" in model.lower():
        return "SahabatAI"
    out = []
    for part in model.replace(":", "-").replace(".", "-").replace("/", "-").split("-"):
        word = ""
        for ch in part:
            word += DIGITS[int(ch)] if ch.isdigit() else ch
        out.append(word[:1].upper() + word[1:])
    return "".join(c for c in "".join(out) if c.isalpha())


def per_model_numbers(summary: dict, analysis: dict, audit: Optional[dict] = None) -> dict:
    ms, D = summary["models"], analysis["diagnostics"]
    nowhere = (audit or {}).get("models") or {}
    n = {}
    for k, m in ms.items():
        a, d, c = macro_name(k), D[k], m.get("cost") or {}
        n.update({f"Score{a}": f(m["composite"]["overall"]["mean"], 3), f"Acad{a}": f(m["composite"]["academic"]["mean"], 3),
                  f"Code{a}": f(m["composite"]["coding"]["mean"], 3), f"Usd{a}": f(c.get("usd_per_100"), 3),
                  f"Kwh{a}": f(c.get("kwh_per_100"), 3), f"Fabr{a}": pct(d["references"]["hallucination_rate"]),
                  f"Doi{a}": pct(d["references"]["doi_correct_rate"]), f"Refs{a}": f(d["references"]["returned_mean"], 1),
                  f"PassOne{a}": f(d["coding"]["pass1"]), f"Repair{a}": f(d["coding"]["pass_repair"]),
                  f"Fix{a}": f(d["latex"]["fix_compiled"]), f"MedS{a}": f(d["latency"]["median_s"], 1),
                  f"Foreign{a}": pct(d["grounded"]["foreign_citation_answers"]),
                  f"Generic{a}": pct(d["structure"]["generic_ch4_rate"]),
                  f"None{a}": pct((nowhere.get(k) or {}).get("no_match_rate")),
                  f"Rank{a}": summary["ranking"].index(k) + 1})
        for t in TASKS:
            n[f"T{t.capitalize()}{a}"] = f(m["tasks"][t]["mean"])
    return n


def numbers_tex(n: dict) -> str:
    lines = ["% generated by paper_bench/make_tables.py from experiments/bench/{summary,analysis,refs_audit}.json — do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else None
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, suffix in (("en", ""), ("id", "-id")):
        (OUT / f"leaderboard{suffix}.tex").write_text(leaderboard_table(summary, lang), encoding="utf-8")
        (OUT / f"diagnostics{suffix}.tex").write_text(diagnostics_table(summary, analysis, audit, lang), encoding="utf-8")
    (OUT / "numbers.tex").write_text(numbers_tex(numbers(summary, analysis, audit)), encoding="utf-8")
    stale = OUT / "models.tex"
    if stale.exists():
        stale.unlink()
    print(f"tables -> {OUT}")


if __name__ == "__main__":
    main()
