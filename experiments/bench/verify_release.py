"""Check the AcademicBench release against itself: recompute the paper's statistics from the released per-item scores.

  python experiments/bench/verify_release.py

Needs only this repository: no API key, no GPU, no private data. From ``per_item.csv`` (every model x item score) it
recomputes the task and overall scores and the ranking (``summary.json``), the task-stratified rank bootstrap with the
paired difference to the leader, and the ranking's sensitivity to the judges, to the private task and to the
reference audit (``analysis.json``); it checks the served models of the main run, the SHA-256 of the public items
(``bench/MANIFEST.json``) and that the numbers printed in the paper (``paper_bench/generated/numbers.tex``) are the
recomputed ones. Exit status 0 when every check agrees within rounding. CI runs it on every push.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "bench"
sys.path[:0] = [str(ROOT), str(BENCH)]

import analyze as A  # noqa: E402
from paper_agent.bench import TASKS  # noqa: E402
from paper_agent.bench.schema import item_dirs, load_items  # noqa: E402


def load_per_item() -> dict:
    per: dict = defaultdict(dict)
    with (BENCH / "per_item.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[r["model"]][r["item"]] = {"task": r["task"], "score": float(r["score"]), "auto": float(r["auto"]),
                                          "judge": float(r["judge"]) if r["judge"] else None, "n_reps": 1}
    return dict(per)


def macros() -> dict:
    text = (ROOT / "paper_bench" / "generated" / "numbers.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text)}


def num(s: str) -> float:
    return float(s.replace("$-$", "-").replace("{,}", ""))


def main() -> int:
    per = load_per_item()
    summary = json.loads((BENCH / "summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((BENCH / "analysis.json").read_text(encoding="utf-8"))
    audit = json.loads((BENCH / "refs_audit.json").read_text(encoding="utf-8"))
    items = load_items(item_dirs(ROOT / "bench" / "items"))
    ms = summary["models"]
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))

    # 1. task and overall scores, ranking
    overall, worst = {}, 0.0
    for m, rows in per.items():
        by = defaultdict(list)
        for r in rows.values():
            by[r["task"]].append(r["score"])
        tm = {t: sum(v) / len(v) for t, v in by.items()}
        overall[m] = sum(tm[t] for t in TASKS) / len(TASKS)
        worst = max([worst, abs(overall[m] - ms[m]["composite"]["overall"]["mean"])]
                    + [abs(tm[t] - ms[m]["tasks"][t]["mean"]) for t in TASKS])
    ranking = sorted(overall, key=lambda m: -overall[m])
    n_items = sum(len(rows) for rows in per.values())
    check(f"task and overall scores ({len(per)} models x {len(TASKS)} tasks, {n_items} item scores)", worst <= 2e-4,
          f"max |diff| {worst:.6f} vs summary.json")
    check("overall ranking", ranking == summary["ranking"], "same order as summary.json"
          if ranking == summary["ranking"] else "order differs")

    # 2. rank bootstrap and paired differences to the leader
    rb, ref = A.rank_bootstrap(per), analysis["rank_bootstrap"]
    d_p = max(abs(rb[m]["p_top1"] - ref[m]["p_top1"]) for m in rb)
    d_ci = max(abs(a - b) for m in rb for a, b in zip(rb[m]["diff_ci"] + [rb[m]["diff_vs_leader"]],
                                                       ref[m]["diff_ci"] + [ref[m]["diff_vs_leader"]]))
    same_ranks = all(rb[m]["rank_ci"] == ref[m]["rank_ci"] for m in rb)
    check("rank bootstrap: P(rank 1), rank CIs, paired difference to the leader [95 % CI]",
          d_p <= 0.002 and d_ci <= 2e-4 and same_ranks,
          f"max |diff| P(top1) {d_p:.4f}, difference/CI {d_ci:.5f}; rank CIs {'identical' if same_ranks else 'differ'}")

    # 3. sensitivity analyses
    js = A.judge_sensitivity(per)
    d_js = max(abs(js[k]["rho"] - analysis["judge_sensitivity"][k]["rho"]) for k in js)
    sh_js = all(js[k]["max_rank_shift"] == analysis["judge_sensitivity"][k]["max_rank_shift"] for k in js)
    check("ranking with prose scored by automatic checks only / judges only", d_js <= 1e-3 and sh_js,
          f"Spearman rho {js['auto_only']['rho']:.3f} / {js['judge_only']['rho']:.3f}")
    po = A.public_only(per)
    ok_po = abs(po["rho"] - analysis["public_only"]["rho"]) <= 1e-3 and \
        po["max_rank_shift"] == analysis["public_only"]["max_rank_shift"]
    check("ranking without the private extraction task", ok_po,
          f"Spearman rho {po['rho']:.3f}, largest rank shift {po['max_rank_shift']}")
    vs, vref = A.verification_sensitivity(per, audit, items), analysis["verification_sensitivity"]
    ok_vs = abs(vs["rho_overall"] - vref["rho_overall"]) <= 1e-3 and vs["max_rank_shift"] == vref["max_rank_shift"] \
        and abs(vs["max_references_gain"] - vref["max_references_gain"]) <= 1e-3
    check("ranking with the works found by the reference audit credited", ok_vs,
          f"{vs['corrected_refs']} references, Spearman rho {vs['rho_overall']:.3f}, "
          f"largest rank shift {vs['max_rank_shift']}")

    # 4. provenance of the main run
    api = [m for m in ms if m.startswith("copilot:")]
    calls = sum(ms[m]["cost"]["calls"] for m in api)
    substituted = [m for m in api if set(ms[m]["cost"]["served"]) != {m.split(":", 1)[1].rsplit("@", 1)[0]}]
    check("served model = requested model (main run, API models)", not substituted,
          f"{calls} calls of {len(api)} models" + (f"; differs: {substituted}" if substituted else ""))

    # 5. public items vs the manifest
    manifest = json.loads((ROOT / "bench" / "MANIFEST.json").read_text(encoding="utf-8"))
    want = {i: h for t in manifest["tasks"].values() for i, h in t["items"].items()}
    bad = [it.id for it in items if want.get(it.id) != it.sha256()]
    private = sorted(set(want) - {it.id for it in items})
    check("public items match bench/MANIFEST.json (SHA-256)", not bad and len(want) == manifest["n_items"],
          f"{len(items)} public items identical, {len(private)} private items listed as hashes only"
          + (f"; changed: {bad[:5]}" if bad else ""))

    # 6. numbers printed in the paper
    mac = macros()
    api_rank = [m for m in ranking if m.startswith("copilot:")]
    loc_rank = [m for m in ranking if m.startswith("ollama:")]
    expect = {"TopScore": overall[ranking[0]], "SecondScore": overall[ranking[1]], "ThirdScore": overall[ranking[2]],
              "FourthScore": overall[ranking[3]], "BestLocalScore": overall[loc_rank[0]],
              "WorstApiScore": overall[api_rank[-1]], "JudgeAutoRho": js["auto_only"]["rho"], "PubRho": po["rho"],
              "VerRho": vs["rho_overall"]}
    for c, m in zip("BCD", ranking[1:4]):
        expect.update({f"Diff{c}": rb[m]["diff_vs_leader"], f"Diff{c}Lo": rb[m]["diff_ci"][0],
                       f"Diff{c}Hi": rb[m]["diff_ci"][1]})
    pct = {"PTopA": rb[ranking[0]]["p_top1"], "PTopB": rb[ranking[1]]["p_top1"], "PTopC": rb[ranking[2]]["p_top1"]}
    off = [k for k, v in expect.items() if abs(num(mac[k]) - v) > 6e-4]
    off += [k for k, v in pct.items() if abs(num(mac[k]) - 100 * v) > 0.51]
    off += [k for k, m in (("TopName", ranking[0]), ("SecondName", ranking[1]), ("ThirdName", ranking[2]))
            if mac[k] != ms[m]["name"]]
    check("numbers printed in the paper (generated/numbers.tex)", not off,
          f"{len(expect) + len(pct) + 3} macros recomputed" + (f"; differ: {off}" if off else ""))

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks agree" + ("" if not failed else f"; {failed} FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
