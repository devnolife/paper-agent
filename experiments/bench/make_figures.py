"""Figures for the SkripsiBench paper and README from ``experiments/bench/summary.json`` (+ ``per_item.csv``).

  python experiments/bench/make_figures.py            # -> paper_bench/figures/*.pdf and docs/img/bench_*.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments" / "bench" / "summary.json"
OUT_PDF = ROOT / "paper_bench" / "figures"
OUT_PNG = ROOT / "docs" / "img"
TASKS = ["structure", "methodology", "extraction", "grounded", "references", "language", "coding", "latex"]
TIER_STYLE = {"economy": ("#2b8cbe", "o"), "standard": ("#e6550d", "s"), "premium": ("#756bb1", "D"),
              "local": ("#31a354", "^")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name: str) -> None:
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF / f"{name}.pdf")
    fig.savefig(OUT_PNG / f"bench_{name}.png", dpi=200)
    plt.close(fig)


def short(name: str) -> str:
    return (name.replace("Qwen2.5-Coder", "Q2.5-Coder").replace("Qwen3-Coder", "Q3-Coder")
            .replace(" (Q8_0)", "").replace("Gemma2 9B CPT SahabatAI", "SahabatAI 9B"))


def pareto(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    pts = []
    for label, m in summary["models"].items():
        q = m["composite"]["overall"]["mean"]
        c = (m.get("cost") or {}).get("usd_per_100")
        if q is None or not c:
            continue
        col, mk = TIER_STYLE.get(m["tier"], ("grey", "x"))
        ax.scatter(c, q, color=col, marker=mk, s=18, zorder=3)
        pts.append((c, q, short(m["name"]), label in summary.get("pareto_usd_per_100_vs_overall", [])))
    front_pts = sorted((c, q, n) for c, q, n, on_front in pts if on_front)
    for i, (c, q, n) in enumerate(front_pts):       # alternate above/below so neighbouring labels do not collide
        last = i == len(front_pts) - 1
        ax.annotate(n, (c, q), fontsize=5.5, xytext=(-4, -9 if last or i % 2 else 5), textcoords="offset points",
                    ha="right" if last else "left")
    if front_pts:
        ax.plot([c for c, _q, _n in front_pts], [q for _c, q, _n in front_pts], color="black", lw=0.6, ls="--",
                zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("US$ per 100 items (API: AI units; local: L40S rental equivalent)")
    ax.set_ylabel("Overall score")
    present = {m["tier"] for m in summary["models"].values()}
    for tier, (col, mk) in TIER_STYLE.items():
        if tier in present:
            ax.scatter([], [], color=col, marker=mk, s=18, label=tier)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    save(fig, "pareto")


def heatmap(summary: dict) -> None:
    ranking = summary["ranking"]
    data = [[summary["models"][k]["tasks"][t]["mean"] for t in TASKS] for k in ranking]
    names = [short(summary["models"][k]["name"]) for k in ranking]
    fig, ax = plt.subplots(figsize=(3.5, 0.17 * len(ranking) + 0.9))
    vals = [[float("nan") if v is None else v for v in row] for row in data]
    im = ax.imshow(vals, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5.5)
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.2, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, "heatmap")


def hallucination(analysis_path: Path, summary: dict) -> None:
    """Listed references not found in Crossref and, from the second-source audit, found in no database, per model."""
    if not analysis_path.exists():
        return
    D = json.loads(analysis_path.read_text(encoding="utf-8"))["diagnostics"]
    audit_path = analysis_path.parent / "refs_audit.json"
    nowhere = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        nowhere = {k: v.get("no_match_rate") for k, v in audit["models"].items()}
    rows = sorted(((D[k]["references"]["hallucination_rate"], nowhere.get(k), short(m["name"]), m["tier"],
                    D[k]["references"]["returned_mean"]) for k, m in summary["models"].items()
                   if k in D and D[k]["references"]["hallucination_rate"] is not None), key=lambda r: (r[0], r[1] or 0))
    fig, ax = plt.subplots(figsize=(3.5, 0.125 * len(rows) + 0.75))
    cols = [TIER_STYLE.get(r[3], ("grey", ""))[0] for r in rows]
    ax.barh(range(len(rows)), [100 * r[0] for r in rows], color=cols, alpha=0.35 if nowhere else 1.0)
    if nowhere:
        ax.barh(range(len(rows)), [100 * (r[1] or 0) for r in rows], color=cols)
    for i, r in enumerate(rows):
        both = f"{100 * r[0]:.0f}%" + (f" / {100 * r[1]:.0f}%" if r[1] is not None else "")
        ax.text(100 * r[0] + 1, i, f"{both} of {20 * r[4]:.0f}", va="center", fontsize=4.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xlabel("Listed references not found in Crossref / in no database (%), of n listed" if nowhere
                  else "Listed references not found in Crossref (%), of n listed")
    tiers = [t for t in TIER_STYLE if any(r[3] == t for r in rows)]
    handles = [Patch(color=TIER_STYLE[t][0], label=t) for t in tiers]
    if nowhere:
        handles += [Patch(facecolor="grey", alpha=0.35, label="not in Crossref"),
                    Patch(facecolor="grey", label="in no database")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6)
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    save(fig, "references")


def energy(summary: dict) -> None:
    rows = []
    for k, m in summary["models"].items():
        c = m.get("cost") or {}
        if m["tier"] == "local" and c.get("kwh_per_100"):
            rows.append((c["kwh_per_100"], short(m["name"]), m["composite"]["overall"]["mean"]))
    if not rows:
        return
    rows.sort()
    fig, ax = plt.subplots(figsize=(3.5, 0.15 * len(rows) + 0.65))
    ax.barh(range(len(rows)), [r[0] for r in rows], color="#31a354")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[1]} ({r[2]:.2f})" if r[2] is not None else r[1] for r in rows], fontsize=6)
    ax.set_xlabel("GPU board energy, kWh per 100 items (overall score in brackets)")
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    save(fig, "energy")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pareto(summary)
    heatmap(summary)
    hallucination(SUMMARY.parent / "analysis.json", summary)
    energy(summary)
    print(f"figures -> {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
