"""Figures for the SkripsiBench paper and README from ``experiments/bench/summary.json`` (+ ``per_item.csv``).

  python experiments/bench/make_figures.py            # -> paper_bench/figures/*.pdf and docs/img/bench_*.png
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    pts = []
    for label, m in summary["models"].items():
        q = m["composite"]["overall"]["mean"]
        c = (m.get("cost") or {}).get("usd_per_100")
        if q is None or not c:
            continue
        col, mk = TIER_STYLE.get(m["tier"], ("grey", "x"))
        ax.scatter(c, q, color=col, marker=mk, s=18, zorder=3)
        pts.append((c, q, short(m["name"]), label in summary.get("pareto_usd_per_100_vs_overall", [])))
    for c, q, n, on_front in pts:
        if on_front:
            ax.annotate(n, (c, q), fontsize=6, xytext=(3, 2), textcoords="offset points")
    front = sorted((c, q) for c, q, _n, f in pts if f)
    if front:
        ax.plot([c for c, _ in front], [q for _, q in front], color="black", lw=0.6, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("US$ per 100 items (API: AI units; local: L40S rental equivalent)")
    ax.set_ylabel("Overall score")
    for tier, (col, mk) in TIER_STYLE.items():
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


def hallucination(per_item: Path, summary: dict) -> None:
    if not per_item.exists():
        return
    by_model = {}
    for row in csv.DictReader(per_item.open(encoding="utf-8")):
        if row["task"] == "references":
            by_model.setdefault(row["model"], []).append(float(row["score"]))
    rows = sorted(((sum(v) / len(v), short(summary["models"][k]["name"]), summary["models"][k]["tier"])
                   for k, v in by_model.items() if k in summary["models"]), reverse=True)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.5, 0.15 * len(rows) + 0.8))
    ax.barh(range(len(rows)), [r[0] for r in rows], color=[TIER_STYLE.get(r[2], ("grey", ""))[0] for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Reference task score (verified, correct DOI)")
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
    fig, ax = plt.subplots(figsize=(3.5, 0.18 * len(rows) + 0.8))
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
    hallucination(SUMMARY.parent / "per_item.csv", summary)
    energy(summary)
    print(f"figures -> {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
