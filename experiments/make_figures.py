#!/usr/bin/env python3
"""Figures and the results table for the paper, computed from experiments/translation/*.json.

    python3 experiments/make_figures.py        # writes paper/figures/fig*.pdf and experiments/translation/summary.json
"""
from __future__ import annotations

import difflib
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "translation"
FIGS = ROOT / "paper" / "figures"
IMG = ROOT / "docs" / "img"                                 # PNG copies for the README
CONFIGS = {"multi": "multi.json", "haiku": "single_copilot_claude-haiku-4.5.json", "gemma3": "single_ollama_gemma3_27b.json"}
LABELS = {"multi": "multi\n(GPT-5 mini → Claude Haiku 4.5\n→ Gemma 3 QA)", "haiku": "single\nClaude Haiku 4.5", "gemma3": "single\nGemma 3 27B (local)"}
WALL = {"multi": 1334, "haiku": 425, "gemma3": 384}          # first full run, seconds (from the run logs)
GLOSSARY = {"multi": (284, 287), "haiku": (286, 287), "gemma3": (281, 287)}

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    """PDF for the paper, PNG for the README."""
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def final_metrics(d: dict) -> dict:
    U = d["units"]
    sims, backs, fails = [], [], 0
    for u in U.values():
        m = u["metrics"]
        if m.get("untranslated"):
            fails += 1
        s = m.get("selected", "draft")
        if m.get(f"sim_{s}") is not None:
            sims.append(m[f"sim_{s}"])
        if m.get(f"back_{s}_sim") is not None:
            backs.append(m[f"back_{s}_sim"])
    return {"units": len(U), "intact": len(U) - fails, "cosine_direct": st.mean(sims), "cosine_back": st.mean(backs),
            "cosine_direct_sd": st.pstdev(sims), "cosine_back_sd": st.pstdev(backs)}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    D = {k: json.loads((DATA / v).read_text(encoding="utf-8")) for k, v in CONFIGS.items()}
    S = {k: final_metrics(d) for k, d in D.items()}
    for k in S:
        S[k]["wall_s"] = WALL[k]
        S[k]["glossary"] = GLOSSARY[k][0] / GLOSSARY[k][1]
    # agreement between configurations' final texts
    def fin(d):
        return {k: u["final"] for k, u in d["units"].items()}
    F = {k: fin(d) for k, d in D.items()}
    agree = {}
    for a, b in (("multi", "haiku"), ("multi", "gemma3"), ("haiku", "gemma3")):
        agree[f"{a}-{b}"] = st.mean(difflib.SequenceMatcher(None, F[a][k], F[b][k]).ratio() for k in F[a] if F[a][k] and F[b].get(k))
    mu = D["multi"]["units"]
    ed = [u["metrics"]["review_edit"] for u in mu.values() if "review_edit" in u["metrics"]]
    S["multi"]["review_changed"] = sum(1 for e in ed if e > 0)
    S["multi"]["review_edit_mean"] = st.mean(ed)
    S["multi"]["selected"] = {s: sum(1 for u in mu.values() if u["metrics"].get("selected") == s) for s in ("draft", "review")}
    (DATA / "summary.json").write_text(json.dumps({"configs": S, "agreement": agree}, indent=1), encoding="utf-8")

    # ---- Fig. 1: architecture --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4", ec="#333", fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, text="", dy=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=6.4, color="#333")

    box(1, 17, 13, 9, "LaTeX / DOCX\nsource", fc="#ffffff")
    box(18, 17, 15, 9, "Mask\n[[n]] placeholders\n<k> tags", fc="#e8f1fb")
    box(37, 17, 15, 9, "Draft\n(role: draft)", fc="#fff3d6")
    box(56, 17, 15, 9, "Post-edit\n(role: review,\ndifferent model)", fc="#fff3d6")
    box(75, 17, 15, 9, "QA: back-translate\n(role: qa, local) +\nmultilingual cosine", fc="#fff3d6")
    box(37, 2, 53, 8, "Deterministic validators: placeholder multiset · number-glued placeholders · echo detection ·\n"
                      "glossary hits · citation gate · build QA  —  reject → retry → fallback model → keep source", fc="#e9f7e9", fs=6.8)
    box(92, 17, 7.5, 9, "Select\n& unmask", fc="#e8f1fb", fs=6.6)
    arrow(14, 21.5, 18, 21.5); arrow(33, 21.5, 37, 21.5); arrow(52, 21.5, 56, 21.5); arrow(71, 21.5, 75, 21.5); arrow(90, 21.5, 92, 21.5)
    for x in (44.5, 63.5, 82.5):
        arrow(x, 17, x, 10.4)
    ax.text(63.5, 28.6, "every call: requested model vs. model that answered (assistant.usage) → provenance log",
            ha="center", va="center", fontsize=6.6, style="italic", color="#444")
    save(fig, "fig1_architecture")

    # ---- Fig. 2: configuration comparison ------------------------------------------------
    keys = ["multi", "haiku", "gemma3"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    panels = [("cosine_direct", "cosine EN↔ID (final)", (0.78, 0.86)), ("cosine_back", "cosine EN↔back-transl.", (0.93, 0.975)),
              ("glossary", "glossary compliance", (0.95, 1.0)), ("wall_s", "wall time (s)", (0, 1500))]
    for ax, (key, title, ylim) in zip(axes, panels):
        vals = [S[k][key] for k in keys]
        err = [S[k].get(f"{key}_sd", 0) / (S[k]["units"] ** 0.5) for k in keys] if key.startswith("cosine") else None
        ax.bar(range(3), vals, color=colors, yerr=err, capsize=2, error_kw={"lw": 0.7})
        ax.set_xticks(range(3)); ax.set_xticklabels(["multi", "Claude\nHaiku", "Gemma 3"], fontsize=6.5)
        ax.set_title(title, fontsize=7.5); ax.set_ylim(*ylim)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=6.3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2_configs")

    # ---- Fig. 3: what the reviewer changed and what QA selected ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    qd = [u["metrics"]["qa_draft"] for u in mu.values() if "qa_draft" in u["metrics"] and "qa_review" in u["metrics"]]
    qr = [u["metrics"]["qa_review"] for u in mu.values() if "qa_draft" in u["metrics"] and "qa_review" in u["metrics"]]
    sel = [u["metrics"]["selected"] for u in mu.values() if "qa_draft" in u["metrics"] and "qa_review" in u["metrics"]]
    ax = axes[0]
    for s, c, m in (("review", "#4c72b0", "o"), ("draft", "#c44e52", "s")):
        xs = [a for a, b, t in zip(qd, qr, sel) if t == s]
        ys = [b for a, b, t in zip(qd, qr, sel) if t == s]
        ax.scatter(xs, ys, s=9, c=c, marker=m, label=f"selected {s} (n={len(xs)})", alpha=0.75, lw=0)
    lo, hi = min(qd + qr) - 0.02, 1.0
    ax.plot([lo, hi], [lo, hi], color="#888", lw=0.7, ls="--")
    ax.set_xlabel("QA score of draft (GPT-5 mini)"); ax.set_ylabel("QA score after post-edit (Claude Haiku 4.5)")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.set_title("(a) per-unit QA before/after post-editing", fontsize=7.5)
    ax = axes[1]
    ax.hist([e for e in ed if e > 0], bins=20, color="#4c72b0")
    ax.set_xlabel("edit ratio of post-editor (changed units only)"); ax.set_ylabel("units")
    ax.set_title(f"(b) post-editor changed {S['multi']['review_changed']}/{len(ed)} units", fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_review")

    # ---- README-only charts ------------------------------------------------------------------
    # (a) requested vs served model (Experiment A)
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    rows = [("Claude Haiku 4.5", "Claude Haiku 4.5"), ("GPT-5 mini", "GPT-5 mini"),
            ("Claude Opus 4.8", "Claude Haiku 4.5"), ("fictitious name", "Claude Haiku 4.5")]
    ax.set_xlim(0, 100); ax.set_ylim(0, len(rows) * 10 + 4); ax.axis("off")
    ax.text(17, len(rows) * 10 + 1.5, "requested", ha="center", fontsize=8, fontweight="bold")
    ax.text(83, len(rows) * 10 + 1.5, "served (usage event)", ha="center", fontsize=8, fontweight="bold")
    for i, (req, srv) in enumerate(rows):
        y = (len(rows) - 1 - i) * 10 + 2
        ok = req == srv
        ax.add_patch(FancyBboxPatch((1, y), 32, 7, boxstyle="round,pad=0.2", fc="#eef2f7", ec="#333", lw=0.8))
        ax.add_patch(FancyBboxPatch((67, y), 32, 7, boxstyle="round,pad=0.2", fc="#e9f7e9" if ok else "#fde2e2", ec="#333", lw=0.8))
        ax.text(17, y + 3.5, req, ha="center", va="center", fontsize=7.6)
        ax.text(83, y + 3.5, srv, ha="center", va="center", fontsize=7.6)
        ax.add_patch(FancyArrowPatch((33.5, y + 3.5), (66.5, y + 3.5), arrowstyle="-|>", mutation_scale=10, lw=0.9,
                                     color="#2e7d32" if ok else "#c62828"))
        ax.text(50, y + 4.6, "as requested" if ok else "silently substituted (no error)", ha="center", fontsize=6.2,
                color="#2e7d32" if ok else "#c62828")
    ax.set_title("Requested vs. served model through the vendor SDK (entry-level account)", fontsize=8)
    save(fig, "readme_substitution")

    # (b) which layer caught which error (Experiment C)
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    errors = [("placeholder moved before a number", "validator", 1), ("QA model echoed its input", "validator", 37),
              ("placeholders dropped (local model)", "validator", 2), ("sentence left untranslated", "post-editor", 1),
              ("wrong technical term", "post-editor + human", 4), ("misspelling", "human", 1),
              ("table cell / heading left or truncated", "human", 2)]
    palette = {"validator": "#2e7d32", "post-editor": "#4c72b0", "post-editor + human": "#8172b2", "human": "#dd8452"}
    ys = range(len(errors))[::-1]
    ax.barh(list(ys), [e[2] for e in errors], color=[palette[e[1]] for e in errors])
    ax.set_yticks(list(ys)); ax.set_yticklabels([e[0] for e in errors], fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(0.7, 60); ax.set_xlabel("occurrences (log scale)", fontsize=7)
    for y, e in zip(ys, errors):
        ax.text(e[2] * 1.15, y, f"{e[2]}  ·  caught by {e[1]}", va="center", fontsize=6.8, color=palette[e[1]])
    ax.set_title("Who caught which error (146 units)", fontsize=8)
    fig.tight_layout()
    save(fig, "readme_errors")
    print(json.dumps({"configs": S, "agreement": agree}, indent=1))


if __name__ == "__main__":
    main()
