"""Multi-model paper review with provenance.

Two critics (roles ``critic`` and ``critic2``, by default a Copilot model and a local model) review the
same paper independently against a fixed rubric and return JSON; a deterministic merge marks which
findings both critics raised (``agreed``) and which only one did. No single model is the judge of
another, and every finding carries the model that actually produced it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import engines
from .citations import strip_comments
from .config import roles as default_roles

RUBRIC = """You are a rigorous peer reviewer for an IEEE conference paper. Review the paper text below.
Return ONLY JSON with this schema:
{"summary": "<3-sentence summary of the contribution>",
 "scores": {"clarity": 1-5, "novelty": 1-5, "soundness": 1-5, "evidence": 1-5, "reproducibility": 1-5},
 "strengths": ["..."],
 "weaknesses": [{"issue": "<one sentence>", "where": "<section or quote>", "severity": "major|minor", "fix": "<concrete suggestion>"}],
 "overclaims": [{"claim": "<quoted or paraphrased>", "why": "<what the evidence does not support>"}],
 "missing": ["<experiment, baseline, statistic or related work that is missing>"],
 "recommendation": "accept|minor revision|major revision|reject"}
Be specific and quote the paper where possible. Judge only what is in the text; do not invent results."""


def paper_text(main_tex: Path, max_chars: int = 60000) -> str:
    """Flattened LaTeX source (inputs resolved, comments stripped, preamble dropped)."""
    main_tex = Path(main_tex)

    def load(p: Path, seen: set) -> str:
        p = p.resolve()
        if p in seen or not p.exists():
            return ""
        seen.add(p)
        txt = strip_comments(p.read_text(encoding="utf-8"))

        def sub(m):
            child = p.parent / m.group(1)
            if child.suffix != ".tex":
                child = child.with_suffix(".tex")
            return load(child, seen)
        return re.sub(r"\\(?:input|include)\{([^}]+)\}", sub, txt)

    text = load(main_tex, set())
    body = text.split("\\begin{document}", 1)[-1]
    body = re.sub(r"\\(usepackage|documentclass|graphicspath|newcommand)\b.*", "", body)
    return body[:max_chars]


def _norm_issue(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def _similar(a: str, b: str) -> bool:
    wa, wb = set(_norm_issue(a).split()), set(_norm_issue(b).split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.35


def review_paper(main_tex: Path, roles: Optional[Dict[str, Tuple[str, str]]] = None, out: Optional[Path] = None,
                 log=print) -> dict:
    roles = roles or default_roles()
    text = paper_text(main_tex)
    critics = [roles[k] for k in ("critic", "critic2") if k in roles]
    reviews: List[dict] = []
    for engine, model in critics:
        t0 = time.time()
        data = engines.chat_json(engine, model, RUBRIC, text, retries=1, timeout=900)
        served = engines.last_served() or f"{engine}:{model}"
        if not isinstance(data, dict):
            log(f"  critic {engine}:{model}: no valid JSON")
            continue
        data["_model_requested"] = f"{engine}:{model}"
        data["_model_served"] = served
        data["_seconds"] = round(time.time() - t0, 1)
        reviews.append(data)
        log(f"  critic {served}: {data.get('recommendation', '?')} in {data['_seconds']} s")
    merged = merge_reviews(reviews)
    result = {"paper": str(main_tex), "reviews": reviews, "merged": merged}
    if out:
        Path(out).write_text(review_markdown(result), encoding="utf-8")
        Path(out).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def merge_reviews(reviews: List[dict]) -> dict:
    issues: List[dict] = []
    for r in reviews:
        for w in r.get("weaknesses", []) or []:
            if not isinstance(w, dict):
                continue
            match = next((i for i in issues if _similar(i["issue"], str(w.get("issue", "")))), None)
            if match:
                match["raised_by"].append(r["_model_served"])
                match["fixes"].append(str(w.get("fix", "")))
            else:
                issues.append({"issue": str(w.get("issue", "")), "where": str(w.get("where", "")),
                               "severity": str(w.get("severity", "")), "fixes": [str(w.get("fix", ""))],
                               "raised_by": [r["_model_served"]]})
    for i in issues:
        i["agreed"] = len(set(i["raised_by"])) >= 2
    scores: Dict[str, List[float]] = {}
    for r in reviews:
        for k, v in (r.get("scores") or {}).items():
            try:
                scores.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass
    return {"issues": sorted(issues, key=lambda i: (not i["agreed"], i["severity"] != "major")),
            "mean_scores": {k: round(sum(v) / len(v), 2) for k, v in scores.items() if v},
            "recommendations": {r["_model_served"]: r.get("recommendation") for r in reviews},
            "n_agreed": sum(1 for i in issues if i["agreed"]), "n_issues": len(issues)}


def review_markdown(res: dict) -> str:
    m = res["merged"]
    rows = [f"# Review — {Path(res['paper']).name}", "",
            f"Critics (model that actually answered): {', '.join(r['_model_served'] for r in res['reviews'])}", "",
            f"Recommendations: {m['recommendations']}", f"Mean scores: {m['mean_scores']}",
            f"Issues: {m['n_issues']} ({m['n_agreed']} raised by both critics)", "", "## Issues", ""]
    for i in m["issues"]:
        tag = "AGREED" if i["agreed"] else "single"
        rows.append(f"- **[{i['severity'] or '?'} · {tag}]** {i['issue']} — _{i['where']}_")
        for f in i["fixes"]:
            if f:
                rows.append(f"  - fix: {f}")
        rows.append(f"  - raised by: {', '.join(i['raised_by'])}")
    for r in res["reviews"]:
        rows += ["", f"## {r['_model_served']} (requested {r['_model_requested']}, {r['_seconds']} s)", "",
                 f"**Summary.** {r.get('summary', '')}", "", "**Strengths**"]
        rows += [f"- {s}" for s in r.get("strengths", []) or []]
        if r.get("overclaims"):
            rows += ["", "**Overclaims**"] + [f"- {o.get('claim', '')} — {o.get('why', '')}" for o in r["overclaims"] if isinstance(o, dict)]
        if r.get("missing"):
            rows += ["", "**Missing**"] + [f"- {s}" for s in r["missing"]]
    return "\n".join(rows) + "\n"
