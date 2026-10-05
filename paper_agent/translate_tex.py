"""EN -> target-language translation of LaTeX sections with a multi-model pipeline.

Pipelines
  multi (default)   draft  : translator model          (role ``draft``)
                    review : post-editor model          (role ``review``) fixes terminology/grammar
                    qa     : back-translation           (role ``qa``) + cross-lingual cosine
                             (sentence-transformers paraphrase-multilingual-MiniLM-L12-v2);
                    select : per unit keep draft or review, whichever scores higher on QA
  single:<engine>:<model>   one model only (QA still measured so configurations can be compared)

Deterministic validators run on every unit and reject model output that violates them (retry,
then fallback model, then the English text is kept and reported):
  * placeholder/tag multiset preserved; placeholders glued to numbers stay glued
  * back-translation that merely echoes its input counts as a failed QA call
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from . import engines
from .config import roles as default_roles
from .latex_mask import PH_RE, TAG_RE, Doc, Unit, fix_ref_words, integrity_ok

SECTIONS_DEFAULT = ["abstract", "introduction", "related", "method", "experiments", "results", "limitations",
                    "conclusion"]

SYSTEM_TRANSLATE = """You are a professional academic translator. Translate each English unit into formal, natural {target} as used in scientific journals (IEEE-style paper).

Rules:
1. Output ONLY JSON: {{"units": [{{"id": <id>, "text": "<translation>"}}, ...]}} — same ids, same order, nothing else.
2. Placeholders like ⟦3⟧ stand for LaTeX code (math, citations, references, identifiers). Copy every placeholder EXACTLY once, in the position where the corresponding element belongs in the {target} sentence. Never drop, add, duplicate, or alter a placeholder. A placeholder glued to a number (e.g. 58⟦3⟧ = "58\\%") must stay glued to that number.
3. Inline tags <1>…</1> mark emphasised spans: keep each pair around the translation of the same words; never drop or reorder tag pairs.
4. Keep LaTeX typography: ``double quotes'' stay as `` and '' (do not use “ ”); --- (em dash) and -- (en dash) stay as they are. Percent signs are already placeholders. Keep decimal POINTS exactly as in the source (0.885, 85.6) because equations and tables use points.
5. Keep unchanged: numbers, statistics (p, U, CI), metric names, model and tool names, rule ids, proper nouns and author names.
6. Use the glossary consistently (English term -> {target} term):
{glossary}
7. Do not summarise, omit, or add sentences. Preserve hedging and precise technical meaning. Headings stay short headings; sentence case as in the source."""

SYSTEM_REVIEW = """You are a senior bilingual (English/{target}) editor of computer-science papers. For each unit you receive the English source and a draft {target} translation. Return an improved {target} version: fix mistranslations, untranslated English, unnatural word order, inconsistent terminology (apply the glossary), grammar and spelling; keep sentences formal and concise. Do NOT add or remove information. If the draft is already correct, return it unchanged.

Hard constraints (violations make the output unusable):
- Placeholders ⟦n⟧ and inline tags <k>…</k> must all be present exactly as in the draft (same set, same count); only their position may change, except that a placeholder glued to a number (58⟦3⟧) stays glued to it.
- Keep LaTeX quotes `` '' and dashes --- / -- as they are; keep decimal points (0.885, 85.6) — do not convert to decimal commas.
- Output ONLY JSON: {{"units": [{{"id": <id>, "text": "<improved translation>", "changes": "<short note or empty>"}}, ...]}}.

Glossary (English -> {target}):
{glossary}"""

SYSTEM_BACK = """You are a literal translator. Translate each unit back into plain English as faithfully as possible (no paraphrase, no improvement). Copy placeholders ⟦n⟧ and tags <k>…</k> unchanged. Output ONLY JSON: {"units": [{"id": <id>, "text": "<english>"}, ...]}."""


def glossary_text(path: Optional[Path]) -> str:
    if not path or not Path(path).exists():
        return "(none)"
    g = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return "\n".join(f"- {k} -> {v}" for k, v in g.items())


def glossary_hits(path: Optional[Path], units: List[Unit]) -> Tuple[int, int]:
    """(#glossary terms found in sources whose target term appears in the final, #found)."""
    if not path or not Path(path).exists():
        return 0, 0
    g = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ok = tot = 0
    for u in units:
        if u.final is None:
            continue
        src, fin = u.masked.lower(), u.final.lower()
        for en, tgt in g.items():
            if en.endswith(" N") or not re.search(r"\b" + re.escape(en.lower()) + r"\b", src):
                continue
            tot += 1
            ok += str(tgt).lower().split(" (")[0] in fin
    return ok, tot


def run_stage(units: List[Unit], stage: str, engine: str, model: str, system: str, batch_chars: int,
              workers: int, build_item: Callable[[Unit], dict], accept: Callable[[Unit, str, dict], bool],
              log=print) -> None:
    """Send units in batches; ``accept`` stores a valid result; failed units are retried once alone."""
    todo = [u for u in units if stage not in u.out]
    batches: List[List[Unit]] = []
    cur: List[Unit] = []
    size = 0
    for u in todo:
        if cur and size + len(u.masked) > batch_chars:
            batches.append(cur); cur, size = [], 0
        cur.append(u); size += len(u.masked)
    if cur:
        batches.append(cur)

    def work(batch: List[Unit]) -> List[Unit]:
        user = json.dumps({"units": [build_item(u) for u in batch]}, ensure_ascii=False, indent=1)
        t0 = time.time()
        data = engines.chat_json(engine, model, system, user)
        items = (data.get("units") if isinstance(data, dict) else data) or []
        got = {int(it["id"]): it for it in items if isinstance(it, dict) and "id" in it and "text" in it}
        dt = (time.time() - t0) / max(1, len(batch))
        failed = []
        for u in batch:
            it = got.get(u.uid)
            if it and accept(u, str(it["text"]).strip(), it):
                u.metrics[f"{stage}_s"] = round(dt, 2)
                u.metrics[f"{stage}_model"] = f"{engine}:{model}"
            else:
                failed.append(u)
        return failed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        failed = [u for fl in ex.map(work, batches) for u in fl]
    if failed:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            failed = [u for fl in ex.map(work, [[u] for u in failed]) for u in fl]
    log(f"  {stage:11s} {engine}:{model}: {len(todo) - len(failed)}/{len(todo)} units ok")


_embedder = None


def embed_sim(a: List[str], b: List[str]) -> List[float]:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    import numpy as np
    clean = [PH_RE.sub(" ", TAG_RE.sub("", s)) for s in a + b]
    v = _embedder.encode(clean, normalize_embeddings=True, batch_size=32)
    return [float(np.dot(x, y)) for x, y in zip(v[: len(a)], v[len(a):])]


def edit_ratio(a: str, b: str) -> float:
    return round(1 - difflib.SequenceMatcher(None, a, b).ratio(), 3)


def parse_pipeline(spec: str, base: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, Tuple[str, str]]:
    r = dict(base or default_roles())
    if spec == "multi":
        return r
    if spec.startswith("single:"):
        _, engine, model = spec.split(":", 2)
        r["draft"] = (engine, model)
        r.pop("review", None)
        return r
    raise ValueError(f"unknown pipeline {spec!r} (multi | single:<engine>:<model>)")


def translate_tex(src_dir: Path, out_dir: Path, files: Optional[List[str]] = None, pipeline: str = "multi",
                  glossary: Optional[Path] = None, target: str = "Bahasa Indonesia", batch_chars: int = 2500,
                  workers: int = 2, qa: bool = True, limit: int = 0, cache_dir: Optional[Path] = None,
                  roles: Optional[Dict[str, Tuple[str, str]]] = None, log=print) -> dict:
    """Translate ``src_dir/<file>.tex`` into ``out_dir``; returns the report dict (also written as JSON/MD)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    roles = parse_pipeline(pipeline, roles)
    cache_dir = Path(cache_dir) if cache_dir else out_dir.parent / "translation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", pipeline)
    cache_path = cache_dir / f"{tag}.json"
    files = files or [p.stem for p in sorted(src_dir.glob("*.tex"))]

    doc = Doc()
    for name in files:
        doc.add_file(name, (src_dir / f"{name}.tex").read_text(encoding="utf-8"))
    units = doc.units[:limit] if limit else doc.units
    n_words = sum(u.words for u in units)
    log(f"{len(units)} units, {n_words} words, {sum(len(u.ph) for u in units)} placeholders, "
        f"{sum(len(u.tags) for u in units)} tags; pipeline={pipeline}")

    if cache_path.exists():
        cached_all = json.loads(cache_path.read_text(encoding="utf-8"))
        for k, v in cached_all.get("served_models", {}).items():
            engines.SERVED[k] = engines.SERVED.get(k, 0) + v
        cached = cached_all.get("units", {})
        for u in units:
            c = cached.get(str(u.uid))
            if c and c.get("masked") == u.masked:
                u.out, u.final, u.metrics = c.get("out", {}), c.get("final"), c.get("metrics", {})
                for stage in ("draft", "review"):   # a back-translation equal to its input is a failed QA call
                    if u.out.get(f"back_{stage}", "").strip() == u.out.get(stage, "\0").strip():
                        u.out.pop(f"back_{stage}", None)
                        for key in (f"sim_{stage}", f"back_{stage}_sim", f"qa_{stage}"):
                            u.metrics.pop(key, None)
    gl = glossary_text(glossary)

    def save() -> None:
        data = {"pipeline": pipeline, "roles": {k: list(v) for k, v in roles.items()},
                "served_models": engines.served_summary(),
                "units": {str(u.uid): {"file": u.file, "line": u.line, "masked": u.masked, "out": u.out,
                                       "final": u.final, "metrics": u.metrics} for u in units}}
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def accept_draft(u: Unit, t: str, it: dict) -> bool:
        if not integrity_ok(u.masked, t):
            return False
        u.out["draft"] = t
        return True

    def item_src(u: Unit) -> dict:
        return {"id": u.uid, "text": u.masked}

    t_start = time.time()
    eng, mod = roles["draft"]
    sys_tr = SYSTEM_TRANSLATE.format(target=target, glossary=gl)
    run_stage(units, "draft", eng, mod, sys_tr, batch_chars, workers, item_src, accept_draft, log)
    save()
    missing = [u for u in units if "draft" not in u.out]
    if missing and roles.get("fallback") and roles["fallback"] != roles["draft"]:
        eng, mod = roles["fallback"]
        run_stage(missing, "draft", eng, mod, sys_tr, batch_chars, 1, item_src, accept_draft, log)
        save()
    for u in units:
        if "draft" not in u.out:
            u.out["draft"] = u.masked
            u.metrics["untranslated"] = True

    if "review" in roles:
        eng, mod = roles["review"]
        sys_rv = SYSTEM_REVIEW.format(target=target, glossary=gl)

        def accept_review(u: Unit, t: str, it: dict) -> bool:
            if not integrity_ok(u.masked, t):
                return False
            u.out["review"] = t
            u.metrics["review_note"] = str(it.get("changes", ""))[:200]
            u.metrics["review_edit"] = edit_ratio(u.out["draft"], t)
            return True

        run_stage([u for u in units if not u.metrics.get("untranslated")], "review", eng, mod, sys_rv,
                  batch_chars, workers, lambda u: {"id": u.uid, "source": u.masked, "draft": u.out["draft"]},
                  accept_review, log)
        save()

    cands = ["draft"] + (["review"] if "review" in roles else [])
    if qa:
        eng, mod = roles["qa"]
        for stage in cands:
            key = f"back_{stage}"

            def accept_back(u: Unit, t: str, it: dict, k: str = key, s: str = stage) -> bool:
                if t.strip() == u.out[s].strip():
                    return False
                u.out[k] = t
                return True

            pend = [u for u in units if stage in u.out and key not in u.out and not u.metrics.get("untranslated")]
            run_stage(pend, key, eng, mod, SYSTEM_BACK, batch_chars, workers,
                      lambda u, s=stage: {"id": u.uid, "text": u.out[s]}, accept_back, log)
        save()
        for stage in cands:
            sel = [u for u in units if stage in u.out and f"sim_{stage}" not in u.metrics]
            if not sel:
                continue
            direct = embed_sim([u.masked for u in sel], [u.out[stage] for u in sel])
            back = embed_sim([u.masked for u in sel], [u.out.get(f"back_{stage}", "") for u in sel])
            for u, d, b in zip(sel, direct, back):
                u.metrics[f"sim_{stage}"] = round(d, 3)
                if f"back_{stage}" in u.out:
                    u.metrics[f"back_{stage}_sim"] = round(b, 3)
                    u.metrics[f"qa_{stage}"] = round(0.5 * d + 0.5 * b, 3)
                else:
                    u.metrics[f"qa_{stage}"] = round(d, 3)
    for u in units:
        best = cands[-1]
        if qa and len(cands) > 1:
            qd, qr = u.metrics.get("qa_draft", 0), u.metrics.get("qa_review", 0)
            best = "review" if qr >= qd - 0.02 else "draft"   # reviewer wins unless clearly worse
        if best not in u.out:
            best = "draft"
        u.final = u.out[best]
        u.metrics["selected"] = best
    save()
    elapsed = time.time() - t_start

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (out_dir / f"{name}.tex").write_text(fix_ref_words(doc.render(name)), encoding="utf-8")

    def mean(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 3) if xs else None

    g_ok, g_tot = glossary_hits(glossary, units)
    report = {
        "pipeline": pipeline, "roles": {k: f"{e}:{m}" for k, (e, m) in roles.items()},
        "units": len(units), "words": n_words,
        "untranslated": sum(1 for u in units if u.metrics.get("untranslated")),
        "served_models": engines.served_summary(), "wall_s": round(elapsed),
        "glossary": {"ok": g_ok, "total": g_tot},
        "stages": {},
        "selected": {s: sum(1 for u in units if u.metrics.get("selected") == s) for s in cands},
        "out_dir": str(out_dir), "cache": str(cache_path),
    }
    for stage in cands:
        st = {"s_per_unit": mean([u.metrics[f"{stage}_s"] for u in units if f"{stage}_s" in u.metrics])}
        if qa:
            st["cosine_direct"] = mean([u.metrics[f"sim_{stage}"] for u in units if f"sim_{stage}" in u.metrics])
            st["cosine_back"] = mean([u.metrics[f"back_{stage}_sim"] for u in units if f"back_{stage}_sim" in u.metrics])
        if stage == "review":
            ed = [u.metrics["review_edit"] for u in units if "review_edit" in u.metrics]
            st["edit_ratio"] = mean(ed)
            st["changed_units"] = sum(1 for e in ed if e > 0)
        report["stages"][stage] = st
    (cache_dir / f"{tag}_report.md").write_text(report_markdown(report), encoding="utf-8")
    log(report_markdown(report))
    return report


def report_markdown(r: dict) -> str:
    rows = [f"# Translation report — pipeline `{r['pipeline']}`", "",
            f"- units: {r['units']} ({r['words']} words); untranslated (kept source language): {r['untranslated']}",
            "- roles: " + ", ".join(f"{k}={v}" for k, v in r["roles"].items()),
            f"- served models (copilot may substitute): {r['served_models']}",
            f"- wall time this run: {r['wall_s']} s",
            f"- glossary compliance: {r['glossary']['ok']}/{r['glossary']['total']}", ""]
    for stage, st in r["stages"].items():
        rows.append(f"## stage `{stage}`")
        rows += [f"- {k}: {v}" for k, v in st.items()]
        rows.append("")
    rows.append(f"- selected: {r['selected']}")
    return "\n".join(rows) + "\n"


def dry_run(src_dir: Path, files: Optional[List[str]] = None, show: int = 12) -> dict:
    src_dir = Path(src_dir)
    files = files or [p.stem for p in sorted(src_dir.glob("*.tex"))]
    doc = Doc()
    for name in files:
        doc.add_file(name, (src_dir / f"{name}.tex").read_text(encoding="utf-8"))
    info = {"units": len(doc.units), "words": sum(u.words for u in doc.units),
            "placeholders": sum(len(u.ph) for u in doc.units), "tags": sum(len(u.tags) for u in doc.units),
            "sample": [f"[{u.uid}] {u.file}:{u.line} {u.masked[:120]}" for u in doc.units[:show]],
            "roundtrip_ok": all(doc.render(n) == (src_dir / f"{n}.tex").read_text(encoding="utf-8") for n in files)}
    return info


if __name__ == "__main__":  # pragma: no cover
    from .cli import main
    sys.exit(main(["translate-tex", *sys.argv[1:]]))
