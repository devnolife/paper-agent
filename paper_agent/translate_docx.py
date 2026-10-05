"""Translate a DOCX paper in place (format preserved) with the same engines.

Units are body paragraphs and table-cell paragraphs. Runs with identical formatting are grouped
into segments; when a paragraph has several segments they are wrapped in tags ``<1>…</1>`` so the
model can move words between formats without breaking bold/italic/superscript. Reference entries,
formulas, identifiers and ``--keep-ids`` units are left untouched. Translations are cached next to
the output (``<out>.translations.json``) so a re-run only sends what changed.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from docx import Document

from . import engines

TAG_RE = re.compile(r"</?(\d+)>")
REF_ENTRY_RE = re.compile(r"^\s*\[\d+\]\s")
KEEP = {"XGBoost", "LightGBM", "CatBoost", "LSTM", "GRU", "SVR", "RF", "RMSE", "MAE", "MAPE", "MSE", "SCADA",
        "RBF", "RandomizedSearchCV", "StandardScaler", "MinMaxScaler", "TimeSeriesSplit", "LLM", "NLI", "RAG"}

SYSTEM = """You are a professional academic translator. Translate the given English text into formal, natural {target} as used in scientific journals (IEEE-style papers).

Rules:
1. Output ONLY the translation of each unit, as JSON: {{"units": [{{"id": <id>, "text": "<translation>"}}, ...]}}. Same ids, same order, nothing else.
2. Keep inline tags such as <1>…</1> exactly where the corresponding words are; never drop, add, or reorder tag pairs.
3. Keep unchanged: numbers, units, citations like [4] or [8]–[10], model and variable names, acronyms, equations, file/dataset names, proper nouns, and line breaks (\\n).
4. Headings in ALL CAPS stay in ALL CAPS; keep heading numbering (I., II., A., B.) and figure/table numbering.
5. Use the glossary consistently:
{glossary}
6. Do not summarise, omit, or add sentences. Preserve hedging and precise technical meaning."""


@dataclass
class DocxUnit:
    uid: int
    para: object
    tagged: str
    segments: List[dict] = field(default_factory=list)
    translation: Optional[str] = None
    skipped: str = ""


def run_signature(run) -> tuple:
    f = run.font
    return (bool(run.bold), bool(run.italic), bool(run.underline), bool(f.superscript), bool(f.subscript),
            f.size, f.name, str(f.color.rgb) if f.color is not None and f.color.type is not None else None)


def iter_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            seen = set()
            for cell in row.cells:
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                for p in cell.paragraphs:
                    yield p


def should_skip(text: str, in_references: bool) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if in_references and REF_ENTRY_RE.match(t):
        return "reference entry"
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", t)
    alpha = [w for w in tokens if re.search(r"[A-Za-z]{3,}", w)]
    if not alpha:
        return "no prose"
    if all(w in KEEP or re.match(r"^[A-Z]+\d+$", w) or (w.isupper() and len(w) <= 7) for w in alpha):
        return "identifier"
    if "=" in t and len(alpha) <= 4:
        return "formula"
    return ""


def build_units(doc, keep_ids: set) -> List[DocxUnit]:
    units: List[DocxUnit] = []
    in_refs = False
    for i, p in enumerate(iter_paragraphs(doc)):
        text = p.text
        if text.strip().upper() in ("REFERENCES", "REFERENCE", "DAFTAR PUSTAKA", "BIBLIOGRAPHY"):
            in_refs = True
        skip = "kept by request" if i in keep_ids else should_skip(text, in_refs)
        runs = [r for r in p.runs if r.text]
        if not runs:
            continue
        segs: List[dict] = []
        for r in runs:
            sig = run_signature(r)
            if segs and segs[-1]["sig"] == sig:
                segs[-1]["runs"].append(r)
            else:
                segs.append({"sig": sig, "runs": [r]})
        if len(segs) == 1:
            tagged = "".join(r.text for r in segs[0]["runs"])
        else:
            parts = []
            for k, s in enumerate(segs, 1):
                s["tag"] = k
                parts.append(f"<{k}>" + "".join(r.text for r in s["runs"]) + f"</{k}>")
            tagged = "".join(parts)
        for s in segs:
            s["src"] = "".join(r.text for r in s["runs"])
        units.append(DocxUnit(uid=i, para=p, tagged=tagged, segments=segs, skipped=skip))
    return units


def tags_of(s: str) -> List[str]:
    return TAG_RE.findall(s)


def translate_batch(batch: List[DocxUnit], engine: str, model: str, system: str, retries: int = 2) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for _attempt in range(retries + 1):
        user = "Translate these units.\n" + json.dumps(
            {"units": [{"id": u.uid, "text": u.tagged} for u in batch]}, ensure_ascii=False, indent=1)
        data = engines.chat_json(engine, model, system, user, retries=0)
        items = (data.get("units") if isinstance(data, dict) else data) or []
        got = {int(it["id"]): str(it["text"]) for it in items if isinstance(it, dict) and "id" in it and "text" in it}
        for u in batch:
            t = got.get(u.uid)
            if t is None or not t.strip():
                continue
            if not tags_of(u.tagged) and tags_of(t):
                t = TAG_RE.sub("", t)
            if tags_of(t) != tags_of(u.tagged):
                continue
            if u.tagged.count("\n") != t.count("\n"):
                t = t.replace("\\n", "\n")
            out[u.uid] = t
        batch = [u for u in batch if u.uid not in out]
        if not batch:
            break
    return out


def _keep_edges(src: str, new: str) -> str:
    if not src.strip():
        return src
    return src[: len(src) - len(src.lstrip())] + new.strip() + src[len(src.rstrip()):]


def apply_translation(u: DocxUnit) -> None:
    t = u.translation or ""
    if len(u.segments) == 1:
        runs = u.segments[0]["runs"]
        runs[0].text = _keep_edges(u.segments[0]["src"], t)
        for r in runs[1:]:
            r.text = ""
        return
    pieces = {int(m.group(1)): m.group(2) for m in re.finditer(r"<(\d+)>(.*?)</\1>", t, re.DOTALL)}
    for s in u.segments:
        runs = s["runs"]
        runs[0].text = _keep_edges(s["src"], pieces.get(s["tag"], ""))
        for r in runs[1:]:
            r.text = ""


def translate_docx(src: Path, out: Optional[Path] = None, engine: str = "ollama", model: str = "gemma3:27b",
                   target: str = "Indonesian", glossary: Optional[Path] = None, batch_chars: int = 3000,
                   workers: int = 1, limit: int = 0, keep_ids: Optional[set] = None, dry_run: bool = False,
                   log=print) -> dict:
    src = Path(src)
    out_path = Path(out) if out else src.with_name(src.stem + "_ID.docx")
    cache_path = out_path.with_suffix(".translations.json")
    gl = "   (none)"
    if glossary and Path(glossary).exists():
        g = yaml.safe_load(Path(glossary).read_text(encoding="utf-8")) or {}
        gl = "\n".join(f"   - {k} → {v}" for k, v in g.items())
    system = SYSTEM.format(target=target, glossary=gl)

    doc = Document(str(src))
    units = build_units(doc, keep_ids or set())
    todo = [u for u in units if not u.skipped]
    log(f"units: {len(units)}  to translate: {len(todo)}  skipped: {len(units) - len(todo)}")
    if dry_run:
        for u in units:
            log(f"  {u.uid:3d} {('SKIP(' + u.skipped + ')') if u.skipped else 'TR':22s} {u.tagged[:70]!r}")
        return {"units": len(units), "to_translate": len(todo), "dry_run": True}
    if limit:
        todo = todo[:limit]

    cache: Dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for u in todo:
        if str(u.uid) in cache and tags_of(cache[str(u.uid)]) == tags_of(u.tagged):
            u.translation = cache[str(u.uid)]
    pending = [u for u in todo if u.translation is None]
    batches: List[List[DocxUnit]] = []
    cur: List[DocxUnit] = []
    size = 0
    for u in pending:
        if cur and size + len(u.tagged) > batch_chars:
            batches.append(cur); cur, size = [], 0
        cur.append(u); size += len(u.tagged)
    if cur:
        batches.append(cur)
    t0 = time.time()

    def run(batch: List[DocxUnit]):
        return batch, translate_batch(batch, engine, model, system)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for bi, (batch, got) in enumerate(pool.map(run, batches), 1):
            for u in batch:
                if u.uid in got:
                    u.translation = got[u.uid]
                    cache[str(u.uid)] = got[u.uid]
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"[{bi}/{len(batches)}] {len(batch)} units … {len(got)}/{len(batch)} ok")
    failed = [u.uid for u in todo if u.translation is None]
    for u in todo:
        if u.translation is not None:
            apply_translation(u)
    doc.save(str(out_path))
    rep = {"output": str(out_path), "translated": len(todo) - len(failed), "failed_ids": failed,
           "wall_s": round(time.time() - t0), "engine": f"{engine}:{model}", "served_models": engines.served_summary()}
    log(f"wrote {out_path}  ({rep['translated']} translated, {len(failed)} left untranslated, {rep['wall_s']} s)")
    return rep


if __name__ == "__main__":  # pragma: no cover
    from .cli import main
    sys.exit(main(["translate-docx", *sys.argv[1:]]))
