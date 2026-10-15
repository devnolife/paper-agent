"""Score run records with the deterministic task scorers; build extraction silver labels from a model panel.

Scores go to ``<run_dir>/scores/<model-slug>.jsonl`` keyed by item and repetition, together with a hash of the
answer(s), the scorer version, the item hash and a fingerprint of reference data kept outside the item (silver/gold
labels), so re-scoring only touches what changed — including a corrected answer key. Network-bound (references) and
sandbox-bound (coding, LaTeX) scoring runs in a small thread pool.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .runner import latest_records
from .schema import Item
from .tasks import Context, registry
from .tasks import extraction as X

SCORER_VERSION = {"structure": 2, "methodology": 2, "extraction": 2, "grounded": 2, "references": 4, "language": 1,
                  "coding": 1, "latex": 2}


def slug(label: str) -> str:
    engine, _, rest = label.partition(":")
    model, _, effort = rest.rpartition("@") if "@" in rest else (rest, "", "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{engine}__{model}__{effort or 'default'}")


def answer_hash(rec: dict) -> str:
    texts = [rec.get("text") or ""] + [f.get("text") or "" for f in rec.get("followups") or []]
    return hashlib.sha256("\x00".join(texts + [rec.get("family") or ""]).encode("utf-8")).hexdigest()[:16]


def _load_scores(path: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(s["item"], s.get("rep", 0))] = s
    return out


def score_runs(items: List[Item], ctx: Context, run_dir: Path, models: Optional[List[str]] = None,
               tasks: Optional[List[str]] = None, workers: int = 4, log: Callable = print) -> dict:
    reg = registry()
    by_id = {it.id: it for it in items}
    item_sha = {it.id: it.sha256()[:16] for it in items}
    sdir = Path(run_dir) / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    totals = {"scored": 0, "reused": 0, "skipped": 0, "failed": 0}
    for model, recs in sorted(latest_records(run_dir).items()):
        if models and model not in models:
            continue
        path = sdir / f"{slug(model)}.jsonl"
        old = _load_scores(path)
        jobs, keep = [], {}
        for key, rec in recs.items():
            it = by_id.get(rec["item"])
            if it is None or (tasks and it.task not in tasks):
                continue
            if rec.get("error"):
                keep[key] = {"item": it.id, "task": it.task, "rep": key[1], "model": model, "error": rec["error"],
                             "score": None}
                continue
            h, ver = answer_hash(rec), SCORER_VERSION[it.task]
            ref = f"{item_sha[it.id]}:{reg[it.task].reference_fingerprint(it, ctx)}"
            prev = old.get(key)
            if prev and prev.get("answer_hash") == h and prev.get("version") == ver and prev.get("ref") == ref \
                    and prev.get("score") is not None:
                keep[key] = prev
                totals["reused"] += 1
                continue
            jobs.append((key, it, rec, h, ver, ref))

        def work(job):
            key, it, rec, h, ver, ref = job
            try:
                res = reg[it.task].score(it, rec, ctx)
            except Exception as exc:  # noqa: BLE001
                res = {"score": None, "error": f"scorer: {type(exc).__name__}: {exc}"[:400]}
            return key, {"item": it.id, "task": it.task, "rep": key[1], "model": model, "answer_hash": h,
                         "version": ver, "ref": ref, **res}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for key, row in pool.map(work, jobs):
                keep[key] = row
                if row.get("score") is None:
                    totals["failed" if row.get("error") else "skipped"] += 1
                else:
                    totals["scored"] += 1
        merged = {**{k: v for k, v in old.items() if k not in keep and k[0] in by_id}, **keep}
        path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for _, v in sorted(merged.items())),
                        encoding="utf-8")
        if jobs:
            log(f"  {model}: {len(jobs)} scored")
    log(f"scores: {totals}")
    return totals


def load_scores(run_dir: Path) -> Dict[str, Dict[Tuple[str, int], dict]]:
    out = {}
    for p in sorted((Path(run_dir) / "scores").glob("*.jsonl")):
        rows = _load_scores(p)
        if rows:
            out[next(iter(rows.values()))["model"]] = rows
    return out


# ----------------------------------------------------------------------------- extraction silver labels
def majority(records: List[dict], min_agree: int) -> dict:
    """Field-wise majority of normalised extraction records. Scalar fields without ``min_agree`` agreeing votes are
    left out (not scored). List fields vote per element: an element is *in* with ≥ ``min_agree`` votes, *out* when
    ≥ ``min_agree`` records omit it, otherwise *undecided* (``<field>__undecided``, ignored when scoring)."""
    out: dict = {}
    if not records:
        return out
    n = len(records)
    for f in X.FIELDS:
        if f in X.LISTS:
            votes = Counter(v for r in records for v in set(r.get(f) or []))
            out[f] = sorted(v for v, c in votes.items() if c >= min_agree)
            und = sorted(v for v, c in votes.items() if c < min_agree and n - c < min_agree)
            if und:
                out[f + "__undecided"] = und
        else:
            votes = Counter(json.dumps(r.get(f)) for r in records)
            val, c = votes.most_common(1)[0]
            if c >= min_agree:
                out[f] = json.loads(val)
    return out


def build_silver(items: List[Item], run_dir: Path, panel: List[str], private_dir: Path, log: Callable = print) -> dict:
    """Silver labels for extraction items from the panel's latest answers: ``*`` = majority of the whole panel
    (≥ 2 of 3), and per vendor family the agreement of the panel members outside that family."""
    recs = latest_records(run_dir)
    fam_of = {}
    for label in panel:
        r = next(iter(recs.get(label, {}).values()), None)
        if r is None:
            raise ValueError(f"panel model {label} has no records in {run_dir}")
        fam_of[label] = r.get("family") or "other"
    silver: Dict[str, dict] = {}
    for it in (i for i in items if i.task == "extraction"):
        got = {}
        for label in panel:
            r = recs[label].get((it.id, 0))
            if r and not r.get("error"):
                got[label] = X.normalize(X.parse_json_loose(r.get("text", "")))
        if len(got) < 2:
            continue
        entry = {"*": majority(list(got.values()), min_agree=(len(got) // 2) + 1)}
        for fam in sorted(set(fam_of.values())):
            others = [g for lab, g in got.items() if fam_of[lab] != fam]
            if len(others) >= 2:
                entry[fam] = majority(others, min_agree=len(others) if len(others) == 2 else (len(others) // 2) + 1)
        silver[it.id] = entry
    path = Path(private_dir) / "silver" / "extraction.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"panel": panel, "families": fam_of, "labels": silver}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    fields = Counter(f for e in silver.values() for f in e["*"])
    log(f"silver: {len(silver)} items, agreed fields {dict(fields)} -> {path}")
    return {"items": len(silver), "fields": dict(fields), "path": str(path)}
