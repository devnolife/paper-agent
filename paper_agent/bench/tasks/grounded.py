"""grounded — write a short literature review from given sources only, citing them as [P1]…[Pn].

Items carry OpenAlex work IDs; titles, authors, years and abstracts are fetched once and frozen in a private
snapshot (``<private>/sources/openalex.json``) so every model sees the same prompt (abstracts are not redistributed).

Deterministic checks: ``coverage`` (share of sources cited), ``cited_sentences`` (share of sentences with a valid
citation), ``invalid_citations`` (numbers outside 1…n), ``foreign_citations`` (author–year citations, which the task
forbids), ``unsupported_numbers`` (numbers that occur in no source), and length. Auto score = mean of coverage,
cited_sentences, no-invalid, no-foreign, number support and length compliance. Faithfulness and quality are judged
by cross-vendor LLM judges and, on a subset, by people.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from ..schema import Item
from . import Context, Task, number_forms, numbers_in, words

SYSTEM = ("Kamu penulis akademik. Kamu menulis tinjauan pustaka berbahasa Indonesia baku yang setia pada sumber: "
          "setiap klaim berasal dari sumber yang diberikan dan diberi sitasi.")
USER = ("Topik: {topic}\n\nSumber (hanya informasi dari sumber ini yang boleh dipakai):\n{sources}\n\n"
        "Tulis tinjauan pustaka {lo}–{hi} kata dalam bahasa Indonesia baku tentang topik di atas berdasarkan sumber-"
        "sumber tersebut. Beri sitasi pada setiap klaim dalam bentuk [P1], [P2], dan seterusnya sesuai daftar sumber. "
        "Jangan memakai bentuk sitasi lain, jangan menambahkan fakta atau angka yang tidak ada di sumber, tanpa judul "
        "dan tanpa daftar pustaka.")

_CITE_GROUP = re.compile(r"\[([^\[\]]*\bP\s*\d+[^\[\]]*)\]")
_FOREIGN = [re.compile(r"\(\s*[A-Z][A-Za-z'’\-]+(?:\s+(?:et\s+al\.?|dkk\.?|&\s+[A-Z][A-Za-z'’\-]+))?,?\s+\d{4}[a-z]?\s*\)"),
            re.compile(r"\b[A-Z][A-Za-z'’\-]+(?:\s+(?:et\s+al\.?|dkk\.?))?\s+\(\d{4}[a-z]?\)")]


def abstract_text(inv: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex ``abstract_inverted_index`` -> plain text."""
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


ABSTRACT_MAX = 1200


def clip(text: str, limit: int = ABSTRACT_MAX) -> str:
    """Abstract cut at the last sentence end before ``limit`` characters (deterministic; keeps prompts short)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0] + " …"


def fetch_openalex(work_id: str) -> Optional[dict]:
    from ...references import get_json

    data = get_json(f"https://api.openalex.org/works/{work_id}", ttl=365 * 86400)
    if not data or not data.get("id"):
        return None
    authors = [((a or {}).get("author") or {}).get("display_name") or "" for a in data.get("authorships") or []]
    return {"id": work_id, "title": data.get("display_name") or data.get("title") or "",
            "year": data.get("publication_year"), "authors": [a for a in authors if a][:6],
            "abstract": clip(abstract_text(data.get("abstract_inverted_index"))), "doi": data.get("doi") or "",
            "language": data.get("language")}


def source_line(i: int, s: dict) -> str:
    au = s.get("authors") or []
    who = (au[0].split()[-1] + (" dkk." if len(au) > 1 else "")) if au else "Anonim"
    return f"[P{i}] {who} ({s.get('year') or 't.t.'}). {s.get('title', '').strip()}. Abstrak: {s.get('abstract', '').strip()}"


def cited_numbers(text: str) -> List[int]:
    out: List[int] = []
    for g in _CITE_GROUP.findall(text or ""):
        for a, b in re.findall(r"P\s*(\d+)(?:\s*[–-]\s*P?\s*(\d+))?", g):
            lo = int(a)
            hi = int(b) if b else lo
            out.extend(range(lo, min(hi, lo + 20) + 1))
    return out


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(“\"])", (text or "").strip())
    return [p for p in parts if len(words(p)) >= 4]


def number_support(text: str, sources: List[dict]) -> Tuple[int, int, List[str]]:
    """(numbers in the text, of which unsupported, the unsupported ones). Citation markers and years of the sources
    are not counted; a number is supported when one of its readings occurs in a source's title or abstract."""
    body = _CITE_GROUP.sub(" ", text or "")
    src = set()
    for s in sources:
        for m in numbers_in(f"{s.get('title', '')} {s.get('abstract', '')}"):
            src |= number_forms(m)
    years = {str(s.get("year")) for s in sources if s.get("year")}
    found = [m for m in numbers_in(body) if m.rstrip("%") not in years]
    bad = [m for m in found if not number_forms(m) & src]
    return len(found), len(bad), bad[:20]


def length_score(n: int, lo: int, hi: int) -> float:
    if lo <= n <= hi:
        return 1.0
    dev = (lo - n) if n < lo else (n - hi)
    return max(0.0, 1.0 - dev / hi)


class GroundedTask(Task):
    name = "grounded"
    json_mode = False
    max_tokens = 1400
    judged = True

    def sources(self, item: Item, ctx: Context) -> List[dict]:
        snap = self._snapshot(ctx)
        out = []
        for wid in item.input["openalex_ids"]:
            s = snap.get(wid)
            if s is None:
                if ctx.offline:
                    raise RuntimeError(f"source {wid} not in the snapshot and offline")
                s = fetch_openalex(wid)
                if s is None:
                    raise RuntimeError(f"OpenAlex work {wid} unavailable")
                snap[wid] = s
                self._save(ctx)
            out.append(s)
        return out

    def _snapshot(self, ctx: Context) -> Dict[str, dict]:
        if "grounded:snap" not in ctx.cache:
            path = ctx.private_dir / "sources" / "openalex.json" if ctx.private_dir else None
            ctx.cache["grounded:snap"] = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
        return ctx.cache["grounded:snap"]

    def _save(self, ctx: Context) -> None:
        if ctx.private_dir:
            path = ctx.private_dir / "sources" / "openalex.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ctx.cache["grounded:snap"], ensure_ascii=False, indent=1), encoding="utf-8")

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        src = self.sources(item, ctx)
        lines = "\n".join(source_line(i, s) for i, s in enumerate(src, 1))
        return SYSTEM, USER.format(topic=item.input["topic"], sources=lines, lo=item.input.get("min_words", 250),
                                   hi=item.input.get("max_words", 400))

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        text = record.get("text", "")
        src = self.sources(item, ctx)
        n = len(src)
        cites = cited_numbers(text)
        valid = {c for c in cites if 1 <= c <= n}
        invalid = [c for c in cites if not 1 <= c <= n]
        sents = sentences(text)
        cited_s = sum(1 for s in sents if any(1 <= c <= n for c in cited_numbers(s)))
        foreign = sum(len(rx.findall(text)) for rx in _FOREIGN)
        n_num, n_bad, bad = number_support(text, src)
        nw = len(words(text))
        lo, hi = item.input.get("min_words", 250), item.input.get("max_words", 400)
        parts = {"coverage": len(valid) / n if n else 0.0,
                 "cited_sentences": cited_s / len(sents) if sents else 0.0,
                 "no_invalid": 1.0 if not invalid else 0.0,
                 "no_foreign": 1.0 if foreign == 0 else 0.0,
                 "number_support": 1.0 - n_bad / n_num if n_num else 1.0,
                 "length": length_score(nw, lo, hi)}
        return {"score": round(sum(parts.values()) / len(parts), 4), "parts": {k: round(v, 4) for k, v in parts.items()},
                "words": nw, "sources": n, "cited": sorted(valid), "invalid_citations": len(invalid),
                "foreign_citations": foreign, "numbers": n_num, "unsupported_numbers": n_bad, "unsupported": bad,
                "sentences": len(sents)}
