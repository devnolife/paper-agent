"""references — ask for n real references (with DOI) on a research topic; verify each in Crossref.

Crossref is the DOI registration agency for journal articles and proceedings, so a work with a DOI exists when the
DOI resolves there to a record matching the title (similarity ≥ 0.90, the same rule as ``paper-agent verify-refs``),
the year (± 1) and the first author; a work given with a wrong or no DOI is looked up by title and author. arXiv
DOIs (10.48550) are checked against arXiv. Crossref answers that are not "found"/"not found" (rate limits, server
errors, network) are retried and, if they persist, make the item *unscored* rather than wrong.

Credit per reference (at most n counted, duplicates ignored): 1.0 exists with the correct DOI, 0.75 exists without a
DOI, 0.5 exists with a wrong DOI, 0 not found (fabricated). Score = total credit / n requested — abstaining costs the
same as a fabricated reference; fabrication is reported separately as ``hallucination_rate`` (not found / returned)
and ``doi_unresolvable`` (DOIs that resolve nowhere).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import List, Optional, Tuple
from urllib.parse import quote

from ..schema import Item
from . import Context, Task, parse_json_loose

SYSTEM = ("Kamu asisten riset akademik yang teliti. Kamu hanya mencantumkan rujukan ilmiah yang benar-benar ada "
          "dan tidak pernah mengarang rujukan.")
USER = ("Topik penelitian: {topic}\n\n"
        "Berikan {n} rujukan ilmiah (artikel jurnal atau makalah prosiding) yang benar-benar ada dan relevan dengan "
        "topik tersebut, lengkap dengan DOI. Jangan mencantumkan rujukan yang tidak kamu yakini keberadaannya; lebih "
        "baik mencantumkan lebih sedikit. Keluarkan HANYA JSON: {{\"rujukan\": [{{\"penulis\": [\"Nama Belakang, Nama "
        "Depan\"], \"tahun\": 2020, \"judul\": \"...\", \"venue\": \"...\", \"doi\": \"10.xxxx/...\"}}]}}")


def _year(v) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(v or ""))
    return int(m.group(0)) if m else None


def _authors(v) -> List[str]:
    if isinstance(v, str):
        v = re.split(r";| and | dan |&", v)
    out = []
    for a in v or []:
        a = str(a).strip()
        if "," in a:                       # "Last, First" -> "First Last" (verify_entry takes the last token)
            last, _, first = a.partition(",")
            a = f"{first.strip()} {last.strip()}".strip()
        if a:
            out.append(a)
    return out


def seeds_from_answer(obj, n: int) -> List[dict]:
    refs = []
    if isinstance(obj, dict):
        refs = obj.get("rujukan") or obj.get("references") or obj.get("referensi") or []
    elif isinstance(obj, list):
        refs = obj
    seeds, seen = [], set()
    for r in refs if isinstance(refs, list) else []:
        if not isinstance(r, dict):
            continue
        title = str(r.get("judul") or r.get("title") or "").strip()
        key = re.sub(r"[^a-z0-9]", "", title.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        seeds.append({"key": f"r{len(seeds) + 1}", "type": "article", "title": title,
                      "authors": _authors(r.get("penulis") or r.get("authors")), "year": _year(r.get("tahun") or r.get("year")),
                      "venue": str(r.get("venue") or r.get("jurnal") or ""), "doi": str(r.get("doi") or "").strip(),
                      "verify": "auto"})
        if len(seeds) >= n:
            break
    return seeds


class Unavailable(RuntimeError):
    """Crossref did not give a definite answer (rate limit, server error, network) after retries."""


class Crossref:
    """Minimal Crossref client for the benchmark: disk cache, polite rate, retries, and 404 kept apart from errors."""
    MIN_INTERVAL = 0.06

    def __init__(self) -> None:
        from ...config import cache_dir
        from ...references import UA
        mail = os.getenv("CROSSREF_EMAIL") or os.getenv("UNPAYWALL_EMAIL") or ""
        self.params = {"mailto": mail} if mail else {}
        self.headers = {"User-Agent": UA.format(mail=mail or "unknown")}
        self.dir = cache_dir() / "bench-crossref"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = 0.0

    def get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        import requests
        key = hashlib.sha1(json.dumps([url, params or {}], sort_keys=True).encode()).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                return None if cached.get("missing") else cached["data"]
            except (ValueError, KeyError, OSError):
                pass
        for attempt in range(6):
            with self._lock:
                wait = self._last + self.MIN_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = requests.get(url, params={**self.params, **(params or {})}, headers=self.headers, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (404, 400):          # no such DOI / malformed DOI: a definite "not found"
                path.write_text(json.dumps({"missing": True, "at": time.time()}), encoding="utf-8")
                return None
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                time.sleep(min(60.0, float(ra)) if ra and ra.isdigit() else 2.0 * 2 ** attempt)
                continue
            if not r.ok:
                break
            try:
                data = r.json()
            except ValueError:
                break
            path.write_text(json.dumps({"data": data, "at": time.time()}), encoding="utf-8")
            return data
        raise Unavailable(url)

    def doi(self, doi: str):
        from ...references import Local
        data = self.get(f"https://api.crossref.org/works/{quote(doi, safe='/')}")
        return Local._crossref_rec(data["message"]) if data and data.get("message") else None

    def search(self, title: str, author: str = "", rows: int = 10) -> list:
        from ...references import Local
        params = {"query.title": title, "rows": rows}
        if author:
            params["query.author"] = author
        data = self.get("https://api.crossref.org/works", params) or {}
        return [Local._crossref_rec(it) for it in (data.get("message") or {}).get("items", [])]


def verify_ref(seed: dict, cr: Crossref) -> dict:
    """{"exists", "doi_ok", "doi_confirmed", "doi_resolves", "via", "notes"} for one listed reference."""
    from ...references import Local, best_match, clean_doi, surname
    notes: List[str] = []
    given = clean_doi(seed.get("doi"))
    resolves = None
    if given:
        if given.startswith("10.48550/"):
            rec = Local().arxiv(given.split("arxiv.", 1)[-1]) if "arxiv." in given else None
        else:
            rec = cr.doi(given)
        resolves = rec is not None
        if rec is not None and best_match(seed, [rec], notes):
            return {"exists": True, "doi_ok": True, "doi_confirmed": given, "doi_resolves": True, "via": "doi",
                    "notes": notes}
    first = surname(seed["authors"][0]) if seed.get("authors") else ""
    m = best_match(seed, cr.search(seed["title"], first), notes)
    if m is None and first:
        m = best_match(seed, cr.search(seed["title"]), notes)
    if m:
        confirmed = clean_doi(m[0].doi)
        return {"exists": True, "doi_ok": bool(given) and given == confirmed, "doi_confirmed": confirmed or None,
                "doi_resolves": resolves, "via": "title", "notes": notes}
    return {"exists": False, "doi_ok": False, "doi_confirmed": None, "doi_resolves": resolves, "via": None,
            "notes": notes}


def credit(exists: bool, doi_given: bool, doi_ok: bool) -> float:
    if not exists:
        return 0.0
    return 1.0 if doi_ok else (0.75 if not doi_given else 0.5)


class ReferencesTask(Task):
    name = "references"
    max_tokens = 1500
    needs_network = True

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        return SYSTEM, USER.format(topic=item.input["topic"], n=item.input.get("n", 5))

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        n = int(item.input.get("n", 5))
        obj = parse_json_loose(record.get("text", ""))
        seeds = seeds_from_answer(obj, n)
        cr = ctx.cache.get("references:crossref")
        if cr is None:
            cr = ctx.cache.setdefault("references:crossref", Crossref())
        rows, total = [], 0.0
        for s in seeds:
            if not s["title"] or s["year"] is None:
                v = {"exists": False, "doi_ok": False, "doi_confirmed": None, "doi_resolves": None, "via": None,
                     "notes": ["missing title or year"]}
            else:
                try:
                    v = verify_ref(s, cr)
                except Unavailable as exc:      # no definite answer: leave the item unscored, retry later
                    return {"score": None, "error": f"verification unavailable: {exc}"[:300]}
            given = bool(s["doi"])
            c = credit(v["exists"], given, v["doi_ok"])
            total += c
            rows.append({"title": s["title"][:200], "year": s["year"], "exists": v["exists"], "via": v["via"],
                         "doi_given": s["doi"] or None, "doi_confirmed": v["doi_confirmed"], "doi_ok": v["doi_ok"],
                         "doi_resolves": v["doi_resolves"], "credit": c})
        returned = len(rows)
        not_found = sum(1 for r in rows if not r["exists"])
        return {"score": round(total / n, 4), "parsed": obj is not None, "returned": returned,
                "found": returned - not_found, "not_found": not_found,
                "hallucination_rate": round(not_found / returned, 4) if returned else None,
                "doi_correct": sum(1 for r in rows if r["doi_ok"]),
                "doi_wrong": sum(1 for r in rows if r["doi_given"] and r["exists"] and not r["doi_ok"]),
                "doi_unresolvable": sum(1 for r in rows if r["doi_given"] and r["doi_resolves"] is False),
                "refs": rows}
