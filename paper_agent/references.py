"""Verify references against independent bibliographic sources and emit BibTeX.

Policy: citations are never written from memory. Every ``verify: auto`` entry of the seed YAML must
match — title similarity >= 0.90 and year within ±1 — in at least two independent sources
(Crossref, OpenAlex, Semantic Scholar, Scopus, arXiv) to be VERIFIED. BibTeX for DOI entries comes
from doi.org content negotiation (validated against the seed); otherwise it is built from the
matched metadata. Books/software marked ``verify: manual`` get BibTeX from the seed and status MANUAL.

Sources: direct Crossref/OpenAlex/arXiv APIs (disk-cached 30 days, 429 cooldown) and, when
configured, the studio-revisi browser-agent (``BROWSER_AGENT_URL``/``BROWSER_AGENT_API_KEY``) for
Semantic Scholar and Scopus.

Seed format (YAML)::

    entries:
      - key: zheng2023
        type: inproceedings | article | arxiv | book | misc
        title: ...
        authors: [First Last, ...]
        year: 2023
        venue: ...            # optional
        doi: ...              # optional (checked, corrected when wrong)
        arxiv: 2306.05685     # optional
        verify: auto | manual
"""
from __future__ import annotations

import difflib
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
import yaml

from .config import cache_dir

TITLE_MIN = 0.90
TITLE_MIN_NO_AUTHOR = 0.95
RELAX_MIN_TITLE_CHARS = 40
YEAR_TOL = 1
AGENT_TIMEOUT = 60
UA = "paper-agent/0.1 (mailto:{mail})"


# ----------------------------------------------------------------------------- http cache
_COOLDOWN: Dict[str, float] = {}
_LAST: Dict[str, float] = {}


def get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, ttl: int = 30 * 86400,
             min_interval: float = 0.0, timeout: float = 30) -> Optional[dict]:
    """GET JSON with a disk cache; None when unavailable (network error, 4xx/5xx, host in 429 cooldown)."""
    key = hashlib.sha1(json.dumps([url, params or {}], sort_keys=True).encode()).hexdigest()
    path = cache_dir() / "http" / f"{key}.json"
    if path.exists():
        try:
            c = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - c.get("at", 0) <= ttl:
                return c.get("data")
        except (ValueError, OSError):
            pass
    host = urlparse(url).netloc
    if _COOLDOWN.get(host, 0) > time.time():
        return None
    if min_interval:
        wait = _LAST.get(host, 0) + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        print(f"    {host}: {exc}", file=sys.stderr)
        return None
    finally:
        _LAST[host] = time.time()
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        _COOLDOWN[host] = time.time() + (float(ra) if ra and ra.isdigit() else 60)
        print(f"    {host}: 429, cooling down", file=sys.stderr)
        return None
    if not r.ok:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": time.time(), "data": data}), encoding="utf-8")
    return data


# ----------------------------------------------------------------------------- helpers
def norm(s: Optional[str]) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def title_sim(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    full = difflib.SequenceMatcher(None, na, nb).ratio()
    ma, mb = norm(a.split(":")[0]), norm(b.split(":")[0])
    main = difflib.SequenceMatcher(None, ma, mb).ratio() if min(len(ma), len(mb)) >= 15 else 0.0
    return max(full, main)


def surname(author: str) -> str:
    parts = (author or "").split()
    return norm(parts[-1]) if parts else ""


def clean_doi(doi: Optional[str]) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (doi or "").strip().lower())


def is_arxiv_doi(doi: str) -> bool:
    return doi.startswith("10.48550/")


@dataclass
class Record:
    source: str
    title: str
    year: Optional[int]
    doi: str = ""
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    url: str = ""
    extra: Dict = field(default_factory=dict)


class RateLimited(Exception):
    pass


@dataclass
class Result:
    key: str
    status: str
    sources: List[str]
    doi: str
    sim: float
    notes: List[str]
    bibtex: str


# ----------------------------------------------------------------------------- sources
class Agent:
    """studio-revisi browser-agent (optional): Semantic Scholar queue + Scopus."""

    def __init__(self, enabled: bool = True):
        self.url = os.getenv("BROWSER_AGENT_URL", "http://127.0.0.1:8780").rstrip("/")
        self.key = os.getenv("BROWSER_AGENT_API_KEY", "")
        self.ok = False
        if enabled and self.key:
            try:
                r = requests.get(f"{self.url}/health", timeout=5)
                self.ok = r.ok and r.json().get("ok") is True
            except (requests.RequestException, ValueError):
                self.ok = False

    def search(self, source: str, q: str, limit: int = 5, **extra) -> Optional[List[Record]]:
        try:
            r = requests.post(f"{self.url}/v1/scholar/search", headers={"X-API-Key": self.key},
                              json={"q": q, "source": source, "limit": limit, **extra}, timeout=AGENT_TIMEOUT)
        except requests.RequestException as exc:
            print(f"    agent {source}: {exc}", file=sys.stderr)
            return None
        if r.status_code == 429:
            raise RateLimited(f"{source}: {r.text[:80]}")
        if not r.ok:
            return None
        return [Record(source=source, title=it.get("title") or "", year=it.get("year"), doi=clean_doi(it.get("doi")),
                       authors=[a for a in (it.get("authors") or []) if a], venue=it.get("journal") or "",
                       url=it.get("url") or "") for it in r.json().get("results", [])]


class Local:
    """Direct Crossref / OpenAlex / Semantic Scholar / arXiv calls (cached)."""

    def __init__(self):
        self.mail = os.getenv("CROSSREF_EMAIL") or os.getenv("UNPAYWALL_EMAIL") or ""
        self.headers = {"User-Agent": UA.format(mail=self.mail or "unknown")}
        self.s2_headers = dict(self.headers)
        if key := os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip():
            self.s2_headers["x-api-key"] = key

    def _mail(self) -> dict:
        return {"mailto": self.mail} if self.mail else {}

    @staticmethod
    def _crossref_rec(it: dict) -> Record:
        parts = (it.get("issued") or {}).get("date-parts") or [[None]]
        return Record("crossref", (it.get("title") or [""])[0], parts[0][0], clean_doi(it.get("DOI")),
                      [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author") or []],
                      (it.get("container-title") or [""])[0], it.get("URL") or "", {"raw": it})

    @staticmethod
    def _openalex_rec(w: dict) -> Record:
        loc = w.get("primary_location") or {}
        return Record("openalex", w.get("display_name") or w.get("title") or "", w.get("publication_year"),
                      clean_doi(w.get("doi")),
                      [((a or {}).get("author") or {}).get("display_name") or "" for a in (w.get("authorships") or [])],
                      ((loc.get("source") or {}).get("display_name")) or "", w.get("id") or "",
                      {"biblio": w.get("biblio") or {}, "type": w.get("type")})

    def crossref_doi(self, doi: str) -> Optional[Record]:
        data = get_json(f"https://api.crossref.org/works/{doi}", params=self._mail(), headers=self.headers)
        return self._crossref_rec(data["message"]) if data and data.get("message") else None

    def openalex_doi(self, doi: str) -> Optional[Record]:
        data = get_json(f"https://api.openalex.org/works/https://doi.org/{doi}", params=self._mail(), headers=self.headers)
        return self._openalex_rec(data) if data and data.get("id") else None

    def search(self, source: str, q: str, limit: int = 10, title: str = "", author: str = "") -> Optional[List[Record]]:
        if source == "crossref":
            params = {"rows": limit, **self._mail()}
            if title:
                params["query.title"] = title
                if author:
                    params["query.author"] = author
            else:
                params["query.bibliographic"] = q
            data = get_json("https://api.crossref.org/works", params=params, headers=self.headers)
            return None if data is None else [self._crossref_rec(it) for it in (data.get("message") or {}).get("items", [])]
        if source == "openalex":
            params = {"per_page": limit, **self._mail()}
            if title:
                params["filter"] = "title.search:" + re.sub(r"[,:()\"]", " ", title)
            else:
                params["search"] = q
            data = get_json("https://api.openalex.org/works", params=params, headers=self.headers)
            return None if data is None else [self._openalex_rec(w) for w in data.get("results", [])]
        if source == "semantic_scholar":
            data = get_json("https://api.semanticscholar.org/graph/v1/paper/search",
                            params={"query": q, "limit": limit, "fields": "title,year,authors,externalIds,venue,journal,url"},
                            headers=self.s2_headers, min_interval=1.1)
            if data is None:
                return None
            out = []
            for p in data.get("data", []):
                ext = p.get("externalIds") or {}
                out.append(Record("semantic_scholar", p.get("title") or "", p.get("year"), clean_doi(ext.get("DOI")),
                                  [a.get("name") or "" for a in p.get("authors") or []],
                                  (p.get("journal") or {}).get("name") or p.get("venue") or "", p.get("url") or "",
                                  {"arxiv": ext.get("ArXiv")}))
            return out
        raise ValueError(source)

    def arxiv(self, arxiv_id: str) -> Optional[Record]:
        r = None
        for attempt in range(2):
            time.sleep(3)
            try:
                r = requests.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id},
                                 headers=self.headers, timeout=20)
            except requests.RequestException:
                r = None
                continue
            if r.status_code == 429:
                wait = r.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit() else 10 * (attempt + 1))
                continue
            break
        if r is None or not r.ok:
            return None
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            entry = ET.fromstring(r.text).find("a:entry", ns)
        except ET.ParseError:
            return None
        if entry is None or entry.find("a:title", ns) is None:
            return None
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        if title.lower().startswith("error"):
            return None
        published = entry.findtext("a:published", "", ns) or ""
        cat = entry.find("{http://arxiv.org/schemas/atom}primary_category")
        return Record("arxiv", title, int(published[:4]) if published[:4].isdigit() else None, "",
                      [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)], "arXiv",
                      f"https://arxiv.org/abs/{arxiv_id}",
                      {"arxiv": arxiv_id, "primary_class": cat.get("term") if cat is not None else ""})

    def bibtex_from_doi(self, doi: str) -> Optional[str]:
        try:
            r = requests.get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8",
                                                               **self.headers}, timeout=30)
        except requests.RequestException:
            return None
        return r.text.strip() if r.ok and r.text.strip().startswith("@") else None


# ----------------------------------------------------------------------------- matching & bibtex
def best_match(seed: Dict, cands: List[Record], notes: List[str]) -> Optional[tuple]:
    first = surname(seed["authors"][0]) if seed.get("authors") else ""
    may_relax = len(norm(seed["title"])) >= RELAX_MIN_TITLE_CHARS and seed.get("type") != "arxiv"
    best = None
    for c in cands:
        sim = title_sim(seed["title"], c.title)
        if c.year is not None and abs(int(c.year) - int(seed["year"])) > YEAR_TOL:
            continue
        author_ok = True
        if first and len(first) >= 3 and c.authors:
            author_ok = any(first in norm(a) for a in c.authors)
        if not author_ok and not may_relax:
            continue
        need = TITLE_MIN if author_ok else TITLE_MIN_NO_AUTHOR
        if sim >= need and (best is None or sim > best[1]):
            best = (c, sim, author_ok)
    if best and not best[2]:
        notes.append(f"{best[0].source}: first-author surname not found (title sim {best[1]:.2f})")
    return best


def _bib_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%")


def normalize_bibtex(text: str) -> str:
    """Fix doi.org/Crossref BibTeX for LaTeX (entities, bare &, month macros, dashes); DOI -> note (IEEEtran.bst ignores doi)."""
    text = html.unescape(text)
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    text = re.sub(r"month\s*=\s*\{?([A-Za-z]+)\}?", lambda m: "month=" + m.group(1).lower()[:3], text)
    text = re.sub(r"(pages\s*=\s*\{[^}]*?)[\u2013\u2014](?=[^}]*\})", r"\1--", text)
    text = text.replace("url={http://dx.doi.org/", "url={https://doi.org/")
    doi = re.search(r"^\s*doi\s*=\s*\{([^}]+)\}", text, re.IGNORECASE | re.MULTILINE)
    if doi:
        text = re.sub(r"^\s*url\s*=\s*\{https?://doi\.org/[^}]*\},?\n", "", text, flags=re.IGNORECASE | re.MULTILINE)
        if not re.search(r"^\s*note\s*=", text, re.IGNORECASE | re.MULTILINE):
            text = re.sub(r",?\s*\n\}\s*$", f",\n  note={{doi: {doi.group(1)}}}\n}}", text)
    elif re.search(r"^\s*note\s*=\s*\{arXiv:", text, re.MULTILINE):
        text = re.sub(r"^\s*url\s*=\s*\{https?://arxiv\.org/[^}]*\},?\n", "", text, flags=re.MULTILINE)
    # a DOI printed through note is text mode: underscores must be escaped (10.1162/tacl_a_00437)
    text = re.sub(r"(note\s*=\s*\{doi: [^}]*\})", lambda m: re.sub(r"(?<!\\)_", r"\\_", m.group(1)), text)
    return re.sub(r",(\s*\n\})", r"\1", text)


def rekey_bibtex(raw: str, key: str) -> str:
    one = " ".join(raw.split())
    one = re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", one)
    one = re.sub(r"(title=\{)(.+?)(\},)", lambda m: f"{m.group(1)}{{{m.group(2)}}}{m.group(3)}", one, count=1)
    body = re.sub(r",\s+(?=[A-Za-z]+=)", ",\n  ", one)
    body = re.sub(r"^(@\w+\{[^,]+,)\s*", r"\1\n  ", body)
    return normalize_bibtex(re.sub(r"\s*\}\s*$", "\n}", body))


def bibtex_from_records(seed: Dict, key: str, recs: Dict[str, Record], doi: str) -> str:
    pick = recs.get("crossref") or recs.get("arxiv") or recs.get("openalex") or recs.get("semantic_scholar")
    authors = next((r.authors for r in (recs.get("crossref"), recs.get("openalex"), recs.get("arxiv"),
                                        recs.get("semantic_scholar")) if r and r.authors), seed.get("authors", []))
    title = pick.title if pick else seed["title"]
    year = seed["year"]
    ar = recs.get("arxiv")
    arxiv_id = (ar.extra.get("arxiv") if ar else None) or seed.get("arxiv")
    typ = seed.get("type", "misc")
    fields = [("author", " and ".join(a for a in authors if a)), ("title", "{" + _bib_escape(title) + "}")]
    if typ == "arxiv" or (typ == "misc" and arxiv_id):
        entry = "misc"
        fields += [("year", str(year)), ("eprint", str(arxiv_id)), ("archivePrefix", "arXiv")]
        if ar and ar.extra.get("primary_class"):
            fields.append(("primaryClass", ar.extra["primary_class"]))
        fields.append(("note", f"arXiv:{arxiv_id}"))
    elif typ == "inproceedings":
        entry = "inproceedings"
        fields += [("booktitle", _bib_escape(seed.get("venue", ""))), ("year", str(year))]
        if arxiv_id:
            fields.append(("note", f"arXiv:{arxiv_id}"))
    elif typ == "article":
        entry = "article"
        cr = recs.get("crossref")
        venue = cr.venue if cr and cr.venue else (seed.get("venue") or (pick.venue if pick else ""))
        fields += [("journal", _bib_escape(venue)), ("year", str(year))]
        bib = (recs["openalex"].extra.get("biblio") if recs.get("openalex") else None) or {}
        if bib.get("volume"):
            fields.append(("volume", str(bib["volume"])))
        if bib.get("issue"):
            fields.append(("number", str(bib["issue"])))
        if bib.get("first_page") and bib.get("last_page"):
            fields.append(("pages", f"{bib['first_page']}--{bib['last_page']}"))
    else:
        entry = typ
        fields += [("year", str(year))]
        for f in ("edition", "publisher", "address", "howpublished", "note"):
            if seed.get(f):
                fields.append((f, str(seed[f])))
    if doi and not is_arxiv_doi(doi):
        fields.append(("doi", doi))
        if not any(k == "note" for k, _ in fields):      # IEEEtran.bst ignores doi; print it via note
            fields.append(("note", "doi: " + doi.replace("_", r"\_")))
    elif arxiv_id:
        fields.append(("url", f"https://arxiv.org/abs/{arxiv_id}"))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{entry}{{{key},\n{body}\n}}"


def bibtex_manual(seed: Dict, key: str) -> str:
    typ = seed.get("type", "misc")
    fields = [("author", " and ".join(seed.get("authors", []))), ("title", "{" + _bib_escape(seed["title"]) + "}"),
              ("year", str(seed["year"]))]
    for f in ("edition", "publisher", "address", "howpublished", "note", "url"):
        if seed.get(f):
            fields.append((f, str(seed[f])))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{typ}{{{key},\n{body}\n}}"


def _bibtex_consistent(bibtex: str, seed: Dict) -> bool:
    y = re.search(r"year=\{(\d{4})\}", bibtex)
    t = re.search(r"title=\{\{(.+?)\}\},", bibtex)
    if not y or abs(int(y.group(1)) - int(seed["year"])) > YEAR_TOL:
        return False
    return bool(t) and title_sim(seed["title"], t.group(1)) >= 0.85


def strip_title_footnote(bibtex: str, seed_title: str) -> str:
    if re.search(r"\d$", seed_title.strip()):
        return bibtex
    return re.sub(r"(title\s*=\s*\{\{[^}]*?[A-Za-z])\d\}\}", r"\1}}", bibtex, count=1)


# ----------------------------------------------------------------------------- verification
def verify_entry(seed: Dict, agent: Agent, local: Local) -> Result:
    key = seed["key"]
    notes: List[str] = []
    if seed.get("verify", "auto") == "manual":
        return Result(key, "MANUAL", [], clean_doi(seed.get("doi")), 0.0,
                      ["book/software/web: BibTeX from seed, needs human check"], bibtex_manual(seed, key))
    recs: Dict[str, Record] = {}
    sims: Dict[str, float] = {}
    seed_doi = clean_doi(seed.get("doi"))
    arxiv_id = str(seed["arxiv"]) if seed.get("arxiv") else ""
    is_preprint = seed.get("type") == "arxiv"

    def accept(rec: Optional[Record], label: str) -> bool:
        if rec is None:
            return False
        m = best_match(seed, [rec], notes)
        if m:
            recs[rec.source], sims[rec.source] = m[0], m[1]
            return True
        notes.append(f"{label}: record does not match title/year ({rec.title[:60]!r}, {rec.year})")
        return False

    seed_doi_confirmed = False
    if seed_doi:
        seed_doi_confirmed |= accept(local.crossref_doi(seed_doi), "crossref DOI")
        seed_doi_confirmed |= accept(local.openalex_doi(seed_doi), "openalex DOI")
    if arxiv_id and "openalex" not in recs:
        accept(local.openalex_doi(f"10.48550/arxiv.{arxiv_id}"), "openalex arXiv-DOI")

    first_author = surname(seed["authors"][0]) if seed.get("authors") else ""
    q = f"{seed['title']} {first_author}".strip()
    for source in ("crossref", "openalex", "semantic_scholar", "scopus"):
        if source in recs or (source in ("semantic_scholar", "scopus") and len(recs) >= 2):
            continue
        if source == "scopus" and not agent.ok:
            continue
        try:
            if source == "scopus":
                cands = agent.search(source, 'TITLE("' + seed["title"].replace('"', " ") + '")', limit=25, sort="cited_by")
            else:
                cands = agent.search(source, q) if agent.ok else None
        except RateLimited as exc:
            notes.append(f"{source}: rate-limited ({exc})")
            continue
        m = best_match(seed, cands, notes) if cands else None
        if m is None and source in ("crossref", "openalex"):
            cands = local.search(source, q, title=seed["title"], author=first_author)
            if cands is None:
                notes.append(f"{source}: unavailable")
                continue
            m = best_match(seed, cands, notes)
        elif m is None and cands is None and source == "semantic_scholar":
            cands = local.search(source, q)
            if cands is None:
                notes.append(f"{source}: unavailable")
                continue
            m = best_match(seed, cands, notes)
        if m:
            recs[source], sims[source] = m[0], m[1]

    if arxiv_id and (is_preprint or len(recs) < 2):
        rec = local.arxiv(arxiv_id)
        if rec is None:
            notes.append("arxiv: unavailable")
        else:
            accept(rec, "arxiv")

    doi = ""
    if not is_preprint:
        found = {r.doi for r in recs.values() if r.doi and not is_arxiv_doi(r.doi)}
        if seed_doi and seed_doi_confirmed:
            doi = seed_doi
            if found - {seed_doi}:
                notes.append(f"other DOIs seen: {sorted(found - {seed_doi})}")
        else:
            if seed_doi:
                notes.append(f"SEED DOI WRONG: {seed_doi} does not resolve to this work")
            if found:
                doi = recs["crossref"].doi if recs.get("crossref") and recs["crossref"].doi else sorted(found)[0]
                if "crossref" not in recs:
                    accept(local.crossref_doi(doi), "crossref DOI")
                if "openalex" not in recs:
                    accept(local.openalex_doi(doi), "openalex DOI")
                if doi not in {r.doi for s, r in recs.items() if s in ("crossref", "openalex")}:
                    notes.append(f"DOI {doi} not confirmed by Crossref/OpenAlex; dropped")
                    doi = ""
                if doi:
                    notes.append(f"DOI adopted from sources: {doi}")
    for s, r in recs.items():
        if r.year is not None and int(r.year) != int(seed["year"]):
            notes.append(f"{s}: year {r.year} (seed {seed['year']})")

    n = len(recs)
    status = "VERIFIED" if n >= 2 else ("SINGLE_SOURCE" if n == 1 else "NOT_FOUND")
    bibtex = ""
    if status != "NOT_FOUND":
        if doi:
            raw = local.bibtex_from_doi(doi)
            bibtex = rekey_bibtex(raw, key) if raw else ""
            if bibtex and not _bibtex_consistent(bibtex, seed):
                notes.append("doi.org BibTeX disagrees with seed title/year; built from metadata instead")
                bibtex = ""
            elif not bibtex:
                notes.append("doi.org BibTeX unavailable; built from metadata")
        if not bibtex:
            bibtex = bibtex_from_records(seed, key, recs, doi)
    return Result(key, status, sorted(recs), doi, max(sims.values()) if sims else 0.0, notes, bibtex)


def load_previous(bib_path: Path, report_path: Path) -> Dict[str, Result]:
    if not (bib_path.exists() and report_path.exists()):
        return {}
    bibs: Dict[str, str] = {}
    for m in re.finditer(r"% status: \w+; sources: [^\n]*\n(@\w+\{([^,\s]+),.*?\n\})\n",
                         bib_path.read_text(encoding="utf-8"), re.DOTALL):
        bibs[m.group(2)] = normalize_bibtex(m.group(1))
    prev: Dict[str, Result] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] in ("key", "Status") or cells[1] not in ("VERIFIED", "SINGLE_SOURCE",
                                                                              "NOT_FOUND", "MANUAL"):
            continue
        key, status, sources, doi, sim, notes = cells
        prev[key] = Result(key, status, [] if sources == "-" else sources.split(", "), "" if doi == "-" else doi,
                           float(sim or 0), [n for n in notes.replace("\\|", "|").split("; ") if n], bibs.get(key, ""))
    return prev


def write_outputs(results: List[Result], bib_path: Path, report_path: Path, seed_name: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [f"% Generated by paper-agent verify-refs on {stamp} from {seed_name}.",
             "% Do not edit by hand. Only VERIFIED/SINGLE_SOURCE/MANUAL entries are emitted;",
             "% `paper-agent check-citations` rejects \\cite of anything that is not VERIFIED or MANUAL.", ""]
    for r in results:
        if r.bibtex:
            lines += [f"% status: {r.status}; sources: {', '.join(r.sources) or '-'}", r.bibtex, ""]
    bib_path.write_text("\n".join(lines), encoding="utf-8")
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    rep = ["# Reference verification report", "", f"Generated {stamp} by `paper-agent verify-refs` from `{seed_name}`.", "",
           "Rule: VERIFIED = title (sim ≥ 0.90) and year (±1) matched in ≥ 2 independent sources; "
           "SINGLE_SOURCE = 1 source; NOT_FOUND = 0; MANUAL = book/software/web from seed (human check).", "",
           "| Status | n |", "|---|---:|"]
    rep += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    rep += ["", "| key | status | sources | DOI | sim | notes |", "|---|---|---|---|---:|---|"]
    for r in results:
        rep.append(f"| {r.key} | {r.status} | {', '.join(r.sources) or '-'} | {r.doi or '-'} | {r.sim:.2f} | "
                   f"{'; '.join(r.notes).replace('|', chr(92) + '|')} |")
    report_path.write_text("\n".join(rep) + "\n", encoding="utf-8")


def verify_references(seed_path: Path, bib_path: Path, report_path: Path, only: Optional[str] = None,
                      use_agent: bool = True, reuse_verified: bool = False, log=print) -> dict:
    seed_path, bib_path, report_path = Path(seed_path), Path(bib_path), Path(report_path)
    seeds = yaml.safe_load(seed_path.read_text(encoding="utf-8"))["entries"]
    keys = [s["key"] for s in seeds]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in seed: {sorted({k for k in keys if keys.count(k) > 1})}")
    if only:
        seeds = [s for s in seeds if s["key"] == only]
        if not seeds:
            raise ValueError(f"key not found: {only}")
    prev = load_previous(bib_path, report_path) if reuse_verified else {}
    agent, local = Agent(enabled=use_agent), Local()
    log(f"browser-agent: {'OK ' + agent.url if agent.ok else 'unavailable → direct APIs'}")
    results: List[Result] = []
    for i, s in enumerate(seeds, 1):
        old = prev.get(s["key"])
        if old and old.status == "VERIFIED" and old.bibtex:
            old.bibtex = strip_title_footnote(old.bibtex, s["title"])
            results.append(old)
            log(f"[{i}/{len(seeds)}] {s['key']} … {old.status} (reused)")
            continue
        try:
            r = verify_entry(s, agent, local)
        except Exception as exc:  # noqa: BLE001 - keep going; the report shows the failure
            r = Result(s["key"], "NOT_FOUND", [], clean_doi(s.get("doi")), 0.0, [f"error: {exc!r}"], "")
        r.bibtex = strip_title_footnote(r.bibtex, s["title"])
        results.append(r)
        log(f"[{i}/{len(seeds)}] {s['key']} … {r.status} [{', '.join(r.sources) or '-'}]"
            + (f"  {'; '.join(r.notes)}" if r.notes else ""))
    write_outputs(results, bib_path, report_path, seed_path.name)
    summary = {st: sum(1 for r in results if r.status == st) for st in ("VERIFIED", "SINGLE_SOURCE", "NOT_FOUND", "MANUAL")}
    log(f"{summary} → {bib_path}, {report_path}")
    return {"summary": summary, "results": [{"key": r.key, "status": r.status, "sources": r.sources, "doi": r.doi,
                                              "notes": r.notes} for r in results],
            "bib": str(bib_path), "report": str(report_path)}
