"""Second-source audit of the references the benchmark scored as not found -> ``refs_audit.json`` (aggregates only).

The references scorer checks Crossref only: a listed work counts as not found when Crossref has no record matching
it by DOI, or by title (similarity >= 0.90), year (±1) and first author. A real work can fail that test (its DOI may
be registered with another agency, it may have no DOI, Crossref's title search may rank it out), so this audit looks
every not-found reference up again, cheapest source first, and stops at the first record that passes the scorer's
own matching rule (``references.best_match``):

  1. doi.org, for a given DOI that Crossref does not know (DataCite, JaLC, ...; free);
  2. Scopus, 3. OpenAlex: title search through the studio-revisi browser-agent
     (``BROWSER_AGENT_URL``/``BROWSER_AGENT_API_KEY``), which holds the API keys and queues the shared rate limits.

Semantic Scholar is not used: during the audit its API answered 429 to every request from our server, with or without
a key, and a source that answers for only some references would make the classes depend on timing.

Each not-found reference ends in one class:

  exact      passes the scorer's rule in another source: a real work the Crossref-only score missed;
  title      title similarity >= 0.90 but the year or first author differ: a real work cited with wrong metadata;
  none       every source answered and none matched;
  unchecked  nothing matched but a source could not be asked (asked again on the next run).

OpenAlex bills each search against a daily budget that other services may share, so at most
``AUDIT_OPENALEX_DAILY_CAP`` searches (default 600) are made per UTC day. Answers are cached under
``PAPER_AGENT_CACHE_DIR``, so reruns only query what is new. ``corrections`` lists, without titles, the credit the
scorer would give each ``exact`` reference; ``analyze.py`` uses it for the sensitivity of the ranking to Crossref.

  BENCH_RUN_DIR=<run dir> python experiments/bench/audit_refs.py [--limit N]
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_agent.bench.runner import latest_records  # noqa: E402
from paper_agent.bench.score import answer_hash, load_scores  # noqa: E402
from paper_agent.bench.tasks import parse_json_loose  # noqa: E402
from paper_agent.bench.tasks.references import credit, seeds_from_answer  # noqa: E402
from paper_agent.config import cache_dir, load_env  # noqa: E402
from paper_agent.references import (UA, Agent, RateLimited, Record, best_match, clean_doi, norm,  # noqa: E402
                                    title_sim)

OUT = ROOT / "experiments" / "bench" / "refs_audit.json"
THRESHOLD = 0.90
LIMIT = 5
SOURCES = ("scopus", "openalex")
PACE = {"doi.org": 0.2, "scopus": 0.5, "openalex": 0.2}    # seconds between calls
CLASSES = ("exact", "title", "none", "unchecked")


class Lookup:
    """Cached, paced lookups; a source that keeps answering 429 is paused for the rest of the run."""

    def __init__(self, openalex_cap: int, log=print) -> None:
        self.agent = Agent()
        self.dir = cache_dir() / "bench-refs-audit"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap, self.log = openalex_cap, log
        self.headers = {"User-Agent": UA.format(mail=os.getenv("CROSSREF_EMAIL") or "unknown")}
        self.paused: Dict[str, str] = {}
        self.calls: Counter = Counter()
        self._last: Dict[str, float] = {}

    def _path(self, *key) -> Path:
        return self.dir / f"{hashlib.sha1(json.dumps(key).encode()).hexdigest()}.json"

    def _pace(self, source: str) -> None:
        wait = self._last.get(source, 0.0) + PACE[source] - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last[source] = time.time()

    def _budget(self) -> Path:
        return self.dir / f"openalex-searches-{datetime.now(timezone.utc):%Y-%m-%d}.txt"

    def openalex_used(self) -> int:
        p = self._budget()
        return int(p.read_text() or 0) if p.exists() else 0

    def _pause(self, source: str, why: str) -> None:
        self.paused[source] = why
        self.log(f"  {source} paused for this run: {why}")

    def search(self, source: str, title: str) -> Optional[List[Record]]:
        """Title search in one source; None when it could not be asked."""
        q = norm(title)              # words only: no quotes, hyphens or upper-case boolean operators to parse
        if not q:
            return None
        if source == "scopus":
            q = f'TITLE("{q}")'
        path = self._path(source, q, LIMIT)
        if path.exists():
            return [Record(**r) for r in json.loads(path.read_text(encoding="utf-8"))]
        if source in self.paused:
            return None
        if not self.agent.ok:
            self._pause(source, "browser-agent not configured or not healthy")
            return None
        if source == "openalex" and self.openalex_used() >= self.cap:
            self._pause(source, f"daily cap of {self.cap} OpenAlex searches reached (resets 00:00 UTC)")
            return None
        recs = None
        for attempt in range(2):
            self._pace(source)
            if source == "openalex":
                self._budget().write_text(str(self.openalex_used() + 1))
            self.calls[source] += 1
            try:
                recs = self.agent.search(source, q, limit=LIMIT)
                break
            except RateLimited as exc:
                if attempt:
                    self._pause(source, f"rate limited ({exc})")
                    return None
                time.sleep(65)
        if recs is None:
            return None
        path.write_text(json.dumps([asdict(r) for r in recs]), encoding="utf-8")
        return recs

    def doi(self, doi: str) -> Union[Record, bool, None]:
        """A DOI registered with any agency: its record, True (no metadata), False (unregistered), None (no answer)."""
        path = self._path("doi.org", doi)
        if path.exists():
            c = json.loads(path.read_text(encoding="utf-8"))
            return Record(**c["record"]) if c.get("record") else c["registered"]
        self._pace("doi.org")
        self.calls["doi.org"] += 1
        try:
            r = requests.get(f"https://doi.org/api/handles/{quote(doi, safe='/')}", headers=self.headers, timeout=20)
        except requests.RequestException:
            return None
        if r.status_code in (400, 404):
            out: dict = {"registered": False}
        elif not r.ok:
            return None
        else:
            try:
                registered = r.json().get("responseCode") == 1
            except ValueError:
                return None
            out = {"registered": registered}
            if registered:
                rec = self._csl(doi)
                if rec is None:
                    return None
                out["record"] = rec or None
        path.write_text(json.dumps(out), encoding="utf-8")
        return Record(**out["record"]) if out.get("record") else out["registered"]

    def _csl(self, doi: str) -> Optional[dict]:
        """CSL-JSON metadata through doi.org content negotiation: a Record dict, {} if the agency has none, None on error."""
        self._pace("doi.org")
        self.calls["doi.org"] += 1
        try:
            r = requests.get(f"https://doi.org/{quote(doi, safe='/')}", timeout=30,
                             headers={**self.headers, "Accept": "application/vnd.citationstyles.csl+json"})
        except requests.RequestException:
            return None
        if r.status_code >= 500 or r.status_code == 429:
            return None
        try:
            m = r.json() if r.ok else {}
        except ValueError:
            m = {}
        if not isinstance(m, dict) or not m.get("title"):
            return {}

        def first(v):
            return (v[0] if v else "") if isinstance(v, list) else (v or "")
        parts = (m.get("issued") or {}).get("date-parts") or [[None]]
        year = parts[0][0] if parts and parts[0] else None
        authors = [" ".join(filter(None, [a.get("given"), a.get("family") or a.get("literal")]))
                   for a in m.get("author") or [] if isinstance(a, dict)]
        return asdict(Record("doi.org", first(m.get("title")), int(year) if str(year or "").isdigit() else None,
                             clean_doi(doi), authors, first(m.get("container-title"))))


def not_found_refs(scores: dict, recs: dict) -> Tuple[List[dict], int]:
    """Every not-found reference with the seed the scorer verified (title, authors, year, DOI)."""
    todo, partial = [], 0
    for model, rows in scores.items():
        for (iid, rep), row in rows.items():
            if row.get("task") != "references" or not row.get("refs"):
                continue
            rec = recs.get(model, {}).get((iid, rep))
            seeds: List[dict] = []
            if rec is not None and answer_hash(rec) == row.get("answer_hash"):
                seeds = seeds_from_answer(parse_json_loose(rec.get("text", "")), 10 ** 6)
            for k, ref in enumerate(row["refs"]):
                if ref.get("exists") is not False:
                    continue
                seed = seeds[k] if k < len(seeds) and seeds[k]["title"][:200] == ref["title"] else None
                if seed is None:         # answer not available: title, year and DOI from the score row
                    partial += 1
                    seed = {"key": f"r{k + 1}", "type": "article", "title": ref["title"], "authors": [],
                            "year": ref.get("year"), "doi": ref.get("doi_given") or ""}
                todo.append({"model": model, "item": iid, "rep": rep, "k": k, "seed": seed, "ref": ref})
    return todo, partial


def classify(seed: dict, ref: dict, look: Lookup) -> Tuple[str, Optional[str], str]:
    """(class, source of the matching record, DOI of that record)."""
    if not seed.get("title") or seed.get("year") is None:
        return "none", None, ""
    given = clean_doi(seed.get("doi"))
    cls, unchecked = "none", False
    if given and ref.get("doi_resolves") is False:
        got = look.doi(given)
        if got is None:
            unchecked = True
        elif isinstance(got, Record):
            if best_match(seed, [got], []):
                return "exact", "doi.org", given
            if title_sim(seed["title"], got.title) >= THRESHOLD:
                cls = "title"
    for source in SOURCES:
        recs = look.search(source, seed["title"])
        if recs is None:
            unchecked = True
            continue
        m = best_match(seed, recs, [])
        if m:
            return "exact", source, clean_doi(m[0].doi)
        if any(title_sim(seed["title"], r.title) >= THRESHOLD for r in recs):
            cls = "title"
    return ("unchecked" if unchecked and cls == "none" else cls), None, ""


def rates(c: dict) -> dict:
    ret, nf = c.get("returned", 0), c.get("not_found", 0)
    out = {k: c.get(k, 0) for k in ("returned", "not_found", *CLASSES)}
    out["via"] = {k[4:]: v for k, v in sorted(c.items()) if k.startswith("via_")}
    out["not_found_rate"] = round(nf / ret, 4) if ret else None                    # the scorer's (Crossref only)
    out["no_match_rate"] = round((c.get("none", 0) + c.get("unchecked", 0)) / ret, 4) if ret else None
    out["share_of_not_found"] = {k: round(c.get(k, 0) / nf, 4) if nf else None for k in CLASSES}
    return out


def audit(scores: dict, todo: List[dict], look: Lookup, log=print) -> dict:
    per_model: Dict[str, Counter] = defaultdict(Counter)
    for model, rows in scores.items():
        for row in rows.values():
            if row.get("task") == "references":
                per_model[model]["returned"] += len(row.get("refs") or [])
                per_model[model]["not_found"] += sum(1 for r in row.get("refs") or [] if r.get("exists") is False)
    corrections = []
    for n, t in enumerate(todo, 1):
        cls, source, doi = classify(t["seed"], t["ref"], look)
        c = per_model[t["model"]]
        c[cls] += 1
        if source:
            c[f"via_{source}"] += 1
        if cls == "exact":
            given = clean_doi(t["seed"].get("doi"))
            corrections.append({"model": t["model"], "item": t["item"], "rep": t["rep"], "ref": t["k"],
                                "source": source, "credit": credit(True, bool(given), bool(given) and given == doi)})
        if n % 50 == 0 or n == len(todo):
            tot = Counter()
            for v in per_model.values():
                tot.update({k: v[k] for k in CLASSES})
            log(f"  {n}/{len(todo)} {dict(tot)} calls {dict(look.calls)}")
    out = {"threshold": THRESHOLD, "sources": ["doi.org", *SOURCES], "openalex_daily_cap": look.cap,
           "models": {m: rates(per_model[m]) for m in sorted(per_model)}}
    for group, pref in (("api", "copilot:"), ("local", "ollama:")):
        tot = Counter()
        for m, c in per_model.items():
            if m.startswith(pref):
                tot.update(c)
        out[group] = rates(tot)
    out["unchecked"] = sum(per_model[m]["unchecked"] for m in per_model)
    out["complete"] = out["unchecked"] == 0
    out["corrections"] = corrections
    out["last_run"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "calls": dict(look.calls),
                       "paused": look.paused, "openalex_searches_today": look.openalex_used()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="check only the first N not-found references (smoke test)")
    args = ap.parse_args()
    load_env(ROOT)
    run_dir = Path(os.environ["BENCH_RUN_DIR"]).expanduser()
    lock = (cache_dir() / "bench-refs-audit.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("refs audit already running; skipped")
        return
    scores = load_scores(run_dir)
    todo, partial = not_found_refs(scores, latest_records(run_dir))
    print(f"{len(todo)} not-found references ({partial} without their parsed answer: title, year and DOI only)")
    look = Lookup(int(os.getenv("AUDIT_OPENALEX_DAILY_CAP", "600")))
    res = audit(scores, todo[:args.limit] if args.limit else todo, look)
    if args.limit:
        print(json.dumps({k: res[k] for k in ("api", "local", "last_run")}, indent=1))
        return
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"refs audit -> {OUT} (unchecked: {res['unchecked']})")


if __name__ == "__main__":
    main()
