"""references — ask for n real references (with DOI) on a research topic; verify each against bibliographic databases.

Verification reuses :func:`paper_agent.references.verify_entry` (Crossref, OpenAlex, Semantic Scholar; title
similarity ≥ 0.90, year ± 1, first-author check): VERIFIED = found in ≥ 2 sources, SINGLE_SOURCE = 1, NOT_FOUND = 0
(hallucinated). DOI correctness: the DOI the model gave equals the DOI the sources confirm for that work.

Credit per reference (at most n counted, duplicates ignored): 1.0 verified + correct DOI, 0.75 verified without DOI,
0.5 verified with a wrong DOI or found in one source only, 0.25 single source with a wrong DOI, 0 not found.
Score = total credit / n requested — abstaining costs the same as a fabricated reference, but fabrication is
reported separately as ``hallucination_rate`` (not found / returned).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

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


def credit(status: str, doi_given: bool, doi_ok: bool) -> float:
    if status == "VERIFIED":
        return 1.0 if doi_ok else (0.75 if not doi_given else 0.5)
    if status == "SINGLE_SOURCE":
        return 0.25 if doi_given and not doi_ok else 0.5
    return 0.0


class ReferencesTask(Task):
    name = "references"
    max_tokens = 1500
    needs_network = True

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        return SYSTEM, USER.format(topic=item.input["topic"], n=item.input.get("n", 5))

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        from ...references import Agent, Local, Result, clean_doi, verify_entry

        n = int(item.input.get("n", 5))
        obj = parse_json_loose(record.get("text", ""))
        seeds = seeds_from_answer(obj, n)
        if "references:local" not in ctx.cache:
            ctx.cache["references:local"] = Local()
            ctx.cache["references:agent"] = Agent(enabled=False)
        local, agent = ctx.cache["references:local"], ctx.cache["references:agent"]
        rows, total = [], 0.0
        for s in seeds:
            if not s["title"] or s["year"] is None:
                res = Result(s["key"], "NOT_FOUND", [], "", 0.0, ["missing title or year"], "")
            else:
                try:
                    res = verify_entry(s, agent, local)
                except Exception as exc:  # noqa: BLE001 - a network failure must not crash scoring
                    return {"score": None, "error": f"verification failed: {exc!r}"[:300]}
            given = clean_doi(s["doi"])
            ok = bool(given) and given == clean_doi(res.doi)
            c = credit(res.status, bool(given), ok)
            total += c
            rows.append({"title": s["title"][:200], "year": s["year"], "status": res.status, "doi_given": given or None,
                         "doi_confirmed": res.doi or None, "doi_ok": ok, "credit": c})
        returned = len(rows)
        not_found = sum(1 for r in rows if r["status"] == "NOT_FOUND")
        return {"score": round(total / n, 4), "parsed": obj is not None, "returned": returned,
                "verified": sum(1 for r in rows if r["status"] == "VERIFIED"), "not_found": not_found,
                "hallucination_rate": round(not_found / returned, 4) if returned else None,
                "doi_correct": sum(1 for r in rows if r["doi_ok"]),
                "doi_wrong": sum(1 for r in rows if r["doi_given"] and not r["doi_ok"] and r["status"] != "NOT_FOUND"),
                "refs": rows}
