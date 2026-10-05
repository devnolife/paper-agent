"""Citation gate: every ``\\cite`` key must exist in the .bib and be VERIFIED or MANUAL in the verification report."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
BIBKEY_RE = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
ROW_RE = re.compile(r"^\|\s*([A-Za-z0-9_:\-]+)\s*\|\s*([A-Z_]+)\s*\|")


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def collect_cites(tex_path: Path, seen: Optional[Set[Path]] = None) -> Dict[str, List[str]]:
    seen = seen if seen is not None else set()
    tex_path = Path(tex_path).resolve()
    if tex_path in seen or not tex_path.exists():
        return {}
    seen.add(tex_path)
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    cites: Dict[str, List[str]] = {}
    for m in CITE_RE.finditer(text):
        for key in (k.strip() for k in m.group(1).split(",")):
            if key:
                cites.setdefault(key, []).append(tex_path.name)
    for m in INPUT_RE.finditer(text):
        child = tex_path.parent / m.group(1)
        if child.suffix != ".tex":
            child = child.with_suffix(".tex")
        for k, v in collect_cites(child, seen).items():
            cites.setdefault(k, []).extend(v)
    return cites


def parse_report(report: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for line in Path(report).read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m and m.group(1) not in ("key", "Status"):
            statuses[m.group(1)] = m.group(2)
    return statuses


def check_citations(tex: Path, bib: Path, report: Optional[Path] = None, allow_single: bool = False) -> dict:
    cites = collect_cites(Path(tex))
    bib_keys = {m.group(2) for m in BIBKEY_RE.finditer(Path(bib).read_text(encoding="utf-8"))}
    statuses = parse_report(report) if report and Path(report).exists() else {}
    accepted = {"VERIFIED", "MANUAL"} | ({"SINGLE_SOURCE"} if allow_single else set())
    problems: List[Tuple[str, str]] = []
    for key in sorted(cites):
        if key not in bib_keys:
            problems.append((key, "missing from .bib"))
        elif statuses and key not in statuses:
            problems.append((key, "no verification status in report"))
        elif statuses and statuses[key] not in accepted:
            problems.append((key, f"status {statuses[key]}"))
    return {"ok": not problems, "cited": len(cites), "bib_keys": len(bib_keys),
            "unused": sorted(bib_keys - set(cites)),
            "manual": sorted(k for k in cites if statuses.get(k) == "MANUAL"),
            "problems": [{"key": k, "why": w, "in": sorted(set(cites[k]))} for k, w in problems],
            "report_used": bool(statuses)}


def format_result(res: dict) -> str:
    out = [f"cited keys: {res['cited']}  bib keys: {res['bib_keys']}  unused bib keys: {len(res['unused'])}"]
    if res["manual"]:
        out.append("MANUAL (human-check) keys cited: " + ", ".join(res["manual"]))
    if not res["report_used"]:
        out.append("(no verification report: only .bib membership was checked)")
    if res["problems"]:
        out.append("VIOLATIONS:")
        out += [f"  {p['key']}: {p['why']}  (cited in {', '.join(p['in'])})" for p in res["problems"]]
    else:
        out.append("OK: all citations are in the .bib" + (" and VERIFIED/MANUAL." if res["report_used"] else "."))
    return "\n".join(out)
