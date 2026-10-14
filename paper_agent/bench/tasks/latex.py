"""latex — fix a broken LaTeX document, typeset a table with booktabs, or convert a reference to BibTeX.

``fix``     the returned document is compiled with pdflatex in the sandbox; score = 0.5 x compiles + 0.5 x share of
            the required text snippets present in the PDF text (content kept).
``table``   the returned ``table`` environment is compiled inside a minimal document; cells parsed from the
            ``tabular`` are compared position-wise with the source table; score = 0.3 x compiles + 0.5 x cell
            accuracy + 0.1 x booktabs rules + 0.1 x caption and label.
``bibtex``  the returned entry is parsed; score = share of correct entry type, key and expected fields (authors'
            family names in order, title, year, venue, volume, number, pages, DOI).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..schema import Item
from . import Context, Task, code_block

SYSTEM = ("Kamu ahli LaTeX dan BibTeX yang membantu penulisan skripsi dan artikel ilmiah. Ikuti instruksi dengan "
          "tepat dan jawab hanya dengan blok kode yang diminta.")
TABLE_DOC = ("\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{booktabs}\n"
             "\\usepackage{amsmath}\n\\usepackage{multirow}\n\\usepackage{array}\n\\begin{document}\n%s\n\\end{document}\n")
COMPILE = ("pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex > /dev/null 2>&1; rc=$?; "
           "[ -f main.pdf ] && pdftotext -layout main.pdf main.txt; exit $rc")


def compile_tex(source: str, ctx: Context) -> dict:
    from .. import sandbox
    import hashlib
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    cache = ctx.cache.setdefault("latex:compile", {})
    if key in cache:
        return cache[key]
    r = sandbox.run(["sh", "-c", COMPILE], {"main.tex": source.encode("utf-8")}, timeout=60, mem_mb=2048,
                    collect=["main.txt", "main.log"])
    log = r.files.get("main.log", b"").decode("utf-8", "replace")
    errors = [ln for ln in log.splitlines() if ln.startswith("! ")][:5]
    res = {"ok": r.returncode == 0 and "main.txt" in r.files, "returncode": r.returncode, "timed_out": r.timed_out,
           "errors": errors, "text": r.files.get("main.txt", b"").decode("utf-8", "replace")}
    cache[key] = res
    return res


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


# ----------------------------------------------------------------------------- tables
def _strip_cmd(s: str, cmd: str) -> str:
    return re.sub(r"\\" + cmd + r"\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)


def clean_cell(s: str) -> str:
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
    s = re.sub(r"\\multirow\{\d+\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
    for cmd in ("textbf", "textit", "emph", "text", "mathrm", "textsuperscript", "textsubscript", "makecell"):
        s = _strip_cmd(s, cmd)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#").replace("\\$", "$")
    s = s.replace("\\textendash", "–").replace("--", "–").replace("~", " ").replace("\\,", " ")
    s = re.sub(r"\$\s*\^\s*\{?2\}?\s*\$|\^\{?2\}?", "²", s)
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    return _flat(s)


def norm_expected_cell(s: str) -> str:
    return _flat(str(s).replace("--", "–").replace("^2", "²"))


def _skip_group(s: str, i: int) -> int:
    """Index after the brace group starting at (or after whitespace from) ``i``; ``i`` if there is none."""
    j = i
    while j < len(s) and s[j] in " \t\n":
        j += 1
    if j >= len(s) or s[j] != "{":
        return i
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return len(s)


def tabular_cells(src: str) -> Optional[List[List[str]]]:
    m = re.search(r"\\begin\{(tabular\*?|tabularx)\}", src or "")
    if not m:
        return None
    env = m.group(1)
    i = m.end()
    if env != "tabular":
        i = _skip_group(src, i)               # width
    i = _skip_group(src, i)                   # column specification
    end = src.find("\\end{" + env + "}", i)
    if end == -1:
        return None
    body = re.sub(r"(?<!\\)%.*", "", src[i:end])
    body = re.sub(r"\\(toprule|midrule|bottomrule|hline|cmidrule(\([^)]*\))?\{[^}]*\}|cline\{[^}]*\})(\[[^\]]*\])?",
                  " ", body)
    rows = []
    for raw in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        if not raw.strip():
            continue
        rows.append([clean_cell(c) for c in re.split(r"(?<!\\)&", raw)])
    return rows


def cell_accuracy(got: Optional[List[List[str]]], exp: List[List[str]]) -> float:
    total = sum(len(r) for r in exp)
    if not got or not total:
        return 0.0
    ok = sum(1 for i, row in enumerate(exp) for j, cell in enumerate(row)
             if i < len(got) and j < len(got[i]) and got[i][j] == norm_expected_cell(cell))
    extra = max(0, sum(len(r) for r in got) - total)
    return max(0.0, (ok - extra) / total)


# ----------------------------------------------------------------------------- bibtex
def parse_bibtex(text: str) -> Optional[dict]:
    m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text or "")
    if not m:
        return None
    entry = {"entry_type": m.group(1).lower(), "key": m.group(2), "fields": {}}
    i = m.end()
    s = text
    while i < len(s):
        fm = re.compile(r"\s*([A-Za-z_-]+)\s*=\s*").match(s, i)
        if not fm:
            break
        name, i = fm.group(1).lower(), fm.end()
        if i < len(s) and s[i] == "{":
            depth, j = 0, i
            while j < len(s):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            val, i = s[i + 1:j], j + 1
        elif i < len(s) and s[i] == '"':
            j = s.find('"', i + 1)
            val, i = s[i + 1:j], j + 1
        else:
            vm = re.compile(r"[^,}\s]+").match(s, i)
            if not vm:
                break
            val, i = vm.group(0), vm.end()
        entry["fields"][name] = re.sub(r"\s+", " ", val).strip()
        cm = re.compile(r"\s*,").match(s, i)
        if not cm:
            break
        i = cm.end()
    return entry


def family_names(author_field: str) -> List[str]:
    out = []
    for a in re.split(r"\s+and\s+", author_field or ""):
        a = a.strip().strip("{}")
        if not a:
            continue
        fam = a.split(",")[0] if "," in a else a.split()[-1]
        out.append(_plain(fam))
    return out


def _plain(s: str) -> str:
    s = re.sub(r"\\[`'^\"~=.uvHcdbtr]\s*\{?([A-Za-z])\}?", r"\1", s or "")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
             "ninth": 9, "tenth": 10, "pertama": 1, "kedua": 2, "ketiga": 3, "keempat": 4, "kelima": 5, "keenam": 6}


def edition_number(s: str) -> Optional[int]:
    m = re.search(r"\d+", s or "")
    if m:
        return int(m.group(0))
    low = (s or "").lower()
    return next((v for k, v in _ORDINALS.items() if k in low), None)


def bib_field_ok(name: str, got: Optional[str], exp) -> bool:
    if got is None:
        return False
    if name == "edition":
        return edition_number(got) is not None and edition_number(got) == edition_number(str(exp))
    if name == "pages":
        return re.sub(r"[-–—]+", "-", got.replace(" ", "")) == re.sub(r"[-–—]+", "-", str(exp).replace(" ", ""))
    if name == "doi":
        return re.sub(r"^https?://(dx\.)?doi\.org/", "", got.strip().lower()) == str(exp).strip().lower()
    return _plain(got) == _plain(str(exp))


class LatexTask(Task):
    name = "latex"
    json_mode = False
    max_tokens = 3000

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        i = item.input
        return SYSTEM, f"{i['instruction'].strip()}\n\n{i['source']}"

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        kind, ref, text = item.input["kind"], item.reference, record.get("text", "")
        if kind == "bibtex":
            block = code_block(text, ("bibtex", "bib", "latex", "tex")) or text
            e = parse_bibtex(block)
            if e is None:
                return {"score": 0.0, "parsed": False}
            checks: Dict[str, bool] = {"entry_type": e["entry_type"] == ref["entry_type"], "key": e["key"] == ref["key"]}
            for name, exp in ref["fields"].items():
                if name == "author_family":
                    checks[name] = family_names(e["fields"].get("author", "")) == [_plain(x) for x in exp]
                else:
                    checks[name] = bib_field_ok(name, e["fields"].get(name), exp)
            return {"score": round(sum(checks.values()) / len(checks), 4), "parsed": True, "checks": checks}
        block = code_block(text, ("latex", "tex"))
        if block is None and "\\begin" in text:
            block = text
        if not block:
            return {"score": 0.0, "parsed": False}
        if kind == "fix":
            res = compile_tex(block, ctx)
            raw = res["text"]
            # words hyphenated at line ends: try them joined ("pembela-\njaran") and as real compounds ("sosial-ekonomi")
            pdfs = {_flat(raw), _flat(re.sub(r"-\s*\n\s*", "", raw)), _flat(re.sub(r"-\s*\n\s*", "-", raw))}
            found = [s for s in ref["must_contain"] if any(_flat(s) in t for t in pdfs)]
            content = len(found) / len(ref["must_contain"]) if ref["must_contain"] else 1.0
            return {"score": round(0.5 * res["ok"] + 0.5 * (content if res["ok"] else 0.0), 4), "compiled": res["ok"],
                    "content": round(content, 4), "errors": res["errors"]}
        if kind == "table":
            env = block
            if "\\documentclass" in env:
                m = re.search(r"\\begin\{table\}.*?\\end\{table\}", env, re.DOTALL)
                env = m.group(0) if m else env
            res = compile_tex(TABLE_DOC % env, ctx)
            cells = tabular_cells(env)
            acc = cell_accuracy(cells, ref["cells"])
            rules = all(r in env for r in ("\\toprule", "\\midrule", "\\bottomrule"))
            cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", env)
            lab = re.search(r"\\label\{([^}]*)\}", env)
            cap_ok = bool(cap) and _flat(cap.group(1)) == _flat(ref.get("caption", "")) and bool(lab) and \
                lab.group(1).strip() == ref.get("label", "")
            score = 0.3 * res["ok"] + 0.5 * acc + 0.1 * rules + 0.1 * cap_ok
            return {"score": round(score, 4), "compiled": res["ok"], "cell_accuracy": round(acc, 4), "booktabs": rules,
                    "caption_label": cap_ok, "errors": res["errors"]}
        raise ValueError(f"unknown latex kind {kind}")
