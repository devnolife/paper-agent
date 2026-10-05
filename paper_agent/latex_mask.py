"""Structure-safe masking of LaTeX for translation.

Each source line becomes translatable *units* (paragraph, ``\\section``/``\\caption`` argument, table
cell, ``\\item`` body). Inside a unit, commands, math, ``\\cite``/``\\ref``/``\\label``, identifiers
(``\\texttt``, ``\\textsc``), ``~``, ``\\%``, ``{,}`` ... are replaced by placeholders ``⟦n⟧`` and the
emphasis commands ``\\emph``/``\\textbf``/``\\textit`` by inline tags ``<k>…</k>``. ``unmask`` reverses
the process; ``integrity_ok`` checks that a translation kept every placeholder/tag exactly once and
kept placeholders glued to numbers (``58⟦3⟧`` = ``58\\%``) in place. Masking then unmasking without
translation reproduces the source byte for byte (see tests).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PH_OPEN, PH_CLOSE = "\u27e6", "\u27e7"          # ⟦ ⟧
PH_RE = re.compile(r"\u27e6(\d+)\u27e7")
NUM_PH_RE = re.compile(r"(\d[\d.,]*)(\u27e6\d+\u27e7)")
TAG_RE = re.compile(r"</?(\d+)>")
TAG_CMDS = {"emph", "textbf", "textit"}
WRAP_RE = re.compile(r"^\\(section|subsection|subsubsection|paragraph|caption|title)\*?\{")
VERBATIM_LINE_RE = re.compile(
    r"^\s*(%|\\(begin|end|label|centering|includegraphics|toprule|midrule|bottomrule|hline|small|"
    r"setlength|noindent|vspace|newpage|footnotesize|scriptsize|maketitle|input|include|bibliography|"
    r"bibliographystyle|usepackage|documentclass|graphicspath|newcommand|def|IEEEoverridecommandlockouts)\b)")
WORD_RE = re.compile(r"[A-Za-z]{2,}")

# the word before ~\ref (models keep "Section" etc. now and then) -> deterministic target-language word
REF_WORDS_ID = {"Fig.": "Gambar", "Figs.": "Gambar", "Figure": "Gambar", "Table": "Tabel", "Tables": "Tabel",
                "Section": "Bagian", "Sections": "Bagian", "Eq.": "Pers.", "Algorithm": "Algoritma"}


@dataclass
class Unit:
    uid: int
    file: str
    line: int
    src: str                                  # raw LaTeX of the unit
    masked: str                               # text sent to the models
    ph: List[str] = field(default_factory=list)
    tags: Dict[int, str] = field(default_factory=dict)
    out: Dict[str, str] = field(default_factory=dict)   # stage -> masked translation
    final: Optional[str] = None               # masked final translation
    metrics: Dict[str, object] = field(default_factory=dict)

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(PH_RE.sub(" ", TAG_RE.sub(" ", self.masked))))


def balanced(s: str, i: int, open_ch: str = "{", close_ch: str = "}") -> int:
    """``s[i] == open_ch``; index just past the matching ``close_ch`` (or ``len(s)``)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s)


class Masker:
    def __init__(self) -> None:
        self.ph: List[str] = []
        self.tags: Dict[int, str] = {}
        self._out: List[str] = []
        self._last_ph = False

    def _keep(self, s: str) -> None:
        if self._last_ph:                      # merge adjacent verbatim spans into one placeholder
            self.ph[-1] += s
            return
        self.ph.append(s)
        self._out.append(f"{PH_OPEN}{len(self.ph) - 1}{PH_CLOSE}")
        self._last_ph = True

    def _text(self, s: str) -> None:
        if s:
            self._out.append(s)
            self._last_ph = False

    def mask(self, s: str) -> str:
        i, n = 0, len(s)
        buf: List[str] = []

        def flush() -> None:
            if buf:
                self._text("".join(buf))
                buf.clear()

        while i < n:
            c = s[i]
            if c == "%":
                flush(); self._keep(s[i:]); break
            if c == "$":
                j = s.find("$", i + 1)
                j = n if j < 0 else j + 1
                flush(); self._keep(s[i:j]); i = j; continue
            if c in "~&":
                flush(); self._keep(c); i += 1; continue
            if c == "{":
                j = balanced(s, i)
                flush(); self._keep(s[i:j]); i = j; continue
            if c == "\\":
                if i + 1 < n and s[i + 1] == "(":
                    j = s.find("\\)", i + 2)
                    j = n if j < 0 else j + 2
                    flush(); self._keep(s[i:j]); i = j; continue
                m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
                if not m:                       # \\, \%, \&, \ , \!, \{ ...
                    flush(); self._keep(s[i:i + 2]); i += 2; continue
                name, j = m.group(1), i + m.end()
                if name in TAG_CMDS and j < n and s[j] == "{":
                    k = balanced(s, j)
                    tag = len(self.tags) + 1
                    self.tags[tag] = name
                    flush()
                    self._text(f"<{tag}>")
                    self.mask(s[j + 1:k - 1])
                    self._text(f"</{tag}>")
                    i = k
                    continue
                while j < n and s[j] in "[{":   # generic command: swallow attached [opt] and {arg} groups
                    j = balanced(s, j, "[", "]") if s[j] == "[" else balanced(s, j)
                flush(); self._keep(s[i:j]); i = j; continue
            buf.append(c)
            i += 1
        flush()
        return "".join(self._out)


def mask_text(src: str) -> Tuple[str, List[str], Dict[int, str]]:
    m = Masker()
    return m.mask(src), m.ph, m.tags


def unmask(text: str, ph: List[str], tags: Dict[int, str]) -> str:
    t = text
    for a, b in (("\u201c", "``"), ("\u201d", "''"), ("\u2018", "`"), ("\u2019", "'"), ("\u2014", "---"),
                 ("\u2013", "--"), ("\u2026", "\\ldots"), ("\u00a0", "~")):
        t = t.replace(a, b)
    t = re.sub(r"(?<!\\)([%&#_$])", r"\\\1", t)   # bare specials typed by the model
    t = re.sub(r"<(\d+)>(.*?)</\1>", lambda m: f"\\{tags.get(int(m.group(1)), 'emph')}{{{m.group(2)}}}", t, flags=re.S)
    t = TAG_RE.sub("", t)
    return PH_RE.sub(lambda m: ph[int(m.group(1))] if int(m.group(1)) < len(ph) else "", t)


def integrity_ok(masked_src: str, cand: str) -> bool:
    if sorted(PH_RE.findall(masked_src)) != sorted(PH_RE.findall(cand)) or \
            sorted(TAG_RE.findall(masked_src)) != sorted(TAG_RE.findall(cand)):
        return False
    for num, ph in NUM_PH_RE.findall(masked_src):     # 58⟦3⟧ must stay glued to its number
        if f"{num}{ph}" not in cand and f"{num.replace('.', ',')}{ph}" not in cand:
            return False
    return True


def split_cells(s: str) -> Optional[List[str]]:
    """Top-level unescaped ``&`` -> table-row cells; None when the line is not a row."""
    cells, depth, i, start, in_math = [], 0, 0, 0, False
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2; continue
        if c == "$":
            in_math = not in_math
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and depth == 0 and not in_math:
            cells.append(s[start:i]); start = i + 1
        i += 1
    if not cells:
        return None
    cells.append(s[start:])
    return cells


def fix_ref_words(tex: str, words: Dict[str, str] = REF_WORDS_ID) -> str:
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")(~\\ref\{)")
    return pat.sub(lambda m: words[m.group(1)] + m.group(2), tex)


@dataclass
class Line:
    parts: List[object]                        # str (verbatim) | Unit

    def render(self) -> str:
        out = []
        for p in self.parts:
            if isinstance(p, Unit):
                text = p.final if p.final is not None else p.masked
                lead = p.src[: len(p.src) - len(p.src.lstrip())]
                trail = p.src[len(p.src.rstrip()):]
                out.append(lead + unmask(text, p.ph, p.tags).strip() + trail)
            else:
                out.append(p)
        return "".join(out)


class Doc:
    """A set of LaTeX files split into translatable units; ``render`` writes them back."""

    def __init__(self) -> None:
        self.units: List[Unit] = []
        self.lines: Dict[str, List[Line]] = {}

    def _unit(self, file: str, lno: int, src: str) -> object:
        masked, ph, tags = mask_text(src)
        u = Unit(len(self.units), file, lno, src, masked, ph, tags)
        if u.words < 1:                        # numbers / placeholders only -> verbatim
            return src
        self.units.append(u)
        return u

    def add_file(self, name: str, text: str) -> None:
        lines: List[Line] = []
        for lno, raw in enumerate(text.split("\n"), 1):
            if not raw.strip() or VERBATIM_LINE_RE.match(raw):
                lines.append(Line([raw])); continue
            indent = raw[: len(raw) - len(raw.lstrip())]
            body, tail = raw.strip(), ""
            if body.endswith("\\\\"):
                body, tail = body[:-2].rstrip(), " \\\\"
            m = WRAP_RE.match(body)
            if m and balanced(body, m.end() - 1) == len(body):
                lines.append(Line([indent + body[: m.end()], self._unit(name, lno, body[m.end():-1]), "}" + tail]))
                continue
            if body.startswith("\\item"):
                lines.append(Line([indent + "\\item", self._unit(name, lno, body[len("\\item"):]), tail]))
                continue
            cells = split_cells(body)
            if cells:
                parts: List[object] = [indent]
                for ci, cell in enumerate(cells):
                    parts.append(self._unit(name, lno, cell))
                    if ci < len(cells) - 1:
                        parts.append("&")
                parts.append(tail)
                lines.append(Line(parts)); continue
            lines.append(Line([indent, self._unit(name, lno, body), tail]))
        self.lines[name] = lines

    def render(self, name: str) -> str:
        return "\n".join(ln.render() for ln in self.lines[name])
