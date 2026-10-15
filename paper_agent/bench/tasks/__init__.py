"""Task registry. Each task module defines a :class:`Task` subclass: prompt construction and a deterministic scorer.

A score is a dict with ``score`` in [0, 1] (the task's headline number) plus task-specific details. Scorers never
call an LLM; prose quality is judged separately (:mod:`paper_agent.bench.judge`) for tasks with ``judged = True``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..schema import Item


@dataclass
class Context:
    """Resources scorers and prompt builders may need."""
    repo: Path                                 # repository root (bench/data, bench/wordlists)
    private_dir: Optional[Path] = None
    python: Optional[str] = None               # interpreter for model-written analysis code (sandboxed)
    offline: bool = False                      # no network: tasks needing it are skipped
    cache: Dict[str, Any] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        p = (self.repo / rel).resolve()
        if self.repo.resolve() not in p.parents:
            raise ValueError(f"path escapes the repository: {rel}")
        return p


class Task:
    name = ""
    json_mode = True
    max_tokens = 1024          # answer budget; thinking models get extra (see ModelInfo.engine_options)
    judged = False
    needs_network = False

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        raise NotImplementedError

    def followup(self, item: Item, text: str, ctx: Context) -> Tuple[Optional[str], dict]:
        """A second user message after the first answer (e.g. a repair request) or None, plus information to store
        in the run record (e.g. the execution result that triggered the repair)."""
        return None, {}

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        raise NotImplementedError

    def reference_fingerprint(self, item: Item, ctx: Context) -> str:
        """Hash of reference data kept outside the item (e.g. silver labels); a change invalidates cached scores."""
        return ""


# ----------------------------------------------------------------------------- parsing helpers
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def _balanced(t: str, opener: str, closer: str) -> Any:
    """First parseable balanced ``opener…closer`` span in ``t`` (strings respected), else None."""
    start = t.find(opener)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
        start = t.find(opener, start + 1)
    return None


def parse_json_loose(text: str) -> Any:
    """JSON from a model answer: plain, fenced, the first balanced ``{…}``, a repaired object (wrong or missing
    closers), then the same for ``[…]``; None if nothing parses."""
    if not text:
        return None
    t = text.strip()
    for cand in [t] + [m.group(2) for m in _FENCE.finditer(t)]:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        got = _balanced(t, opener, closer)
        if got is not None:
            return got
        i = t.find(opener)
        if i != -1:
            fixed = repair_json(t[i:])
            if fixed:
                try:
                    return json.loads(fixed)
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def repair_json(text: str) -> Optional[str]:
    """Balance brackets outside strings: a wrong closer is replaced by the expected one and missing closers are
    appended (``{"a": [1, 2]]`` -> ``{"a": [1, 2]}``; a cut-off answer is closed). Only the first object/array from
    its opening bracket is considered."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    out, stack, in_str, esc = [], [], False, False
    pairs = {"{": "}", "[": "]"}
    for ch in text[min(starts):]:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack:
                break
            ch = stack.pop()
            out.append(ch)
            if not stack:
                break
            continue
        out.append(ch)
    if in_str:
        out.append('"')
    out.extend(reversed(stack))
    return "".join(out)


def code_block(text: str, langs: Tuple[str, ...]) -> Optional[str]:
    """The longest fenced block whose language is in ``langs`` (or untagged); None if there is none."""
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text or "")]
    tagged = [b for lang, b in blocks if lang in langs]
    pool = tagged or [b for lang, b in blocks if not lang]
    if pool:
        return max(pool, key=len)
    # an unterminated fence (answer cut off) still carries the code
    m = re.search(r"```[ \t]*(?:" + "|".join(map(re.escape, langs)) + r")?[^\n]*\n(.*)\Z", text or "", re.DOTALL)
    return m.group(1) if m else None


def words(text: str) -> list:
    return re.findall(r"[0-9A-Za-zÀ-ÿ]+(?:[-'][0-9A-Za-zÀ-ÿ]+)*", text or "")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z%]+", " ", (s or "").lower())).strip()


NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*%?")


def number_forms(s: str) -> set:
    """Readings of a number string under Indonesian and English conventions: ``1.250`` -> {1250, 1.25};
    ``0,85`` -> {0.85}; ``85%`` -> {85}. Returned as canonical strings."""
    s = s.strip().rstrip("%")
    out = set()

    def canon(x: str) -> str:
        try:
            v = float(x)
        except ValueError:
            return x
        return str(int(v)) if v == int(v) else repr(round(v, 10))

    if re.fullmatch(r"\d+", s):
        return {canon(s)}
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):         # 1.250 or 1.250.000: Indonesian thousands (or EN decimal)
        out.add(canon(s.replace(".", "")))
        if s.count(".") == 1:
            out.add(canon(s))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):          # 1,250: English thousands (or Indonesian decimal)
        out.add(canon(s.replace(",", "")))
        if s.count(",") == 1:
            out.add(canon(s.replace(",", ".")))
    if not out:
        if s.count(",") == 1 and "." not in s:
            out.add(canon(s.replace(",", ".")))
        elif s.count(".") == 1 and "," not in s:
            out.add(canon(s))
        else:                                              # 1.234,5 (ID) or 1,234.5 (EN)
            out.add(canon(s.replace(".", "").replace(",", ".")))
            out.add(canon(s.replace(",", "")))
    return out


def numbers_in(text: str) -> list:
    return NUMBER.findall(text or "")


def registry() -> Dict[str, Task]:
    from . import (coding, extraction, grounded, language, latex, methodology, references,  # noqa: F401
                   structure)
    return {t.name: t for t in (structure.StructureTask(), methodology.MethodologyTask(),
                                extraction.ExtractionTask(), grounded.GroundedTask(),
                                references.ReferencesTask(), language.LanguageTask(), coding.CodingTask(),
                                latex.LatexTask())}
