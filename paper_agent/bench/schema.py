"""Benchmark items: JSONL files, one item per line, and a manifest of their SHA-256 hashes.

Public items live in ``bench/items/<task>.jsonl`` of the repository; private items (derived from a local thesis
dataset that cannot be redistributed) live outside it (``--private-dir``, ``<dir>/items/<task>.jsonl``) and are
published only as hashes in ``bench/MANIFEST.json``, so a result can be tied to the exact item set without the text.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import TASKS

PRIVACY = ("public", "private")


@dataclass
class Item:
    id: str
    task: str
    privacy: str
    input: dict
    reference: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "task": self.task, "privacy": self.privacy, "input": self.input,
                "reference": self.reference, "meta": self.meta}

    def sha256(self) -> str:
        """Hash of what defines the item (id, task, input, reference); ``meta`` may change without invalidating it."""
        core = {"id": self.id, "task": self.task, "input": self.input, "reference": self.reference}
        return hashlib.sha256(canonical(core).encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def item_from_dict(d: dict) -> Item:
    missing = [k for k in ("id", "task", "privacy", "input") if k not in d]
    if missing:
        raise ValueError(f"item {d.get('id', '?')}: missing {missing}")
    if d["task"] not in TASKS:
        raise ValueError(f"item {d['id']}: unknown task {d['task']!r}")
    if d["privacy"] not in PRIVACY:
        raise ValueError(f"item {d['id']}: privacy must be one of {PRIVACY}")
    return Item(d["id"], d["task"], d["privacy"], d["input"], d.get("reference") or {}, d.get("meta") or {})


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def item_dirs(public_dir: Path, private_dir: Optional[Path] = None) -> List[Path]:
    dirs = [Path(public_dir)]
    if private_dir:
        dirs.append(Path(private_dir) / "items")
    return dirs


def load_items(dirs: List[Path], tasks: Optional[Iterable[str]] = None) -> List[Item]:
    """All items of the given tasks from ``<dir>/<task>.jsonl`` files; ids must be unique."""
    wanted = list(tasks or TASKS)
    items: List[Item] = []
    seen: Dict[str, Path] = {}
    for d in dirs:
        for task in wanted:
            path = Path(d) / f"{task}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                it = item_from_dict(row)
                if it.task != task:
                    raise ValueError(f"{path}: item {it.id} has task {it.task!r}")
                if it.id in seen:
                    raise ValueError(f"duplicate item id {it.id} in {path} and {seen[it.id]}")
                seen[it.id] = path
                items.append(it)
    return items


def build_manifest(items: List[Item], note: str = "") -> dict:
    tasks: Dict[str, dict] = {}
    for it in sorted(items, key=lambda i: i.id):
        t = tasks.setdefault(it.task, {"public": 0, "private": 0, "items": {}})
        t[it.privacy] += 1
        t["items"][it.id] = it.sha256()
    return {"benchmark": "AcademicBench", "created": time.strftime("%Y-%m-%d"), "note": note,
            "n_items": len(items), "tasks": tasks}


def check_manifest(manifest: dict, items: List[Item]) -> List[str]:
    """Differences between a manifest and the items at hand (missing, extra or changed ids)."""
    want = {i: h for t in manifest.get("tasks", {}).values() for i, h in t.get("items", {}).items()}
    have = {it.id: it.sha256() for it in items}
    problems = [f"missing item {i}" for i in sorted(set(want) - set(have))]
    problems += [f"item not in manifest {i}" for i in sorted(set(have) - set(want))]
    problems += [f"item changed {i}" for i in sorted(set(want) & set(have)) if want[i] != have[i]]
    return problems
