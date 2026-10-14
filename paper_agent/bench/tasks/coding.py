"""coding — research data analysis in Python on a synthetic dataset; the program runs in the sandbox.

The model answers with one program that prints a JSON object. It runs offline in :mod:`paper_agent.bench.sandbox`
with the task's data files. Each expected key is compared within tolerance. If the first program crashes, times
out, prints no JSON or misses keys, the model gets one repair turn with the error (never with the expected values).

Score = share of correct keys of the *first* program; ``pass1`` (all keys right at once) and ``pass_repair`` (all
right after at most one repair) are reported alongside.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Optional, Tuple

from ..schema import Item
from . import Context, Task, code_block, parse_json_loose

SYSTEM = ("Kamu asisten analisis data penelitian. Kamu menulis program Python yang benar, lengkap, dan dapat "
          "langsung dijalankan.")
ENV_NOTE = ("Lingkungan eksekusi: Python 3.11 tanpa akses internet; pustaka yang tersedia: numpy, pandas, scipy, "
            "statsmodels, scikit-learn (dan pustaka standar). Berkas data ada di direktori kerja: {files}.")
TAIL = ("Tulis SATU program Python lengkap yang mencetak tepat satu objek JSON ke stdout dengan kunci: {keys}. "
        "Nilai numerik ditulis sebagai angka (bukan string, tidak dibulatkan). Jawab hanya dengan satu blok "
        "```python berisi program tersebut.")
REPAIR = ("{prompt}\n\nProgram berikut sudah dijalankan tetapi gagal ({why}):\n```python\n{code}\n```\n"
          "Pesan dari eksekusi (dipotong):\n```\n{err}\n```\n"
          "Perbaiki programnya. Jawab hanya dengan satu blok ```python berisi program lengkap.")


def _preview(data: bytes, n: int = 5) -> Tuple[str, int]:
    lines = data.decode("utf-8", "replace").splitlines()
    return "\n".join(lines[: n + 1]), max(0, len(lines) - 1)


def last_json(stdout: str) -> Any:
    s = (stdout or "").strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
    return parse_json_loose(s)


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def value_ok(got, exp, tol: Dict[str, float]) -> bool:
    if isinstance(exp, bool):
        return got is exp or (isinstance(got, str) and got.strip().lower() == str(exp).lower())
    if isinstance(exp, (int, float)):
        g = _num(got)
        if g is None or math.isnan(g):
            return False
        a, r = float(tol.get("abs", 1e-6)), float(tol.get("rel", 1e-6))
        return abs(g - exp) <= a or abs(g - exp) <= r * abs(exp)
    if isinstance(exp, str):
        return isinstance(got, (str, int, float)) and str(got).strip().lower() == exp.strip().lower()
    if isinstance(exp, list):
        return isinstance(got, list) and len(got) == len(exp) and all(value_ok(g, e, tol) for g, e in zip(got, exp))
    if isinstance(exp, dict):
        return isinstance(got, dict) and all(k in got and value_ok(got[k], v, tol) for k, v in exp.items())
    return got == exp


def compare(out, expected: dict, tolerance: dict) -> Dict[str, bool]:
    default = tolerance.get("default", {})
    return {k: isinstance(out, dict) and k in out and value_ok(out[k], v, tolerance.get(k, default))
            for k, v in expected.items()}


class CodingTask(Task):
    name = "coding"
    json_mode = False
    max_tokens = 2500

    def files(self, item: Item, ctx: Context) -> Dict[str, bytes]:
        return {name: ctx.path(rel).read_bytes() for name, rel in item.input["files"].items()}

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        files = self.files(item, ctx)
        parts = [item.input["instruction"].strip(), "", ENV_NOTE.format(files=", ".join(files))]
        for name, data in files.items():
            head, rows = _preview(data)
            parts += [f"Pratinjau {name} (baris awal dari {rows} baris data):", "```", head, "```"]
        parts.append(TAIL.format(keys=", ".join(item.input.get("output_keys") or item.reference["expected"])))
        return SYSTEM, "\n".join(parts)

    def execute(self, item: Item, code: Optional[str], ctx: Context) -> dict:
        if not code:
            return {"ok": False, "why": "tidak ada blok kode", "stderr": "", "out": None}
        from .. import sandbox
        key = hashlib.sha256((item.id + "\0" + code).encode("utf-8")).hexdigest()[:20]
        cache = ctx.cache.setdefault("coding:exec", {})
        if key in cache:
            return cache[key]
        r = sandbox.run_python(code, self.files(item, ctx), ctx.python or sandbox.default_python(), timeout=90)
        out = last_json(r.stdout) if r.returncode == 0 else None
        expected = item.reference["expected"]
        missing = [k for k in expected if not (isinstance(out, dict) and k in out)]
        if r.timed_out:
            why = "melewati batas waktu 90 detik"
        elif r.returncode != 0:
            why = f"kode keluar {r.returncode}"
        elif not isinstance(out, dict):
            why = "keluaran bukan satu objek JSON"
        elif missing:
            why = "kunci JSON tidak lengkap: " + ", ".join(missing)
        else:
            why = ""
        res = {"ok": not why, "why": why, "returncode": r.returncode, "timed_out": r.timed_out, "seconds": r.seconds,
               "sandboxed": r.sandboxed, "stderr": r.stderr[-1500:], "stdout_tail": r.stdout[-500:], "out": out,
               "code_sha": key}
        cache[key] = res
        return res

    def followup(self, item: Item, text: str, ctx: Context) -> Tuple[Optional[str], dict]:
        code = code_block(text, ("python", "py", "python3"))
        res = self.execute(item, code, ctx)
        if res["ok"]:
            return None, {"exec": [res]}
        _, user = self.prompt(item, ctx)
        err = res.get("stderr") or res.get("stdout_tail") or res["why"]
        return REPAIR.format(prompt=user, why=res["why"], code=(code or text or "")[:6000], err=err[-1500:]), {"exec": [res]}

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        ref = item.reference
        attempts = [record.get("text", "")] + [f.get("text", "") for f in record.get("followups") or []]
        cached = {e.get("code_sha"): e for e in record.get("exec") or [] if isinstance(e, dict)}
        ctx.cache.setdefault("coding:exec", {}).update({k: v for k, v in cached.items() if k})
        results = []
        for text in attempts:
            res = self.execute(item, code_block(text, ("python", "py", "python3")), ctx)
            keys = compare(res.get("out"), ref["expected"], ref.get("tolerance", {}))
            results.append({"ok_run": res["ok"], "why": res["why"], "keys": keys,
                            "acc": sum(keys.values()) / len(keys) if keys else 0.0})
        first, last = results[0], results[-1]
        return {"score": round(first["acc"], 4), "pass1": first["acc"] == 1.0, "pass_repair": last["acc"] == 1.0,
                "acc_final": round(last["acc"], 4), "ran_first": first["ok_run"], "repaired": len(results) > 1,
                "attempts": results}
