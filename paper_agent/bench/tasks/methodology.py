"""methodology — multiple-choice questions on research methodology and statistics (Indonesian). Score: accuracy."""
from __future__ import annotations

import re
from typing import Optional, Tuple

from ..schema import Item
from . import Context, Task, parse_json_loose

SYSTEM = ("Kamu dosen metodologi penelitian dan statistika di perguruan tinggi Indonesia. Jawab soal pilihan ganda "
          "berikut dengan tepat. Keluarkan HANYA JSON: {\"jawaban\": \"A\"|\"B\"|\"C\"|\"D\"}.")


def chosen(text: str) -> Optional[str]:
    data = parse_json_loose(text)
    if isinstance(data, dict):
        for k in ("jawaban", "answer", "pilihan", "jawab"):
            v = data.get(k)
            if isinstance(v, str):
                m = re.match(r"\s*\(?([ABCDabcd])\b", v)
                if m:
                    return m.group(1).upper()
    m = re.search(r"(?:jawaban|answer)\W{0,5}\(?([ABCD])\b", text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text or "")
    return letters[0] if len(set(letters)) == 1 else None


class MethodologyTask(Task):
    name = "methodology"
    max_tokens = 256

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        opts = item.input["options"]
        user = item.input["question"] + "\n\n" + "\n".join(f"{k}. {opts[k]}" for k in sorted(opts))
        return SYSTEM, user

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        got = chosen(record.get("text", ""))
        key = item.reference["answer"]
        return {"score": 1.0 if got == key else 0.0, "chosen": got, "key": key, "parsed": got is not None}
