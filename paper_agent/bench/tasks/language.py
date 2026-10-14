"""language — formal academic Indonesian: an abstract from a structured research summary, or a paraphrase of an
informal student draft into standard (baku) Indonesian.

Deterministic checks (word lists in ``bench/wordlists``): length; English leakage (share of English function words
among function words); non-standard words (``id_nonstandard.tsv``, e.g. *analisa*, *praktek*, *gak*); recall of the
numbers and key terms that must survive; numbers not present in the source (fabrication); overly long sentences
(> 40 words); a single paragraph for abstracts. Auto score = mean of these components. Fluency and faithfulness are
judged by cross-vendor LLM judges and, on a subset, by people.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from ..schema import Item
from . import Context, Task, number_forms, numbers_in, words

SYSTEM = ("Kamu penyunting naskah akademik berbahasa Indonesia. Kamu menulis dalam bahasa Indonesia baku sesuai KBBI "
          "dan EYD Edisi V, ringkas, runtut, dan setia pada informasi yang diberikan.")
FIELDS_ORDER = ("judul", "latar", "tujuan", "metode", "hasil", "kesimpulan")
LONG_SENTENCE = 40


def summary_text(summary: dict) -> str:
    keys = [k for k in FIELDS_ORDER if k in summary] + [k for k in summary if k not in FIELDS_ORDER]
    return "\n".join(f"{k.capitalize()}: {summary[k]}" for k in keys)


def load_wordlists(ctx: Context) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    if "language:lists" not in ctx.cache:
        base = ctx.repo / "bench" / "wordlists"
        nonstd: Dict[str, str] = {}
        p = base / "id_nonstandard.tsv"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split("\t")
                if len(parts) == 2 and parts[0] and not parts[0].startswith("#"):
                    nonstd[parts[0].lower()] = parts[1]

        def lst(name: str) -> Set[str]:
            q = base / name
            return {w.strip().lower() for w in q.read_text(encoding="utf-8").splitlines()
                    if w.strip() and not w.startswith("#")} if q.exists() else set()
        ctx.cache["language:lists"] = (nonstd, lst("id_stopwords.txt"), lst("en_stopwords.txt"))
    return ctx.cache["language:lists"]


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if words(p)]


def recall_numbers(text: str, wanted: List[str]) -> Tuple[float, List[str]]:
    have = set()
    for m in numbers_in(text):
        have |= number_forms(m)
    missing = [w for w in wanted if not number_forms(w) & have]
    return (1.0 - len(missing) / len(wanted) if wanted else 1.0), missing


def recall_terms(text: str, terms: List[str]) -> Tuple[float, List[str]]:
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^0-9a-zà-ÿ]+", " ", (text or "").lower())) + " "
    missing = [k for k in terms if " " + re.sub(r"[^0-9a-zà-ÿ]+", " ", k.lower()).strip() + " " not in t]
    return (1.0 - len(missing) / len(terms) if terms else 1.0), missing


def length_score(n: int, lo: int, hi: int) -> float:
    if lo <= n <= hi:
        return 1.0
    dev = (lo - n) if n < lo else (n - hi)
    return max(0.0, 1.0 - dev / hi)


class LanguageTask(Task):
    name = "language"
    json_mode = False
    max_tokens = 900
    judged = True

    def source_text(self, item: Item) -> str:
        i = item.input
        return summary_text(i["summary"]) if i.get("kind") == "abstract" else (i.get("text") or "")

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        i = item.input
        label = "RINGKASAN PENELITIAN" if i.get("kind") == "abstract" else "DRAF MAHASISWA"
        user = f"{i['instruction']}\n\n{label}:\n{self.source_text(item)}\n\nTulis hanya teks hasilnya, tanpa judul atau komentar."
        return SYSTEM, user

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        text = (record.get("text") or "").strip()
        nonstd, id_sw, en_sw = load_wordlists(ctx)
        toks = [w.lower() for w in words(text)]
        n = len(toks)
        i, ref = item.input, item.reference
        lo, hi = int(i.get("min_words", 0)), int(i.get("max_words", 10_000))
        id_hits = sum(1 for w in toks if w in id_sw)
        en_hits = sum(1 for w in toks if w in en_sw)
        en_ratio = en_hits / (id_hits + en_hits) if id_hits + en_hits else 1.0
        bad_words = sorted({w for w in toks if w in nonstd})
        bad_count = sum(1 for w in toks if w in nonstd)
        num_recall, num_missing = recall_numbers(text, list(ref.get("numbers") or []))
        src = set()
        for m in numbers_in(self.source_text(item)):
            src |= number_forms(m)
        out_nums = numbers_in(text)
        fabricated = [m for m in out_nums if not number_forms(m) & src]
        term_recall, term_missing = recall_terms(text, list(ref.get("key_terms") or []))
        sents = sentences(text)
        long_share = sum(1 for s in sents if len(words(s)) > LONG_SENTENCE) / len(sents) if sents else 1.0
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        parts = {"length": length_score(n, lo, hi),
                 "indonesian": 1.0 if en_ratio <= 0.03 else max(0.0, 1.0 - 5 * en_ratio),
                 "standard_words": max(0.0, 1.0 - 0.25 * bad_count),
                 "numbers_kept": num_recall,
                 "no_new_numbers": 1.0 - len(fabricated) / len(out_nums) if out_nums else 1.0,
                 "key_terms": term_recall,
                 "sentence_length": 1.0 - long_share}
        if i.get("kind") == "abstract":
            parts["one_paragraph"] = 1.0 if len(paragraphs) == 1 and not re.search(r"(?m)^\s*(#|\*\*|abstrak\b)", text,
                                                                                  re.IGNORECASE) else 0.0
        return {"score": round(sum(parts.values()) / len(parts), 4), "parts": {k: round(v, 4) for k, v in parts.items()},
                "words": n, "english_ratio": round(en_ratio, 4), "nonstandard": bad_words, "nonstandard_count": bad_count,
                "numbers_missing": num_missing, "numbers_fabricated": fabricated[:20], "terms_missing": term_missing,
                "sentences": len(sents), "long_sentences": round(long_share, 4)}
