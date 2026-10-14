"""LLM judges for prose tasks (``grounded``, ``language``): judges from vendor families other than the candidate's.

Each judge sees the task prompt the candidate received and the candidate's answer — never the model name — and
returns 1–5 ratings on five criteria. Every answer is rated by all judges except one of the candidate's own vendor
family (three judges from three vendors -> two or three ratings per answer); judge leniency is estimated on the
answers all judges rated and removed (:func:`leniency_offsets`) so that families judged by different subsets stay
comparable. Jobs are shuffled with a fixed seed. Results go to ``<run_dir>/judges/<judge-slug>.jsonl`` keyed by
candidate, item, repetition and answer hash.
"""
from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import engines
from .models import ModelInfo, ModelSpec, family_of
from .runner import latest_records
from .schema import Item
from .score import answer_hash, slug
from .tasks import Context, parse_json_loose, registry

DEFAULT_JUDGES = ["copilot:gpt-6.1-sol@low", "copilot:claude-sonnet-5@low", "copilot:gemini-4-argon"]
CRITERIA = {
    "grounded": {"kesetiaan_sumber": "setiap klaim didukung sumber yang diberikan; tidak ada fakta/angka karangan",
                 "cakupan": "memakai dan menyintesis sumber-sumber yang relevan, bukan sekadar meringkas satu per satu",
                 "keruntutan": "alur logis, paragraf tersusun, transisi jelas",
                 "bahasa_baku": "bahasa Indonesia akademik baku (KBBI/EYD V), istilah tepat",
                 "kepatuhan": "format sitasi [P#], panjang sesuai, tanpa judul/daftar pustaka"},
    "language": {"kesetiaan_isi": "setia pada ringkasan/draf; tidak menambah atau mengubah informasi",
                 "kelengkapan": "semua informasi penting (dan angka) dipertahankan",
                 "keruntutan": "alur logis dan padu",
                 "bahasa_baku": "bahasa Indonesia akademik baku (KBBI/EYD V), tanpa kata tidak baku/campuran asing",
                 "kepatuhan": "bentuk dan panjang sesuai instruksi"},
}
SYSTEM = ("Kamu penilai naskah akademik berbahasa Indonesia yang tegas dan konsisten. Kamu menilai jawaban sebuah "
          "sistem terhadap tugas yang diberikan, hanya berdasarkan teks tugas dan jawaban.")
USER = """TUGAS YANG DIBERIKAN KEPADA SISTEM:
<<<
{task_prompt}
>>>

JAWABAN SISTEM:
<<<
{answer}
>>>

Nilai jawaban pada skala 1–5 (1 = sangat buruk, 3 = cukup, 5 = sangat baik) untuk setiap kriteria:
{criteria}
Keluarkan HANYA JSON: {{"skor": {{{keys}}}, "alasan": "<maksimal 3 kalimat>"}}"""


def pick_judges(candidate_family: str, judges: List[str], k: Optional[int] = None) -> List[str]:
    """Judges not of the candidate's vendor family (all of them unless ``k`` is given)."""
    out = [j for j in judges if family_of(ModelSpec.parse(j).model) != candidate_family]
    return out if k is None else out[:k]


def leniency_offsets(rows: List[dict]) -> Dict[str, float]:
    """Per-judge leniency: mean of (judge score − mean of all judges) over the answers that *every* judge rated.
    Candidates of a judge's own vendor are rated by the other judges only, so without this correction a lenient or
    strict judge would shift whole vendor families."""
    by_answer: Dict[tuple, Dict[str, float]] = {}
    judges = set()
    for r in rows:
        if r.get("score") is None:
            continue
        judges.add(r["judge"])
        by_answer.setdefault((r["model"], r["item"], r.get("rep", 0), r["answer_hash"]), {})[r["judge"]] = r["score"]
    common = [v for v in by_answer.values() if set(v) == judges]
    if len(judges) < 2 or not common:
        return {j: 0.0 for j in judges}
    return {j: sum(v[j] - sum(v.values()) / len(v) for v in common) / len(common) for j in judges}


def judge_prompt(task: str, task_prompt: str, answer: str) -> Tuple[str, str]:
    crit = CRITERIA[task]
    lines = "\n".join(f"- {k}: {v}" for k, v in crit.items())
    keys = ", ".join(f'"{k}": 1-5' for k in crit)
    return SYSTEM, USER.format(task_prompt=task_prompt, answer=answer or "(kosong)", criteria=lines, keys=keys)


def parse_ratings(task: str, text: str) -> Optional[Dict[str, int]]:
    obj = parse_json_loose(text)
    if not isinstance(obj, dict):
        return None
    sk = obj.get("skor") if isinstance(obj.get("skor"), dict) else obj
    out = {}
    for k in CRITERIA[task]:
        v = sk.get(k)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        if not 1 <= v <= 5:
            return None
        out[k] = v
    return out


def norm_rating(r: Dict[str, int]) -> float:
    return (sum(r.values()) / len(r) - 1) / 4


def run_judges(items: List[Item], ctx: Context, run_dir: Path, judges: Optional[List[str]] = None,
               models: Optional[List[str]] = None, workers: int = 4, seed: int = 11, timeout: float = 600,
               log: Callable = print) -> dict:
    judges = judges or list(DEFAULT_JUDGES)
    reg = registry()
    by_id = {it.id: it for it in items if reg[it.task].judged}
    jdir = Path(run_dir) / "judges"
    jdir.mkdir(parents=True, exist_ok=True)
    done: Dict[str, set] = {}
    for j in judges:
        p = jdir / f"{slug(j)}.jsonl"
        done[j] = set()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ratings"):
                    done[j].add((r["model"], r["item"], r.get("rep", 0), r["answer_hash"]))
    jobs = []
    for model, recs in latest_records(run_dir).items():
        if models and model not in models:
            continue
        for (iid, rep), rec in recs.items():
            it = by_id.get(iid)
            if it is None or rec.get("error"):
                continue
            h = answer_hash(rec)
            for j in pick_judges(rec.get("family") or "other", judges):
                if (model, iid, rep, h) not in done[j]:
                    jobs.append((j, model, it, rep, rec, h))
    random.Random(seed).shuffle(jobs)
    log(f"judging: {len(jobs)} calls with {judges}")
    locks = {j: threading.Lock() for j in judges}
    stats = {"ok": 0, "failed": 0, "aiu": 0.0}

    def work(job):
        j, model, it, rep, rec, h = job
        spec = ModelSpec.parse(j)
        _, task_prompt = reg[it.task].prompt(it, ctx)
        system, user = judge_prompt(it.task, task_prompt, rec.get("text", ""))
        ratings, reply, err = None, None, None
        for attempt in range(3):
            try:
                reply = engines.chat_reply(spec.engine, spec.model, system, user, timeout=timeout, json_mode=True,
                                           effort=spec.effort)
                ratings = parse_ratings(it.task, reply.text)
                if ratings:
                    break
                err = "unparseable ratings"
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"[:300]
                time.sleep(5 * 3 ** attempt)
        row = {"judge": j, "judge_served": reply.served if reply else None, "model": model, "item": it.id,
               "task": it.task, "rep": rep, "answer_hash": h, "ratings": ratings,
               "score": None if not ratings else round(norm_rating(ratings), 4),
               "reason": (parse_json_loose(reply.text) or {}).get("alasan") if reply and ratings else None,
               "aiu": reply.usage.aiu if reply else None, "error": None if ratings else err,
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        return j, row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            j, row = fut.result()
            with locks[j], (jdir / f"{slug(j)}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok" if row["ratings"] else "failed"] += 1
            stats["aiu"] += row["aiu"] or 0
            if i % 50 == 0 or i == len(futs):
                log(f"  [{i}/{len(futs)}] ok={stats['ok']} failed={stats['failed']} aiu={stats['aiu']:.2f}")
    stats["aiu"] = round(stats["aiu"], 3)
    return stats


def load_judgements(run_dir: Path) -> Dict[Tuple[str, str, int], List[dict]]:
    """{(candidate, item, rep): [latest valid row per judge]} — only rows matching the candidate's current answer
    are meaningful; callers compare ``answer_hash``."""
    out: Dict[Tuple[str, str, int], Dict[str, dict]] = {}
    for p in sorted((Path(run_dir) / "judges").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ratings"):
                out.setdefault((r["model"], r["item"], r.get("rep", 0)), {})[r["judge"] + "|" + r["answer_hash"]] = r
    return {k: list(v.values()) for k, v in out.items()}


def judge_models(judges: List[str]) -> List[ModelInfo]:
    return [ModelInfo(ModelSpec.parse(j), "standard", family_of(ModelSpec.parse(j).model), j) for j in judges]
