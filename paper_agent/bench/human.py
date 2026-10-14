"""Blind sheets for two human raters and their analysis.

``make_sheets`` samples prose answers (grounded + language, spread over models) and extraction silver labels,
assigns anonymous ids (``H01`` …, ``E01`` …) and writes, under ``<run_dir>/human/``: ``PETUNJUK.md`` (instructions),
``bacaan.md`` (what to read), ``rater1/`` and ``rater2/`` (identical CSV sheets to fill in) and ``key.json`` (which
model/item each id is — not for the raters). ``ingest`` reads the filled sheets: Krippendorff's alpha between the
raters, Spearman correlation between the human mean and the LLM-judge mean, accuracy of the silver labels, and gold
extraction labels where both raters agree (``<private>/gold/extraction.json``).
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import stats as S
from .judge import leniency_offsets, load_judgements
from .runner import latest_records
from .schema import Item
from .score import answer_hash
from .tasks import Context, registry
from .tasks import extraction as X

PROSE_COLS = ["kesetiaan", "kelengkapan", "keruntutan", "bahasa_baku", "kepatuhan"]
JUDGE_TO_COL = {"kesetiaan_sumber": "kesetiaan", "kesetiaan_isi": "kesetiaan", "cakupan": "kelengkapan",
                "kelengkapan": "kelengkapan", "keruntutan": "keruntutan", "bahasa_baku": "bahasa_baku",
                "kepatuhan": "kepatuhan"}
GUIDE = """# Petunjuk penilaian (SkripsiBench)

Anda menilai **jawaban sistem AI** tanpa mengetahui sistem mana yang menulisnya. Kerjakan sendiri, jangan
berdiskusi dengan penilai lain sampai selesai.

## A. Tulisan (`penilaian_tulisan.csv`, bacaan di `bacaan.md` bagian A)
Untuk setiap id (H01, H02, …) baca TUGAS lalu JAWABAN, beri nilai 1–5 (1 sangat buruk, 3 cukup, 5 sangat baik):
- **kesetiaan** — isi setia pada sumber/ringkasan/draf; tidak ada fakta atau angka karangan.
- **kelengkapan** — tinjauan pustaka: memakai dan menyintesis sumber yang relevan; abstrak/parafrase: semua
  informasi penting dan angka tetap ada.
- **keruntutan** — alur logis, padu, transisi jelas.
- **bahasa_baku** — bahasa Indonesia akademik baku (KBBI/EYD V), istilah tepat, tanpa kata tidak baku.
- **kepatuhan** — mengikuti instruksi (format sitasi [P#], panjang, satu paragraf, tanpa judul, dsb.).
Kolom `catatan` boleh diisi singkat.

## B. Label ekstraksi (`penilaian_ekstraksi.csv`, bacaan di `bacaan.md` bagian B)
Untuk setiap id (E01, …) baca cuplikan skripsi, lalu untuk tiap baris (bidang) isi kolom `benar` dengan 1 bila
`nilai_label` sesuai cuplikan atau 0 bila tidak. Bila 0, tulis nilai yang benar di `nilai_seharusnya` memakai
kosakata yang sama (daftar dipisah koma; kosong/null bila memang tidak disebutkan).
Kosakata: lihat bagian akhir `bacaan.md`.
"""


def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _silver(ctx: Context) -> dict:
    p = ctx.private_dir / "silver" / "extraction.json" if ctx.private_dir else None
    if not p or not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("labels", data)


def make_sheets(items: List[Item], ctx: Context, run_dir: Path, n_prose: int = 30, n_extraction: int = 20,
                seed: int = 5, log: Callable = print) -> dict:
    reg = registry()
    rng = random.Random(seed)
    by_id = {it.id: it for it in items}
    hdir = Path(run_dir) / "human"
    if (hdir / "key.json").exists():
        raise SystemExit(f"{hdir}/key.json exists — sheets were already made (delete the folder to redo)")
    recs = latest_records(run_dir)
    pool: Dict[str, List[tuple]] = {"grounded": [], "language": []}
    for model, rows in recs.items():
        for (iid, rep), r in rows.items():
            it = by_id.get(iid)
            if it and it.task in pool and not r.get("error") and rep == 0 and (r.get("text") or "").strip():
                pool[it.task].append((model, it, r))
    picked: List[tuple] = []
    for task, share in (("grounded", n_prose // 2), ("language", n_prose - n_prose // 2)):
        cands = pool[task][:]
        rng.shuffle(cands)
        models = sorted({c[0] for c in cands})
        rng.shuffle(models)
        used_items: Dict[str, int] = {}
        chosen: List[tuple] = []
        while len(chosen) < share and cands:
            progress = False
            for m in models:
                if len(chosen) >= share:
                    break
                opts = [c for c in cands if c[0] == m]
                if not opts:
                    continue
                opts.sort(key=lambda c: used_items.get(c[1].id, 0))
                c = opts[0]
                cands.remove(c)
                chosen.append(c)
                used_items[c[1].id] = used_items.get(c[1].id, 0) + 1
                progress = True
            if not progress:
                break
        picked += chosen
    rng.shuffle(picked)
    key, md, rows = {}, ["# Bahan penilaian", "", "## A. Tulisan", ""], []
    for i, (model, it, r) in enumerate(picked, 1):
        hid = f"H{i:02d}"
        _, task_prompt = reg[it.task].prompt(it, ctx)
        key[hid] = {"model": model, "item": it.id, "task": it.task, "answer_hash": answer_hash(r)}
        md += [f"### {hid}", "", "**TUGAS**", "", "```text", task_prompt.strip(), "```", "", "**JAWABAN**", "",
               (r.get("text") or "").strip(), "", "---", ""]
        rows.append([hid, it.task] + [""] * len(PROSE_COLS) + [""])
    silver = _silver(ctx)
    ext = [it for it in items if it.task == "extraction" and it.id in silver]
    rng.shuffle(ext)
    ext = sorted(ext[:n_extraction], key=lambda i: i.id)
    md += ["## B. Label ekstraksi", ""]
    erows = []
    for j, it in enumerate(ext, 1):
        eid = f"E{j:02d}"
        rec = silver[it.id].get("*", {})
        key[eid] = {"item": it.id, "task": "extraction", "silver": rec}
        md += [f"### {eid}", "", "```text", it.input["excerpt"].strip(), "```", "", "| bidang | nilai_label |", "|---|---|"]
        for f in X.FIELDS:
            if f in rec:
                v = rec[f]
                shown = ", ".join(v) if isinstance(v, list) else ("null" if v is None else str(v).lower())
                md.append(f"| {f} | {shown} |")
                erows.append([eid, f, shown, "", "", ""])
        md += ["", "---", ""]
    md += ["## Kosakata ekstraksi", ""] + [f"- **{f}**: {', '.join(v)}" for f, v in {**X.ENUMS, **X.LISTS}.items()]
    for rater in ("rater1", "rater2"):
        _write_csv(hdir / rater / "penilaian_tulisan.csv", ["id", "tugas"] + PROSE_COLS + ["catatan"], rows)
        _write_csv(hdir / rater / "penilaian_ekstraksi.csv",
                   ["id", "bidang", "nilai_label", "benar", "nilai_seharusnya", "catatan"], erows)
    (hdir / "PETUNJUK.md").write_text(GUIDE, encoding="utf-8")
    (hdir / "bacaan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (hdir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"human sheets: {len(picked)} prose, {len(ext)} extraction items -> {hdir}")
    return {"prose": len(picked), "extraction": len(ext), "dir": str(hdir)}


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _int(v) -> Optional[int]:
    try:
        x = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if 1 <= x <= 5 else None


def _parse_value(field: str, s: str):
    s = (s or "").strip()
    if field in X.LISTS:
        return sorted({X.norm_value(v) for v in s.split(",") if v.strip()} & set(X.LISTS[field]))
    if field == "n_sampel":
        return int(s) if s.isdigit() else None
    if field == "hipotesis":
        return s.lower() in ("true", "1", "ya")
    if not s or s.lower() == "null":
        return None
    v = X.norm_value(s)
    return v if v in X.ENUMS.get(field, []) else None


def ingest(items: List[Item], ctx: Context, run_dir: Path, log: Callable = print) -> dict:
    hdir = Path(run_dir) / "human"
    key = json.loads((hdir / "key.json").read_text(encoding="utf-8"))
    r1 = {r["id"]: r for r in _read_csv(hdir / "rater1" / "penilaian_tulisan.csv")}
    r2 = {r["id"]: r for r in _read_csv(hdir / "rater2" / "penilaian_tulisan.csv")}
    res: dict = {"prose": {}, "extraction": {}}
    ids = sorted(i for i in key if i.startswith("H"))
    alpha = {}
    for col in PROSE_COLS:
        units = [[_int(r1.get(i, {}).get(col)), _int(r2.get(i, {}).get(col))] for i in ids]
        a = S.krippendorff_alpha(units, "interval")
        alpha[col] = None if a is None else round(a, 3)
    means_h = {}
    for i in ids:
        vals = [_int(r.get(i, {}).get(c)) for r in (r1, r2) for c in PROSE_COLS]
        vals = [v for v in vals if v is not None]
        if vals:
            means_h[i] = (statistics.mean(vals) - 1) / 4
    unit_means = []
    for i in ids:
        row = []
        for r in (r1, r2):
            v = [x for x in (_int(r.get(i, {}).get(c)) for c in PROSE_COLS) if x is not None]
            row.append((statistics.mean(v) - 1) / 4 if v else None)
        unit_means.append(row)
    a_all = S.krippendorff_alpha(unit_means, "interval")
    judg = load_judgements(run_dir)
    offsets = leniency_offsets([r for rows in judg.values() for r in rows])
    hj, jj = [], []
    for i in ids:
        k = key[i]
        js = [j["score"] - offsets.get(j["judge"], 0.0) for j in judg.get((k["model"], k["item"], 0), [])
              if j.get("answer_hash") == k["answer_hash"]]
        if i in means_h and js:
            hj.append(means_h[i])
            jj.append(sum(js) / len(js))
    sp = S.spearman(hj, jj)
    res["prose"] = {"n": len(ids), "rated": len(means_h), "alpha_by_criterion": alpha,
                    "alpha_mean_score": None if a_all is None else round(a_all, 3),
                    "spearman_judge_vs_human": {"n": sp["n"], "rho": None if sp["rho"] is None else round(sp["rho"], 3),
                                                "p": None if sp["p"] is None else round(sp["p"], 4)}}
    e1 = _read_csv(hdir / "rater1" / "penilaian_ekstraksi.csv")
    e2 = {(r["id"], r["bidang"]): r for r in _read_csv(hdir / "rater2" / "penilaian_ekstraksi.csv")}
    agree_units, correct, total, gold = [], 0, 0, {}
    for r in e1:
        k2 = e2.get((r["id"], r["bidang"]))
        b1 = r.get("benar", "").strip()
        b2 = (k2 or {}).get("benar", "").strip()
        if b1 not in ("0", "1") or b2 not in ("0", "1"):
            continue
        agree_units.append([int(b1), int(b2)])
        total += 2
        correct += int(b1) + int(b2)
        item = key[r["id"]]["item"]
        f = r["bidang"]
        silver_val = key[r["id"]]["silver"].get(f)
        if b1 == b2 == "1":
            gold.setdefault(item, {})[f] = silver_val
        elif b1 == b2 == "0":
            v1, v2 = _parse_value(f, r.get("nilai_seharusnya")), _parse_value(f, (k2 or {}).get("nilai_seharusnya"))
            if v1 == v2:
                gold.setdefault(item, {})[f] = v1
    a_ext = S.krippendorff_alpha(agree_units, "nominal")
    res["extraction"] = {"fields_rated": len(agree_units), "silver_accuracy": round(correct / total, 3) if total else None,
                         "alpha_correctness": None if a_ext is None else round(a_ext, 3),
                         "gold_items": len(gold), "gold_fields": sum(len(v) for v in gold.values())}
    if gold and ctx.private_dir:
        gp = ctx.private_dir / "gold" / "extraction.json"
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps({"labels": gold}, ensure_ascii=False, indent=1), encoding="utf-8")
    (hdir / "hasil.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(res, ensure_ascii=False, indent=1))
    return res
