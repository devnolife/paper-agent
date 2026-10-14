"""structure — outline of thesis chapter III (methods) and IV (results and discussion) for a title in a study programme.

Reference: the typical sub-section *families* of that programme (share of theses that contain each family, ≥ 25 %),
aggregated from a local Indonesian undergraduate thesis dataset. The model's sub-section titles are mapped to the
same families with the same regular expressions; results-chapter titles that match no family count as *topical*
(titled after the research questions). Per chapter: prevalence-weighted recall, precision, F1; score = mean F1 of
both chapters. ``generic_ch4`` flags a results chapter built around assumption/instrument/hypothesis tests in a
programme whose theses typically do not have them.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..schema import Item
from . import Context, Task, parse_json_loose

CH3_FAMILY = [
    (r"^(jenis|tipe|pendekatan|desain|rancangan|metode) (dan |& )?(pendekatan |jenis |desain )?(penelitian|riset)|"
     r"^jenis penelitian|^research (design|type)|^pendekatan", "Jenis/pendekatan/desain penelitian"),
    (r"lokasi|tempat|waktu|setting", "Lokasi dan waktu penelitian"),
    (r"populasi|sampel|sample", "Populasi dan sampel"),
    (r"informan|subjek|subyek|objek penelitian|sasaran penelitian", "Subjek/informan penelitian"),
    (r"variabel|definisi operasional|operasional variabel", "Variabel & definisi operasional"),
    (r"sumber data|jenis data|jenis dan sumber", "Jenis dan sumber data"),
    (r"instrumen", "Instrumen penelitian"),
    (r"pengumpulan data|teknik pengumpulan|metode pengumpulan|collecting", "Teknik pengumpulan data"),
    (r"keabsahan|validitas data|pengecekan|kredibilitas", "Keabsahan data"),
    (r"uji (validitas|reliabilitas|instrumen)", "Uji validitas & reliabilitas"),
    (r"analisis data|teknik analisis|metode analisis|analisa data|data analysis", "Teknik analisis data"),
    (r"hipotesis", "Hipotesis statistik"),
    (r"prosedur|tahap|langkah|alur|kerangka (penelitian|kerja)", "Prosedur/tahapan penelitian"),
    (r"alat dan bahan|bahan dan alat|^alat|^bahan", "Alat dan bahan"),
    (r"fokus|deskripsi fokus", "Fokus penelitian"),
    (r"perlakuan|rancangan percobaan|desain eksperimen|parameter", "Rancangan percobaan & parameter"),
    (r"model pengembangan|pengembangan (produk|media|sistem)", "Model/prosedur pengembangan"),
    (r"siklus|indikator keberhasilan", "Siklus & indikator keberhasilan (PTK)"),
    (r"kebutuhan|perancangan|desain sistem|use case|flowchart", "Analisis kebutuhan & perancangan sistem"),
    (r"pengujian", "Metode pengujian"),
]
CH4_FAMILY = [
    (r"gambaran umum|deskripsi (lokasi|objek|wilayah|daerah|umum)|profil|sejarah|letak|keadaan (umum|geografis)|"
     r"kondisi (umum|geografis)|visi|struktur organisasi", "Gambaran umum lokasi/objek"),
    (r"karakteristik|identitas|profil (responden|informan)|deskripsi responden", "Karakteristik responden/informan"),
    (r"deskripsi (data|variabel|hasil)|statistik deskriptif|analisis deskriptif|tanggapan responden",
     "Deskripsi variabel/data"),
    (r"uji (validitas|reliabilitas|instrumen)", "Uji instrumen"),
    (r"asumsi|normalitas|multikolinear|heteroskedast|autokorelasi|linearitas|prasyarat|homogenitas",
     "Uji asumsi/prasyarat"),
    (r"hipotesis|uji t|uji f|regresi|koefisien|determinasi|korelasi", "Uji hipotesis/regresi"),
    (r"siklus|pra ?siklus", "Hasil per siklus (PTK)"),
    (r"validasi|uji coba|kelayakan|kepraktisan|keefektifan", "Validasi & uji coba produk"),
    (r"implementasi|pengujian sistem|black ?box|tampilan|antarmuka", "Implementasi & pengujian sistem"),
    (r"analisis (kebutuhan|sistem)|perancangan", "Analisis & perancangan"),
    (r"pendapatan|biaya|penerimaan|kelayakan usaha|r ?/ ?c", "Analisis biaya/pendapatan"),
    (r"pertumbuhan|tinggi tanaman|jumlah daun|bobot|produksi|kadar|aktivitas|hasil uji|rendemen|skrining",
     "Hasil pengamatan/uji (lab/lapangan)"),
    (r"pembahasan|discussion", "Pembahasan"),
    (r"hasil penelitian|hasil analisis|temuan|findings|results?|^hasil", "Hasil penelitian"),
]
TOPICAL = "Subbab topikal (mengikuti rumusan masalah)"
GENERIC_CH4 = {"Uji asumsi/prasyarat", "Uji hipotesis/regresi", "Uji instrumen"}

SYSTEM = ("Kamu dosen pembimbing skripsi S1 di perguruan tinggi Indonesia. Kamu memahami kebiasaan penulisan skripsi "
          "di tiap program studi.")
USER = ("Program studi: {program} ({faculty})\nJudul skripsi: {title}\n\n"
        "Susun judul subbab tingkat pertama BAB III (Metode Penelitian) dan BAB IV (Hasil Penelitian dan Pembahasan) "
        "yang lazim dipakai skripsi di program studi ini untuk judul tersebut. Cukup judul subbab, tanpa isi. "
        "Keluarkan HANYA JSON: {{\"bab3\": [\"3.1 ...\", \"3.2 ...\"], \"bab4\": [\"4.1 ...\", \"4.2 ...\"]}}")


def family(text: str, fams: list) -> str:
    t = re.sub(r"[^a-z/ ]+", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    for rx, name in fams:
        if re.search(rx, t):
            return name
    return ""


def _title(h) -> str:
    if isinstance(h, dict):
        for k in ("judul", "title", "subbab", "sub_bab", "nama", "heading"):
            if isinstance(h.get(k), str):
                return h[k]
        return next((v for v in h.values() if isinstance(v, str)), "")
    return str(h)


def headings(obj, chapter: str) -> List[str]:
    """Sub-section titles of chapter III/IV: a plain list, a nested object, or a ``chapters`` list."""
    if not isinstance(obj, dict):
        return []
    v = obj.get("bab3" if chapter == "III" else "bab4")
    if v is None:
        v = obj.get("BAB III" if chapter == "III" else "BAB IV")
    if isinstance(v, dict):
        v = next((v[k] for k in ("subbab", "sub_bab", "subsections", "bagian") if isinstance(v.get(k), list)), [])
    if not v:
        for ch in obj.get("chapters") or []:
            if isinstance(ch, dict) and str(ch.get("number", "")).strip().upper() == chapter:
                v = ch.get("subsections") or ch.get("subbab") or []
    return [_title(h) for h in v] if isinstance(v, list) else []


def chapter_score(titles: List[str], fams: list, typical: List[Tuple[str, float]], topical: bool) -> dict:
    got: List[str] = []
    for t in titles:
        f = family(t, fams) or (TOPICAL if topical else "")
        if f and f not in got:
            got.append(f)
    weights = {f: float(p) for f, p in typical}
    total = sum(weights.values())
    hit = [f for f in got if f in weights]
    recall = sum(weights[f] for f in hit) / total if total else 0.0
    precision = len(hit) / len(got) if got else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"families": got, "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4)}


class StructureTask(Task):
    name = "structure"
    max_tokens = 768

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        i = item.input
        return SYSTEM, USER.format(program=i["program"], faculty=i["faculty"], title=i["title"])

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        obj = parse_json_loose(record.get("text", ""))
        h3, h4 = headings(obj, "III"), headings(obj, "IV")
        ref = item.reference
        s3 = chapter_score(h3, CH3_FAMILY, ref["bab3"], False)
        s4 = chapter_score(h4, CH4_FAMILY, ref["bab4"], True)
        typical4 = {f for f, _ in ref["bab4"]}
        generic = bool((set(s4["families"]) & GENERIC_CH4) - typical4)
        return {"score": round((s3["f1"] + s4["f1"]) / 2, 4), "ch3": s3, "ch4": s4, "generic_ch4": generic,
                "parsed": bool(h3 or h4), "n_ch3": len(h3), "n_ch4": len(h4)}
