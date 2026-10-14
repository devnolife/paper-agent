"""Build ``bench/items/references.jsonl``: research topics (Indonesian) on which a model must list real references.

  python bench/tools/make_references.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CREATED = "2026-10-14"
TOPICS = [
    ("ekonomi", "Pengaruh literasi keuangan terhadap perilaku menabung dan berinvestasi mahasiswa"),
    ("pendidikan", "Model problem-based learning untuk meningkatkan kemampuan berpikir kritis siswa sekolah menengah"),
    ("teknik_informatika", "Deteksi penyakit daun tanaman padi menggunakan convolutional neural network"),
    ("ekonomi", "Kualitas layanan dan kepuasan nasabah pada perbankan syariah"),
    ("kesehatan", "Aktivitas antioksidan ekstrak daun kelor (Moringa oleifera)"),
    ("sosial_politik", "Partisipasi masyarakat dalam perencanaan pembangunan desa"),
    ("kesehatan", "Pengaruh penggunaan media sosial terhadap kesehatan mental remaja"),
    ("sosial_politik", "Adopsi e-government dan kualitas pelayanan publik di negara berkembang"),
    ("teknik_informatika", "Analisis sentimen ulasan aplikasi seluler menggunakan machine learning"),
    ("kesehatan", "Faktor risiko stunting pada anak balita di negara berpendapatan rendah dan menengah"),
    ("ekonomi", "Kepatuhan pajak pelaku usaha mikro, kecil, dan menengah"),
    ("pertanian", "Simpanan karbon pada sistem agroforestri"),
    ("ekonomi", "Technology acceptance model pada adopsi pembayaran digital"),
    ("pertanian", "Pengaruh probiotik terhadap pertumbuhan dan imunitas ikan nila (Oreochromis niloticus)"),
    ("ekonomi", "Kepemimpinan transformasional dan kinerja karyawan"),
    ("pendidikan", "Gamifikasi dalam pembelajaran dan keterlibatan siswa"),
    ("pertanian", "Pengaruh pupuk organik terhadap kesuburan tanah dan hasil tanaman"),
    ("teknik_informatika", "Internet of Things untuk pemantauan konsumsi energi listrik rumah tangga"),
    ("pendidikan", "Pendidikan karakter di sekolah dasar dan menengah"),
    ("agama_islam", "Peran zakat dalam pengentasan kemiskinan"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/items/references.jsonl")
    args = ap.parse_args()
    rows = [{"id": f"references-{i:03d}", "task": "references", "privacy": "public", "input": {"topic": t, "n": 5},
             "reference": {}, "meta": {"field": f, "created": CREATED}} for i, (f, t) in enumerate(TOPICS, 1)]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {args.out}")


if __name__ == "__main__":
    main()
