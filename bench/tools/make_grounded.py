"""Build ``bench/items/grounded.jsonl``: literature-review topics, each with six OpenAlex works (with abstracts).

Works are chosen by OpenAlex relevance search (articles 2015–2025 with an abstract, ≥ 20 citations). Items store the
OpenAlex ids only; titles/authors/years/abstracts (clipped at a sentence end, ≤ 1 200 characters) are written to a
private snapshot that the benchmark reads, plus a SHA-256 of each item's sources so a reconstruction can be checked.

  python bench/tools/make_grounded.py --snapshot <private-dir>/sources/openalex.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_agent.bench.tasks.grounded import abstract_text, clip  # noqa: E402
from paper_agent.config import load_env  # noqa: E402
from paper_agent.references import get_json  # noqa: E402

CREATED = "2026-10-14"
N_SOURCES = 6
TOPICS = [
    ("kesehatan", "Dampak penggunaan media sosial terhadap kesehatan mental remaja", "social media use adolescent mental health"),
    ("pendidikan", "Problem-based learning dan kemampuan berpikir kritis peserta didik", "problem-based learning critical thinking"),
    ("ekonomi", "Literasi keuangan dan perilaku keuangan generasi muda", "financial literacy financial behavior young adults"),
    ("teknik_informatika", "Deteksi penyakit tanaman berbasis deep learning", "deep learning plant disease detection"),
    ("kesehatan", "Faktor risiko stunting pada anak balita", "stunting risk factors children under five"),
    ("ekonomi", "Adopsi pembayaran digital oleh usaha mikro dan kecil", "mobile payment adoption small business"),
    ("ekonomi", "Kualitas layanan dan kepuasan nasabah perbankan syariah", "service quality customer satisfaction Islamic banks"),
    ("kesehatan", "Aktivitas antioksidan ekstrak tumbuhan obat", "antioxidant activity plant extract DPPH"),
    ("pertanian", "Probiotik dalam budidaya ikan nila", "probiotics Nile tilapia growth"),
    ("ekonomi", "Kepemimpinan transformasional dan kinerja karyawan", "transformational leadership employee performance"),
    ("pendidikan", "Gamifikasi dalam pendidikan", "gamification education student engagement"),
    ("sosial_politik", "E-government dan kualitas pelayanan publik", "e-government adoption public services"),
    ("teknik_informatika", "Analisis sentimen ulasan daring dengan machine learning", "sentiment analysis online reviews machine learning"),
    ("pertanian", "Simpanan karbon pada sistem agroforestri", "agroforestry carbon sequestration"),
    ("pertanian", "Pupuk organik, kesuburan tanah, dan hasil tanaman", "organic fertilizer soil fertility crop yield"),
    ("teknik_informatika", "Internet of Things untuk pemantauan energi di rumah tangga", "internet of things smart home energy monitoring"),
    ("pendidikan", "Pendidikan karakter di sekolah", "character education students school"),
    ("agama_islam", "Zakat dan pengentasan kemiskinan", "zakat poverty alleviation"),
    ("pendidikan", "Kecemasan matematika dan prestasi belajar", "mathematics anxiety achievement students"),
    ("kesehatan", "Kelelahan kerja (burnout) pada tenaga kesehatan", "burnout healthcare workers"),
    ("teknik_sipil", "Perubahan penggunaan lahan perkotaan dan risiko banjir", "urbanization land use change flood risk"),
    ("pendidikan", "Pembelajaran daring pada masa pandemi COVID-19", "online learning COVID-19 university students"),
    ("ekonomi", "Pemasaran hijau dan niat beli konsumen", "green marketing purchase intention"),
    ("sosial_politik", "Partisipasi masyarakat dalam pembangunan perdesaan", "community participation rural development"),
    ("ekonomi", "Kepatuhan pajak usaha kecil dan menengah", "tax compliance small and medium enterprises"),
    ("pendidikan", "Kecerdasan buatan generatif dalam pendidikan tinggi", "ChatGPT higher education students"),
    ("pendidikan", "Pola asuh orang tua dan perkembangan anak usia dini", "parenting styles early childhood development"),
    ("kesehatan", "Resistensi antibiotik pada bakteri patogen", "antibiotic resistance pathogenic bacteria"),
    ("sosial_politik", "Pariwisata berbasis masyarakat dan keberlanjutan", "community-based tourism sustainability"),
    ("kesehatan", "Pengetahuan dan praktik keamanan pangan penjamah makanan", "food safety knowledge practices food handlers"),
]


def search(query: str, mail: str) -> list:
    params = {"search": query, "per_page": 50,
              "filter": "has_abstract:true,type:article,publication_year:2015-2025,cited_by_count:>20"}
    if mail:
        params["mailto"] = mail
    data = get_json("https://api.openalex.org/works", params=params, ttl=365 * 86400) or {}
    return data.get("results") or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="private JSON snapshot of the sources")
    ap.add_argument("--out", default="bench/items/grounded.jsonl")
    args = ap.parse_args()
    load_env()
    mail = os.getenv("CROSSREF_EMAIL", "")
    snap_path = Path(args.snapshot)
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
    rows = []
    for i, (field, topic, query) in enumerate(TOPICS, 1):
        chosen = []
        for w in search(query, mail):
            ab = abstract_text(w.get("abstract_inverted_index"))
            if len(ab) < 500 or (w.get("language") or "en") not in ("en", "id"):
                continue
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            authors = [((a or {}).get("author") or {}).get("display_name") or "" for a in w.get("authorships") or []]
            snap[wid] = {"id": wid, "title": w.get("display_name") or "", "year": w.get("publication_year"),
                         "authors": [a for a in authors if a][:6], "abstract": clip(ab), "doi": w.get("doi") or "",
                         "language": w.get("language")}
            chosen.append(wid)
            if len(chosen) == N_SOURCES:
                break
        if len(chosen) < N_SOURCES:
            raise SystemExit(f"topic {i} ({query}): only {len(chosen)} usable works")
        src_hash = hashlib.sha256(json.dumps([snap[w] for w in chosen], ensure_ascii=False, sort_keys=True)
                                  .encode("utf-8")).hexdigest()
        rows.append({"id": f"grounded-{i:03d}", "task": "grounded", "privacy": "public",
                     "input": {"topic": topic, "openalex_ids": chosen, "min_words": 250, "max_words": 400},
                     "reference": {"sources_sha256": src_hash},
                     "meta": {"field": field, "query": query, "created": CREATED}})
        print(f"{i:2d} {topic[:60]:60} {chosen}")
        time.sleep(0.2)
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {args.out}; snapshot {len(snap)} works -> {snap_path}")


if __name__ == "__main__":
    main()
