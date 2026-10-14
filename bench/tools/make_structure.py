"""Build ``bench/items/structure.jsonl``: thesis titles (written for this benchmark) per study programme, with the
programme's typical chapter III/IV sub-section families as reference.

The reference comes from an aggregated profile of a local Indonesian undergraduate thesis dataset (one university):
for each programme, the share of theses whose chapter contains a sub-section of each family (≥ 25 %). Only these
aggregates enter the items — no thesis text, names or institution.

  python bench/tools/make_structure.py --profile /path/to/profil-prodi.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CREATED = "2026-10-14"
# programme -> titles (synthetic; places are fictitious). Larger programmes get two titles.
TITLES = {
    "Agribisnis": ["Analisis Pendapatan dan Kelayakan Usahatani Jagung Hibrida di Desa Tanjung Harapan",
                   "Pengaruh Harga dan Kualitas Produk terhadap Keputusan Pembelian Beras Organik pada Konsumen Rumah "
                   "Tangga di Kecamatan Sukamaju"],
    "Agroteknologi": ["Pengaruh Dosis Pupuk Kandang Ayam terhadap Pertumbuhan dan Hasil Tanaman Kacang Hijau "
                      "(Vigna radiata L.)"],
    "Akuntansi": ["Pengaruh Likuiditas dan Ukuran Perusahaan terhadap Profitabilitas Perusahaan Sektor Makanan dan "
                  "Minuman yang Terdaftar di Bursa Efek Indonesia",
                  "Pengaruh Pemahaman Perpajakan dan Sanksi Pajak terhadap Kepatuhan Wajib Pajak UMKM"],
    "Bimbingan dan Konseling Pendidikan Islam": ["Pengaruh Layanan Bimbingan Kelompok terhadap Kepercayaan Diri Santri "
                                                 "di Pondok Pesantren Darul Falah"],
    "Budidaya Perairan": ["Pengaruh Penambahan Probiotik pada Pakan terhadap Pertumbuhan dan Sintasan Benih Ikan Nila "
                          "(Oreochromis niloticus)"],
    "Ekonomi Pembangunan": ["Pengaruh Pertumbuhan Ekonomi dan Tingkat Pengangguran Terbuka terhadap Kemiskinan di "
                            "Kabupaten Sukamaju Periode 2010–2023"],
    "Farmasi": ["Uji Aktivitas Antioksidan Ekstrak Etanol Daun Kelor (Moringa oleifera Lam.) dengan Metode DPPH",
                "Formulasi dan Uji Stabilitas Fisik Sediaan Gel Ekstrak Daun Jambu Biji (Psidium guajava L.)"],
    "Hukum Ekonomi Syariah": ["Tinjauan Hukum Ekonomi Syariah terhadap Praktik Jual Beli Hasil Panen dengan Sistem Ijon "
                              "di Desa Tanjung Harapan"],
    "Hukum Keluarga (Ahwal Syakhshiyah)": ["Tinjauan Hukum Islam terhadap Praktik Nikah Siri di Kecamatan Sukamaju",
                                           "Peran Hakim Mediator dalam Menekan Angka Perceraian di Pengadilan Agama "
                                           "Kelas IB Sukamaju"],
    "Ilmu Administrasi Negara": ["Implementasi Kebijakan Pelayanan Administrasi Kependudukan Berbasis Daring pada Dinas "
                                 "Kependudukan dan Pencatatan Sipil Kabupaten Sukamaju",
                                 "Efektivitas Program Bantuan Sosial Pangan dalam Penanggulangan Kemiskinan di Kelurahan "
                                 "Tanjung Harapan"],
    "Ilmu Komunikasi": ["Strategi Komunikasi Pemasaran UMKM Kuliner melalui Instagram di Era Digital",
                        "Pola Komunikasi Orang Tua dan Anak dalam Mencegah Kecanduan Gawai di Kelurahan Sukamaju"],
    "Ilmu Pemerintahan": ["Partisipasi Masyarakat dalam Musyawarah Perencanaan Pembangunan Desa di Desa Tanjung Harapan"],
    "Informatika": ["Klasifikasi Penyakit Daun Padi Menggunakan Convolutional Neural Network dengan Arsitektur "
                    "MobileNetV2",
                    "Analisis Sentimen Ulasan Pengguna Aplikasi Transportasi Daring Menggunakan Metode Naive Bayes"],
    "Kehutanan": ["Potensi Simpanan Karbon pada Tegakan Hutan Rakyat di Desa Tanjung Harapan",
                  "Kontribusi Hasil Hutan Bukan Kayu terhadap Pendapatan Rumah Tangga Masyarakat Sekitar Hutan"],
    "Komunikasi dan Penyiaran Islam": ["Strategi Dakwah Dai Muda melalui Media Sosial TikTok dalam Menjangkau Generasi Z"],
    "Manajemen": ["Pengaruh Kualitas Pelayanan dan Kepercayaan terhadap Kepuasan Nasabah Bank Syariah di Kota Sukamaju",
                  "Pengaruh Disiplin Kerja dan Motivasi terhadap Kinerja Pegawai pada Kantor Kecamatan Tanjung Harapan"],
    "Pendidikan Agama Islam": ["Peran Guru Pendidikan Agama Islam dalam Membentuk Karakter Religius Peserta Didik di "
                               "SMP Negeri 3 Sukamaju"],
    "Pendidikan Bahasa dan Sastra Indonesia": ["Analisis Penggunaan Gaya Bahasa Kiasan dalam Lirik Lagu Pop Indonesia"],
    "Pendidikan Dokter": ["Hubungan Pola Makan dan Aktivitas Fisik dengan Kejadian Obesitas pada Remaja di SMA Negeri 2 "
                          "Sukamaju"],
    "Pendidikan Guru Sekolah Dasar": ["Pengaruh Model Problem Based Learning terhadap Hasil Belajar IPAS Siswa Kelas V "
                                      "SD Negeri 7 Sukamaju",
                                      "Peningkatan Keterampilan Membaca Pemahaman melalui Model Cooperative Integrated "
                                      "Reading and Composition pada Siswa Kelas IV SD Inpres Tanjung Harapan"],
    "Pendidikan Sosiologi": ["Dampak Pernikahan Usia Dini terhadap Keberlanjutan Pendidikan Remaja Perempuan di Desa "
                             "Tanjung Harapan"],
    "Perencanaan Wilayah dan Kota": ["Analisis Kesesuaian Pemanfaatan Lahan terhadap Rencana Tata Ruang Wilayah di "
                                     "Kawasan Perkotaan Sukamaju"],
    "Teknik Elektro": ["Rancang Bangun Sistem Monitoring Konsumsi Energi Listrik Rumah Tangga Berbasis Internet of "
                       "Things"],
    "Teknik Pengairan": ["Analisis Debit Banjir Rancangan dan Kapasitas Saluran Drainase di Kawasan Perkotaan Sukamaju"],
    "Teknologi Pendidikan": ["Pengaruh Penggunaan Media Video Animasi terhadap Motivasi dan Hasil Belajar IPA Siswa "
                             "Kelas VIII"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="aggregated programme profile JSON")
    ap.add_argument("--out", default="bench/items/structure.jsonl")
    args = ap.parse_args()
    prof = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    rows, n = [], 0
    for program, titles in TITLES.items():
        p = prof["prodi"][program]
        if p.get("data_terbatas") or not p.get("kerangka_bab3") or not p.get("kerangka_bab4"):
            raise SystemExit(f"{program}: profile too thin for a reference")
        faculty = prof["fakultas"][p["fakultas"]]["nama"]
        approach = p.get("pendekatan") or []
        for title in titles:
            n += 1
            rows.append({"id": f"structure-{n:03d}", "task": "structure", "privacy": "public",
                         "input": {"program": program, "faculty": faculty, "title": title},
                         "reference": {"bab3": p["kerangka_bab3"], "bab4": p["kerangka_bab4"], "n_theses": p["n"]},
                         "meta": {"dominant_approach": approach[0][0] if approach else None,
                                  "source": "aggregate profile of a local Indonesian undergraduate thesis dataset "
                                            "(one university)", "created": CREATED}})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {args.out}")


if __name__ == "__main__":
    main()
