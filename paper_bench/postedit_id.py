"""Manual post-edits of the Indonesian sections, applied after ``paper-agent translate-tex``.

  paper-agent translate-tex --src paper_bench/sections --out paper_bench/sections_id --glossary paper_bench/glossary_id.yaml
  python paper_bench/postedit_id.py

Each edit replaces an exact string of the machine translation: a mistranslation, a term outside the glossary, or a
spelling against EYD. The translation cache keeps unchanged units identical, so the edits apply again after a new
run; an edit whose text the new translation no longer contains is reported (re-read that passage).
"""
from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sections_id"

EDITS = [  # (file, machine translation, post-edit)
    ("abstract", r"dengan sebuah \emph{skripsi}: sebuah skripsi yang bab metode dan bab hasil mengikuti",
     r"dengan \emph{skripsi}: karya tulis akhir yang bab metode dan bab hasilnya mengikuti"),
    ("abstract", "menghasilkan artefak memisahkan mereka.", "yang membedakan mereka adalah kemampuan menghasilkan artefak."),
    ("abstract", "dan melacak pemeriksaan deterministik", "dan selaras dengan pemeriksaan deterministik"),
    ("abstract", "yang disesuaikan untuk Indonesia", "yang disesuaikan untuk bahasa Indonesia"),
    ("abstract", r"(Krippendorff's $\alpha=\AlphaJudges{}$)", r"($\alpha$ Krippendorff $=\AlphaJudges{}$)"),
    ("introduction", r"diakhiri dengan \emph{skripsi}, sebuah skripsi yang bab-babnya",
     r"diakhiri dengan \emph{skripsi}, karya tulis akhir yang bab-babnya"),
    ("introduction", "model bahasa besar (LLMs)", "model bahasa besar (LLM)"),
    ("introduction", "Kami mempersembahkan", "Kami menyajikan"),
    ("introduction", "225 butir publik baru ditulis dan", "225 butir publik yang baru ditulis dan"),
    ("introduction", "paling pada rujukan dan kode dan paling sedikit pada bahasa Indonesia baku",
     "paling jauh pada rujukan dan kode, paling dekat pada bahasa Indonesia baku"),
    ("related", r"\section{Pekerjaan terkait}", r"\section{Penelitian Terkait}"),
    ("related", "Sumber-sumber ini menguji", "Benchmark-benchmark ini menguji"),
    ("benchmark", r"\emph{Publ.} butir publik", r"\emph{Pub.} butir publik"),
    ("benchmark", r"\subsection{Tugas dan item}", r"\subsection{Tugas dan butir}"),
    ("benchmark", "Empat prinsip membimbing benchmark.", "Empat prinsip memandu benchmark ini."),
    ("benchmark", "kit menyiapkan lembar buta untuk memeriksa mereka terhadap penilai manusia",
     "kit menyiapkan lembar penilaian tersamar untuk membandingkan mereka dengan penilai manusia"),
    ("benchmark", "(sebagaimana dilaporkan oleh serving stack)", "(sebagaimana dilaporkan oleh sistem yang melayaninya)"),
    ("benchmark", "field BibTeX", "isian BibTeX"),
    ("benchmark", "bidang BibTeX", "isian BibTeX"),
    ("benchmark", "kunci jawaban metodologi diaudit sebelum model dijalankan oleh peninjau LLM terpisah yang diberi "
                  "instruksi sebagai pemeriksa",
     "sebelum model mana pun dijalankan, kunci jawaban metodologi diaudit oleh peninjau LLM terpisah yang "
     "diinstruksikan berperan sebagai penguji"),
    ("benchmark", r"(mis.\ ``population and sample'', ``data analysis technique'', ``general description of the "
                  r"research site'', ``discussion'')",
     r"(mis.\ ``populasi dan sampel'', ``teknik analisis data'', ``gambaran umum lokasi penelitian'', "
     r"``pembahasan'')"),
    ("benchmark", "Judul-judul baru dan fiktif", "Judul-judulnya baru dan fiktif"),
    ("benchmark", r"Kami juga menandai sebuah bab hasil \emph{generic} yang dibangun di sekitar asumsi, instrumen, atau "
                  r"uji hipotesis dalam sebuah program studi yang skripsinya tidak memilikinya.",
     r"Kami juga menandai bab hasil \emph{generik} yang berpusat pada uji asumsi, uji instrumen, atau uji hipotesis "
     r"pada program studi yang skripsinya tidak memuat uji tersebut."),
    ("benchmark", "analisis bisnis pertanian", "analisis usaha tani"),
    ("benchmark", "rekaman JSON", "catatan JSON"),
    ("benchmark", "seorang kandidat dinilai terhadap mayoritas", "sebuah model kandidat dinilai terhadap mayoritas"),
    ("benchmark", "diberitahu bahwa lebih sedikit lebih baik daripada rujukan karangan",
     "diberi tahu bahwa mencantumkan lebih sedikit lebih baik daripada mengarang"),
    ("benchmark", "Jawaban Crossref selain ``found'' atau ``not found''",
     "Jawaban Crossref selain ``ditemukan'' atau ``tidak ditemukan''"),
    ("benchmark", "membiarkan butir tidak diberi skor daripada dinyatakan salah",
     "membuat butir tidak diberi skor alih-alih dinyatakan salah"),
    ("benchmark", "Skor adalah 1 untuk karya yang ada dengan DOI yang benar",
     "Kredit bernilai 1 untuk karya yang ada dengan DOI yang benar"),
    ("benchmark", "skor dibagi lima, sehingga tidak menjawab memiliki biaya yang sama dengan rujukan",
     "jumlah kredit dibagi lima, sehingga tidak menjawab bernilai sama dengan mencantumkan rujukan"),
    ("benchmark", "sebuah audit mencari setiap kegagalan lagi dengan aturan yang sama",
     "sebuah audit mencari ulang setiap rujukan yang tidak ditemukan dengan aturan yang sama"),
    ("benchmark", "dan mengklasifikasikannya sebagai ditemukan (dengan hanya kecocokan judul nyata) atau tidak ditemukan "
                  "di mana pun.",
     "dan mengklasifikasikannya sebagai ditemukan, hanya cocok dengan judul karya nyata, atau tidak ditemukan di mana "
     "pun."),
    ("benchmark", "beserta bentuk KBBI/EYD mereka", "beserta bentuk bakunya menurut KBBI/EYD"),
    ("benchmark", "Program yang crash, habis waktu atau", "Program yang galat (crash), melewati batas waktu, atau"),
    ("benchmark", r"\paragraph{LaTeX} 8 dokumen dengan 1 hingga 3 kesalahan sisipan",
     r"\paragraph{LaTeX} Delapan dokumen dengan satu hingga tiga kesalahan sisipan"),
    ("benchmark", "; 6 tabel harus", "; enam tabel harus"),
    ("benchmark", "; 6 rujukan APA", "; enam rujukan APA"),
    ("benchmark", "mencantumkan lima referensi nyata", "mencantumkan lima rujukan nyata"),
    ("benchmark", "mengonversi referensi APA ke BibTeX", "mengonversi rujukan APA ke BibTeX"),
    ("benchmark", "dari kutipan skripsi yang dimasker", "dari cuplikan skripsi yang disamarkan"),
    ("setup", r"\section{Pengaturan Eksperimental}", r"\section{Pengaturan Eksperimen}"),
    ("setup", "(Python package 1.0.9,", "(paket Python 1.0.9,"),
    ("setup", r"US\$1.09 per hour", r"US\$1.09 per jam"),
    ("setup", r"diintegrasikan dari \texttt{nvidia-smi} pembacaan setiap", r"diintegrasikan dari pembacaan \texttt{nvidia-smi} setiap"),
    ("setup", "permintaan mengalami time out", "permintaan melewati batas waktu"),
    ("setup", "dan butir prosa mendapat skor sebagai rata-rata dari skor juri deterministik dan terkalibrasi",
     "dan skor butir prosa adalah rata-rata skor deterministiknya dan skor juri yang terkalibrasi"),
    ("setup", "apa yang tidak dapat dinilai oleh penilai", "yang tidak dapat dinilai oleh pemberi skor"),
    ("results", r"\subsection{Para pemimpin tidak berbeda signifikan}", r"\subsection{Model-model teratas tidak berbeda signifikan}"),
    ("results", r"mengurutkan model-model \NModels{} menurut", r"mengurutkan \NModels{} model menurut"),
    ("results", "selisih berpasangan terhadap pemimpin adalah", "selisih berpasangan terhadap model teratas adalah"),
    ("results", "Ketiga pemimpin tersebut oleh karena itu tidak berbeda signifikan, yang tidak membuatnya ekuivalen:",
     "Dengan demikian ketiga model teratas tidak berbeda signifikan, tetapi itu tidak berarti setara:"),
    ("results", "sepuluh model lokal mengikuti setelah jeda", "sepuluh model lokal menyusul setelah selisih yang lebar"),
    ("results", "Pemimpin berdekatan dalam kualitas tetapi tidak dalam harga.",
     "Model-model teratas berdekatan dalam kualitas, tetapi tidak dalam harga."),
    ("results", r"\CheapGap{} di bawah pemimpin, dengan", r"\CheapGap{} di bawah model teratas, dengan"),
    ("results", r"\CostRatio{} kali lebih sedikit daripada", r"\CostRatio{} kali lebih murah daripada"),
    ("results", "Bukan untuk dua tugas yang padat pengetahuan", "Bukan dua tugas yang menuntut pengetahuan"),
    ("results", r"(\LangApiMin{} terendah)", r"(terendah \LangApiMin{})"),
    ("results", "artefak memisahkan mereka (Tabel", "artefak membedakan mereka (Tabel"),
    ("results", "Audit mengkualifikasi kegagalan tersebut.", "Audit memperjelas kegagalan tersebut."),
    ("results", "karya yang skor Crossref-saja lewatkan", "karya yang terlewat oleh skor yang hanya memakai Crossref"),
    ("results", "seperti yang dilakukan penilai akan menaikkan", "seperti yang dilakukan pemberi skor akan menaikkan"),
    ("results", "dan meninggalkan peringkat hampir tidak berubah", "dan membuat peringkat hampir tidak berubah"),
    ("results", "Karena skor menghitung abstensi seperti rujukan", "Karena skor memperlakukan abstensi sama dengan rujukan"),
    ("results", "adalah angka yang harus dibaca untuk keamanan", "adalah angka yang perlu diperhatikan demi keamanan"),
    ("results", "membangun bab~IV di sekitar uji asumsi", "menyusun bab~IV di seputar uji asumsi"),
    ("results", "tidak mengorganisir hasilnya", "tidak menyusun hasilnya"),
    ("results", "Antar rata-rata kelompok, celah terbesar terdapat pada rujukan",
     "Di antara rata-rata kelompok, selisih terbesar terdapat pada rujukan"),
    ("results", r"(\SahLang{} melawan \GemmaFourLang{})", r"(\SahLang{} berbanding \GemmaFourLang{})"),
    ("results", r"dengan penilaian \NJudgeCalls{}.", r"dengan \NJudgeCalls{} penilaian."),
    ("results", r"rata-rata penilaian mereka sesuai dengan $\alpha=\AlphaJudges{}$ Krippendorff (interval)",
     r"rata-rata penilaian mereka sepakat dengan $\alpha$ Krippendorff sebesar \AlphaJudges{} (interval)"),
    ("results", r"\RhoAnsGrd{} (grounded) dan \RhoAnsLng{} (language)",
     r"\RhoAnsGrd{} (berlandaskan sumber) dan \RhoAnsLng{} (bahasa)"),
    ("results", "event penggunaan SDK", "peristiwa penggunaan SDK"),
    ("discussion", "Model API terdepan tidak berbeda signifikan pada benchmark ini",
     "Model-model API teratas tidak berbeda signifikan pada benchmark ini"),
    ("discussion", r"Model baik termurah bukanlah yang memiliki \emph{mini} atau \emph{flash} dalam namanya",
     r"Model bagus yang termurah bukanlah yang namanya memuat \emph{mini} atau \emph{flash}"),
    ("discussion", "biaya terukur per butir lebih menjadi panduan yang baik daripada label tier",
     "biaya terukur per butir merupakan panduan yang lebih baik daripada label tier"),
    ("discussion", r"\paragraph{Rujukan memerlukan pemeriksaan, apa pun modelnya.}",
     r"\paragraph{Rujukan perlu diperiksa, apa pun modelnya}"),
    ("discussion", "dalam dokumen yang diperiksa", "dalam dokumen yang diujikan"),
    ("discussion", "masalah pengambilan, bukan masalah generasi",
     "masalah penelusuran (retrieval), bukan masalah pembangkitan (generation)"),
    ("discussion", "ketika data mungkin tidak boleh keluar", "ketika data tidak boleh keluar"),
    ("discussion", "mereka tidak dapat dipercaya dengan rujukan", "mereka tidak dapat diandalkan untuk rujukan"),
    ("discussion", "sekaligus masih memisahkan model lokal", "tetapi masih membedakan model lokal"),
    ("discussion", r"model tier premium \NPremium{} (diperkirakan sekitar US\$50) dan pemeriksaan usaha tinggi",
     r"\NPremium{} model tier premium (diperkirakan sekitar US\$50) dan pemeriksaan reasoning effort tinggi"),
    ("discussion", "Struktur rujukan dan cuplikan ekstraksi berasal", "Acuan tugas struktur dan cuplikan ekstraksi berasal"),
    ("discussion", "yang untuk seorang kandidat berasal", "yang untuk sebuah model kandidat berasal"),
    ("discussion", "kunci butir belum diperiksa", "kunci jawaban butir belum diperiksa"),
    ("discussion", "menguntungkan frase LLM", "menguntungkan gaya bahasa LLM"),
    ("discussion", "dalam satu stack penyajian", "dalam satu sistem penyaji"),
    ("conclusion", r"Melewati \NModels{} model,", r"Di antara \NModels{} model,"),
]

EVERYWHERE = [  # applied to every file after the edits above
    (r"Krippendorff's $\alpha$", r"$\alpha$ Krippendorff"),
    (r"Spearman's $\rho$", r"$\rho$ Spearman"),
    (r"Cronbach's $\alpha$", r"$\alpha$ Cronbach"),
    ("~October~", "~Oktober~"),
    ("per 100 item", "per 100 butir"),
    ("per item", "per butir"),
    ("per-butir", "per butir"),
    ("lembar buta", "lembar penilaian tersamar"),
    ("pengkodean", "pengodean"),
    ("skoring", "penskoran"),
    ("manapun", "mana pun"),
    ("satupun", "satu pun"),
]


def main() -> int:
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted(OUT.glob("*.tex"))}
    missing = []
    for name, old, new in EDITS:
        if old in texts.get(name, ""):
            texts[name] = texts[name].replace(old, new)
        elif new not in texts.get(name, ""):
            missing.append(f"{name}: {old[:70]}")
    for old, new in EVERYWHERE:
        for name in texts:
            texts[name] = texts[name].replace(old, new)
    for name, text in texts.items():
        (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"post-edits: {len(EDITS) - len(missing)}/{len(EDITS)} applied (+{len(EVERYWHERE)} everywhere)")
    for m in missing:
        print(f"  not found (translation changed? re-read): {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
