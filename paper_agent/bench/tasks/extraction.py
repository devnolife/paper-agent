"""extraction — fill a closed-vocabulary methods record from a (masked) thesis excerpt. Items are private.

Reference: silver labels from a panel of strong models of different vendors (field-wise majority), stored outside
the items (``<private>/silver/extraction.json``) as ``{item_id: {"*": record, "<family>": record, ...}}``: a
candidate is scored against the panel record that excludes its own vendor family when one exists, so no model is
graded by itself. Fields the panel disagreed on are absent from the record and not scored (``null`` means the
panel agreed the excerpt does not state it). A human-verified subset
(``<private>/gold/extraction.json``) measures the silver labels' accuracy.

Score: mean over scored fields (enums exact, lists set-F1, sample size exact, hypothesis flag exact).
``evidence_verbatim``: share of the quoted evidence strings found verbatim in the excerpt (fabricated quotes).
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

from ..schema import Item
from . import Context, Task, parse_json_loose

APPROACH = ["kuantitatif", "kualitatif", "campuran", "rnd", "ptk", "eksperimen_lab", "eksperimen_lapangan",
            "eksperimen_kelas", "sistem", "eksperimen_komputasi", "literatur", "normatif", "perancangan_arsitektur",
            "lainnya"]
DATA_KIND = ["primer", "sekunder", "keduanya"]
COLLECTION = ["kuesioner", "wawancara", "observasi", "dokumentasi", "tes", "pengukuran_lab", "pengukuran_lapangan",
              "fgd", "studi_pustaka", "eksperimen_perlakuan", "survei_lapangan_pemetaan", "lainnya"]
ANALYSIS = ["statistik_deskriptif", "uji_validitas_reliabilitas", "uji_asumsi_klasik", "regresi_linier_sederhana",
            "regresi_linier_berganda", "regresi_lain", "korelasi", "uji_t", "anova", "uji_nonparametrik", "chi_square",
            "sem_pls", "analisis_jalur", "n_gain", "miles_huberman", "analisis_tematik", "analisis_isi", "triangulasi",
            "rancangan_acak_ral_rak", "uji_lanjut_bnj_bnt_duncan", "analisis_pendapatan_rc", "analisis_kelayakan_bc_npv",
            "persentase_kategori", "analisis_swot", "analisis_spasial_gis", "pengujian_blackbox_uat", "sus",
            "metrik_ml", "analisis_normatif_hukum", "analisis_kadar_lab", "lainnya"]
TOOLS = ["spss", "excel", "smartpls", "eviews", "amos", "nvivo", "r", "python", "arcgis_qgis", "autocad_sketchup",
         "minitab", "stata", "lainnya", "tidak_disebutkan"]
ENUMS = {"pendekatan": APPROACH, "data_jenis": DATA_KIND}
LISTS = {"teknik_pengumpulan": COLLECTION, "analisis": ANALYSIS, "alat": TOOLS}
FIELDS = ("pendekatan", "data_jenis", "teknik_pengumpulan", "analisis", "alat", "n_sampel", "hipotesis")

SYSTEM = ("Kamu analis metodologi penelitian. Kamu membaca cuplikan skripsi S1 berbahasa Indonesia lalu mengisi satu "
          "objek JSON. Aturan: isi HANYA berdasarkan cuplikan; bila tidak disebutkan pakai null atau []; jangan "
          "mengarang. Untuk bidang berkosakata pakai persis salah satu nilai yang diizinkan.")
USER = """Kosakata yang diizinkan:
- pendekatan (satu): {approach}
- data_jenis (satu): {data}
- teknik_pengumpulan (daftar): {collection}
- analisis (daftar; teknik analisis data yang benar-benar dipakai): {analysis}
- alat (daftar; perangkat lunak analisis): {tools}
- n_sampel: bilangan bulat jumlah sampel/responden/informan yang dianalisis (bukan populasi), atau null
- hipotesis: true bila penelitian merumuskan hipotesis statistik/penelitian, selain itu false
- bukti: kutipan PERSIS (≤ 25 kata, disalin dari cuplikan) yang mendukung pendekatan, data, dan analisis

Keluarkan HANYA JSON:
{{"pendekatan": "...", "data_jenis": "...", "teknik_pengumpulan": ["..."], "analisis": ["..."], "alat": ["..."],
 "n_sampel": null, "hipotesis": false, "bukti": {{"pendekatan": "...", "data": "...", "analisis": "..."}}}}

CUPLIKAN SKRIPSI (nama orang dan identitas sudah disamarkan):
{excerpt}"""


def norm_value(v) -> str:
    s = str(v or "").strip().lower()
    s = s.replace("&", "").replace("-", "_").replace("/", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_")


def normalize(obj) -> Dict[str, object]:
    """Model answer -> record with normalised enum/list values (values outside the vocabulary are dropped)."""
    out: Dict[str, object] = {f: None for f in FIELDS}
    if not isinstance(obj, dict):
        return out
    for f, vocab in ENUMS.items():
        v = norm_value(obj.get(f))
        out[f] = v if v in vocab else None
    for f, vocab in LISTS.items():
        raw = obj.get(f)
        raw = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) and raw else [])
        out[f] = sorted({norm_value(x) for x in raw} & set(vocab))
    n = obj.get("n_sampel")
    if isinstance(n, str):
        m = re.search(r"\d[\d.]*", n)
        n = int(m.group(0).replace(".", "")) if m else None
    out["n_sampel"] = int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0 else None
    h = obj.get("hipotesis")
    out["hipotesis"] = h if isinstance(h, bool) else (str(h).strip().lower() in ("true", "ya", "ada") if h else False)
    return out


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def evidence_verbatim(obj, excerpt: str) -> Optional[float]:
    quotes = [q for q in ((obj or {}).get("bukti") or {}).values() if isinstance(q, str) and q.strip()] \
        if isinstance(obj, dict) and isinstance(obj.get("bukti"), dict) else []
    if not quotes:
        return None
    src = _flat(excerpt)
    ok = sum(1 for q in quotes if _flat(q).strip(" .\"'“”…") in src)
    return ok / len(quotes)


def field_scores(got: Dict[str, object], ref: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for f in FIELDS:
        if f not in ref:           # the panel disagreed on this field: not scored
            continue
        r, g = ref[f], got.get(f)
        if f in LISTS:
            skip = set(ref.get(f + "__undecided") or [])
            rs, gs = set(r or []) - skip, set(g or []) - skip
            if not rs and not gs:
                out[f] = 1.0
            else:
                tp = len(rs & gs)
                out[f] = 2 * tp / (len(rs) + len(gs)) if (rs or gs) else 1.0
        else:
            out[f] = 1.0 if g == r else 0.0
    return out


def reference_for(silver: dict, family: str) -> Optional[dict]:
    if not silver:
        return None
    return silver.get(family) or silver.get("*")


class ExtractionTask(Task):
    name = "extraction"
    max_tokens = 900

    def prompt(self, item: Item, ctx: Context) -> Tuple[str, str]:
        return SYSTEM, USER.format(approach=", ".join(APPROACH), data=", ".join(DATA_KIND),
                                   collection=", ".join(COLLECTION), analysis=", ".join(ANALYSIS),
                                   tools=", ".join(TOOLS), excerpt=item.input["excerpt"])

    def _labels(self, ctx: Context, kind: str) -> dict:
        key = f"extraction:{kind}"
        if key not in ctx.cache:
            path = ctx.private_dir / kind / "extraction.json" if ctx.private_dir else None
            data = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
            ctx.cache[key] = data.get("labels", data) if isinstance(data, dict) else {}
        return ctx.cache[key]

    def reference_fingerprint(self, item: Item, ctx: Context) -> str:
        import hashlib
        labels = [self._labels(ctx, "silver").get(item.id), self._labels(ctx, "gold").get(item.id)]
        return hashlib.sha256(json.dumps(labels, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def score(self, item: Item, record: dict, ctx: Context) -> dict:
        obj = parse_json_loose(record.get("text", ""))
        got = normalize(obj)
        fam = record.get("family") or "*"
        ref = reference_for(self._labels(ctx, "silver").get(item.id, {}), fam)
        res: dict = {"parsed": isinstance(obj, dict), "record": got,
                     "evidence_verbatim": evidence_verbatim(obj, item.input["excerpt"])}
        if ref is None:
            res["score"] = None
            res["note"] = "no silver label yet"
            return res
        fs = field_scores(got, ref)
        res["fields"] = fs
        res["score"] = round(sum(fs.values()) / len(fs), 4) if fs else None
        gold = self._labels(ctx, "gold").get(item.id)
        if gold:
            gfs = field_scores(got, gold)
            res["score_gold"] = round(sum(gfs.values()) / len(gfs), 4) if gfs else None
        return res
