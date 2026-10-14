#!/usr/bin/env python3
"""Build the SkripsiBench ``coding`` task: 30 research data-analysis items on synthetic data.

For every task the generator

1. draws a data set with ``numpy.random.default_rng(seed)`` and writes the canonical CSV
   ``bench/data/coding/<id>.csv``;
2. runs the reference solution (the complete program stored in the item) in a subprocess whose working directory is a
   fresh temporary directory holding only the data file under its sandbox name (``data.csv``); the JSON object it
   prints becomes ``reference.expected``, so expected values always come from the canonical file on disk;
3. recomputes the answer independently (manual formulas or a second library) and asserts agreement, asserts that
   plausible mistakes (ddof = 0, Welch instead of Student, unadjusted p-values, ...) fall outside the tolerance, and
   checks that no answer sits on a decision boundary;
4. writes ``bench/items/coding.jsonl``, reads it back, re-runs every stored solution against the stored files and
   scores the output like the harness (last JSON object on stdout, per-key tolerance) — every item must score 1.0.

Outputs are byte-identical across runs. Run from the repository root with the shared research runtime::

    python bench/tools/make_coding.py

Temporary directories follow :mod:`tempfile` (set ``TMPDIR`` to move them). ``SKRIPSIBENCH_PYTHON`` overrides the
interpreter used for the subprocesses (default: the interpreter running this script).
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # single-threaded BLAS: bit-for-bit reproducible results

import bisect
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "bench" / "data" / "coding"
ITEMS_PATH = ROOT / "bench" / "items" / "coding.jsonl"
PYTHON = os.environ.get("SKRIPSIBENCH_PYTHON", sys.executable)
CREATED = "2026-10-14"
SANDBOX_NAME = "data.csv"
DEFAULT_TOL = {"abs": 1e-4, "rel": 1e-3}
CROSS_TOL = {"abs": 1e-8, "rel": 1e-8}
MAX_WORDS, MAX_KEYS, MAX_ROWS, MAX_COLS = 220, 8, 300, 12
FOOTER = ("Jawab dengan satu program Python lengkap yang membaca `data.csv` dari direktori kerja dan mencetak HANYA "
          "satu objek JSON berisi kunci di atas, tanpa teks lain. Angka ditulis mentah: tidak dibulatkan dan bukan "
          "string. "
          "Pustaka yang tersedia hanya pustaka standar, numpy, pandas, scipy, statsmodels dan scikit-learn (tanpa "
          "internet).")


@dataclass
class Task:
    topic: str
    difficulty: str
    libs: list[str]
    data: Callable[[np.random.Generator], pd.DataFrame]
    body: str
    keys: dict[str, str]
    solution: str
    # independent recomputation: [(key, value, method) or (key, value, method, tolerance)]
    cross: Callable[[Path], list]
    tol: dict[str, dict] = field(default_factory=dict)
    # plausible mistakes that must be rejected: [(label, key, wrong value)]
    wrong: Callable[[Path, dict], list] | None = None
    # extra assertions on the data/expected answer (no value on a decision boundary, intended pattern present)
    sanity: Callable[[Path, dict], None] | None = None


REGISTRY: list[Callable[[], Task]] = []


def register(fn: Callable[[], Task]) -> Callable[[], Task]:
    REGISTRY.append(fn)
    return fn


def program(src: str) -> str:
    return src.strip("\n") + "\n"


# ---------------------------------------------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------------------------------------------

def likert(x: np.ndarray, lo: int = 1, hi: int = 5) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi).astype(int)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert len(df) <= MAX_ROWS and df.shape[1] <= MAX_COLS, (path.name, df.shape)
    for col in df.columns:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", col), col
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# independent statistics (used only for cross-checks)
# ---------------------------------------------------------------------------------------------------------------

def ols_np(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """OLS via least squares; ``X`` already holds the constant column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sse = float(resid @ resid)
    se = np.sqrt(np.diag(sse / (n - k) * np.linalg.inv(X.T @ X)))
    t = beta / se
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": 2 * stats.t.sf(np.abs(t), n - k), "r2": r2,
            "adj": 1 - (1 - r2) * (n - 1) / (n - k), "F": (r2 / (k - 1)) / ((1 - r2) / (n - k)), "resid": resid}


def anova1_np(groups: list[np.ndarray]) -> dict[str, float]:
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = len(groups) - 1, allv.size - len(groups)
    f = (ssb / dfb) / (ssw / dfw)
    return {"F": f, "p": stats.f.sf(f, dfb, dfw), "mse": ssw / dfw, "dfb": dfb, "dfw": dfw}


def average_ranks(values: list[float]) -> list[float]:
    """Ranks 1..n with ties sharing their average rank (pure Python)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def mwu_manual(a: list[float], b: list[float], continuity: bool = True) -> dict[str, float]:
    """Mann-Whitney U for ``a`` with the tie-corrected normal approximation."""
    allv = list(a) + list(b)
    r = average_ranks(allv)
    n1, n2 = len(a), len(b)
    big_n = n1 + n2
    u1 = sum(r[:n1]) - n1 * (n1 + 1) / 2
    tie = sum(t ** 3 - t for t in Counter(allv).values())
    sigma = math.sqrt(n1 * n2 / 12 * ((big_n + 1) - tie / (big_n * (big_n - 1))))
    z = (abs(u1 - n1 * n2 / 2) - (0.5 if continuity else 0.0)) / sigma
    return {"u": u1, "p": min(1.0, 2 * stats.norm.sf(z)), "mr_a": sum(r[:n1]) / n1, "mr_b": sum(r[n1:]) / n2}


def shapiro_manual(x: np.ndarray) -> tuple[float, float]:
    """Shapiro-Wilk W and p after Royston (1992, 1995; algorithm AS R94), n >= 12."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    assert n >= 12
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    mm = float(m @ m)
    u = 1 / math.sqrt(n)

    def poly(c: list[float], v: float) -> float:
        return sum(ci * v ** i for i, ci in enumerate(c))

    an = poly([0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056], u) + m[-1] / math.sqrt(mm)
    an1 = poly([0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633], u) + m[-2] / math.sqrt(mm)
    phi = (mm - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * an ** 2 - 2 * an1 ** 2)
    a = m / math.sqrt(phi)
    a[-1], a[-2], a[0], a[1] = an, an1, -an, -an1
    w = float((a @ x) ** 2 / ((x - x.mean()) @ (x - x.mean())))
    ln = math.log(n)
    mu = 0.0038915 * ln ** 3 - 0.083751 * ln ** 2 - 0.31082 * ln - 1.5861
    sigma = math.exp(0.0030302 * ln ** 2 - 0.082676 * ln - 0.4803)
    return w, float(stats.norm.sf((math.log(1 - w) - mu) / sigma))


def tukey_sf_manual(q: float, k: int, df: int) -> float:
    """P(Q > q) of the studentized range by direct double integration (independent of scipy.stats)."""
    def range_cdf(w: float) -> float:
        def f(z: float) -> float:
            return math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * (special.ndtr(z) - special.ndtr(z - w)) ** (k - 1)
        return k * integrate.quad(f, -9, 9, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    logc = (df / 2) * math.log(df) - special.gammaln(df / 2) - (df / 2 - 1) * math.log(2)

    def dens(s: float) -> float:  # density of S = sqrt(chi2_df / df)
        return math.exp(logc + (df - 1) * math.log(s) - df * s * s / 2)

    hi = 1 + 14 / math.sqrt(2 * df)
    total = integrate.quad(dens, 1e-9, hi, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    assert abs(total - 1) < 1e-9, total
    cdf = integrate.quad(lambda s: dens(s) * range_cdf(q * s), 1e-9, hi, epsabs=1e-12, epsrel=1e-11, limit=200)[0]
    return 1 - cdf


def logit_manual(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Unpenalised logistic regression by Newton-Raphson; Wald SE from the inverse Hessian."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ b)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-13:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ybar = float(y.mean())
    ll0 = y.size * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    return {"b": b, "se": se, "p": 2 * stats.norm.sf(np.abs(b / se)), "ll": ll, "ll0": ll0, "prob": p}


def quantile_pos(values: list[float], p: float, pos: Callable[[int, float], float]) -> float:
    v = sorted(values)
    h = min(max(pos(len(v), p), 0.0), len(v) - 1.0)
    lo = math.floor(h)
    return v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def q_linear(values: list[float], p: float) -> float:  # numpy/pandas default, 0-based position (n - 1)p
    return quantile_pos(values, p, lambda n, q: (n - 1) * q)


def q_weibull(values: list[float], p: float) -> float:  # SPSS/Minitab, 1-based position (n + 1)p
    return quantile_pos(values, p, lambda n, q: (n + 1) * q - 1)


def near_int(x: Fraction, eps: Fraction = Fraction(1, 10_000)) -> bool:
    return min(x - math.floor(x), math.ceil(x) - x) <= eps


# ---------------------------------------------------------------------------------------------------------------
# tasks (order = item id; seed = 100 + position)
# ---------------------------------------------------------------------------------------------------------------

@register
def t_deskriptif_likert() -> Task:
    butir = [f"kp{j}" for j in range(1, 7)]
    label = ["sangat tidak puas", "tidak puas", "cukup puas", "puas", "sangat puas"]
    batas = [1.8, 2.6, 3.4, 4.2]

    def data(rng):
        n = 120
        theta = rng.normal(0, 1, n)
        shift = [0.0, -0.2, 0.1, 0.35, -0.1, 0.05]
        cols = {c: likert(3.55 + s + 0.7 * theta + rng.normal(0, 0.7, n)) for c, s in zip(butir, shift)}
        return pd.DataFrame({"id_responden": [f"R{i:03d}" for i in range(1, n + 1)], **cols})

    def skor(path):
        X = read(path)[butir].to_numpy(dtype=float)
        return X, X.sum(axis=1) / 6

    def cross(path):
        X, s = skor(path)
        n, m = s.size, s.sum() / s.size
        srt = np.sort(s)
        return [("rerata", m, "jumlah/n (numpy)"),
                ("sd", math.sqrt(((s - m) ** 2).sum() / (n - 1)), "rumus manual ddof=1"),
                ("median", (srt[(n - 1) // 2] + srt[n // 2]) / 2, "median manual data terurut"),
                ("kategori", label[bisect.bisect_left(batas, m)], "bisect batas interval"),
                ("butir_tertinggi", butir[int(np.argmax(X.mean(axis=0)))], "numpy argmax"),
                ("n_skor_di_atas_3_4", int(np.count_nonzero(s > 3.4)), "numpy count")]

    def wrong(path, exp):
        return [("SD populasi ddof=0", "sd", float(np.std(skor(path)[1])))]

    def sanity(path, exp):
        X, s = skor(path)
        assert min(abs(s.mean() - b) for b in batas) > 0.05
        top = np.sort(X.mean(axis=0))
        assert top[-1] - top[-2] > 0.03

    return Task(
        topic="statistik_deskriptif_likert", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi jawaban 120 nasabah bank atas angket kepuasan: `id_responden` dan enam butir Likert `kp1`–`kp6` (1 = sangat tidak puas … 5 = sangat puas). Skor tiap responden = rata-rata keenam butirnya. Dari skor responden hitung rata-rata, simpangan baku sampel (ddof = 1) dan median. Kategorikan rata-rata skor (tanpa pembulatan) dengan panjang interval (5 − 1)/5 = 0,8: 1,00–1,80 "sangat tidak puas"; >1,80–2,60 "tidak puas"; >2,60–3,40 "cukup puas"; >3,40–4,20 "puas"; >4,20–5,00 "sangat puas". Tentukan juga butir dengan rata-rata tertinggi dan banyak responden yang skornya > 3,40.

Kunci JSON: `rerata`, `sd`, `median` (ketiganya dari skor responden), `kategori` (salah satu label di atas, persis), `butir_tertinggi` (nama kolom, mis. "kp1"), `n_skor_di_atas_3_4` (bilangan bulat).
""",
        keys={"rerata": "rata-rata skor responden", "sd": "simpangan baku sampel skor responden (ddof=1)",
              "median": "median skor responden", "kategori": "kategori interval rata-rata skor",
              "butir_tertinggi": "nama kolom butir dengan rata-rata tertinggi",
              "n_skor_di_atas_3_4": "banyak responden dengan skor > 3,40"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
butir = [f"kp{i}" for i in range(1, 7)]
skor = df[butir].mean(axis=1)
rerata = float(skor.mean())
if rerata <= 1.8:
    kategori = "sangat tidak puas"
elif rerata <= 2.6:
    kategori = "tidak puas"
elif rerata <= 3.4:
    kategori = "cukup puas"
elif rerata <= 4.2:
    kategori = "puas"
else:
    kategori = "sangat puas"
hasil = {
    "rerata": rerata,
    "sd": float(skor.std(ddof=1)),
    "median": float(skor.median()),
    "kategori": kategori,
    "butir_tertinggi": str(df[butir].mean().idxmax()),
    "n_skor_di_atas_3_4": int((skor > 3.4).sum()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_cronbach() -> Task:
    butir = [f"x{j}" for j in range(1, 11)]

    def data(rng):
        n = 40
        theta = rng.normal(0, 1, n)
        loads = [0.75, 0.6, 0.7, 0.55, 0.65, 0.7, 0.05, 0.6, 0.5, 0.7]
        cols = {c: likert(3.3 + ld * theta + rng.normal(0, 0.8, n)) for c, ld in zip(butir, loads)}
        return pd.DataFrame({"id_responden": [f"M{i:02d}" for i in range(1, n + 1)], **cols})

    def alpha_cov(X):
        C = np.cov(X, rowvar=False)
        k = X.shape[1]
        return k / (k - 1) * (1 - np.trace(C) / C.sum()), float(np.trace(C)), float(C.sum())

    def cross(path):
        df = read(path)
        a, sv, tv = alpha_cov(df[butir].to_numpy(float))
        a7 = alpha_cov(df[[b for b in butir if b != "x7"]].to_numpy(float))[0]
        return [("alpha", a, "matriks kovarians: 1 - trace/jumlah"), ("jumlah_varians_butir", sv, "trace kovarians"),
                ("varians_total", tv, "jumlah elemen kovarians"), ("alpha_tanpa_x7", a7, "kovarians 9 butir"),
                ("kesimpulan", "reliabel" if a >= 0.7 else "tidak reliabel", "aturan 0,70")]

    def wrong(path, exp):
        R = np.corrcoef(read(path)[butir].to_numpy(float), rowvar=False)
        rbar = (R.sum() - 10) / 90
        return [("alpha terstandar", "alpha", 10 * rbar / (1 + 9 * rbar))]

    def sanity(path, exp):
        assert abs(exp["alpha"] - 0.70) > 0.02 and exp["alpha_tanpa_x7"] > exp["alpha"] + 0.01

    return Task(
        topic="reliabilitas_cronbach_alpha", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi hasil uji coba angket motivasi belajar pada 40 mahasiswa: `id_responden` dan sepuluh butir Likert `x1`–`x10` (1–5, tidak ada butir terbalik). Hitung Cronbach's alpha: α = k/(k − 1) × (1 − Σs²ᵢ/s²ₜ), dengan k banyak butir, s²ᵢ varians butir ke-i dan s²ₜ varians skor total (jumlah semua butir per responden); semua varians memakai ddof = 1. Hitung pula alpha bila butir `x7` dihapus (rumus sama untuk sembilan butir sisanya). Instrumen "reliabel" bila alpha sepuluh butir ≥ 0,70; selain itu "tidak reliabel".

Kunci JSON: `alpha` (10 butir), `jumlah_varians_butir` (Σs²ᵢ, 10 butir), `varians_total` (s²ₜ, 10 butir), `alpha_tanpa_x7`, `kesimpulan` ("reliabel" atau "tidak reliabel").
""",
        keys={"alpha": "Cronbach's alpha 10 butir", "jumlah_varians_butir": "jumlah varians butir (ddof=1)",
              "varians_total": "varians skor total (ddof=1)", "alpha_tanpa_x7": "alpha bila butir x7 dihapus",
              "kesimpulan": "reliabel bila alpha >= 0,70"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")


def cronbach(items):
    k = items.shape[1]
    jumlah_varians_butir = items.var(ddof=1).sum()
    varians_total = items.sum(axis=1).var(ddof=1)
    alpha = k / (k - 1) * (1 - jumlah_varians_butir / varians_total)
    return float(alpha), float(jumlah_varians_butir), float(varians_total)


butir = [f"x{i}" for i in range(1, 11)]
alpha, jumlah_varians_butir, varians_total = cronbach(df[butir])
alpha_tanpa_x7 = cronbach(df[[b for b in butir if b != "x7"]])[0]
hasil = {
    "alpha": alpha,
    "jumlah_varians_butir": jumlah_varians_butir,
    "varians_total": varians_total,
    "alpha_tanpa_x7": alpha_tanpa_x7,
    "kesimpulan": "reliabel" if alpha >= 0.70 else "tidak reliabel",
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_validitas_butir() -> Task:
    butir = [f"b{j}" for j in range(1, 11)]
    r_tabel = 0.361

    def data(rng):
        n = 30
        theta = rng.normal(0, 1, n)
        loads = [0.8, 0.65, 0.05, 0.7, 0.75, 0.6, 0.1, 0.7, 0.2, 0.8]
        cols = {c: likert(3.4 + ld * theta + rng.normal(0, 0.75, n)) for c, ld in zip(butir, loads)}
        return pd.DataFrame({"id_responden": [f"U{i:02d}" for i in range(1, n + 1)], **cols})

    def cross(path):
        X = read(path)[butir].to_numpy(float)
        tot = X.sum(axis=1)
        r = [float(np.corrcoef(X[:, j], tot)[0, 1]) for j in range(10)]
        return [("r_butir", r, "numpy corrcoef"), ("jumlah_valid", sum(v >= r_tabel for v in r), "hitung manual"),
                ("nomor_tidak_valid", [j + 1 for j, v in enumerate(r) if v < r_tabel], "hitung manual"),
                ("r_maks", max(r), "max")]

    def wrong(path, exp):
        X = read(path)[butir].to_numpy(float)
        tot = X.sum(axis=1)
        rc = [float(np.corrcoef(X[:, j], tot - X[:, j])[0, 1]) for j in range(10)]
        return [("korelasi item-total terkoreksi", "r_butir", rc)]

    def sanity(path, exp):
        assert all(abs(v - r_tabel) > 0.02 for v in exp["r_butir"])
        assert 2 <= len(exp["nomor_tidak_valid"]) <= 4

    return Task(
        topic="validitas_butir_korelasi_item_total", difficulty="medium", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi uji coba angket pada 30 responden: `id_responden` dan butir Likert `b1`–`b10` (1–5). Uji validitas tiap butir dengan korelasi Pearson antara skor butir dan skor total, dengan skor total = jumlah kesepuluh butir TERMASUK butir yang sedang diuji (korelasi item-total tanpa koreksi). Butir valid bila r ≥ r_tabel = 0,361 (n = 30, α = 0,05 dua sisi).

Kunci JSON: `r_butir` (daftar 10 nilai r berurutan b1, b2, …, b10), `jumlah_valid` (bilangan bulat), `nomor_tidak_valid` (daftar nomor butir yang tidak valid sebagai bilangan bulat, urut naik, mis. [2, 5]), `r_maks` (r terbesar).
""",
        keys={"r_butir": "daftar r item-total (tidak dikoreksi) b1..b10",
              "jumlah_valid": "banyak butir dengan r >= 0,361",
              "nomor_tidak_valid": "daftar nomor butir dengan r < 0,361", "r_maks": "r item-total terbesar"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
butir = [f"b{i}" for i in range(1, 11)]
total = df[butir].sum(axis=1)
r = [float(df[b].corr(total)) for b in butir]
r_tabel = 0.361
hasil = {
    "r_butir": r,
    "jumlah_valid": sum(1 for v in r if v >= r_tabel),
    "nomor_tidak_valid": [i + 1 for i, v in enumerate(r) if v < r_tabel],
    "r_maks": max(r),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_korelasi() -> Task:
    def data(rng):
        n = 48
        jam = np.clip(rng.normal(12, 4.5, n), 2, 28).round(1)
        ipk = np.clip(2.55 + 0.045 * jam + rng.normal(0, 0.28, n), 1.8, 4.0).round(2)
        z = (ipk - ipk.mean()) / ipk.std()
        cemas = np.digitize(-0.3 * z + rng.normal(0, 1, n), [-1.3, -0.45, 0.45, 1.3]) + 1
        return pd.DataFrame({"id_mahasiswa": [f"MHS{i:02d}" for i in range(1, n + 1)], "jam_belajar_per_minggu": jam,
                             "ipk": ipk, "tingkat_kecemasan": cemas.astype(int)})

    def cross(path):
        df = read(path)
        n = len(df)
        x, y = df["jam_belajar_per_minggu"].to_numpy(float), df["ipk"].to_numpy(float)
        dx, dy = x - x.mean(), y - y.mean()
        r = float(dx @ dy / math.sqrt((dx @ dx) * (dy @ dy)))
        t = r * math.sqrt((n - 2) / (1 - r * r))
        sp = stats.spearmanr(df["tingkat_kecemasan"], df["ipk"])
        return [("r_pearson", r, "rumus manual"), ("p_pearson", 2 * stats.t.sf(abs(t), n - 2), "uji t manual"),
                ("rho_spearman", sp.statistic, "scipy.stats.spearmanr"),
                ("p_spearman", sp.pvalue, "scipy.stats.spearmanr (aproksimasi t)")]

    def wrong(path, exp):
        return [("p satu sisi", "p_spearman", exp["p_spearman"] / 2)]

    def sanity(path, exp):
        assert exp["p_pearson"] < 0.05 and exp["rho_spearman"] < 0 and 0.001 < exp["p_spearman"] < 0.2

    return Task(
        topic="korelasi_pearson_spearman", difficulty="easy", libs=["pandas", "numpy", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi data 48 mahasiswa: `id_mahasiswa`, `jam_belajar_per_minggu`, `ipk` (0–4) dan `tingkat_kecemasan` (ordinal 1 = sangat rendah … 5 = sangat tinggi). (1) Hitung korelasi Pearson antara `jam_belajar_per_minggu` dan `ipk` beserta p-value dua sisi (uji t, df = n − 2). (2) Hitung korelasi Spearman antara `tingkat_kecemasan` dan `ipk` (peringkat rata-rata untuk nilai kembar) beserta p-value dua sisi dari pendekatan distribusi t: t = ρ·√((n − 2)/(1 − ρ²)), df = n − 2 (bukan uji eksak atau permutasi).

Kunci JSON: `r_pearson`, `p_pearson`, `rho_spearman`, `p_spearman`.
""",
        keys={"r_pearson": "r Pearson jam belajar-IPK", "p_pearson": "p dua sisi r Pearson",
              "rho_spearman": "rho Spearman kecemasan-IPK", "p_spearman": "p dua sisi rho (aproksimasi t, df=n-2)"},
        solution=program('''
import json

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
n = len(df)
r, p_r = stats.pearsonr(df["jam_belajar_per_minggu"], df["ipk"])
rho = df["tingkat_kecemasan"].corr(df["ipk"], method="spearman")
t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
p_rho = 2 * stats.t.sf(abs(t), n - 2)
hasil = {"r_pearson": float(r), "p_pearson": float(p_r), "rho_spearman": float(rho), "p_spearman": float(p_rho)}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_regresi_sederhana() -> Task:
    def data(rng):
        n = 36
        promosi = rng.uniform(4, 25, n).round(2)
        jual = np.rint(140 + 6.2 * promosi + rng.normal(0, 28, n)).astype(int)
        return pd.DataFrame({"periode": [f"{2022 + i // 12}-{i % 12 + 1:02d}" for i in range(n)],
                             "biaya_promosi_juta": promosi, "penjualan_unit": jual})

    def cross(path):
        df = read(path)
        lr = stats.linregress(df["biaya_promosi_juta"], df["penjualan_unit"])
        how = "scipy.stats.linregress"
        return [("intersep", lr.intercept, how), ("koef_promosi", lr.slope, how), ("se_koef_promosi", lr.stderr, how),
                ("t_koef_promosi", lr.slope / lr.stderr, how), ("p_koef_promosi", lr.pvalue, how),
                ("r2", lr.rvalue ** 2, how)]

    def wrong(path, exp):
        n = len(read(path))
        return [("adjusted R²", "r2", 1 - (1 - exp["r2"]) * (n - 1) / (n - 2))]

    return Task(
        topic="regresi_linear_sederhana", difficulty="easy", libs=["pandas", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi data bulanan sebuah UMKM selama 36 bulan: `periode` (tahun-bulan), `biaya_promosi_juta` (juta rupiah) dan `penjualan_unit`. Estimasi regresi linear sederhana dengan OLS (dengan intersep): penjualan_unit = a + b·biaya_promosi_juta + e. Uji signifikansi b dengan uji t dua sisi.

Kunci JSON: `intersep` (a), `koef_promosi` (b), `se_koef_promosi` (galat baku b), `t_koef_promosi` (t hitung b), `p_koef_promosi` (p-value dua sisi b), `r2` (koefisien determinasi R²).
""",
        keys={"intersep": "intersep a", "koef_promosi": "koefisien b", "se_koef_promosi": "galat baku b",
              "t_koef_promosi": "t hitung b", "p_koef_promosi": "p dua sisi b", "r2": "R-squared"},
        solution=program('''
import json

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data.csv")
X = sm.add_constant(df[["biaya_promosi_juta"]])
model = sm.OLS(df["penjualan_unit"], X).fit()
b = "biaya_promosi_juta"
hasil = {
    "intersep": float(model.params["const"]),
    "koef_promosi": float(model.params[b]),
    "se_koef_promosi": float(model.bse[b]),
    "t_koef_promosi": float(model.tvalues[b]),
    "p_koef_promosi": float(model.pvalues[b]),
    "r2": float(model.rsquared),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong)


@register
def t_regresi_berganda() -> Task:
    pred = ["roa", "der", "ukuran_perusahaan"]

    def data(rng):
        n = 100
        roa = np.clip(rng.normal(6.5, 3.5, n), -4, 20).round(2)
        der = np.clip(rng.lognormal(np.log(0.9), 0.5, n), 0.1, 4.5).round(2)
        ukuran = rng.normal(29.6, 1.3, n).round(3)
        pbv = np.clip(1.1 + 0.13 * roa + 0.22 * der + 0.0 * ukuran + rng.normal(0, 0.55, n), 0.15, None).round(2)
        return pd.DataFrame({"kode_emiten": [f"EMT{i:02d}" for i in range(1, 21) for _ in range(5)],
                             "tahun": [y for _ in range(20) for y in range(2019, 2024)],
                             "roa": roa, "der": der, "ukuran_perusahaan": ukuran, "pbv": pbv})

    def fit(path):
        df = read(path)
        X = np.column_stack([np.ones(len(df)), df[pred].to_numpy(float)])
        return ols_np(df["pbv"].to_numpy(float), X)

    def cross(path):
        o = fit(path)
        how = "numpy lstsq + (X'X)^-1"
        return [("konstanta", o["beta"][0], how), ("b_roa", o["beta"][1], how), ("b_der", o["beta"][2], how),
                ("b_ukuran", o["beta"][3], how), ("p_der", o["p"][2], how), ("p_ukuran", o["p"][3], how),
                ("adj_r2", o["adj"], "rumus manual"), ("f_hitung", o["F"], "rumus manual")]

    def wrong(path, exp):
        o = fit(path)
        return [("R² biasa", "adj_r2", o["r2"]), ("p dari normal baku", "p_der", 2 * stats.norm.sf(abs(o["t"][2])))]

    def sanity(path, exp):
        assert exp["p_der"] < 0.05 < exp["p_ukuran"]

    return Task(
        topic="regresi_linear_berganda", difficulty="medium", libs=["pandas", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi data 20 perusahaan manufaktur tahun 2019–2023 (100 baris): `kode_emiten`, `tahun`, `roa` (return on assets, %), `der` (debt to equity ratio, kali), `ukuran_perusahaan` (ln total aset) dan `pbv` (price to book value). Estimasi regresi linear berganda OLS pada data gabungan (pooled), dengan konstanta: pbv = b0 + b1·roa + b2·der + b3·ukuran_perusahaan + e. Gunakan uji t dua sisi untuk tiap koefisien dan uji F simultan.

Kunci JSON: `konstanta` (b0), `b_roa`, `b_der`, `b_ukuran` (b1–b3), `p_der` dan `p_ukuran` (p-value uji t untuk b2 dan b3), `adj_r2` (adjusted R²), `f_hitung` (statistik F).
""",
        keys={"konstanta": "b0", "b_roa": "koefisien roa", "b_der": "koefisien der",
              "b_ukuran": "koefisien ukuran_perusahaan", "p_der": "p dua sisi koefisien der",
              "p_ukuran": "p dua sisi koefisien ukuran_perusahaan", "adj_r2": "adjusted R-squared",
              "f_hitung": "statistik F simultan"},
        solution=program('''
import json

import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data.csv")
model = smf.ols("pbv ~ roa + der + ukuran_perusahaan", data=df).fit()
hasil = {
    "konstanta": float(model.params["Intercept"]),
    "b_roa": float(model.params["roa"]),
    "b_der": float(model.params["der"]),
    "b_ukuran": float(model.params["ukuran_perusahaan"]),
    "p_der": float(model.pvalues["der"]),
    "p_ukuran": float(model.pvalues["ukuran_perusahaan"]),
    "adj_r2": float(model.rsquared_adj),
    "f_hitung": float(model.fvalue),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_asumsi_klasik() -> Task:
    pred = ["inflasi", "bi_rate", "kurs"]

    def data(rng):
        n = 96
        inf = np.empty(n)
        inf[0] = 3.5
        for t in range(1, n):
            inf[t] = 3.5 + 0.85 * (inf[t - 1] - 3.5) + rng.normal(0, 0.35)
        inf = np.clip(inf, 1.3, 6.0).round(2)
        bi = (2.6 + 0.55 * inf + rng.normal(0, 0.35, n)).round(2)
        kurs = (13.2 + 0.018 * np.arange(n) + rng.normal(0, 0.25, n)).round(3)
        e, prev = np.empty(n), 0.0
        for t in range(n):
            prev = 0.45 * prev + rng.normal(0, 70 + 90 * (kurs[t] - 12.8))
            e[t] = prev
        ihsg = (5400 - 95 * inf - 160 * bi + 120 * kurs + e).round(2)
        return pd.DataFrame({"periode": [f"{2016 + i // 12}-{i % 12 + 1:02d}" for i in range(n)], "inflasi": inf,
                             "bi_rate": bi, "kurs": kurs, "ihsg": ihsg})

    def parts(path):
        df = read(path)
        Xp = df[pred].to_numpy(float)
        X = np.column_stack([np.ones(len(df)), Xp])
        return Xp, X, ols_np(df["ihsg"].to_numpy(float), X)["resid"]

    def cross(path):
        Xp, X, e = parts(path)
        n = e.size
        w, p = shapiro_manual(e)
        vif = np.diag(np.linalg.inv(np.corrcoef(Xp, rowvar=False)))
        lm = n * ols_np(e ** 2, X)["r2"]
        how_v = "diagonal invers matriks korelasi"
        return [("sw_w", w, "Royston AS R94 manual", {"abs": 1e-5, "rel": 0.0}),
                ("sw_p", p, "Royston AS R94 manual", {"abs": 1e-5, "rel": 1e-4}),
                ("vif_inflasi", vif[0], how_v), ("vif_bi_rate", vif[1], how_v), ("vif_kurs", vif[2], how_v),
                ("bp_lm", lm, "n·R² regresi bantu (numpy)"), ("bp_p", stats.chi2.sf(lm, 3), "chi2(3)"),
                ("dw", float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2)), "rumus manual")]

    def wrong(path, exp):
        Xp, X, e = parts(path)
        y, others = Xp[:, 2], Xp[:, :2]  # kurs on inflasi, bi_rate without constant: uncentred R²
        ssr = float(((y - others @ np.linalg.lstsq(others, y, rcond=None)[0]) ** 2).sum())
        g = e ** 2 / (e @ e / e.size)
        fitted = X @ np.linalg.lstsq(X, g, rcond=None)[0]
        return [("VIF tanpa konstanta (R² tak terpusat)", "vif_kurs", float(y @ y) / ssr),
                ("Breusch-Pagan asli (non-robust)", "bp_lm", float(((fitted - g.mean()) ** 2).sum() / 2))]

    def sanity(path, exp):
        assert exp["bp_p"] < 0.05 and exp["dw"] < 1.6 and all(1.3 < exp[k] < 10 for k in ("vif_inflasi", "vif_bi_rate"))

    return Task(
        topic="uji_asumsi_klasik_regresi", difficulty="hard", libs=["pandas", "scipy", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi data bulanan Januari 2016–Desember 2023 (96 baris, urut waktu): `periode`, `inflasi` (%), `bi_rate` (%), `kurs` (ribu rupiah per USD) dan `ihsg`. Estimasi OLS dengan konstanta: ihsg = b0 + b1·inflasi + b2·bi_rate + b3·kurs + e, lalu uji asumsi klasik model ini:
1. Normalitas: uji Shapiro–Wilk pada residual.
2. Multikolinearitas: VIF tiap prediktor, VIFⱼ = 1/(1 − R²ⱼ), dengan R²ⱼ dari regresi prediktor j pada dua prediktor lainnya DENGAN konstanta.
3. Heteroskedastisitas: Breusch–Pagan versi Koenker: regresikan kuadrat residual pada konstanta dan ketiga prediktor; LM = n·R² regresi bantu itu, p dari χ² dengan df = 3 (sama dengan `het_breuschpagan(resid, exog)` bawaan statsmodels).
4. Autokorelasi: statistik Durbin–Watson residual menurut urutan baris.

Kunci JSON: `sw_w` dan `sw_p` (statistik W dan p-value Shapiro–Wilk), `vif_inflasi`, `vif_bi_rate`, `vif_kurs`, `bp_lm` dan `bp_p` (statistik LM dan p-value Breusch–Pagan), `dw` (Durbin–Watson).
""",
        keys={"sw_w": "W Shapiro-Wilk residual", "sw_p": "p Shapiro-Wilk residual", "vif_inflasi": "VIF inflasi",
              "vif_bi_rate": "VIF bi_rate", "vif_kurs": "VIF kurs", "bp_lm": "LM Breusch-Pagan (Koenker, n·R²)",
              "bp_p": "p Breusch-Pagan (chi2, df=3)", "dw": "statistik Durbin-Watson"},
        solution=program('''
import json

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

df = pd.read_csv("data.csv")
X = sm.add_constant(df[["inflasi", "bi_rate", "kurs"]])
model = sm.OLS(df["ihsg"], X).fit()
resid = model.resid
w, p_sw = stats.shapiro(resid)
vif = {kolom: variance_inflation_factor(X.values, i) for i, kolom in enumerate(X.columns) if kolom != "const"}
lm, p_lm, _, _ = het_breuschpagan(resid, model.model.exog)
hasil = {
    "sw_w": float(w),
    "sw_p": float(p_sw),
    "vif_inflasi": float(vif["inflasi"]),
    "vif_bi_rate": float(vif["bi_rate"]),
    "vif_kurs": float(vif["kurs"]),
    "bp_lm": float(lm),
    "bp_p": float(p_lm),
    "dw": float(durbin_watson(resid)),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_uji_t_student() -> Task:
    def data(rng):
        a = np.clip(np.rint(rng.normal(79, 6.5, 32)), 0, 100)
        b = np.clip(np.rint(rng.normal(74, 11, 30)), 0, 100)
        kelompok = np.array(["eksperimen"] * 32 + ["kontrol"] * 30)
        nilai = np.concatenate([a, b]).astype(int)
        perm = rng.permutation(62)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 63)], "kelompok": kelompok[perm],
                             "nilai_posttest": nilai[perm]})

    def groups(path):
        df = read(path)
        return [df.loc[df["kelompok"] == g, "nilai_posttest"].to_numpy(float) for g in ("eksperimen", "kontrol")]

    def cross(path):
        a, b = groups(path)
        n1, n2 = a.size, b.size
        sp2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (n1 + n2 - 2)
        t = (a.mean() - b.mean()) / math.sqrt(sp2 * (1 / n1 + 1 / n2))
        return [("t_hitung", t, "rumus pooled manual"), ("df", n1 + n2 - 2, "n1 + n2 - 2"),
                ("p_value", 2 * stats.t.sf(abs(t), n1 + n2 - 2), "distribusi t manual"),
                ("rerata_eksperimen", a.sum() / n1, "jumlah/n"), ("rerata_kontrol", b.sum() / n2, "jumlah/n")]

    def wrong(path, exp):
        w = stats.ttest_ind(*groups(path), equal_var=False)
        return [("uji Welch", "p_value", w.pvalue), ("uji Welch", "t_hitung", w.statistic)]

    def sanity(path, exp):
        assert exp["p_value"] < 0.05

    return Task(
        topic="uji_t_independen_student", difficulty="easy", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi nilai posttest siswa: `id_siswa`, `kelompok` ("eksperimen" atau "kontrol") dan `nilai_posttest`. Lakukan uji t dua sampel independen Student (varians diasumsikan sama, pooled), dua sisi, untuk membandingkan rata-rata kedua kelompok; statistik t dihitung untuk selisih eksperimen − kontrol.

Kunci JSON: `t_hitung`, `df` (derajat bebas, bilangan bulat), `p_value` (dua sisi), `rerata_eksperimen`, `rerata_kontrol`.
""",
        keys={"t_hitung": "t Student (eksperimen - kontrol)", "df": "derajat bebas n1+n2-2",
              "p_value": "p dua sisi", "rerata_eksperimen": "rata-rata kelompok eksperimen",
              "rerata_kontrol": "rata-rata kelompok kontrol"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
a = df.loc[df["kelompok"] == "eksperimen", "nilai_posttest"]
b = df.loc[df["kelompok"] == "kontrol", "nilai_posttest"]
t, p = stats.ttest_ind(a, b, equal_var=True)
hasil = {
    "t_hitung": float(t),
    "df": int(len(a) + len(b) - 2),
    "p_value": float(p),
    "rerata_eksperimen": float(a.mean()),
    "rerata_kontrol": float(b.mean()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_uji_welch() -> Task:
    def data(rng):
        a = np.clip(rng.normal(74, 12.5, 25), 0, 100).round(1)
        b = np.clip(rng.normal(78, 6.5, 40), 0, 100).round(1)
        metode = np.array(["daring"] * 25 + ["luring"] * 40)
        nilai = np.concatenate([a, b])
        perm = rng.permutation(65)
        return pd.DataFrame({"id_mahasiswa": [f"MHS{i:02d}" for i in range(1, 66)], "metode": metode[perm],
                             "nilai_ujian": nilai[perm]})

    def groups(path):
        df = read(path)
        return [df.loc[df["metode"] == g, "nilai_ujian"].to_numpy(float) for g in ("daring", "luring")]

    def cross(path):
        a, b = groups(path)
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        t = (a.mean() - b.mean()) / math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        return [("t_hitung", t, "rumus Welch manual"), ("df_welch", df, "Welch-Satterthwaite manual"),
                ("p_value", 2 * stats.t.sf(abs(t), df), "distribusi t manual"),
                ("selisih_rerata", a.mean() - b.mean(), "numpy")]

    def wrong(path, exp):
        return [("uji t Student", "p_value", stats.ttest_ind(*groups(path), equal_var=True).pvalue)]

    def sanity(path, exp):
        assert exp["p_value"] > 0.05

    return Task(
        topic="uji_t_welch", difficulty="easy", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi nilai ujian mahasiswa dari dua metode perkuliahan: `id_mahasiswa`, `metode` ("daring" atau "luring") dan `nilai_ujian`. Varians kedua kelompok tidak diasumsikan sama, jadi lakukan uji t Welch dua sisi (derajat bebas Welch–Satterthwaite) untuk selisih rata-rata daring − luring.

Kunci JSON: `t_hitung`, `df_welch` (derajat bebas Welch–Satterthwaite, desimal), `p_value` (dua sisi), `selisih_rerata` (rata-rata daring − rata-rata luring).
""",
        keys={"t_hitung": "t Welch (daring - luring)", "df_welch": "derajat bebas Welch-Satterthwaite",
              "p_value": "p dua sisi", "selisih_rerata": "rata-rata daring - rata-rata luring"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
a = df.loc[df["metode"] == "daring", "nilai_ujian"]
b = df.loc[df["metode"] == "luring", "nilai_ujian"]
res = stats.ttest_ind(a, b, equal_var=False)
hasil = {
    "t_hitung": float(res.statistic),
    "df_welch": float(res.df),
    "p_value": float(res.pvalue),
    "selisih_rerata": float(a.mean() - b.mean()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_uji_t_berpasangan() -> Task:
    def data(rng):
        n = 28
        pre = np.clip(np.rint(rng.normal(62, 9, n)), 0, 100)
        post = np.clip(pre + np.rint(rng.normal(5.5, 8, n)), 0, 100)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, n + 1)], "pretest": pre.astype(int),
                             "posttest": post.astype(int)})

    def cross(path):
        df = read(path)
        d = (df["posttest"] - df["pretest"]).to_numpy(float)
        n, m = d.size, d.mean()
        sd = math.sqrt(((d - m) ** 2).sum() / (n - 1))
        t = m / (sd / math.sqrt(n))
        return [("t_hitung", t, "rumus manual"), ("df", n - 1, "n - 1"),
                ("p_value", 2 * stats.t.sf(abs(t), n - 1), "distribusi t manual"),
                ("rerata_selisih", m, "numpy"), ("sd_selisih", sd, "rumus manual ddof=1")]

    def wrong(path, exp):
        df = read(path)
        return [("uji t independen", "p_value", stats.ttest_ind(df["posttest"], df["pretest"]).pvalue)]

    def sanity(path, exp):
        assert exp["p_value"] < 0.05

    return Task(
        topic="uji_t_berpasangan", difficulty="easy", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi skor 28 siswa sebelum dan sesudah pembelajaran dengan media interaktif: `id_siswa`, `pretest`, `posttest`. Lakukan uji t sampel berpasangan (paired t-test) dua sisi pada selisih d = posttest − pretest.

Kunci JSON: `t_hitung`, `df` (bilangan bulat), `p_value` (dua sisi), `rerata_selisih` (rata-rata d), `sd_selisih` (simpangan baku sampel d, ddof = 1).
""",
        keys={"t_hitung": "t berpasangan (posttest - pretest)", "df": "derajat bebas n-1", "p_value": "p dua sisi",
              "rerata_selisih": "rata-rata selisih", "sd_selisih": "simpangan baku sampel selisih"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
d = df["posttest"] - df["pretest"]
t, p = stats.ttest_rel(df["posttest"], df["pretest"])
hasil = {
    "t_hitung": float(t),
    "df": int(len(d) - 1),
    "p_value": float(p),
    "rerata_selisih": float(d.mean()),
    "sd_selisih": float(d.std(ddof=1)),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_mann_whitney() -> Task:
    def data(rng):
        def skor(n, loc):
            theta = rng.normal(0, 1, n)
            return sum(likert(loc + 0.6 * theta + rng.normal(0, 0.7, n)) for _ in range(5))

        a, b = skor(26, 3.95), skor(24, 3.2)
        kelas = np.array(["pbl"] * 26 + ["konvensional"] * 24)
        nilai = np.concatenate([a, b]).astype(int)
        perm = rng.permutation(50)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 51)], "kelas": kelas[perm],
                             "skor_sikap": nilai[perm]})

    def groups(path):
        df = read(path)
        return [df.loc[df["kelas"] == g, "skor_sikap"].tolist() for g in ("pbl", "konvensional")]

    def cross(path):
        m = mwu_manual(*groups(path))
        how = "peringkat manual + aproksimasi normal (ties, kontinuitas)"
        return [("u_pbl", m["u"], how), ("p_value", m["p"], how), ("rerata_peringkat_pbl", m["mr_a"], how),
                ("rerata_peringkat_konvensional", m["mr_b"], how)]

    def wrong(path, exp):
        a, b = groups(path)
        return [("tanpa koreksi kontinuitas", "p_value", mwu_manual(a, b, continuity=False)["p"]),
                ("U kelompok b", "u_pbl", len(a) * len(b) - exp["u_pbl"])]

    def sanity(path, exp):
        a, b = groups(path)
        assert len(set(a + b)) < len(a + b)  # ties present
        assert exp["p_value"] < 0.05

    return Task(
        topic="uji_mann_whitney", difficulty="medium", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi skor sikap ilmiah siswa (data ordinal): `id_siswa`, `kelas` ("pbl" atau "konvensional") dan `skor_sikap`. Lakukan uji Mann–Whitney U dua sisi dengan a = skor kelas "pbl" dan b = skor kelas "konvensional", memakai pendekatan normal dengan koreksi ties dan koreksi kontinuitas, persis semantik `scipy.stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic', use_continuity=True)`. Statistik yang dilaporkan adalah U kelompok a: U = R_a − n_a(n_a + 1)/2, dengan R_a jumlah peringkat kelompok a (peringkat dari gabungan kedua kelompok, peringkat rata-rata untuk nilai kembar).

Kunci JSON: `u_pbl`, `p_value`, `rerata_peringkat_pbl`, `rerata_peringkat_konvensional` (rata-rata peringkat tiap kelas dalam peringkat gabungan).
""",
        keys={"u_pbl": "U untuk kelompok pbl", "p_value": "p dua sisi (normal, koreksi ties & kontinuitas)",
              "rerata_peringkat_pbl": "rata-rata peringkat kelas pbl",
              "rerata_peringkat_konvensional": "rata-rata peringkat kelas konvensional"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
df["peringkat"] = df["skor_sikap"].rank(method="average")
a = df.loc[df["kelas"] == "pbl", "skor_sikap"]
b = df.loc[df["kelas"] == "konvensional", "skor_sikap"]
res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
rerata_peringkat = df.groupby("kelas")["peringkat"].mean()
hasil = {
    "u_pbl": float(res.statistic),
    "p_value": float(res.pvalue),
    "rerata_peringkat_pbl": float(rerata_peringkat["pbl"]),
    "rerata_peringkat_konvensional": float(rerata_peringkat["konvensional"]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_wilcoxon() -> Task:
    def data(rng):
        n = 26
        sebelum = np.rint(rng.normal(152, 12, n))
        d = np.rint(rng.normal(-2.5, 6.5, n))
        d[rng.choice(n, 2, replace=False)] = 0
        return pd.DataFrame({"id_responden": [f"L{i:02d}" for i in range(1, n + 1)],
                             "sebelum": sebelum.astype(int), "sesudah": (sebelum + d).astype(int)})

    def manual(path):
        df = read(path)
        d = [int(s) - int(b) for s, b in zip(df["sesudah"], df["sebelum"]) if s != b]
        r = average_ranks([abs(v) for v in d])
        n = len(d)
        wp = sum(ri for ri, v in zip(r, d) if v > 0)
        wm = sum(ri for ri, v in zip(r, d) if v < 0)
        tie = sum(t ** 3 - t for t in Counter(abs(v) for v in d).values())
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie / 48)
        return df, n, wp, wm, sd

    def cross(path):
        df, n, wp, wm, sd = manual(path)
        z = (wp - n * (n + 1) / 4) / sd
        res = stats.wilcoxon(df["sesudah"], df["sebelum"], zero_method="wilcox", correction=False,
                             method="asymptotic")
        sign = 1.0 if wp > n * (n + 1) / 4 else -1.0
        return [("n_efektif", n, "hitung manual"), ("w_plus", wp, "peringkat manual"),
                ("w_minus", wm, "peringkat manual"), ("z", z, "rumus manual"),
                ("z", sign * abs(res.zstatistic), "scipy.stats.wilcoxon |z|"),
                ("p_value", 2 * stats.norm.sf(abs(z)), "rumus manual"),
                ("p_value", res.pvalue, "scipy.stats.wilcoxon asimtotik")]

    def wrong(path, exp):
        df, n, wp, wm, sd = manual(path)
        zc = (abs(wp - n * (n + 1) / 4) - 0.5) / sd
        return [("dengan koreksi kontinuitas", "p_value", 2 * stats.norm.sf(zc)),
                ("W- dipakai sebagai W+", "w_plus", wm)]

    def sanity(path, exp):
        df, n, wp, wm, sd = manual(path)
        absd = [abs(int(s) - int(b)) for s, b in zip(df["sesudah"], df["sebelum"]) if s != b]
        assert len(df) - n >= 2 and len(set(absd)) < len(absd)  # zeros and ties present
        assert exp["p_value"] < 0.05

    return Task(
        topic="uji_wilcoxon_peringkat_bertanda", difficulty="hard", libs=["pandas", "numpy", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi tekanan darah sistolik (mmHg) 26 lansia sebelum dan sesudah program senam: `id_responden`, `sebelum`, `sesudah`. Lakukan uji peringkat bertanda Wilcoxon dua sisi pada d = sesudah − sebelum dengan aturan: (1) buang pasangan dengan d = 0; n = banyak selisih tak nol; (2) beri peringkat |d| dari yang terkecil, peringkat rata-rata untuk nilai kembar; (3) W+ = jumlah peringkat untuk d > 0, W− = jumlah peringkat untuk d < 0; (4) z = (W+ − n(n + 1)/4) / √(n(n + 1)(2n + 1)/24 − Σ(t³ − t)/48), dengan t ukuran tiap kelompok nilai |d| kembar, tanpa koreksi kontinuitas; (5) p = 2·(1 − Φ(|z|)). Ini setara `scipy.stats.wilcoxon(sesudah, sebelum, zero_method='wilcox', correction=False, method='asymptotic')`.

Kunci JSON: `n_efektif` (n, bilangan bulat), `w_plus`, `w_minus`, `z`, `p_value`.
""",
        keys={"n_efektif": "banyak selisih tak nol", "w_plus": "jumlah peringkat selisih positif",
              "w_minus": "jumlah peringkat selisih negatif", "z": "z dari W+ (koreksi ties, tanpa kontinuitas)",
              "p_value": "p dua sisi normal"},
        solution=program('''
import json

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
d = (df["sesudah"] - df["sebelum"]).to_numpy(dtype=float)
d = d[d != 0]
n = len(d)
peringkat = stats.rankdata(np.abs(d))
w_plus = peringkat[d > 0].sum()
w_minus = peringkat[d < 0].sum()
_, t = np.unique(np.abs(d), return_counts=True)
varians = n * (n + 1) * (2 * n + 1) / 24 - np.sum(t ** 3 - t) / 48
z = (w_plus - n * (n + 1) / 4) / np.sqrt(varians)
hasil = {
    "n_efektif": int(n),
    "w_plus": float(w_plus),
    "w_minus": float(w_minus),
    "z": float(z),
    "p_value": float(2 * stats.norm.sf(abs(z))),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_anova_tukey() -> Task:
    urutan = ["ceramah", "diskusi", "pbl"]
    pasangan = [(0, 1), (0, 2), (1, 2)]

    def data(rng):
        nilai = np.concatenate([np.clip(np.rint(rng.normal(m, 8, 20)), 0, 100) for m in (70, 74, 79)]).astype(int)
        metode = np.repeat(urutan, 20)
        perm = rng.permutation(60)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 61)], "metode": metode[perm],
                             "nilai": nilai[perm]})

    def groups(path):
        df = read(path)
        return df, [df.loc[df["metode"] == m, "nilai"].to_numpy(float) for m in urutan]

    def cross(path):
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        df, g = groups(path)
        a = anova1_np(g)
        pt = []
        for i, j in pasangan:
            q = abs(g[i].mean() - g[j].mean()) / math.sqrt(a["mse"] / 2 * (1 / g[i].size + 1 / g[j].size))
            pt.append(tukey_sf_manual(q, 3, a["dfw"]))
        tk = pairwise_tukeyhsd(df["nilai"], df["metode"])
        # statsmodels orders pairs as np.triu_indices over the sorted group labels
        assert [str(v) for v in tk.groupsunique] == urutan
        assert list(zip(*np.triu_indices(3, 1))) == pasangan
        return [("f_hitung", a["F"], "JK manual"), ("p_value", a["p"], "F manual"), ("df_antar", a["dfb"], "k - 1"),
                ("df_dalam", a["dfw"], "N - k"),
                ("p_tukey", pt, "integrasi numerik studentized range", {"abs": 1e-7, "rel": 1e-6}),
                ("p_tukey", [float(v) for v in tk.pvalues], "statsmodels pairwise_tukeyhsd")]

    def wrong(path, exp):
        df, g = groups(path)
        a = anova1_np(g)
        lsd = []
        for i, j in pasangan:
            t = abs(g[i].mean() - g[j].mean()) / math.sqrt(a["mse"] * (1 / g[i].size + 1 / g[j].size))
            lsd.append(float(2 * stats.t.sf(t, a["dfw"])))
        return [("p tanpa penyesuaian (LSD)", "p_tukey", lsd),
                ("Bonferroni", "p_tukey", [min(1.0, 3 * v) for v in lsd])]

    def sanity(path, exp):
        assert exp["p_value"] < 0.05 and min(exp["p_tukey"]) < 0.05 < max(exp["p_tukey"])

    return Task(
        topic="anova_satu_arah_tukey_hsd", difficulty="medium", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi nilai hasil belajar siswa yang diajar dengan tiga metode: `id_siswa`, `metode` ("ceramah", "diskusi", "pbl") dan `nilai`. Lakukan ANOVA satu arah untuk menguji perbedaan rata-rata nilai antarmetode, lalu uji lanjut Tukey HSD: p-value tiap pasangan disesuaikan memakai distribusi studentized range dengan k = 3 kelompok serta KT galat dan df galat dari ANOVA.

Kunci JSON: `f_hitung`, `p_value` (ANOVA), `df_antar` dan `df_dalam` (bilangan bulat), `p_tukey` (daftar tiga p-value Tukey berurutan: ceramah–diskusi, ceramah–pbl, diskusi–pbl).
""",
        keys={"f_hitung": "F ANOVA satu arah", "p_value": "p ANOVA", "df_antar": "df antarkelompok",
              "df_dalam": "df galat", "p_tukey": "p Tukey HSD: ceramah-diskusi, ceramah-pbl, diskusi-pbl"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
urutan = ["ceramah", "diskusi", "pbl"]
grup = [df.loc[df["metode"] == m, "nilai"] for m in urutan]
f, p = stats.f_oneway(*grup)
tukey = stats.tukey_hsd(*grup)
hasil = {
    "f_hitung": float(f),
    "p_value": float(p),
    "df_antar": len(urutan) - 1,
    "df_dalam": int(len(df) - len(urutan)),
    "p_tukey": [float(tukey.pvalue[0, 1]), float(tukey.pvalue[0, 2]), float(tukey.pvalue[1, 2])],
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_kruskal_wallis() -> Task:
    urutan = ["SMA", "SMK", "MA"]

    def data(rng):
        parts = []
        for asal, n, shift in (("SMA", 30, 0.0), ("SMK", 25, 0.18), ("MA", 20, -0.08)):
            theta = rng.normal(0, 1, n)
            parts.append(sum(likert(3.5 + shift + 0.5 * theta + rng.normal(0, 0.8, n)) for _ in range(10)))
        asal = np.repeat(urutan, [30, 25, 20])
        skor = np.concatenate(parts).astype(int)
        perm = rng.permutation(75)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 76)], "asal_sekolah": asal[perm],
                             "skor_minat": skor[perm]})

    def manual(path):
        df = read(path)
        vals = df["skor_minat"].tolist()
        lab = df["asal_sekolah"].tolist()
        r = average_ranks(vals)
        big_n = len(vals)
        rs = [[ri for ri, g in zip(r, lab) if g == u] for u in urutan]
        h0 = 12 / (big_n * (big_n + 1)) * sum(sum(x) ** 2 / len(x) for x in rs) - 3 * (big_n + 1)
        c = 1 - sum(t ** 3 - t for t in Counter(vals).values()) / (big_n ** 3 - big_n)
        return df, h0, c, [sum(x) / len(x) for x in rs]

    def cross(path):
        df, h0, c, mr = manual(path)
        kw = stats.kruskal(*[df.loc[df["asal_sekolah"] == u, "skor_minat"] for u in urutan])
        return [("h", h0 / c, "peringkat manual + koreksi ties"), ("h", kw.statistic, "scipy.stats.kruskal"),
                ("df", 2, "k - 1"), ("p_value", stats.chi2.sf(h0 / c, 2), "chi2(2)"),
                ("rerata_peringkat", mr, "peringkat manual")]

    def wrong(path, exp):
        df, h0, c, mr = manual(path)
        return [("H tanpa koreksi ties", "h", h0)]

    def sanity(path, exp):
        assert exp["p_value"] > 0.05

    return Task(
        topic="uji_kruskal_wallis", difficulty="medium", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi skor minat berwirausaha siswa (data ordinal) dari tiga asal sekolah: `id_siswa`, `asal_sekolah` ("SMA", "SMK", "MA") dan `skor_minat`. Lakukan uji Kruskal–Wallis: beri peringkat gabungan semua siswa (peringkat rata-rata untuk nilai kembar), hitung H = 12/(N(N + 1))·Σ(Rⱼ²/nⱼ) − 3(N + 1), lalu bagi dengan faktor koreksi ties 1 − Σ(t³ − t)/(N³ − N); p dari distribusi χ² dengan df = k − 1.

Kunci JSON: `h` (H terkoreksi ties), `df` (bilangan bulat), `p_value`, `rerata_peringkat` (daftar rata-rata peringkat berurutan SMA, SMK, MA).
""",
        keys={"h": "H Kruskal-Wallis terkoreksi ties", "df": "derajat bebas k-1", "p_value": "p chi-square",
              "rerata_peringkat": "rata-rata peringkat SMA, SMK, MA"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
urutan = ["SMA", "SMK", "MA"]
h, p = stats.kruskal(*[df.loc[df["asal_sekolah"] == g, "skor_minat"] for g in urutan])
peringkat = df["skor_minat"].rank(method="average")
hasil = {
    "h": float(h),
    "df": len(urutan) - 1,
    "p_value": float(p),
    "rerata_peringkat": [float(peringkat[df["asal_sekolah"] == g].mean()) for g in urutan],
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_chi_square() -> Task:
    baris = ["SD", "SMP", "SMA", "Perguruan Tinggi"]
    kolom = ["kurang", "baik", "lebih"]

    def data(rng):
        n = 180
        pend = rng.choice(baris, n, p=[0.27, 0.31, 0.33, 0.09])
        peluang = {"SD": [0.36, 0.56, 0.08], "SMP": [0.24, 0.66, 0.10], "SMA": [0.15, 0.72, 0.13],
                   "Perguruan Tinggi": [0.07, 0.70, 0.23]}
        gizi = [str(rng.choice(kolom, p=peluang[str(e)])) for e in pend]
        return pd.DataFrame({"id_balita": [f"B{i:03d}" for i in range(1, n + 1)], "pendidikan_ibu": pend,
                             "status_gizi": gizi})

    def manual(path):
        df = read(path)
        cnt = Counter(zip(df["pendidikan_ibu"], df["status_gizi"]))
        O = np.array([[cnt[(r, c)] for c in kolom] for r in baris], dtype=float)
        E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
        return O, E

    def cross(path):
        O, E = manual(path)
        chi2 = float(((O - E) ** 2 / E).sum())
        how = "tabel & frekuensi harapan manual"
        return [("chi2", chi2, how), ("df", 6, "(r-1)(c-1)"), ("p_value", stats.chi2.sf(chi2, 6), "chi2(6)"),
                ("n_sel_harapan_kurang_5", int((E < 5).sum()), how),
                ("persen_sel_harapan_kurang_5", float((E < 5).sum() / 12 * 100), how), ("harapan_min", E.min(), how)]

    def sanity(path, exp):
        O, E = manual(path)
        assert np.abs(E - 5).min() > 0.05 and exp["n_sel_harapan_kurang_5"] >= 2 and exp["p_value"] < 0.05

    return Task(
        topic="uji_chi_square_independensi", difficulty="medium", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi data 180 balita: `id_balita`, `pendidikan_ibu` ("SD", "SMP", "SMA", "Perguruan Tinggi") dan `status_gizi` ("kurang", "baik", "lebih"). Susun tabel kontingensi 4 × 3 frekuensi teramati, lalu lakukan uji chi-square independensi Pearson tanpa koreksi kontinuitas (Yates): frekuensi harapan Eᵢⱼ = (total baris i × total kolom j)/N, χ² = Σ(O − E)²/E, p dari distribusi χ² dengan df = (r − 1)(c − 1).

Kunci JSON: `chi2`, `df` (bilangan bulat), `p_value`, `n_sel_harapan_kurang_5` (banyak sel dengan E < 5), `persen_sel_harapan_kurang_5` (persentase sel tersebut dari 12 sel, skala 0–100), `harapan_min` (E terkecil).
""",
        keys={"chi2": "statistik chi-square Pearson", "df": "derajat bebas", "p_value": "p chi-square",
              "n_sel_harapan_kurang_5": "banyak sel dengan frekuensi harapan < 5",
              "persen_sel_harapan_kurang_5": "persentase sel dengan frekuensi harapan < 5",
              "harapan_min": "frekuensi harapan terkecil"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
tabel = pd.crosstab(df["pendidikan_ibu"], df["status_gizi"])
res = stats.chi2_contingency(tabel, correction=False)
harapan = res.expected_freq
hasil = {
    "chi2": float(res.statistic),
    "df": int(res.dof),
    "p_value": float(res.pvalue),
    "n_sel_harapan_kurang_5": int((harapan < 5).sum()),
    "persen_sel_harapan_kurang_5": float((harapan < 5).mean() * 100),
    "harapan_min": float(harapan.min()),
}
print(json.dumps(hasil))
'''),
        cross=cross, sanity=sanity)


@register
def t_n_gain() -> Task:
    def kat(g):
        return "tinggi" if g >= 0.7 else "sedang" if g >= 0.3 else "rendah"

    def data(rng):
        def kelas(n, pre_mu, g_mu, g_sd):
            pre = np.clip(np.rint(rng.normal(pre_mu, 10, n)), 10, 85)
            post = np.clip(np.rint(pre + rng.normal(g_mu, g_sd, n) * (100 - pre)), 0, 100)
            return pre, post

        pe, qe = kelas(30, 45, 0.56, 0.16)
        pk, qk = kelas(30, 46, 0.2, 0.17)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 61)],
                             "kelas": ["eksperimen"] * 30 + ["kontrol"] * 30,
                             "pretest": np.concatenate([pe, pk]).astype(int),
                             "posttest": np.concatenate([qe, qk]).astype(int)})

    def per_kelas(path):
        df = read(path)
        out = {}
        for k in ("eksperimen", "kontrol"):
            sub = df[df["kelas"] == k]
            out[k] = [(int(q) - int(p)) / (100 - int(p)) for p, q in zip(sub["pretest"], sub["posttest"])]
        return df, out

    def cross(path):
        _, g = per_kelas(path)
        me, mk = sum(g["eksperimen"]) / 30, sum(g["kontrol"]) / 30
        return [("ngain_eksperimen", me, "Python murni"), ("kategori_eksperimen", kat(me), "aturan Hake"),
                ("ngain_kontrol", mk, "Python murni"), ("kategori_kontrol", kat(mk), "aturan Hake"),
                ("n_tinggi_eksperimen", sum(v >= 0.7 for v in g["eksperimen"]), "Python murni")]

    def wrong(path, exp):
        df = read(path)
        sub = df[df["kelas"] == "eksperimen"]
        pre, post = sub["pretest"].mean(), sub["posttest"].mean()
        return [("g dari rata-rata kelas", "ngain_eksperimen", (post - pre) / (100 - pre))]

    def sanity(path, exp):
        _, g = per_kelas(path)
        allg = g["eksperimen"] + g["kontrol"]
        assert all(abs(v - c) > 1e-6 for v in allg for c in (0.3, 0.7)) and min(allg) < 0
        assert all(abs(exp[k] - c) > 0.03 for k in ("ngain_eksperimen", "ngain_kontrol") for c in (0.3, 0.7))
        assert exp["kategori_eksperimen"] != exp["kategori_kontrol"]

    return Task(
        topic="n_gain_hake", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi skor tes siswa (skala 0–100, skor maksimum 100): `id_siswa`, `kelas` ("eksperimen" atau "kontrol"), `pretest` dan `posttest`. Hitung N-Gain (Hake) tiap siswa: g = (posttest − pretest)/(100 − pretest); nilai g negatif tetap dipakai. N-Gain kelas = rata-rata g seluruh siswa di kelas itu (bukan g dari rata-rata pretest dan posttest kelas). Kategorikan N-Gain kelas: g ≥ 0,7 "tinggi"; 0,3 ≤ g < 0,7 "sedang"; g < 0,3 "rendah".

Kunci JSON: `ngain_eksperimen`, `kategori_eksperimen`, `ngain_kontrol`, `kategori_kontrol` (kategori berupa "tinggi", "sedang" atau "rendah"), `n_tinggi_eksperimen` (banyak siswa kelas eksperimen dengan g individu ≥ 0,7, bilangan bulat).
""",
        keys={"ngain_eksperimen": "rata-rata g individu kelas eksperimen", "kategori_eksperimen": "kategori Hake",
              "ngain_kontrol": "rata-rata g individu kelas kontrol", "kategori_kontrol": "kategori Hake",
              "n_tinggi_eksperimen": "banyak siswa eksperimen dengan g >= 0,7"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
df["g"] = (df["posttest"] - df["pretest"]) / (100 - df["pretest"])


def kategori(g):
    if g >= 0.7:
        return "tinggi"
    if g >= 0.3:
        return "sedang"
    return "rendah"


eksperimen = df.loc[df["kelas"] == "eksperimen", "g"]
kontrol = df.loc[df["kelas"] == "kontrol", "g"]
hasil = {
    "ngain_eksperimen": float(eksperimen.mean()),
    "kategori_eksperimen": kategori(eksperimen.mean()),
    "ngain_kontrol": float(kontrol.mean()),
    "kategori_kontrol": kategori(kontrol.mean()),
    "n_tinggi_eksperimen": int((eksperimen >= 0.7).sum()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_ral_bnt() -> Task:
    perlakuan = ["P0", "P1", "P2", "P3", "P4"]

    def data(rng):
        rows = [(p, u, float(np.round(m + rng.normal(0, 2.6), 1)))
                for p, m in zip(perlakuan, (41.5, 44.8, 48.6, 52.3, 50.1)) for u in range(1, 5)]
        return pd.DataFrame(rows, columns=["perlakuan", "ulangan", "tinggi_tanaman_cm"])

    def manual(path):
        df = read(path)
        g = [df.loc[df["perlakuan"] == p, "tinggi_tanaman_cm"].to_numpy(float) for p in perlakuan]
        a = anova1_np(g)
        bnt = stats.t.ppf(0.975, a["dfw"]) * math.sqrt(2 * a["mse"] / 4)
        means = [x.mean() for x in g]
        diffs = [abs(means[i] - means[j]) for i in range(5) for j in range(i + 1, 5)]
        return df, a, bnt, means, diffs

    def cross(path):
        df, a, bnt, means, diffs = manual(path)
        how = "JK manual"
        return [("f_hitung", a["F"], how), ("p_value", a["p"], how), ("kt_galat", a["mse"], how),
                ("bnt_5persen", bnt, "t(0,975; db galat)·√(2·KTG/r)"),
                ("kk_persen", math.sqrt(a["mse"]) / df["tinggi_tanaman_cm"].to_numpy(float).mean() * 100, how),
                ("n_pasangan_berbeda", sum(d > bnt for d in diffs), "10 selisih vs BNT"),
                ("perlakuan_terbaik", perlakuan[int(np.argmax(means))], "argmax")]

    def wrong(path, exp):
        df, a, bnt, means, diffs = manual(path)
        return [("t satu sisi (kuantil 0,95)", "bnt_5persen", stats.t.ppf(0.95, a["dfw"]) * math.sqrt(a["mse"] / 2))]

    def sanity(path, exp):
        df, a, bnt, means, diffs = manual(path)
        srt = sorted(means)
        assert min(abs(d - bnt) for d in diffs) > 0.05 and srt[-1] - srt[-2] > 0.1
        assert exp["p_value"] < 0.05 and 1 <= exp["n_pasangan_berbeda"] <= 9

    return Task(
        topic="ral_anova_bnt", difficulty="medium", libs=["pandas", "numpy", "scipy", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi hasil percobaan Rancangan Acak Lengkap (RAL) dosis pupuk organik cair pada tanaman cabai: `perlakuan` (P0–P4), `ulangan` (1–4) dan `tinggi_tanaman_cm`. Lakukan ANOVA satu faktor RAL (kolom `ulangan` hanya penomoran, bukan faktor). Hitung uji lanjut BNT (LSD) α = 0,05: BNT = t × √(2·KT galat / r), dengan t = kuantil 0,975 distribusi t berderajat bebas db galat dan r = 4 ulangan; dua perlakuan berbeda nyata bila |selisih rata-ratanya| > BNT. Hitung juga koefisien keragaman KK = √(KT galat) / rata-rata umum × 100.

Kunci JSON: `f_hitung`, `p_value`, `kt_galat` (kuadrat tengah galat), `bnt_5persen`, `kk_persen`, `n_pasangan_berbeda` (banyak pasangan berbeda nyata dari 10 pasangan perlakuan, bilangan bulat), `perlakuan_terbaik` (perlakuan dengan rata-rata tertinggi, mis. "P2").
""",
        keys={"f_hitung": "F ANOVA RAL", "p_value": "p ANOVA", "kt_galat": "kuadrat tengah galat",
              "bnt_5persen": "nilai BNT 5%", "kk_persen": "koefisien keragaman (%)",
              "n_pasangan_berbeda": "banyak pasangan dengan selisih > BNT",
              "perlakuan_terbaik": "perlakuan dengan rata-rata tertinggi"},
        solution=program('''
import json
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("data.csv")
model = smf.ols("tinggi_tanaman_cm ~ C(perlakuan)", data=df).fit()
tabel = anova_lm(model)
kt_galat = float(tabel.loc["Residual", "mean_sq"])
db_galat = int(tabel.loc["Residual", "df"])
bnt = stats.t.ppf(0.975, db_galat) * np.sqrt(2 * kt_galat / 4)
rerata = df.groupby("perlakuan")["tinggi_tanaman_cm"].mean()
berbeda = sum(1 for a, b in combinations(rerata.to_numpy(), 2) if abs(a - b) > bnt)
hasil = {
    "f_hitung": float(tabel.loc["C(perlakuan)", "F"]),
    "p_value": float(tabel.loc["C(perlakuan)", "PR(>F)"]),
    "kt_galat": kt_galat,
    "bnt_5persen": float(bnt),
    "kk_persen": float(np.sqrt(kt_galat) / df["tinggi_tanaman_cm"].mean() * 100),
    "n_pasangan_berbeda": int(berbeda),
    "perlakuan_terbaik": str(rerata.idxmax()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_rak() -> Task:
    varietas = ["V1", "V2", "V3", "V4"]

    def data(rng):
        blok = rng.normal(0, 0.1, 5)
        rows = [(v, k + 1, float(np.round(5.3 + ev + blok[k] + rng.normal(0, 0.32), 2)))
                for v, ev in zip(varietas, (0.0, 0.55, 1.05, 0.3)) for k in range(5)]
        return pd.DataFrame(rows, columns=["varietas", "kelompok", "hasil_ton_per_ha"])

    def manual(path):
        df = read(path)
        blok = sorted(set(df["kelompok"]))
        Y = np.zeros((4, len(blok)))
        for v, k, y in zip(df["varietas"], df["kelompok"], df["hasil_ton_per_ha"]):
            Y[varietas.index(v), blok.index(k)] = y
        t, b = Y.shape
        gm = Y.mean()
        sst = ((Y - gm) ** 2).sum()
        ssa = b * ((Y.mean(axis=1) - gm) ** 2).sum()
        ssb = t * ((Y.mean(axis=0) - gm) ** 2).sum()
        dfe = (t - 1) * (b - 1)
        mse = (sst - ssa - ssb) / dfe
        return Y, ssa / (t - 1) / mse, ssb / (b - 1) / mse, mse, dfe

    def cross(path):
        Y, fa, fb, mse, dfe = manual(path)
        how = "JK manual dari matriks perlakuan x blok"
        return [("f_perlakuan", fa, how), ("p_perlakuan", stats.f.sf(fa, 3, dfe), how), ("f_kelompok", fb, how),
                ("p_kelompok", stats.f.sf(fb, 4, dfe), how), ("kt_galat", mse, how), ("db_galat", dfe, "(t-1)(b-1)")]

    def wrong(path, exp):
        Y = manual(path)[0]
        return [("ANOVA satu arah tanpa blok", "f_perlakuan", anova1_np(list(Y))["F"])]

    def sanity(path, exp):
        assert exp["p_perlakuan"] < 0.01 and exp["f_kelompok"] > 1

    return Task(
        topic="rak_anova_tanpa_interaksi", difficulty="medium", libs=["pandas", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi hasil percobaan Rancangan Acak Kelompok (RAK) empat varietas padi pada lima kelompok (blok): `varietas` (V1–V4), `kelompok` (1–5; kode blok, perlakukan sebagai faktor kategorik, bukan angka) dan `hasil_ton_per_ha`. Lakukan ANOVA RAK aditif tanpa interaksi: hasil = μ + varietas + kelompok + galat (seimbang, satu pengamatan per kombinasi).

Kunci JSON: `f_perlakuan` dan `p_perlakuan` (faktor varietas), `f_kelompok` dan `p_kelompok`, `kt_galat` (kuadrat tengah galat), `db_galat` (derajat bebas galat, bilangan bulat).
""",
        keys={"f_perlakuan": "F varietas", "p_perlakuan": "p varietas", "f_kelompok": "F kelompok",
              "p_kelompok": "p kelompok", "kt_galat": "kuadrat tengah galat", "db_galat": "derajat bebas galat"},
        solution=program('''
import json

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("data.csv")
model = smf.ols("hasil_ton_per_ha ~ C(varietas) + C(kelompok)", data=df).fit()
tabel = anova_lm(model, typ=2)
galat = tabel.loc["Residual"]
hasil = {
    "f_perlakuan": float(tabel.loc["C(varietas)", "F"]),
    "p_perlakuan": float(tabel.loc["C(varietas)", "PR(>F)"]),
    "f_kelompok": float(tabel.loc["C(kelompok)", "F"]),
    "p_kelompok": float(tabel.loc["C(kelompok)", "PR(>F)"]),
    "kt_galat": float(galat["sum_sq"] / galat["df"]),
    "db_galat": int(galat["df"]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_anova_dua_arah() -> Task:
    dosis, var = ["D0", "D1", "D2"], ["lokal", "unggul"]

    def data(rng):
        mu = {("D0", "lokal"): 188, ("D0", "unggul"): 172, ("D1", "lokal"): 204, ("D1", "unggul"): 209,
              ("D2", "lokal"): 201, ("D2", "unggul"): 212}
        rows = [(d, v, u, float(np.round(mu[(d, v)] + rng.normal(0, 11), 1))) for d in dosis for v in var
                for u in range(1, 5)]
        return pd.DataFrame(rows, columns=["dosis_pupuk", "varietas", "ulangan", "bobot_buah_g"])

    def cross(path):
        df = read(path)
        Y = np.zeros((3, 2, 4))
        fill = Counter()
        for d, v, y in zip(df["dosis_pupuk"], df["varietas"], df["bobot_buah_g"]):
            i, j = dosis.index(d), var.index(v)
            Y[i, j, fill[(i, j)]] = y
            fill[(i, j)] += 1
        gm = Y.mean()
        ma, mb, mc = Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=2)
        ssa = 2 * 4 * ((ma - gm) ** 2).sum()
        ssb = 3 * 4 * ((mb - gm) ** 2).sum()
        ssab = 4 * ((mc - ma[:, None] - mb[None, :] + gm) ** 2).sum()
        sse = ((Y - mc[:, :, None]) ** 2).sum()
        dfe = 3 * 2 * 3
        mse = sse / dfe
        fa, fb, fab = ssa / 2 / mse, ssb / 1 / mse, ssab / 2 / mse
        how = "JK manual desain seimbang"
        return [("f_dosis", fa, how), ("p_dosis", stats.f.sf(fa, 2, dfe), how), ("f_varietas", fb, how),
                ("p_varietas", stats.f.sf(fb, 1, dfe), how), ("f_interaksi", fab, how),
                ("p_interaksi", stats.f.sf(fab, 2, dfe), how), ("kt_galat", mse, how), ("db_galat", dfe, "ab(r-1)")]

    def wrong(path, exp):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        t3 = anova_lm(smf.ols("bobot_buah_g ~ C(dosis_pupuk) * C(varietas)", data=read(path)).fit(), typ=3)
        return [("tipe III dengan kontras treatment", "f_dosis", float(t3.loc["C(dosis_pupuk)", "F"])),
                ("tipe III dengan kontras treatment", "f_varietas", float(t3.loc["C(varietas)", "F"]))]

    def sanity(path, exp):
        assert exp["p_dosis"] < 0.05 and exp["p_interaksi"] < 0.05 and exp["p_varietas"] > 0.05

    return Task(
        topic="anova_dua_arah_interaksi", difficulty="hard", libs=["pandas", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi percobaan faktorial 3 × 2 dalam RAL dengan 4 ulangan per kombinasi (seimbang): `dosis_pupuk` ("D0", "D1", "D2"), `varietas` ("lokal", "unggul"), `ulangan` dan `bobot_buah_g`. Lakukan ANOVA dua arah dengan interaksi: bobot = μ + dosis + varietas + dosis×varietas + galat (`ulangan` bukan faktor). Gunakan jumlah kuadrat klasik desain seimbang (tipe II; pada data seimbang ini sama dengan tipe I dan dengan tipe III berkontras jumlah-nol).

Kunci JSON: `f_dosis`, `p_dosis`, `f_varietas`, `p_varietas`, `f_interaksi`, `p_interaksi`, `kt_galat` (kuadrat tengah galat), `db_galat` (bilangan bulat).
""",
        keys={"f_dosis": "F dosis pupuk", "p_dosis": "p dosis pupuk", "f_varietas": "F varietas",
              "p_varietas": "p varietas", "f_interaksi": "F interaksi dosis x varietas",
              "p_interaksi": "p interaksi", "kt_galat": "kuadrat tengah galat", "db_galat": "derajat bebas galat"},
        solution=program('''
import json

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("data.csv")
model = smf.ols("bobot_buah_g ~ C(dosis_pupuk) * C(varietas)", data=df).fit()
tabel = anova_lm(model, typ=2)
galat = tabel.loc["Residual"]
hasil = {
    "f_dosis": float(tabel.loc["C(dosis_pupuk)", "F"]),
    "p_dosis": float(tabel.loc["C(dosis_pupuk)", "PR(>F)"]),
    "f_varietas": float(tabel.loc["C(varietas)", "F"]),
    "p_varietas": float(tabel.loc["C(varietas)", "PR(>F)"]),
    "f_interaksi": float(tabel.loc["C(dosis_pupuk):C(varietas)", "F"]),
    "p_interaksi": float(tabel.loc["C(dosis_pupuk):C(varietas)", "PR(>F)"]),
    "kt_galat": float(galat["sum_sq"] / galat["df"]),
    "db_galat": int(galat["df"]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_regresi_logistik() -> Task:
    coef_tol = {"abs": 2e-3, "rel": 2e-3}

    def data(rng):
        n = 240
        while True:  # redraw until no fitted probability lies within 0.003 of the 0.5 cut-off (stable accuracy)
            pend = np.clip(rng.lognormal(np.log(6.0), 0.4, n), 2.0, 18.0).round(1)
            rasio = np.clip(rng.beta(3, 6, n) * 0.9 + 0.05, 0.05, 0.8).round(2)
            kontrak = rng.random(n) < 0.4
            eta = -1.0 - 0.2 * pend + 4.0 * rasio + 0.8 * kontrak
            gagal = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
            X = np.column_stack([np.ones(n), pend, rasio, kontrak.astype(float)])
            if np.abs(logit_manual(X, gagal.astype(float))["prob"] - 0.5).min() > 0.003:
                break
        return pd.DataFrame({"id_debitur": [f"D{i:03d}" for i in range(1, n + 1)], "pendapatan_juta": pend,
                             "rasio_cicilan": rasio, "status_kerja": np.where(kontrak, "kontrak", "tetap"),
                             "gagal_bayar": gagal})

    def design(path):
        df = read(path)
        X = np.column_stack([np.ones(len(df)), df["pendapatan_juta"], df["rasio_cicilan"],
                             (df["status_kerja"] == "kontrak").astype(float)]).astype(float)
        return X, df["gagal_bayar"].to_numpy(float)

    def cross(path):
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        X, y = design(path)
        m = logit_manual(X, y)
        how = "Newton-Raphson numpy"
        sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(X[:, 1:], y)
        how_sk = "sklearn LogisticRegression(C=inf)"
        with warnings.catch_warnings():  # default lbfgs settings, as a typical analyst would run them
            warnings.simplefilter("ignore", ConvergenceWarning)
            sk0 = LogisticRegression(C=np.inf).fit(X[:, 1:], y)
        how_sk0 = "sklearn LogisticRegression(C=inf), solver bawaan"
        default_fit = [(k, float(v), how_sk0, coef_tol) for k, v in
                       zip(("konstanta", "b_pendapatan", "b_rasio", "b_kontrak"), [sk0.intercept_[0], *sk0.coef_[0]])]
        return default_fit + [
                ("konstanta", m["b"][0], how), ("b_pendapatan", m["b"][1], how), ("b_rasio", m["b"][2], how),
                ("b_kontrak", m["b"][3], how), ("or_kontrak", math.exp(m["b"][3]), how),
                ("p_rasio", m["p"][2], "Wald dari invers Hessian"), ("mcfadden_r2", 1 - m["ll"] / m["ll0"], how),
                ("akurasi", float(((m["prob"] >= 0.5) == (y == 1)).mean()), how),
                ("konstanta", float(sk.intercept_[0]), how_sk, coef_tol),
                ("b_pendapatan", float(sk.coef_[0][0]), how_sk, coef_tol),
                ("b_rasio", float(sk.coef_[0][1]), how_sk, coef_tol),
                ("b_kontrak", float(sk.coef_[0][2]), how_sk, coef_tol)]

    def wrong(path, exp):
        from sklearn.linear_model import LogisticRegression
        X, y = design(path)
        sk = LogisticRegression(max_iter=10_000).fit(X[:, 1:], y)
        return [("sklearn bawaan (L2, C=1)", "b_rasio", float(sk.coef_[0][1])),
                ("sklearn bawaan (L2, C=1)", "b_kontrak", float(sk.coef_[0][2]))]

    def sanity(path, exp):
        X, y = design(path)
        prob = logit_manual(X, y)["prob"]
        assert np.abs(prob - 0.5).min() > 0.003 and 0.2 < y.mean() < 0.5
        assert exp["p_rasio"] < 0.05

    return Task(
        topic="regresi_logistik_biner", difficulty="hard", libs=["pandas", "numpy", "statsmodels"], data=data,
        body="""
Berkas `data.csv` berisi data 240 debitur koperasi: `id_debitur`, `pendapatan_juta` (pendapatan per bulan, juta rupiah), `rasio_cicilan` (cicilan/pendapatan), `status_kerja` ("tetap" atau "kontrak") dan `gagal_bayar` (1 = gagal bayar, 0 = lancar). Estimasi regresi logistik biner dengan maximum likelihood TANPA regularisasi, dengan konstanta: logit P(gagal_bayar = 1) = b0 + b1·pendapatan_juta + b2·rasio_cicilan + b3·kontrak, dengan kontrak = 1 untuk "kontrak" dan 0 untuk "tetap" (kategori acuan). p-value dari uji Wald z dua sisi. McFadden R² = 1 − LL_model/LL_nol (LL_nol: model konstanta saja). Akurasi: prediksi 1 bila peluang ≥ 0,5, selain itu 0.

Kunci JSON: `konstanta` (b0), `b_pendapatan`, `b_rasio`, `b_kontrak`, `or_kontrak` (odds ratio = e^b3), `p_rasio` (p-value b2), `mcfadden_r2`, `akurasi` (proporsi 0–1).
""",
        keys={"konstanta": "b0", "b_pendapatan": "koefisien pendapatan_juta", "b_rasio": "koefisien rasio_cicilan",
              "b_kontrak": "koefisien dummy kontrak (acuan: tetap)", "or_kontrak": "odds ratio kontrak = exp(b3)",
              "p_rasio": "p Wald dua sisi b_rasio", "mcfadden_r2": "pseudo R-squared McFadden",
              "akurasi": "akurasi klasifikasi cut-off 0,5"},
        tol={k: coef_tol for k in ("konstanta", "b_pendapatan", "b_rasio", "b_kontrak", "or_kontrak")}
        | {"p_rasio": {"abs": 2e-4, "rel": 5e-3}},
        solution=program('''
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data.csv")
df["kontrak"] = (df["status_kerja"] == "kontrak").astype(int)
X = sm.add_constant(df[["pendapatan_juta", "rasio_cicilan", "kontrak"]].astype(float))
model = sm.Logit(df["gagal_bayar"], X).fit(disp=0)
prediksi = (model.predict(X) >= 0.5).astype(int)
hasil = {
    "konstanta": float(model.params["const"]),
    "b_pendapatan": float(model.params["pendapatan_juta"]),
    "b_rasio": float(model.params["rasio_cicilan"]),
    "b_kontrak": float(model.params["kontrak"]),
    "or_kontrak": float(np.exp(model.params["kontrak"])),
    "p_rasio": float(model.pvalues["rasio_cicilan"]),
    "mcfadden_r2": float(model.prsquared),
    "akurasi": float((prediksi == df["gagal_bayar"]).mean()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_metrik_klasifikasi() -> Task:
    kelas = ["negatif", "netral", "positif"]

    def data(rng):
        peluang = {"negatif": [0.78, 0.12, 0.10], "netral": [0.22, 0.45, 0.33], "positif": [0.06, 0.10, 0.84]}
        y_true = np.repeat(kelas, [48, 32, 70])
        y_pred = np.array([kelas[int(rng.choice(3, p=peluang[str(t)]))] for t in y_true])
        perm = rng.permutation(y_true.size)
        return pd.DataFrame({"id_ulasan": [f"U{i:03d}" for i in range(1, y_true.size + 1)], "y_true": y_true[perm],
                             "y_pred": y_pred[perm]})

    def per_kelas(path):
        df = read(path)
        pairs = list(zip(df["y_true"], df["y_pred"]))
        out = {}
        for c in kelas:
            tp = sum(t == c and p == c for t, p in pairs)
            prec = tp / sum(p == c for _, p in pairs)
            rec = tp / sum(t == c for t, _ in pairs)
            out[c] = (prec, rec, 2 * prec * rec / (prec + rec))
        return pairs, out

    def cross(path):
        pairs, pk = per_kelas(path)
        how = "penghitungan manual"
        return [("akurasi", sum(t == p for t, p in pairs) / len(pairs), how),
                ("presisi_makro", sum(v[0] for v in pk.values()) / 3, how),
                ("recall_makro", sum(v[1] for v in pk.values()) / 3, how),
                ("f1_makro", sum(v[2] for v in pk.values()) / 3, how), ("f1_netral", pk["netral"][2], how),
                ("cm_netral_positif", sum(t == "netral" and p == "positif" for t, p in pairs), how)]

    def wrong(path, exp):
        from sklearn.metrics import f1_score
        df = read(path)
        p, r = exp["presisi_makro"], exp["recall_makro"]
        return [("F1 dari presisi & recall makro", "f1_makro", 2 * p * r / (p + r)),
                ("F1 berbobot", "f1_makro", f1_score(df["y_true"], df["y_pred"], average="weighted"))]

    return Task(
        topic="metrik_klasifikasi_multikelas", difficulty="medium", libs=["pandas", "scikit-learn"], data=data,
        body="""
Berkas `data.csv` berisi hasil klasifikasi sentimen ulasan aplikasi oleh model Naive Bayes: `id_ulasan`, `y_true` (label sebenarnya) dan `y_pred` (label prediksi), dengan kelas "negatif", "netral" dan "positif". Hitung akurasi serta presisi, recall dan F1 tiap kelas, lalu presisi makro, recall makro dan F1 makro sebagai rata-rata aritmetika tanpa bobot dari nilai ketiga kelas (F1 makro = rata-rata F1 per kelas, bukan F1 dari presisi makro dan recall makro).

Kunci JSON: `akurasi`, `presisi_makro`, `recall_makro`, `f1_makro`, `f1_netral` (F1 kelas netral) — semuanya proporsi 0–1 — dan `cm_netral_positif` (banyak ulasan berlabel sebenarnya "netral" yang diprediksi "positif", bilangan bulat).
""",
        keys={"akurasi": "akurasi", "presisi_makro": "presisi makro", "recall_makro": "recall makro",
              "f1_makro": "rata-rata F1 per kelas", "f1_netral": "F1 kelas netral",
              "cm_netral_positif": "sel matriks konfusi: sebenarnya netral, diprediksi positif"},
        solution=program('''
import json

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

df = pd.read_csv("data.csv")
y_true, y_pred = df["y_true"], df["y_pred"]
kelas = ["negatif", "netral", "positif"]
cm = confusion_matrix(y_true, y_pred, labels=kelas)
hasil = {
    "akurasi": float(accuracy_score(y_true, y_pred)),
    "presisi_makro": float(precision_score(y_true, y_pred, labels=kelas, average="macro")),
    "recall_makro": float(recall_score(y_true, y_pred, labels=kelas, average="macro")),
    "f1_makro": float(f1_score(y_true, y_pred, labels=kelas, average="macro")),
    "f1_netral": float(f1_score(y_true, y_pred, labels=["netral"], average=None)[0]),
    "cm_netral_positif": int(cm[kelas.index("netral"), kelas.index("positif")]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong)


@register
def t_sus() -> Task:
    def data(rng):
        n = 40
        u = rng.normal(0, 1, n)
        cols = {f"q{i}": likert(3.85 + 0.6 * u + rng.normal(0, 0.7, n)) if i % 2 else
                likert(2.2 - 0.6 * u + rng.normal(0, 0.7, n)) for i in range(1, 11)}
        return pd.DataFrame({"id_responden": [f"R{i:02d}" for i in range(1, n + 1)], **cols})

    def skor(path, balik=True):
        df = read(path)
        out = []
        for _, row in df.iterrows():
            tot = sum((row[f"q{i}"] - 1) if (i % 2 or not balik) else (5 - row[f"q{i}"]) for i in range(1, 11))
            out.append(float(tot) * 2.5)
        return out

    def cross(path):
        s = skor(path)
        n, m = len(s), sum(s) / len(s)
        srt = sorted(s)
        how = "aturan Brooke, Python murni"
        return [("rerata_sus", m, how), ("sd_sus", math.sqrt(sum((v - m) ** 2 for v in s) / (n - 1)), how),
                ("median_sus", (srt[(n - 1) // 2] + srt[n // 2]) / 2, how),
                ("n_minimal_68", sum(v >= 68 for v in s), how), ("sus_min", min(s), how), ("sus_max", max(s), how)]

    def wrong(path, exp):
        s = skor(path, balik=False)
        return [("butir genap tidak dibalik", "rerata_sus", sum(s) / len(s))]

    def sanity(path, exp):
        assert 5 <= exp["n_minimal_68"] <= 35 and 60 < exp["rerata_sus"] < 80

    return Task(
        topic="skor_system_usability_scale", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi jawaban 40 responden uji usabilitas aplikasi dengan System Usability Scale (SUS): `id_responden` dan `q1`–`q10` (skala 1–5, urutan butir standar SUS). Hitung skor SUS tiap responden menurut aturan Brooke: kontribusi butir ganjil (q1, q3, q5, q7, q9) = skor − 1, kontribusi butir genap (q2, q4, q6, q8, q10) = 5 − skor; skor SUS = jumlah kontribusi × 2,5 (rentang 0–100).

Kunci JSON: `rerata_sus`, `sd_sus` (simpangan baku sampel, ddof = 1), `median_sus`, `n_minimal_68` (banyak responden dengan SUS ≥ 68, bilangan bulat), `sus_min`, `sus_max`.
""",
        keys={"rerata_sus": "rata-rata skor SUS", "sd_sus": "simpangan baku sampel skor SUS",
              "median_sus": "median skor SUS", "n_minimal_68": "banyak responden dengan SUS >= 68",
              "sus_min": "skor SUS terendah", "sus_max": "skor SUS tertinggi"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
ganjil = [f"q{i}" for i in (1, 3, 5, 7, 9)]
genap = [f"q{i}" for i in (2, 4, 6, 8, 10)]
sus = ((df[ganjil] - 1).sum(axis=1) + (5 - df[genap]).sum(axis=1)) * 2.5
hasil = {
    "rerata_sus": float(sus.mean()),
    "sd_sus": float(sus.std(ddof=1)),
    "median_sus": float(sus.median()),
    "n_minimal_68": int((sus >= 68).sum()),
    "sus_min": float(sus.min()),
    "sus_max": float(sus.max()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_usahatani() -> Task:
    variabel = ["biaya_benih", "biaya_pupuk", "biaya_pestisida", "biaya_tenaga_kerja"]
    tetap = ["penyusutan_alat", "pajak_lahan"]

    def data(rng):
        n = 40
        luas = rng.uniform(0.25, 2.0, n).round(2)
        faktor = np.ones(n)
        faktor[rng.choice(n, 4, replace=False)] = rng.uniform(0.3, 0.45, 4)
        produksi = np.rint(luas * rng.normal(5600, 650, n) * faktor).astype(int)
        harga = (np.rint(rng.uniform(5300, 6300, n) / 50) * 50).astype(int)

        def biaya(mu, sd):
            return (np.rint(luas * rng.normal(mu, sd, n) / 1000) * 1000).astype(int)

        cols = {"biaya_benih": biaya(450_000, 50_000), "biaya_pupuk": biaya(2_400_000, 300_000),
                "biaya_pestisida": biaya(950_000, 200_000), "biaya_tenaga_kerja": biaya(9_800_000, 1_100_000),
                "penyusutan_alat": (np.rint(rng.uniform(250_000, 600_000, n) / 1000) * 1000).astype(int),
                "pajak_lahan": (np.rint(luas * rng.uniform(120_000, 200_000, n) / 1000) * 1000).astype(int)}
        return pd.DataFrame({"id_petani": [f"P{i:02d}" for i in range(1, n + 1)], "luas_lahan_ha": luas,
                             "produksi_kg": produksi, "harga_jual_per_kg": harga, **cols})

    def manual(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        tr = [int(r["produksi_kg"]) * int(r["harga_jual_per_kg"]) for r in rows]
        tc = [sum(int(r[c]) for c in variabel + tetap) for r in rows]
        return rows, tr, tc

    def cross(path):
        rows, tr, tc = manual(path)
        n = len(rows)
        how = "aritmetika bilangan bulat (modul csv)"
        luas = sum(Fraction(r["luas_lahan_ha"]) for r in rows)
        return [("rerata_penerimaan", sum(tr) / n, how), ("rerata_total_biaya", sum(tc) / n, how),
                ("rerata_pendapatan", (sum(tr) - sum(tc)) / n, how), ("rc_ratio", sum(tr) / sum(tc), how),
                ("n_petani_rc_di_atas_1", sum(a > b for a, b in zip(tr, tc)), how),
                ("bep_harga", sum(tc) / sum(int(r["produksi_kg"]) for r in rows), how),
                ("pendapatan_per_ha", float((sum(tr) - sum(tc)) / luas), "pecahan eksak")]

    def wrong(path, exp):
        rows, tr, tc = manual(path)
        return [("rata-rata R/C individu", "rc_ratio", sum(a / b for a, b in zip(tr, tc)) / len(tr))]

    def sanity(path, exp):
        rows, tr, tc = manual(path)
        assert all(abs(a / b - 1) > 1e-3 for a, b in zip(tr, tc)) and 2 <= len(tr) - exp["n_petani_rc_di_atas_1"] <= 6

    return Task(
        topic="analisis_usahatani_rc_bep", difficulty="medium", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi data usaha tani padi 40 petani dalam satu musim tanam (nilai uang dalam rupiah): `id_petani`, `luas_lahan_ha`, `produksi_kg`, `harga_jual_per_kg`, biaya variabel (`biaya_benih`, `biaya_pupuk`, `biaya_pestisida`, `biaya_tenaga_kerja`) dan biaya tetap (`penyusutan_alat`, `pajak_lahan`). Per petani: penerimaan TR = produksi_kg × harga_jual_per_kg; TC = total biaya variabel + total biaya tetap; pendapatan = TR − TC; R/C = TR/TC.

Kunci JSON: `rerata_penerimaan`, `rerata_total_biaya`, `rerata_pendapatan` (rata-rata per petani); `rc_ratio` (R/C usaha tani = ΣTR/ΣTC seluruh petani); `n_petani_rc_di_atas_1` (banyak petani dengan R/C individu > 1, bilangan bulat); `bep_harga` (BEP harga dalam Rp/kg = ΣTC/Σproduksi_kg); `pendapatan_per_ha` (Σpendapatan/Σluas_lahan_ha).
""",
        keys={"rerata_penerimaan": "rata-rata TR per petani", "rerata_total_biaya": "rata-rata TC per petani",
              "rerata_pendapatan": "rata-rata pendapatan per petani", "rc_ratio": "sum TR / sum TC",
              "n_petani_rc_di_atas_1": "banyak petani dengan R/C > 1", "bep_harga": "sum TC / sum produksi (Rp/kg)",
              "pendapatan_per_ha": "sum pendapatan / sum luas lahan"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
tr = df["produksi_kg"] * df["harga_jual_per_kg"]
biaya_variabel = df[["biaya_benih", "biaya_pupuk", "biaya_pestisida", "biaya_tenaga_kerja"]].sum(axis=1)
biaya_tetap = df[["penyusutan_alat", "pajak_lahan"]].sum(axis=1)
tc = biaya_variabel + biaya_tetap
pendapatan = tr - tc
hasil = {
    "rerata_penerimaan": float(tr.mean()),
    "rerata_total_biaya": float(tc.mean()),
    "rerata_pendapatan": float(pendapatan.mean()),
    "rc_ratio": float(tr.sum() / tc.sum()),
    "n_petani_rc_di_atas_1": int((tr / tc > 1).sum()),
    "bep_harga": float(tc.sum() / df["produksi_kg"].sum()),
    "pendapatan_per_ha": float(pendapatan.sum() / df["luas_lahan_ha"].sum()),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_outlier_iqr() -> Task:
    kolom = ["pendapatan_ribu", "pengeluaran_pangan_ribu", "jumlah_anggota"]

    def data(rng):
        n = 120
        pend = np.rint(rng.lognormal(np.log(3800), 0.5, n)).astype(int)
        peng = np.rint(np.clip(0.38 * pend + rng.normal(0, 220, n), 300, None)).astype(int)
        idx = rng.choice(n, 3, replace=False)
        peng[idx] += rng.integers(1500, 2500, 3)
        # fixed household-size counts (Q1 = 2, Q3 = 5, upper fence 9.5) plus three extended families
        anggota = rng.permutation(np.repeat(np.arange(1, 9), [8, 26, 22, 24, 20, 9, 4, 4]).tolist() + [10, 11, 13])
        return pd.DataFrame({"id_rt": [f"RT{i:03d}" for i in range(1, n + 1)], "pendapatan_ribu": pend,
                             "pengeluaran_pangan_ribu": peng, "jumlah_anggota": anggota})

    def manual(path, q=q_linear):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out = {}
        for c in kolom:
            v = [float(r[c]) for r in rows]
            q1, q3 = q(v, 0.25), q(v, 0.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            out[c] = (q1, q3, lo, hi, sum(x < lo or x > hi for x in v), v)
        return out

    def cross(path):
        m = manual(path)
        how = "interpolasi linear manual (n-1)p"
        return [("q1_pendapatan", m[kolom[0]][0], how), ("q3_pendapatan", m[kolom[0]][1], how),
                ("batas_atas_pendapatan", m[kolom[0]][3], how), ("n_outlier_pendapatan", m[kolom[0]][4], how),
                ("n_outlier_pengeluaran", m[kolom[1]][4], how), ("n_outlier_anggota", m[kolom[2]][4], how)]

    def wrong(path, exp):
        m = manual(path, q=q_weibull)
        return [("kuartil metode Weibull (n+1)p", "q1_pendapatan", m[kolom[0]][0]),
                ("kuartil metode Weibull (n+1)p", "batas_atas_pendapatan", m[kolom[0]][3])]

    def sanity(path, exp):
        for c, (q1, q3, lo, hi, cnt, v) in manual(path).items():
            assert min(min(abs(x - lo), abs(x - hi)) for x in v) > 1e-6, c
            assert cnt >= 1, c

    return Task(
        topic="deteksi_outlier_iqr", difficulty="medium", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi data 120 rumah tangga: `id_rt`, `pendapatan_ribu` dan `pengeluaran_pangan_ribu` (ribu rupiah per bulan) serta `jumlah_anggota` (orang). Deteksi pencilan per kolom dengan aturan IQR: Q1 dan Q3 dihitung dengan interpolasi linear (metode bawaan `pandas.Series.quantile`/`numpy.percentile`: posisi (n − 1)·p pada data terurut, indeks mulai 0), IQR = Q3 − Q1, batas bawah = Q1 − 1,5·IQR, batas atas = Q3 + 1,5·IQR; pencilan = nilai < batas bawah atau > batas atas.

Kunci JSON: `q1_pendapatan`, `q3_pendapatan`, `batas_atas_pendapatan`, `n_outlier_pendapatan`, `n_outlier_pengeluaran`, `n_outlier_anggota` (tiga terakhir: banyak pencilan, bilangan bulat).
""",
        keys={"q1_pendapatan": "Q1 pendapatan (linear)", "q3_pendapatan": "Q3 pendapatan (linear)",
              "batas_atas_pendapatan": "Q3 + 1,5 IQR pendapatan", "n_outlier_pendapatan": "banyak pencilan pendapatan",
              "n_outlier_pengeluaran": "banyak pencilan pengeluaran pangan",
              "n_outlier_anggota": "banyak pencilan jumlah anggota"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")


def pencilan(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(q1), float(q3), float(atas), int(((s < bawah) | (s > atas)).sum())


q1, q3, atas, n_pendapatan = pencilan(df["pendapatan_ribu"])
hasil = {
    "q1_pendapatan": q1,
    "q3_pendapatan": q3,
    "batas_atas_pendapatan": atas,
    "n_outlier_pendapatan": n_pendapatan,
    "n_outlier_pengeluaran": pencilan(df["pengeluaran_pangan_ribu"])[3],
    "n_outlier_anggota": pencilan(df["jumlah_anggota"])[3],
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_data_hilang() -> Task:
    kolom = ["usia", "pendapatan_juta", "skor_literasi", "skor_kepuasan"]

    def data(rng):
        n = 150
        usia = rng.integers(18, 41, n).astype(float)
        pend = np.clip(rng.lognormal(np.log(4.5), 0.45, n), 1.0, 25.0).round(1)
        lit = np.clip(np.rint(rng.normal(66, 11, n)), 20, 100)
        kep = np.clip(np.rint(30 + 0.45 * lit + rng.normal(0, 8, n)), 10, 100)
        bobot = np.exp(-(lit - lit.mean()) / 8)
        hilang = [rng.choice(n, 7, replace=False), rng.choice(n, 12, replace=False, p=bobot / bobot.sum()),
                  rng.choice(n, 9, replace=False), rng.choice(n, 6, replace=False)]
        for arr, idx in zip((usia, pend, lit, kep), hilang):
            arr[idx] = np.nan
        return pd.DataFrame({"id_responden": [f"R{i:03d}" for i in range(1, n + 1)],
                             "usia": pd.array(usia, dtype="Int64"), "pendapatan_juta": pend,
                             "skor_literasi": pd.array(lit, dtype="Int64"),
                             "skor_kepuasan": pd.array(kep, dtype="Int64")})

    def manual(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        lengkap = [r for r in rows if all(r[c] != "" for c in kolom)]
        return rows, lengkap

    def cross(path):
        rows, lengkap = manual(path)
        lit = [float(r["skor_literasi"]) for r in lengkap]
        kep = [float(r["skor_kepuasan"]) for r in lengkap]
        semua = [float(r["skor_literasi"]) for r in rows if r["skor_literasi"] != ""]
        how = "modul csv, Python murni"
        return [("missing_usia", sum(r["usia"] == "" for r in rows), how),
                ("missing_pendapatan", sum(r["pendapatan_juta"] == "" for r in rows), how),
                ("missing_literasi", sum(r["skor_literasi"] == "" for r in rows), how),
                ("missing_kepuasan", sum(r["skor_kepuasan"] == "" for r in rows), how),
                ("n_lengkap", len(lengkap), how), ("rerata_literasi_listwise", sum(lit) / len(lit), how),
                ("rerata_literasi_tersedia", sum(semua) / len(semua), how),
                ("r_listwise", float(np.corrcoef(lit, kep)[0, 1]), "numpy corrcoef")]

    def wrong(path, exp):
        df = read(path)
        pair = df[["skor_literasi", "skor_kepuasan"]].dropna()
        return [("rerata tanpa listwise", "rerata_literasi_listwise", exp["rerata_literasi_tersedia"]),
                ("korelasi pairwise (2 kolom saja)", "r_listwise",
                 float(pair["skor_literasi"].corr(pair["skor_kepuasan"])))]

    return Task(
        topic="penanganan_data_hilang", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi data survei literasi keuangan 150 responden: `id_responden`, `usia`, `pendapatan_juta`, `skor_literasi` dan `skor_kepuasan`; sel kosong berarti data hilang (missing). (1) Hitung banyak nilai hilang per kolom. (2) Lakukan penghapusan listwise: buang setiap baris yang memiliki nilai hilang pada salah satu dari empat kolom tersebut. (3) Dari data listwise hitung rata-rata `skor_literasi` dan korelasi Pearson antara `skor_literasi` dan `skor_kepuasan`. (4) Hitung juga rata-rata `skor_literasi` dari semua nilai yang tersedia (tanpa penghapusan listwise).

Kunci JSON: `missing_usia`, `missing_pendapatan`, `missing_literasi`, `missing_kepuasan`, `n_lengkap` (banyak baris setelah listwise) — semuanya bilangan bulat — serta `rerata_literasi_listwise`, `rerata_literasi_tersedia`, `r_listwise`.
""",
        keys={"missing_usia": "banyak nilai hilang usia", "missing_pendapatan": "banyak nilai hilang pendapatan_juta",
              "missing_literasi": "banyak nilai hilang skor_literasi",
              "missing_kepuasan": "banyak nilai hilang skor_kepuasan", "n_lengkap": "banyak baris lengkap",
              "rerata_literasi_listwise": "rata-rata skor_literasi setelah listwise",
              "rerata_literasi_tersedia": "rata-rata skor_literasi dari semua nilai tersedia",
              "r_listwise": "r Pearson literasi-kepuasan setelah listwise"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
kolom = ["usia", "pendapatan_juta", "skor_literasi", "skor_kepuasan"]
hilang = df[kolom].isna().sum()
lengkap = df.dropna(subset=kolom)
hasil = {
    "missing_usia": int(hilang["usia"]),
    "missing_pendapatan": int(hilang["pendapatan_juta"]),
    "missing_literasi": int(hilang["skor_literasi"]),
    "missing_kepuasan": int(hilang["skor_kepuasan"]),
    "n_lengkap": int(len(lengkap)),
    "rerata_literasi_listwise": float(lengkap["skor_literasi"].mean()),
    "rerata_literasi_tersedia": float(df["skor_literasi"].mean()),
    "r_listwise": float(lengkap["skor_literasi"].corr(lengkap["skor_kepuasan"])),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong)


@register
def t_ringkasan_kelompok() -> Task:
    sektor = ["kuliner", "fashion", "kriya", "jasa"]

    def data(rng):
        n = 160
        sek = [str(s) for s in rng.choice(sektor, n, p=[0.35, 0.25, 0.15, 0.25])]
        wilayah = rng.choice(["utara", "selatan", "timur", "barat"], n)
        mu = {"kuliner": 17, "fashion": 24, "kriya": 13, "jasa": 20}
        lam = {"kuliner": 3, "fashion": 4, "kriya": 2, "jasa": 3}
        omzet = [float(np.round(rng.lognormal(np.log(mu[s]), 0.45), 2)) for s in sek]
        karyawan = [int(1 + rng.poisson(lam[s])) for s in sek]
        return pd.DataFrame({"id_umkm": [f"UMKM{i:03d}" for i in range(1, n + 1)], "sektor": sek, "wilayah": wilayah,
                             "omzet_juta": omzet, "jumlah_karyawan": karyawan,
                             "lama_usaha_tahun": rng.integers(1, 21, n)})

    def cross(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        grp = {s: [r for r in rows if r["sektor"] == s] for s in sektor}
        mean = [sum(float(r["omzet_juta"]) for r in grp[s]) / len(grp[s]) for s in sektor]
        med = []
        for s in sektor:
            v = sorted(int(r["jumlah_karyawan"]) for r in grp[s])
            med.append((v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2)
        ks = [float(r["omzet_juta"]) for r in rows if r["sektor"] == "kuliner" and r["wilayah"] == "selatan"]
        how = "modul csv, Python murni"
        return [("jumlah_umkm", [len(grp[s]) for s in sektor], how), ("rerata_omzet", mean, how),
                ("median_karyawan", med, how), ("sektor_omzet_tertinggi", sektor[mean.index(max(mean))], how),
                ("rerata_omzet_kuliner_selatan", sum(ks) / len(ks), how)]

    def sanity(path, exp):
        srt = sorted(exp["rerata_omzet"])
        assert srt[-1] - srt[-2] > 0.5

    return Task(
        topic="ringkasan_per_kelompok", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi data 160 UMKM: `id_umkm`, `sektor` ("kuliner", "fashion", "kriya", "jasa"), `wilayah` ("utara", "selatan", "timur", "barat"), `omzet_juta` (omzet per bulan, juta rupiah), `jumlah_karyawan` dan `lama_usaha_tahun`. Buat ringkasan per sektor.

Kunci JSON: `jumlah_umkm` (daftar banyak UMKM per sektor), `rerata_omzet` (daftar rata-rata omzet per sektor), `median_karyawan` (daftar median jumlah karyawan per sektor) — ketiga daftar berurutan kuliner, fashion, kriya, jasa; `sektor_omzet_tertinggi` (nama sektor dengan rata-rata omzet tertinggi); `rerata_omzet_kuliner_selatan` (rata-rata omzet UMKM sektor kuliner di wilayah selatan).
""",
        keys={"jumlah_umkm": "banyak UMKM: kuliner, fashion, kriya, jasa",
              "rerata_omzet": "rata-rata omzet: kuliner, fashion, kriya, jasa",
              "median_karyawan": "median karyawan: kuliner, fashion, kriya, jasa",
              "sektor_omzet_tertinggi": "sektor dengan rata-rata omzet tertinggi",
              "rerata_omzet_kuliner_selatan": "rata-rata omzet kuliner di wilayah selatan"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
urutan = ["kuliner", "fashion", "kriya", "jasa"]
per_sektor = df.groupby("sektor")
rerata = per_sektor["omzet_juta"].mean().reindex(urutan)
kuliner_selatan = df[(df["sektor"] == "kuliner") & (df["wilayah"] == "selatan")]
hasil = {
    "jumlah_umkm": [int(v) for v in per_sektor.size().reindex(urutan)],
    "rerata_omzet": [float(v) for v in rerata],
    "median_karyawan": [float(v) for v in per_sektor["jumlah_karyawan"].median().reindex(urutan)],
    "sektor_omzet_tertinggi": str(rerata.idxmax()),
    "rerata_omzet_kuliner_selatan": float(kuliner_selatan["omzet_juta"].mean()),
}
print(json.dumps(hasil))
'''),
        cross=cross, sanity=sanity)


@register
def t_pertumbuhan() -> Task:
    def data(rng):
        tumbuh = rng.normal(0.025, 0.045, 14)
        produksi, luas = [212_000.0], [43_000.0]
        for g in tumbuh:
            produksi.append(produksi[-1] * (1 + g))
            luas.append(luas[-1] * (1 + rng.normal(0.004, 0.015)))
        df = pd.DataFrame({"tahun": np.arange(2010, 2025), "produksi_ton": np.round(produksi, 2),
                           "luas_panen_ha": np.round(luas, 2)})
        return df.iloc[rng.permutation(15)].reset_index(drop=True)

    def series(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: int(r["tahun"]))
        return ([int(r["tahun"]) for r in rows], [float(r["produksi_ton"]) for r in rows],
                [float(r["luas_panen_ha"]) for r in rows])

    def cross(path):
        th, x, luas = series(path)
        g = [(x[i] - x[i - 1]) / x[i - 1] * 100 for i in range(1, len(x))]
        how = "Python murni setelah diurutkan"
        return [("pertumbuhan_persen", g, how), ("rerata_pertumbuhan_persen", sum(g) / len(g), how),
                ("cagr_persen", ((x[-1] / x[0]) ** (1 / 14) - 1) * 100, how), ("ma3_2024", sum(x[-3:]) / 3, how),
                ("tahun_pertumbuhan_tertinggi", th[1 + g.index(max(g))], how),
                ("produktivitas_2024", x[-1] / luas[-1], how)]

    def wrong(path, exp):
        th, x, luas = series(path)
        df = read(path)
        raw = df["produksi_ton"].tolist()
        return [("eksponen 1/15", "cagr_persen", ((x[-1] / x[0]) ** (1 / 15) - 1) * 100),
                ("data tidak diurutkan", "pertumbuhan_persen", [(raw[i] - raw[i - 1]) / raw[i - 1] * 100
                                                               for i in range(1, len(raw))]),
                ("data tidak diurutkan", "ma3_2024", sum(raw[-3:]) / 3)]

    def sanity(path, exp):
        g = sorted(exp["pertumbuhan_persen"])
        th = read(path)["tahun"].tolist()
        assert g[-1] - g[-2] > 0.1 and th != sorted(th) and g[0] < 0

    return Task(
        topic="pertumbuhan_dan_rata_rata_bergerak", difficulty="medium", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi produksi padi sebuah kabupaten tahun 2010–2024: `tahun`, `produksi_ton`, `luas_panen_ha`; baris dalam berkas TIDAK urut tahun. Urutkan menurut tahun, lalu hitung untuk produksi: pertumbuhan tahunan (%) = (Xₜ − Xₜ₋₁)/Xₜ₋₁ × 100 untuk 2011–2024; rata-rata aritmetika ke-14 pertumbuhan tersebut; CAGR (%) = ((X₂₀₂₄/X₂₀₁₀)^(1/14) − 1) × 100; rata-rata bergerak sederhana 3 tahun untuk 2024 (rata-rata produksi 2022, 2023, 2024); tahun dengan pertumbuhan tertinggi; serta produktivitas 2024 = produksi/luas panen (ton/ha).

Kunci JSON: `pertumbuhan_persen` (daftar 14 angka berurutan 2011–2024), `rerata_pertumbuhan_persen`, `cagr_persen`, `ma3_2024`, `tahun_pertumbuhan_tertinggi` (bilangan bulat), `produktivitas_2024`. Persen dalam skala 0–100.
""",
        keys={"pertumbuhan_persen": "pertumbuhan tahunan 2011..2024 (%)",
              "rerata_pertumbuhan_persen": "rata-rata aritmetika pertumbuhan (%)", "cagr_persen": "CAGR 2010-2024 (%)",
              "ma3_2024": "rata-rata bergerak 3 tahun 2022-2024",
              "tahun_pertumbuhan_tertinggi": "tahun dengan pertumbuhan tertinggi",
              "produktivitas_2024": "produksi/luas panen 2024 (ton/ha)"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv").sort_values("tahun").reset_index(drop=True)
x = df["produksi_ton"]
tumbuh = ((x - x.shift(1)) / x.shift(1) * 100).iloc[1:]
periode = int(df["tahun"].iloc[-1] - df["tahun"].iloc[0])
hasil = {
    "pertumbuhan_persen": [float(v) for v in tumbuh],
    "rerata_pertumbuhan_persen": float(tumbuh.mean()),
    "cagr_persen": float(((x.iloc[-1] / x.iloc[0]) ** (1 / periode) - 1) * 100),
    "ma3_2024": float(x.rolling(3).mean().iloc[-1]),
    "tahun_pertumbuhan_tertinggi": int(df.loc[tumbuh.idxmax(), "tahun"]),
    "produktivitas_2024": float(x.iloc[-1] / df["luas_panen_ha"].iloc[-1]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_pemilihan_uji() -> Task:
    def data(rng):
        a = np.clip(np.rint(rng.normal(77, 8, 28)), 0, 100)
        b = np.clip(np.rint(63 + rng.gamma(1.3, 9.0, 27)), 0, 100)
        model = np.array(["discovery"] * 28 + ["konvensional"] * 27)
        nilai = np.concatenate([a, b]).astype(int)
        perm = rng.permutation(55)
        return pd.DataFrame({"id_siswa": [f"S{i:02d}" for i in range(1, 56)], "model": model[perm],
                             "nilai": nilai[perm]})

    def groups(path):
        df = read(path)
        return [df.loc[df["model"] == g, "nilai"].tolist() for g in ("discovery", "konvensional")]

    def cross(path):
        a, b = groups(path)
        (_, pa), (_, pb) = shapiro_manual(np.array(a)), shapiro_manual(np.array(b))
        m = mwu_manual(a, b)
        tol = {"abs": 1e-5, "rel": 1e-4}
        return [("sw_p_discovery", pa, "Royston AS R94 manual", tol),
                ("sw_p_konvensional", pb, "Royston AS R94 manual", tol),
                ("uji", "uji t independen" if pa > 0.05 and pb > 0.05 else "uji mann-whitney", "aturan keputusan"),
                ("statistik", m["u"], "U manual"), ("p_value", m["p"], "Mann-Whitney manual")]

    def wrong(path, exp):
        a, b = groups(path)
        return [("uji t walau tidak normal", "p_value", stats.ttest_ind(a, b).pvalue),
                ("tanpa koreksi kontinuitas", "p_value", mwu_manual(a, b, continuity=False)["p"])]

    def sanity(path, exp):
        assert exp["sw_p_discovery"] > 0.15 and exp["sw_p_konvensional"] < 0.01 and exp["uji"] == "uji mann-whitney"

    return Task(
        topic="pemilihan_uji_berdasarkan_normalitas", difficulty="hard", libs=["pandas", "scipy"], data=data,
        body="""
Berkas `data.csv` berisi nilai tes siswa: `id_siswa`, `model` ("discovery" atau "konvensional") dan `nilai`. Ikuti prosedur: (1) uji normalitas Shapiro–Wilk pada nilai tiap kelompok secara terpisah; (2) bila p-value KEDUA kelompok > 0,05, lakukan uji t independen Student (varians sama) dua sisi dengan t untuk selisih discovery − konvensional; bila tidak, lakukan uji Mann–Whitney dua sisi dengan a = discovery dan b = konvensional, pendekatan normal dengan koreksi ties dan koreksi kontinuitas (semantik `scipy.stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic', use_continuity=True)`), statistik U = R_a − n_a(n_a + 1)/2 untuk kelompok discovery.

Kunci JSON: `sw_p_discovery`, `sw_p_konvensional`, `uji` (persis "uji t independen" atau "uji mann-whitney"), `statistik` (t atau U dari uji terpilih), `p_value` (uji terpilih).
""",
        keys={"sw_p_discovery": "p Shapiro-Wilk kelompok discovery",
              "sw_p_konvensional": "p Shapiro-Wilk kelompok konvensional",
              "uji": "uji terpilih: 'uji t independen' atau 'uji mann-whitney'",
              "statistik": "statistik uji terpilih (t atau U discovery)", "p_value": "p dua sisi uji terpilih"},
        solution=program('''
import json

import pandas as pd
from scipy import stats

df = pd.read_csv("data.csv")
a = df.loc[df["model"] == "discovery", "nilai"]
b = df.loc[df["model"] == "konvensional", "nilai"]
p_a = stats.shapiro(a).pvalue
p_b = stats.shapiro(b).pvalue
if p_a > 0.05 and p_b > 0.05:
    uji = "uji t independen"
    res = stats.ttest_ind(a, b, equal_var=True)
else:
    uji = "uji mann-whitney"
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
hasil = {
    "sw_p_discovery": float(p_a),
    "sw_p_konvensional": float(p_b),
    "uji": uji,
    "statistik": float(res.statistic),
    "p_value": float(res.pvalue),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


@register
def t_ukuran_sampel() -> Task:
    e5, e10, chi2, prop, d = Fraction(1, 20), Fraction(1, 10), Fraction(3841, 1000), Fraction(1, 2), Fraction(1, 20)

    def exact(nh):
        big_n = sum(nh)
        s5 = Fraction(big_n) / (1 + big_n * e5 ** 2)
        s10 = Fraction(big_n) / (1 + big_n * e10 ** 2)
        km = chi2 * big_n * prop * (1 - prop) / (d ** 2 * (big_n - 1) + chi2 * prop * (1 - prop))
        n5 = math.ceil(s5)
        return big_n, s5, s10, km, [Fraction(n5 * h, big_n) for h in nh]

    def data(rng):
        while True:  # redraw until no quantity sits within 1e-4 of an integer and round-half != ceiling for Slovin 5 %
            nh = [int(v) for v in rng.integers(140, 380, 8)]
            big_n, s5, s10, km, alok = exact(nh)
            if not any(near_int(f) for f in [s5, s10, km, *alok]) and s5 - math.floor(s5) < Fraction(1, 2):
                break
        return pd.DataFrame({"sekolah": [f"SMP Negeri {i}" for i in range(1, 9)], "jumlah_siswa": nh})

    def cross(path):
        with open(path, newline="", encoding="utf-8") as fh:
            nh = [int(r["jumlah_siswa"]) for r in csv.DictReader(fh)]
        big_n, s5, s10, km, alok = exact(nh)
        how = "pecahan eksak (fractions) + ceiling"
        alokasi = [math.ceil(f) for f in alok]
        return [("n_populasi", big_n, how), ("n_slovin_5", math.ceil(s5), how), ("n_slovin_10", math.ceil(s10), how),
                ("n_krejcie_morgan", math.ceil(km), how), ("alokasi_slovin_5", alokasi, how),
                ("total_alokasi", sum(alokasi), how)]

    def wrong(path, exp):
        with open(path, newline="", encoding="utf-8") as fh:
            nh = [int(r["jumlah_siswa"]) for r in csv.DictReader(fh)]
        s5 = exact(nh)[1]
        return [("dibulatkan ke terdekat", "n_slovin_5", int(round(s5)))]

    return Task(
        topic="ukuran_sampel_slovin_krejcie_morgan", difficulty="easy", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi populasi penelitian: `sekolah` dan `jumlah_siswa` (banyak siswa kelas VIII tiap sekolah); populasi N = jumlah seluruh siswa. Hitung: (1) ukuran sampel Slovin n = N/(1 + N·e²) untuk e = 0,05 dan e = 0,10; (2) ukuran sampel Krejcie–Morgan s = χ²·N·P(1 − P)/(d²(N − 1) + χ²·P(1 − P)) dengan χ² = 3,841, P = 0,5 dan d = 0,05; (3) alokasi proporsional sampel Slovin e = 0,05 ke tiap sekolah: nₕ = n·Nₕ/N, memakai n Slovin yang sudah dibulatkan. Semua hasil dibulatkan KE ATAS ke bilangan bulat (ceiling).

Kunci JSON (semuanya bilangan bulat): `n_populasi`, `n_slovin_5`, `n_slovin_10`, `n_krejcie_morgan`, `alokasi_slovin_5` (daftar nₕ berurutan sesuai baris berkas), `total_alokasi` (jumlah daftar tersebut).
""",
        keys={"n_populasi": "N", "n_slovin_5": "Slovin e=0,05 (ceiling)", "n_slovin_10": "Slovin e=0,10 (ceiling)",
              "n_krejcie_morgan": "Krejcie-Morgan chi2=3,841, P=0,5, d=0,05 (ceiling)",
              "alokasi_slovin_5": "alokasi proporsional per sekolah (ceiling)", "total_alokasi": "jumlah alokasi"},
        solution=program('''
import json
import math

import pandas as pd

df = pd.read_csv("data.csv")
N = int(df["jumlah_siswa"].sum())


def slovin(e):
    return math.ceil(N / (1 + N * e ** 2))


chi2, P, d = 3.841, 0.5, 0.05
krejcie_morgan = math.ceil(chi2 * N * P * (1 - P) / (d ** 2 * (N - 1) + chi2 * P * (1 - P)))
n5 = slovin(0.05)
alokasi = [math.ceil(n5 * int(nh) / N) for nh in df["jumlah_siswa"]]
hasil = {
    "n_populasi": N,
    "n_slovin_5": n5,
    "n_slovin_10": slovin(0.10),
    "n_krejcie_morgan": krejcie_morgan,
    "alokasi_slovin_5": alokasi,
    "total_alokasi": sum(alokasi),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong)


@register
def t_pembersihan_tabulasi() -> Task:
    kolom = ["jenis_kelamin", "fakultas", "pakai_paylater"]
    varian = {
        "laki-laki": ["L", "l", "Laki-laki", "laki-laki", "LAKI-LAKI", " L", "Laki-Laki "],
        "perempuan": ["P", "p", "Perempuan", "perempuan", "PEREMPUAN", "P ", " perempuan"],
        "ekonomi": ["Ekonomi", "ekonomi", "EKONOMI", " Ekonomi", "ekonomi "],
        "teknik": ["Teknik", "teknik", "TEKNIK", "Teknik ", " teknik"],
        "hukum": ["Hukum", "hukum", "HUKUM", " Hukum", "hukum "],
        "ya": ["Ya", "ya", "YA", "ya ", " Ya"],
        "tidak": ["Tidak", "tidak", "TIDAK", "tidak ", " Tidak"],
    }

    def data(rng):
        n = 210
        fak = [str(v) for v in rng.choice(["ekonomi", "teknik", "hukum"], n, p=[0.4, 0.35, 0.25])]
        jk = [str(v) for v in rng.choice(["laki-laki", "perempuan"], n)]
        dasar = {"ekonomi": 0.56, "teknik": 0.33, "hukum": 0.44}
        ya = rng.random(n) < np.array([dasar[f] + (0.08 if j == "perempuan" else -0.04) for f, j in zip(fak, jk)])

        def pilih(kat):
            v = varian[kat]
            return v[int(rng.integers(len(v)))]

        rows = [{"id_responden": f"MHS{i + 1:04d}", "jenis_kelamin": pilih(jk[i]), "fakultas": pilih(fak[i]),
                 "pakai_paylater": pilih("ya" if ya[i] else "tidak")} for i in range(n)]
        kosong = rng.choice(n, 9, replace=False)
        for r in kosong:
            rows[r][kolom[int(rng.integers(3))]] = None
        ganda = rng.choice(np.setdiff1d(np.arange(n), kosong), 12, replace=False)
        urut = [(float(i), row) for i, row in enumerate(rows)]
        urut += [(float(rng.uniform(s + 0.5, n)), dict(rows[s])) for s in ganda]
        urut.sort(key=lambda kv: kv[0])
        return pd.DataFrame([row for _, row in urut])

    def bersih(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for c in kolom:
                v = r[c].strip().lower()
                r[c] = {"l": "laki-laki", "p": "perempuan"}.get(v, v) if c == "jenis_kelamin" else v
        seen, unik = set(), []
        for r in rows:
            if r["id_responden"] not in seen:
                seen.add(r["id_responden"])
                unik.append(r)
        final = [r for r in unik if all(r[c] != "" for c in kolom)]
        return rows, unik, final

    def cross(path):
        rows, unik, final = bersih(path)

        def persen(c, v):
            sub = [r for r in final if r[c] == v]
            return 100 * sum(r["pakai_paylater"] == "ya" for r in sub) / len(sub)

        how = "modul csv, pembersihan Python murni"
        return [("n_duplikat_dihapus", len(rows) - len(unik), how), ("n_kosong_dihapus", len(unik) - len(final), how),
                ("n_bersih", len(final), how), ("persen_ya_ekonomi", persen("fakultas", "ekonomi"), how),
                ("persen_ya_hukum", persen("fakultas", "hukum"), how),
                ("persen_ya_teknik", persen("fakultas", "teknik"), how),
                ("persen_ya_laki_laki", persen("jenis_kelamin", "laki-laki"), how),
                ("persen_ya_perempuan", persen("jenis_kelamin", "perempuan"), how)]

    def wrong(path, exp):
        rows, unik, final = bersih(path)
        ya = [r for r in final if r["pakai_paylater"] == "ya"]
        with open(path, newline="", encoding="utf-8") as fh:  # without the "l"/"p" mapping
            raw = list(csv.DictReader(fh))
        seen, nomap = set(), []
        for r in raw:
            if r["id_responden"] in seen:
                continue
            seen.add(r["id_responden"])
            vals = {c: r[c].strip().lower() for c in kolom}
            if all(vals.values()):
                nomap.append(vals)
        laki = [r for r in nomap if r["jenis_kelamin"] == "laki-laki"]
        return [("persentase kolom (bukan baris)", "persen_ya_ekonomi",
                 100 * sum(r["fakultas"] == "ekonomi" for r in ya) / len(ya)),
                ('"l" tidak diubah menjadi "laki-laki"', "persen_ya_laki_laki",
                 100 * sum(r["pakai_paylater"] == "ya" for r in laki) / len(laki)),
                ("duplikat tidak dihapus", "n_bersih", len(final) + len(rows) - len(unik))]

    def sanity(path, exp):
        rows, unik, final = bersih(path)
        assert {r["jenis_kelamin"] for r in final} == {"laki-laki", "perempuan"}
        assert {r["fakultas"] for r in final} == {"ekonomi", "hukum", "teknik"}
        assert {r["pakai_paylater"] for r in final} == {"ya", "tidak"}

    return Task(
        topic="pembersihan_data_tabulasi_silang", difficulty="hard", libs=["pandas"], data=data,
        body="""
Berkas `data.csv` berisi data mentah survei penggunaan paylater mahasiswa: `id_responden`, `jenis_kelamin`, `fakultas`, `pakai_paylater`. Penulisan kategori tidak konsisten (huruf besar/kecil, spasi di awal/akhir, "L"/"P" untuk laki-laki/perempuan), ada baris yang tercatat ganda dan ada sel kosong. Bersihkan berurutan: (1) pada ketiga kolom kategori hapus spasi di awal/akhir dan ubah ke huruf kecil, lalu ubah "l" menjadi "laki-laki" dan "p" menjadi "perempuan"; (2) hapus baris dengan `id_responden` ganda, pertahankan kemunculan pertama; (3) hapus baris yang memiliki sel kosong. Lalu hitung persentase responden yang menjawab "ya" di dalam tiap kategori (persentase baris tabulasi silang, skala 0–100).

Kunci JSON: `n_duplikat_dihapus`, `n_kosong_dihapus`, `n_bersih` (banyak baris akhir) — bilangan bulat; `persen_ya_ekonomi`, `persen_ya_hukum`, `persen_ya_teknik` (per fakultas); `persen_ya_laki_laki`, `persen_ya_perempuan` (per jenis kelamin).
""",
        keys={"n_duplikat_dihapus": "baris ganda yang dihapus", "n_kosong_dihapus": "baris bersel kosong yang dihapus",
              "n_bersih": "banyak baris setelah pembersihan", "persen_ya_ekonomi": "% ya di fakultas ekonomi",
              "persen_ya_hukum": "% ya di fakultas hukum", "persen_ya_teknik": "% ya di fakultas teknik",
              "persen_ya_laki_laki": "% ya pada laki-laki", "persen_ya_perempuan": "% ya pada perempuan"},
        solution=program('''
import json

import pandas as pd

df = pd.read_csv("data.csv")
kolom = ["jenis_kelamin", "fakultas", "pakai_paylater"]
for k in kolom:
    df[k] = df[k].str.strip().str.lower()
df["jenis_kelamin"] = df["jenis_kelamin"].replace({"l": "laki-laki", "p": "perempuan"})
n_awal = len(df)
df = df.drop_duplicates(subset="id_responden", keep="first")
n_unik = len(df)
df = df.dropna(subset=kolom)
fakultas = pd.crosstab(df["fakultas"], df["pakai_paylater"], normalize="index") * 100
jenis_kelamin = pd.crosstab(df["jenis_kelamin"], df["pakai_paylater"], normalize="index") * 100
hasil = {
    "n_duplikat_dihapus": int(n_awal - n_unik),
    "n_kosong_dihapus": int(n_unik - len(df)),
    "n_bersih": int(len(df)),
    "persen_ya_ekonomi": float(fakultas.loc["ekonomi", "ya"]),
    "persen_ya_hukum": float(fakultas.loc["hukum", "ya"]),
    "persen_ya_teknik": float(fakultas.loc["teknik", "ya"]),
    "persen_ya_laki_laki": float(jenis_kelamin.loc["laki-laki", "ya"]),
    "persen_ya_perempuan": float(jenis_kelamin.loc["perempuan", "ya"]),
}
print(json.dumps(hasil))
'''),
        cross=cross, wrong=wrong, sanity=sanity)


# ---------------------------------------------------------------------------------------------------------------
# running and scoring (mirrors the harness: last JSON object on stdout, per-key tolerance)
# ---------------------------------------------------------------------------------------------------------------

def to_py(v: Any) -> Any:
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [to_py(x) for x in v]
    return v


def close(got: Any, exp: Any, tol: dict[str, float]) -> bool:
    if isinstance(exp, str):
        return isinstance(got, str) and got.strip().lower() == exp.strip().lower()
    if isinstance(exp, list):
        return isinstance(got, list) and len(got) == len(exp) and all(close(g, e, tol) for g, e in zip(got, exp))
    if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
        return False
    diff = abs(got - exp)
    return diff <= tol["abs"] or diff <= tol["rel"] * abs(exp)


def deviation(got: Any, exp: Any) -> float:
    if isinstance(exp, list):
        return max((deviation(g, e) for g, e in zip(got, exp)), default=0.0)
    return 0.0 if isinstance(exp, str) else abs(got - exp)


def last_json_object(text: str) -> dict | None:
    dec, found, i = json.JSONDecoder(), None, 0
    while (j := text.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            found = obj
        i = end
    return found


def score(got: dict | None, expected: dict, tolerance: dict) -> float:
    if not isinstance(got, dict):
        return 0.0
    ok = sum(k in got and close(got[k], e, tolerance.get(k, tolerance["default"])) for k, e in expected.items())
    return ok / len(expected)


def run_solution(code: str, files: dict[str, Path]) -> subprocess.CompletedProcess:
    """Run a program in a fresh temporary directory that holds only the data files under their sandbox names."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="skripsibench-coding-") as tmp:
        for name, src in files.items():
            shutil.copyfile(src, Path(tmp) / name)
        assert sorted(os.listdir(tmp)) == sorted(files)
        return subprocess.run([PYTHON, "-c", code], cwd=tmp, env=env, capture_output=True, text=True, timeout=300)


def check_value(v: Any) -> None:
    if isinstance(v, str):
        assert v.strip()
    elif isinstance(v, list):
        assert v and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v), v
    else:
        assert isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), v


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tasks = [fn() for fn in REGISTRY]
    assert len(tasks) == 30, len(tasks)
    assert len({t.topic for t in tasks}) == 30
    built = []
    for pos, task in enumerate(tasks, start=1):
        tid, seed = f"coding-{pos:03d}", 100 + pos
        path = DATA_DIR / f"{tid}.csv"
        write_csv(task.data(np.random.default_rng(seed)), path)
        built.append((tid, seed, task, path))
    assert sorted(p.name for p in DATA_DIR.iterdir()) == sorted(b[3].name for b in built), "stray files in data dir"

    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(lambda b: run_solution(b[2].solution, {SANDBOX_NAME: b[3]}), built))

    items, rows = [], []
    for (tid, seed, task, path), run in zip(built, runs):
        assert run.returncode == 0 and run.stderr == "", (tid, run.stderr)
        expected = json.loads(run.stdout)  # the reference prints exactly one JSON object and nothing else
        assert isinstance(expected, dict) and list(expected) == list(task.keys), (tid, list(expected))
        assert 1 <= len(expected) <= MAX_KEYS and task.difficulty in ("easy", "medium", "hard")
        for value in expected.values():
            check_value(value)
        tolerance = {"default": dict(DEFAULT_TOL), **task.tol}
        assert set(task.tol) <= set(expected), tid
        if task.sanity:
            task.sanity(path, expected)

        entries = task.cross(path)
        assert {e[0] for e in entries} == set(expected), (tid, set(expected) - {e[0] for e in entries})
        max_dev = 0.0
        for key, value, how, *tol in entries:
            value = to_py(value)
            assert close(value, expected[key], tol[0] if tol else CROSS_TOL), (tid, key, how, value, expected[key])
            max_dev = max(max_dev, deviation(value, expected[key]))
        wrongs = task.wrong(path, expected) if task.wrong else []
        for label, key, value in wrongs:
            value = to_py(value)
            assert not close(value, expected[key], tolerance.get(key, tolerance["default"])), \
                (tid, "mistake accepted", label, key, value, expected[key])

        instruction = task.body.strip() + "\n\n" + FOOTER
        words = len(instruction.split())
        assert words <= MAX_WORDS, (tid, words)
        for key in expected:
            assert f"`{key}`" in instruction, (tid, key)
        items.append({
            "id": tid, "task": "coding", "privacy": "public",
            "input": {"instruction": instruction, "files": {SANDBOX_NAME: path.relative_to(ROOT).as_posix()},
                      "output_keys": dict(task.keys)},
            "reference": {"expected": expected, "tolerance": tolerance, "solution": task.solution},
            "meta": {"topic": task.topic, "libs": list(task.libs), "seed": seed, "difficulty": task.difficulty,
                     "created": CREATED},
        })
        methods = sorted({e[2] for e in entries})
        rows.append((tid, task.topic, task.difficulty, len(expected), len(entries), len(wrongs), max_dev, words,
                     methods))

    ITEMS_PATH.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")

    # validate the stored items exactly as the harness will see them
    stored = [json.loads(line) for line in ITEMS_PATH.read_text(encoding="utf-8").splitlines()]
    assert [it["id"] for it in stored] == [f"coding-{i:03d}" for i in range(1, 31)]

    def validate(item: dict) -> float:
        files = {name: ROOT / rel for name, rel in item["input"]["files"].items()}
        run = run_solution(item["reference"]["solution"], files)
        return score(last_json_object(run.stdout), item["reference"]["expected"], item["reference"]["tolerance"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(validate, stored))
    assert all(s == 1.0 for s in scores), scores

    print(f"{'id':<11} {'topic':<38} {'level':<6} {'keys':>4} {'cross':>5} {'wrong':>5} {'max|dev|':>9} "
          f"{'words':>5} score")
    for (tid, topic, level, nk, nc, nw, dev, words, _), s in zip(rows, scores):
        print(f"{tid:<11} {topic:<38} {level:<6} {nk:>4} {nc:>5} {nw:>5} {dev:>9.1e} {words:>5} {s:.2f}")
    print("difficulty:", dict(Counter(r[2] for r in rows)))
    print("cross-check methods:")
    for tid, *_, methods in rows:
        print(f"  {tid}: " + "; ".join(methods))
    print("p-values:")
    for it in stored:
        ps = {k: v for k, v in it["reference"]["expected"].items() if re.search(r"(^p_|_p$|_p_)", k)}
        if ps:
            print(f"  {it['id']}: " + ", ".join(f"{k}={[round(x, 4) for x in v] if isinstance(v, list) else f'{v:.4g}'}"
                                                for k, v in ps.items()))
    outputs = sorted(DATA_DIR.glob("coding-*.csv")) + [ITEMS_PATH]
    digest = hashlib.sha256("".join(f"{p.name}:{sha256(p)}\n" for p in outputs).encode()).hexdigest()
    print(f"{ITEMS_PATH.relative_to(ROOT)} sha256 {sha256(ITEMS_PATH)}")
    print(f"all {len(outputs)} outputs sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
