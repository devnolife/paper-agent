"""Statistics without third-party packages (CI installs only the core dependencies).

Bootstrap confidence intervals, paired comparisons (paired bootstrap, sign test, Wilcoxon signed-rank with normal
approximation), Holm correction, Spearman correlation (t approximation for p), Krippendorff's alpha (nominal,
ordinal, interval; any number of coders, missing ratings allowed) and the cost–quality Pareto front.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stdev(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _percentile(sorted_xs: List[float], q: float) -> float:
    if not sorted_xs:
        raise ValueError("empty")
    k = (len(sorted_xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return sorted_xs[int(k)] if f == c else sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def bootstrap_ci(xs: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI of the mean."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], xs[0]
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def paired_bootstrap(xs: Sequence[float], ys: Sequence[float], n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean difference x - y over paired items with a bootstrap CI and two-sided p (share of resamples crossing 0)."""
    d = [x - y for x, y in zip(xs, ys) if x is not None and y is not None]
    if not d:
        return {"n": 0, "diff": None, "ci": (None, None), "p": None}
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    below = sum(1 for b in boots if b <= 0) / n_boot
    above = sum(1 for b in boots if b >= 0) / n_boot
    return {"n": n, "diff": sum(d) / n, "ci": (_percentile(boots, 0.025), _percentile(boots, 0.975)),
            "p": min(1.0, 2 * min(below, above))}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p for k successes in n trials with p = 0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk * (1 + 1e-9)))


def sign_test(xs: Sequence[float], ys: Sequence[float]) -> dict:
    pos = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None and x > y)
    neg = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None and x < y)
    return {"pos": pos, "neg": neg, "ties": len(list(zip(xs, ys))) - pos - neg, "p": binom_two_sided(pos, pos + neg)}


def rankdata(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test (zero differences dropped, average ranks for ties, normal approximation with tie
    correction and continuity correction). Fine for n >= 10; report the sign test alongside for small n."""
    d = [x - y for x, y in zip(xs, ys) if x is not None and y is not None and x != y]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": 1.0}
    r = rankdata([abs(v) for v in d])
    w_plus = sum(rk for rk, v in zip(r, d) if v > 0)
    mu = n * (n + 1) / 4
    ties: Dict[float, int] = {}
    for v in r:
        ties[v] = ties.get(v, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    if var <= 0:
        return {"n": n, "w_plus": w_plus, "p": 1.0}
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return {"n": n, "w_plus": w_plus, "z": z, "p": min(1.0, 2 * norm_sf(max(0.0, z)))}


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values, same order as the input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    return betainc(df / 2, 0.5, df / (df + t * t)) if df > 0 else 1.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> dict:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return {"n": n, "rho": None, "p": None}
    rho = pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))
    if rho is None:
        return {"n": n, "rho": None, "p": None}
    if abs(rho) >= 1:
        return {"n": n, "rho": rho, "p": 0.0}
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return {"n": n, "rho": rho, "p": t_two_sided_p(t, n - 2)}


def krippendorff_alpha(units: Sequence[Sequence[Optional[float]]], level: str = "interval") -> Optional[float]:
    """Krippendorff's alpha. ``units``: one list per unit with each coder's value (None = missing)."""
    pairable = [[v for v in u if v is not None] for u in units]
    pairable = [u for u in pairable if len(u) >= 2]
    if not pairable:
        return None
    values = sorted({v for u in pairable for v in u})
    o: Dict[Tuple[float, float], float] = {}
    for u in pairable:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[(a, b)] = o.get((a, b), 0.0) + 1.0 / (m - 1)
    n_c = {c: sum(o.get((c, k), 0.0) for k in values) for c in values}
    n = sum(n_c.values())
    if n <= 1:
        return None

    def delta2(c: float, k: float) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return (c - k) ** 2
        if level == "ordinal":
            lo, hi = (c, k) if c <= k else (k, c)
            s = sum(n_c[g] for g in values if lo <= g <= hi)
            return (s - (n_c[c] + n_c[k]) / 2) ** 2
        raise ValueError(level)

    d_o = sum(w * delta2(c, k) for (c, k), w in o.items()) / n
    d_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in values for k in values) / (n * (n - 1))
    return None if d_e == 0 else 1.0 - d_o / d_e


def pareto_front(points: Sequence[Tuple[float, float, str]]) -> List[str]:
    """Labels on the cost–quality front: no other point is at most as costly and strictly better (or cheaper and
    at least as good)."""
    front = []
    for c, q, lab in points:
        dominated = any((c2 <= c and q2 > q) or (c2 < c and q2 >= q) for c2, q2, _ in points)
        if not dominated:
            front.append(lab)
    return front
