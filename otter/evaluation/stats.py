"""Dependency-light statistics: chi-square tail, exact McNemar, bootstrap CIs.

Only numpy is required (scipy is not installed in .venv)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def _gammainc_upper_reg(s: float, x: float) -> float:
    """Q(s, x) = Gamma(s, x) / Gamma(s) (regularized upper incomplete gamma)."""
    if x <= 0:
        return 1.0
    if x < s + 1:  # series for P, then Q = 1 - P
        term = 1.0 / s
        total = term
        k = s
        for _ in range(10_000):
            k += 1
            term *= x / k
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + s * math.log(x) - math.lgamma(s))
        return max(0.0, 1.0 - p)
    # continued fraction (Lentz) for Q
    tiny = 1e-300
    b = x + 1 - s
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(x: float, df: int) -> float:
    return _gammainc_upper_reg(df / 2.0, x / 2.0)


def chi2_uniform_test(counts: Sequence[int]) -> tuple[float, float]:
    """Goodness-of-fit of `counts` against a uniform distribution; returns (chi2, p)."""
    n = sum(counts)
    k = len(counts)
    if n == 0 or k < 2:
        return 0.0, 1.0
    e = n / k
    stat = sum((c - e) ** 2 / e for c in counts)
    return stat, chi2_sf(stat, k - 1)


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of probabilities <= P(k))."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    pk = pmf[k]
    return min(1.0, sum(q for q in pmf if q <= pk * (1 + 1e-7)))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correctness vectors (same item order)."""
    assert len(a_correct) == len(b_correct)
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)   # A right, B wrong
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)   # B right, A wrong
    n = b01 + b10
    p = binom_test_two_sided(min(b01, b10), n) if n else 1.0
    return {"a_only": b01, "b_only": b10, "discordant": n, "p_value": p}


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean; returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0, groups: Optional[Sequence[str]] = None) -> dict:
    """Paired bootstrap of mean(a) - mean(b). Two-sided p-value = 2*min(P(d<=0), P(d>=0)) under resampling.

    If `groups` is given, resampling is clustered by group (e.g. lesson) to respect dependence."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    assert x.shape == y.shape
    if x.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    d = x - y
    if groups is None:
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        diffs = d[idx].mean(axis=1)
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        members = [np.flatnonzero(g == u) for u in uniq]
        sums = np.array([d[m].sum() for m in members])
        sizes = np.array([m.size for m in members])
        pick = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
        diffs = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_value": float(min(1.0, p))}
