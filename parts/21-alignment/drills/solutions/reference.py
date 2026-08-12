"""
Reference solutions for the Part XXI drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------ what evidence you have

def zero_failure_bound(n, conf=0.95):
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def rule_of_three(n):
    return 3.0 / n


def trials_needed(p, conf=0.95):
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 1
    return int(np.ceil(np.log(1 - conf) / np.log1p(-p)))


def attack_success(p, n):
    return 1.0 - (1.0 - p) ** n


def binomial_interval(k, n, conf=0.95):
    # Wilson, which behaves at the edges where the normal approximation does not
    # and — the property the tests check — agrees with the exact zero-failure
    # bound closely enough to be usable for the same purpose.
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        return 0.0, zero_failure_bound(n, conf)
    if k == n:
        return 1.0 - zero_failure_bound(n, conf), 1.0
    z = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else _z(conf)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _z(conf):
    from math import sqrt
    # inverse normal CDF at (1+conf)/2, by bisection — no scipy dependency
    target = (1 + conf) / 2

    def cdf(x):
        return 0.5 * (1 + _erf(x / sqrt(2)))

    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _erf(x):
    import math
    return math.erf(x)


# ------------------------------------------------------------- superposition

def features_kept(W, frac=0.5):
    norms = np.linalg.norm(np.asarray(W, float), axis=0)
    if norms.max() <= 0:
        return 0
    return int((norms > frac * norms.max()).sum())


def mean_overlap(W):
    W = np.asarray(W, float)
    n = np.linalg.norm(W, axis=0)
    keep = n > 1e-12
    Wn = W[:, keep] / n[keep]
    G = np.abs(Wn.T @ Wn)
    m = G.shape[0]
    if m < 2:
        return 0.0
    return float(G[~np.eye(m, dtype=bool)].mean())


def max_orthogonal(n_dims):
    return int(n_dims)


# ------------------------------------------------------- dictionary recovery

def best_cosine(true_dirs, found_dirs):
    T = np.asarray(true_dirs, float)
    F = np.asarray(found_dirs, float)
    T = T / np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
    F = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
    return np.abs(T @ F.T).max(axis=1)


def recovered_count(true_dirs, found_dirs, thresh=0.9):
    return int((best_cosine(true_dirs, found_dirs) > thresh).sum())


# ---------------------------------------------------- probing against ablation

def ablate_direction(H, d):
    H = np.asarray(H, float)
    d = np.asarray(d, float)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return H.copy()
    d = d / nd
    return H - np.outer(H @ d, d)


def causal_effect(acc_intact, acc_ablated, acc_random):
    raw = acc_intact - acc_ablated
    # the network degrades a little under ANY perturbation of this size, so the
    # random-direction drop is the zero point. Reporting `raw` credits the
    # direction with damage that had nothing to do with it.
    return raw, raw - (acc_intact - acc_random)


# ------------------------------------------------------------------ patching

def restoration(acc_patched, acc_corrupt, acc_clean):
    denom = acc_clean - acc_corrupt
    if abs(denom) < 1e-12:
        return 0.0
    return (acc_patched - acc_corrupt) / denom


def steering_vector(h_pos, h_neg):
    v = np.asarray(h_pos, float).mean(0) - np.asarray(h_neg, float).mean(0)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


# ---------------------------------------------------------------- Goodhart

def best_of_n_true(true, proxy, n, rng=None):
    true = np.asarray(true, float)
    proxy = np.asarray(proxy, float)
    k = len(true) // n
    if k == 0:
        return float(true[int(np.argmax(proxy))])
    t = true[:k * n].reshape(k, n)
    p = proxy[:k * n].reshape(k, n)
    return float(t[np.arange(k), p.argmax(1)].mean())


def tail_ratio(x, q=0.999):
    x = np.asarray(x, float)
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.quantile(np.abs(x), q) / s)


def ensemble_proxy(proxies):
    return np.mean([np.asarray(p, float) for p in proxies], axis=0)


# --------------------------------------------------------- fairness criteria

def demographic_parity(y_pred, group):
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    return {g: float(y_pred[group == g].mean()) for g in np.unique(group)}


def equalised_odds(y_pred, y_true, group):
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    group = np.asarray(group)
    out = {}
    for g in np.unique(group):
        m = group == g
        pos, neg = m & y_true, m & ~y_true
        tpr = float(y_pred[pos].mean()) if pos.sum() else float("nan")
        fpr = float(y_pred[neg].mean()) if neg.sum() else float("nan")
        out[g] = (tpr, fpr)
    return out


def calibration_by_group(p_pred, y_true, group, bins=5):
    p_pred = np.asarray(p_pred, float)
    y_true = np.asarray(y_true, float)
    group = np.asarray(group)
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for g in np.unique(group):
        m = group == g
        gaps, weights = [], []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            sel = m & (p_pred >= lo) & (p_pred < hi if i < bins - 1 else p_pred <= hi)
            if sel.sum() == 0:
                continue
            gaps.append(abs(p_pred[sel].mean() - y_true[sel].mean()))
            weights.append(sel.sum())
        out[g] = float(np.average(gaps, weights=weights)) if gaps else 0.0
    return out


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
