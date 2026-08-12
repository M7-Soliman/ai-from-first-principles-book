"""
Reference solutions for the Part XXIV drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------ coverage

def coverage(lo, hi, y):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def coverage_by_group(lo, hi, y, group):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    y = np.asarray(y, float); group = np.asarray(group)
    return {g: float(np.mean((y[group == g] >= lo[group == g])
                             & (y[group == g] <= hi[group == g])))
            for g in np.unique(group)}


def interval_width(lo, hi):
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


# --------------------------------------------------------- the two kinds

def decompose_variance(member_means, member_vars):
    m = np.asarray(member_means, float)
    v = np.asarray(member_vars, float)
    return v.mean(axis=0), m.var(axis=0)


def gaussian_interval(mu, sd, alpha=0.05):
    from math import sqrt
    z = _z(1 - alpha / 2)
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    return mu - z * sd, mu + z * sd


def _z(q):
    import math
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------ Gaussian processes

def rbf_kernel(a, b, ell=1.0, amp=1.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a[:, None] - b[None, :]
    return amp ** 2 * np.exp(-0.5 * (d / ell) ** 2)


def gp_posterior(xtr, ytr, xq, ell=1.0, amp=1.0, noise=0.1):
    xtr = np.asarray(xtr, float); ytr = np.asarray(ytr, float); xq = np.asarray(xq, float)
    K = rbf_kernel(xtr, xtr, ell, amp) + noise ** 2 * np.eye(len(xtr))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    Ks = rbf_kernel(xq, xtr, ell, amp)
    mean = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    var = np.clip(rbf_kernel(xq, xq, ell, amp).diagonal() - np.sum(v ** 2, axis=0),
                  1e-12, None)
    return mean, var


def log_marginal_likelihood(xtr, ytr, ell=1.0, amp=1.0, noise=0.1):
    xtr = np.asarray(xtr, float); ytr = np.asarray(ytr, float)
    n = len(xtr)
    K = rbf_kernel(xtr, xtr, ell, amp) + noise ** 2 * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    return float(-0.5 * ytr @ alpha - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2 * np.pi))


# ---------------------------------------------------- conformal prediction

def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")          # the sample is too small for this level
    return float(s[k - 1])


def conformal_interval(pred, q):
    pred = np.asarray(pred, float)
    return pred - q, pred + q


def normalised_scores(y, pred, sigma):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    return np.abs(y - pred) / sigma


def conformal_set(probs, q):
    p = np.asarray(probs, float)
    return (1.0 - p) <= q


def weighted_conformal_quantile(scores, weights, alpha):
    s = np.asarray(scores, float); w = np.asarray(weights, float)
    order = np.argsort(s)
    s, w = s[order], w[order]
    cw = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cw, 1 - alpha)
    return float(s[min(idx, len(s) - 1)])


# ---------------------------------------------------------------- decisions

def optimal_threshold(cost_fp, cost_fn):
    return float(cost_fp / (cost_fp + cost_fn))


def expected_cost(p, y, threshold, cost_fp, cost_fn):
    p = np.asarray(p, float); y = np.asarray(y).astype(bool)
    act = p >= threshold
    return float(np.mean(cost_fp * (act & ~y) + cost_fn * (~act & y)))


def value_of_information(p_outcome, p_signal, p_outcome_given_signal, utility):
    U = np.asarray(utility, float)
    p0 = np.asarray(p_outcome, float)
    ps = np.asarray(p_signal, float)
    pos = np.asarray(p_outcome_given_signal, float)   # (n_signals, n_outcomes)
    without = float(np.max(U @ p0))
    with_ = float(sum(ps[s] * np.max(U @ pos[s]) for s in range(len(ps))))
    return with_ - without


def abstain_band(cost_fp, cost_fn, cost_abstain):
    # acting costs (1-p) c_fp; not acting costs p c_fn; abstaining costs c_abstain
    lo = cost_abstain / cost_fn                     # below this, not acting is cheaper
    hi = 1.0 - cost_abstain / cost_fp               # above this, acting is cheaper
    if lo >= hi:
        return float("nan"), float("nan")
    return float(lo), float(hi)


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
