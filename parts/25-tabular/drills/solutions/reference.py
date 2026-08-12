"""
Reference solutions for the Part XXV drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------ splitting

def sse(y):
    y = np.asarray(y, float)
    return float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0


def split_gain(GL, nL, GR, nR):
    GL = np.asarray(GL, float); GR = np.asarray(GR, float)
    nL = np.asarray(nL, float); nR = np.asarray(nR, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (np.where(nL > 0, GL ** 2 / np.maximum(nL, 1e-12), 0.0)
             + np.where(nR > 0, GR ** 2 / np.maximum(nR, 1e-12), 0.0)
             - (GL + GR) ** 2 / np.maximum(nL + nR, 1e-12))
    return g


def best_split(x, y, min_samples=1):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); xs, ys = x[o], y[o]
    n = len(ys)
    cs = np.cumsum(ys)
    idx = np.arange(1, n)
    # only positions where the value actually changes are valid thresholds
    valid = (xs[1:] != xs[:-1]) & (idx >= min_samples) & (n - idx >= min_samples)
    if not valid.any():
        return 0.0, None
    g = split_gain(cs[:-1], idx, cs[-1] - cs[:-1], n - idx)
    g = np.where(valid, g, -np.inf)
    b = int(np.argmax(g))
    return (float(g[b]), float(xs[b])) if g[b] > 0 else (0.0, None)


def quantile_bins(x, n_bins):
    x = np.asarray(x, float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def histogram_best_split(xb, y, n_bin_values, min_samples=1):
    xb = np.asarray(xb, int); y = np.asarray(y, float)
    G = np.bincount(xb, weights=y, minlength=n_bin_values)
    N = np.bincount(xb, minlength=n_bin_values).astype(float)
    GL, NL = np.cumsum(G), np.cumsum(N)
    GT, NT = GL[-1], NL[-1]
    g = split_gain(GL, NL, GT - GL, NT - NL)
    valid = (NL >= min_samples) & (NT - NL >= min_samples)
    g = np.where(valid, g, -np.inf)
    b = int(np.argmax(g))
    return (float(g[b]), b) if np.isfinite(g[b]) and g[b] > 0 else (0.0, None)


# ------------------------------------------------------------------ ensembles

def averaging_variance(sigma2, rho, B):
    return float(rho * sigma2 + (1.0 - rho) * sigma2 / B)


def variance_floor(sigma2, rho):
    return float(rho * sigma2)


def oob_mask(n, seed=0):
    r = np.random.default_rng(seed)
    drawn = r.integers(0, n, n)
    m = np.ones(n, dtype=bool)
    m[np.unique(drawn)] = False
    return m


def permutation_importance(predict, X, y, feature, n_repeats=5, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    base = float(np.mean((predict(X) - y) ** 2))
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        Xp = X.copy()
        Xp[:, feature] = Xp[r.permutation(len(Xp)), feature]
        out.append(float(np.mean((predict(Xp) - y) ** 2)))
    return float(np.mean(out) - base)


# ------------------------------------------------------------------- boosting

def negative_gradient(y, F, objective="ls"):
    y = np.asarray(y, float); F = np.asarray(F, float)
    if objective == "ls":
        return y - F
    return y - 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))


def hessian(y, F, objective="ls"):
    F = np.asarray(F, float)
    if objective == "ls":
        return np.ones_like(F)
    p = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
    return np.clip(p * (1 - p), 1e-12, None)


def leaf_value(g, h, lam=0.0):
    g = np.asarray(g, float); h = np.asarray(h, float)
    return float(g.sum() / (h.sum() + lam))


def structured_gain(gL, hL, gR, hR, lam=0.0, gamma=0.0):
    def t(g, h):
        return (np.asarray(g, float).sum() ** 2) / (np.asarray(h, float).sum() + lam)
    return float(0.5 * (t(gL, hL) + t(gR, hR)
                        - t(np.concatenate([np.ravel(gL), np.ravel(gR)]),
                            np.concatenate([np.ravel(hL), np.ravel(hR)]))) - gamma)


# --------------------------------------------------------------- calibration

def expected_calibration_error(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def isotonic_fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x)
    xs, v = x[o].copy(), y[o].astype(float).copy()
    w = np.ones(len(v))
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            nw = w[i] + w[i + 1]
            v = np.delete(v, i + 1); w = np.delete(w, i + 1); xs = np.delete(xs, i + 1)
            v[i], w[i] = nv, nw
            i = max(i - 1, 0)
        else:
            i += 1
    return xs, v


def auc(scores, y):
    s = np.asarray(scores, float); y = np.asarray(y).astype(int)
    o = np.argsort(s)
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------- ordered rows

def lag_matrix(y, n_lags, horizon=1):
    y = np.asarray(y, float)
    n = len(y)
    rows, tgt = [], []
    for t in range(n_lags, n - horizon + 1):
        rows.append(y[t - n_lags:t][::-1])       # most recent lag first
        tgt.append(y[t + horizon - 1])
    return np.array(rows), np.array(tgt)


def rolling_origin_splits(n, n_splits, min_train, horizon=1):
    span = (n - min_train - horizon + 1)
    if span <= 0 or n_splits <= 0:
        return
    step = max(1, span // n_splits)
    for k in range(n_splits):
        end = min_train + k * step
        te_lo = end + horizon - 1
        te_hi = min(te_lo + step, n)
        if te_lo >= n:
            break
        yield np.arange(end), np.arange(te_lo, te_hi)


def seasonal_naive(y, season, horizon=1):
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    src = np.arange(len(y)) - season
    ok = src >= 0
    out[ok] = y[src[ok]]
    return out


def pinball_loss(y, q_pred, tau):
    y = np.asarray(y, float); q = np.asarray(q_pred, float)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
