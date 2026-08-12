"""
Reference solutions for the Part XXIII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import itertools
from math import comb, ceil, log, sqrt

import numpy as np
from scipy.optimize import linprog

import drills


# --------------------------------------------------------------- concentration

def hoeffding_bound(n, eps, span=1.0):
    return float(min(2 * np.exp(-2 * n * eps ** 2 / span ** 2), 1.0))


def bernstein_bound(n, eps, var, span=1.0):
    return float(min(2 * np.exp(-n * eps ** 2 / (2 * var + 2 * span * eps / 3)), 1.0))


def empirical_tail(samples, mu, eps):
    s = np.asarray(samples, float)
    return float(np.mean(np.abs(s - mu) >= eps))


def samples_needed(eps, delta, span=1.0):
    # 2 exp(-2 n eps^2 / span^2) <= delta
    return int(ceil(span ** 2 * log(2 / delta) / (2 * eps ** 2)))


# ---------------------------------------------------------- uniform convergence

def finite_class_bound(n_hypotheses, n, delta=0.05):
    return float(sqrt((log(n_hypotheses) + log(2 / delta)) / (2 * n)))


def sauer_bound(m, vcdim):
    if vcdim >= m:
        return 2 ** m
    return sum(comb(m, i) for i in range(vcdim + 1))


def vc_bound(vcdim, n, delta=0.05):
    # the log term goes negative once vcdim > 2n; flooring it keeps the
    # expression real and, in that regime, reduces the bound to sqrt(vcdim/n)
    inner = vcdim * (max(np.log(2 * n / vcdim), 0.0) + 1) + np.log(4 / delta)
    return float(np.sqrt(inner / n))


def is_vacuous(bound):
    return bool(bound >= 1.0)


def is_separable(X, y):
    X = np.asarray(X, float)
    n, d = X.shape
    s = np.where(np.asarray(y) > 0, 1.0, -1.0)
    A = -np.column_stack([X * s[:, None], s])
    res = linprog(c=np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n),
                  bounds=[(None, None)] * (d + 1), method="highs")
    return res.status == 0


def shatters(X):
    m = len(X)
    return all(is_separable(X, np.array(lab))
               for lab in itertools.product([0, 1], repeat=m))


# ---------------------------------------------------------------- complexity

def rademacher_linear(X, bound=1.0, n_draws=200, seed=0):
    # sup over ||w|| <= B of (1/n) sum_i sigma_i <w, x_i> = (B/n) ||sum_i sigma_i x_i||
    X = np.asarray(X, float)
    n = len(X)
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n_draws):
        s = rng.choice([-1.0, 1.0], size=n)
        tot += np.linalg.norm(s @ X)
    return float(bound * tot / n_draws / n)


def margin_fraction(scores, y, gamma):
    s = np.asarray(scores, float); yy = np.asarray(y, float)
    return float(np.mean(yy * s >= gamma))


# ------------------------------------------------------------ online learning

def regret(losses, choices):
    L = np.asarray(losses, float)
    c = np.asarray(choices, int)
    mine = L[np.arange(len(c)), c].sum()
    return float(mine - L.sum(axis=0).min())


def follow_the_leader(losses):
    L = np.asarray(losses, float)
    T, K = L.shape
    out, cum = [], np.zeros(K)
    for t in range(T):
        out.append(0 if t == 0 else int(np.argmin(cum)))
        cum += L[t]
    return np.array(out)


def multiplicative_weights(losses, eta):
    L = np.asarray(losses, float)
    T, K = L.shape
    w = np.ones(K)
    total = 0.0
    for t in range(T):
        p = w / w.sum()
        total += float(p @ L[t])
        w *= np.exp(-eta * L[t])
    return total


# ------------------------------------------------------------ distribution shift

def importance_weights(x, p_mean, q_mean, sigma=1.0, clip=None):
    x = np.asarray(x, float)
    logw = (-((x - q_mean) ** 2) + ((x - p_mean) ** 2)) / (2 * sigma ** 2)
    w = np.exp(logw)
    return np.clip(w, 0, clip) if clip is not None else w


def weighted_least_squares(X, y, w):
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float)
    r = np.sqrt(w)[:, None]
    return np.linalg.lstsq(X * r, y * np.sqrt(w), rcond=None)[0]


def label_shift_weights(confusion, target_pred_dist):
    C = np.asarray(confusion, float)
    q = np.asarray(target_pred_dist, float)
    # C is p(pred i | true j); source prior is recoverable from the column sums
    # only if C was built from counts, so take it as given via the identity
    # p(pred) = C @ p(true).
    target_prior = np.linalg.solve(C, q)
    target_prior = np.clip(target_prior, 1e-12, None)
    target_prior = target_prior / target_prior.sum()
    source_prior = np.full(len(q), 1.0 / len(q))
    return target_prior / source_prior


def worst_group_accuracy(pred, y, group):
    pred = np.asarray(pred); y = np.asarray(y); group = np.asarray(group)
    accs = {}
    for g in np.unique(group):
        m = group == g
        accs[g] = float(np.mean(pred[m] == y[m])) if m.sum() else float("nan")
    return float(min(accs.values())), accs




# ------------------------------------------------------- movement G: width

def ntk_gram(jacobian):
    J = np.asarray(jacobian, float)
    return J @ J.T


def relative_kernel_change(K0, K1):
    K0 = np.asarray(K0, float)
    K1 = np.asarray(K1, float)
    return float(np.linalg.norm(K1 - K0) / np.linalg.norm(K0))


def init_scales(scheme, m, d):
    if scheme == "standard":
        return (1.0 / np.sqrt(d), 1.0 / np.sqrt(m), 1.0, 1.0)
    if scheme == "ntk":
        # the 1/sqrt(m) lives in the forward pass, so the init is unit scale
        return (1.0, 1.0, 1.0, 1.0)
    if scheme == "mup":
        return (1.0 / np.sqrt(d), 1.0 / m, float(m), 1.0 / m)
    raise ValueError(scheme)


def loglog_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def is_width_stable(widths, values, tol=0.25):
    return bool(abs(loglog_slope(widths, values)) <= tol)


def _gh(n=201):
    """Gauss-Hermite nodes and weights for E[f(z)], z ~ N(0,1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def _corr_map(c, q, sigma_w2, sigma_b2):
    """One step of the correlation map at variance q, by two-dimensional
    Gauss-Hermite quadrature over a correlated pair."""
    x, w = _gh()
    z1 = x[:, None]
    z2 = (c * x[:, None] + np.sqrt(max(1.0 - c * c, 0.0)) * x[None, :])
    W = w[:, None] * w[None, :]
    s = np.sqrt(q)
    num = float((W * np.tanh(s * z1) * np.tanh(s * z2)).sum())
    return sigma_w2 * num + sigma_b2


def _var_fixed_point(sigma_w2, sigma_b2, iters=300):
    x, w = _gh()
    q = 1.0
    for _ in range(iters):
        q = sigma_w2 * float((w * np.tanh(np.sqrt(q) * x) ** 2).sum()) + sigma_b2
    return q


def correlation_fixed_point(sigma_w2, sigma_b2, depth=400, tol=1e-9):
    q = _var_fixed_point(sigma_w2, sigma_b2)
    c = 0.5
    for _ in range(depth):
        nxt = _corr_map(c, q, sigma_w2, sigma_b2) / q
        nxt = float(np.clip(nxt, -1.0, 1.0))
        if abs(nxt - c) < tol:
            return nxt
        c = nxt
    return c


def chi1(sigma_w2, sigma_b2):
    q = _var_fixed_point(sigma_w2, sigma_b2)
    x, w = _gh()
    dphi2 = (1.0 - np.tanh(np.sqrt(q) * x) ** 2) ** 2
    return float(sigma_w2 * (w * dphi2).sum())


def critical_sigma_w2(sigma_b2, lo=0.1, hi=10.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi1(mid, sigma_b2) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def transfer_spread(best_lrs):
    b = np.asarray(best_lrs, float)
    return float(b.max() / b.min())


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
