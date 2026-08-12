"""
Reference solutions for the Part XXII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------ the graph tools

def parents(graph, node):
    return set(graph.get(node, set()))


def _children(graph, node):
    return {n for n, ps in graph.items() if node in ps}


def descendants(graph, node):
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        for c in _children(graph, cur):
            if c not in seen:
                seen.add(c); stack.append(c)
    return seen


def _neighbours(graph, node):
    return parents(graph, node) | _children(graph, node)


def paths(graph, a, b):
    out = []

    def walk(cur, path, seen):
        if cur == b:
            out.append(list(path)); return
        for nxt in _neighbours(graph, cur):
            if nxt in seen:
                continue
            seen.add(nxt); path.append(nxt)
            walk(nxt, path, seen)
            path.pop(); seen.remove(nxt)

    walk(a, [a], {a})
    return out


def is_collider(graph, path, i):
    if i <= 0 or i >= len(path) - 1:
        return False
    node = path[i]
    ps = parents(graph, node)
    return path[i - 1] in ps and path[i + 1] in ps


def path_is_blocked(graph, path, given):
    given = set(given)
    for i in range(1, len(path) - 1):
        node = path[i]
        if is_collider(graph, path, i):
            # a collider blocks UNLESS it or a descendant is conditioned on
            if node not in given and not (descendants(graph, node) & given):
                return True
        else:
            if node in given:
                return True
    return False


def d_separated(graph, a, b, given=frozenset()):
    return all(path_is_blocked(graph, p, given) for p in paths(graph, a, b))


def satisfies_backdoor(graph, x, y, z):
    z = set(z)
    if z & descendants(graph, x):
        return False
    px = parents(graph, x)
    for p in paths(graph, x, y):
        if len(p) > 1 and p[1] in px:            # a path leaving x through an arrow INTO x
            if not path_is_blocked(graph, p, z):
                return False
    return True


# ------------------------------------------------------------- the estimators

def naive_difference(y, t):
    y = np.asarray(y, float); t = np.asarray(t).astype(bool)
    return float(y[t].mean() - y[~t].mean())


def backdoor_adjust(y, t, z, bins=10):
    y = np.asarray(y, float); t = np.asarray(t).astype(bool); z = np.asarray(z, float)
    edges = np.quantile(z, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(z, edges[1:-1]), 0, bins - 1)
    tot, wsum = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0 or t[m].sum() == 0 or (~t[m]).sum() == 0:
            continue
        d = y[m & t].mean() - y[m & ~t].mean()
        # weight by the bin's share of EVERYONE — that is what the backdoor
        # formula says. Weighting by its share of the treated gives the
        # conditional back, which is the quantity we were trying to leave.
        w = m.sum()
        tot += w * d; wsum += w
    return float(tot / wsum) if wsum else float("nan")


def fit_propensity(z, t, iters=25):
    z = np.asarray(z, float)
    X = np.column_stack([np.ones(len(z)), z.reshape(len(z), -1)])
    t = np.asarray(t, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (t - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def ipw_estimate(y, t, e, clip=0.0):
    y = np.asarray(y, float); t = np.asarray(t).astype(bool); e = np.asarray(e, float)
    if clip:
        e = np.clip(e, clip, 1 - clip)
    w1 = 1.0 / e[t]
    w0 = 1.0 / (1 - e[~t])
    return float(np.sum(w1 * y[t]) / np.sum(w1) - np.sum(w0 * y[~t]) / np.sum(w0))


def aipw_estimate(y, t, e, m1, m0):
    y = np.asarray(y, float); t = np.asarray(t).astype(float)
    e = np.asarray(e, float); m1 = np.asarray(m1, float); m0 = np.asarray(m0, float)
    return float(np.mean(m1 - m0
                         + t * (y - m1) / e
                         - (1 - t) * (y - m0) / (1 - e)))


def effective_sample_size(w):
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return 0.0
    return float((w.sum() ** 2) / np.sum(w ** 2) / len(w))


def overlap_range(e, t):
    e = np.asarray(e, float); t = np.asarray(t).astype(bool)
    if t.sum() == 0 or (~t).sum() == 0:
        return float("nan"), float("nan")
    lo = max(e[t].min(), e[~t].min())
    hi = min(e[t].max(), e[~t].max())
    if lo > hi:
        return float("nan"), float("nan")
    return float(lo), float(hi)


# --------------------------------------------------------------- the designs

def _ols(X, y):
    X = np.column_stack([np.ones(len(X)), np.asarray(X).reshape(len(X), -1)])
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def two_stage_least_squares(y, x, z):
    a = _ols(z, x)
    xhat = a[0] + np.asarray(z, float) * a[1]
    return float(_ols(xhat, y)[1])


def first_stage_strength(x, z):
    x = np.asarray(x, float); z = np.asarray(z, float)
    a = _ols(z, x); fit = a[0] + a[1] * z
    ss_model = np.sum((fit - x.mean()) ** 2)
    ss_resid = np.sum((x - fit) ** 2)
    df = len(x) - 2
    return float(ss_model / (ss_resid / df))


def did_estimate(y, treated, after):
    y = np.asarray(y, float)
    tr = np.asarray(treated).astype(bool); af = np.asarray(after).astype(bool)
    return float((y[tr & af].mean() - y[tr & ~af].mean())
                 - (y[~tr & af].mean() - y[~tr & ~af].mean()))


def rdd_estimate(y, running, cutoff, bandwidth):
    y = np.asarray(y, float); r = np.asarray(running, float)
    m = np.abs(r - cutoff) <= bandwidth
    hi = m & (r >= cutoff); lo = m & (r < cutoff)
    if hi.sum() < 2 or lo.sum() < 2:
        return float("nan")
    ch = _ols(r[hi] - cutoff, y[hi])
    cl = _ols(r[lo] - cutoff, y[lo])
    return float(ch[0] - cl[0])          # both intercepts sit AT the cutoff


# ---------------------------------------------------- policies and transfer

def off_policy_value(rewards, actions, logged_prob, target_policy_prob, clip=None):
    r = np.asarray(rewards, float)
    ratio = np.asarray(target_policy_prob, float) / np.asarray(logged_prob, float)
    if clip is not None:
        ratio = np.minimum(ratio, clip)
    return float(np.mean(ratio * r))


def self_normalised_value(rewards, actions, logged_prob, target_policy_prob):
    r = np.asarray(rewards, float)
    ratio = np.asarray(target_policy_prob, float) / np.asarray(logged_prob, float)
    s = ratio.sum()
    return float(np.sum(ratio * r) / s) if s > 0 else float("nan")


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
