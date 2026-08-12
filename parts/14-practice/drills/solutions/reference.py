"""
Reference solutions for the Part XIV drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ------------------------------------------------------------------- metrics ----

def confusion_counts(y_true, scores, threshold):
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(scores) >= threshold
    return (int(np.sum(p & y)), int(np.sum(p & ~y)),
            int(np.sum(~p & y)), int(np.sum(~p & ~y)))


def precision_recall(y_true, scores, threshold):
    tp, fp, fn, _ = confusion_counts(y_true, scores, threshold)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def f1(precision, recall):
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def pr_curve(y_true, scores, n_points=200):
    s = np.asarray(scores, float)
    ths = np.linspace(s.min(), s.max(), n_points)
    ps, rs = [], []
    for t in ths:
        p, r = precision_recall(y_true, s, t)
        ps.append(p)
        rs.append(r)
    return np.array(ps), np.array(rs), ths


def coverage_accuracy(y_true, y_pred, confidence, threshold):
    m = np.asarray(confidence) >= threshold
    cov = float(m.mean())
    if m.sum() == 0:
        return 0.0, 1.0
    acc = float((np.asarray(y_pred)[m] == np.asarray(y_true)[m]).mean())
    return cov, acc


# --------------------------------------------------------------- calibration ----

def reliability_bins(confidence, correct, n_bins=10, min_count=1):
    c = np.asarray(confidence, float)
    k = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bc, ba, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.sum() < min_count:
            continue
        bc.append(float(c[m].mean()))
        ba.append(float(k[m].mean()))
        bn.append(int(m.sum()))
    return np.array(bc), np.array(ba), np.array(bn)


def expected_calibration_error(confidence, correct, n_bins=10):
    bc, ba, bn = reliability_bins(confidence, correct, n_bins)
    if len(bc) == 0:
        return 0.0
    return float(np.sum(bn * np.abs(bc - ba)) / np.sum(bn))


def softmax_T(logits, T=1.0):
    z = np.asarray(logits, float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits, y, lo=0.05, hi=100.0, iters=60):
    logits = np.asarray(logits, float)
    y = np.asarray(y, int)

    def nll(T):
        p = softmax_T(logits, T)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    a, b = np.log(lo), np.log(hi)
    gr = (np.sqrt(5.0) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(np.exp(c)) < nll(np.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return float(np.exp((a + b) / 2))


# ---------------------------------------------------------- gradient checking ----

def central_difference(f, x, eps):
    return (f(x + eps / 2.0) - f(x - eps / 2.0)) / eps


def complex_step(f, x, eps):
    return float(np.imag(f(complex(x, eps))) / eps)


def projected_check(g, jac, x, seed=0, eps=1e-6, trials=3):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    J = np.asarray(jac(x), float)
    n, m = J.shape
    worst = 0.0
    for _ in range(trials):
        u = r.normal(size=n)
        v = r.normal(size=m)
        num = (u @ g(x + eps / 2 * v) - u @ g(x - eps / 2 * v)) / eps
        ana = u @ (J @ v)
        denom = max(abs(ana), 1e-12)
        worst = max(worst, abs(num - ana) / denom)
    return float(worst)


# ------------------------------------------------------------------ diagnosis ----

def diagnose(train_err, test_err, target, tol=0.05):
    if train_err > test_err + tol:
        return "measurement-error"
    if train_err <= target + tol and test_err <= target + tol:
        return "working"
    if train_err <= target + tol:
        return "overfitting-or-broken-eval"
    return "underfitting-or-bug"


def update_ratio(params, updates):
    out = []
    for p, u in zip(params, updates):
        pn = float(np.linalg.norm(np.asarray(p, float)))
        un = float(np.linalg.norm(np.asarray(u, float)))
        if pn == 0.0:
            out.append(np.inf if un > 0 else 0.0)
        else:
            out.append(un / pn)
    return out


def worst_mistakes(y_true, probs, k=10):
    y = np.asarray(y_true, int)
    p = np.asarray(probs, float)
    p_true = p[np.arange(len(y)), y]
    return np.argsort(p_true)[:k]


# --------------------------------------------------------------------- search ----

def log_uniform(lo, hi, size, seed=0):
    r = np.random.default_rng(seed)
    return np.exp(r.uniform(np.log(lo), np.log(hi), size))


def grid_points(bounds, budget):
    D = len(bounds)
    n = max(2, int(np.floor(budget ** (1.0 / D))))
    axes = [np.linspace(lo, hi, n) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def distinct_values_per_axis(points, decimals=9):
    P = np.asarray(points, float)
    return [len(np.unique(np.round(P[:, j], decimals))) for j in range(P.shape[1])]


def successive_halving(observe, n_arms, total_budget, eta=2):
    alive = list(range(n_arms))
    spent, e = 0, 1
    while len(alive) > 1 and spent < total_budget:
        spent += len(alive) * e
        obs = {a: observe(a, e) for a in alive}
        keep = max(1, len(alive) // eta)
        alive = sorted(alive, key=lambda a: -obs[a])[:keep]
        e *= eta
    return alive[0]


def best_of_k_inflation(true_score, noise_sd, k, trials=2000, seed=0):
    r = np.random.default_rng(seed)
    draws = true_score + r.normal(0, noise_sd, (trials, k))
    return float(draws.max(axis=1).mean() - true_score)


# ------------------------------------------------------------------ reporting ----

def mean_and_spread(values):
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else float("inf")
    return mean, sd, half


def difference_is_detectable(a_runs, b_runs):
    a = np.asarray(a_runs, float)
    b = np.asarray(b_runs, float)
    if len(a) < 2 or len(b) < 2:
        return False
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return bool(abs(a.mean() - b.mean()) > 1.96 * se)


def smallest_detectable_difference(spread, n_runs, z=1.96):
    if n_runs < 2:
        return float("inf")
    return float(z * spread * np.sqrt(2.0 / n_runs))


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
