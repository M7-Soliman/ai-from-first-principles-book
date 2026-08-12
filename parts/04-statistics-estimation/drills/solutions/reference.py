"""
Reference solutions for the Part V drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))

Z = {0.95: 1.959963985, 0.99: 2.575829304, 0.90: 1.644853627}


def _z(level):
    if level in Z:
        return Z[level]
    # good enough for anything else: invert Phi by bisection
    lo, hi = 0.0, 10.0
    target = 0.5 * (1 + level)
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------ estimators ----

def sample_variance(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return float(((x - x.mean()) ** 2).sum() / (len(x) - ddof))


def standard_error(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(sample_variance(x, ddof=1) / len(x)))


def mse_decomposition(estimates, truth):
    e = np.asarray(estimates, dtype=float)
    bias2 = float((e.mean() - truth) ** 2)
    var = float(e.var())                       # ddof=0: this IS the population
    mse = float(((e - truth) ** 2).mean())
    return bias2, var, mse


def shrinkage_estimator(x, toward=0.0, weight=0.5):
    return float((1 - weight) * np.mean(x) + weight * toward)


# ------------------------------------------------------------- intervals ----

def confidence_interval(x, level=0.95):
    m, se, z = float(np.mean(x)), standard_error(x), _z(level)
    return (m - z * se, m + z * se)


def coverage(true_mean, true_sd, n, level=0.95, trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        lo, hi = confidence_interval(rng.normal(true_mean, true_sd, n), level)
        hits += lo <= true_mean <= hi
    return hits / trials


def bootstrap_ci(data, stat, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(data)
    reps = [stat(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------- testing ----

def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def two_sample_pvalue(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    se = math.sqrt(standard_error(a) ** 2 + standard_error(b) ** 2)
    if se == 0:
        return 1.0
    z = (a.mean() - b.mean()) / se
    return float(2 * (1 - _phi(abs(z))))


def pvalues_under_null(n=30, trials=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([two_sample_pvalue(rng.normal(size=n), rng.normal(size=n))
                     for _ in range(trials)])


def power(effect, n=30, trials=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(effect, 1.0, n)
        hits += two_sample_pvalue(a, b) < alpha
    return hits / trials


def bonferroni(pvalues, alpha=0.05):
    p = np.asarray(pvalues, dtype=float)
    return p < alpha / len(p)


def best_of_k_bias(k, trials=2000, n=200, seed=0):
    rng = np.random.default_rng(seed)
    se = math.sqrt(0.25 / n)                   # SE of an accuracy near 0.5
    best = rng.normal(0.5, se, size=(trials, k)).max(axis=1)
    return float(best.mean() - 0.5)


def paired_difference_ci(a, b, level=0.95):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return confidence_interval(d, level)


def unpaired_difference_ci(a, b, level=0.95):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a.mean() - b.mean()
    se = math.sqrt(standard_error(a) ** 2 + standard_error(b) ** 2)
    z = _z(level)
    return (diff - z * se, diff + z * se)


# ----------------------------------------------------------- estimation -----

def map_bernoulli(heads, n, prior_a=1.0, prior_b=1.0):
    return float((heads + prior_a - 1) / (n + prior_a + prior_b - 2))


def ridge_weights(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def bias_variance_curve(degrees, n_train=25, n_sets=200, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    truth = lambda t: np.sin(2 * np.pi * t)          # noqa: E731
    grid = np.linspace(0, 1, 60)
    f_true = truth(grid)

    bias2, var, total = [], [], []
    for d in degrees:
        preds = np.empty((n_sets, len(grid)))
        for s in range(n_sets):
            xt = rng.uniform(0, 1, n_train)
            yt = truth(xt) + rng.normal(0, noise, n_train)
            coef = np.polyfit(xt, yt, d)
            preds[s] = np.polyval(coef, grid)
        mean_pred = preds.mean(axis=0)
        bias2.append(float(((mean_pred - f_true) ** 2).mean()))
        var.append(float(preds.var(axis=0).mean()))
        total.append(float(((preds - f_true) ** 2).mean()))
    return np.array(bias2), np.array(var), np.array(total)


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
