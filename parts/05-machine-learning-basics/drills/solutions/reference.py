"""
Reference solutions for the Part VI drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ---------------------------------------------------------------- splits ----

def train_val_test_split(n, fracs=(0.7, 0.15, 0.15), seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    return idx[:a], idx[a:b], idx[b:]


def kfold_indices(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, val))
    return out


def cross_val_score(fit_predict, X, y, k=5, seed=0):
    scores = []
    for tr, va in kfold_indices(len(X), k, seed):
        pred = fit_predict(X[tr], y[tr], X[va])
        scores.append(float(np.mean(np.asarray(pred) == y[va])))
    return float(np.mean(scores))


# -------------------------------------------------------------- the models --

def fit_linear_regression(X, y):
    w, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return w


def fit_ridge(X, y, lam):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))       # exponent <= 0
    ez = np.exp(z[neg])                            # exponent < 0
    out[neg] = ez / (1.0 + ez)
    return out


def fit_logistic(X, y, lr=0.1, steps=2000):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(steps):
        grad = X.T @ (sigmoid(X @ w) - y) / n      # the sigma' factor cancelled
        w -= lr * grad
    return w


def knn_predict(X_train, y_train, X_test, k=5):
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    d2 = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    nn = np.argsort(d2, axis=1)[:, :k]
    votes = np.asarray(y_train)[nn]
    return np.array([np.bincount(v).argmax() for v in votes])


def kmeans(X, k, seed=0, iters=50):
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                centres[j] = X[labels == j].mean(axis=0)
    return centres, labels


def pca_fit(X, k):
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD route: already descending, and never forms X'X
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].T
    explained = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return components, mean, explained


def pca_reconstruct(X, components, mean):
    Xc = np.asarray(X, float) - mean
    return (Xc @ components) @ components.T + mean


# --------------------------------------------------------------- kernels ----

def polynomial_kernel(A, B, p=2):
    return (np.asarray(A, float) @ np.asarray(B, float).T) ** p


def quadratic_feature_map(A):
    A = np.atleast_2d(np.asarray(A, float))
    x1, x2 = A[:, 0], A[:, 1]
    return np.stack([x1 ** 2, np.sqrt(2.0) * x1 * x2, x2 ** 2], axis=1)


def rbf_kernel(A, B, gamma=1.0):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


# --------------------------------------------------------- generalisation ---

def learning_curve(fit_predict, X, y, sizes, seed=0):
    tr_idx, _, te_idx = train_val_test_split(len(X), (0.7, 0.0, 0.3), seed)
    train_err, test_err = [], []
    for n in sizes:
        sub = tr_idx[:n]
        p_tr = fit_predict(X[sub], y[sub], X[sub])
        p_te = fit_predict(X[sub], y[sub], X[te_idx])
        train_err.append(float(np.mean(np.asarray(p_tr) != y[sub])))
        test_err.append(float(np.mean(np.asarray(p_te) != y[te_idx])))
    return np.array(train_err), np.array(test_err)


def intrinsic_dimension(X, threshold=0.95):
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    frac = np.cumsum(s ** 2) / (s ** 2).sum()
    return int(np.searchsorted(frac, threshold) + 1)


def nearest_farthest_ratio(n, d, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(n, d))
    q = rng.uniform(size=d)
    dist = np.sqrt(((pts - q) ** 2).sum(axis=1))
    return float(dist.max() / dist.min())


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
