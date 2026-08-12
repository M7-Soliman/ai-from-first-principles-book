"""
Reference solutions for the Part VIII drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


def ridge_weights(X, y, alpha):
    X = np.asarray(X, float); y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def kept_fraction(X, w_reg, w_ols):
    lam, Q = np.linalg.eigh(np.asarray(X, float).T @ np.asarray(X, float))
    return lam, (Q.T @ np.asarray(w_reg)) / (Q.T @ np.asarray(w_ols))


def predicted_kept_fraction(eigenvalues, alpha):
    lam = np.asarray(eigenvalues, float)
    return lam / (lam + alpha)


def lasso_weights(X, y, alpha, lr=0.004, iters=4000):
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        w -= lr * (X.T @ (X @ w - y)) / len(y)
        w = np.sign(w) * np.maximum(np.abs(w) - lr * alpha, 0.0)   # the prox
    return w


def effective_alpha_from_noise(n, sigma):
    return float(n) * float(sigma) ** 2


def noise_augmented_fit(X, y, sigma, reps=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xa = np.vstack([X + rng.normal(0, sigma, X.shape) for _ in range(reps)])
    ya = np.tile(y, reps)
    return np.linalg.lstsq(Xa, ya, rcond=None)[0]


def gd_path(X, y, lr, steps):
    X = np.asarray(X, float); y = np.asarray(y, float)
    H, b = X.T @ X, X.T @ y
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * (H @ w - b)
    return w


def early_stopping_kept_fraction(eigenvalues, lr, steps):
    lam = np.asarray(eigenvalues, float)
    return 1.0 - (1.0 - lr * lam) ** steps


def alpha_for_stopping(lr, steps):
    return 1.0 / (steps * lr)


def ensemble_error(v, c, k):
    return v / k + (k - 1) / k * c


def bag_predict(fit_predict, X, y, X_test, k=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    preds = []
    for _ in range(k):
        idx = rng.integers(0, n, n)          # with replacement
        preds.append(np.asarray(fit_predict(X[idx], y[idx], X_test), float))
    return np.mean(preds, axis=0)


def dropout_mask(shape, p, rng):
    return (rng.uniform(size=shape) > p).astype(float) / (1.0 - p)


def dropout_forward(X, W, b, p, rng):
    a = np.maximum(np.asarray(X, float) @ np.asarray(W, float).T + b, 0.0)
    m = dropout_mask(a.shape, p, rng)
    return a * m, m


def _sigmoid(z):
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[neg]); out[neg] = e / (1 + e)
    return out


def fgsm_perturbation(w, X, y, eps):
    X = np.asarray(X, float); w = np.asarray(w, float)
    resid = _sigmoid(X @ w) - np.asarray(y, float)
    return X + eps * np.sign(np.outer(resid, w))


def random_perturbation(X, eps, rng):
    return np.asarray(X, float) + eps * np.sign(rng.normal(size=np.shape(X)))


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
