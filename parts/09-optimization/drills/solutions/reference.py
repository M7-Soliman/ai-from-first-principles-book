"""
Reference solutions for the Part IX drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# --------------------------------------------------------- instruments ----

def gHg(grad_fn, theta, h_rel=1e-5):
    theta = np.asarray(theta, float)
    g = np.asarray(grad_fn(theta), float)
    n = np.linalg.norm(g)
    if n == 0:
        return 0.0
    h = h_rel / n
    g2 = np.asarray(grad_fn(theta + h * g), float)
    return float((g2 - g) @ g / h)


def step_ceiling(grad_fn, theta):
    g = np.asarray(grad_fn(np.asarray(theta, float)), float)
    q = gHg(grad_fn, theta)
    if q <= 0:
        return np.inf
    return float(2.0 * (g @ g) / q)


def gradient_error_vs_batch(X, y, sizes, trials, rng):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(X)
    g_full = -X.T @ y / n
    out = []
    for b in sizes:
        errs = []
        for _ in range(trials):
            idx = rng.integers(0, n, int(b))
            g = -X[idx].T @ y[idx] / int(b)
            errs.append(np.linalg.norm(g - g_full) / np.linalg.norm(g_full))
        out.append(float(np.mean(errs)))
    return np.array(out)


def newton_step_error_vs_batch(X, y, sizes, trials, rng):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(X)
    H = X.T @ X / n
    step_full = np.linalg.solve(H, -X.T @ y / n)
    out = []
    for b in sizes:
        errs = []
        for _ in range(trials):
            idx = rng.integers(0, n, int(b))
            g = -X[idx].T @ y[idx] / int(b)
            errs.append(np.linalg.norm(np.linalg.solve(H, g) - step_full)
                        / np.linalg.norm(step_full))
        out.append(float(np.mean(errs)))
    return np.array(out)


# ------------------------------------------------------------ symmetry ----

def relu_scaling_loss(x, y, w1, w2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.mean((w2 * np.maximum(w1 * x, 0.0) - y) ** 2))


def hyperbola_spread(x, y, c, w1_values):
    vals = np.array([relu_scaling_loss(x, y, float(w), c / float(w))
                     for w in w1_values])
    return float((vals.max() - vals.min()) / vals.mean())


# --------------------------------------------------------------- cliffs ----

def deep_product_loss(w, L, target=1.0):
    return 0.5 * (np.asarray(w, float) ** L - target) ** 2


def deep_product_grad(w, L, target=1.0):
    w = np.asarray(w, float)
    return (w ** L - target) * L * w ** (L - 1)


def clipped_step(w, grad, lr, threshold):
    g = np.asarray(grad, float)
    n = np.linalg.norm(g)
    scale = min(1.0, threshold / n) if n > 0 else 1.0
    return np.asarray(w, float) - lr * scale * g


# ---------------------------------------------------- momentum, Nesterov ----

def momentum_path(grad_fn, theta0, lr, alpha, steps, nesterov=False):
    theta = np.array(theta0, dtype=float)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        g = np.asarray(grad_fn(theta + alpha * v if nesterov else theta), float)
        v = alpha * v - lr * g
        theta = theta + v
        path.append(theta.copy())
    return np.array(path)


def iterations_to_reach(path, tol):
    d = np.linalg.norm(np.asarray(path, float), axis=1)
    idx = np.where(d < tol)[0]
    return int(idx[0]) if len(idx) else None


# ------------------------------------------------------- initialisation ----

def variance_profile(widths, rule, batch, rng):
    Ws = []
    for m, n in zip(widths[:-1], widths[1:]):
        if rule == "fan_in":
            s = 1.0 / np.sqrt(m)
        elif rule == "fan_out":
            s = 1.0 / np.sqrt(n)
        elif rule == "glorot":
            s = np.sqrt(2.0 / (m + n))
        else:
            raise ValueError(rule)
        Ws.append(rng.normal(size=(m, n)) * s)
    a = rng.normal(size=(batch, widths[0]))
    fwd = [float(a.var())]
    for W in Ws:
        a = a @ W
        fwd.append(float(a.var()))
    d = rng.normal(size=(batch, widths[-1]))
    bwd = [float(d.var())]
    for W in reversed(Ws):
        d = d @ W.T
        bwd.append(float(d.var()))
    return np.array(fwd), np.array(bwd[::-1])


def output_bias_for_marginal(labels, n_classes):
    counts = np.bincount(np.asarray(labels, int), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1e-12)
    return np.log(counts / counts.sum())


# --------------------------------------------------------- second order ----

def damped_newton_step(H, g, alpha):
    H = np.asarray(H, float)
    return -np.linalg.solve(H + alpha * np.eye(H.shape[0]), np.asarray(g, float))


def steepest_descent_residuals(A, b, iters):
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.zeros_like(b)
    out = [float(np.linalg.norm(b - A @ x))]
    for _ in range(iters):
        g = A @ x - b
        denom = g @ A @ g
        if denom == 0:
            out.append(float(np.linalg.norm(b - A @ x)))
            continue
        x = x - (g @ g) / denom * g
        out.append(float(np.linalg.norm(b - A @ x)))
    return np.array(out)


def conjugate_gradient_residuals(A, b, iters):
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    out = [float(np.linalg.norm(r))]
    for _ in range(iters):
        Ap = A @ p
        denom = p @ Ap
        if denom == 0:
            out.append(float(np.linalg.norm(b - A @ x)))
            continue
        al = (r @ r) / denom
        x = x + al * p
        rn = r - al * Ap
        rr = r @ r
        p = rn + (rn @ rn) / rr * p
        r = rn
        out.append(float(np.linalg.norm(b - A @ x)))
    return np.array(out)


# ------------------------------------------------ meta-algorithms ----------

def batchnorm_forward(H, gamma, beta, delta=1e-8):
    H = np.asarray(H, float)
    m = H.shape[0]
    mu = H.mean(axis=0)
    xc = H - mu
    var = (xc ** 2).mean(axis=0)
    sigma = np.sqrt(var + delta)
    xhat = xc / sigma
    out = gamma * xhat + beta
    return out, (xhat, xc, sigma, gamma, m)


def batchnorm_backward(dout, cache):
    xhat, xc, sigma, gamma, m = cache
    dout = np.asarray(dout, float)
    dbeta = dout.sum(axis=0)
    dgamma = (dout * xhat).sum(axis=0)
    dxhat = dout * gamma
    dH = (dxhat - dxhat.mean(axis=0) - xhat * (dxhat * xhat).mean(axis=0)) / sigma
    return dH, dgamma, dbeta


def coordinate_descent_sweeps(alpha, tol=1e-6, max_sweeps=100000):
    x1, x2 = 1.0, -1.0
    for k in range(1, max_sweeps + 1):
        x1 = x2 / (1.0 + alpha)
        x2 = x1 / (1.0 + alpha)
        if max(abs(x1), abs(x2)) < tol:
            return k
    return max_sweeps


def polyak_average(path, alpha=None):
    path = np.asarray(path, float)
    out = np.empty_like(path)
    if alpha is None:
        acc = np.zeros_like(path[0])
        for t in range(len(path)):
            acc = acc + path[t]
            out[t] = acc / (t + 1)
    else:
        e = path[0].copy()
        out[0] = e
        for t in range(1, len(path)):
            e = alpha * e + (1 - alpha) * path[t]
            out[t] = e
    return out


def gaussian_blur(values, dx, sigma):
    v = np.asarray(values, float)
    if sigma <= 0:
        return v.copy()
    half = int(5 * sigma / dx)
    k = np.arange(-half, half + 1) * dx
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(v, half, mode="edge"), w, mode="valid")


def local_descent(values, grid, start):
    v = np.asarray(values, float)
    j = int(np.argmin(np.abs(np.asarray(grid, float) - start)))
    while True:
        nbrs = [i for i in (j - 1, j + 1) if 0 <= i < len(v)]
        k = min(nbrs, key=lambda i: v[i])
        if v[k] >= v[j]:
            return j
        j = k


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
