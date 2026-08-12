"""
Reference solutions for the Part III drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ------------------------------------------------------------- stability ----

def machine_epsilon(dtype=np.float64):
    one = dtype(1.0)
    eps = dtype(1.0)
    while one + eps / dtype(2.0) != one:
        eps = dtype(eps / dtype(2.0))
    return float(eps)


def logsumexp(z):
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return float(m + np.log(np.exp(z - m).sum()))


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    return z - logsumexp(z)                # never exp-then-log


def quadratic_roots(a, b, c):
    a, b, c = float(a), float(b), float(c)
    disc = np.sqrt(b * b - 4 * a * c)
    # sign(b) makes the two terms ADD, so nothing cancels
    sign = 1.0 if b >= 0 else -1.0
    q = -0.5 * (b + sign * disc)
    r1 = q / a
    r2 = c / q if q != 0 else 0.0
    return tuple(sorted((r1, r2)))


def kahan_sum(xs):
    total = 0.0
    comp = 0.0                              # running estimate of what was lost
    for x in np.asarray(xs, dtype=float):
        y = x - comp
        t = total + y
        comp = (t - total) - y              # the bits that fell off
        total = t
    return float(total)


def welford_var(xs):
    n = 0
    mean = 0.0
    m2 = 0.0                                # sum of squared deviations
    for x in np.asarray(xs, dtype=float):
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)                # uses both old and new mean
    return float(m2 / n) if n else 0.0


# ----------------------------------------------------------- conditioning ---

def make_matrix(m, n, kappa, rng):
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -np.log10(kappa), n)
    return U @ np.diag(s) @ V.T


def solve_normal(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)      # squares the conditioning


def _householder_qr(A):
    """Householder QR.

    Modified Gram-Schmidt is the textbook algorithm and is NOT good enough here:
    its loss of orthogonality grows like eps * kappa, so at kappa = 1e11 the
    computed Q is visibly non-orthogonal and the solve returns garbage. Householder
    reflections hold |Q^T Q - I| at ~1e-15 regardless of conditioning, which is why
    every serious library uses them.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k].copy()
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (1.0 if x[0] >= 0 else -1.0)   # sign avoids cancellation
        v = x + e
        nv = np.linalg.norm(v)
        if nv < 1e-300:
            continue
        v = v / nv
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q[:, :n], R[:n, :]


def _back_sub(R, y):
    """Solve an upper-triangular system by back substitution."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def solve_qr(A, b):
    Q, R = _householder_qr(A)
    return _back_sub(R, Q.T @ np.asarray(b, dtype=float))


def solve_svd(A, b, rcond=1e-12):
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    keep = s > rcond * s.max()
    inv = np.zeros_like(s)
    inv[keep] = 1.0 / s[keep]              # truncate, never divide by ~0
    return Vt.T @ (inv * (U.T @ np.asarray(b, dtype=float)))


# ------------------------------------------------------------ optimisers ----

def _run(step, x0, steps):
    x = np.asarray(x0, dtype=float).copy()
    path = [x.copy()]
    for t in range(1, steps + 1):
        x = step(x, t)
        path.append(x.copy())
        if not np.all(np.isfinite(x)):
            break
    return np.array(path)


def gd(grad, x0, lr, steps):
    return _run(lambda x, t: x - lr * np.asarray(grad(x), dtype=float), x0, steps)


def gd_momentum(grad, x0, lr, steps, beta=0.9):
    v = np.zeros_like(np.asarray(x0, dtype=float))

    def step(x, t):
        nonlocal v
        v = beta * v + np.asarray(grad(x), dtype=float)
        return x - lr * v
    return _run(step, x0, steps)


def newton(grad, hess, x0, steps):
    return _run(lambda x, t: x - np.linalg.solve(np.atleast_2d(hess(x)),
                                                 np.asarray(grad(x), dtype=float)),
                x0, steps)


def rmsprop(grad, x0, lr, steps, rho=0.9, eps=1e-8):
    s = np.zeros_like(np.asarray(x0, dtype=float))

    def step(x, t):
        nonlocal s
        g = np.asarray(grad(x), dtype=float)
        s = rho * s + (1 - rho) * g ** 2
        return x - lr * g / (np.sqrt(s) + eps)
    return _run(step, x0, steps)


def adam(grad, x0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    v = np.zeros_like(np.asarray(x0, dtype=float))
    s = np.zeros_like(v)

    def step(x, t):
        nonlocal v, s
        g = np.asarray(grad(x), dtype=float)
        v = b1 * v + (1 - b1) * g
        s = b2 * s + (1 - b2) * g ** 2
        vh = v / (1 - b1 ** t)             # bias correction: both start at zero
        sh = s / (1 - b2 ** t)
        return x - lr * vh / (np.sqrt(sh) + eps)
    return _run(step, x0, steps)


def backtracking_line_search(f, grad_f, x, d, alpha0=1.0, c=1e-4, rho=0.5,
                             max_iter=50):
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    fx = f(x)
    slope = float(np.asarray(grad_f(x), dtype=float) @ d)
    a = alpha0
    for _ in range(max_iter):
        if f(x + a * d) <= fx + c * a * slope:
            return a
        a *= rho
    return a


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
