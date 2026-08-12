"""
Reference solutions for the Part XI drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


def rnn_forward(x, U, W, V, b, c, h0):
    x = np.asarray(x, float)
    T = len(x)
    H = np.empty((T, len(h0)))
    O = np.empty((T, V.shape[0]))
    h = np.asarray(h0, float)
    for t in range(T):
        h = np.tanh(b + W @ h + U @ x[t])
        H[t] = h
        O[t] = c + V @ h
    return H, O


def softmax_rows(O):
    O = np.asarray(O, float)
    z = O - O.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sequence_loss(O, y):
    P = softmax_rows(O)
    y = np.asarray(y, int)
    return float(-np.log(P[np.arange(len(y)), y] + 1e-300).mean())


def bptt(x, y, U, W, V, b, c, h0, k=None):
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    T = len(x)
    H, O = rnn_forward(x, U, W, V, b, c, h0)
    P = softmax_rows(O)
    loss = float(-np.log(P[np.arange(T), y] + 1e-300).mean())

    dU = np.zeros_like(U); dW = np.zeros_like(W); dV = np.zeros_like(V)
    db = np.zeros_like(b); dc = np.zeros_like(c)
    dO = P.copy()
    dO[np.arange(T), y] -= 1.0
    dO /= T

    for t in range(T):                       # each output, backward in time
        dV += np.outer(dO[t], H[t])
        dc += dO[t]
        dh = V.T @ dO[t]
        stop = 0 if k is None else max(0, t - k + 1)
        for s in range(t, stop - 1, -1):
            da = dh * (1.0 - H[s] ** 2)
            db += da
            dU += np.outer(da, x[s])
            h_prev = H[s - 1] if s > 0 else np.asarray(h0, float)
            dW += np.outer(da, h_prev)
            dh = W.T @ da
    return loss, dict(dU=dU, dW=dW, dV=dV, db=db, dc=dc)


def jacobian_product_norm(W, T, nonlinearity="linear"):
    W = np.asarray(W, float)
    d = W.shape[0]
    out = [1.0]
    if nonlinearity == "linear":
        M = np.eye(d)
        for _ in range(T):
            M = W @ M
            out.append(float(np.linalg.norm(M, 2)))
        return out
    h = np.ones(d) / np.sqrt(d)
    M = np.eye(d)
    for _ in range(T):
        h = np.tanh(W @ h)
        M = (np.diag(1.0 - h ** 2) @ W) @ M
        out.append(float(np.linalg.norm(M, 2)))
    return out


def spectral_radius(W):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, float)))))


def scale_to_radius(W, rho):
    W = np.asarray(W, float)
    r = spectral_radius(W)
    return W * (rho / r) if r > 0 else W


def clip_by_norm(g, threshold):
    g = np.asarray(g, float)
    n = np.linalg.norm(g)
    return g * (threshold / n) if n > threshold else g.copy()


def clip_elementwise(g, threshold):
    return np.clip(np.asarray(g, float), -threshold, threshold)


def clip_angle(g, threshold, mode):
    g = np.asarray(g, float)
    c = clip_by_norm(g, threshold) if mode == "norm" else clip_elementwise(g, threshold)
    denom = np.linalg.norm(g) * np.linalg.norm(c)
    if denom == 0:
        return 0.0
    return float(np.arccos(np.clip((g @ c) / denom, -1.0, 1.0)))


def leaky_weights(alpha, T):
    t = np.arange(T)
    return (1.0 - alpha) * alpha ** t


def effective_memory(alpha):
    return float("inf") if alpha >= 1 else 1.0 / (1.0 - alpha)


def sigmoid(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def lstm_step(x, h_prev, s_prev, p):
    def gate(k):
        return p["W" + k] @ x + p["U" + k] @ h_prev + p["b" + k]
    f = sigmoid(gate("f"))
    i = sigmoid(gate("g"))
    q = sigmoid(gate("o"))
    s = f * s_prev + i * np.tanh(gate("c"))
    h = np.tanh(s) * q
    return h, s, dict(f=f, i=i, q=q)


def lstm_run(X, p, h0, s0):
    X = np.asarray(X, float)
    h, s = np.asarray(h0, float), np.asarray(s0, float)
    H, S, G = [], [], []
    for t in range(len(X)):
        h, s, g = lstm_step(X[t], h, s, p)
        H.append(h.copy()); S.append(s.copy()); G.append(g)
    return np.array(H), np.array(S), G


def retention(forget_mean, steps):
    return float(forget_mean) ** int(steps)


def rnn_params(n_in, n_h, n_out):
    return n_h * n_in + n_h * n_h + n_out * n_h + n_h + n_out


def table_params(k, tau):
    return float(k) ** tau


def unshared_params(n_in, n_h, n_out, tau):
    return tau * rnn_params(n_in, n_h, n_out)


import drills  # noqa: E402


# ------------------------------------------- movement G: what came after

def attention_flops(L, d):
    return 4.0 * L ** 2 * d + 6.0 * L * d ** 2


def linear_flops(L, d):
    return 4.0 * L * d ** 2 + 6.0 * L * d ** 2


def cost_crossover(d):
    # 4 L^2 d + 6 L d^2 = 10 L d^2  ->  4 L^2 d = 4 L d^2  ->  L = d.
    # The projection terms cancel, so the crossover is exactly the model
    # dimension. Comparing the quadratic term alone against the full linear
    # cost gives 2.5d and is the mistake that is easy to make in your head.
    return float(d)


def generation_flops(L, d, kind="attention"):
    if kind == "attention":
        return 4.0 * L * d + 6.0 * d ** 2
    if kind in ("linear", "recurrent"):
        return 10.0 * d ** 2
    raise ValueError(kind)


def _phi(x):
    return np.where(x > 0, x + 1.0, np.exp(x))


def linear_attention_quadratic(Q, K, V):
    Qp, Kp = _phi(np.asarray(Q, float)), _phi(np.asarray(K, float))
    W = Qp @ Kp.T
    W = W / W.sum(1, keepdims=True)
    return W @ np.asarray(V, float)


def linear_attention_recurrent(Q, K, V):
    Q, K, V = np.asarray(Q, float), np.asarray(K, float), np.asarray(V, float)
    Qp, Kp = _phi(Q), _phi(K)
    S = Kp.T @ V                      # the whole state, m x v
    z = Kp.sum(0)                     # the normaliser is its own running sum
    return (Qp @ S) / (Qp @ z)[:, None]


def scan_sequential(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = np.zeros_like(b)
    prev = 0.0
    for t in range(len(b)):
        prev = a[t] * prev + b[t]
        h[t] = prev
    return h


def scan_associative(a, b):
    a, b = np.asarray(a, float).copy(), np.asarray(b, float).copy()
    L, step = len(b), 1
    while step < L:
        a_s = np.concatenate([np.ones(step), a[:-step]])
        b_s = np.concatenate([np.zeros(step), b[:-step]])
        b = a * b_s + b
        a = a * a_s
        step *= 2
    return b


def scan_depth(L, parallel=True):
    return int(np.ceil(np.log2(L))) if parallel else int(L)


def discretise(A, B, delta):
    A, B = np.asarray(A, float), np.asarray(B, float)
    Abar = np.exp(delta * A)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bbar = np.where(np.abs(A) > 1e-12, (Abar - 1.0) / A * B, B * delta)
    return Abar, Bbar


def ssm_kernel(Abar, Bbar, C, L):
    Abar, Bbar, C = (np.asarray(v, float) for v in (Abar, Bbar, C))
    powers = Abar[None, :] ** np.arange(L)[:, None]
    return (powers * Bbar[None, :]) @ C


def half_life(lam):
    # Exact: lam**h = 1/2. The familiar ln2/(1-lam) is the limit as lam -> 1,
    # and it is 5% high already at lam = 0.9.
    return float(-np.log(2.0) / np.log(float(lam)))


def selective_scan(x, gate):
    x, g = np.asarray(x, float), np.asarray(gate, float)
    h = np.zeros_like(x)
    prev = 0.0
    for t in range(len(x)):
        prev = g[t] * prev + (1.0 - g[t]) * x[t]
        h[t] = prev
    return h


def hybrid_cache(n_layers, n_attention, d, L):
    return int(2 * n_attention * L * d)


for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
