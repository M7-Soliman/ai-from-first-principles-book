"""
Reference solutions for the Part XII drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ------------------------------------------------------------- the operation ----

def softmax(x, axis=-1):
    x = np.asarray(x, float)
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    Q, K, V = (np.asarray(a, float) for a in (Q, K, V))
    d_k = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    return softmax(s, axis=-1) @ V


def causal_mask(n):
    return np.triu(np.ones((n, n), dtype=bool), 1)


def attention_entropy(Q, K, scale=True):
    Q, K = np.asarray(Q, float), np.asarray(K, float)
    s = Q @ K.T
    if scale:
        s = s / np.sqrt(Q.shape[-1])
    p = softmax(s, axis=-1)
    return float(-(p * np.log(p + 1e-300)).sum(-1).mean())


# ------------------------------------------------------------------ heads ----

def split_heads(X, n_head):
    X = np.asarray(X, float)
    n, d = X.shape
    return X.reshape(n, n_head, d // n_head).transpose(1, 0, 2)


def merge_heads(H):
    H = np.asarray(H, float)
    n_head, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, n_head * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_head, mask=None):
    X = np.asarray(X, float)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    Qh, Kh, Vh = (split_heads(a, n_head) for a in (Q, K, V))
    outs = [attention(Qh[h], Kh[h], Vh[h], mask) for h in range(n_head)]
    return merge_heads(np.stack(outs)) @ Wo


# --------------------------------------------------------------- position ----

def sinusoidal_encoding(n, d, base=10000.0):
    pos = np.arange(n)[:, None].astype(float)
    i = np.arange(0, d, 2).astype(float)
    div = base ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe


def rope(x, pos, base=10000.0):
    x = np.asarray(x, float)
    d = x.shape[-1]
    h = d // 2
    inv = base ** (-np.arange(h, dtype=float) / h)
    ang = np.asarray(pos, float)[..., None] * inv
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def rope_score(q, k, m, n, base=10000.0):
    qm = rope(np.asarray(q, float), np.array(float(m)), base)
    kn = rope(np.asarray(k, float), np.array(float(n)), base)
    return float((qm * kn).sum())


def is_permutation_equivariant(fn, X, perm, tol=1e-8):
    a = np.asarray(fn(np.asarray(X)[perm]), float)
    b = np.asarray(fn(np.asarray(X)), float)[perm]
    return bool(np.abs(a - b).max() < tol)


# ------------------------------------------------------------- the counts ----

def attention_params(d):
    return 4 * d * d


def ffn_params(d, mult=4):
    return 2 * mult * d * d


def block_flops(n, d, mult=4):
    return {"proj": 4 * n * d * d,
            "scores": 2 * n * n * d,
            "ffn": 2 * mult * n * d * d}


def crossover_length(d, mult=4):
    return mult * d


def kv_cache_bytes(n, d, n_layers, bytes_per=2, n_kv_head=None, n_head=None):
    frac = 1.0
    if n_kv_head is not None and n_head is not None:
        frac = n_kv_head / n_head
    return int(2 * n_layers * n * d * bytes_per * frac)


# ------------------------------------------------------------- generation ----

def _step(x_all, Wq, Wk, Wv):
    q = x_all[-1:] @ Wq
    K, V = x_all @ Wk, x_all @ Wv
    return attention(q, K, V)


def generate_no_cache(x0, Wq, Wk, Wv, steps):
    X = np.asarray(x0, float).reshape(1, -1)
    for _ in range(steps):
        X = np.concatenate([X, _step(X, Wq, Wk, Wv)], axis=0)
    return X


def generate_with_cache(x0, Wq, Wk, Wv, steps):
    x = np.asarray(x0, float).reshape(1, -1)
    out = [x[0]]
    K, V = x @ Wk, x @ Wv
    for _ in range(steps):
        q = x @ Wq
        nxt = attention(q, K, V)
        K = np.concatenate([K, nxt @ Wk], axis=0)
        V = np.concatenate([V, nxt @ Wv], axis=0)
        out.append(nxt[0])
        x = nxt
    return np.array(out)


# ---------------------------------------------------------- online softmax ----

def online_softmax_weighted_sum(scores, V, tile):
    scores = np.asarray(scores, float)
    V = np.asarray(V, float)
    m = -np.inf
    l = 0.0
    acc = np.zeros(V.shape[1])
    for i in range(0, len(scores), tile):
        s = scores[i:i + tile]
        v = V[i:i + tile]
        m_new = max(m, float(s.max()))
        corr = np.exp(m - m_new) if np.isfinite(m) else 0.0
        w = np.exp(s - m_new)
        acc = acc * corr + w @ v
        l = l * corr + float(w.sum())
        m = m_new
    return acc / l


# ------------------------------------------------------------------ shapes ----

def effective_rank(X):
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    tot = sv.sum()
    if tot <= 0:
        return 1.0
    p = sv / tot
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def attention_only_depth(X, layers, skip=False, seed=0):
    r = np.random.default_rng(seed)
    X = np.asarray(X, float)
    d = X.shape[1]
    for _ in range(layers):
        Wq, Wk, Wv = (r.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
        Y = attention(X @ Wq, X @ Wk, X @ Wv)
        X = X + Y if skip else Y
    return X


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
