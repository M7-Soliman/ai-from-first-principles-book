"""
Reference solutions for the Part XIX drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# -------------------------------------------------------------- the roofline

def arithmetic_intensity(n_params, tokens, bytes_per_weight=2):
    return (2.0 * n_params * tokens) / (n_params * bytes_per_weight)


def ridge_point(peak_flops, bandwidth_bytes_per_s):
    return float(peak_flops) / float(bandwidth_bytes_per_s)


def achievable_flops(intensity, peak_flops, bandwidth_bytes_per_s):
    return float(min(peak_flops, intensity * bandwidth_bytes_per_s))


def is_memory_bound(n_params, tokens, peak_flops, bandwidth, bytes_per_weight=2):
    return arithmetic_intensity(n_params, tokens, bytes_per_weight) < \
        ridge_point(peak_flops, bandwidth)


def decode_time_per_token(n_params, bytes_per_weight, bandwidth, batch=1,
                          cache_bytes=0):
    return (n_params * bytes_per_weight + cache_bytes) / (bandwidth * max(batch, 1))


# --------------------------------------------------------------- the KV cache

def kv_cache_bytes(tokens, batch, n_layers, n_kv_heads, head_dim, bytes_per=2):
    return 2.0 * batch * tokens * n_layers * n_kv_heads * head_dim * bytes_per


def context_where_cache_equals_weights(n_params, n_layers, n_kv_heads,
                                       head_dim, bytes_per=2):
    per_token = kv_cache_bytes(1, 1, n_layers, n_kv_heads, head_dim, bytes_per)
    return float(n_params * bytes_per / per_token)


def max_concurrent_sequences(device_bytes, n_params, tokens, n_layers,
                             n_kv_heads, head_dim, bytes_per=2):
    left = device_bytes - n_params * bytes_per
    if left <= 0:
        return 0
    per = kv_cache_bytes(tokens, 1, n_layers, n_kv_heads, head_dim, bytes_per)
    return int(left // per)


# ------------------------------------------------------------- quantization

def quantize_symmetric(w, bits, axis=None, group=None):
    w = np.asarray(w, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    if group:
        n, m = w.shape
        wg = w.reshape(n, m // group, group)
        s = np.abs(wg).max(-1, keepdims=True) / qmax
        s = np.where(s == 0, 1e-12, s)
        return (np.round(wg / s).clip(-qmax - 1, qmax) * s).reshape(n, m)
    if axis is None:
        s = max(np.abs(w).max() / qmax, 1e-12)
    else:
        s = np.abs(w).max(axis=1 - axis, keepdims=True) / qmax
        s = np.where(s == 0, 1e-12, s)
    return np.round(w / s).clip(-qmax - 1, qmax) * s


def quantization_error(w, wq):
    w = np.asarray(w, float)
    return float(np.linalg.norm(np.asarray(wq, float) - w) / np.linalg.norm(w))


def quantized_bytes(n_params, bits, group=None, scale_bits=16):
    b = n_params * bits / 8.0
    if group:
        b += (n_params / group) * scale_bits / 8.0
    return float(b)


def outlier_channels(w, factor=5.0):
    w = np.asarray(w, float)
    colmax = np.abs(w).max(0)
    med = np.median(colmax)
    return np.flatnonzero(colmax > factor * med)


def mixed_precision_error(w, bits, keep_cols, axis=0):
    w = np.asarray(w, float)
    keep = np.asarray(keep_cols, int)
    rest = np.setdiff1d(np.arange(w.shape[1]), keep)
    q = w.copy()
    # the point of mixed precision is that the SCALES are computed without the
    # outliers. Quantizing the whole matrix and then overwriting the outlier
    # columns leaves every remaining weight sharing an inflated range, which
    # defeats the technique entirely.
    q[:, rest] = quantize_symmetric(w[:, rest], bits, axis=axis)
    return quantization_error(w, q)


# ---------------------------------------------------------------- sparsity

def magnitude_prune(w, sparsity):
    w = np.asarray(w, float).copy()
    if sparsity <= 0:
        return w
    thr = np.quantile(np.abs(w), sparsity)
    w[np.abs(w) < thr] = 0.0
    return w


def prune_2of4(w):
    w = np.asarray(w, float).copy()
    shp = w.shape
    g = w.reshape(-1, 4)
    keep = np.argsort(-np.abs(g), axis=1)[:, :2]
    out = np.zeros_like(g)
    rows = np.repeat(np.arange(len(g)), 2)
    out[rows, keep.ravel()] = g[rows, keep.ravel()]
    return out.reshape(shp)


def sparsity_of(w):
    return float((np.asarray(w) == 0).mean())


def sparse_storage_bytes(w, value_bytes=2, index_bytes=4):
    w = np.asarray(w)
    nnz = int((w != 0).sum())
    rows = w.shape[0] if w.ndim > 1 else 1
    return float(nnz * (value_bytes + index_bytes) + (rows + 1) * index_bytes)


def dense_storage_bytes(w, value_bytes=2):
    return float(np.asarray(w).size * value_bytes)


# -------------------------------------------------------------- distillation

def softmax_T(logits, T=1.0):
    z = np.asarray(logits, float) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    s = np.asarray(student_logits, float)
    lab = np.asarray(labels, int)
    ls = s - s.max(-1, keepdims=True)
    logp = ls - np.log(np.exp(ls).sum(-1, keepdims=True))
    hard = -logp[np.arange(len(lab)), lab].mean()
    pt = softmax_T(teacher_logits, T)
    ps = softmax_T(s, T)
    kl = (pt * (np.log(pt + 1e-30) - np.log(ps + 1e-30))).sum(-1).mean()
    return float((1 - alpha) * hard + alpha * (T ** 2) * kl)


def dark_knowledge(teacher_logits, T=1.0):
    p = softmax_T(teacher_logits, T)
    return float((-(p * np.log(p + 1e-30)).sum(-1)).mean())


# ------------------------------------------------------------------ adapters

def lora_params(d_in, d_out, rank):
    p = rank * (d_in + d_out)
    return int(p), float(p) / float(d_in * d_out)


def lora_forward(x, W0, A, B, scale=1.0):
    x = np.asarray(x, float)
    return x @ np.asarray(W0, float).T + scale * ((x @ np.asarray(A, float).T)
                                                  @ np.asarray(B, float).T)


def merge_lora(W0, A, B, scale=1.0):
    return np.asarray(W0, float) + scale * (np.asarray(B, float) @ np.asarray(A, float))


def rank_for_energy(delta, frac=0.9):
    s = np.linalg.svd(np.asarray(delta, float), compute_uv=False)
    c = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(c, frac) + 1)


# -------------------------------------------------------------------- serving

def batching_speedup(n_params, batch, bandwidth, peak_flops, bytes_per=2):
    def tput(b):
        ai = arithmetic_intensity(n_params, b, bytes_per)
        return achievable_flops(ai, peak_flops, bandwidth) / (2.0 * n_params)
    return float(tput(batch) / tput(1))


def speculative_speedup(accept_rate, k, draft_cost_fraction):
    a = float(accept_rate)
    if a >= 1.0:
        exp_acc = float(k)
    else:
        exp_acc = (1.0 - a ** (k + 1)) / (1.0 - a) - 1.0
    return float((1.0 + exp_acc) / (1.0 + k * draft_cost_fraction))


def paged_waste(seq_lengths, block_size, max_len):
    L = np.asarray(seq_lengths, float)
    used = L.sum()
    naive = len(L) * max_len
    paged = (np.ceil(L / block_size) * block_size).sum()
    return float(1 - used / naive), float(1 - used / paged)


def request_cost(n_params, tokens_in, tokens_out, batch, bandwidth, peak_flops,
                 bytes_per=2):
    prefill = 2.0 * n_params * tokens_in / peak_flops
    decode = tokens_out * (n_params * bytes_per) / (bandwidth * max(batch, 1))
    return float(prefill + decode)


# --------------------------------------------------------------------- patch

for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
