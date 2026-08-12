"""
Figures for Part XIX — Efficiency and Deployment.

Every claim this part makes that can be measured, is measured here.

Several are exact arithmetic — the roofline, the KV cache, the cost model —
because those are not empirical questions. The rest are measurements, and three
of them show a technique failing to deliver what it is usually credited with:
unstructured sparsity buys no speedup, per-tensor quantization is destroyed by
a handful of outliers, and a fine-tuning update is low rank only in a sense
that has to be stated carefully.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train

Runtime: about two minutes for --cheap, fifteen for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "ef"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
CYAN, LIGHT, INK = "#0e7490", "#cbd5e1", "#1e293b"
MAX_PT = 900

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "axes.linewidth": 0.9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def save(fig, name):
    p = os.path.join(OUT, name + ".svg")
    fig.savefig(p, format="svg")
    plt.close(fig)
    h = float(re.search(r'height="([\d.]+)pt"', open(p).read(1200)).group(1))
    if h > MAX_PT:
        raise ValueError(f"{name}.svg is {h:.0f}pt")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------- 1. prefill against decode

def fig_roofline():
    """Why generation is memory-bound and prefill is not. Exact arithmetic:
    a matrix multiply's arithmetic intensity is set by its shapes, and one
    token at a time gives you the worst possible shape."""
    N = 7e9
    BYTES = 2
    PEAK_FLOPS = 1.0e15          # a plausible sustained dense rate
    BANDWIDTH = 2.0e12           # bytes/s of memory bandwidth
    RIDGE = PEAK_FLOPS / BANDWIDTH

    # arithmetic intensity of one decoder pass over B sequences of T new tokens
    def intensity(tokens):
        flops = 2 * N * tokens                 # forward only
        bytes_moved = N * BYTES                # every weight, once
        return flops / bytes_moved

    toks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096])
    ai = intensity(toks)
    achieved = np.minimum(PEAK_FLOPS, ai * BANDWIDTH)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    x = np.logspace(-1, 4, 300)
    ax.plot(x, np.minimum(PEAK_FLOPS, x * BANDWIDTH) / 1e12, color=GRAY, lw=2.0)
    ax.axvline(RIDGE, color=LIGHT, lw=1.2, ls="--")
    ax.text(RIDGE * 1.15, 3, f"ridge at {RIDGE:.0f}", fontsize=8, color=GRAY)
    for t, c, lab in ((1, ORANGE, "decode: 1 token"),
                      (64, BLUE, "64 tokens"),
                      (4096, GREEN, "prefill: 4096")):
        ax.plot([intensity(t)], [min(PEAK_FLOPS, intensity(t) * BANDWIDTH) / 1e12],
                "o", color=c, ms=8, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte)")
    ax.set_ylabel("achievable TFLOP/s")
    ax.set_title("The roofline", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    tidy(ax)

    ax = axes[1]
    ax.plot(toks, achieved / PEAK_FLOPS * 100, "o-", color=BLUE, lw=1.9, ms=4)
    ax.axhline(100, color=GRAY, lw=1.2, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens processed per weight load")
    ax.set_ylabel("% of peak compute reachable")
    ax.set_title("Why batching is the whole game", fontsize=10.5, color=INK)
    tidy(ax)

    print(f"    a {N/1e9:.0f}B model in bf16 moves {N*BYTES/1e9:.1f} GB of weights per pass")
    print(f"    the hardware's ridge point is {RIDGE:.0f} FLOP/byte")
    for t in (1, 8, 64, 512, 4096):
        print(f"      {t:5d} tokens: intensity {intensity(t):8.1f}   "
              f"{min(PEAK_FLOPS, intensity(t)*BANDWIDTH)/PEAK_FLOPS*100:5.1f}% of peak   "
              f"{'memory-bound' if intensity(t) < RIDGE else 'compute-bound'}")
    print(f"    generating one token at a time reaches "
          f"{intensity(1)*BANDWIDTH/PEAK_FLOPS*100:.2f}% of the hardware's compute.")
    print("    the weights are read in full to produce a single token. Everything in")
    print("    movement F is about amortising that read over more tokens.")
    fig.tight_layout()
    save(fig, "01-roofline")
    return dict(ridge=RIDGE, ai1=float(intensity(1)))


# ----------------------------------------------------------- 2. the KV cache

def fig_kv_cache():
    """Exact: how much memory the cache takes, and what the three standard
    reductions buy."""
    d, L, heads, V = 4096, 32, 32, 128000
    head_dim = d // heads
    N = 12 * d * d * L + V * d
    GB = 1024 ** 3

    def kv_bytes(tokens, batch, n_kv_heads, bytes_per=2):
        return 2 * batch * tokens * L * n_kv_heads * head_dim * bytes_per

    ctxs = np.array([1024, 4096, 16384, 65536, 262144])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    for nkv, c, lab in ((heads, ORANGE, "multi-head (32 kv heads)"),
                        (8, BLUE, "grouped-query (8)"),
                        (1, GREEN, "multi-query (1)")):
        ax.plot(ctxs, [kv_bytes(t, 1, nkv) / GB for t in ctxs], "o-",
                color=c, lw=1.9, ms=4, label=lab)
    ax.axhline(N * 2 / GB, color=GRAY, lw=1.3, ls="--",
               label=f"the weights ({N*2/GB:.0f} GiB)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length"); ax.set_ylabel("KV cache, GiB (batch 1)")
    ax.set_title("The cache overtakes the model", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.2, frameon=False)
    tidy(ax)

    ax = axes[1]
    budget = 80 * GB - N * 2
    for nkv, c, lab in ((heads, ORANGE, "multi-head"), (8, BLUE, "grouped-query"),
                        (1, GREEN, "multi-query")):
        seqs = [max(0, int(budget // kv_bytes(t, 1, nkv))) for t in ctxs]
        ax.plot(ctxs, seqs, "o-", color=c, lw=1.9, ms=4, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length"); ax.set_ylabel("concurrent sequences")
    ax.set_title("And decides how many users fit", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    print(f"    model: {N/1e9:.1f}B parameters, {N*2/GB:.1f} GiB in bf16")
    for t in (4096, 65536):
        mh = kv_bytes(t, 1, heads) / GB
        gq = kv_bytes(t, 1, 8) / GB
        mq = kv_bytes(t, 1, 1) / GB
        print(f"    context {t:6d}: cache {mh:7.2f} GiB multi-head, {gq:6.2f} grouped, "
              f"{mq:5.2f} multi-query")
    cross = kv_bytes(1, 1, heads)
    t_eq = int(N * 2 / cross)
    print(f"    the cache equals the weights at about {t_eq:,} tokens of context")
    print(f"    on an 80 GiB device, {budget/GB:.0f} GiB is left for cache after weights;")
    print(f"    at 64k context that is {int(budget//kv_bytes(65536,1,heads))} concurrent "
          f"sequences with multi-head and "
          f"{int(budget//kv_bytes(65536,1,1))} with multi-query.")
    fig.tight_layout()
    save(fig, "02-kv-cache")
    return dict(t_eq=t_eq)


# --------------------------------------------- 3. quantization granularity

def quantize(w, bits, axis=None, group=None):
    """Symmetric integer quantization, and dequantization back. axis=None is
    per-tensor; axis=0 is per-output-channel; group splits each row."""
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


def fig_quant_granularity():
    """What granularity buys, and what a handful of outliers costs. Measured on
    a weight matrix with the heavy tail that real ones have."""
    rng = np.random.default_rng(0)
    n, m = 512, 512
    W = rng.normal(0, 0.02, (n, m))
    # a realistic outlier structure: a few input channels with much larger values
    out_cols = rng.choice(m, 6, replace=False)
    W[:, out_cols] *= 120.0

    def err(Wq):
        return float(np.linalg.norm(Wq - W) / np.linalg.norm(W))

    bits_list = [8, 6, 4, 3]
    schemes = [("per-tensor", dict(axis=None)),
               ("per-channel", dict(axis=0)),
               ("per-group (128)", dict(group=128))]
    res = {}
    for name, kw in schemes:
        res[name] = [err(quantize(W, b, **kw)) for b in bits_list]
        print(f"    {name:16s} " + "  ".join(
            f"{b}-bit {e:.4f}" for b, e in zip(bits_list, res[name])))

    # the same, with the outliers removed
    Wc = W.copy(); Wc[:, out_cols] /= 120.0
    clean = {}
    for name, kw in schemes:
        qmaxerr = float(np.linalg.norm(quantize(Wc, 8, **kw) - Wc) / np.linalg.norm(Wc))
        clean[name] = qmaxerr
    print("    with the outlier channels removed, 8-bit error becomes:")
    for name in clean:
        print(f"      {name:16s} {clean[name]:.4f}  "
              f"(was {res[name][bits_list.index(8)]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for (name, _), c in zip(schemes, (ORANGE, BLUE, GREEN)):
        ax.plot(bits_list, res[name], "o-", color=c, lw=1.9, ms=5, label=name)
    ax.set_xticks(bits_list); ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("bits"); ax.set_ylabel("relative error")
    ax.set_title("Granularity is what buys accuracy", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    tidy(ax)

    ax = axes[1]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, [res[n][bits_list.index(8)] for n, _ in schemes], w,
           color=ORANGE, label="with outliers")
    ax.bar(x + w/2, [clean[n] for n, _ in schemes], w,
           color=GREEN, label="outliers removed")
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in schemes], fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("relative error at 8 bits")
    ax.set_title("Six columns of 512 do this", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    tidy(ax)

    r = res["per-tensor"][bits_list.index(8)] / clean["per-tensor"]
    print(f"    six outlier columns out of {m} raise per-tensor 8-bit error by {r:.0f}x.")
    print("    they widen the range every other weight must share, so the step size")
    print("    grows and the ordinary weights lose their precision to them.")
    fig.tight_layout()
    save(fig, "03-quant-granularity")
    return dict(res=res, clean=clean)


# ------------------------------------------- 4. unstructured sparsity is slow

def fig_sparsity():
    """Unstructured sparsity removes arithmetic and does not remove time. This
    is a timing measurement, not a claim about FLOPs."""
    import time
    import scipy.sparse as sp

    n = 2048
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n)).astype(np.float32)
    x = rng.normal(size=(n, 256)).astype(np.float32)

    def timeit(f, reps=30):
        for _ in range(5):
            f()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return float(np.min(ts))          # min, not median: noise is one-sided

    dense_t = timeit(lambda: A @ x)
    levels = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]
    dense_times, sparse_times, kept = [], [], []
    for s in levels:
        M = A.copy()
        if s > 0:
            thr = np.quantile(np.abs(M), s)
            M[np.abs(M) < thr] = 0.0
        nnz = float((M != 0).mean())
        kept.append(nnz)
        dense_times.append(timeit(lambda: M @ x))
        S = sp.csr_matrix(M)
        sparse_times.append(timeit(lambda: S @ x))
        print(f"    {s*100:5.1f}% zeros: dense {dense_times[-1]*1e3:7.3f} ms   "
              f"CSR {sparse_times[-1]*1e3:7.3f} ms   "
              f"dense speedup {dense_t/dense_times[-1]:.2f}x   "
              f"CSR speedup {dense_t/sparse_times[-1]:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(np.array(levels) * 100, dense_t / np.array(dense_times), "o-",
            color=ORANGE, lw=1.9, ms=5, label="zeros in a dense array")
    ax.plot(np.array(levels) * 100, dense_t / np.array(sparse_times), "s-",
            color=BLUE, lw=1.9, ms=5, label="CSR sparse kernel")
    ax.plot(np.array(levels) * 100, 1 / np.maximum(np.array(kept), 1e-9),
            color=GRAY, lw=1.5, ls="--", label="what the FLOP count promises")
    ax.set_yscale("log")
    ax.set_xlabel("% of weights zeroed"); ax.set_ylabel("speedup over dense")
    ax.set_title("Removed arithmetic is not removed time",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=7.4, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--")
    ax.plot(np.array(levels) * 100, dense_t / np.array(sparse_times), "s-",
            color=BLUE, lw=1.9, ms=5)
    ax.set_xlabel("% of weights zeroed")
    ax.set_ylabel("CSR speedup over dense")
    ax.set_title("The crossover, if there is one", fontsize=10.5, color=INK)
    tidy(ax)

    i90 = levels.index(0.9)
    print(f"    at 90% sparsity the FLOP count promises {1/kept[i90]:.0f}x and the")
    print(f"    measured speedups are {dense_t/dense_times[i90]:.2f}x dense and "
          f"{dense_t/sparse_times[i90]:.2f}x with a sparse kernel.")
    print("    zeros in a dense array cost exactly as much as any other number, and a")
    print("    sparse format trades the arithmetic for indexing and irregular access.")
    fig.tight_layout()
    save(fig, "05-sparsity")
    return dict(levels=levels, dense=dense_times, sparse=sparse_times, kept=kept)


CHEAP = [fig_roofline, fig_kv_cache, fig_quant_granularity, fig_sparsity]

if __name__ == "__main__":
    print("Part XIX — efficiency and deployment figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
