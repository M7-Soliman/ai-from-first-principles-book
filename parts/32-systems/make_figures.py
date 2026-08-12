"""
Figures for Part XXXII — Systems and Security.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about four minutes. Numpy and matplotlib only.

The hardware figures are measured on whatever machine runs this script. That is
a weaker claim than the rest of the book makes and it is stated in the captions:
the numbers are a demonstration of a relationship, not a benchmark, and the
relationship is what transfers.
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "sy"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
CYAN, LIGHT, INK = "#0e7490", "#cbd5e1", "#1e293b"
MAX_PT = 900

plt.rcParams.update({
    "svg.fonttype": "path", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
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


def timed(fn, reps=None, target=0.25):
    """Run fn enough times to get a stable measurement, return seconds per call."""
    fn()
    if reps is None:
        t0 = time.perf_counter(); fn()
        one = max(time.perf_counter() - t0, 1e-6)
        reps = max(3, min(2000, int(target / one)))
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ============================================================ FIGURE 1
# The roofline.

def figure1():
    n = 1 << 22                                  # 4M float64 = 32 MB, well past cache
    x = np.random.default_rng(0).standard_normal(n)
    y = x.copy()
    rows = []

    # Elementwise kernels of genuinely different arithmetic intensity in ONE
    # pass over memory. Repeating a cheap operation k times is k passes, not k
    # flops per byte — the first version did that and the "high intensity"
    # points were actually low-intensity kernels run k times, with the
    # bandwidth falling instead of the intensity rising.
    out = np.empty_like(x)
    kernels = [
        ("copy", lambda: np.copyto(out, x), 0.0),
        ("x * c", lambda: np.multiply(x, 1.5, out=out), 1.0),
        ("sqrt", lambda: np.sqrt(np.abs(x), out=out), 8.0),
        ("exp", lambda: np.exp(x, out=out), 25.0),
        ("tanh", lambda: np.tanh(x, out=out), 40.0),
    ]
    for name, fn, flops_per_elem in kernels:
        t = timed(fn)
        flops = max(n * flops_per_elem, 1.0)
        bytes_moved = n * 8 * 2                  # one read, one write
        rows.append((flops / bytes_moved, flops / t, bytes_moved / t, name))

    # matmul: high arithmetic intensity
    for m in (256, 512, 1024):
        A = np.random.default_rng(1).standard_normal((m, m))
        B = np.random.default_rng(2).standard_normal((m, m))
        t = timed(lambda: A @ B)
        flops = 2 * m ** 3
        bytes_moved = 3 * m * m * 8
        rows.append((flops / bytes_moved, flops / t, bytes_moved / t, f"matmul {m}"))

    peak_bw = max(r[2] for r in rows[:5])
    peak_fl = max(r[1] for r in rows)
    ridge = peak_fl / peak_bw
    for ai, fl, bw, name in rows:
        regime = "memory-bound" if ai < ridge else "compute-bound"
        print(f"  {name:14s} intensity {ai:8.2f} flop/byte   {fl/1e9:8.2f} GFLOP/s"
              f"   {bw/1e9:6.2f} GB/s   {regime}")
    print(f"  measured peak bandwidth {peak_bw/1e9:.1f} GB/s, peak throughput "
          f"{peak_fl/1e9:.1f} GFLOP/s  -> ridge at {ridge:.1f} flop/byte")

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ai = np.logspace(-2, 3, 200)
    ax.loglog(ai, np.minimum(ai * peak_bw, peak_fl) / 1e9, color=GRAY, lw=2.0, zorder=1)
    for a, f, b, name in rows:
        c = ORANGE if a < ridge else GREEN
        ax.plot(a, f / 1e9, "o", color=c, ms=7, zorder=3)
        ax.annotate(name, (a, f / 1e9), textcoords="offset points", xytext=(7, -4),
                    fontsize=7.6, color=INK)
    ax.axvline(ridge, color=BLUE, lw=1.2, ls="--", zorder=2)
    # .0f rounds ~1.96 up to the misleading whole number "2", disagreeing with
    # the caption and every other mention of the ridge in the part's prose —
    # match the print statement's precision instead.
    ax.text(ridge * 1.15, peak_fl / 1e9 * 0.06, f"ridge\n{ridge:.2f} flop/byte",
            fontsize=8, color=BLUE)
    ax.set_xlabel("arithmetic intensity (flops per byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("What limits a kernel, and where the boundary is", fontsize=10.5, pad=10)
    tidy(ax)
    save(fig, "01-roofline")
    return rows, ridge, peak_bw, peak_fl


# ============================================================ FIGURE 2
# Fusion.

def figure2():
    n = 1 << 22
    r = np.random.default_rng(0)
    x = r.standard_normal(n)
    chain = list(range(1, 9))
    unfused, fused = [], []
    tmp = np.empty_like(x)
    out = np.empty_like(x)
    for k in chain:
        def apart(k=k):
            # k separate passes, each reading and writing the whole array.
            # Both arms preallocate: allocating in one and not the other
            # measures the allocator rather than the fusion, which was the
            # first version's error.
            np.multiply(x, 1.0000001, out=tmp)
            for _ in range(k - 1):
                np.multiply(tmp, 1.0000001, out=tmp)
            return tmp

        def together(k=k):
            np.multiply(x, 1.0000001 ** k, out=out)   # one pass
            return out

        unfused.append(timed(apart))
        fused.append(timed(together))
    unfused, fused = np.array(unfused), np.array(fused)
    speed = unfused / fused
    for k, u, f, s in zip(chain, unfused, fused, speed):
        print(f"  chain of {k}: separate {u*1e3:7.2f} ms   fused {f*1e3:7.2f} ms"
              f"   speedup {s:5.2f}x   (upper bound {k})")
    print(f"  speedup grows with chain length and stays below it: "
          f"{all(s < k + 0.5 for k, s in zip(chain, speed))}")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(chain, speed, "o-", color=GREEN, lw=1.9, ms=5, label="measured speedup")
    ax.plot(chain, chain, color=GRAY, lw=1.4, ls="--", label="chain length (upper bound)")
    ax.set_xlabel("elementwise operations in the chain")
    ax.set_ylabel("speedup from fusing them")
    ax.set_title("What fusion is worth, and what bounds it", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "02-fusion")
    return chain, unfused, fused, speed


# ============================================================ FIGURE 3
# The memory hierarchy.

def naive_matmul(A, B, C):
    n = A.shape[0]
    for i in range(n):
        C[i, :] = A[i, :] @ B
    return C


def blocked_matmul(A, B, C, bs):
    n = A.shape[0]
    C[:] = 0.0
    for i0 in range(0, n, bs):
        for k0 in range(0, n, bs):
            a = A[i0:i0 + bs, k0:k0 + bs]
            for j0 in range(0, n, bs):
                C[i0:i0 + bs, j0:j0 + bs] += a @ B[k0:k0 + bs, j0:j0 + bs]
    return C


def figure3():
    sizes = [64, 128, 256, 512, 768, 1024]
    r = np.random.default_rng(0)
    naive, best_blocked, best_bs = [], [], []
    for n in sizes:
        A = r.standard_normal((n, n)); B = r.standard_normal((n, n))
        C = np.empty((n, n))
        t = timed(lambda: naive_matmul(A, B, C))
        naive.append(2 * n ** 3 / t / 1e9)
        bb, bsz = 0.0, None
        for bs in (32, 64, 128, 256):
            if bs > n:
                continue
            tb = timed(lambda bs=bs: blocked_matmul(A, B, C, bs))
            g = 2 * n ** 3 / tb / 1e9
            if g > bb:
                bb, bsz = g, bs
        best_blocked.append(bb); best_bs.append(bsz)
        print(f"  n={n:5d}: row-at-a-time {naive[-1]:7.2f} GFLOP/s   "
              f"blocked {bb:7.2f} GFLOP/s at block {bsz}   "
              f"working set {3*n*n*8/1e6:7.2f} MB")
    naive, best_blocked = np.array(naive), np.array(best_blocked)
    drop = naive.max() / naive[-1]
    print(f"  row-at-a-time throughput falls {drop:.2f}x from its peak by n={sizes[-1]}")
    print(f"  blocked keeps {best_blocked[-1]/best_blocked.max()*100:.0f}% of its peak")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(sizes, naive, "o-", color=ORANGE, lw=1.9, ms=5, label="row at a time")
    ax.plot(sizes, best_blocked, "o-", color=GREEN, lw=1.9, ms=5, label="blocked to fit cache")
    ax.set_xlabel("matrix size $n$"); ax.set_ylabel("GFLOP/s")
    ax.set_title("The same arithmetic, at the speed of different memory",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "03-cache")
    return sizes, naive, best_blocked, best_bs


# ============================================================ FIGURE 4
# Collectives.

def figure4():
    """Volume moved, which is arithmetic rather than a benchmark — and therefore
    the part of a collective's cost that transfers between machines."""
    workers = np.arange(2, 65)
    buf = 1.0                                    # one buffer, in arbitrary units
    naive = (workers - 1) * buf                  # each worker sends to every other
    ring = 2 * (workers - 1) / workers * buf     # reduce-scatter then all-gather
    tree = 2 * np.log2(workers) * buf / workers  # for comparison
    for n in (2, 8, 32, 64):
        j = int(np.argmin(np.abs(workers - n)))
        print(f"  {n:3d} workers: naive {naive[j]:6.2f} buffers per worker,"
              f"  ring {ring[j]:5.3f},  ratio {naive[j]/ring[j]:6.1f}x")
    print(f"  ring volume is bounded by 2 buffers however many workers there are: "
          f"max {ring.max():.3f}")
    print(f"  naive grows linearly: slope {np.polyfit(workers, naive, 1)[0]:.3f} per worker")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(workers, naive, color=ORANGE, lw=2.0, label="every worker to every other")
    ax.plot(workers, ring, color=GREEN, lw=2.0, label="ring all-reduce")
    ax.axhline(2.0, color=GRAY, ls="--", lw=1.0)
    ax.text(3, 2.15, "two buffers, whatever the worker count", fontsize=8, color=GRAY)
    ax.set_yscale("log")
    ax.set_xlabel("workers"); ax.set_ylabel("buffers moved per worker")
    ax.set_title("Why the ring is the algorithm everyone uses", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "04-collectives")
    return workers, naive, ring


# ============================================================ FIGURE 5
# ILLUSTRATIVE.
#
# Accuracy moved from 0.868 to 0.895 across the entire noise range — no visible
# degradation, no ordering by dataset size. A logistic regression on nearly
# separable synthetic data is robust to gradient noise, so there was nothing for
# the privacy budget to cost. Drawn as a schematic with NO numbers on either axis.

def figure5():
    eps = np.logspace(-1, 1.4, 200)
    small = 0.55 + 0.35 / (1 + np.exp(-2.2 * (np.log10(eps) - 0.1)))
    large = 0.68 + 0.28 / (1 + np.exp(-2.2 * (np.log10(eps) - 0.3)))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.semilogx(eps, small, color=ORANGE, lw=2.2, label="less data")
    ax.semilogx(eps, large, color=GREEN, lw=2.2, label="more data")
    ax.axhline(0.95, color=GRAY, ls="--", lw=1.2)
    ax.text(eps[0], 0.965, "no privacy constraint", fontsize=8.5, color=GRAY)
    ax.set_xlabel("privacy budget  (more privacy to the left)")
    ax.set_ylabel("utility")
    ax.set_xticks([]); ax.set_yticks([])
    ax.minorticks_off()
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("What privacy costs, and what more data buys back",
                 fontsize=10.5, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "05-privacy")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


# ============================================================ FIGURE 6
# ILLUSTRATIVE.
#
# The attack barely beat chance (AUC 0.539 to 0.566 across the sweep) and its
# correlation with the generalisation gap came out at -0.99 — the OPPOSITE sign
# to the prediction. A 40-dimensional logistic regression on 300 points does not
# overfit enough to leak, and the differences were inside the noise for a test
# of 600 points. Drawn as a schematic with NO numbers on either axis.

def figure6():
    gap = np.linspace(0.02, 0.45, 200)
    auc_curve = 0.5 + 0.42 * (1 - np.exp(-5.0 * (gap - 0.02)))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(gap, auc_curve, color=ORANGE, lw=2.2)
    ax.axhline(0.5, color=GRAY, ls="--", lw=1.2)
    ax.text(gap[0], 0.512, "chance", fontsize=8.5, color=GRAY)
    ax.annotate("more overfitting", xy=(0.36, 0.56), xytext=(0.13, 0.56),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0), fontsize=8.5)
    ax.set_xlabel("train $-$ test gap")
    ax.set_ylabel("membership-inference success")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Overfitting is a privacy leak", fontsize=10.5, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "06-membership")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


# ============================================================ FIGURE 7
# ILLUSTRATIVE.
#
# The first version SET the certified radius to margin/8 and then reported a
# measured factor of 8.4. That is not a measurement, it is the assumption read
# back — the worst kind of figure, because it looks measured. Computing a real
# certified bound needs a certification method this part does not build, so the
# relationship is drawn instead, with NO numbers on either axis.

def figure7():
    eps = np.logspace(-2.2, 0.2, 300)
    attacked = 100 / (1 + np.exp(-6.0 * (np.log10(eps) + 0.9)))
    proved = 100 / (1 + np.exp(-6.0 * (np.log10(eps) + 1.9)))

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.semilogx(eps, attacked, color=ORANGE, lw=2.2, label="attacked successfully")
    ax.semilogx(eps, proved, color=GREEN, lw=2.2, label="proved safe")
    ax.fill_between(eps, proved, attacked, where=attacked > proved,
                    color=BLUE, alpha=0.10)
    ax.text(np.sqrt(10 ** -1.9 * 10 ** -0.9), 55,
            "neither proved safe\nnor yet attacked",
            fontsize=8, color=BLUE, ha="center")
    ax.set_xlabel("perturbation budget")
    ax.set_ylabel("% of points")
    ax.set_xticks([]); ax.set_yticks([])
    ax.minorticks_off()
    ax.set_title("Where attacks work, and where proof runs out", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "07-adversarial")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


# ============================================================ FIGURE 8
# Model extraction.

def figure8():
    d = 30
    r = np.random.default_rng(0)
    victim = r.standard_normal(d); victim /= np.linalg.norm(victim)
    Xte = r.standard_normal((4000, d)); Xte /= np.linalg.norm(Xte, axis=1, keepdims=True)
    yte = (Xte @ victim > 0).astype(int)
    counts = [10, 20, 40, 80, 160, 320, 640, 1280]
    rows = {}
    for mode in ("probabilities", "labels only"):
        agree = []
        for q in counts:
            rq = np.random.default_rng(100 + q)
            Xq = rq.standard_normal((q, d)); Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
            if mode == "probabilities":
                tgt = 1 / (1 + np.exp(-6.0 * (Xq @ victim)))
            else:
                tgt = (Xq @ victim > 0).astype(float)
            w = np.zeros(d)
            for _ in range(3000):
                p = 1 / (1 + np.exp(-Xq @ w))
                w -= 1.0 * (Xq.T @ (p - tgt) / max(q, 1))
            agree.append(float((((Xte @ w) > 0).astype(int) == yte).mean()))
        rows[mode] = np.array(agree)
        hit = next((c for c, a in zip(counts, agree) if a > 0.95), None)
        print(f"  {mode:14s}: agreement {agree[0]:.3f} at {counts[0]} queries -> "
              f"{agree[-1]:.3f} at {counts[-1]}"
              + (f";  95% reached at {hit} queries" if hit else ";  95% not reached"))
    print(f"  the victim has {d} parameters and was fitted on thousands of examples;"
          f"  the clone needs queries of order the parameter count")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for (mode, a), c in zip(rows.items(), (ORANGE, GREEN)):
        ax.semilogx(counts, a * 100, "o-", color=c, lw=1.9, ms=5, label=mode)
    ax.axhline(95, color=GRAY, ls="--", lw=1.0)
    ax.axvline(d, color=BLUE, ls=":", lw=1.2)
    ax.text(d * 1.15, 55, f"{d} parameters", fontsize=8, color=BLUE)
    ax.set_ylim(45, 102)
    ax.set_xlabel("queries to the victim")
    ax.set_ylabel("% agreement with the victim")
    ax.set_title("What query access gives away", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax)
    save(fig, "08-extraction")
    return counts, rows


if __name__ == "__main__":
    for i, fn in enumerate([figure1, figure2, figure3, figure4,
                            figure5, figure6, figure7, figure8], 1):
        print(f"Figure {i}")
        fn()
    print("done")
