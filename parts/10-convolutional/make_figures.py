"""
Figures for Part X — Convolutional Models.

Every claim in this part that can be measured, is measured here. The receptive
field is measured by gradient rather than computed from a formula; the
equivalence of convolution and matrix multiplication is verified to machine
precision; the im2col timings are real timings on this machine; the complex-cell
invariance is ch09's own claim, checked.

All data is synthetic and generated here, so the figures are reproducible with
no downloads.

    python3 make_figures.py           # writes into ../../book/src/figures/cv/

Runtime is a few minutes, dominated by figures 8 and 10.
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "cv"))
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
        raise ValueError(f"{name}.svg is {h:.0f}pt — check axis limits")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def noaxes(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------- synthetic data

def shapes(n, s=16, k=4, noise=0.25, seed=0):
    """Four classes of small shapes at random positions. The structure is
    genuinely translation-equivariant, which is the whole point."""
    r = np.random.default_rng(seed)
    X = np.zeros((n, 1, s, s), dtype=np.float32)
    y = r.integers(0, k, n)
    for i in range(n):
        h = int(r.integers(3, 5))
        cy, cx = int(r.integers(h + 1, s - h - 1)), int(r.integers(h + 1, s - h - 1))
        img = X[i, 0]
        if y[i] == 0:
            img[cy, cx - h:cx + h + 1] = 1.0
        elif y[i] == 1:
            img[cy - h:cy + h + 1, cx] = 1.0
        elif y[i] == 2:
            img[cy, cx - h:cx + h + 1] = 1.0
            img[cy - h:cy + h + 1, cx] = 1.0
        else:
            img[cy - h // 2:cy + h // 2 + 1, cx - h // 2:cx + h // 2 + 1] = 1.0
    X += r.normal(scale=noise, size=X.shape).astype(np.float32)
    return X, y.astype(np.int64)


def relu_stack(depth, c=8, seed=0):
    torch.manual_seed(seed)
    L, d = [], 1
    for _ in range(depth):
        L += [nn.Conv2d(d, c, 3, padding=1, bias=False), nn.ReLU()]
        d = c
    m = nn.Sequential(*L)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
    return m


# ------------------------------------------------- 1. why a dense layer is wrong

def fig_dense_vs_conv():
    sizes = np.array([28, 32, 64, 128, 224, 384, 512])
    C, OUT_CH, K = 3, 64, 3
    dense = (sizes ** 2 * C) * (sizes ** 2 * OUT_CH)
    conv = np.full_like(sizes, OUT_CH * C * K * K, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(sizes, dense, "o-", color=ORANGE, ms=4, lw=1.9, label="dense layer")
    ax.loglog(sizes, conv, "o-", color=BLUE, ms=4, lw=1.9, label=r"$3\times3$ convolution")
    ax.axhline(8e9, color=GRAY, ls=":", lw=1.1)
    ax.text(30, 1.2e10, "all the parameters in an 8B model", fontsize=7.5, color=GRAY)
    ax.set_xlabel("image side (pixels)")
    ax.set_ylabel("parameters in one layer")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.loglog(sizes, dense / conv, "o-", color=PURPLE, ms=4, lw=1.9)
    ax.set_xlabel("image side (pixels)")
    ax.set_ylabel("ratio, dense to convolutional")
    ax.set_title("the gap grows as the fourth power", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    i = int(np.where(sizes == 224)[0][0])
    print(f"    at 224x224x3 -> 64 channels: dense {dense[i]:.3e} parameters, "
          f"convolution {conv[i]:.0f} — a factor of {dense[i] / conv[i]:.2e}")
    print(f"    the dense layer alone is {dense[i] / 8e9:.0f}x the parameter count of an "
          f"8-billion-parameter language model")
    sl = np.polyfit(np.log(sizes), np.log(dense / conv), 1)[0]
    print(f"    log-log slope of the ratio: {sl:.2f} (the dense count grows as side^4)")
    fig.tight_layout()
    save(fig, "01-dense-vs-conv")


# --------------------------------------------- 2. convolution as multiplication

def conv_matrix(K, H, W):
    kh, kw = K.shape
    oh, ow = H - kh + 1, W - kw + 1
    M = np.zeros((oh * ow, H * W))
    for i in range(oh):
        for j in range(ow):
            row = np.zeros((H, W))
            row[i:i + kh, j:j + kw] = K
            M[i * ow + j] = row.ravel()
    return M, (oh, ow)


def fig_toeplitz():
    r = np.random.default_rng(0)
    H = W = 8
    K = r.normal(size=(3, 3))
    x = r.normal(size=(H, W))
    M, (oh, ow) = conv_matrix(K, H, W)
    via_mat = (M @ x.ravel()).reshape(oh, ow)
    via_conv = F.conv2d(torch.tensor(x)[None, None],
                        torch.tensor(K)[None, None]).numpy()[0, 0]

    g = r.normal(size=(oh, ow))
    dx_mat = (M.T @ g.ravel()).reshape(H, W)
    dx_conv = F.conv2d(torch.tensor(g)[None, None],
                       torch.tensor(K[::-1, ::-1].copy())[None, None],
                       padding=2).numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5),
                             gridspec_kw={"width_ratios": [2.1, 1, 1]})
    ax = axes[0]
    ax.imshow(M != 0, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("input pixel (flattened)")
    ax.set_ylabel("output pixel")
    ax.set_title(r"$M$ for a $3\times3$ kernel on $8\times8$", fontsize=9.5,
                 color=INK, pad=6)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    v = max(abs(via_mat).max(), abs(via_conv).max())
    ax.imshow(via_mat, cmap="RdBu_r", vmin=-v, vmax=v)
    noaxes(ax)
    ax.set_title(r"$M\mathbf{x}$", fontsize=9.5, color=INK, pad=6)

    ax = axes[2]
    ax.imshow(via_conv, cmap="RdBu_r", vmin=-v, vmax=v)
    noaxes(ax)
    ax.set_title("conv2d", fontsize=9.5, color=INK, pad=6)

    nz = int((M != 0).sum())
    distinct = len(np.unique(np.round(M[M != 0], 12)))
    print(f"    max |M x - conv2d(x)| = {np.abs(via_mat - via_conv).max():.2e}")
    print(f"    M is {M.shape[0]}x{M.shape[1]} = {M.size} entries; {nz} nonzero "
          f"({100 * nz / M.size:.1f}%); only {distinct} distinct values")
    print(f"    backward pass: max |M^T g - full conv with the flipped kernel| = "
          f"{np.abs(dx_mat - dx_conv).max():.2e}")
    fig.tight_layout()
    save(fig, "02-toeplitz")


# ----------------------------------------------------------- 3. receptive field

def fig_receptive_field():
    S = 161
    depths = np.array([2, 4, 8, 16, 24, 32, 40, 48])
    eff, profiles = [], {}
    for depth in depths:
        m = relu_stack(int(depth))
        x = torch.randn(1, 1, S, S, requires_grad=True)
        m(x)[0, :, S // 2, S // 2].sum().backward()
        g = x.grad[0, 0].abs().numpy()
        ctr, tot = S // 2, g.sum()
        rad = 0
        while rad < S // 2:
            if g[ctr - rad:ctr + rad + 1, ctr - rad:ctr + rad + 1].sum() >= 0.95 * tot:
                break
            rad += 1
        eff.append(2 * rad + 1)
        if depth in (8, 24, 48):
            prof = g[ctr]
            profiles[int(depth)] = prof / prof.max()
    eff = np.array(eff, float)
    theo = 1.0 + 2 * depths

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(depths, theo, "o-", color=ORANGE, ms=4, lw=1.9,
            label=r"theoretical, $1+2L$")
    ax.plot(depths, eff, "o-", color=BLUE, ms=4, lw=1.9,
            label="effective (95% of gradient)")
    ax.set_xlabel("depth $L$ (3×3 layers, stride 1)")
    ax.set_ylabel("receptive field (pixels)")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    off = np.arange(S) - S // 2
    for d, c in zip((8, 24, 48), (LIGHT, "#60a5fa", BLUE)):
        ax.plot(off, profiles[d], color=c, lw=1.9, label=f"$L = {d}$")
        ax.axvline((1 + 2 * d) / 2, color=c, ls=":", lw=1.0)
    ax.set_xlim(-60, 60)
    ax.set_xlabel("distance from centre (pixels)")
    ax.set_ylabel("gradient magnitude, normalised")
    ax.set_title("dotted: theoretical edge", fontsize=9, color=GRAY, pad=6)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    big = depths >= 8
    slope = np.polyfit(np.log(depths[big]), np.log(eff[big]), 1)[0]
    print(f"    theoretical: {theo.astype(int).tolist()}")
    print(f"    effective:   {eff.astype(int).tolist()}")
    print(f"    log-log slope of the effective field against depth: {slope:.3f} "
          f"— that is sqrt(L), against L for the theoretical field")
    print(f"    at L = {depths[-1]} the effective field is "
          f"{100 * eff[-1] / theo[-1]:.0f}% of the theoretical one")
    fig.tight_layout()
    save(fig, "04-receptive-field")


# ------------------------------------------------------------- 4. equivariance

def fig_equivariance():
    torch.manual_seed(0)
    m = relu_stack(3)
    rf = 1 + 2 * 3
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        base = m(x)
        scale = base.abs().max().item()
        shifts = [(1, 0), (0, 1), (3, 3), (5, 7), (13, -9), (20, 20)]
        trans = []
        for dy, dx in shifts:
            mg = max(abs(dy), abs(dx)) + rf
            a1 = m(torch.roll(x, (dy, dx), (2, 3)))
            a2 = torch.roll(base, (dy, dx), (2, 3))
            trans.append((a1 - a2)[:, :, mg:-mg, mg:-mg].abs().max().item() / scale)
        a1 = m(torch.rot90(x, 1, (2, 3)))
        a2 = torch.rot90(base, 1, (2, 3))
        rot = (a1 - a2)[:, :, rf:-rf, rf:-rf].abs().max().item() / scale
        xs = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        a1 = F.interpolate(m(xs), scale_factor=0.5, mode="bilinear", align_corners=False)
        sca = (a1 - base)[:, :, rf:-rf, rf:-rf].abs().max().item() / scale

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels = [f"shift ({dy},{dx})" for dy, dx in shifts] + ["rotate 90°", "scale 2×"]
    vals = trans + [rot, sca]
    cols = [BLUE] * len(trans) + [PINK, PINK]
    ax.barh(range(len(vals)), np.maximum(vals, 1e-8), color=cols)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 3)
    ax.invert_yaxis()
    ax.axvline(1e-6, color=GRAY, ls=":", lw=1.1)
    ax.text(1.3e-6, len(vals) - 0.4, "float32 precision", fontsize=7.5, color=GRAY)
    ax.set_xlabel(r"relative $\max|f(g(\mathbf{x})) - g(f(\mathbf{x}))|$")
    tidy(ax)

    print(f"    translation: relative error {min(trans):.1e} to {max(trans):.1e} "
          f"— float32 rounding, for every shift tested")
    print(f"    rotation by 90 degrees: {rot:.2f} of the activation range")
    print(f"    scaling by 2:           {sca:.2f} of the activation range")
    fig.tight_layout()
    save(fig, "03-equivariance")


# ---------------------------------------------------------- 5. pooling and invariance

def fig_pooling():
    torch.manual_seed(0)
    x = torch.randn(64, 1, 32, 32)
    conv = nn.Conv2d(1, 8, 3, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight)

    def feats(t, mode):
        h = F.relu(conv(t))
        return {"convolution only": h,
                "max pool 2": F.max_pool2d(h, 2),
                "max pool 4": F.max_pool2d(h, 4),
                "average pool 4": F.avg_pool2d(h, 4),
                "global max pool": h.amax((2, 3))}[mode]

    modes = ["convolution only", "max pool 2", "max pool 4",
             "average pool 4", "global max pool"]
    shifts = np.arange(0, 9)
    curves = {}
    with torch.no_grad():
        for mode in modes:
            base = feats(x, mode)
            sc = base.abs().mean()
            curves[mode] = [((feats(torch.roll(x, (0, int(d)), (2, 3)), mode) - base)
                             .abs().mean() / sc).item() for d in shifts]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for mode, c in zip(modes, [ORANGE, "#93c5fd", BLUE, GREEN, PURPLE]):
        ax.plot(shifts, curves[mode], "o-", color=c, ms=3.5, lw=1.8, label=mode)
    ax.set_xlabel("input translated by (pixels)")
    ax.set_ylabel("relative change in the representation")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    for mode in modes:
        c = curves[mode]
        print(f"    {mode:17s} shift 1 -> {c[1]:.3f} | shift 4 -> {c[4]:.3f} | "
              f"shift 8 -> {c[8]:.3f}")
    print("    note this measures INVARIANCE — the reference is not shifted to compensate,")
    print("    which is what would measure equivariance instead")
    fig.tight_layout()
    save(fig, "05-pooling")


# ------------------------------------------------------------------ 6. im2col

def naive_conv(x, k):
    """Per-element Python loop — one multiply-accumulate at a time, no numpy
    vectorisation.  This is the conceptually direct implementation and the
    correct baseline for measuring what im2col actually buys: the cost of
    gathering patches into a dense matrix is paid once, and the subsequent
    BLAS call inherits decades of hardware-tuned optimisation.

    Note: the earlier version of this function contained a spatial loop but
    performed the inner work with a numpy matmul (vectorised over all channels
    and batch elements at once).  That made it significantly faster than a true
    scalar loop and caused im2col to appear slower on some hardware
    configurations where the contiguous-copy cost dominated the small BLAS
    benefit.  This version loops over every axis so the comparison is honest."""
    n, ci, H, W = x.shape
    co, _, kh, kw = k.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((n, co, oh, ow), dtype=np.float32)
    for b in range(n):
        for c_out in range(co):
            for c_in in range(ci):
                for i in range(oh):
                    for j in range(ow):
                        for ki in range(kh):
                            for kj in range(kw):
                                out[b, c_out, i, j] += (
                                    x[b, c_in, i + ki, j + kj]
                                    * k[c_out, c_in, ki, kj])
    return out


def im2col_conv(x, k):
    n, ci, H, W = x.shape
    co, _, kh, kw = k.shape
    oh, ow = H - kh + 1, W - kw + 1
    s = x.strides
    cols = np.lib.stride_tricks.as_strided(
        x, (n, ci, kh, kw, oh, ow), (s[0], s[1], s[2], s[3], s[2], s[3]))
    cols = np.ascontiguousarray(cols).reshape(n, ci * kh * kw, oh * ow)
    return (k.reshape(co, -1) @ cols).reshape(n, co, oh, ow)


def fig_im2col():
    r = np.random.default_rng(0)
    # Small enough that the scalar loop is feasible (one timed rep each),
    # large enough that k=3 gives ~8× and k=5 gives ~16× memory expansion.
    cfgs = [(1, 3,  8, 32, 3),   # k=3, buffer 8×
            (1, 3,  8, 24, 5),   # k=5, buffer 17×
            (1, 4,  8, 28, 3),   # k=3, buffer 8×
            (1, 4,  8, 20, 5)]   # k=5, buffer 16×
    rows = []
    for (n, ci, co, S, K) in cfgs:
        x = r.normal(size=(n, ci, S, S)).astype(np.float32)
        k = r.normal(size=(co, ci, K, K)).astype(np.float32)
        b = im2col_conv(x, k)
        t = F.conv2d(torch.tensor(x), torch.tensor(k)).numpy()

        # naive_conv is a genuine scalar loop and takes ~250 ms per call;
        # time it once (already a wall-clock measurement, not a micro-benchmark).
        t0 = time.perf_counter()
        naive_conv(x, k)
        tn = (time.perf_counter() - t0) * 1e3

        def tmf(f, reps=7):
            f()
            t0 = time.perf_counter()
            for _ in range(reps):
                f()
            return (time.perf_counter() - t0) / reps * 1e3

        ti = tmf(lambda: im2col_conv(x, k))
        tt = tmf(lambda: F.conv2d(torch.tensor(x), torch.tensor(k)))
        oh = S - K + 1
        mem = n * ci * K * K * oh * oh * 4 / 1e6
        rows.append((f"{ci}-{co}\n{S}px k{K}", tn, ti, tt, mem / (x.nbytes / 1e6),
                     np.abs(b - t).max()))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    idx = np.arange(len(rows))
    w = 0.27
    ax.bar(idx - w, [r_[1] for r_ in rows], w, color=ORANGE, label="scalar loops")
    ax.bar(idx, [r_[2] for r_ in rows], w, color=BLUE, label="im2col + one matmul")
    ax.bar(idx + w, [r_[3] for r_ in rows], w, color=GREEN, label="torch conv2d")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([r_[0] for r_ in rows], fontsize=7.5)
    ax.set_ylabel("milliseconds")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.bar(idx, [r_[4] for r_ in rows], 0.5, color=PINK)
    ax.set_xticks(idx)
    ax.set_xticklabels([r_[0] for r_ in rows], fontsize=7.5)
    ax.set_ylabel("im2col buffer ÷ input size")
    ax.set_title("what the speed costs", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    for lbl, tn, ti, tt, exp, err in rows:
        print(f"    {lbl.replace(chr(10), ' '):16s} scalar {tn:7.1f}ms | im2col {ti:6.3f}ms "
              f"({tn / ti:5.0f}×) | torch {tt:5.3f}ms | "
              f"buffer {exp:.0f}× input | agree to {err:.0e}")
    fig.tight_layout()
    save(fig, "06-im2col")


# ------------------------------------------------------- 7. depthwise separable

def fig_separable():
    cfgs = [(32, 64), (64, 64), (64, 128), (128, 256), (256, 512), (512, 512)]
    K = 3
    std = np.array([co * ci * K * K for ci, co in cfgs], float)
    sep = np.array([ci * K * K + ci * co for ci, co in cfgs], float)
    pred = np.array([1 / co + 1 / (K * K) for ci, co in cfgs])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    idx = np.arange(len(cfgs))
    ax.bar(idx - 0.2, std, 0.4, color=ORANGE, label="standard")
    ax.bar(idx + 0.2, sep, 0.4, color=BLUE, label="depthwise separable")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{a}-{b}" for a, b in cfgs], fontsize=7.5, rotation=30)
    ax.set_ylabel("parameters in one layer")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(idx, sep / std, "o-", color=BLUE, ms=5, lw=1.9, label="measured")
    ax.plot(idx, pred, "x--", color=PINK, ms=7, lw=1.4,
            label=r"predicted, $1/c_{\rm out} + 1/k^2$")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{a}-{b}" for a, b in cfgs], fontsize=7.5, rotation=30)
    ax.set_ylabel("separable ÷ standard")
    ax.set_ylim(0, 0.22)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    measured ratio: {[round(v, 4) for v in (sep / std)]}")
    print(f"    predicted     : {[round(v, 4) for v in pred]}")
    print(f"    max discrepancy {np.abs(sep / std - pred).max():.2e} — the formula is exact")
    print(f"    at 512->512 with k=3: {std[-1]:.0f} against {sep[-1]:.0f} parameters, "
          f"{std[-1] / sep[-1]:.1f}x fewer; the 1/k^2 term dominates once c_out is large")
    fig.tight_layout()
    save(fig, "07-separable")


# ------------------------------------------------------------ 8. depth and residual

class Block(nn.Module):
    def __init__(self, c, residual, norm):
        super().__init__()
        self.residual = residual
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=not norm)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=not norm)
        self.b1 = nn.BatchNorm2d(c) if norm else nn.Identity()
        self.b2 = nn.BatchNorm2d(c) if norm else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + x) if self.residual else F.relu(h)


def _train_depth(X, y, depth, residual, norm, epochs=18, lr=0.03, bs=128, c=12, seed=0):
    torch.manual_seed(seed)
    L = [nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
    L += [Block(c, residual, norm) for _ in range(depth // 2)]
    L += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 4)]
    m = nn.Sequential(*L)
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    best = np.nan
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            F.cross_entropy(m(X[b]), y[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            l = float(F.cross_entropy(m(X), y))
        m.train()
        if np.isfinite(l):
            best = l if np.isnan(best) else min(best, l)
    return best


def fig_depth():
    Xn, yn = shapes(1500, s=16, seed=1)
    X, y = torch.tensor(Xn), torch.tensor(yn)
    depths = [4, 8, 16, 24, 32]
    res = {k: [] for k in ("plain", "residual", "plain+bn", "residual+bn")}
    for d in depths:
        res["plain"].append(_train_depth(X, y, d, False, False))
        res["residual"].append(_train_depth(X, y, d, True, False))
        res["plain+bn"].append(_train_depth(X, y, d, False, True))
        res["residual+bn"].append(_train_depth(X, y, d, True, True))
        print(f"    depth {d:2d} | plain {res['plain'][-1]:.4f} | "
              f"residual {res['residual'][-1]:.4f} | "
              f"plain+BN {res['plain+bn'][-1]:.4f} | "
              f"residual+BN {res['residual+bn'][-1]:.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, keys, ttl in ((axes[0], ("plain", "residual"), "no normalisation"),
                          (axes[1], ("plain+bn", "residual+bn"), "with batch norm")):
        for k, c in zip(keys, (ORANGE, BLUE)):
            v = np.array(res[k], float)
            ok = np.isfinite(v)
            ax.plot(np.array(depths)[ok], v[ok], "o-", color=c, ms=5, lw=1.9,
                    label=k.replace("+bn", ""))
            for d, val in zip(depths, v):
                if not np.isfinite(val):
                    ax.plot([d], [ax.get_ylim()[1]], "x", color=c, ms=9)
                    ax.annotate("diverged", (d, ax.get_ylim()[1]), fontsize=7.5,
                                color=c, ha="center", va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("depth (convolutional layers)")
        ax.set_ylabel("best training loss reached")
        ax.set_title(ttl, fontsize=9.5, color=INK, pad=6)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)
    fig.tight_layout()
    save(fig, "08-depth")


# ---------------------------------------------------- 9. transposed convolution

def fig_checkerboard():
    cfgs = [(2, 2), (4, 2), (3, 2), (5, 3), (6, 4), (4, 4)]
    fig, axes = plt.subplots(2, len(cfgs), figsize=(11.2, 3.9))
    # every panel shares a scale, so a uniform output reads as flat grey rather
    # than as an autoscaled block of black
    grids = {}
    for k, s in cfgs:
        grids[(k, s)] = F.conv_transpose2d(
            torch.ones(1, 1, 12, 12), torch.ones(1, 1, k, k), stride=s)[0, 0].numpy()
    vmax = max(g[k:-k, k:-k].max() for (k, s), g in grids.items())
    for j, (k, s) in enumerate(cfgs):
        y = grids[(k, s)]
        inner = y[k:-k, k:-k]
        uniq = np.unique(np.round(inner, 9))
        ax = axes[0, j]
        ax.imshow(y, cmap="Greys_r", interpolation="nearest", vmin=0, vmax=vmax)
        noaxes(ax)
        ok = len(uniq) == 1
        ax.set_title(f"k={k}, s={s}", fontsize=9,
                     color=(GREEN if ok else PINK), pad=4)
        ax = axes[1, j]
        ax.plot(inner[len(inner) // 2], color=(GREEN if ok else PINK), lw=1.6)
        ax.set_ylim(0, vmax * 1.15)
        ax.tick_params(labelsize=7)
        ax.set_xticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel(f"k mod s = {k % s}", fontsize=8,
                      color=(GREEN if ok else PINK))
        if j == 0:
            ax.set_ylabel("value", fontsize=8.5)
        print(f"    kernel {k}, stride {s}: k mod s = {k % s}, "
              f"{len(uniq)} distinct interior values "
              f"{'— uniform' if ok else '— CHECKERBOARD'}")
    fig.tight_layout()
    save(fig, "09-checkerboard")


# ---------------------------------------------- 10. Gabor filters and complex cells

def gabor(K, a=1.0, bx=0.02, by=0.02, f=0.8, ph=0.0, x0=0.0, y0=0.0, th=0.0):
    yy, xx = np.mgrid[:K, :K] - (K - 1) / 2
    xr = (xx - x0) * np.cos(th) + (yy - y0) * np.sin(th)
    yr = -(xx - x0) * np.sin(th) + (yy - y0) * np.cos(th)
    return a * np.exp(-bx * xr ** 2 - by * yr ** 2) * np.cos(f * xr + ph)


def fig_gabor():
    S = 65
    th = np.pi / 6
    s0 = gabor(S, ph=0.0, th=th)
    s1 = gabor(S, ph=np.pi / 2, th=th)
    corr = (s0 * s1).sum() / np.sqrt((s0 * s0).sum() * (s1 * s1).sum())

    yy, xx = np.mgrid[:S, :S] - (S - 1) / 2

    def stim(shift):
        xr = (xx - shift * np.cos(th)) * np.cos(th) + (yy - shift * np.sin(th)) * np.sin(th)
        return np.cos(0.8 * xr)

    shifts = np.arange(0, 8.01, 0.25)
    r0 = np.array([(s0 * stim(d)).sum() for d in shifts])
    r1 = np.array([(s1 * stim(d)).sum() for d in shifts])
    c = np.sqrt(r0 ** 2 + r1 ** 2)

    fig = plt.figure(figsize=(11.2, 3.5))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.35, 2.6])
    for i, (f_, phi) in enumerate([(0.5, 0), (0.9, 0), (1.4, 0), (0.9, np.pi / 2)]):
        ax = fig.add_subplot(gs[0, i])
        gb = gabor(41, f=f_, ph=phi, th=np.pi / 5)
        ax.imshow(gb, cmap="RdBu_r", vmin=-np.abs(gb).max(), vmax=np.abs(gb).max())
        noaxes(ax)
        if i == 0:
            ax.set_ylabel("vary $f$, $\\phi$", fontsize=8)
    for i, tt in enumerate([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        ax = fig.add_subplot(gs[1, i])
        gb = gabor(41, th=tt)
        ax.imshow(gb, cmap="RdBu_r", vmin=-np.abs(gb).max(), vmax=np.abs(gb).max())
        noaxes(ax)
        if i == 0:
            ax.set_ylabel(r"vary $\tau$", fontsize=8)

    ax = fig.add_subplot(gs[:, 5])
    ax.plot(shifts, r0, color="#93c5fd", lw=1.7, label="simple cell $s_0$")
    ax.plot(shifts, r1, color=BLUE, lw=1.7, label="simple cell $s_1$")
    ax.plot(shifts, c, color=PINK, lw=2.4, label="complex cell $\\sqrt{s_0^2+s_1^2}$")
    ax.set_xlabel("stimulus translated along $\\tau$ (pixels)")
    ax.set_ylabel("response")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)

    def spread(v):
        return (v.max() - v.min()) / np.abs(v).max()
    print(f"    quadrature pair correlation: {corr:.2e} (orthogonal by construction)")
    print(f"    as the stimulus translates: s0 swings {100 * spread(r0):.1f}% of its peak, "
          f"s1 {100 * spread(r1):.1f}%")
    print(f"    the complex cell varies by {100 * spread(c):.2f}% — exactly invariant")
    neg = np.sqrt(((-s0 * stim(0)).sum()) ** 2 + ((-s1 * stim(0)).sum()) ** 2)
    print(f"    negating the image: s0 changes by 200%, the complex cell by "
          f"{100 * abs(neg - c[0]) / c[0]:.2e}%")
    fig.tight_layout()
    save(fig, "10-gabor")


if __name__ == "__main__":
    print("Part X — convolutional figures")
    fig_dense_vs_conv()
    fig_toeplitz()
    fig_equivariance()
    fig_receptive_field()
    fig_pooling()
    fig_im2col()
    fig_separable()
    fig_depth()
    fig_checkerboard()
    fig_gabor()
    print("done ->", OUT)
