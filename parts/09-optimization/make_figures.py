"""
Figures for Part IX — Optimization.

Every claim in this part that can be measured, is measured here. The rising
gradient norm is a real training run, not a sketch; the conjugate-gradient
termination is real arithmetic in real floating point; the batch-norm learning
rate range is a real sweep over real networks; the cliff is a real loss surface.

    python3 make_figures.py           # writes into ../../book/src/figures/op/

Runtime is a few minutes, dominated by figures 2 and 9.
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "op"))
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


# ----------------------------------------------------------------- shared nets

def spiral(n_per, k=3, turns=2.2, jitter=0.16, seed=0):
    """Interleaved spiral arms. Jitter controls class overlap."""
    r = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        t = np.sqrt(r.random(n_per)) * turns * 2 * np.pi
        rad = t / (turns * 2 * np.pi)
        ang = t + c * 2 * np.pi / k + r.normal(scale=jitter, size=n_per)
        X.append(np.c_[rad * np.cos(ang), rad * np.sin(ang)])
        y.append(np.full(n_per, c))
    return np.vstack(X), np.concatenate(y)


class MLP:
    """Flat-vector ReLU MLP with hand-written backprop. No framework."""

    def __init__(self, dims, seed=0):
        r = np.random.default_rng(seed)
        self.shapes, parts = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            self.shapes += [(a, b), (b,)]
            parts += [(r.normal(size=(a, b)) * np.sqrt(2 / a)).ravel(), np.zeros(b)]
        self.v0 = np.concatenate(parts)

    def split(self, v):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(v[i:i + n].reshape(s))
            i += n
        return out

    def loss_grad(self, v, X, y):
        P = self.split(v)
        Ws, bs = P[0::2], P[1::2]
        n, L = len(X), len(Ws)
        a, cache = X, []
        for i in range(L):
            z = a @ Ws[i] + bs[i]
            cache.append(a)
            a = np.maximum(z, 0) if i < L - 1 else z
        z = a - a.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n
        gs = [None] * (2 * L)
        for i in range(L - 1, -1, -1):
            gs[2 * i], gs[2 * i + 1] = cache[i].T @ d, d.sum(0)
            if i:
                d = (d @ Ws[i].T) * (cache[i] > 0)
        return loss, np.concatenate([g.ravel() for g in gs]), (p.argmax(1) != y).mean()


# ------------------------------------------------------- 1. the three objectives

def _logreg_run(sep, scale, seed, d=2, ntr=40, iters=200_000, lr=0.5):
    """Full-batch logistic regression. d > 2 adds pure-noise coordinates."""
    r = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = sep
    ytr = r.integers(0, 2, ntr)
    Xtr = np.where(ytr[:, None] == 1, mu, -mu) + r.normal(scale=scale, size=(ntr, d))
    yte = r.integers(0, 2, 8000)
    Xte = np.where(yte[:, None] == 1, mu, -mu) + r.normal(scale=scale, size=(8000, d))
    A = np.c_[Xtr, np.ones(ntr)] * np.where(ytr == 1, 1.0, -1.0)[:, None]
    B = np.c_[Xte, np.ones(len(Xte))] * np.where(yte == 1, 1.0, -1.0)[:, None]
    w = np.zeros(d + 1)
    ri = set(np.unique(np.round(np.logspace(0, np.log10(iters), 300)).astype(int)).tolist())
    H = []
    for t in range(1, iters + 1):
        w -= lr * (A.T @ (-1.0 / (1.0 + np.exp(A @ w)))) / ntr
        if t in ri:
            H.append((t, np.logaddexp(0.0, -(A @ w)).mean(), (A @ w <= 0).mean(),
                      (B @ w <= 0).mean(), np.linalg.norm(w)))
    return np.array(H).T


def fig_three_objectives():
    """Surrogate loss, 0-1 loss, and the fact that the loss has no minimum."""
    t, nll, e_tr, e_te, nw = _logreg_run(1.35, 0.85, seed=9)
    k = int(np.argmax(e_tr == 0))
    best = int(np.argmin(e_te))
    to, _, e_tr_o, e_te_o, _ = _logreg_run(1.4, 1.0, seed=3, d=12, lr=0.01,
                                           iters=300_000)
    bo = int(np.argmin(e_te_o))
    ko = int(np.argmax(e_tr_o == 0))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    ax = axes[0, 0]
    ax.loglog(t, nll, color=BLUE, lw=2.0, label="training loss (surrogate)")
    ax.loglog(t, np.maximum(e_tr, 1e-4), color=ORANGE, lw=1.8,
              label="training error (0–1)")
    ax.axvline(t[k], color=GRAY, ls=":", lw=1.1)
    ax.set_ylim(3e-5, 1.2)
    ax.annotate("0–1 error hits zero;\nthe surrogate keeps falling",
                xy=(t[k], nll[k]), xytext=(t[k] * 1.5, 0.35), fontsize=8, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss / error")
    ax.set_title("(a) separated classes", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    tidy(ax)

    ax = axes[0, 1]
    ax.semilogx(t, e_te, color=GREEN, lw=2.0)
    ax.axvline(t[k], color=GRAY, ls=":", lw=1.1)
    ax.plot([t[best]], [e_te[best]], "o", color=PINK, ms=7, mec="white", mew=1.2)
    ax.annotate(f"best test error,\n{t[best] / t[k]:.0f}x later", xy=(t[best], e_te[best]),
                xytext=(t[best] / 300, e_te[best] + 0.0016), fontsize=8, color=PINK,
                arrowprops=dict(arrowstyle="->", color=PINK, lw=0.8))
    ax.set_xlabel("iteration")
    ax.set_ylabel("test error (0–1)")
    ax.set_title("(b) the same run, test error only", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    ax = axes[1, 0]
    ax.semilogx(t, nw, color=PURPLE, lw=2.0)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|w\|$")
    ax.set_title("(c) no minimum exists to converge to", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    ax = axes[1, 1]
    ax.semilogx(to, e_te_o, color=GREEN, lw=2.0)
    ax.axvline(to[ko], color=GRAY, ls=":", lw=1.1)
    ax.plot([to[bo]], [e_te_o[bo]], "o", color=PINK, ms=7, mec="white", mew=1.2)
    ax.annotate(f"best at iteration {to[bo]:.0f},\nlong before", xy=(to[bo], e_te_o[bo]),
                xytext=(to[bo] * 1.4, e_te_o[bo] + 0.009), fontsize=8, color=PINK,
                arrowprops=dict(arrowstyle="->", color=PINK, lw=0.8))
    ax.set_xlabel("iteration")
    ax.set_ylabel("test error (0–1)")
    ax.set_title("(d) ten noise coordinates added", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    print(f"    separated: train 0-1 hits 0 at iter {t[k]:.0f}, surrogate there {nll[k]:.4f}")
    print(f"      test error {e_te[k]:.4f} at that moment -> best {e_te[best]:.4f} "
          f"at iter {t[best]:.0f} ({t[best] / t[k]:.0f}x later, "
          f"{100 * (e_te[k] - e_te[best]) / e_te[k]:.0f}% relative gain)")
    print(f"      surrogate ends at {nll[-1]:.5f}, never 0; ||w|| {nw[0]:.2f} -> {nw[-1]:.1f}, "
          f"last decade alone adds {nw[-1] - nw[np.argmax(t > t[-1] / 10)]:.1f}")
    print(f"    with noise coords: train 0-1 hits 0 at iter {to[ko]:.0f}; test error best "
          f"{e_te_o[bo]:.4f} at iter {to[bo]:.0f} (before that), rises to {e_te_o[-1]:.4f} "
          f"(+{100 * (e_te_o[-1] - e_te_o[bo]) / e_te_o[bo]:.0f}% relative)")
    fig.tight_layout()
    save(fig, "01-three-objectives")


# ------------------------------------------- 2. no critical point is ever reached

def fig_gradient_norm():
    """Gradient norm and g'Hg during a successful run that never converges."""
    Xtr, ytr = spiral(2000, jitter=0.45, seed=1)
    Xva, yva = spiral(1500, jitter=0.45, seed=2)
    net = MLP([2, 16, 16, 3], seed=5)
    v = net.v0.copy()
    rng = np.random.default_rng(0)
    lr, bs, epochs, n = 0.05, 64, 300, len(Xtr)

    scat_ep, scat_g, H = [], [], []
    for ep in range(epochs):
        idx = rng.permutation(n)
        mb = []
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            _, g, _ = net.loss_grad(v, Xtr[b], ytr[b])
            mb.append(np.linalg.norm(g))
            v = v - lr * g
        scat_ep.append(ep)
        scat_g.append(mb[rng.integers(len(mb))])
        L, gf, etr = net.loss_grad(v, Xtr, ytr)
        _, _, eva = net.loss_grad(v, Xva, yva)
        step = 1e-5 / (np.linalg.norm(gf) + 1e-12)
        _, g2, _ = net.loss_grad(v + step * gf, Xtr, ytr)
        H.append((ep, L, np.mean(mb), float((g2 - gf) @ gf / step), etr, eva,
                  np.linalg.norm(v), float(gf @ gf)))
    a = np.array(H)

    def smooth(x, k=15):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    ax = axes[0]
    ax.scatter(scat_ep, scat_g, s=5, color=LIGHT, zorder=1)
    ax.plot(np.arange(len(smooth(a[:, 2]))) + 7, smooth(a[:, 2]), color=BLUE, lw=2, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("minibatch gradient norm")
    tidy(ax)

    ax = axes[1]
    ax.semilogy(np.arange(len(smooth(a[:, 3]))) + 7, np.maximum(smooth(a[:, 3]), 1e-6),
                color=ORANGE, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathbf{g}^\top H \mathbf{g}$")
    tidy(ax)

    ax = axes[2]
    ax.plot(a[:, 0], a[:, 5], color=GREEN, lw=2.2, label="validation error")
    ax.plot(a[:, 0], a[:, 4], color=INK, lw=1.0, ls="--", label="training error")
    ax.set_xlabel("epoch")
    ax.set_ylabel("classification error")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    w = 20
    print(f"    minibatch |g|: {a[:w, 2].mean():.3f} -> {a[-w:, 2].mean():.3f} "
          f"(x{a[-w:, 2].mean() / a[:w, 2].mean():.2f})")
    print(f"    g'Hg:          {a[:w, 3].mean():.4g} -> {a[-w:, 3].mean():.4g} "
          f"(x{a[-w:, 3].mean() / a[:w, 3].mean():.0f})")
    print(f"    validation err {a[0, 5]:.3f} -> {a[-1, 5]:.3f} (best {a[:, 5].min():.3f}); "
          f"||w|| {a[0, 6]:.1f} -> {a[-1, 6]:.1f}")
    gg0, gg1 = a[:w, 7].mean(), a[-w:, 7].mean()
    hh0, hh1 = a[:w, 3].mean(), a[-w:, 3].mean()
    print(f"    full-batch g'g: {gg0:.4g} -> {gg1:.4g} (x{gg1 / gg0:.1f})")
    print(f"    max stable step 2 g'g / g'Hg: {2 * gg0 / hh0:.2f} -> {2 * gg1 / hh1:.3f} "
          f"— it shrank {(2 * gg0 / hh0) / (2 * gg1 / hh1):.0f}x while lr stayed at {lr}")
    fig.tight_layout()
    save(fig, "02-gradient-norm")


# -------------------------------------------------------------- 3. batch size

def fig_batch_size():
    """Estimate noise, throughput, and why second-order methods need big batches."""
    r = np.random.default_rng(7)
    n, d = 20000, 40
    Xf = r.normal(size=(n, d))
    wt = r.normal(size=d)
    yf = Xf @ wt + r.normal(scale=1.0, size=n)
    gfull = Xf.T @ (Xf @ np.zeros(d) - yf) / n

    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                      8192, 16384])
    err_g, err_hg = [], []
    # Ill-conditioned Hessian for the second-order panel.
    S = np.diag(np.logspace(0, 4.0, d))
    Q = np.linalg.qr(r.normal(size=(d, d)))[0]
    Xi = r.normal(size=(n, d)) @ (Q @ np.sqrt(S) @ Q.T)
    yi = Xi @ wt + r.normal(scale=1.0, size=n)
    Hi = Xi.T @ Xi / n
    gi_full = -Xi.T @ yi / n
    step_full = np.linalg.solve(Hi, gi_full)
    kappa = np.linalg.cond(Hi)

    for b in sizes:
        eg, eh = [], []
        for _ in range(160):
            idx = r.integers(0, n, b)
            g = Xf[idx].T @ (-yf[idx]) / b
            eg.append(np.linalg.norm(g - gfull) / np.linalg.norm(gfull))
            gi = -Xi[idx].T @ yi[idx] / b
            eh.append(np.linalg.norm(np.linalg.solve(Hi, gi) - step_full)
                      / np.linalg.norm(step_full))
        err_g.append(np.mean(eg))
        err_hg.append(np.mean(eh))
    err_g, err_hg = np.array(err_g), np.array(err_hg)

    # Real throughput on this machine.
    W1, W2 = r.normal(size=(d, 512)), r.normal(size=(512, 512))
    per_ex = []
    for b in sizes:
        Xb = r.normal(size=(b, d))
        reps = max(3, int(3e7 / (b * 512 * 512)))
        np.maximum(Xb @ W1, 0) @ W2
        t0 = time.perf_counter()
        for _ in range(reps):
            np.maximum(Xb @ W1, 0) @ W2
        per_ex.append((time.perf_counter() - t0) / reps / b * 1e6)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    ax = axes[0]
    ax.loglog(sizes, err_g, "o-", color=BLUE, ms=4, lw=1.8, label="measured")
    ax.loglog(sizes, err_g[0] / np.sqrt(sizes), "--", color=GRAY, lw=1.2,
              label=r"$1/\sqrt{b}$")
    ax.set_xlabel("batch size $b$")
    ax.set_ylabel("relative gradient error")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogx(sizes, per_ex, "o-", color=ORANGE, ms=4, lw=1.8, base=2)
    ax.set_xlabel("batch size $b$")
    ax.set_ylabel("µs per example")
    ax.set_title("throughput, this machine", fontsize=9, color=GRAY, pad=6)
    tidy(ax)

    ax = axes[2]
    ax.loglog(sizes, err_g, "o-", color=BLUE, ms=4, lw=1.8, label=r"$\hat{\mathbf{g}}$")
    ax.loglog(sizes, err_hg, "o-", color=PINK, ms=4, lw=1.8,
              label=r"$H^{-1}\hat{\mathbf{g}}$")
    ax.axhline(0.1, color=GRAY, ls=":", lw=1.1)
    ax.set_xlabel("batch size $b$")
    ax.set_ylabel("relative error of the step")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    def need(e, thr):
        k = np.where(e < thr)[0]
        return int(sizes[k[0]]) if len(k) else None

    i128, i4096 = int(np.argmin(abs(sizes - 128))), int(np.argmin(abs(sizes - 4096)))
    print(f"    gradient error: b=128 {err_g[i128]:.4f}, b=4096 {err_g[i4096]:.4f} "
          f"— 32x the compute for {err_g[i128] / err_g[i4096]:.1f}x the accuracy")
    print(f"    per-example cost: b=1 {per_ex[0]:.2f}us, cheapest at b="
          f"{sizes[int(np.argmin(per_ex))]} at {min(per_ex):.2f}us "
          f"(b=1 is {per_ex[0] / min(per_ex):.0f}x more expensive per example)")
    print(f"    kappa(H) = {kappa:.0f}")
    for thr in (0.5, 0.25, 0.1):
        print(f"      to hold step error under {thr:.0%}: gradient b={need(err_g, thr)}, "
              f"Newton b={need(err_hg, thr)}")
    print(f"      at b=128 the Newton step is {err_hg[i128] / err_g[i128]:.0f}x "
          f"noisier than the gradient step")
    fig.tight_layout()
    save(fig, "03-batch-size")


# ------------------------------------------------------ 4. weight-space symmetry

def fig_symmetry():
    """The ReLU scaling hyperbola: a continuum of exactly equivalent minima."""
    r = np.random.default_rng(3)
    x = np.linspace(0.2, 2.0, 60)
    ystar = 1.7 * np.maximum(1.1 * x, 0) + r.normal(scale=0.05, size=60)

    def loss(w1, w2, alpha=0.0):
        return np.mean((w2 * np.maximum(w1 * x, 0) - ystar) ** 2) + alpha * (w1 ** 2 + w2 ** 2)

    g1 = np.linspace(0.08, 4.0, 320)
    g2 = np.linspace(0.08, 4.0, 320)
    G1, G2 = np.meshgrid(g1, g2)
    Z = np.array([[loss(a, b) for a in g1] for b in g2])
    Za = np.array([[loss(a, b, 0.06) for a in g1] for b in g2])

    # Walk the hyperbola w1*w2 = c through the optimum and check the loss is flat.
    c = 1.1 * 1.7
    ws = np.linspace(c / 3.95, 3.95, 400)
    along = np.array([loss(w, c / w) for w in ws])
    dev = (along.max() - along.min()) / along.mean()

    i, j = np.unravel_index(np.argmin(Za), Za.shape)
    w1a, w2a = g1[j], g2[i]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, Zi, ttl in ((axes[0], Z, "no weight decay"),
                        (axes[1], Za, r"with weight decay $\alpha = 0.06$")):
        cs = ax.contourf(G1, G2, np.log10(Zi + 1e-6), levels=26, cmap="Blues_r")
        ax.contour(G1, G2, np.log10(Zi + 1e-6), levels=14, colors="white",
                   linewidths=0.4, alpha=0.55)
        ax.set_xlabel(r"$w_1$ (incoming)")
        ax.set_ylabel(r"$w_2$ (outgoing)")
        ax.set_title(ttl, fontsize=9.5, color=INK, pad=6)
        tidy(ax, grid=False)
    axes[0].plot(ws, c / ws, color=ORANGE, lw=2.2)
    axes[0].text(2.3, 1.55, r"$w_1 w_2 = $ const", color=ORANGE, fontsize=9)
    axes[1].plot(ws, c / ws, color=ORANGE, lw=1.2, ls="--", alpha=0.8)
    axes[1].plot([w1a], [w2a], "o", color=PINK, ms=7, mec="white", mew=1.2)
    axes[1].annotate("unique\nminimum", xy=(w1a, w2a), xytext=(w1a + 0.55, w2a + 0.95),
                     color=PINK, fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color=PINK, lw=0.9))

    print(f"    loss along the hyperbola w1*w2={c:.2f}: min {along.min():.8f}, "
          f"max {along.max():.8f}, relative spread {dev:.2e}")
    print(f"    with alpha=0.06 the minimum is isolated at w1={w1a:.3f}, w2={w2a:.3f}, "
          f"ratio {w2a / w1a:.3f} (symmetric point would be 1.000)")
    print(f"    a 3-layer 64-unit net has 64!^3 = 10^{3 * np.sum(np.log10(np.arange(1, 65))):.0f} "
          f"permutation-equivalent settings")
    fig.tight_layout()
    save(fig, "04-symmetry")


# ------------------------------------------------------------------ 5. cliffs

def fig_cliff():
    """Deep multiplicative parameterisation: a real cliff, and what clipping does."""
    L = 6
    target = 1.0

    def f(w):
        return 0.5 * (w ** L - target) ** 2

    def df(w):
        return (w ** L - target) * L * w ** (L - 1)

    w = np.linspace(0.2, 1.9, 1400)
    lo = f(w)
    gr = np.abs(df(w))

    w0, lr = 1.45, 0.02
    g0 = df(w0)
    w_plain = w0 - lr * g0
    thr = 1.0
    g_clip = g0 * min(1.0, thr / abs(g0))
    w_clip = w0 - lr * g_clip

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(w, lo, color=BLUE, lw=2.0)
    ax.axvline(w0, color=GRAY, ls=":", lw=1.1)
    ax.text(w0 - 0.04, 62, "you are here", color=GRAY, fontsize=8.5, ha="right")
    ax.set_ylim(-3, 75)
    ax.set_xlim(0.2, 1.6)
    ax.set_xlabel(r"$w$   (all six factors equal)")
    ax.set_ylabel(r"$J(w)$")
    ax.set_title(r"$J = \frac{1}{2}(w^6 - 1)^2$, on linear axes", fontsize=9.5, color=INK,
                 pad=6)
    tidy(ax)

    ax = axes[1]
    ww = np.linspace(-5.6, 2.1, 1400)
    ax.plot(ww, np.log10(f(ww) + 1e-12), color=LIGHT, lw=1.8, zorder=1)
    ax.plot([w0], [np.log10(f(w0))], "o", color=INK, ms=6, zorder=4)
    ax.annotate("", xy=(w_plain, np.log10(f(w_plain))), xytext=(w0, np.log10(f(w0))),
                arrowprops=dict(arrowstyle="->", color=PINK, lw=1.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(w_plain + 0.15, np.log10(f(w_plain)) + 0.15,
            f"one unclipped step:\n$J$ rises {f(w_plain) / f(w0):.0e}-fold", color=PINK,
            fontsize=8.5, ha="left")
    ax.plot([w_clip], [np.log10(f(w_clip))], "o", color=GREEN, ms=6, zorder=4)
    ax.text(w0 - 0.5, np.log10(f(w0)) - 2.6, "clipped at 1.0:\nstays on the cliff",
            color=GREEN, fontsize=8.5, ha="right")
    ax.set_ylim(-5, 11)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$\log_{10} J(w)$")
    tidy(ax)

    print(f"    at w={w0}: J={f(w0):.3f}, gradient={g0:.2f}")
    print(f"    unclipped step (lr={lr}) lands at w={w_plain:.3f}, J={f(w_plain):.3f} "
          f"— {f(w_plain) / f(w0):.1f}x worse")
    print(f"    clipped step lands at w={w_clip:.3f}, J={f(w_clip):.3f}")
    print(f"    gradient at w=1.9 is {abs(df(1.9)) / abs(df(1.1)):.1f}x the gradient at w=1.1 "
          f"({abs(df(1.9)):.1f} vs {abs(df(1.1)):.3f})")
    fig.tight_layout()
    save(fig, "05-cliff")


# ------------------------------------------------------------- 6. Nesterov

def fig_nesterov():
    """Momentum versus Nesterov momentum on an ill-conditioned quadratic."""
    lam = np.array([1.0, 22.0])

    def grad(p):
        return lam * p

    def run(nesterov, steps=120, lr=0.03, alpha=0.9):
        p, v = np.array([-9.0, 3.2]), np.zeros(2)
        path = [p.copy()]
        for _ in range(steps):
            g = grad(p + alpha * v) if nesterov else grad(p)
            v = alpha * v - lr * g
            p = p + v
            path.append(p.copy())
        return np.array(path)

    pm, pn = run(False), run(True)

    x = np.linspace(-11, 6, 300)
    y = np.linspace(-4.2, 4.2, 300)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (lam[0] * X ** 2 + lam[1] * Y ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.contour(X, Y, Z, levels=np.logspace(-0.5, 2.2, 16), colors=LIGHT, linewidths=0.7)
    ax.plot(pm[:, 0], pm[:, 1], "-o", color=ORANGE, ms=2.2, lw=1.2, label="momentum")
    ax.plot(pn[:, 0], pn[:, 1], "-o", color=BLUE, ms=2.2, lw=1.2, label="Nesterov")
    ax.plot([0], [0], "*", color=INK, ms=11)
    ax.set_xlabel(r"$\theta_1$ (flat)")
    ax.set_ylabel(r"$\theta_2$ (steep)")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax, grid=False)

    ax = axes[1]
    ax.semilogy(np.linalg.norm(pm, axis=1), color=ORANGE, lw=1.9, label="momentum")
    ax.semilogy(np.linalg.norm(pn, axis=1), color=BLUE, lw=1.9, label="Nesterov")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta - \theta^*\|$")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    dm, dn = np.linalg.norm(pm[-1]), np.linalg.norm(pn[-1])
    print(f"    after 120 steps: momentum {dm:.2e} from the optimum, Nesterov {dn:.2e} "
          f"({dm / dn:.1f}x closer)")
    def reach(path, tol=0.01):
        d = np.linalg.norm(path, axis=1)
        i = np.where(d < tol)[0]
        return int(i[0]) if len(i) else None
    print(f"    iterations to get within 0.01 of the optimum: "
          f"momentum {reach(pm)}, Nesterov {reach(pn)}")
    print(f"    both use lr=0.03, alpha=0.9; curvature ratio {lam[1] / lam[0]:.0f}:1, "
          f"lr*lambda_max = {0.03 * lam[1]:.2f} (inside the stability limit)")
    fig.tight_layout()
    save(fig, "06-nesterov")


# ------------------------------------------------------- 7. initialisation scale

def fig_init_scale():
    """Glorot's compromise, measured: forward variance vs backward variance."""
    widths = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    depth = len(widths) - 1
    r = np.random.default_rng(1)
    batch = 512

    def chain(rule):
        Ws = []
        for m, n in zip(widths[:-1], widths[1:]):
            if rule == "fan_in":
                s = 1 / np.sqrt(m)
            elif rule == "fan_out":
                s = 1 / np.sqrt(n)
            else:
                s = np.sqrt(2 / (m + n))
            Ws.append(r.normal(size=(m, n)) * s)
        a = r.normal(size=(batch, widths[0]))
        fwd = [a.var()]
        for W in Ws:
            a = a @ W
            fwd.append(a.var())
        d = r.normal(size=(batch, widths[-1]))
        bwd = [d.var()]
        for W in reversed(Ws):
            d = d @ W.T
            bwd.append(d.var())
        return np.array(fwd), np.array(bwd)[::-1]

    rules = [("fan_in", r"$1/\sqrt{m}$  (fan-in)", BLUE),
             ("fan_out", r"$1/\sqrt{n}$  (fan-out)", ORANGE),
             ("glorot", r"$\sqrt{2/(m+n)}$  (Glorot)", GREEN)]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    stats = {}
    for rule, lab, col in rules:
        fwd, bwd = chain(rule)
        axes[0].semilogy(fwd / fwd[0], color=col, lw=1.9, label=lab)
        axes[1].semilogy(bwd / bwd[-1], color=col, lw=1.9, label=lab)
        stats[rule] = (fwd[-1] / fwd[0], bwd[0] / bwd[-1])
    for ax, t in ((axes[0], "forward: activation variance"),
                  (axes[1], "backward: gradient variance")):
        ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
        ax.set_xlabel("layer")
        ax.set_ylabel("variance, relative to input")
        ax.set_title(t, fontsize=9.5, color=INK, pad=6)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)

    for rule, lab, _ in rules:
        f, b = stats[rule]
        worst = max(f, 1 / f, b, 1 / b)
        print(f"    {rule:8s}: forward x{f:.3g} over {depth} layers, backward x{b:.3g} "
              f"— worst drift {worst:.0f}x")
    print(f"    widths halve each layer ({widths[0]} down to {widths[-1]}), so m/n = 2 "
          f"and the per-layer factors are 1 / 0.5, 2 / 1, 4/3 / 2/3")
    fig.tight_layout()
    save(fig, "07-init-scale")


# ------------------------------------------------------- 8. conjugate gradients

def fig_conjugate():
    """CG terminates in n steps — until rounding error destroys the property."""
    n = 12

    def build(kappa, seed=2):
        r = np.random.default_rng(seed)
        Q = np.linalg.qr(r.normal(size=(n, n)))[0]
        A = Q @ np.diag(np.logspace(0, np.log10(kappa), n)) @ Q.T
        return A, r.normal(size=n)

    def steepest(A, b, iters):
        x, out = np.zeros(n), [np.linalg.norm(b)]
        for _ in range(iters):
            g = A @ x - b
            x = x - (g @ g) / (g @ A @ g) * g        # exact line search
            out.append(np.linalg.norm(b - A @ x))
        return np.array(out)

    def cg(A, b, iters):
        x = np.zeros(n)
        rr = b - A @ x
        p = rr.copy()
        out = [np.linalg.norm(rr)]
        for _ in range(iters):
            Ap = A @ p
            al = (rr @ rr) / (p @ Ap)
            x = x + al * p
            rn = rr - al * Ap
            p = rn + (rn @ rn) / (rr @ rr) * p
            rr = rn
            out.append(np.linalg.norm(b - A @ x))
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    res = {}
    for ax, kappa in zip(axes, (10, 1000)):
        A, b = build(kappa)
        sd, c = steepest(A, b, 2 * n), cg(A, b, 2 * n)
        res[kappa] = (sd, c)
        ax.semilogy(np.maximum(sd, 1e-18), color=ORANGE, lw=1.9, label="steepest descent")
        ax.semilogy(np.maximum(c, 1e-18), color=BLUE, lw=1.9, label="conjugate gradients")
        ax.axvline(n, color=GRAY, ls=":", lw=1.2)
        ax.text(n + 0.5, 1e-4, f"$n = {n}$", color=GRAY, fontsize=9)
        ax.set_ylim(1e-17, 1e2)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\|A\mathbf{x} - \mathbf{b}\|$")
        ax.set_title(rf"$\kappa = {kappa}$", fontsize=9.5, color=INK, pad=6)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)

    for kappa in (10, 1000):
        sd, c = res[kappa]
        below = np.where(c < 1e-10)[0]
        print(f"    kappa={kappa}: CG residual at n={n} is {c[n]:.2e} "
              f"(from {c[0]:.2f}); first below 1e-10 at step "
              f"{below[0] if len(below) else '>2n'}")
        print(f"      steepest descent after 2n={2 * n} steps: {sd[-1]:.2e}")
    fig.tight_layout()
    save(fig, "08-conjugate")


# --------------------------------------------------------- 9. batch normalization

def fig_batchnorm():
    """How much wider is the usable learning rate range with batch norm?"""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    Xn, yn = spiral(700, jitter=0.30, seed=1)
    X = torch.tensor(Xn, dtype=torch.float32)
    y = torch.tensor(yn, dtype=torch.long)

    def he(lin):
        nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
        return lin

    def build(bn, depth=8, width=48):
        layers, d = [], 2
        for _ in range(depth):
            layers.append(he(nn.Linear(d, width, bias=not bn)))
            if bn:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            d = width
        layers.append(he(nn.Linear(d, 3)))
        return nn.Sequential(*layers)

    lrs = np.logspace(-3.5, 0.7, 20)
    res = {}
    for bn in (False, True):
        finals = []
        for lr in lrs:
            torch.manual_seed(0)
            net = build(bn)
            opt = torch.optim.SGD(net.parameters(), lr=float(lr))
            best = float("inf")
            for ep in range(80):
                perm = torch.randperm(len(X))
                for s in range(0, len(X), 128):
                    b = perm[s:s + 128]
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(net(X[b]), y[b])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    full = float(nn.functional.cross_entropy(net(X), y))
                net.train()
                if np.isfinite(full):
                    best = min(best, full)
            finals.append(best if np.isfinite(best) else np.nan)
        res[bn] = np.array(finals)

    thr = 0.35

    def width_of(v):
        ok = np.where((v < thr) & np.isfinite(v))[0]
        return (lrs[ok[0]], lrs[ok[-1]]) if len(ok) else (np.nan, np.nan)

    lo0, hi0 = width_of(res[False])
    lo1, hi1 = width_of(res[True])

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.semilogx(lrs, res[False], "o-", color=ORANGE, ms=4, lw=1.8, label="plain, 8 layers")
    ax.semilogx(lrs, res[True], "o-", color=BLUE, ms=4, lw=1.8, label="with batch norm")
    ax.axhline(thr, color=GRAY, ls=":", lw=1.1)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best training loss reached")
    ax.legend(fontsize=8.5, frameon=False)
    tidy(ax)

    print(f"    both arms use He initialisation, so this isolates the normalisation")
    print(f"    usable range (best loss < {thr}): plain {lo0:.4g}–{hi0:.4g} "
          f"({hi0 / lo0:.0f}x wide), batch norm {lo1:.4g}–{hi1:.4g} ({hi1 / lo1:.0f}x wide)")
    print(f"    widening factor {(hi1 / lo1) / (hi0 / lo0):.0f}x; "
          f"largest usable rate {hi1 / hi0:.0f}x higher")
    print(f"    best loss anywhere: plain {np.nanmin(res[False]):.4f}, "
          f"bn {np.nanmin(res[True]):.4f} "
          f"— normalisation widened the range, it did not lower the floor")
    fig.tight_layout()
    save(fig, "09-batchnorm")


# --------------------------------------------------------- 10. Polyak averaging

def fig_polyak():
    """Averaging the trajectory lands closer than any point on it."""
    lam = np.array([1.0, 45.0])
    rng = np.random.default_rng(0)
    p = np.array([-8.0, 3.0])
    lr, noise, steps, alpha = 0.043, 4.0, 800, 0.995
    path, run_mean, ema = [p.copy()], [p.copy()], [p.copy()]
    acc, e = p.copy(), p.copy()
    for t in range(1, steps + 1):
        p = p - lr * (lam * p + rng.normal(scale=noise, size=2))
        acc = acc + p
        e = alpha * e + (1 - alpha) * p
        path.append(p.copy())
        run_mean.append(acc / (t + 1))
        ema.append(e.copy())
    path, run_mean, ema = map(np.array, (path, run_mean, ema))

    x = np.linspace(-9.5, 5, 300)
    y = np.linspace(-1.6, 3.4, 300)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (lam[0] * X ** 2 + lam[1] * Y ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.contour(X, Y, Z, levels=np.logspace(-0.6, 2.4, 14), colors=LIGHT, linewidths=0.7)
    ax.plot(path[:, 0], path[:, 1], color=LIGHT, lw=0.9, zorder=1)
    ax.plot(path[:, 0], path[:, 1], ".", color=GRAY, ms=1.8, zorder=2)
    ax.plot(run_mean[:, 0], run_mean[:, 1], color=GREEN, lw=2.1, zorder=3,
            label="running mean")
    ax.plot([0], [0], "*", color=INK, ms=11, zorder=4)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax, grid=False)

    ax = axes[1]
    ax.semilogy(np.linalg.norm(path, axis=1), color=GRAY, lw=1.2, label="SGD iterate")
    ax.semilogy(np.linalg.norm(run_mean, axis=1), color=GREEN, lw=1.9, label="running mean")
    ax.semilogy(np.linalg.norm(ema, axis=1), color=BLUE, lw=1.9, label="EMA")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\theta - \theta^*\|$")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    tail = slice(-150, None)
    d_raw = np.linalg.norm(path[tail], axis=1).mean()
    d_run = np.linalg.norm(run_mean[tail], axis=1).mean()
    d_ema = np.linalg.norm(ema[tail], axis=1).mean()
    print(f"    mean distance to the optimum over the last 150 steps:")
    print(f"      SGD iterate {d_raw:.3f} | running mean {d_run:.3f} "
          f"({d_raw / d_run:.0f}x closer) | EMA alpha={alpha} {d_ema:.3f} "
          f"({d_raw / d_ema:.1f}x closer)")
    print(f"    the plain running mean wins because this problem is convex and "
          f"stationary; that is exactly the assumption the EMA gives up")
    fig.tight_layout()
    save(fig, "10-polyak")


# --------------------------------------------- 11. continuation and its failure

def fig_continuation():
    """Blurring a hard objective, and the case where blurring cannot help."""
    # Compute on a grid far wider than 5*sigma, then display only the middle.
    # On a +/-8 grid the kernel is nearly as wide as the array and edge padding
    # contaminates everything: the curvature of a blurred quadratic measures
    # -0.93 instead of the exact -1.
    th = np.linspace(-24, 24, 9601)
    dx = th[1] - th[0]
    view = np.abs(th) <= 8.0

    def J(t):
        return 0.10 * t ** 2 + 1.5 * np.cos(2.2 * t) + 0.6 * np.sin(0.9 * t) + 0.35 * t

    def blur(vals, sigma):
        if sigma <= 0:
            return vals.copy()
        k = np.arange(-int(5 * sigma / dx), int(5 * sigma / dx) + 1) * dx
        w = np.exp(-0.5 * (k / sigma) ** 2)
        w /= w.sum()
        return np.convolve(np.pad(vals, len(k) // 2, mode="edge"), w, mode="valid")

    def descend(v, start):
        """Strict local descent on the grid, from the nearest point to start."""
        j = int(np.argmin(np.abs(th - start)))
        while True:
            k = min((i for i in (j - 1, j + 1) if 0 <= i < len(v)), key=lambda i: v[i])
            if v[k] >= v[j]:
                return j
            j = k

    base = J(th)
    start = 5.5
    sigmas = [3.0, 1.6, 0.8, 0.0]
    cols = [LIGHT, "#93c5fd", "#3b82f6", BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    cur, path = start, []
    for s_, c in zip(sigmas, cols):
        v = blur(base, s_)
        ax.plot(th[view], v[view], color=c, lw=2.0 if s_ == 0 else 1.3,
                label=(r"true $J$" if s_ == 0 else f"$\\sigma = {s_}$"))
        j = descend(v, cur)
        cur = th[j]
        path.append(cur)
        ax.plot([cur], [v[j]], "o", color=c, ms=6, mec="white", mew=1.0, zorder=5)
    stuck = descend(base, start)
    ax.plot([th[stuck]], [base[stuck]], "X", color=PINK, ms=9, zorder=6)
    ax.plot([start], [base[int(np.argmin(np.abs(th - start)))]], "o", color=INK, ms=5,
            zorder=6)
    ax.text(th[stuck] + 0.25, base[stuck] + 0.5, "descent on $J$\nstops here", color=PINK,
            fontsize=8)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$J^{(i)}(\theta)$")
    ax.set_ylim(-4.5, 9)
    ax.set_xlim(-8, 8)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    tidy(ax)

    ax = axes[1]
    bad = -0.5 * th ** 2
    for s_, c in zip([3.0, 1.6, 0.0], [LIGHT, "#93c5fd", PINK]):
        ax.plot(th, blur(bad, s_), color=c, lw=2.0 if s_ == 0 else 1.3,
                label=(r"$J = -\frac{1}{2}\theta^\top\theta$" if s_ == 0 else f"$\\sigma = {s_}$"))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$J^{(i)}(\theta)$")
    ax.set_title("blurring that never convexifies", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=7.5, frameon=False, loc="lower center")
    tidy(ax)

    g = int(np.argmin(base))
    print(f"    global minimum at theta = {th[g]:.3f}, J = {base[g]:.3f}")
    print(f"    plain local descent from theta = {start} stops at "
          f"theta = {th[stuck]:.3f}, J = {base[stuck]:.3f}")
    print("    continuation path: " +
          " -> ".join(f"sigma={s_}: {c:.2f}" for s_, c in zip(sigmas, path)) +
          f"  (J = {base[descend(base, path[-1])]:.3f})")
    b = blur(bad, 3.0)
    c2 = np.gradient(np.gradient(b, dx), dx)[len(b) // 2]
    shift = (b - bad)[np.abs(th) < 6]
    print(f"    blurring -1/2*theta^T theta: curvature stays {c2:.5f} (exactly -1) and the "
          f"whole curve drops by {shift.mean():.4f} +/- {shift.std():.1e} "
          f"(exactly -sigma^2/2 = {-0.5 * 3.0 ** 2:.4f})")
    print("      a Gaussian blur of a quadratic adds a constant and nothing else, "
          "so it never convexifies")
    fig.tight_layout()
    save(fig, "11-continuation")


if __name__ == "__main__":
    print("Part IX — optimization figures")
    fig_three_objectives()
    fig_gradient_norm()
    fig_batch_size()
    fig_symmetry()
    fig_cliff()
    fig_nesterov()
    fig_init_scale()
    fig_conjugate()
    fig_batchnorm()
    fig_polyak()
    fig_continuation()
    print("done ->", OUT)
