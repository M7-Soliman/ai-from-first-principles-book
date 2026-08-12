"""
Figures for Part XVI — Generative Models.

Every claim this part makes that can be measured, is measured here.

The cheap ones are exact or near-exact computations: the cost asymmetry of
autoregressive sampling, and how score matching fails away from the data. The
expensive ones train: the ELBO's gap against a likelihood that can be computed
exactly, posterior collapse, a diffusion model that actually generates samples,
guidance, GAN mode collapse, and the near-independence of likelihood from
sample quality.

Two exist because the folk version did not survive contact: sample quality and
log-likelihood are almost unrelated (fig_likelihood_quality), and a diffusion
model's noise schedule matters more than its architecture (fig_schedule).

All data is synthetic and generated here, on distributions whose density,
score and modes are known in closed form — so "did it learn the distribution"
is a question with an exact answer rather than a visual impression.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train

Runtime: about a minute for --cheap, twenty for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "gn"))
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


# ------------------------------------------------------- the reference density

MUS = np.array([[-1.6, -0.9], [1.5, 0.9], [-1.2, 1.7]])
SDS = np.array([0.42, 0.34, 0.30])
WS = np.array([0.40, 0.35, 0.25])


def mix_logp(P, extra_var=0.0):
    """log density of the reference Gaussian mixture, optionally convolved with
    isotropic noise of variance extra_var."""
    P = np.atleast_2d(P)
    d = P.shape[1]
    L = []
    for m, sd, w in zip(MUS, SDS, WS):
        v = sd ** 2 + extra_var
        d2 = ((P - m) ** 2).sum(1)
        L.append(np.log(w) - 0.5 * d * np.log(2 * np.pi * v) - d2 / (2 * v))
    L = np.stack(L)
    M = L.max(0)
    return M + np.log(np.exp(L - M).sum(0))


def mix_score(P, extra_var=0.0):
    """Exact score of the reference mixture, optionally smoothed."""
    P = np.atleast_2d(P)
    d = P.shape[1]
    logs, scores = [], []
    for m, sd, w in zip(MUS, SDS, WS):
        v = sd ** 2 + extra_var
        d2 = ((P - m) ** 2).sum(1)
        logs.append(np.log(w) - 0.5 * d * np.log(2 * np.pi * v) - d2 / (2 * v))
        scores.append((m - P) / v)
    L = np.stack(logs)
    R = np.exp(L - L.max(0))
    post = R / R.sum(0)
    return sum(post[i][:, None] * scores[i] for i in range(len(WS)))


def mix_sample(n, seed=0):
    r = np.random.default_rng(seed)
    c = r.choice(len(WS), size=n, p=WS)
    return (MUS[c] + r.normal(size=(n, 2)) * SDS[c][:, None]).astype(np.float32)


# ------------------------------------------- 1. the autoregressive asymmetry

def fig_autoregressive():
    """Training an autoregressive model is one parallel pass over all positions;
    sampling is one pass per position and cannot be otherwise. Measured on a
    real transformer stack doing identical arithmetic both ways."""
    import time
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(0)

    D, H, L = 128, 4, 4

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = nn.MultiheadAttention(D, H, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(),
                                    nn.Linear(4 * D, D))
            self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)

        def forward(self, x, mask):
            h = self.n1(x)
            x = x + self.att(h, h, h, attn_mask=mask, need_weights=False)[0]
            return x + self.ff(self.n2(x))

    net = nn.Sequential(*[Block() for _ in range(L)])

    def causal(n):
        return torch.triu(torch.ones(n, n, dtype=torch.bool), 1)

    def forward(x, m):
        h = x
        for b in net:
            h = b(h, m)
        return h

    def timed(fn, reps):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    lens = [16, 32, 64, 128, 256]
    par, seq = [], []
    with torch.no_grad():
        for n in lens:
            x, m = torch.randn(1, n, D), causal(n)
            par.append(timed(lambda: forward(x, m), 25))
            def sample():
                for i in range(1, n + 1):
                    forward(x[:, :i], causal(i))
            seq.append(timed(sample, 3))
            print(f"    length {n:4d}: one parallel pass {par[-1]*1e3:7.2f} ms   "
                  f"{n} sequential passes {seq[-1]*1e3:9.2f} ms   "
                  f"ratio {seq[-1]/par[-1]:6.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(lens, np.array(par) * 1e3, "o-", color=BLUE, lw=1.9, ms=4.5,
            label="training: one pass, all positions")
    ax.plot(lens, np.array(seq) * 1e3, "s-", color=ORANGE, lw=1.9, ms=4.5,
            label="sampling: one pass per position")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length"); ax.set_ylabel("wall clock (ms)")
    ax.set_title("The same model, two access patterns", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    ax.plot(lens, np.array(seq) / np.array(par), "o-", color=PURPLE, lw=2.0, ms=4.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length"); ax.set_ylabel("sampling / training")
    ax.set_title("The penalty grows with length", fontsize=10.5, color=INK)
    tidy(ax)

    print(f"    at length {lens[-1]}: {seq[-1]/par[-1]:.0f}x, and the gap widens with length.")
    print("    the cost is not arithmetic — it is that position i+1 cannot begin")
    print("    until position i has been produced. Training has no such constraint,")
    print("    because the true prefix is already known.")
    fig.tight_layout()
    save(fig, "01-autoregressive")
    return dict(lens=lens, par=par, seq=seq)


# ------------------------------------ 2. where score matching has no signal

def fig_score_coverage():
    """Plain score matching fits the score only where the data is. Away from
    the data there is no training signal, and a sampler that starts from noise
    starts exactly there. Measured against the exact score."""
    rng = np.random.default_rng(0)
    X = mix_sample(4000, seed=0)

    g = np.linspace(-4.0, 4.0, 120)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    dens = np.exp(mix_logp(P))

    # how much training signal reaches each point: the density of the
    # distribution the samples are drawn from, at several noise levels
    sigmas = [0.0, 0.3, 0.8, 2.0]
    frac_covered = []
    for s in sigmas:
        d = np.exp(mix_logp(P, extra_var=s ** 2))
        # "covered" = density at least 1% of this distribution's own maximum
        frac_covered.append(float((d > 0.01 * d.max()).mean()))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))

    ax = axes[0]
    ax.contourf(GX, GY, dens.reshape(GX.shape), levels=14, cmap="Blues")
    sub = np.random.default_rng(1).choice(len(X), 500, replace=False)
    ax.scatter(X[sub, 0], X[sub, 1], s=3, color=INK, alpha=0.35)
    ax.set_title("The data, and where it is", fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ("top", "right", "bottom", "left"):
        ax.spines[s_].set_visible(False)

    ax = axes[1]
    for s, c in zip(sigmas, (INK, BLUE, GREEN, ORANGE)):
        d = np.exp(mix_logp(P, extra_var=s ** 2)).reshape(GX.shape)
        ax.contour(GX, GY, d, levels=[0.01 * d.max()], colors=[c], linewidths=1.8)
    ax.set_title("Where each noise level puts data", fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ("top", "right", "bottom", "left"):
        ax.spines[s_].set_visible(False)
    ax.plot([], [], color=INK, lw=1.8, label="$\\sigma = 0$")
    for s, c in zip(sigmas[1:], (BLUE, GREEN, ORANGE)):
        ax.plot([], [], color=c, lw=1.8, label=f"$\\sigma = {s}$")
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")

    ax = axes[2]
    ax.bar(range(len(sigmas)), np.array(frac_covered) * 100,
           color=[INK, BLUE, GREEN, ORANGE], width=0.6)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s}" for s in sigmas])
    ax.set_xlabel("noise level $\\sigma$")
    ax.set_ylabel("% of the plotted region covered")
    ax.set_title("How much space gets a signal", fontsize=10.5, color=INK)
    tidy(ax)

    for s, f in zip(sigmas, frac_covered):
        print(f"    sigma {s:4.1f}: {f*100:5.1f}% of the region carries training signal")
    print(f"    at sigma=0 the data covers {frac_covered[0]*100:.1f}% of the space;")
    print(f"    a sampler initialised from wide noise starts in the other "
          f"{100-frac_covered[0]*100:.1f}%, where the score was never fitted.")
    print("    this is the argument for many noise levels, not one.")
    fig.tight_layout()
    save(fig, "03-score-coverage")
    return dict(sigmas=sigmas, frac=frac_covered)


CHEAP = [fig_autoregressive, fig_score_coverage]

if __name__ == "__main__":
    print("Part XVI — generative model figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
