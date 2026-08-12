"""
Figures for Part XII — Attention and Transformers.

Every claim this part makes that can be measured, is measured here. The sqrt(d)
is derived from the measured variance of a dot product; RoPE's relative-position
property is checked to float64 rounding; the attention/feedforward crossover is
timed on this machine; the induction head is found in a model small enough to
read.

All data is synthetic and generated here.

    python3 make_figures.py                # everything
    python3 make_figures.py --cheap        # only the ones that need no training

Runtime: about a minute for --cheap, twenty for the rest.
"""
import os
import re
import sys
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
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "at"))
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


def noaxes(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tm(f, reps=5):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps * 1e3


# ------------------------------------------------------------- 1. path length

def fig_path_length():
    ns = np.array([16, 64, 256, 1024, 4096, 16384])
    attn = np.ones_like(ns)
    rnn = ns - 1
    conv = np.ceil((ns - 1) / 2)                      # stacked 3x3, stride 1
    dilated = np.ceil(np.log2((ns - 1) / 2 + 1))      # dilations 1, 2, 4, 8, ...

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for y, c, lab in ((rnn, ORANGE, "recurrence"), (conv, GREEN, "stacked $3\\times3$"),
                      (dilated, PURPLE, "dilated convolution"), (attn, BLUE, "attention")):
        ax.loglog(ns, np.maximum(y, 1), "o-", color=c, ms=4, lw=1.9, label=lab)
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("steps between the two ends")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    d = 256
    seq = np.array([64, 256, 1024, 4096])
    ax.loglog(seq, 2 * seq ** 2 * d, "o-", color=BLUE, ms=4, lw=1.9,
              label="attention, $O(n^2 d)$")
    ax.loglog(seq, 8 * seq * d ** 2, "o-", color=ORANGE, ms=4, lw=1.9,
              label="feedforward, $O(n d^2)$")
    ax.axvline(4 * d, color=GRAY, ls=":", lw=1.2)
    ax.text(4 * d * 1.1, 3e9, "$n = 4d$", fontsize=8, color=GRAY)
    ax.set_xlabel("sequence length $n$   ($d = 256$)")
    ax.set_ylabel("multiply-accumulates")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    path length at n={ns[-1]}: attention 1, recurrence {rnn[-1]}, "
          f"stacked conv {conv[-1]:.0f}, dilated {dilated[-1]:.0f}")
    print(f"    the two costs cross at n = 4d — {4*d} for d = {d}")
    fig.tight_layout()
    save(fig, "02-path-length")


# ------------------------------------------------------------------ 2. sqrt d

def fig_sqrt_d():
    torch.manual_seed(0)
    n = 64
    ds = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    raw_std, raw_ent, sc_ent, var_ratio = [], [], [], []

    def ent(s):
        p = F.softmax(s, dim=-1)
        return float(-(p * (p + 1e-30).log()).sum(-1).mean())

    for d in ds:
        q, k = torch.randn(n, d), torch.randn(n, d)
        raw = q @ k.T
        raw_std.append(float(raw.std()))
        raw_ent.append(ent(raw))
        sc_ent.append(ent(raw / d ** 0.5))
        qq = torch.randn(max(4000 // d, 200), d)
        kk = torch.randn_like(qq)
        var_ratio.append(float((qq * kk).sum(-1).var()) / d)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(ds, raw_std, "o-", color=ORANGE, ms=4, lw=1.9, label="measured std of $q\\cdot k$")
    ax.loglog(ds, np.sqrt(ds), "--", color=GRAY, lw=1.3, label="$\\sqrt{d}$")
    ax.set_xlabel("head dimension $d$")
    ax.set_ylabel("standard deviation of a logit")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogx(ds, raw_ent, "o-", color=ORANGE, ms=4, lw=1.9, label="unscaled")
    ax.semilogx(ds, sc_ent, "o-", color=BLUE, ms=4, lw=1.9, label="divided by $\\sqrt{d}$")
    ax.axhline(np.log(n), color=GRAY, ls=":", lw=1.2)
    ax.text(5, np.log(n) * 0.955, "uniform attention, $\\ln 64$", fontsize=7.5, color=GRAY)
    ax.set_xlabel("head dimension $d$")
    ax.set_ylabel("entropy of the attention weights")
    ax.set_ylim(0, np.log(n) * 1.1)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    Var[q.k] / d over d = {ds[0]}..{ds[-1]}: "
          f"{min(var_ratio):.3f} to {max(var_ratio):.3f} — the variance IS d, so "
          f"dividing by sqrt(d) is what makes the logits O(1)")
    print(f"    unscaled entropy falls {raw_ent[0]:.2f} -> {raw_ent[-1]:.3f} "
          f"(uniform would be {np.log(n):.3f}); scaled stays {min(sc_ent):.2f}-{max(sc_ent):.2f}")
    fig.tight_layout()
    save(fig, "01-sqrt-d")


# --------------------------------------------------- 3. permutation and position

def fig_permutation():
    torch.manual_seed(0)
    d, n = 32, 14
    Wq, Wk, Wv = (torch.randn(d, d) / d ** 0.5 for _ in range(3))

    def attn(X):
        q, k, v = X @ Wq, X @ Wk, X @ Wv
        return F.softmax(q @ k.T / d ** 0.5, dim=-1) @ v

    X = torch.randn(n, d)
    errs_plain, errs_pos = [], []
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)

    for _ in range(12):
        p = torch.randperm(n)
        a = attn(X[p])
        b = attn(X)[p]
        errs_plain.append(float((a - b).abs().max()))
        a2 = attn(X[p] + pe)
        b2 = (attn(X + pe))[p]
        errs_pos.append(float((a2 - b2).abs().max()))
    scale = float(attn(X).abs().max())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(np.array(errs_plain) / scale, "o", color=BLUE, ms=5, label="no positional encoding")
    ax.plot(np.array(errs_pos) / scale, "o", color=PINK, ms=5, label="with positional encoding")
    ax.set_yscale("log")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=1.1)
    ax.text(0.2, 1.4e-6, "float32 precision", fontsize=7.5, color=GRAY)
    ax.set_xlabel("random permutation")
    ax.set_ylabel("relative $\\max|f(PX) - Pf(X)|$")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    im = ax.imshow(pe.T.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("position")
    ax.set_ylabel("dimension")
    ax.set_title("sinusoidal encoding", fontsize=9.5, color=INK, pad=6)
    ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    print(f"    without positions: relative error {max(errs_plain)/scale:.1e} — exactly "
          f"permutation equivariant, which here is a DEFECT")
    print(f"    with positions:    relative error {min(errs_pos)/scale:.2f} to "
          f"{max(errs_pos)/scale:.2f} — the symmetry is broken, as intended")
    fig.tight_layout()
    save(fig, "03-permutation")


# --------------------------------------------------------------------- 4. RoPE

def _rope(x, pos, base=10000.0):
    d = x.shape[-1]
    half = d // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float64) / half)
    ang = pos.unsqueeze(-1).double() * inv
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half].double(), x[..., half:].double()
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def fig_rope():
    torch.manual_seed(0)
    d = 64
    q0 = torch.randn(d, dtype=torch.float64)
    k0 = torch.randn(d, dtype=torch.float64)

    bases = [0, 100, 1000, 10000]
    offsets = np.arange(-20, 21)
    curves = []
    for b in bases:
        row = []
        for off in offsets:
            m, n = b + 30 + off, b + 30
            qm = _rope(q0.unsqueeze(0), torch.tensor([float(m)]))[0]
            kn = _rope(k0.unsqueeze(0), torch.tensor([float(n)]))[0]
            row.append(float((qm * kn).sum()))
        curves.append(row)
    curves = np.array(curves)
    spread = float(np.abs(curves - curves[0]).max())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for b, row, c in zip(bases, curves, (ORANGE, GREEN, PURPLE, BLUE)):
        ax.plot(offsets, row, lw=2.2 if b == 0 else 1.2,
                color=c, alpha=1.0 if b == 0 else 0.75,
                label=f"anchored at position {b + 30}")
    ax.set_xlabel("offset $m - n$")
    ax.set_ylabel("$q_m \\cdot k_n$")
    ax.legend(fontsize=7.5, frameon=False)
    tidy(ax)

    ax = axes[1]
    dev = np.abs(curves - curves[0]).max(axis=1)
    ax.semilogy(range(len(bases)), np.maximum(dev, 1e-18), "o", color=PINK, ms=7)
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([str(b + 30) for b in bases], fontsize=8.5)
    ax.set_xlabel("absolute position of the anchor")
    ax.set_ylabel("deviation from the curve at position 30")
    ax.set_ylim(1e-18, 1e-2)
    ax.axhline(2.2e-16, color=GRAY, ls=":", lw=1.1)
    ax.text(0.05, 3e-16, "float64 precision", fontsize=7.5, color=GRAY)
    tidy(ax)

    print(f"    the four curves coincide to {spread:.1e} — q.k depends on m-n ALONE, "
          f"not on where in the sequence the pair sits")
    fig.tight_layout()
    save(fig, "04-rope")


# ------------------------------------------------------------ 5. the cost, timed

def fig_cost():
    torch.manual_seed(0)
    d = 256
    Wq, Wk, Wv, Wo = (torch.randn(d, d) / d ** 0.5 for _ in range(4))
    W1 = torch.randn(d, 4 * d) / d ** 0.5
    W2 = torch.randn(4 * d, d) / (4 * d) ** 0.5
    ns = [64, 128, 256, 512, 1024, 2048]
    ta, tf = [], []
    for n in ns:
        X = torch.randn(n, d)

        def A():
            q, k, v = X @ Wq, X @ Wk, X @ Wv
            return (F.softmax(q @ k.T / d ** 0.5, -1) @ v) @ Wo

        def Ff():
            return F.gelu(X @ W1) @ W2
        ta.append(tm(A))
        tf.append(tm(Ff))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(ns, ta, "o-", color=BLUE, ms=4, lw=1.9, label="attention")
    ax.loglog(ns, tf, "o-", color=ORANGE, ms=4, lw=1.9, label="feedforward")
    ax.axvline(4 * d, color=GRAY, ls=":", lw=1.2)
    ax.text(4 * d * 1.08, min(ta) * 1.4, "$n = 4d$", fontsize=8, color=GRAY)
    ax.set_xlabel("sequence length $n$   ($d = 256$)")
    ax.set_ylabel("milliseconds")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    seq = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ax.loglog(seq, seq ** 2 * 4 / 1e9, "o-", color=PINK, ms=4, lw=1.9)
    ax.axhline(40, color=GRAY, ls=":", lw=1.2)
    ax.text(600, 48, "a 40 GB accelerator", fontsize=7.5, color=GRAY)
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("one score matrix (GB, float32)")
    tidy(ax)

    cross = next((n for n, a, f in zip(ns, ta, tf) if a > f), None)
    print(f"    measured crossover between {ns[ns.index(cross)-1]} and {cross}; "
          f"predicted at n = 4d = {4*d}")
    for n, a, f in zip(ns, ta, tf):
        print(f"      n={n:5d}  attention {a:7.2f} ms   feedforward {f:7.2f} ms   "
              f"ratio {a/f:.2f}")
    print(f"    a single 65,536-token score matrix is "
          f"{65536**2*4/1e9:.1f} GB — before any of the other layers")
    fig.tight_layout()
    save(fig, "05-cost")


# ------------------------------------------------------- 6. where the model is

def fig_breakdown():
    ds = [512, 768, 1024, 2048, 4096, 8192]
    attn = [4 * d * d for d in ds]
    ffn = [8 * d * d for d in ds]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    idx = np.arange(len(ds))
    ax.bar(idx, attn, 0.6, color=BLUE, label="attention  ($4d^2$)")
    ax.bar(idx, ffn, 0.6, bottom=attn, color=ORANGE, label="feedforward  ($8d^2$)")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(d) for d in ds], fontsize=8, rotation=30)
    ax.set_xlabel("$d_{\\rm model}$")
    ax.set_ylabel("parameters per block")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    frac = [a / (a + f) for a, f in zip(attn, ffn)]
    ax.plot(idx, frac, "o-", color=BLUE, ms=5, lw=1.9, label="attention")
    ax.plot(idx, [1 - x for x in frac], "o-", color=ORANGE, ms=5, lw=1.9,
            label="feedforward")
    ax.set_ylim(0, 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(d) for d in ds], fontsize=8, rotation=30)
    ax.set_xlabel("$d_{\\rm model}$")
    ax.set_ylabel("share of the block's parameters")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    attention holds {100*frac[0]:.1f}% of a block's parameters at every width; "
          f"the feedforward layer holds {100*(1-frac[0]):.1f}%")
    print("    the ratio is exactly 1:2 and does not depend on d — the part of the")
    print("    architecture the part is named after is the minority of it")
    fig.tight_layout()
    save(fig, "06-breakdown")


# ----------------------------------------------------------------- 7. KV cache

def fig_kv_cache():
    torch.manual_seed(0)
    d, n_layers = 4096, 32
    lens = np.array([1024, 4096, 16384, 65536, 131072])
    gb = 2 * n_layers * lens * d * 2 / 1e9

    # measured: generating with and without a cache, small model
    dh, T = 128, 256
    Wq, Wk, Wv = (torch.randn(dh, dh) / dh ** 0.5 for _ in range(3))

    def gen_no_cache(T):
        # the honest baseline: no cache means no memory of any earlier computation, so
        # every step reruns the FULL forward pass over the whole prefix so far — every
        # query against every key, not just the one new row — and keeps only the last
        # output. This is what makes the per-step cost O(t^2 d) rather than O(td), and
        # the total O(n^3) rather than O(n^2), matching Sec.28's definition.
        X = torch.randn(1, dh)
        for _ in range(T - 1):
            q, k, v = X @ Wq, X @ Wk, X @ Wv          # recompute EVERY query, key, value
            out_all = F.softmax(q @ k.T / dh ** 0.5, -1) @ v   # the full t x t attention
            X = torch.cat([X, out_all[-1:]], 0)
        return X

    def gen_cache(T):
        x = torch.randn(1, dh)
        K = x @ Wk
        V = x @ Wv
        for _ in range(T - 1):
            q = x @ Wq
            out = F.softmax(q @ K.T / dh ** 0.5, -1) @ V
            K = torch.cat([K, out @ Wk], 0)            # append only the new one
            V = torch.cat([V, out @ Wv], 0)
            x = out
        return x

    Ts = [32, 64, 128, 256, 512]
    t_no, t_yes = [], []
    for T_ in Ts:
        t_no.append(tm(lambda T_=T_: gen_no_cache(T_), reps=3))
        t_yes.append(tm(lambda T_=T_: gen_cache(T_), reps=3))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(lens, gb, "o-", color=PINK, ms=5, lw=1.9)
    ax.axhline(14, color=GRAY, ls=":", lw=1.2)
    ax.text(1200, 16, "the weights of a 7B model in bf16", fontsize=7.5, color=GRAY)
    ax.set_xlabel("tokens in context")
    ax.set_ylabel("KV cache (GB, bf16)")
    ax.set_title("32 layers, $d = 4096$", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    ax = axes[1]
    ax.loglog(Ts, t_no, "o-", color=ORANGE, ms=4, lw=1.9, label="recompute every step")
    ax.loglog(Ts, t_yes, "o-", color=BLUE, ms=4, lw=1.9, label="cached")
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("milliseconds")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    i = int(np.where(lens == 131072)[0][0])
    print(f"    KV cache at 131,072 tokens: {gb[i]:.1f} GB — around five times the "
          f"weights of the 7B model holding it")
    print(f"    generating {Ts[-1]} tokens: {t_no[-1]:.0f} ms recomputing against "
          f"{t_yes[-1]:.0f} ms cached — {t_no[-1]/t_yes[-1]:.1f}x")
    fig.tight_layout()
    save(fig, "09-kv-cache")


CHEAP = [fig_path_length, fig_sqrt_d, fig_permutation, fig_rope,
         fig_cost, fig_breakdown, fig_kv_cache]

if __name__ == "__main__":
    print("Part XII — attention figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE           # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
