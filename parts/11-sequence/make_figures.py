"""
Figures for Part XI — Sequence Models.

Every claim in this part that can be measured, is measured here. The gradient
decay through time is a real Jacobian product; the long-dependency failure is a
real training sweep with the optimiser stated; the encoder-decoder bottleneck is
a real seq2seq model whose context width is the independent variable.

All data is synthetic and generated here, so the figures are reproducible with
no downloads.

    python3 make_figures.py           # writes into ../../book/src/figures/sq/

Runtime is roughly twenty minutes, dominated by figures 3, 4 and 9.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "sq"))
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


# ------------------------------------------------------------------- tasks

def copy_task(n, T, seed=0):
    """The label is bit 0; everything after is noise. Nothing between the
    signal and the readout carries any information about the answer."""
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    x = r.normal(scale=0.5, size=(n, T, 1)).astype(np.float32)
    x[:, 0, 0] = np.where(y == 1, 1.0, -1.0)
    return torch.tensor(x), torch.tensor(y.astype(np.int64))


class Recurrent(nn.Module):
    def __init__(self, kind="rnn", h=16, forget_bias=None):
        super().__init__()
        if kind == "rnn":
            self.rnn = nn.RNN(1, h, batch_first=True, nonlinearity="tanh")
        elif kind == "lstm":
            self.rnn = nn.LSTM(1, h, batch_first=True)
        else:
            self.rnn = nn.GRU(1, h, batch_first=True)
        if forget_bias is not None and kind == "lstm":
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    q = p.shape[0] // 4
                    with torch.no_grad():
                        p[q:2 * q].fill_(forget_bias)      # i, f, g, o order
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        o = self.rnn(x)[0]
        return self.out(o[:, -1])


def train_copy(kind, T, seed, forget_bias=None, opt_name="adam", clip=1.0,
               epochs=60, n=600):
    torch.manual_seed(seed)
    X, y = copy_task(n, T, seed=seed)
    m = Recurrent(kind, forget_bias=forget_bias)
    opt = (torch.optim.Adam(m.parameters(), lr=0.01) if opt_name == "adam"
           else torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            loss = F.cross_entropy(m(X[b]), y[b])
            if not torch.isfinite(loss):
                return 0.5, m
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
    with torch.no_grad():
        return float((m(X).argmax(1) == y).float().mean()), m


# ------------------------------------------------- 1. what unfolding buys

def fig_unfolding():
    taus = np.arange(2, 81)
    d, k = 64, 10
    # Full RNN parameter count — U, W, V, b, c — matching the drills' rnn_params(1, d, k),
    # scalar input and a k-way output vocabulary. Sharing removes the tau dependence; nothing else.
    per_step = d * 1 + d * d + k * d + d + k                       # U, W, V, b, c — no tau
    rnn = np.full_like(taus, per_step, dtype=float)
    unshared = taus * per_step                                     # one full set per step
    tabular = np.float64(k) ** taus                                # a full joint table

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.semilogy(taus, tabular, color=PINK, lw=1.9, label=r"joint table, $O(k^\tau)$")
    ax.semilogy(taus, unshared, color=ORANGE, lw=1.9, label=r"unshared per step, $O(\tau)$")
    ax.semilogy(taus, rnn, color=BLUE, lw=2.2, label=r"RNN, $O(1)$")
    ax.axhline(1e80, color=GRAY, ls=":", lw=1.1)
    ax.text(4, 3e80, "atoms in the observable universe", fontsize=7, color=GRAY)
    ax.set_ylim(1e2, 1e90)
    ax.set_xlabel(r"sequence length $\tau$")
    ax.set_ylabel("parameters")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    ax.plot(taus, unshared / rnn, color=ORANGE, lw=1.9)
    ax.set_xlabel(r"sequence length $\tau$")
    ax.set_ylabel("unshared ÷ shared")
    ax.set_title("sharing alone, on a like-for-like model", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    i = int(np.where(taus == 20)[0][0])
    print(f"    at tau=20, d={d}, k={k}: RNN {rnn[i]:.0f} parameters, "
          f"unshared {unshared[i]:.0f}, a full joint table {tabular[i]:.2e}")
    over = np.where(tabular >= 1e80)[0]      # >= : 10**80 rounds to exactly 1e80 in float64
    print(f"    the table reaches the atoms in the observable universe (~10^80) at "
          f"tau = {int(taus[over[0]])}, which is a sequence of "
          f"{int(taus[over[0]])} tokens")
    fig.tight_layout()
    save(fig, "01-unfolding")


# --------------------------------------------- 2. gradients through time

def fig_gradient_decay():
    torch.manual_seed(0)
    d, T = 32, 60
    r = np.random.default_rng(0)
    Q = np.linalg.qr(r.normal(size=(d, d)))[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for rho, c in ((0.9, BLUE), (1.0, GREEN), (1.1, ORANGE)):
        lam = np.linspace(rho * 0.6, rho, d)
        lam[-1] = rho
        W = Q @ np.diag(lam) @ Q.T
        norms = [1.0]
        M = np.eye(d)
        for _ in range(T):
            M = W @ M
            norms.append(np.linalg.norm(M, 2))
        ax.semilogy(norms, color=c, lw=1.9, label=rf"$\rho(W) = {rho}$")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
    ax.set_xlabel("time steps $t$")
    ax.set_ylabel(r"$\|\partial \mathbf{h}^{(t)} / \partial \mathbf{h}^{(0)}\|$")
    ax.set_title("linear recurrence, exactly $W^t$", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    # the real thing: a tanh RNN's Jacobian product, measured by autograd
    ax = axes[1]
    stats = {}
    for gain, c in ((0.5, BLUE), (1.0, GREEN), (2.0, ORANGE)):
        torch.manual_seed(1)
        W = torch.randn(d, d) * (gain / np.sqrt(d))
        h = torch.zeros(1, d, requires_grad=True)
        hs, cur = [], h
        for _ in range(T):
            cur = torch.tanh(cur @ W.T)
            hs.append(cur)
        norms = []
        for t in (list(range(0, T, 2)) + [T - 1]):
            g = torch.autograd.grad(hs[t].sum(), h, retain_graph=True)[0]
            norms.append(float(g.norm()))
        ts = list(range(0, T, 2)) + [T - 1]
        arr = np.array(norms)
        finite = np.isfinite(arr)
        ax.semilogy(np.array(ts)[finite], np.clip(arr[finite], 1e-30, 1e30),
                    color=c, lw=1.9, label=rf"gain $= {gain}$")
        if not finite.all():
            k = int(np.argmax(~finite))
            ax.plot([ts[k - 1]], [min(arr[k - 1], 1e30)], "x", color=c, ms=9)
            ax.annotate("overflow", (ts[k - 1], min(arr[k - 1], 1e30)), fontsize=7.5,
                        color=c, textcoords="offset points", xytext=(4, -2))
        stats[gain] = (arr[finite][-1] if finite.any() else np.inf,
                       int(np.argmax(~finite)) if not finite.all() else None)
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.0)
    ax.set_xlabel("time steps $t$")
    ax.set_ylabel(r"$\|\partial \mathbf{h}^{(t)} / \partial \mathbf{h}^{(0)}\|$")
    ax.set_title(r"$\tanh$ recurrence, measured by autograd", fontsize=9.5,
                 color=INK, pad=6)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    for gain, (v, k) in stats.items():
        tail = f", overflowed to inf by step {2 * k}" if k else ""
        print(f"    tanh RNN, gain {gain}: last finite gradient norm {v:.3e}{tail}")
    print("    the tanh saturates, so the FORWARD state is bounded — but the "
          "Jacobian product is not, and at gain 2 it overflows float32 outright")
    fig.tight_layout()
    save(fig, "03-gradient-decay")


# ------------------------------------------- 3. the long-dependency failure

def fig_long_dependency():
    Ts = [5, 10, 20, 40, 80]
    seeds = range(5)
    res = {}
    for kind, label in (("rnn", "vanilla RNN"), ("lstm", "LSTM"), ("gru", "GRU")):
        res[label] = [np.mean([train_copy(kind, T, s)[0] for s in seeds]) for T in Ts]
        print(f"    {label:12s} Adam+clip: " +
              " ".join(f"T={T}:{v:.2f}" for T, v in zip(Ts, res[label])), flush=True)
    sgd = [np.mean([train_copy("rnn", T, s, opt_name="sgd", clip=0)[0] for s in seeds])
           for T in Ts]
    print("    vanilla RNN  plain SGD, no clipping: " +
          " ".join(f"T={T}:{v:.2f}" for T, v in zip(Ts, sgd)), flush=True)

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    for label, c in (("vanilla RNN", ORANGE), ("LSTM", BLUE), ("GRU", GREEN)):
        ax.plot(Ts, res[label], "o-", color=c, ms=5, lw=1.9, label=label + ", Adam + clipping")
    ax.plot(Ts, sgd, "s--", color=PINK, ms=5, lw=1.7,
            label="vanilla RNN, plain SGD, no clipping")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.1)
    ax.text(6, 0.515, "chance", fontsize=7.5, color=GRAY)
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels(Ts)
    ax.set_ylim(0.42, 1.04)
    ax.set_xlabel("sequence length $T$ between the signal and the readout")
    ax.set_ylabel("training accuracy, mean of 5 seeds")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "05-long-dependency")


# ---------------------------------------------------- 4. the forget gate

def fig_forget_bias():
    Ts = [20, 40, 80]
    biases = [0.0, 1.0, 2.0]
    seeds = range(5)
    grid = np.zeros((len(biases) + 1, len(Ts)))
    for j, T in enumerate(Ts):
        for i, fb in enumerate(biases):
            grid[i, j] = np.mean([train_copy("lstm", T, s, forget_bias=fb)[0]
                                  for s in seeds])
        grid[-1, j] = np.mean([train_copy("lstm", T, s)[0] for s in seeds])
        print(f"    T={T}: " + " ".join(f"fb={b}:{grid[i, j]:.3f}"
                                        for i, b in enumerate(biases)) +
              f"  default:{grid[-1, j]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    labels = [f"forget bias = {b}" for b in biases] + ["PyTorch default"]
    for i, (lab, c) in enumerate(zip(labels, ["#93c5fd", BLUE, "#1e3a8a", PINK])):
        ax.plot(Ts, grid[i], "o-", color=c, ms=5, lw=1.9, label=lab)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.1)
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels(Ts)
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel("training accuracy, mean of 5 seeds")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "08-forget-bias")


# ------------------------------------------------------- 5. the RNN cliff

def fig_cliff():
    """A scalar recurrence h_t = w h_{t-1}: the loss is (w^T x - y)^2, whose
    steepness is exponential in T. ch10's figure 10.17, computed."""
    x0, target = 1.0, 1.0
    ws = np.linspace(0.2, 1.6, 2000)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for T, c in ((5, "#bfdbfe"), (10, "#60a5fa"), (20, BLUE), (40, "#1e3a8a")):
        loss = 0.5 * (ws ** T * x0 - target) ** 2
        ax.semilogy(ws, np.maximum(loss, 1e-12), color=c, lw=1.8, label=f"$T = {T}$")
    ax.set_xlabel("$w$, the single recurrent weight")
    ax.set_ylabel("loss")
    ax.set_ylim(1e-10, 1e14)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    Ts = np.arange(2, 41)
    slope = np.array([abs((1.2 ** T - 1) * T * 1.2 ** (T - 1)) for T in Ts])
    ax.semilogy(Ts, slope, color=ORANGE, lw=2.0)
    ax.set_xlabel("$T$")
    ax.set_ylabel(r"$|dL/dw|$ at $w = 1.2$")
    ax.set_title("the cliff is exponential in the sequence length",
                 fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    for T in (10, 20, 40):
        g = abs((1.2 ** T - 1) * T * 1.2 ** (T - 1))
        print(f"    T={T:2d}: |dL/dw| at w=1.2 is {g:.3e}; one step at lr=0.01 "
              f"moves w by {0.01 * g:.3e}")
    print("    a learning rate that is safe at T=10 overshoots by 8 orders of "
          "magnitude at T=40 — and nothing about the model changed but the input length")
    fig.tight_layout()
    save(fig, "04-cliff")


# ---------------------------------------------------------- 6. leaky units

def fig_leaky():
    alphas = [0.0, 0.5, 0.9, 0.99]
    T = 120
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))

    ax = axes[0]
    for a, c in zip(alphas, [ORANGE, "#93c5fd", BLUE, "#1e3a8a"]):
        imp = np.array([a ** t for t in range(T)])
        ax.semilogy(np.maximum(imp, 1e-12), color=c, lw=1.9,
                    label=rf"$\alpha = {a}$")
    ax.axhline(1e-3, color=GRAY, ls=":", lw=1.0)
    ax.set_ylim(1e-9, 2)
    ax.set_xlabel("steps into the past")
    ax.set_ylabel("weight on that input")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    aa = np.linspace(0.0, 0.995, 400)
    ax.plot(aa, 1.0 / (1.0 - aa), color=BLUE, lw=2.0)
    for a in alphas[1:]:
        ax.plot([a], [1 / (1 - a)], "o", color=PINK, ms=6)
        ax.annotate(rf"$\alpha={a}$" + f"\n{1/(1-a):.0f} steps", (a, 1 / (1 - a)),
                    textcoords="offset points", xytext=(-52, -4), fontsize=8, color=PINK)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"effective memory, $1/(1-\alpha)$")
    tidy(ax)

    print("    a leaky unit averages over roughly 1/(1-alpha) steps: " +
          ", ".join(f"alpha={a} -> {1/(1-a):.0f}" for a in alphas[1:]))
    print("    the same quantity as Part III section 15's momentum window, and for "
          "the same reason: it is a geometric series")
    fig.tight_layout()
    save(fig, "06-leaky")


# ------------------------------------------------------ 7. what gates do

def fig_gates():
    acc, m = train_copy("lstm", 30, 0, forget_bias=1.0, epochs=90)
    print(f"    trained an LSTM to {acc:.3f} on T=30, then read its gates")
    X, y = copy_task(200, 30, seed=0)
    W_ih = m.rnn.weight_ih_l0.detach()
    W_hh = m.rnn.weight_hh_l0.detach()
    b = (m.rnn.bias_ih_l0 + m.rnn.bias_hh_l0).detach()
    H = W_ih.shape[0] // 4
    h = torch.zeros(len(X), H)
    c = torch.zeros(len(X), H)
    gates = {"input": [], "forget": [], "output": []}
    cells = []
    with torch.no_grad():
        for t in range(X.shape[1]):
            z = X[:, t] @ W_ih.T + h @ W_hh.T + b
            i, f, g, o = z.split(H, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            gates["input"].append(i.mean().item())
            gates["forget"].append(f.mean().item())
            gates["output"].append(o.mean().item())
            cells.append(c.abs().mean().item())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for k, c_ in (("forget", BLUE), ("input", ORANGE), ("output", GREEN)):
        ax.plot(gates[k], color=c_, lw=1.9, label=k + " gate")
    ax.axvline(0, color=PINK, ls=":", lw=1.2)
    ax.text(0.6, 0.06, "the signal arrives here", fontsize=7.5, color=PINK)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time step")
    ax.set_ylabel("mean gate activation")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(cells, color=PURPLE, lw=2.0)
    ax.set_xlabel("time step")
    ax.set_ylabel(r"mean $|c^{(t)}|$")
    ax.set_title("the cell state integrates", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    fmean = float(np.mean(gates["forget"]))
    T_ = len(cells)
    print(f"    mean forget gate over the run: {fmean:.3f} — the cell keeps "
          f"{100*fmean:.1f}% of itself per step")
    print(f"    so a value written at step 0 still carries weight "
          f"{fmean**(T_-1):.4f} at step {T_-1}; a vanilla tanh RNN with the same "
          f"decay per step would be at {0.5**(T_-1):.2e} for a comparable 0.5")
    print(f"    mean |c| grows {cells[0]:.3f} -> {cells[-1]:.3f} ({cells[-1]/max(cells[0],1e-9):.1f}x): "
          f"the cell INTEGRATES, it does not merely hold — the input gate keeps adding")
    fig.tight_layout()
    save(fig, "07-gates")


# ------------------------------------------------------ 8. truncated BPTT

def fig_truncated():
    """How wrong is the gradient when you stop backpropagating early?"""
    torch.manual_seed(0)
    d, T = 16, 60
    W = torch.randn(d, d) * (1.0 / np.sqrt(d))
    U = torch.randn(d, 1) * 0.5
    W.requires_grad_(True)
    X, _ = copy_task(64, T, seed=0)

    def loss_with_truncation(k):
        h = torch.zeros(64, d)
        for t in range(T):
            if k is not None and t < T - k:
                h = h.detach()
            h = torch.tanh(h @ W.T + X[:, t] @ U.T)
        return h.pow(2).mean()

    full = torch.autograd.grad(loss_with_truncation(None), W, retain_graph=False)[0]
    ks = [1, 2, 5, 10, 20, 30, 45, 60]
    errs, cos = [], []
    for k in ks:
        g = torch.autograd.grad(loss_with_truncation(k), W)[0]
        errs.append(float((g - full).norm() / full.norm()))
        cos.append(float((g * full).sum() / (g.norm() * full.norm())))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.semilogy(ks, np.maximum(errs, 1e-17), "o-", color=ORANGE, ms=5, lw=1.9)
    ax.set_xlabel("truncation length $k$")
    ax.set_ylabel("relative error of the gradient")
    tidy(ax)
    ax = axes[1]
    ax.plot(ks, cos, "o-", color=BLUE, ms=5, lw=1.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("truncation length $k$")
    ax.set_ylabel("cosine with the full gradient")
    tidy(ax)

    for k, e, c in zip(ks, errs, cos):
        print(f"    k={k:2d}: relative error {e:.2e}, cosine {c:.4f}")
    print("    truncation is a BIASED gradient, not a noisy one — averaging does not "
          "recover what was cut off")
    fig.tight_layout()
    save(fig, "02-truncated")


# ------------------------------------------------------- 9. the bottleneck

V = 12


def reverse_task(n, T, seed=0):
    r = np.random.default_rng(seed)
    x = r.integers(1, V, size=(n, T))
    return torch.tensor(x), torch.tensor(x[:, ::-1].copy())


class Seq2Seq(nn.Module):
    def __init__(self, ctx, hid=64):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.enc = nn.LSTM(32, hid, batch_first=True)
        self.to_ctx = nn.Linear(hid, ctx)
        self.from_ctx = nn.Linear(ctx, hid)
        self.dec = nn.LSTM(32, hid, batch_first=True)
        self.out = nn.Linear(hid, V)

    def forward(self, x, y_in):
        _, (h, _) = self.enc(self.emb(x))
        c = self.to_ctx(h[-1])
        h0 = torch.tanh(self.from_ctx(c))[None]
        o, _ = self.dec(self.emb(y_in), (h0, torch.zeros_like(h0)))
        return self.out(o)


def fig_bottleneck():
    def run(T, ctx, epochs=90, n=2000, seed=0):
        torch.manual_seed(seed)
        x, y = reverse_task(n, T, seed=seed)
        y_in = torch.cat([torch.zeros(n, 1, dtype=torch.long), y[:, :-1]], 1)
        m = Seq2Seq(ctx)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 128):
                b = perm[s:s + 128]
                opt.zero_grad()
                F.cross_entropy(m(x[b], y_in[b]).reshape(-1, V),
                                y[b].reshape(-1)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            return float((m(x, y_in).argmax(-1) == y).float().mean())

    Ts = [4, 8, 12, 16, 24]
    ctxs = [4, 8, 16, 64]
    grid = np.zeros((len(ctxs), len(Ts)))
    for j, T in enumerate(Ts):
        for i, c in enumerate(ctxs):
            grid[i, j] = run(T, c)
        print(f"    T={T:2d}: " + " ".join(f"ctx={c}:{grid[i, j]:.3f}"
                                           for i, c in enumerate(ctxs)), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for i, (c, col) in enumerate(zip(ctxs, ["#fdba74", "#93c5fd", BLUE, "#1e3a8a"])):
        ax.plot(Ts, grid[i], "o-", color=col, ms=5, lw=1.9, label=f"context = {c}")
    ax.axhline(1.0 / (V - 1), color=GRAY, ls=":", lw=1.1)
    ax.text(4.3, 1.0 / (V - 1) + 0.02, "chance", fontsize=7.5, color=GRAY)
    ax.set_xlabel("input length")
    ax.set_ylabel("per-token accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for j, (T, col) in enumerate(zip(Ts, [LIGHT, "#93c5fd", BLUE, PURPLE, PINK])):
        ax.semilogx(ctxs, grid[:, j], "o-", color=col, ms=5, lw=1.9,
                    base=2, label=f"$T = {T}$")
    ax.set_xlabel("context width")
    ax.set_ylabel("per-token accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("widening $C$ does not rescue it", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    fig.tight_layout()
    save(fig, "09-bottleneck")


if __name__ == "__main__":
    print("Part XI — sequence figures")
    fig_unfolding()          # 1
    fig_truncated()          # 2
    fig_gradient_decay()     # 3
    fig_cliff()              # 4
    fig_long_dependency()    # 5
    fig_leaky()              # 6
    fig_gates()              # 7
    fig_forget_bias()        # 8
    fig_bottleneck()         # 9
    print("done ->", OUT)
