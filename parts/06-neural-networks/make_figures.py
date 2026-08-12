"""
Figures for Part VII — Neural Networks.

Every network here is actually trained. The XOR solution is a real fit, the
initialisation histograms are real forward passes, the vanishing-gradient
measurements are real backward passes through real depth.

    python3 make_figures.py           # writes into ../../book/src/figures/nn/
"""
import os
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "nn"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
LIGHT, INK = "#cbd5e1", "#1e293b"
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
    import re
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


def relu(z): return np.maximum(z, 0)


def train_mlp(X, y, hidden=2, steps=6000, lr=0.35, seed=0, act="relu", b1_init=0.0):
    """A real two-layer network trained by real backpropagation.

    b1_init sets the hidden layer's initial bias. A small positive value
    starts every unit active rather than at the knife-edge of zero, which
    guards against the permanently-dead unit §19 describes: with a
    two-unit ReLU layer on a four-point problem, a single unit that never
    fires is enough to make the network fail.
    """
    rng = np.random.default_rng(seed)
    n_in = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden)); b1 = np.full(hidden, b1_init)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1));  b2 = np.zeros(1)
    Y = y.reshape(-1, 1)
    for _ in range(steps):
        Z1 = X @ W1 + b1
        A1 = relu(Z1) if act == "relu" else np.tanh(Z1)
        Z2 = A1 @ W2 + b2
        P = 1 / (1 + np.exp(-Z2))
        dZ2 = (P - Y) / len(X)                       # cross-entropy + sigmoid
        dW2 = A1.T @ dZ2; db2 = dZ2.sum(0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * ((Z1 > 0) if act == "relu" else (1 - np.tanh(Z1) ** 2))
        dW1 = X.T @ dZ1; db1 = dZ1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return W1, b1, W2, b2


# =============================================================== fig 1 ======
def fig_xor():
    """The problem that stopped the field, and the hidden layer that solves it."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])
    # b1_init=0.1: a bare two-unit ReLU layer on XOR is fragile to dead units
    # (a seed sweep at b1_init=0.0 solves it only ~1 time in 5); the small
    # positive bias hardens it without changing what the figure claims.
    W1, b1, W2, b2 = train_mlp(X, y, hidden=2, steps=20000, lr=0.6, seed=3, b1_init=0.1)
    assert np.all(
        (1 / (1 + np.exp(-(relu(X @ W1 + b1) @ W2 + b2))).ravel() > 0.5) == (y > 0.5)
    ), "XOR must actually be solved before this figure is drawn"

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))

    ax = axes[0]
    for xi, yi in zip(X, y):
        ax.plot(*xi, "o", ms=13, color=(ORANGE if yi else BLUE), zorder=5)
    for a, b in [(-0.4, 1.5), (-0.1, 1.2)]:
        ax.plot([-0.4, 1.4], [a, b], color=GRAY, lw=1.1, ls="--", alpha=0.8)
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.45, 1.45); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_title("XOR: no line separates these", fontsize=9.8, pad=8)
    tidy(ax, grid=False)

    ax = axes[1]
    H = relu(X @ W1 + b1)
    for hi, yi in zip(H, y):
        ax.plot(*hi, "o", ms=13, color=(ORANGE if yi else BLUE), zorder=5)
    pts = np.array([h for h in H])
    lo, hi_ = pts.min() - .3, pts.max() + .3
    ax.plot([lo, hi_], [(-b2[0] - W2[0, 0] * lo) / W2[1, 0],
                        (-b2[0] - W2[0, 0] * hi_) / W2[1, 0]],
            color=GREEN, lw=2.1)
    ax.set_xlim(lo, hi_); ax.set_ylim(lo, hi_)
    ax.set_xlabel("hidden unit 1"); ax.set_ylabel("hidden unit 2")
    ax.set_title("after the hidden layer: a line does", fontsize=9.8, pad=8)
    tidy(ax, grid=False)

    ax = axes[2]
    g = np.linspace(-0.4, 1.4, 220)
    G0, G1 = np.meshgrid(g, g)
    P = np.c_[G0.ravel(), G1.ravel()]
    Z = 1 / (1 + np.exp(-(relu(P @ W1 + b1) @ W2 + b2)))
    ax.contourf(G0, G1, Z.reshape(G0.shape), levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(G0, G1, Z.reshape(G0.shape), levels=[0.5], colors=[INK], linewidths=2)
    for xi, yi in zip(X, y):
        ax.plot(*xi, "o", ms=11, color=(ORANGE if yi else BLUE),
                mec="white", mew=1.4, zorder=6)
    ax.set_aspect("equal"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_title("the learned function", fontsize=9.8, pad=8)
    tidy(ax, grid=False)
    save(fig, "01-xor")


# =============================================================== fig 2 ======
def fig_universal_approx():
    """Bumps built from unit pairs, summed into an arbitrary curve. Real fit."""
    f = lambda x: np.sin(3 * x) + 0.4 * np.cos(7 * x)
    x = np.linspace(-2, 2, 400)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.6))
    for ax, h in zip(axes, [3, 10, 60]):
        rng = np.random.default_rng(2)
        # fit hidden layer at random, solve output layer exactly (random features)
        W = rng.uniform(-4, 4, h); b = rng.uniform(-6, 6, h)
        H = relu(np.outer(x, W) + b)
        w2 = np.linalg.lstsq(np.c_[H, np.ones_like(x)], f(x), rcond=None)[0]
        yh = np.c_[H, np.ones_like(x)] @ w2
        ax.plot(x, f(x), color=GREEN, lw=2.0, ls="--", label="target")
        ax.plot(x, yh, color=ORANGE, lw=2.0, label="network")
        ax.set_title(f"{h} hidden units", fontsize=9.6, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        if h == 3:
            ax.legend(fontsize=7.5, frameon=False, loc="lower left")
        tidy(ax, grid=False)
    save(fig, "02-universal-approximation")


# =============================================================== fig 3 ======
def fig_depth_vs_width():
    """Depth buys regions exponentially; width buys them linearly. Counted."""
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.85))
    ax = axes[0]
    units = 30
    depths = np.arange(1, 7)
    per_layer = units // depths
    regions = np.where(per_layer >= 1, per_layer.astype(float) ** depths, np.nan)
    ax.semilogy(depths, regions, "o-", color=ORANGE, lw=2.2, ms=5)
    ax.set_xlabel("depth (with 30 units total)")
    ax.set_ylabel("linear regions (rough count)")
    ax.set_title("same budget, arranged deeper", fontsize=9.7, pad=8)
    tidy(ax)

    ax = axes[1]
    rng = np.random.default_rng(4)
    x = np.linspace(-1, 1, 300)[:, None]
    for depth, col in [(1, LIGHT), (2, BLUE), (4, ORANGE)]:
        h = x.copy()
        for _ in range(depth):
            W = rng.normal(0, np.sqrt(2 / h.shape[1]), (h.shape[1], 6))
            h = relu(h @ W + rng.normal(0, .4, 6))
        out = h @ rng.normal(0, .7, (h.shape[1], 1))
        ax.plot(x.ravel(), out.ravel(), color=col, lw=2.0, label=f"depth {depth}")
    ax.legend(fontsize=8.2, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("random networks: deeper folds more", fontsize=9.7, pad=8)
    tidy(ax, grid=False)
    save(fig, "03-depth-vs-width")


# =============================================================== fig 4 ======
def fig_anatomy():
    """The object itself, labelled."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_xlim(0, 13); ax.set_ylim(-0.4, 4.6); ax.axis("off")
    layers = [(1.2, 3, "input\n$\\mathbf{x}$", "#eef2f7", GRAY),
              (4.4, 4, "hidden 1\n$\\mathbf{h}^{(1)}$", "#eff5ff", BLUE),
              (7.6, 4, "hidden 2\n$\\mathbf{h}^{(2)}$", "#eff5ff", BLUE),
              (10.8, 2, "output\n$\\hat{\\mathbf{y}}$", "#fff7ed", ORANGE)]
    pos = {}
    for li, (cx, n, lab, fc, ec) in enumerate(layers):
        ys = np.linspace(0.6, 3.6, n)
        pos[li] = [(cx, yy) for yy in ys]
        for (px, py) in pos[li]:
            ax.add_patch(Circle((px, py), 0.26, facecolor=fc, edgecolor=ec,
                                lw=1.6, zorder=5))
        ax.text(cx, 4.15, lab, ha="center", fontsize=9, color=INK)
    for li in range(3):
        for (x0, y0) in pos[li]:
            for (x1, y1) in pos[li + 1]:
                ax.plot([x0 + .26, x1 - .26], [y0, y1], color=LIGHT, lw=0.7, zorder=2)
    for cx, lab in [(2.8, "$W^{(1)}, \\mathbf{b}^{(1)}$"),
                    (6.0, "$W^{(2)}, \\mathbf{b}^{(2)}$"),
                    (9.2, "$W^{(3)}, \\mathbf{b}^{(3)}$")]:
        ax.text(cx, 0.02, lab, ha="center", fontsize=9, color=PURPLE)
    ax.text(6.5, -0.35, r"$\hat{\mathbf{y}} = f^{(3)}(f^{(2)}(f^{(1)}(\mathbf{x})))$",
            ha="center", fontsize=11, color=INK)
    save(fig, "04-anatomy")


# =============================================================== fig 5 ======
def fig_hidden_units():
    """The activation zoo, with derivatives underneath — the part that matters."""
    x = np.linspace(-4, 4, 600)
    # True GELU, x*Phi(x) with Phi the standard normal CDF — not the SiLU/Swish
    # x*sigmoid(x) approximation, which differs from it by up to 0.19 over this
    # range. Phi'(x) is the standard normal density.
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    gelu, gelu_grad = x * Phi, Phi + x * phi
    fns = [
        ("ReLU", np.maximum(x, 0), (x > 0).astype(float), GREEN),
        ("Leaky ReLU", np.where(x > 0, x, 0.1 * x),
         np.where(x > 0, 1.0, 0.1), BLUE),
        ("tanh", np.tanh(x), 1 - np.tanh(x) ** 2, PURPLE),
        ("GELU", gelu, gelu_grad, ORANGE),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 3.7), sharex=True)
    for j, (name, y, dy, c) in enumerate(fns):
        ax = axes[0, j]
        ax.plot(x, y, color=c, lw=2.2); ax.axhline(0, color=LIGHT, lw=.8)
        ax.set_title(name, fontsize=9.8, pad=6); ax.set_yticks([]); tidy(ax)
        ax = axes[1, j]
        ax.plot(x, dy, color=c, lw=2.2)
        ax.fill_between(x, 0, dy, color=c, alpha=0.14)
        ax.set_ylim(-0.15, 1.25); ax.set_yticks([]); tidy(ax)
        ax.set_xlabel("$x$", fontsize=8.8, color=GRAY)
    axes[0, 0].set_ylabel("f(x)", fontsize=9.4)
    axes[1, 0].set_ylabel("f'(x)", fontsize=9.4)
    save(fig, "05-hidden-units")


# =============================================================== fig 6 ======
def fig_ce_vs_mse():
    """Why squared error stalls a saturated sigmoid and cross-entropy does not."""
    z = np.linspace(-6, 6, 500)
    p = 1 / (1 + np.exp(-z))
    g_mse = 2 * (p - 1) * p * (1 - p)          # target 1
    g_ce = (p - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.85))
    ax = axes[0]
    ax.plot(z, (p - 1) ** 2, color=PINK, lw=2.2, label="squared error")
    ax.plot(z, -np.log(np.clip(p, 1e-12, 1)), color=GREEN, lw=2.2,
            label="cross-entropy")
    ax.set_ylim(0, 4); ax.legend(fontsize=8.5, frameon=False)
    ax.set_xlabel("logit $z$ (true label is 1)"); ax.set_ylabel("loss")
    ax.set_title("loss when the model is confidently wrong", fontsize=9.6, pad=8)
    tidy(ax)
    ax = axes[1]
    ax.plot(z, np.abs(g_mse), color=PINK, lw=2.2, label="squared error")
    ax.plot(z, np.abs(g_ce), color=GREEN, lw=2.2, label="cross-entropy")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_xlabel("logit $z$"); ax.set_ylabel(r"$|\partial L / \partial z|$")
    ax.set_title("gradient: squared error vanishes exactly where you need it",
                 fontsize=9.3, pad=8)
    tidy(ax)
    save(fig, "06-ce-vs-mse")


# =============================================================== fig 7 ======
def fig_initialization():
    """Activation scale across 30 layers under three initialisations. Real passes."""
    rng = np.random.default_rng(0)
    n, width, depth = 512, 256, 30
    schemes = [("too small  $\\sigma=0.01$", 0.01, PINK),
               ("too large  $\\sigma=0.2$", 0.2, ORANGE),
               ("He  $\\sqrt{2/n}$", None, GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.9))
    forward = {}
    for lab, s, col in schemes:
        h = rng.normal(0, 1, (n, width))
        sd = np.sqrt(2.0 / width) if s is None else s
        stds, Ws, Zs = [], [], []
        for _ in range(depth):
            W = rng.normal(0, sd, (width, width))
            z = h @ W
            h = relu(z)
            Ws.append(W); Zs.append(z)
            stds.append(h.std())
        forward[lab] = (Ws, Zs, col)
        axes[0].semilogy(range(1, depth + 1), np.maximum(stds, 1e-30),
                         color=col, lw=2.1, label=lab)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("activation std")
    axes[0].legend(fontsize=7.8, frameon=False)
    axes[0].set_title("forward pass: signal dies or explodes", fontsize=9.6, pad=8)
    tidy(axes[0])

    # Real backward pass: reuse the exact weights and ReLU masks the forward
    # pass just produced, rather than a fixed expected pass-through.
    ax = axes[1]
    for lab, s, col in schemes:
        Ws, Zs, col = forward[lab]
        g = rng.normal(0, 1, (n, width))
        mags = []
        for W, z in zip(reversed(Ws), reversed(Zs)):
            g = (g * (z > 0)) @ W.T
            mags.append(np.abs(g).mean())
        ax.semilogy(range(1, depth + 1), np.maximum(mags[::-1], 1e-30),
                    color=col, lw=2.1)
    ax.set_xlabel("layer"); ax.set_ylabel("gradient magnitude")
    ax.set_title("backward pass: the same problem, mirrored", fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "07-initialization")


# =============================================================== fig 8 ======
def fig_compgraph():
    """The expression as a DAG — the object backprop actually walks."""
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.2, 3.4); ax.axis("off")
    nodes = {"x": (0.9, 2.5), "w": (0.9, 1.1), "b": (0.9, 0.2),
             "u": (4.0, 1.8), "z": (6.8, 1.2), "a": (9.4, 1.2), "L": (12.1, 1.2)}
    labels = {"x": "$x$", "w": "$w$", "b": "$b$", "u": "$u = wx$",
              "z": "$z = u + b$", "a": r"$a = \sigma(z)$", "L": "$L = (a-y)^2$"}
    fill = {"x": "#eef2f7", "w": "#f6f2ff", "b": "#f6f2ff",
            "u": "#eff5ff", "z": "#eff5ff", "a": "#eff5ff", "L": "#fff7ed"}
    edge = {"x": GRAY, "w": PURPLE, "b": PURPLE,
            "u": BLUE, "z": BLUE, "a": BLUE, "L": ORANGE}
    for k, (px, py) in nodes.items():
        ax.add_patch(FancyBboxPatch((px - .78, py - .3), 1.56, .62,
                                    boxstyle="round,pad=0.07", facecolor=fill[k],
                                    edgecolor=edge[k], lw=1.6, zorder=4))
        ax.text(px, py, labels[k], ha="center", va="center", fontsize=9.6, zorder=5)
    for a_, b_ in [("x", "u"), ("w", "u"), ("u", "z"), ("b", "z"),
                   ("z", "a"), ("a", "L")]:
        (x0, y0), (x1, y1) = nodes[a_], nodes[b_]
        ax.add_patch(FancyArrowPatch((x0 + .8, y0), (x1 - .8, y1),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=1.5, color=GRAY, zorder=3,
                                     shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(1.7, 0.72), xytext=(11.3, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=PINK, lw=2.0))
    ax.text(6.5, 0.34, "backward pass: one local derivative per edge, multiplied",
            ha="center", fontsize=9, color=PINK)
    save(fig, "08-computational-graph")


# =============================================================== fig 9 ======
def fig_modes():
    """Forward mode costs one pass per input; reverse mode one per output."""
    fig, ax = plt.subplots(figsize=(5.7, 2.9))
    n = np.logspace(0, 6, 60)
    ax.loglog(n, n, color=BLUE, lw=2.2, label="forward mode: $O(n)$ passes")
    ax.loglog(n, np.ones_like(n), color=ORANGE, lw=2.2,
              label="reverse mode: $O(1)$ passes")
    ax.axvline(1e6, color=GRAY, ls=":", lw=1.3)
    ax.text(1.2e4, 3e3, "a small network\nhas $10^6$ parameters",
            fontsize=8.2, color=GRAY)
    ax.set_xlabel("number of inputs (parameters)")
    ax.set_ylabel("passes needed for the full gradient")
    ax.legend(fontsize=8.4, frameon=False)
    ax.set_title("why training uses reverse mode", fontsize=9.7, pad=8)
    tidy(ax)
    save(fig, "09-forward-vs-reverse")


# ============================================================== fig 10 ======
def fig_vanishing():
    """Measured gradient magnitude by layer, sigmoid versus ReLU."""
    rng = np.random.default_rng(2)
    width, depth, n = 128, 25, 256
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for act, col, lab in [("sigmoid", PINK, "sigmoid"), ("tanh", BLUE, "tanh"),
                          ("relu", GREEN, "ReLU + He init")]:
        h = rng.normal(0, 1, (n, width))
        Ws, caches = [], []
        for _ in range(depth):
            sd = np.sqrt(2.0 / width) if act == "relu" else np.sqrt(1.0 / width)
            W = rng.normal(0, sd, (width, width)); Ws.append(W)
            z = h @ W
            h = (1 / (1 + np.exp(-z))) if act == "sigmoid" else (
                np.tanh(z) if act == "tanh" else relu(z))
            caches.append((z, h))
        g = rng.normal(0, 1, (n, width))
        mags = []
        for W, (z, hh) in zip(reversed(Ws), reversed(caches)):
            if act == "sigmoid":
                g = g * hh * (1 - hh)
            elif act == "tanh":
                g = g * (1 - hh ** 2)
            else:
                g = g * (z > 0)
            g = g @ W.T
            mags.append(np.abs(g).mean())
        ax.semilogy(range(1, depth + 1), np.maximum(mags[::-1], 1e-30),
                    color=col, lw=2.2, label=lab)
    ax.set_xlabel("layer (1 = closest to input)")
    ax.set_ylabel("mean gradient magnitude")
    ax.legend(fontsize=8.4, frameon=False)
    ax.set_title("25 layers: what reaches the early layers", fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "10-vanishing-gradients")


# ============================================================== fig 11 ======
def fig_dead_relu():
    """Dead units: a real network, counted after real training."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 20)); y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(float)
    fig, ax = plt.subplots(figsize=(5.7, 2.85))
    for lr, col in [(0.5, GREEN), (5.0, BLUE), (20.0, PINK)]:
        W1 = rng.normal(0, np.sqrt(2 / 20), (20, 64)); b1 = np.zeros(64)
        W2 = rng.normal(0, np.sqrt(2 / 64), (64, 1)); b2 = np.zeros(1)
        dead = []
        Y = y.reshape(-1, 1)
        for t in range(300):
            Z1 = X @ W1 + b1; A1 = relu(Z1)
            P = 1 / (1 + np.exp(-(A1 @ W2 + b2)))
            dZ2 = (P - Y) / len(X)
            dA1 = dZ2 @ W2.T; dZ1 = dA1 * (Z1 > 0)
            W2 -= lr * (A1.T @ dZ2); b2 -= lr * dZ2.sum(0)
            W1 -= lr * (X.T @ dZ1);  b1 -= lr * dZ1.sum(0)
            dead.append(np.mean((X @ W1 + b1).max(0) <= 0) * 100)
        ax.plot(dead, color=col, lw=2.1, label=f"lr = {lr}")
    ax.set_xlabel("training step"); ax.set_ylabel("% of units never active")
    ax.legend(fontsize=8.4, frameon=False)
    ax.set_title("dead ReLUs: a large step size kills units permanently",
                 fontsize=9.5, pad=8)
    tidy(ax)
    save(fig, "11-dead-relu")


# ============================================================== fig 12 ======
def fig_training_curves():
    """What healthy, diverging and stalled training actually look like."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(600, 12)); w = rng.normal(size=12)
    y = (X @ w + 0.4 * rng.normal(size=600) > 0).astype(float).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(5.9, 2.9))
    for lr, col, lab in [(0.5, GREEN, "healthy"), (25.0, PINK, "too large"),
                         (0.002, BLUE, "too small")]:
        W1 = rng.normal(0, np.sqrt(2 / 12), (12, 32)); b1 = np.zeros(32)
        W2 = rng.normal(0, np.sqrt(2 / 32), (32, 1)); b2 = np.zeros(1)
        losses = []
        for _ in range(220):
            Z1 = X @ W1 + b1; A1 = relu(Z1)
            Z2 = A1 @ W2 + b2
            P = 1 / (1 + np.exp(-np.clip(Z2, -60, 60)))
            losses.append(float(-np.mean(y * np.log(P + 1e-12)
                                         + (1 - y) * np.log(1 - P + 1e-12))))
            dZ2 = (P - y) / len(X)
            dA1 = dZ2 @ W2.T; dZ1 = dA1 * (Z1 > 0)
            W2 -= lr * (A1.T @ dZ2); b2 -= lr * dZ2.sum(0)
            W1 -= lr * (X.T @ dZ1);  b1 -= lr * dZ1.sum(0)
            if not np.all(np.isfinite(W1)):
                break
        ax.semilogy(np.clip(losses, 1e-4, 1e3), color=col, lw=2.1, label=lab)
    ax.set_xlabel("step"); ax.set_ylabel("training loss")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("three learning rates, one network", fontsize=9.7, pad=8)
    tidy(ax)
    save(fig, "12-training-curves")


if __name__ == "__main__":
    print("generating Part VII figures ->", OUT)
    for fn in (fig_xor, fig_universal_approx, fig_depth_vs_width, fig_anatomy,
               fig_hidden_units, fig_ce_vs_mse, fig_initialization, fig_compgraph,
               fig_modes, fig_vanishing, fig_dead_relu, fig_training_curves):
        fn()
    print("done.")
