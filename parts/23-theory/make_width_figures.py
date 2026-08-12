"""
Figures 8 to 12 for Part XXIII — the appended movement G, Width.

Predictions for every one of these were recorded before the
script was run. Figures 1 to 7 live in
make_figures.py and predate that discipline; nothing is invented for them here.

    python3 make_width_figures.py

Runtime: about seven minutes. Numpy only — no autograd anywhere. Every network
below is a two- or three-layer MLP whose forward pass, Jacobian and gradient are
written out by hand, which is slower to write and much easier to trust: the
claims in this movement are about the exact scaling of quantities with width, and
a framework that silently rescales an initialisation would destroy all of them.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "lt"))
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


# ---------------------------------------------------------------------------
# A one-hidden-layer network, written out by hand.
#
#   f(x) = (alpha / m^b_out) * V @ phi( (1 / m^b_in) * W x )
#
# The exponents b_in and b_out are the whole subject. Standard parameterisation
# puts the width scaling in the *initialisation* and none in the forward pass;
# NTK parameterisation puts 1/sqrt(m) in the forward pass; muP differs from NTK
# in the output layer and in the per-layer learning rates. Writing them as
# explicit exponents rather than as three separate classes is what makes the
# comparison honest.
# ---------------------------------------------------------------------------

def init_net(m, d, rng, scheme):
    """W is (m, d), V is (1, m). Returns weights and the forward/backward scales."""
    if scheme == "standard":
        # Var = 1/fan_in on both layers, no scaling in the forward pass.
        W = rng.normal(scale=1.0 / np.sqrt(d), size=(m, d))
        V = rng.normal(scale=1.0 / np.sqrt(m), size=(1, m))
        s_in, s_out = 1.0, 1.0
        lr_W, lr_V = 1.0, 1.0
    elif scheme == "ntk":
        W = rng.normal(size=(m, d))
        V = rng.normal(size=(1, m))
        s_in, s_out = 1.0 / np.sqrt(d), 1.0 / np.sqrt(m)
        lr_W, lr_V = 1.0, 1.0
    elif scheme == "mup":
        # Three things at once, and all three are needed. The output layer is
        # initialised at 1/m (variance 1/m^2) so the network starts at the zero
        # function; its learning rate falls like 1/m; and the *input* layer's
        # learning rate GROWS like m. That last one is the easy one to omit, and
        # omitting it makes muP look lazier than NTK parameterisation, which is
        # the exact opposite of the property it exists to have.
        W = rng.normal(scale=1.0 / np.sqrt(d), size=(m, d))
        V = rng.normal(scale=1.0 / m, size=(1, m))
        s_in, s_out = 1.0, 1.0
        lr_W, lr_V = float(m), 1.0 / m
    else:
        raise ValueError(scheme)
    return dict(W=W, V=V, s_in=s_in, s_out=s_out, lr_W=lr_W, lr_V=lr_V)


def forward(net, X):
    """Returns (pre-activations, activations, outputs). tanh throughout."""
    Z = (X @ net["W"].T) * net["s_in"]
    H = np.tanh(Z)
    Y = (H @ net["V"].T) * net["s_out"]
    return Z, H, Y[:, 0]


def grads(net, X, resid):
    """Gradients of (1/2n)*sum(resid^2) with n = len(X). Hand-derived."""
    n = len(X)
    Z, H, _ = forward(net, X)
    gV = (resid @ H)[None, :] * net["s_out"] / n
    dH = np.outer(resid, net["V"][0]) * net["s_out"]
    dZ = dH * (1.0 - np.tanh(Z) ** 2)
    gW = (dZ.T @ X) * net["s_in"] / n
    return gW, gV


def ntk_features(net, X):
    """The Jacobian of the output with respect to every parameter, flattened.

    The empirical NTK is J J^T. Building J explicitly is O(params * n) memory,
    which is why the input dimension and sample count below are small: the point
    is the width scaling, and holding everything else fixed is what makes the
    scaling readable.
    """
    Z, H, _ = forward(net, X)
    dphi = 1.0 - np.tanh(Z) ** 2
    # d out / d V  ->  H * s_out
    JV = H * net["s_out"]
    # d out / d W  ->  outer(V * dphi * s_out * s_in, x)
    coef = dphi * net["V"][0][None, :] * net["s_out"] * net["s_in"]
    JW = (coef[:, :, None] * X[:, None, :]).reshape(len(X), -1)
    return np.concatenate([JW, JV], axis=1)


def train(net, X, y, lr, steps):
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            _, _, out = forward(net, X)
            resid = out - y
            gW, gV = grads(net, X, resid)
            net["W"] -= lr * net["lr_W"] * gW
            net["V"] -= lr * net["lr_V"] * gV
    return net


def train_to(net, X, y, lr, target, max_steps):
    """Train until the training loss reaches `target`, and report the steps used.

    Every comparison in this movement is between networks of different width or
    different parameterisation, and a fixed step budget does not compare them: a
    wider network reaches a lower loss in the same number of steps, so anything
    measured afterwards is partly measuring how far each one got. Matching on the
    training loss is what makes the width the only difference.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(max_steps):
            _, _, out = forward(net, X)
            resid = out - y
            loss = float(np.mean(resid ** 2))
            if not np.isfinite(loss):
                return net, t, np.inf
            if loss <= target:
                return net, t, loss
            gW, gV = grads(net, X, resid)
            net["W"] -= lr * net["lr_W"] * gW
            net["V"] -= lr * net["lr_V"] * gV
    _, _, out = forward(net, X)
    return net, max_steps, float(np.mean((out - y) ** 2))


# ---------------------------------------------------------------------------
# Figure 8 — the kernel stops moving
# ---------------------------------------------------------------------------

def figure8(d=6, n=48, max_steps=20000, lr=0.5, seed=0):
    widths = [16, 32, 64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    y = np.sign(X[:, 0] * X[:, 1]) * 0.8

    rel, used, reached = [], [], []
    for m in widths:
        net = init_net(m, d, np.random.default_rng(seed + m), "ntk")
        K0 = ntk_features(net, X)
        K0 = K0 @ K0.T
        _, _, f0 = forward(net, X)
        start = float(np.mean((f0 - y) ** 2))
        # Halving each network's own initial loss is the matched criterion. A
        # fixed target is unreachable for the wide nets and a fixed step budget
        # lets each one get a different distance, so either would measure
        # progress rather than width.
        net, t, loss = train_to(net, X, y, lr, 0.5 * start, max_steps)
        K1 = ntk_features(net, X)
        K1 = K1 @ K1.T
        rel.append(np.linalg.norm(K1 - K0) / np.linalg.norm(K0))
        used.append(t)
        reached.append(loss)
        print(f"  width {m:5d}: relative kernel change {rel[-1]:.4f}"
              f"   (loss {start:.4f} -> {loss:.4f} in {t} steps)")
    print(f"  every width halved its own initial loss: "
          f"{all(t < max_steps for t in used)}")

    rel = np.array(rel)
    w = np.array(widths, float)
    slope, intercept = np.polyfit(np.log(w), np.log(rel), 1)
    print(f"  fitted exponent {slope:+.3f}   (a scaling argument suggests -0.5)")
    print(f"  at the widest network measured the kernel still moves by "
          f"{rel[-1] * 100:.1f}% — the limit is approached, not attained")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.loglog(w, rel, "o-", color=BLUE, lw=2.0, ms=5, label="measured")
    ax.loglog(w, np.exp(intercept) * w ** slope, "--", color=GRAY, lw=1.3,
              label=f"fit: slope {slope:.2f}")
    ax.loglog(w, rel[0] * (w / w[0]) ** -0.5, ":", color=ORANGE, lw=1.5,
              label=r"$m^{-1/2}$")
    ax.set_xlabel("width $m$")
    ax.set_ylabel("relative change in the NTK during training")
    ax.set_title("The kernel freezes, slowly", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "08-ntk-freeze")
    return widths, rel, slope


# ---------------------------------------------------------------------------
# Figure 9 — lazy and rich
# ---------------------------------------------------------------------------

def figure9(d=8, n=120, ntest=400, m=512, target=0.02, max_steps=40000, seed=1):
    """A target depending on a single direction: the classic feature-learning task.

    The output multiplier alpha interpolates the regimes. Large alpha means the
    function moves a long way for a small parameter change, so the parameters
    barely have to move at all — that is exactly what "lazy" means, and it is why
    the regime is a property of the scale rather than of the architecture.

    Every arm is trained to the *same training loss* rather than for the same
    number of steps. Without that the lazy arms simply have not converged, and a
    comparison of test errors is a comparison of how far each one got.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    Xt = rng.normal(size=(ntest, d)) / np.sqrt(d)
    f = lambda Z: np.tanh(3.0 * (Z @ u))
    y, yt = f(X), f(Xt)

    alphas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    test, feat, reached, steps_used, align = [], [], [], [], []
    for a in alphas:
        net = init_net(m, d, np.random.default_rng(seed + 7), "ntk")
        net["s_out"] *= a
        _, H0, _ = forward(net, X)
        W0 = net["W"].copy()
        # the loss curvature scales like alpha^2, so the step does too
        net, t, loss = train_to(net, X, y, 0.5 / a ** 2, target, max_steps)
        _, H1, _ = forward(net, X)
        _, _, outt = forward(net, Xt)
        test.append(float(np.mean((outt - yt) ** 2)))
        feat.append(float(np.linalg.norm(H1 - H0) / np.linalg.norm(H0)))
        reached.append(loss)
        steps_used.append(t)
        # how much of the first-layer *change* points along the true direction
        dW = net["W"] - W0
        align.append(float(np.abs(dW @ u).sum() / (np.abs(dW).sum() + 1e-12)))
        print(f"  alpha {a:5.2f}: test {test[-1]:.4f}  features moved {feat[-1]:.4f}"
              f"  alignment {align[-1]:.3f}   (loss {loss:.4f} in {t} steps)")

    matched = all(abs(r - target) < 0.002 for r in reached)
    print(f"  all arms reached the same training loss {target}: {matched}")
    print(f"  across the sweep, test error {test[0]:.4f} -> {test[-1]:.4f} "
          f"({test[-1] / test[0]:.1f}x) at identical training loss")
    print(f"  alignment with the true direction {align[0]:.3f} -> {align[-1]:.3f} "
          f"({align[0] / align[-1]:.1f}x): the lazy network fits the data without "
          f"finding the direction the data depends on")
    print(f"  feature movement {feat[0]:.4f} -> {feat[-1]:.4f}, and it plateaus "
          f"rather than reaching zero — laziness saturates")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.7, 2.95))
    ax.semilogx(alphas, feat, "o-", color=BLUE, lw=2.0, ms=5, base=2,
                label="features moved")
    ax.semilogx(alphas, align, "s--", color=GREEN, lw=1.6, ms=4, base=2,
                label="alignment with the true direction")
    ax.set_xlabel(r"output scale $\alpha$")
    ax.set_ylabel("relative movement / alignment")
    ax.set_title("What the network learns", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.semilogx(alphas, test, "o-", color=ORANGE, lw=2.0, ms=5, base=2)
    ax2.set_xlabel(r"output scale $\alpha$")
    ax2.set_ylabel("test error at equal training loss")
    ax2.set_title("What it costs", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("Lazy and rich are the same network at different scales",
                 fontsize=10.5, y=1.03)
    save(fig, "09-lazy-rich")
    return alphas, feat, test, align, reached


# ---------------------------------------------------------------------------
# Figure 10 — the coordinate check
# ---------------------------------------------------------------------------

def figure10(d=6, n=64, lr=0.2, seed=2):
    """What one optimiser step does to the *output*, against width.

    The quantity that decides stability is the change in the function, not the
    change in an activation. Under standard parameterisation the output-layer
    update is coherent with the hidden activations that produced it, so summing
    over m units gives a change in the output that grows with m — and a learning
    rate tuned at one width is wrong at another for that reason alone.

    The second panel is what separates the two stable schemes: NTK parameterisation
    keeps the output stable by making the features stop moving, which is what
    "lazy" means; muP keeps the output stable while the features still move.
    """
    widths = [64, 128, 256, 512, 1024, 2048, 4096]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    y = np.sign(X[:, 0]) * 0.5

    out = {}
    for scheme in ("standard", "ntk", "mup"):
        df, dh = [], []
        for m in widths:
            net = init_net(m, d, np.random.default_rng(seed + m), scheme)
            _, H0, f0 = forward(net, X)
            gW, gV = grads(net, X, f0 - y)
            net["W"] -= lr * net["lr_W"] * gW
            net["V"] -= lr * net["lr_V"] * gV
            _, H1, f1 = forward(net, X)
            df.append(float(np.abs(f1 - f0).mean()))
            dh.append(float(np.linalg.norm(H1 - H0) / np.linalg.norm(H0)))
        out[scheme] = (np.array(df), np.array(dh))
        r = widths[-1] / widths[0]
        print(f"  {scheme:9s}: |df| {df[0]:.3e} -> {df[-1]:.3e} "
              f"({df[-1] / df[0]:8.2f}x over {r:.0f}x width);  "
              f"relative feature move {dh[0]:.3e} -> {dh[-1]:.3e} "
              f"({dh[-1] / dh[0]:.3f}x)")
    print("  a stable parameterisation is one whose |df| curve is flat; a "
          "feature-learning one also has a flat feature curve")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    for scheme, c in zip(("standard", "ntk", "mup"), (ORANGE, PURPLE, BLUE)):
        ax.loglog(widths, out[scheme][0], "o-", color=c, lw=1.8, ms=4.5,
                  label=scheme)
        ax2.loglog(widths, out[scheme][1], "o-", color=c, lw=1.8, ms=4.5,
                   label=scheme)
    ax.set_xlabel("width $m$")
    ax.set_ylabel(r"$|\Delta f|$ after one step")
    ax.set_title("Is the update stable?", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    ax2.set_xlabel("width $m$")
    ax2.set_ylabel("relative feature movement")
    ax2.set_title("Do the features still move?", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("A coordinate check takes one step, not a training run",
                 fontsize=10.5, y=1.03)
    save(fig, "10-coord-check")
    return widths, out


# ---------------------------------------------------------------------------
# Figure 11 — hyperparameter transfer
# ---------------------------------------------------------------------------

def figure11(d=6, n=96, ntest=400, steps=400, seed=3):
    widths = [64, 128, 256, 512, 1024]
    lrs = np.logspace(-4.5, 2.5, 57)
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    Xt = rng.normal(size=(ntest, d)) / np.sqrt(d)
    f = lambda Z: np.tanh(2.5 * (Z @ u))
    y, yt = f(X), f(Xt)

    curves, best = {}, {}
    for scheme in ("standard", "mup"):
        rows = []
        for m in widths:
            losses = []
            for lr in lrs:
                net = init_net(m, d, np.random.default_rng(seed + 11), scheme)
                net = train(net, X, y, lr, steps)
                with np.errstate(over="ignore", invalid="ignore"):
                    _, _, o = forward(net, Xt)
                    v = float(np.mean((o - yt) ** 2))
                losses.append(v if np.isfinite(v) else np.nan)
            rows.append(np.array(losses))
        curves[scheme] = rows
        best[scheme] = [lrs[int(np.nanargmin(r))] for r in rows]
        span = max(best[scheme]) / min(best[scheme])
        edge = any(b in (lrs[0], lrs[-1]) for b in best[scheme])
        print(f"  {scheme:9s}: best lr by width "
              + " ".join(f"{b:.3g}" for b in best[scheme])
              + f"   (spread {span:.1f}x over {widths[-1] // widths[0]}x width)")
        if edge:
            print(f"    WARNING: an optimum sits on the edge of the swept grid "
                  f"[{lrs[0]:.3g}, {lrs[-1]:.3g}] — it has not been found")
        step = lrs[1] / lrs[0]
        print(f"    grid resolution is a factor of {step:.2f} per point, so any "
              f"drift below that is not resolved")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.95), sharey=True)
    for ax, scheme, title in zip(axes, ("standard", "mup"),
                                 ("Standard parameterisation", r"$\mu$P")):
        for m, r, c in zip(widths, curves[scheme],
                           (LIGHT, CYAN, GREEN, ORANGE, BLUE)):
            ax.semilogx(lrs, r, "-", color=c, lw=1.7, label=f"$m={m}$")
        for m, r, c in zip(widths, curves[scheme],
                           (LIGHT, CYAN, GREEN, ORANGE, BLUE)):
            j = int(np.nanargmin(r))
            ax.plot([lrs[j]], [r[j]], "*", color=c, ms=12, zorder=5)
        ax.set_xlabel("learning rate")
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylim(0, 0.6)
        tidy(ax)
    axes[0].set_ylabel("test error")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Whether the best learning rate moves with width",
                 fontsize=10.5, y=1.03)
    save(fig, "11-transfer")
    return widths, lrs, curves, best


# ---------------------------------------------------------------------------
# Figure 12 — signal propagation
# ---------------------------------------------------------------------------

def propagate(sigma_w2, depth, m, d, rng, sigma_b2=0.05, n_pairs=64):
    """Correlation between two random inputs after each layer, and the gradient
    norm ratio across the whole stack.

    Everything is forward-mode: the gradient factor for a tanh network is a
    product of diagonal derivative matrices and weight matrices, whose typical
    scale is what decides vanishing or exploding. Computing it as a product of
    per-layer factors avoids ever building the full Jacobian.
    """
    x1 = rng.normal(size=(n_pairs, d)) / np.sqrt(d)
    x2 = rng.normal(size=(n_pairs, d)) / np.sqrt(d)
    # start correlated so the approach to the fixed point is visible
    x2 = 0.7 * x1 + np.sqrt(1 - 0.7 ** 2) * x2

    W0 = rng.normal(scale=np.sqrt(sigma_w2 / d), size=(m, d))
    h1 = x1 @ W0.T
    h2 = x2 @ W0.T
    h1, h2 = np.tanh(h1), np.tanh(h2)

    corrs, gfac = [], 1.0
    for _ in range(depth):
        W = rng.normal(scale=np.sqrt(sigma_w2 / m), size=(m, m))
        b = rng.normal(scale=np.sqrt(sigma_b2), size=m)
        z1, z2 = h1 @ W.T + b, h2 @ W.T + b
        dphi = 1.0 - np.tanh(z1) ** 2
        # E[chi] for one layer: sigma_w^2 * E[phi'(z)^2]
        gfac *= sigma_w2 * float((dphi ** 2).mean())
        h1, h2 = np.tanh(z1), np.tanh(z2)
        c = ((h1 * h2).sum(1)
             / (np.linalg.norm(h1, axis=1) * np.linalg.norm(h2, axis=1)))
        corrs.append(float(c.mean()))
    return np.array(corrs), gfac


def deep_init(L, m, d, sigma_w2, sigma_b2, rng):
    Ws = [rng.normal(scale=np.sqrt(sigma_w2 / d), size=(m, d))]
    bs = [rng.normal(scale=np.sqrt(sigma_b2), size=m)]
    for _ in range(L - 1):
        Ws.append(rng.normal(scale=np.sqrt(sigma_w2 / m), size=(m, m)))
        bs.append(rng.normal(scale=np.sqrt(sigma_b2), size=m))
    Wo = rng.normal(scale=1.0 / np.sqrt(m), size=(1, m))
    return Ws, bs, Wo


def deep_forward(Ws, bs, Wo, X):
    H, acts = X, []
    for W, b in zip(Ws, bs):
        Z = H @ W.T + b
        H = np.tanh(Z)
        acts.append((Z, H))
    return (H @ Wo.T)[:, 0], acts


def deep_train(Ws, bs, Wo, X, y, lr, steps):
    """Plain gradient descent on a depth-L tanh MLP, backprop written out.

    Nothing here is adaptive: no normalisation, no residual connections, no
    Adam. All three of those exist to defeat the effect being measured, so
    including any of them would hide it.
    """
    n = len(X)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            out, acts = deep_forward(Ws, bs, Wo, X)
            resid = (out - y) / n
            gWo = (resid @ acts[-1][1])[None, :]
            dH = np.outer(resid, Wo[0])
            for i in range(len(Ws) - 1, -1, -1):
                Z, _ = acts[i]
                dZ = dH * (1.0 - np.tanh(Z) ** 2)
                Hprev = X if i == 0 else acts[i - 1][1]
                gW = dZ.T @ Hprev
                gb = dZ.sum(0)
                dH = dZ @ Ws[i]
                Ws[i] -= lr * gW
                bs[i] -= lr * gb
            Wo -= lr * gWo
        out, _ = deep_forward(Ws, bs, Wo, X)
        v = float(np.mean((out - y) ** 2))
    return v if np.isfinite(v) else np.nan


def figure12(m=384, d=16, depth=150, show=48, seed=4):
    sigmas = np.array([0.6, 1.0, 1.4, 1.76, 2.2, 3.0, 4.5])
    curves, gfacs = {}, []
    for sg in sigmas:
        c, g = propagate(sg, depth, m, d, np.random.default_rng(seed + int(sg * 100)))
        curves[sg] = c
        gfacs.append(g)
        print(f"  sigma_w^2 {sg:4.2f}: correlation at depth {depth} {c[-1]:.4f}"
              f"   gradient factor over {depth} layers {g:.3e}")

    # Depth scale: layers for |c - c_inf| to fall by 1/e, with c_inf taken at the
    # bottom of a much deeper stack than the one plotted. Estimating the fixed
    # point from the plotted depth is what made an earlier version put the
    # slowest approach in the chaotic phase rather than at criticality.
    scales = []
    for sg in sigmas:
        c = curves[sg]
        gap = np.abs(c - c[-1])
        thr = gap[0] / np.e
        k = np.flatnonzero(gap <= thr)
        scales.append(int(k[0]) + 1 if len(k) else depth)
    crit = sigmas[int(np.argmax(scales))]
    print(f"  slowest approach at sigma_w^2 = {crit}: {max(scales)} layers, "
          f"against {min(scales)} at the extremes")
    ok = [float(sg) for sg, g in zip(sigmas, gfacs) if 0.1 < g < 10]
    print(f"  gradient factor within 10x of one only for sigma_w^2 in {ok}")

    # Trainability: does the phase diagram predict anything, or only describe
    # initialisation? Plain gradient descent, no normalisation, no residuals.
    depths = [2, 4, 8, 16, 32, 64]
    probe = [1.0, 1.76, 3.0]
    rng = np.random.default_rng(seed + 1)
    Xs = rng.normal(size=(64, d)) / np.sqrt(d)
    ys = np.tanh(2.0 * Xs[:, 0]) * 0.8
    # The learning rate is swept per depth. A fixed rate would measure whether
    # 0.15 happens to be stable at depth L, which is a statement about the
    # gradient scale rather than about trainability, and it reported every deep
    # network as untrainable including the critical one.
    trainable, lrs_t = {}, np.logspace(-3.5, -0.5, 7)
    for sg in probe:
        row = []
        for L in depths:
            best = np.inf
            for lr in lrs_t:
                Ws, bs, Wo = deep_init(L, 64, d, sg, 0.05,
                                       np.random.default_rng(seed + L))
                v = deep_train(Ws, bs, Wo, Xs, ys, lr, 400)
                if np.isfinite(v):
                    best = min(best, v)
            row.append(best if np.isfinite(best) else np.nan)
        trainable[sg] = np.array(row, float)
        txt = " ".join("diverged" if np.isnan(v) else f"{v:.3f}" for v in row)
        print(f"  trainability at sigma_w^2 {sg:4.2f} (best of {len(lrs_t)} rates): "
              f"loss by depth {depths} = {txt}")
    base = float(np.mean(ys ** 2))
    print(f"  predicting the mean would give {base:.3f}; a run at or above that "
          f"has not trained")

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(6.9, 2.6))
    cols = (BLUE, CYAN, GREEN, INK, ORANGE, PINK, PURPLE)
    for sg, c in zip(sigmas, cols):
        ax.plot(np.arange(1, show + 1), curves[sg][:show], "-", color=c, lw=1.6,
                label=f"{sg}")
    ax.set_xlabel("layer")
    ax.set_ylabel("correlation of two inputs")
    ax.set_title("Ordered / critical / chaotic", fontsize=9.5, pad=7)
    ax.legend(frameon=False, fontsize=6.6, ncol=2, title=r"$\sigma_w^2$",
              title_fontsize=6.6)
    tidy(ax)

    ax2.semilogy(sigmas, gfacs, "o-", color=ORANGE, lw=1.9, ms=4.5)
    ax2.axhline(1.0, color=GRAY, ls="--", lw=1.1)
    ax2.set_xlabel(r"$\sigma_w^2$")
    ax2.set_ylabel(f"gradient scale, {depth} layers")
    ax2.set_title("What survives the depth", fontsize=9.5, pad=7)
    tidy(ax2)

    for sg, c in zip(probe, (CYAN, INK, ORANGE)):
        ax3.loglog(depths, trainable[sg], "o-", color=c, lw=1.8, ms=4.5,
                   base=2, label=f"{sg}")
    ax3.axhline(base, color=GRAY, ls="--", lw=1.1)
    ax3.set_xlabel("depth")
    ax3.set_ylabel("best training loss reached")
    ax3.set_title("Does it predict trainability?", fontsize=9.5, pad=7)
    ax3.legend(frameon=False, fontsize=6.8, title=r"$\sigma_w^2$", title_fontsize=6.8)
    tidy(ax3)

    fig.suptitle("Initialisation decides how deep information travels",
                 fontsize=10.5, y=1.04)
    save(fig, "12-signal-prop")
    return sigmas, curves, gfacs, scales, trainable


if __name__ == "__main__":
    for i, fn in enumerate([figure8, figure9, figure10, figure11, figure12], 8):
        print(f"Figure {i}")
        fn()
    print("done")
