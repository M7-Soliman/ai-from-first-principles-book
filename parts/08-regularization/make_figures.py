"""
Figures for Part VIII — Regularization.

Every claim in this part that can be measured, is measured here. The
eigen-shrinkage curve is computed from a real Hessian; the early-stopping
equivalence is a real gradient-descent trajectory compared against a real
ridge path; the bagging variance reduction is measured over hundreds of
resampled fits; the adversarial example is a real gradient attack.

    python3 make_figures.py           # writes into ../../book/src/figures/rg/
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "rg"))
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


# --------------------------------------------------------------------------
# 1. What L2 actually does: shrink each eigen-direction by lam/(lam+a).
# --------------------------------------------------------------------------
def fig_eigen_shrinkage():
    rng = np.random.default_rng(0)
    n, d = 50, 40          # barely more data than parameters
    # a design whose eigenvalues span several orders of magnitude
    scale = np.logspace(1.2, -2.2, d)
    X = rng.normal(size=(n, d)) * scale
    w_true = rng.normal(size=d)
    y = X @ w_true + rng.normal(0, 1.5, n)

    H = X.T @ X                       # the Hessian of the squared-error loss
    lam, Q = np.linalg.eigh(H)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    coords_ols = Q.T @ w_ols          # in the eigenbasis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    for a, c in [(1.0, BLUE), (30.0, ORANGE), (1000.0, GREEN)]:
        w_r = np.linalg.solve(H + a * np.eye(d), X.T @ y)
        ratio = (Q.T @ w_r) / coords_ols
        ax1.semilogx(lam, ratio, "o", ms=3.5, color=c, label=f"$\\alpha$ = {a:g}")
        ax1.semilogx(np.sort(lam), (np.sort(lam) / (np.sort(lam) + a)),
                     "-", lw=1.2, color=c, alpha=0.55)
    ax1.set_xlabel("eigenvalue of $X^\\top X$  (curvature)")
    ax1.set_ylabel("kept fraction")
    ax1.set_title("measured points, predicted curve $\\lambda/(\\lambda+\\alpha)$",
                  fontsize=9, color=GRAY)
    ax1.set_ylim(-0.05, 1.08)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    tidy(ax1)

    alphas = np.logspace(-3, 5, 60)
    err = []
    Xte = rng.normal(size=(2000, d)) * scale
    yte = Xte @ w_true
    for a in alphas:
        w_r = np.linalg.solve(H + a * np.eye(d), X.T @ y)
        err.append(np.mean((Xte @ w_r - yte) ** 2))
    err = np.array(err)
    ax2.loglog(alphas, err, "-", color=PURPLE, lw=1.8)
    best = alphas[np.argmin(err)]
    ax2.axvline(best, color=GRAY, ls="--", lw=1.0)
    ax2.text(best * 1.3, err.max() * 0.5, f"best $\\alpha$ ≈ {best:.0f}",
             fontsize=8, color=GRAY)
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("test error")
    ax2.set_title("and the optimum is strictly interior", fontsize=9, color=GRAY)
    tidy(ax2)

    print(f"    best alpha {best:.2f}; err {err[0]:.3f} -> {err.min():.3f} ({err[0]/err.min():.1f}x better)")
    fig.tight_layout()
    save(fig, "01-eigen-shrinkage")


# --------------------------------------------------------------------------
# 2. L1 versus L2: the coefficient paths, and where sparsity comes from.
# --------------------------------------------------------------------------
def fig_l1_vs_l2():
    rng = np.random.default_rng(3)
    n, d = 80, 8
    X = rng.normal(size=(n, d))
    w_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ w_true + rng.normal(0, 0.4, n)

    def lasso(X, y, a, iters=4000, lr=0.004):
        """Proximal gradient: a gradient step, then soft-threshold."""
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            w -= lr * (X.T @ (X @ w - y)) / len(y)
            w = np.sign(w) * np.maximum(np.abs(w) - lr * a, 0.0)   # the prox
        return w

    alphas = np.logspace(-3, 0.6, 45)
    L2 = np.array([np.linalg.solve(X.T @ X + a * len(y) * np.eye(d), X.T @ y)
                   for a in alphas])
    L1 = np.array([lasso(X, y, a) for a in alphas])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for j in range(d):
        live = w_true[j] != 0
        ax1.semilogx(alphas, L2[:, j], lw=1.5,
                     color=BLUE if live else LIGHT)
        ax2.semilogx(alphas, L1[:, j], lw=1.5,
                     color=ORANGE if live else LIGHT)
    for ax, t in ((ax1, "$L^2$: everything shrinks, nothing vanishes"),
                  (ax2, "$L^1$: the useless ones hit exactly zero")):
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xlabel("$\\alpha$")
        ax.set_title(t, fontsize=9, color=GRAY)
        tidy(ax)
    ax1.set_ylabel("coefficient")

    zero_l1 = (np.abs(L1[-8]) < 1e-8).sum()
    zero_l2 = (np.abs(L2[-8]) < 1e-8).sum()
    print(f"    at a large alpha: L1 has {zero_l1}/8 exact zeros, L2 has {zero_l2}")
    fig.tight_layout()
    save(fig, "02-l1-vs-l2")


# --------------------------------------------------------------------------
# 3. The constrained reading: contours meeting a ball.
# --------------------------------------------------------------------------
def fig_constraint_view():
    A = np.array([[3.0, 1.4], [1.4, 1.0]])
    w_ols = np.array([2.4, 1.6])

    def loss(W1, W2):
        d1, d2 = W1 - w_ols[0], W2 - w_ols[1]
        return A[0, 0]*d1*d1 + 2*A[0, 1]*d1*d2 + A[1, 1]*d2*d2

    g = np.linspace(-1.2, 3.4, 400)
    W1, W2 = np.meshgrid(g, g)
    Z = loss(W1, W2)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, p, title in ((axes[0], 2, "$L^2$: a round ball, tangency anywhere"),
                         (axes[1], 1, "$L^1$: corners on the axes")):
        ax.contour(W1, W2, Z, levels=np.geomspace(0.05, 30, 9),
                   colors=GRAY, linewidths=0.7, alpha=0.65)
        r = 1.0
        # solve the constrained problem numerically on a fine grid
        mask = (np.abs(W1) ** p + np.abs(W2) ** p) ** (1.0 / p) <= r
        Zc = np.where(mask, Z, np.inf)
        i, j = np.unravel_index(np.argmin(Zc), Zc.shape)
        wc = (W1[i, j], W2[i, j])

        th = np.linspace(0, 2 * np.pi, 800)
        if p == 2:
            ax.plot(r*np.cos(th), r*np.sin(th), color=BLUE, lw=1.8)
        else:
            ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], color=ORANGE, lw=1.8)
        ax.plot(*w_ols, "o", color=GRAY, ms=6)
        ax.annotate("unpenalised\noptimum", w_ols, textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=GRAY)
        ax.plot(*wc, "o", color=BLUE if p == 2 else ORANGE, ms=8)
        ax.set_title(title, fontsize=9, color=GRAY)
        ax.set_xlim(-1.2, 3.4); ax.set_ylim(-1.2, 3.4)
        ax.set_aspect("equal")
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        tidy(ax, grid=False)
        print(f"    p={p}: constrained optimum ({wc[0]:.3f}, {wc[1]:.3f})")

    fig.tight_layout()
    save(fig, "03-constraint-view")


# --------------------------------------------------------------------------
# 4. Early stopping IS L2 — two different procedures, one path.
# --------------------------------------------------------------------------
def fig_early_stopping():
    """Gradient descent on a quadratic keeps a fraction 1-(1-eps*lam)^tau of
    each eigen-direction; ridge keeps lam/(lam+alpha). Setting alpha=1/(tau*eps)
    lines the two families up. They are NOT the same curve -- the measured gap
    is reported below and stated in the caption."""
    rng = np.random.default_rng(1)
    n, d = 60, 30
    scale = np.logspace(0.6, -1.4, d)
    X = rng.normal(size=(n, d)) * scale
    w_true = rng.normal(size=d)
    y = X @ w_true + rng.normal(0, 0.25, n)
    H = X.T @ X
    lam, Q = np.linalg.eigh(H)
    b = X.T @ y
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    coords_ols = Q.T @ w_ols

    lr = 0.9 / lam.max()
    checkpoints = [10, 60, 400, 3000]
    w = np.zeros(d)
    traj = {}
    for t in range(1, max(checkpoints) + 1):
        w -= lr * (H @ w - b)
        if t in checkpoints:
            traj[t] = w.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    grid = np.logspace(np.log10(lam.min()), np.log10(lam.max()), 300)
    gaps = []
    for t, c in zip(checkpoints, (BLUE, ORANGE, GREEN, PURPLE)):
        a = 1.0 / (t * lr)
        ax1.semilogx(grid, 1 - (1 - lr * grid) ** t, "-", lw=1.8, color=c,
                     label=f"stop at $\\tau$={t}")
        ax1.semilogx(grid, grid / (grid + a), "--", lw=1.4, color=c, alpha=0.8)
        gaps.append(np.abs((1 - (1 - lr * grid) ** t) - grid / (grid + a)).max())
    ax1.set_xlabel("eigenvalue (curvature)")
    ax1.set_ylabel("fraction of the direction kept")
    ax1.set_title("solid = early stopping, dashed = ridge",
                  fontsize=9, color=GRAY)
    ax1.set_ylim(-0.05, 1.08)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    tidy(ax1)

    for t, c in zip(checkpoints, (BLUE, ORANGE, GREEN, PURPLE)):
        a = 1.0 / (t * lr)
        w_r = np.linalg.solve(H + a * np.eye(d), b)
        ax2.plot(Q.T @ w_r, Q.T @ traj[t], "o", ms=3.5, alpha=0.8, color=c)
    lims = [-3, 3]
    ax2.plot(lims, lims, "--", color=GRAY, lw=1.0)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("ridge coefficient, $\\alpha = 1/(\\tau\\epsilon)$")
    ax2.set_ylabel("gradient descent at step $\\tau$")
    ax2.set_title("real trajectory against real ridge path",
                  fontsize=9, color=GRAY)
    tidy(ax2)

    print(f"    max gap in kept-fraction per checkpoint: "
          + ", ".join(f"{g:.3f}" for g in gaps))
    fig.tight_layout()
    save(fig, "05-early-stopping")


# --------------------------------------------------------------------------
# 5. Noise on the inputs is a penalty on the weights.
# --------------------------------------------------------------------------
def fig_noise_is_penalty():
    rng = np.random.default_rng(4)
    n, d = 120, 12
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + rng.normal(0, 0.3, n)

    sigmas = np.linspace(0.0, 1.4, 12)
    noisy_norm, ridge_norm = [], []
    for s in sigmas:
        # fit on many noisy copies -- this is dataset augmentation with noise
        reps = 60
        Xa = np.vstack([X + rng.normal(0, s, X.shape) for _ in range(reps)])
        ya = np.tile(y, reps)
        w_noise = np.linalg.lstsq(Xa, ya, rcond=None)[0]
        noisy_norm.append(np.linalg.norm(w_noise))
        # the penalty it is predicted to equal: alpha = n * sigma^2
        a = len(y) * s ** 2
        w_ridge = np.linalg.solve(X.T @ X + a * np.eye(d), X.T @ y)
        ridge_norm.append(np.linalg.norm(w_ridge))

    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.plot(sigmas, noisy_norm, "o", ms=5, color=ORANGE,
            label="fit on inputs corrupted with noise $\\sigma$")
    ax.plot(sigmas, ridge_norm, "-", lw=1.8, color=BLUE,
            label="ridge with $\\alpha = n\\sigma^2$, no noise at all")
    ax.set_xlabel("input noise $\\sigma$")
    ax.set_ylabel("$\\|w\\|_2$")
    ax.set_title("two unrelated-looking procedures, one curve",
                 fontsize=9, color=GRAY)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    err = np.max(np.abs(np.array(noisy_norm) - np.array(ridge_norm))
                 / np.array(ridge_norm))
    print(f"    max relative disagreement {err:.3f}")
    fig.tight_layout()
    save(fig, "04-noise-is-penalty")


# --------------------------------------------------------------------------
# 6. Bagging: variance falls with the ensemble, bias does not.
# --------------------------------------------------------------------------
def fig_bagging():
    rng = np.random.default_rng(5)
    grid = np.linspace(0.05, 0.95, 60)   # away from the edges, where
    truth = np.sin(2 * np.pi * grid)     # polynomials blow up

    def one_fit(seed, deg=6, n=40, noise=0.35):
        """Returns (xt, yt, predictions) — training data kept for bootstrap."""
        r = np.random.default_rng(seed)
        xt = r.uniform(0, 1, n)
        yt = np.sin(2 * np.pi * xt) + r.normal(0, noise, n)
        return xt, yt, np.polyval(np.polyfit(xt, yt, deg), grid)

    def boot_fit(xt, yt, seed, deg=6, clip=12.0):
        """Fit on a bootstrap resample; clip extremes from degenerate draws."""
        r = np.random.default_rng(seed)
        idx = r.integers(0, len(xt), len(xt))
        pred = np.polyval(np.polyfit(xt[idx], yt[idx], deg), grid)
        return np.clip(pred, -clip, clip)

    sizes = [1, 2, 5, 10, 25, 50]
    bias2, var, total = [], [], []
    trials = 250
    for k in sizes:
        preds = np.empty((trials, len(grid)))
        for t in range(trials):
            members = [one_fit(1000 * t + i)[2] for i in range(k)]
            preds[t] = np.mean(members, axis=0)
        m = preds.mean(axis=0)
        bias2.append(((m - truth) ** 2).mean())
        var.append(preds.var(axis=0).mean())
        total.append(((preds - truth) ** 2).mean())

    # ------------------------------------------------------------------
    # Bootstrap-resample variance: what §18's pitfall box quotes.
    # Each trial draws one full training set, then measures the variance
    # of (a) a full-data fit, (b) one bootstrap member, (c) a 20-member
    # bootstrap ensemble — across 300 independent training sets.
    # Seeds are well-separated from the independent-sample block above.
    # ------------------------------------------------------------------
    BOOT_TRIALS, N_ENS = 300, 20
    full_v, mem_v, ens_v = [], [], []
    for t in range(BOOT_TRIALS):
        xt, yt, fp = one_fit(500000 + t)
        full_v.append(fp)
        mem_v.append(boot_fit(xt, yt, seed=600000 + t))
        mems = [boot_fit(xt, yt, seed=700000 + t * N_ENS + i)
                for i in range(N_ENS)]
        ens_v.append(np.mean(mems, axis=0))
    var_full = np.array(full_v).var(axis=0).mean()
    var_mem  = np.array(mem_v).var(axis=0).mean()
    var_ens  = np.array(ens_v).var(axis=0).mean()
    boot_ratio = var_mem / var_ens

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax1.loglog(sizes, var, "o-", color=ORANGE, lw=1.8, ms=5, label="variance")
    ax1.loglog(sizes, bias2, "o-", color=BLUE, lw=1.8, ms=5, label="bias$^2$")
    ax1.loglog(sizes, np.array(var)[0] / np.array(sizes), "--",
               color=GRAY, lw=1.0, label="$1/k$")
    ax1.set_xlabel("ensemble size $k$")
    ax1.set_ylabel("error component")
    ax1.set_title("variance falls like $1/k$; bias does not move",
                  fontsize=9, color=GRAY)
    ax1.legend(fontsize=7.5, frameon=False)
    tidy(ax1)

    for k, c in ((1, LIGHT), (25, ORANGE)):
        members = [one_fit(7000 + i)[2] for i in range(k)]
        ax2.plot(grid, np.mean(members, axis=0), color=c, lw=1.8,
                 label=f"k = {k}")
    ax2.plot(grid, truth, "--", color=INK, lw=1.2, label="truth")
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_xlabel("$x$")
    ax2.set_title("one degree-6 fit, versus 25 averaged",
                  fontsize=9, color=GRAY)
    ax2.legend(fontsize=7.5, frameon=False)
    tidy(ax2)

    print(f"    variance {var[0]:.4f} -> {var[-1]:.5f} over k=1..50 "
          f"(ratio {var[0]/var[-1]:.1f}, ideal {sizes[-1]})")
    print(f"    bias^2   {bias2[0]:.4f} -> {bias2[-1]:.4f} (should barely move)")
    print(f"    bootstrap pitfall ({BOOT_TRIALS} trials, k_ens={N_ENS}):")
    print(f"      var_full_data={var_full:.4f}  var_boot_member={var_mem:.4f}"
          f"  var_boot_ens={var_ens:.4f}  ratio={boot_ratio:.1f}x")
    fig.tight_layout()
    save(fig, "06-bagging")


# --------------------------------------------------------------------------
# 7. Dropout: a real network, trained with and without.
# --------------------------------------------------------------------------
def fig_dropout():
    rng = np.random.default_rng(6)

    def make(n, seed):
        r = np.random.default_rng(seed)
        X = r.uniform(-3, 3, size=(n, 2))
        y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 4.0).astype(float)
        flip = r.uniform(size=n) < 0.12          # label noise, to overfit
        y[flip] = 1 - y[flip]
        return X, y

    Xtr, ytr = make(90, 0)
    Xte, yte = make(3000, 1)

    def train(p_drop, steps=6000, h=64, lr=0.06, seed=0):
        r = np.random.default_rng(seed)
        W1 = r.normal(0, 1.0, (h, 2)); b1 = np.zeros(h)
        W2 = r.normal(0, np.sqrt(1 / h), h); b2 = 0.0
        tr_hist, te_hist = [], []
        for t in range(steps):
            z1 = Xtr @ W1.T + b1
            a1 = np.maximum(z1, 0)
            if p_drop > 0:
                mask = (r.uniform(size=a1.shape) > p_drop) / (1 - p_drop)
                a1 = a1 * mask
            z2 = a1 @ W2 + b2
            p = 1 / (1 + np.exp(-z2))
            d2 = (p - ytr) / len(ytr)
            gW2 = a1.T @ d2; gb2 = d2.sum()
            d1 = np.outer(d2, W2) * (z1 > 0)
            if p_drop > 0:
                d1 = d1 * mask
            gW1 = d1.T @ Xtr; gb1 = d1.sum(axis=0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
            if t % 40 == 0:
                for XX, yy, hist in ((Xtr, ytr, tr_hist), (Xte, yte, te_hist)):
                    a = np.maximum(XX @ W1.T + b1, 0)
                    pr = 1 / (1 + np.exp(-(a @ W2 + b2)))
                    hist.append(np.mean((pr > 0.5) != yy))
        return np.array(tr_hist), np.array(te_hist)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    xs = np.arange(0, 6000, 40)
    for p, c, lbl in ((0.0, GRAY, "no dropout"), (0.5, ORANGE, "dropout 0.5")):
        tr, te = train(p)
        ax1.plot(xs, tr, color=c, lw=1.4, ls="--")
        ax1.plot(xs, te, color=c, lw=1.8, label=lbl)
        print(f"    p={p}: final train err {tr[-1]:.3f}  test err {te[-1]:.3f}")
    ax1.set_xlabel("step")
    ax1.set_ylabel("error  (dashed = train)")
    ax1.set_title("dropout raises training error and lowers test error",
                  fontsize=9, color=GRAY)
    ax1.legend(fontsize=7.5, frameon=False)
    tidy(ax1)

    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    gaps, tests = [], []
    for p in ps:                       # average three seeds -- one is noisy
        runs = [train(p, seed=s) for s in (0, 1, 2)]
        te_m = float(np.mean([r[1][-1] for r in runs]))
        tr_m = float(np.mean([r[0][-1] for r in runs]))
        gaps.append(te_m - tr_m); tests.append(te_m)
        print(f"      p={p:.1f}  train {tr_m:.3f}  test {te_m:.3f}")
    ax2.plot(ps, tests, "o-", color=ORANGE, lw=1.8, ms=5, label="test error")
    ax2.plot(ps, gaps, "o-", color=BLUE, lw=1.8, ms=5, label="generalisation gap")
    ax2.set_xlabel("dropout probability")
    ax2.set_ylabel("error")
    ax2.set_title("and too much of it underfits", fontsize=9, color=GRAY)
    ax2.legend(fontsize=7.5, frameon=False)
    tidy(ax2)
    print(f"    best dropout p = {ps[int(np.argmin(tests))]}")

    fig.tight_layout()
    save(fig, "07-dropout")


# --------------------------------------------------------------------------
# 8. An adversarial example: a real gradient attack on a real classifier.
# --------------------------------------------------------------------------
def fig_adversarial():
    rng = np.random.default_rng(7)
    n, d = 400, 64
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = (X @ w_true > 0).astype(float)

    # fit a logistic model properly
    w = np.zeros(d)
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= 0.5 * X.T @ (p - y) / n
    acc = np.mean(((X @ w) > 0) == (y > 0))

    eps = np.linspace(0, 0.35, 25)
    adv_acc, rand_acc = [], []
    for e in eps:
        # fast gradient sign: step every input along sign(grad) of its loss
        grad_sign = np.sign(np.outer(1 / (1 + np.exp(-(X @ w))) - y, w))
        Xa = X + e * grad_sign
        adv_acc.append(np.mean(((Xa @ w) > 0) == (y > 0)))
        Xr = X + e * np.sign(rng.normal(size=X.shape))
        rand_acc.append(np.mean(((Xr @ w) > 0) == (y > 0)))

    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.plot(eps, adv_acc, "o-", color=PINK, lw=1.8, ms=4,
            label="perturbation chosen by the gradient")
    ax.plot(eps, rand_acc, "o-", color=GRAY, lw=1.8, ms=4,
            label="random perturbation of the same size")
    ax.axhline(0.5, color=LIGHT, lw=1.0, ls="--")
    ax.set_xlabel("perturbation size $\\epsilon$ per input coordinate")
    ax.set_ylabel("accuracy")
    ax.set_title(f"clean accuracy {acc:.3f} — and the two directions are "
                 "not remotely equivalent", fontsize=9, color=GRAY)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    i = int(np.argmax(np.array(adv_acc) < 0.55))
    print(f"    clean {acc:.3f}; adversarial reaches chance at eps={eps[i]:.3f} "
          f"where random noise leaves {rand_acc[i]:.3f}")
    fig.tight_layout()
    save(fig, "08-adversarial")


if __name__ == "__main__":
    print("Part VIII — regularization figures")
    fig_eigen_shrinkage()
    fig_l1_vs_l2()
    fig_constraint_view()
    fig_early_stopping()
    fig_noise_is_penalty()
    fig_bagging()
    fig_dropout()
    fig_adversarial()
    print("done ->", OUT)
