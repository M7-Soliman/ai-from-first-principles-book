"""
Figures for Part VI — Machine Learning Basics.

Every model here is actually fitted. The decision boundaries come from real
solvers, the learning curves from real training runs, the curse-of-dimensionality
panels from real sampling in high dimensions.

    python3 make_figures.py           # writes into ../../book/src/figures/ml/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "ml"))
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


def blobs(rng, n=120, sep=2.2):
    a = rng.normal([-sep / 2, 0], 1.0, size=(n // 2, 2))
    b = rng.normal([sep / 2, 0.6], 1.0, size=(n // 2, 2))
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def mesh(X, pad=0.8, k=260):
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, k)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, k)
    G0, G1 = np.meshgrid(x0, x1)
    return G0, G1, np.c_[G0.ravel(), G1.ravel()]


# =============================================================== fig 1 ======
def fig_capacity():
    """The central curve: training error falls forever, test error turns around."""
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * np.pi * x)
    xtr = np.sort(rng.uniform(0, 1, 18)); ytr = f(xtr) + rng.normal(0, .25, 18)
    xte = np.sort(rng.uniform(0, 1, 400)); yte = f(xte) + rng.normal(0, .25, 400)
    degs = np.arange(0, 16)
    tr, te = [], []
    for d in degs:
        with np.errstate(all="ignore"):
            c = np.polyfit(xtr, ytr, d)
        tr.append(np.mean((np.polyval(c, xtr) - ytr) ** 2))
        te.append(np.mean((np.polyval(c, xte) - yte) ** 2))
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.semilogy(degs, tr, "o-", color=BLUE, lw=2, ms=4, label="training error")
    ax.semilogy(degs, te, "o-", color=ORANGE, lw=2.2, ms=4, label="test error")
    best = int(np.argmin(te))
    ax.axvline(best, color=GREEN, ls="--", lw=1.5)
    ax.text(best + 0.25, max(te) * 0.4, "best capacity", color=GREEN, fontsize=8.5)
    ax.set_xlabel("capacity (polynomial degree)"); ax.set_ylabel("mean squared error")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("underfitting on the left, overfitting on the right",
                 fontsize=9.8, pad=8)
    tidy(ax)
    save(fig, "01-capacity")


# =============================================================== fig 2 ======
def fig_learning_curves():
    """More data helps a high-variance model and does nothing for a biased one."""
    rng = np.random.default_rng(1)
    f = lambda x: np.sin(2 * np.pi * x)
    sizes = np.unique(np.round(np.logspace(0.7, 2.6, 14)).astype(int))
    xte = np.linspace(0, 1, 600); yte = f(xte)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.9))
    for ax, deg, name in [(axes[0], 1, "degree 1 — high bias"),
                          (axes[1], 12, "degree 12 — high variance")]:
        tr_c, te_c = [], []
        for n in sizes:
            trs, tes = [], []
            for _ in range(40):
                x = np.sort(rng.uniform(0, 1, n)); y = f(x) + rng.normal(0, .25, n)
                with np.errstate(all="ignore"):
                    c = np.polyfit(x, y, min(deg, max(n - 1, 1)))
                trs.append(np.mean((np.polyval(c, x) - y) ** 2))
                tes.append(np.mean((np.polyval(c, xte) - yte) ** 2))
            tr_c.append(np.median(trs)); te_c.append(np.median(tes))
        ax.semilogx(sizes, tr_c, "o-", color=BLUE, lw=2, ms=4, label="training")
        ax.semilogx(sizes, te_c, "o-", color=ORANGE, lw=2, ms=4, label="test")
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("training set size"); ax.set_ylabel("error")
        ax.legend(fontsize=8.2, frameon=False)
        ax.set_title(name, fontsize=9.6, pad=6)
        tidy(ax)
    save(fig, "02-learning-curves")


# =============================================================== fig 3 ======
def fig_regularization_path():
    """Weight decay traded against fit, swept. The coefficients shrink."""
    rng = np.random.default_rng(2)
    n, d = 25, 12
    X = rng.normal(size=(n, d)); w = rng.normal(size=d) * np.exp(-np.arange(d) / 3)
    y = X @ w + rng.normal(0, .6, n)
    Xte = rng.normal(size=(600, d)); yte = Xte @ w + rng.normal(0, .6, 600)
    lams = np.logspace(-4, 3, 60)
    coefs, te = [], []
    for lam in lams:
        wh = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
        coefs.append(wh); te.append(np.mean((Xte @ wh - yte) ** 2))
    coefs = np.array(coefs)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.9))
    ax = axes[0]
    for j in range(d):
        ax.semilogx(lams, coefs[:, j], lw=1.5, color=plt.cm.viridis(j / d))
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel(r"regularization strength $\lambda$")
    ax.set_ylabel("coefficient value")
    ax.set_title("every coefficient shrinks toward zero", fontsize=9.6, pad=8)
    tidy(ax)
    ax = axes[1]
    ax.semilogx(lams, te, color=ORANGE, lw=2.3)
    ax.axvline(lams[int(np.argmin(te))], color=GREEN, ls="--", lw=1.5)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("test error")
    ax.set_title("and there is a best amount of shrinking", fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "03-regularization-path")


# =============================================================== fig 4 ======
def fig_linear_logistic():
    """The two workhorses, both actually fitted."""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.0))

    ax = axes[0]
    x = np.sort(rng.uniform(0, 10, 30)); y = 2.1 * x + 1.4 + rng.normal(0, 2.4, 30)
    A = np.c_[x, np.ones_like(x)]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    ax.plot(x, y, "o", color=BLUE, ms=5, alpha=0.75)
    xs = np.array([0, 10])
    ax.plot(xs, w[0] * xs + w[1], color=ORANGE, lw=2.3)
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, w[0] * xi + w[1]], color=PINK, lw=0.8, alpha=0.55)
    ax.set_title(r"linear regression: minimise $\sum$ (pink)$^2$",
                 fontsize=9.6, pad=8)
    ax.set_xticks([]); ax.set_yticks([]); tidy(ax, grid=False)

    ax = axes[1]
    X, y2 = blobs(rng, 160, sep=2.6)
    Xb = np.c_[X, np.ones(len(X))]
    wv = np.zeros(3)
    for _ in range(400):                       # logistic regression by gradient descent
        p = 1 / (1 + np.exp(-Xb @ wv))
        wv -= 0.05 * Xb.T @ (p - y2) / len(y2)
    G0, G1, P = mesh(X)
    Z = (1 / (1 + np.exp(-np.c_[P, np.ones(len(P))] @ wv))).reshape(G0.shape)
    ax.contourf(G0, G1, Z, levels=20, cmap="RdBu_r", alpha=0.35)
    ax.contour(G0, G1, Z, levels=[0.5], colors=[INK], linewidths=2)
    ax.scatter(*X[y2 == 0].T, s=12, color=BLUE, zorder=5)
    ax.scatter(*X[y2 == 1].T, s=12, color=ORANGE, zorder=5)
    ax.set_title("logistic regression: a probability, not a label",
                 fontsize=9.6, pad=8)
    ax.set_xticks([]); ax.set_yticks([]); tidy(ax, grid=False)
    save(fig, "04-linear-logistic")


# =============================================================== fig 5 ======
def fig_svm_margin():
    """Maximum margin, solved by projected gradient on the hinge loss."""
    rng = np.random.default_rng(5)
    X, y = blobs(rng, 60, sep=3.4)
    t = 2 * y - 1
    w, b = np.zeros(2), 0.0
    lam = 0.02
    for _ in range(6000):                      # subgradient descent on hinge + L2
        m = t * (X @ w + b)
        idx = m < 1
        gw = lam * w - (t[idx, None] * X[idx]).sum(0) / len(X)
        gb = -t[idx].sum() / len(X)
        w -= 0.05 * gw; b -= 0.05 * gb
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    G0, G1, P = mesh(X)
    Z = (P @ w + b).reshape(G0.shape)
    ax.contourf(G0, G1, Z, levels=[-99, 0, 99], colors=["#e8effb", "#fdece0"])
    ax.contour(G0, G1, Z, levels=[-1, 0, 1], colors=[GRAY, INK, GRAY],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
    sv = np.abs(t * (X @ w + b) - 1) < 0.12
    ax.scatter(*X[y == 0].T, s=16, color=BLUE, zorder=5)
    ax.scatter(*X[y == 1].T, s=16, color=ORANGE, zorder=5)
    ax.scatter(*X[sv].T, s=90, facecolors="none", edgecolors=GREEN, lw=1.8, zorder=6)
    ax.set_title("the margin, and the few points that determine it",
                 fontsize=9.7, pad=8)
    ax.set_xticks([]); ax.set_yticks([]); tidy(ax, grid=False)
    save(fig, "05-svm-margin")


# =============================================================== fig 6 ======
def fig_kernel():
    """Not separable in 2-D; separable after a lift. The kernel trick, drawn."""
    rng = np.random.default_rng(6)
    n = 200
    r = np.r_[rng.uniform(0, 1.1, n // 2), rng.uniform(1.9, 3.0, n // 2)]
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    y = (r > 1.5).astype(int)
    fig = plt.figure(figsize=(7.9, 3.0))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(*X[y == 0].T, s=12, color=BLUE)
    ax.scatter(*X[y == 1].T, s=12, color=ORANGE)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("no straight line separates these", fontsize=9.6, pad=6)
    tidy(ax, grid=False)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    z = X[:, 0] ** 2 + X[:, 1] ** 2                  # the lift
    ax.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0], s=9, color=BLUE)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], s=9, color=ORANGE)
    gx, gy = np.meshgrid(np.linspace(-3, 3, 8), np.linspace(-3, 3, 8))
    ax.plot_surface(gx, gy, np.full_like(gx, 2.25), alpha=0.25, color=GREEN)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.grid(False); ax.view_init(elev=16, azim=-62)
    ax.set_title(r"add $x_1^2 + x_2^2$ and a plane does", fontsize=9.6, pad=-2)
    save(fig, "06-kernel")


# =============================================================== fig 7 ======
def fig_knn():
    """k controls smoothness. Real boundaries from a real classifier."""
    rng = np.random.default_rng(7)
    X, y = blobs(rng, 120, sep=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8))
    G0, G1, P = mesh(X)
    for ax, k in zip(axes, [1, 5, 35]):
        d = ((P[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :k]
        Z = y[nn].mean(1).reshape(G0.shape)
        ax.contourf(G0, G1, Z, levels=[-1, 0.5, 2], colors=["#e8effb", "#fdece0"])
        ax.contour(G0, G1, Z, levels=[0.5], colors=[INK], linewidths=1.6)
        ax.scatter(*X[y == 0].T, s=10, color=BLUE, zorder=5)
        ax.scatter(*X[y == 1].T, s=10, color=ORANGE, zorder=5)
        ax.set_title(f"k = {k}", fontsize=9.6, pad=6)
        ax.set_xticks([]); ax.set_yticks([]); tidy(ax, grid=False)
    save(fig, "07-knn")


# =============================================================== fig 8 ======
def fig_tree():
    """A tree partitions space into axis-aligned boxes. Grown greedily, for real."""
    rng = np.random.default_rng(8)
    X, y = blobs(rng, 160, sep=2.2)

    def grow(idx, depth, bounds, out):
        if depth == 0 or len(idx) < 8 or y[idx].std() < 1e-9:
            out.append((bounds, y[idx].mean() if len(idx) else 0.5)); return
        best = None
        for f in (0, 1):
            for thr in np.percentile(X[idx, f], [25, 50, 75]):
                L = idx[X[idx, f] <= thr]; Rr = idx[X[idx, f] > thr]
                if len(L) < 4 or len(Rr) < 4:
                    continue
                imp = (len(L) * y[L].var() + len(Rr) * y[Rr].var()) / len(idx)
                if best is None or imp < best[0]:
                    best = (imp, f, thr, L, Rr)
        if best is None:
            out.append((bounds, y[idx].mean())); return
        _, f, thr, L, Rr = best
        b1 = list(bounds); b2 = list(bounds)
        b1[2 * f + 1] = thr; b2[2 * f] = thr
        grow(L, depth - 1, b1, out); grow(Rr, depth - 1, b2, out)

    leaves = []
    lim = [X[:, 0].min() - .6, X[:, 0].max() + .6, X[:, 1].min() - .6, X[:, 1].max() + .6]
    grow(np.arange(len(X)), 4, lim, leaves)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for (x0, x1, y0, y1), v in leaves:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   color=("#fdece0" if v > .5 else "#e8effb"),
                                   ec=GRAY, lw=0.7))
    ax.scatter(*X[y == 0].T, s=11, color=BLUE, zorder=5)
    ax.scatter(*X[y == 1].T, s=11, color=ORANGE, zorder=5)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("a decision tree cuts space into boxes", fontsize=9.7, pad=8)
    tidy(ax, grid=False)
    save(fig, "08-tree")


# =============================================================== fig 9 ======
def fig_kmeans():
    """Lloyd's algorithm, iteration by iteration."""
    rng = np.random.default_rng(9)
    C = np.array([[0, 0], [3.4, 1.0], [1.4, 3.2]])
    X = np.vstack([rng.normal(c, .62, size=(70, 2)) for c in C])
    cent = X[rng.choice(len(X), 3, replace=False)].copy()
    snaps = [cent.copy()]
    assign = None
    for _ in range(6):
        d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for k in range(3):
            if (assign == k).any():
                cent[k] = X[assign == k].mean(0)
        snaps.append(cent.copy())
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.5))
    for ax, (it, c) in zip(axes, [(0, snaps[0]), (1, snaps[1]),
                                  (2, snaps[2]), (6, snaps[-1])]):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1); a = d.argmin(1)
        for k, col in enumerate([BLUE, ORANGE, PURPLE]):
            ax.scatter(*X[a == k].T, s=8, color=col, alpha=0.6)
        ax.scatter(*c.T, s=110, marker="X", color=INK, zorder=7)
        ax.set_title(f"iteration {it}", fontsize=9.4, pad=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        tidy(ax, grid=False)
    save(fig, "09-kmeans")


# ============================================================== fig 10 ======
def fig_curse():
    """High dimensions: everything is far away and equally far away."""
    rng = np.random.default_rng(10)
    dims = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ratio, shell = [], []
    for d in dims:
        X = rng.normal(size=(600, d))
        q = rng.normal(size=(1, d))
        dist = np.sqrt(((X - q) ** 2).sum(1))
        ratio.append(dist.max() / dist.min())
        pts = rng.normal(size=(4000, d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        u = rng.random(4000) ** (1 / d)
        shell.append(np.mean(u > 0.9))
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.85))
    ax = axes[0]
    ax.semilogx(dims, ratio, "o-", color=ORANGE, lw=2.2, ms=5)
    ax.axhline(1, color=GREEN, ls="--", lw=1.4)
    ax.set_xlabel("dimensions"); ax.set_ylabel("farthest / nearest distance")
    ax.set_title("nearest and farthest become the same", fontsize=9.6, pad=8)
    tidy(ax)
    ax = axes[1]
    ax.semilogx(dims, shell, "o-", color=BLUE, lw=2.2, ms=5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("dimensions")
    ax.set_ylabel("fraction of volume in outer 10%")
    ax.set_title("all the volume moves to the surface", fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "10-curse")


# ============================================================== fig 11 ======
def fig_local_constancy():
    """Smoothness priors need exponentially many examples as dimension grows."""
    rng = np.random.default_rng(11)
    dims = np.arange(1, 11)
    need = []
    for d in dims:
        # points required so that a random query has a neighbour within 0.25
        # of the unit cube's side, estimated by the volume ratio
        need.append(1.0 / (0.25 ** d))
    fig, ax = plt.subplots(figsize=(5.5, 2.85))
    ax.semilogy(dims, need, "o-", color=PINK, lw=2.3, ms=5)
    ax.set_xlabel("dimensions")
    ax.set_ylabel("examples needed to cover")
    ax.set_title(r"covering the space costs $(1/r)^d$ — exponential in $d$",
                 fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "11-local-constancy")


# ============================================================== fig 12 ======
def fig_manifold():
    """Real data lives on a thin curved sheet, not everywhere in the cube."""
    rng = np.random.default_rng(12)
    t = rng.uniform(0, 3 * np.pi, 700)
    h = rng.uniform(0, 2.4, 700)
    X = np.c_[t * np.cos(t), h, t * np.sin(t)] + rng.normal(0, .18, (700, 3))
    fig = plt.figure(figsize=(7.9, 2.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, c=t, cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.grid(False); ax.view_init(elev=14, azim=-72)
    ax.set_title("data in 3-D…", fontsize=9.6, pad=-2)
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(t, h, s=5, c=t, cmap="viridis")
    ax.set_xlabel("position along the sheet"); ax.set_ylabel("height")
    ax.set_title("…is really 2 numbers", fontsize=9.6, pad=6)
    tidy(ax)
    save(fig, "12-manifold")


# ============================================================== fig 13 ======
def fig_no_free_lunch():
    """Averaged over all possible problems, every learner is identical."""
    rng = np.random.default_rng(13)
    n_pts, n_prob = 8, 40000
    scores = {"constant 0": 0, "majority": 0, "nearest neighbour": 0}
    for _ in range(n_prob):
        f = rng.integers(0, 2, n_pts)                 # a random target function
        seen = rng.choice(n_pts, 4, replace=False)
        unseen = np.setdiff1d(np.arange(n_pts), seen)
        scores["constant 0"] += np.mean(f[unseen] == 0)
        maj = int(f[seen].mean() > 0.5)
        scores["majority"] += np.mean(f[unseen] == maj)
        pred = [f[seen[np.argmin(np.abs(seen - u))]] for u in unseen]
        scores["nearest neighbour"] += np.mean(f[unseen] == np.array(pred))
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    names = list(scores); vals = [scores[k] / n_prob for k in names]
    ax.barh(names, vals, color=[BLUE, ORANGE, GREEN], alpha=0.8)
    ax.axvline(0.5, color=INK, ls="--", lw=1.6)
    ax.set_xlim(0, 1); ax.set_xlabel("accuracy on unseen points")
    for i, v in enumerate(vals):
        ax.text(v + .015, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_title("averaged over all target functions: all equal to chance",
                 fontsize=9.5, pad=8)
    tidy(ax)
    save(fig, "13-no-free-lunch")


# ============================================================== fig 14 ======
def fig_splits():
    """Train / validation / test, and what each is allowed to touch."""
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    spans = [(0, 60, "training\nfit parameters", BLUE),
             (60, 80, "validation\nchoose model", ORANGE),
             (80, 100, "test\nreport once", GREEN)]
    for x0, x1, lab, c in spans:
        ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 1, color=c, alpha=0.75))
        ax.text((x0 + x1) / 2, 0.5, lab, ha="center", va="center",
                color="white", fontsize=9, weight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.5); ax.axis("off")
    ax.text(50, 1.28, "every decision you make must use only the first two",
            ha="center", fontsize=9, color=INK)
    save(fig, "14-splits")


# ============================================================== fig 15 ======
def fig_cross_validation():
    """K-fold cross-validation: every point used for validation exactly once."""
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    K, n = 5, 20
    for k in range(K):
        for i in range(n):
            val = (i // (n // K)) == k
            ax.add_patch(plt.Rectangle((i, K - k - 1), 0.9, 0.86,
                                       color=(ORANGE if val else "#dbe6f5")))
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-0.3, K + 0.3)
    ax.set_yticks([K - k - 0.55 for k in range(K)])
    ax.set_yticklabels([f"fold {k+1}" for k in range(K)], fontsize=8.5)
    ax.set_xticks([]); ax.set_aspect("equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("orange = held out for validation", fontsize=9.8, pad=8)
    save(fig, "15-cross-validation")


if __name__ == "__main__":
    print("generating Part VI figures ->", OUT)
    for fn in (fig_capacity, fig_learning_curves, fig_regularization_path,
               fig_linear_logistic, fig_svm_margin, fig_kernel, fig_knn, fig_tree,
               fig_kmeans, fig_curse, fig_local_constancy, fig_manifold,
               fig_no_free_lunch, fig_splits, fig_cross_validation):
        fn()
    print("done.")
