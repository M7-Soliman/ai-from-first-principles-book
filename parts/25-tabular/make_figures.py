"""
Figures for Part XXV — Structured and Tabular Learning.

Everything is built on `trees.py` rather than imported from a library, so every
number here can be regenerated from this repository alone.

    python3 make_figures.py

Runtime: about six minutes, most of it in figures 6 and 7.
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trees import Tree, Forest, GBM, fit_bins, apply_bins

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "tb"))
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


def tabular(n=4000, d=12, noise=0.4, seed=0):
    """A generating process with the structure tabular data actually has:
    interactions, a threshold, an irrelevant majority of columns."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, d))
    y = (2.0 * (X[:, 0] > 0.5)                      # a threshold
         + 1.5 * X[:, 1] * X[:, 2]                  # an interaction
         + np.sin(3 * X[:, 3])                      # a smooth term
         + 0.8 * X[:, 4]
         + noise * r.standard_normal(n))
    return X, y


# --------------------------------------------------------- 1. a single tree

def fig_depth():
    """A tree's capacity is its depth, and the classical curve is visible in a
    way it rarely is for a network: train error falls to zero, test error turns
    around, and the turn is at a depth you can point to."""
    print("\n[1] one tree, and where its depth stops helping")
    X, y = tabular(4000, seed=0)
    Xtr, ytr, Xte, yte = X[:2500], y[:2500], X[2500:], y[2500:]
    edges = fit_bins(Xtr, 64)
    Btr, Bte = apply_bins(Xtr, edges), apply_bins(Xte, edges)

    depths = list(range(1, 17))
    tr_e, te_e, leaves = [], [], []
    for d in depths:
        t = Tree(max_depth=d, min_samples=1, n_bins=64).fit(Btr, ytr, None, 65)
        tr_e.append(float(np.mean((t.predict(Btr) - ytr) ** 2)))
        te_e.append(float(np.mean((t.predict(Bte) - yte) ** 2)))
        leaves.append(t.n_leaves())
        if d in (1, 4, 8, 12, 16):
            print(f"    depth {d:2d}: leaves {leaves[-1]:5d}  train {tr_e[-1]:.4f}  test {te_e[-1]:.4f}")
    best = depths[int(np.argmin(te_e))]
    print(f"    test error is lowest at depth {best}, and rises "
          f"{te_e[-1]/min(te_e):.2f}x by depth {depths[-1]}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(depths, tr_e, "o-", color=BLUE, lw=1.9, ms=4, label="training")
    ax.plot(depths, te_e, "o-", color=ORANGE, lw=1.9, ms=4, label="test")
    ax.axvline(best, color=GREEN, lw=1.2, ls="--")
    ax.text(best + 0.3, max(te_e) * 0.9, f"best: depth {best}", fontsize=8, color=GREEN)
    ax.set_xlabel("maximum depth"); ax.set_ylabel("mean squared error")
    ax.set_title("One tree", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogy(depths, leaves, "o-", color=PURPLE, lw=1.9, ms=4)
    ax.set_xlabel("maximum depth"); ax.set_ylabel("leaves")
    ax.set_title("Capacity, counted", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "01-tree-depth")


# ------------------------------------------------------------- 2. averaging

def fig_bagging():
    """Two things about averaging that measure cleanly: the error falls as trees
    are added, and offering each split fewer features makes the trees less alike.

    What is NOT here is the step from the second to lower ensemble error. See
    fig_subsampling_illus.
    """
    print("\n[2] averaging, and what subsampling does to tree agreement")
    fracs = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
    REPS = 6
    C = []
    for ff in fracs:
        c1 = []
        for rep in range(REPS):
            X, y = tabular(4000, seed=200 + rep)
            Xtr, ytr, Xte = X[:2500], y[:2500], X[2500:]
            f = Forest(n_trees=40, max_depth=12, min_samples=2,
                       feature_fraction=ff, seed=rep).fit(Xtr, ytr)
            Bte = apply_bins(Xte, f.edges)
            P = np.array([t.predict(Bte) for t in f.trees])
            M = np.corrcoef(P)
            c1.append(float(M[~np.eye(len(M), dtype=bool)].mean()))
        C.append(np.array(c1))
        print(f"    features {ff:4.2f}:  mean pairwise correlation "
              f"{C[-1].mean():.3f} +/- {C[-1].std(ddof=1):.3f}")
    cm = np.array([c.mean() for c in C])
    print(f"    monotone in the feature fraction: "
          f"{all(cm[i] < cm[i+1] for i in range(len(cm)-1))}")

    curves = {}
    for nt, lab in ((60, "60 trees"),):
        X, y = tabular(4000, seed=200)
        Xtr, ytr, Xte, yte = X[:2500], y[:2500], X[2500:], y[2500:]
        f = Forest(n_trees=nt, max_depth=12, min_samples=2, seed=0).fit(Xtr, ytr)
        curves[lab] = [float(np.mean((p - yte) ** 2)) for p in f.staged_predict(Xte)]
    e = curves["60 trees"]
    print(f"    averaging: one tree {e[0]:.3f} -> 60 trees {e[-1]:.3f} "
          f"({e[0]/e[-1]:.2f}x)")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(range(1, len(e) + 1), e, color=GREEN, lw=1.9)
    ax.set_xlabel("trees averaged"); ax.set_ylabel("test mean squared error")
    ax.set_title("Error falls as trees are added", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.errorbar(fracs, cm, yerr=[c.std(ddof=1) for c in C], fmt="o-",
                color=PURPLE, lw=1.9, ms=5, capsize=3)
    ax.set_xlabel("fraction of features offered at each split")
    ax.set_ylabel("mean correlation between trees")
    ax.set_title("Offering fewer features makes them differ",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "02-bagging")


def fig_subsampling_illus():
    """Illustrative. The shape reported in the literature for the trade between
    decorrelation and individual tree quality.

    Not measured here: neither generator in this part separates it. On twelve
    features with five informative, and on sixty correlated informative
    features, the paired difference between the best subsampling fraction and
    using every feature is within noise at ten repetitions. Drawn with no
    numbers on either axis.
    """
    print("\n[2b] the subsampling trade — illustrative, no numbers")
    x = np.linspace(0.05, 1.0, 200)
    single = 1.0 + 0.9 * np.exp(-3.2 * x)          # tree quality: worse when starved
    corr = 0.18 + 0.62 * x                          # agreement: rises with features
    ensemble = single * (corr + (1 - corr) / 40.0)  # the averaging identity, in shape

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.plot(x, ensemble / ensemble.max(), color=GREEN, lw=2.2, label="ensemble error")
    ax.plot(x, single / single.max(), color=GRAY, lw=1.5, ls="--",
            label="a single tree's error")
    ax.plot(x, corr / corr.max(), color=PURPLE, lw=1.5, ls=":",
            label="agreement between trees")
    j = int(np.argmin(ensemble))
    ax.plot([x[j]], [ensemble[j] / ensemble.max()], "o", color=GREEN, ms=7, zorder=5)
    ax.annotate("the reported optimum", (x[j], ensemble[j] / ensemble.max()),
                textcoords="offset points", xytext=(14, 16), fontsize=8.5, color=GREEN)
    ax.set_xlabel("fraction of features offered at each split $\longrightarrow$")
    ax.set_ylabel("relative")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Two effects pulling opposite ways", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "03-subsampling-illus")


# ------------------------------------------------------------- 4. boosting

def fig_boosting():
    """Boosting fits the residual, so its test error falls with every tree until
    it does not. Shrinkage decides where that turn is, and the trade is the
    number of trees you have to pay for."""
    print("\n[4] boosting, and what shrinkage buys")
    X, y = tabular(4000, seed=3)
    Xtr, ytr, Xte, yte = X[:2500], y[:2500], X[2500:], y[2500:]

    curves = {}
    for lr, col in ((1.0, ORANGE), (0.3, BLUE), (0.1, GREEN), (0.03, PURPLE)):
        g = GBM(n_trees=400, lr=lr, max_depth=3, seed=4).fit(Xtr, ytr)
        errs = [float(np.mean((F - yte) ** 2)) for F in g.staged_decision(Xte)]
        curves[lr] = (errs, col)
        k = int(np.argmin(errs)) + 1
        conv = "converged" if k < 380 else "STILL IMPROVING at the budget"
        print(f"    lr {lr:<5}: best test mse {min(errs):.4f} after {k:3d} trees"
              f"   (at 400 trees {errs[-1]:.4f})  [{conv}]")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for lr, (e, col) in curves.items():
        ax.plot(range(1, len(e) + 1), e, color=col, lw=1.7, label=f"learning rate {lr}")
    ax.set_xscale("log")
    ax.set_xlabel("trees"); ax.set_ylabel("test mean squared error")
    ax.set_ylim(0, max(curves[1.0][0][:5]) * 1.05)
    ax.set_title("Each tree fits what is left", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    lrs = sorted(curves)
    best = [min(curves[l][0]) for l in lrs]
    ntree = [int(np.argmin(curves[l][0])) + 1 for l in lrs]
    ax.semilogx(lrs, best, "o-", color=BLUE, lw=1.9, ms=5)
    ax.set_xlabel("learning rate"); ax.set_ylabel("best error within 400 trees", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    ax2.semilogx(lrs, ntree, "s--", color=ORANGE, lw=1.6, ms=4)
    ax2.set_ylabel("trees used", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=8.5)
    ax.set_title("At a fixed budget of 400 trees", fontsize=10.5, color=INK)
    tidy(ax, grid=True)
    save(fig, "04-boosting")


# ------------------------------------------------------- 5. histogram splits

def fig_binning():
    """Exact split-finding considers every distinct value of every feature.
    Binning considers a fixed number of thresholds instead. Measure what the
    approximation costs in accuracy, and count what it saves — the count is
    exact arithmetic, whereas a wall-clock comparison in this implementation
    would be measuring Python rather than the algorithm."""
    print("\n[5] the approximation that made boosted trees fast")
    bins = [4, 8, 16, 32, 64, 128, 255]
    N_TRAIN = 4000
    errs = []
    for nb in bins:
        reps = []
        for sd in range(5):                     # the first version of this figure
            X, y = tabular(6000, seed=50 + sd)  # used one seed and reported noise
            Xtr, ytr, Xte, yte = X[:4000], y[:4000], X[4000:], y[4000:]
            g = GBM(n_trees=150, lr=0.1, max_depth=4, n_bins=nb, seed=6).fit(Xtr, ytr)
            reps.append(float(np.mean((g.predict(Xte) - yte) ** 2)))
        errs.append(float(np.mean(reps)))
        print(f"    {nb:3d} bins:  test mse {errs[-1]:.4f}   "
              f"(mean of 5 seeds, sd {np.std(reps):.4f})")
    b_best = bins[int(np.argmin(errs))]
    e_best = min(errs)
    print(f"    best at {b_best} bins ({e_best:.4f}); the finest binning is "
          f"{errs[-1]:.4f}, which is WORSE")
    print(f"    so binning is not only a speed approximation — past a point the")
    print(f"    coarser grid is also acting as a regulariser on the split choice.")
    print(f"    candidate splits per feature per node: {b_best} against {N_TRAIN} "
          f"for exact search — a factor of {N_TRAIN/b_best:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.semilogx(bins, errs, "o-", color=BLUE, lw=1.9, ms=5, base=2)
    ax.axvline(b_best, color=GREEN, lw=1.2, ls="--")
    ax.text(b_best * 1.15, max(errs) * 0.85, f"best: {b_best} bins",
            fontsize=8, color=GREEN)
    ax.set_xlabel("bins per feature"); ax.set_ylabel("test mean squared error")
    ax.set_title("What the approximation costs", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.loglog(bins, bins, "o-", color=ORANGE, lw=1.9, ms=5, base=2,
              label="binned: thresholds per feature")
    ax.axhline(N_TRAIN, color=INK, lw=1.4, ls="--", label=f"exact: {N_TRAIN} values")
    ax.set_xlabel("bins per feature")
    ax.set_ylabel("candidate splits evaluated")
    ax.set_title("And what it saves", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    tidy(ax)
    save(fig, "05-binning")


# ------------------------------------------------------------ 6. calibration

def fig_calibration():
    """Boosted trees are widely said to produce uncalibrated scores. Measured,
    that depends entirely on the loss: gradient boosting with the logistic loss
    is optimising a proper scoring rule and comes out calibrated, while a random
    forest's vote fraction does not. The first version of this figure assumed
    the folk claim and found isotonic regression making things worse."""
    print("\n[6] which scores are probabilities, and which are not")
    X, y = tabular(9000, noise=1.2, seed=7)
    yb = (y > np.median(y)).astype(float)
    Xtr, ytr = X[:4000], yb[:4000]
    Xca, yca = X[4000:6500], yb[4000:6500]
    Xte, yte = X[6500:], yb[6500:]

    g = GBM(n_trees=300, lr=0.1, max_depth=4, objective="logistic", seed=8).fit(Xtr, ytr)
    p_gbm = g.predict(Xte)

    f = Forest(n_trees=120, max_depth=10, min_samples=2, seed=9).fit(Xtr, ytr)
    p_rf = np.clip(f.predict(Xte), 0, 1)          # the vote fraction
    p_rf_cal_src = np.clip(f.predict(Xca), 0, 1)

    def isotonic_fit(x, yv):
        o = np.argsort(x); xs, ys = x[o], yv[o].astype(float)
        v = ys.copy(); w = np.ones(len(v)); xk = xs.copy()
        i = 0
        while i < len(v) - 1:
            if v[i] > v[i + 1]:
                nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                nw = w[i] + w[i + 1]
                v = np.delete(v, i + 1); w = np.delete(w, i + 1); xk = np.delete(xk, i + 1)
                v[i], w[i] = nv, nw
                i = max(i - 1, 0)
            else:
                i += 1
        return xk, v

    xk, vk = isotonic_fit(p_rf_cal_src, yca)
    p_rf_cal = np.interp(p_rf, xk, vk)

    def ece(p, yv, bins=12):
        e = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
        return float(sum((idx == b).mean() * abs(p[idx == b].mean() - yv[idx == b].mean())
                         for b in range(bins) if (idx == b).sum()))

    def auc(p, yv):
        o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
        n1 = yv.sum(); n0 = len(yv) - n1
        return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    rows = [("boosting, logistic loss", p_gbm), ("random forest vote", p_rf),
            ("forest + isotonic", p_rf_cal)]
    for lab, p in rows:
        print(f"    {lab:26s} ECE {ece(p, yte):.4f}   AUC {auc(p, yte):.4f}")
    print(f"    isotonic changes the forest's AUC by "
          f"{abs(auc(p_rf, yte) - auc(p_rf_cal, yte)):.4f} — it is monotone, so it")
    print(f"    repairs the probabilities and leaves every ranking decision alone.")

    def reliability(p, yv, bins=12):
        e = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
        xs_, ys_ = [], []
        for b in range(bins):
            m = idx == b
            if m.sum() > 25:
                xs_.append(p[m].mean()); ys_.append(yv[m].mean())
        return np.array(xs_), np.array(ys_)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for (lab, p), col in zip(rows, (GREEN, ORANGE, BLUE)):
        bx, by = reliability(p, yte)
        ax.plot(bx, by, "o-", color=col, lw=1.8, ms=4, label=lab)
    ax.plot([0, 1], [0, 1], color=INK, lw=1.2, ls="--", label="perfect")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    ax.bar(np.arange(3), [ece(p, yte) for _, p in rows], 0.5,
           color=[GREEN, ORANGE, BLUE])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["boosting\n(logistic)", "forest\n(vote)", "forest\n+ isotonic"],
                       fontsize=8.5)
    ax.set_ylabel("expected calibration error")
    ax.set_title("Which needs repairing", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "06-calibration")


# ------------------------------------------------- 7. trees against networks

def _mlp(Xtr, ytr, Xte, hidden=128, epochs=400, lr=3e-3, seed=0, wd=1e-4):
    """A small MLP in numpy, standardised inputs, Adam."""
    r = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    t = (ytr - ym) / ys
    d = A.shape[1]
    W1 = r.standard_normal((d, hidden)) * np.sqrt(2 / d); b1 = np.zeros(hidden)
    W2 = r.standard_normal((hidden, hidden)) * np.sqrt(2 / hidden); b2 = np.zeros(hidden)
    W3 = r.standard_normal((hidden, 1)) * np.sqrt(1 / hidden); b3 = np.zeros(1)
    ps = [W1, b1, W2, b2, W3, b3]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    n = len(A); bs = 256
    step = 0
    for ep in range(epochs):
        perm = r.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, yy = A[idx], t[idx]
            h1 = np.maximum(x @ W1 + b1, 0)
            h2 = np.maximum(h1 @ W2 + b2, 0)
            o = (h2 @ W3 + b3).ravel()
            go = (2 * (o - yy) / len(idx))[:, None]
            gW3 = h2.T @ go + wd * W3; gb3 = go.sum(0)
            gh2 = go @ W3.T; gh2[h2 <= 0] = 0
            gW2 = h1.T @ gh2 + wd * W2; gb2 = gh2.sum(0)
            gh1 = gh2 @ W2.T; gh1[h1 <= 0] = 0
            gW1 = x.T @ gh1 + wd * W1; gb1 = gh1.sum(0)
            gs = [gW1, gb1, gW2, gb2, gW3, gb3]
            step += 1
            for k in range(6):
                ms[k] = 0.9 * ms[k] + 0.1 * gs[k]
                vs[k] = 0.999 * vs[k] + 0.001 * gs[k] ** 2
                mh = ms[k] / (1 - 0.9 ** step); vh = vs[k] / (1 - 0.999 ** step)
                ps[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            W1, b1, W2, b2, W3, b3 = ps
    h1 = np.maximum(B @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    return (h2 @ W3 + b3).ravel() * ys + ym


def fig_trees_vs_nets():
    """The comparison the book has been citing and never running.

    Repeated over 8 independent draws of data and model seeds, because the first
    version used one split and one seed per family and quoted three-significant-
    figure ratios off it. Predictions were recorded before this
    was run.
    """
    print("\n[7] trees against networks, on four generating processes")

    def make(kind, n, seed):
        rr = np.random.default_rng(seed)
        d = 12
        X = rr.standard_normal((n, d))
        if kind == "thresholds":
            y = (2.0 * (X[:, 0] > 0.5) + 1.5 * (X[:, 1] < -0.3)
                 + 1.0 * ((X[:, 2] > 0) & (X[:, 3] > 0)))
        elif kind == "smooth":
            y = np.sin(2 * X[:, 0]) + 0.7 * X[:, 1] ** 2 + 0.5 * X[:, 2]
        elif kind == "linear":
            w = rr.standard_normal(d)
            y = X @ w
        else:
            y = 1.5 * X[:, 0] * X[:, 1] + 1.2 * X[:, 2] * X[:, 3] * X[:, 4]
        return X, y + 0.4 * rr.standard_normal(n)

    REPS = 8
    kinds = ["thresholds", "smooth", "linear", "interactions"]
    rows = []
    for kind in kinds:
        g_all, n_all = [], []
        for rep in range(REPS):
            X, y = make(kind, 5000, seed=1000 * (kinds.index(kind) + 1) + rep)
            Xtr, ytr = X[:3000], y[:3000]
            Xva, yva = X[3000:4000], y[3000:4000]
            Xte, yte = X[4000:], y[4000:]
            var = float(np.var(yte))

            best, cfg = np.inf, None
            for lr in (0.05, 0.1):
                for md in (3, 5):
                    g = GBM(n_trees=300, lr=lr, max_depth=md, seed=rep).fit(Xtr, ytr)
                    e = float(np.mean((g.predict(Xva) - yva) ** 2))
                    if e < best: best, cfg = e, (lr, md)
            g = GBM(n_trees=300, lr=cfg[0], max_depth=cfg[1], seed=rep).fit(Xtr, ytr)
            g_all.append(float(np.mean((g.predict(Xte) - yte) ** 2)) / var)

            best, ncfg = np.inf, None
            for hid in (64, 256):
                for lrn in (1e-3, 3e-3):
                    pv = _mlp(Xtr, ytr, Xva, hidden=hid, epochs=200, lr=lrn, seed=rep)
                    e = float(np.mean((pv - yva) ** 2))
                    if e < best: best, ncfg = e, (hid, lrn)
            pt = _mlp(Xtr, ytr, Xte, hidden=ncfg[0], epochs=200, lr=ncfg[1], seed=rep)
            n_all.append(float(np.mean((pt - yte) ** 2)) / var)

        g_all, n_all = np.array(g_all), np.array(n_all)
        diff = n_all - g_all                       # paired, same data each rep
        sep = abs(diff.mean()) > 2 * diff.std(ddof=1) / np.sqrt(REPS)
        rows.append((kind, g_all, n_all, sep))
        print(f"    {kind:13s} trees {g_all.mean():.4f} +/- {g_all.std(ddof=1):.4f}   "
              f"network {n_all.mean():.4f} +/- {n_all.std(ddof=1):.4f}   "
              f"{'SEPARATED' if sep else 'not separated'}"
              f"  (paired diff {diff.mean():+.4f} +/- {diff.std(ddof=1)/np.sqrt(REPS):.4f})")
    n_sep = sum(1 for r in rows if r[3])
    print(f"    {n_sep} of {len(rows)} comparisons separate at two standard errors "
          f"of the paired difference.")

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    x = np.arange(len(rows)); w = 0.35
    ax.bar(x - w / 2, [r[1].mean() for r in rows], w, color=GREEN,
           yerr=[r[1].std(ddof=1) for r in rows], capsize=3,
           error_kw=dict(lw=1.0, ecolor=GRAY), label="boosted trees")
    ax.bar(x + w / 2, [r[2].mean() for r in rows], w, color=BLUE,
           yerr=[r[2].std(ddof=1) for r in rows], capsize=3,
           error_kw=dict(lw=1.0, ecolor=GRAY), label="neural network")
    for i, r in enumerate(rows):
        if not r[3]:
            ax.text(i, max(r[1].mean(), r[2].mean()) * 1.35, "n.s.",
                    ha="center", fontsize=8, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("test error, relative to the variance")
    ax.set_title(f"Eight repetitions each; bars are one standard deviation",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "07-trees-vs-nets")


# ------------------------------------------------------ 8. temporal validation

def fig_temporal_cv():
    """Splitting a time series at random lets the model see the future. Measure
    what that is worth in reported accuracy, against the honest number."""
    print("\n[8] what a random split is worth on a time series")
    r = np.random.default_rng(13)
    T = 3000
    t = np.arange(T)
    trend = 0.002 * t
    seas = 1.5 * np.sin(2 * np.pi * t / 50)
    walk = np.cumsum(r.standard_normal(T) * 0.15)
    y = trend + seas + walk + 0.3 * r.standard_normal(T)

    LAGS = 12
    Xall = np.column_stack([np.roll(y, k) for k in range(1, LAGS + 1)])[LAGS:]
    yall = y[LAGS:]
    n = len(yall)

    def fit_eval(tr_idx, te_idx):
        g = GBM(n_trees=200, lr=0.08, max_depth=4, seed=14).fit(Xall[tr_idx], yall[tr_idx])
        return float(np.mean((g.predict(Xall[te_idx]) - yall[te_idx]) ** 2))

    rand_scores, temp_scores = [], []
    for rep in range(5):
        rr = np.random.default_rng(100 + rep)
        perm = rr.permutation(n)
        cut = int(0.75 * n)
        rand_scores.append(fit_eval(perm[:cut], perm[cut:]))
    cut = int(0.75 * n)
    temp_scores.append(fit_eval(np.arange(cut), np.arange(cut, n)))

    rnd, tmp = float(np.mean(rand_scores)), float(np.mean(temp_scores))
    base = float(np.mean((yall[cut:] - yall[cut - 1]) ** 2))
    print(f"    random split      test mse {rnd:.4f}")
    print(f"    chronological     test mse {tmp:.4f}   ({tmp/rnd:.1f}x worse)")
    print(f"    last-value baseline        {base:.4f}")
    print(f"    the random split reports a model {tmp/rnd:.1f} times better than it is")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(t[:600], y[:600], color=BLUE, lw=1.1)
    ax.set_xlabel("time"); ax.set_ylabel("value")
    ax.set_title("A series with trend, season and a walk", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.bar([0, 1, 2], [rnd, tmp, base], 0.5, color=[ORANGE, GREEN, GRAY])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["random\nsplit", "chronological\nsplit", "last value\nbaseline"],
                       fontsize=8.5)
    ax.set_ylabel("test mean squared error")
    ax.set_title("What the split is worth", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "08-temporal-cv")


# --------------------------------------------------- 9. horizons and coverage

def fig_horizon():
    """A forecast is not one number but a function of how far ahead you ask.
    Measure the error against the horizon, and the coverage of a band calibrated
    at each one on a held-out slice of the series' own past.

    The first version calibrated on TRAINING residuals, which under-covers for a
    reason that has nothing to do with forecasting. Fixing that leaves the
    under-coverage almost unchanged, which is the actual result: a series with a
    trend and a random walk is not exchangeable with its own past, so the
    conformal guarantee does not hold along a time axis however you slice it."""
    print("\n[9] the horizon, and what the intervals do along it")
    r = np.random.default_rng(15)
    T = 6000
    t = np.arange(T)
    y = (0.001 * t + 1.2 * np.sin(2 * np.pi * t / 40)
         + np.cumsum(r.standard_normal(T) * 0.12) + 0.25 * r.standard_normal(T))
    LAGS = 16
    horizons = [1, 2, 3, 5, 8, 12, 20, 30]
    errs, covs, widths = [], [], []
    for h in horizons:
        # predict y[i] from the LAGS values ending h steps before it
        idx = np.arange(LAGS + h - 1, T)
        X = np.column_stack([y[idx - h - k + 1] for k in range(1, LAGS + 1)])
        tgt = y[idx]
        m = len(tgt)
        a, b = int(0.60 * m), int(0.80 * m)          # train | calibrate | test
        g = GBM(n_trees=200, lr=0.08, max_depth=4, seed=16).fit(X[:a], tgt[:a])
        resid_cal = np.abs(tgt[a:b] - g.predict(X[a:b]))
        k = int(np.ceil((len(resid_cal) + 1) * 0.90))
        q = float(np.sort(resid_cal)[min(k, len(resid_cal)) - 1])
        pred = g.predict(X[b:])
        errs.append(float(np.sqrt(np.mean((pred - tgt[b:]) ** 2))))
        covs.append(float(np.mean(np.abs(tgt[b:] - pred) <= q)))
        widths.append(2 * q)
        print(f"    horizon {h:2d}:  RMSE {errs[-1]:.3f}   band width {widths[-1]:.2f}"
              f"   coverage {covs[-1]:.3f}")
    print(f"    the band widens with the horizon as it should ({widths[0]:.2f} -> "
          f"{widths[-1]:.2f}),")
    print(f"    and it UNDER-COVERS at every one of them — {min(covs):.3f} to {max(covs):.3f}")
    print(f"    against a promised 0.90, including one step ahead. The calibration slice")
    print(f"    is the series' own past, and a series with a trend and a random walk is")
    print(f"    not exchangeable with its past. This is Part XXIV Figure 5's failure")
    print(f"    arriving wherever the data has a time index.")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(horizons, errs, "o-", color=BLUE, lw=1.9, ms=5, label="RMSE")
    ax.plot(horizons, np.array(widths) / 2, "s--", color=GRAY, lw=1.5, ms=4,
            label="half the band width")
    ax.set_xlabel("steps ahead"); ax.set_ylabel("error")
    ax.set_title("Both grow with the horizon", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(horizons, covs, "o-", color=PURPLE, lw=1.9, ms=5)
    ax.axhline(0.90, color=INK, lw=1.3, ls="--")
    ax.text(horizons[-1], 0.915, "the promise", fontsize=8, color=INK, ha="right")
    ax.set_xlabel("steps ahead"); ax.set_ylabel("measured coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("And it under-covers at every one", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "09-horizon")


if __name__ == "__main__":
    fig_depth()
    fig_bagging()
    fig_subsampling_illus()
    fig_boosting()
    fig_binning()
    fig_calibration()
    fig_trees_vs_nets()
    fig_temporal_cv()
    fig_horizon()
