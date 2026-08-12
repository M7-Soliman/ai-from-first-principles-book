"""
Figures for Part XV — Representation Learning.

Every claim this part makes that can be measured, is measured here.

The cheap ones are exact computations: the region count of a distributed
representation against a symbolic one, hubness in high dimensions, and what a
neighbour-embedding does to distance. The expensive ones train: a linear
autoencoder against PCA, a denoising autoencoder against the analytic score,
contrastive learning with and without negatives, superposition, and probe
selectivity.

Two of these exist because the folk version did not survive contact:
t-SNE cluster sizes carry no information (fig_neighbour_embedding), and a
probe's accuracy alone does not establish that a representation encodes
anything (fig_probe, in train_figures.py).

All data is synthetic and generated here.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train

Runtime: about a minute for --cheap, fifteen for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "rp"))
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


# --------------------------------------------------- 1. distributed vs symbolic

def regions_from_hyperplanes(n, d):
    """General-position count (Schläfli; Zaslavsky's arrangement theorem is the
    generalisation to hyperplanes in any position): n hyperplanes in general
    position in R^d cut it into sum_{j=0}^{d} C(n, j) regions. Exact integer
    arithmetic."""
    from math import comb
    return sum(comb(n, j) for j in range(0, d + 1))


def fig_distributed():
    """The counting argument behind distributed representations, computed
    exactly rather than asserted."""
    ns = np.arange(1, 41)
    dims = [2, 5, 10]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    for d, c in zip(dims, (BLUE, ORANGE, PURPLE)):
        r = [regions_from_hyperplanes(int(n), d) for n in ns]
        ax.plot(ns, r, color=c, lw=1.8, label=f"distributed, $d={d}$")
    ax.plot(ns, ns, color=GRAY, lw=1.8, ls="--", label="symbolic (one region each)")
    ax.set_yscale("log")
    ax.set_xlabel("number of features $n$")
    ax.set_ylabel("distinguishable regions")
    ax.set_title("What $n$ features can distinguish", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    # the exact numbers the caption quotes
    print("    exact region counts (general position):")
    for d in dims:
        for n in (10, 20, 40):
            print(f"      n={n:3d}, d={d:2d} -> {regions_from_hyperplanes(n, d):,} regions "
                  f"({regions_from_hyperplanes(n, d)/n:.1f}x symbolic)")

    # right: how many examples a symbolic learner needs to cover the same set
    ax = axes[1]
    d = 10
    r = np.array([regions_from_hyperplanes(int(n), d) for n in ns], dtype=float)
    ax.plot(ns, r / ns, color=GREEN, lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("number of features $n$")
    ax.set_ylabel("regions per feature")
    ax.set_title("Regions per feature, $d=10$", fontsize=10.5, color=INK)
    tidy(ax)
    n40 = regions_from_hyperplanes(40, 10)
    ax.annotate(f"{n40:,} regions\nfrom 40 features",
                xy=(40, n40 / 40), xytext=(19, n40 / 40 * 0.06),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))

    print(f"    40 binary features in d=10 distinguish {n40:,} regions;")
    print(f"    a symbolic representation with 40 symbols distinguishes 40.")
    print(f"    ratio {n40/40:,.0f}x — and this is POLYNOMIAL in n, not exponential:")
    print(f"    the exponential is in d. Doubling n from 20 to 40 multiplies regions by "
          f"{n40/regions_from_hyperplanes(20,10):.0f}x, not 2^20.")
    fig.tight_layout()
    save(fig, "01-distributed")


# ------------------------------------------------------------------ 2. hubness

def fig_hubness():
    """In high dimensions a few points become everyone's neighbour, and a large
    fraction become nobody's. Exact measurement on Gaussian data — no model, no
    training. The right panel is the mechanism, not the effect."""
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    n, k = 3000, 10
    dims = [2, 5, 10, 25, 50, 100, 200, 500]
    skews, maxhub, orphan, corr = [], [], [], []
    detail = {}

    for d in dims:
        X = rng.normal(size=(n, d))
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn = np.argpartition(D, k, axis=1)[:, :k]
        cnt = np.bincount(nn.ravel(), minlength=n)     # N_k occurrence count
        m, sd = cnt.mean(), cnt.std()
        skews.append(float(((cnt - m) ** 3).mean() / sd ** 3))
        maxhub.append(int(cnt.max()))
        orphan.append(float((cnt == 0).mean()))
        norms = np.linalg.norm(X, axis=1)
        corr.append(float(np.corrcoef(cnt, -norms)[0, 1]))
        if d == 100:
            detail = dict(cnt=cnt, norms=norms)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    ax.plot(dims, maxhub, "o-", color=ORANGE, lw=1.8, ms=4, label="biggest hub")
    ax.axhline(k, color=GRAY, lw=1.1, ls="--", label=f"the mean, {k}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"times a point is in a top-$10$")
    ax.set_title("A few points become everyone's neighbour",
                 fontsize=10.5, color=INK)
    ax2 = ax.twinx()
    ax2.plot(dims, np.array(orphan) * 100, "s--", color=BLUE, lw=1.4, ms=3.5)
    ax2.set_ylabel("% that are nobody's neighbour", color=BLUE, fontsize=9)
    ax2.tick_params(labelsize=8.5, colors=BLUE)
    ax2.spines["top"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    ax.scatter(detail["norms"], detail["cnt"], s=5, color=PURPLE, alpha=0.45)
    ax.set_xlabel(r"distance from the centre, $\|\mathbf{x}\|$")
    ax.set_ylabel(r"$N_{10}$ — times chosen as a neighbour")
    ax.set_title(f"Why, at $d=100$: $r={corr[dims.index(100)]:+.2f}$",
                 fontsize=10.5, color=INK)
    tidy(ax)

    print("     d | skew(N10) | biggest hub | nobody's neighbour | corr with centrality")
    for d, s_, m, o, c in zip(dims, skews, maxhub, orphan, corr):
        print(f"    {d:4d} | {s_:8.2f}  | {m:11d} | {o*100:16.1f}% | {c:+.2f}")
    print(f"    the mean count is {k} by construction, at every dimension.")
    print(f"    at d=2 the biggest hub is {maxhub[0]} and {orphan[0]*100:.1f}% are orphans;")
    print(f"    at d=500 the biggest hub is {maxhub[-1]} ({maxhub[-1]/k:.1f}x the mean) "
          f"and {orphan[-1]*100:.0f}% are orphans.")
    print("    the mechanism: hubs are the points nearest the centre of the cloud.")
    fig.tight_layout()
    save(fig, "07-hubness")


# ----------------------------------------------------- 3. neighbour embeddings

def fig_neighbour_embedding():
    """What t-SNE preserves and what it does not. Three well-separated blobs of
    DELIBERATELY different spread and separation, and the question is whether
    either survives the projection."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(0)
    d = 50
    # three clusters: tight, medium, wide — and very different pairwise gaps
    specs = [(0.30, 200), (1.00, 200), (2.50, 200)]
    centres = np.zeros((3, d))
    centres[1, 0] = 14.0
    centres[2, 1] = 45.0                       # cluster 2 is much further away
    X, lab = [], []
    for i, (sd, m) in enumerate(specs):
        X.append(centres[i] + rng.normal(scale=sd, size=(m, d)))
        lab += [i] * m
    X = np.vstack(X)
    lab = np.array(lab)

    Z = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=0).fit_transform(X)

    def spread(P, m):
        c = P[m].mean(0)
        return float(np.sqrt(((P[m] - c) ** 2).sum(1).mean()))

    def gap(P, a, b):
        return float(np.linalg.norm(P[lab == a].mean(0) - P[lab == b].mean(0)))

    hi = [spread(X, lab == i) for i in range(3)]
    lo = [spread(Z, lab == i) for i in range(3)]
    ghi = [gap(X, 0, 1), gap(X, 0, 2)]
    glo = [gap(Z, 0, 1), gap(Z, 0, 2)]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for i, c in enumerate((BLUE, ORANGE, PURPLE)):
        ax.scatter(Z[lab == i, 0], Z[lab == i, 1], s=7, color=c, alpha=0.75,
                   label=f"true spread {hi[i]:.2f}")
    ax.set_title("t-SNE of three clusters", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w/2, np.array(hi) / hi[0], w, color=GRAY, label="original")
    ax.bar(x + w/2, np.array(lo) / lo[0], w, color=ORANGE, label="t-SNE")
    ax.set_xticks(x)
    ax.set_xticklabels(["tight", "medium", "wide"])
    ax.set_ylabel("cluster spread, relative to tightest")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_title("Relative cluster size is not preserved",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    tidy(ax)

    print(f"    original spreads: {hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}  "
          f"(ratios 1 : {hi[1]/hi[0]:.1f} : {hi[2]/hi[0]:.1f})")
    print(f"    t-SNE spreads:    {lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f}  "
          f"(ratios 1 : {lo[1]/lo[0]:.1f} : {lo[2]/lo[0]:.1f})")
    print(f"    a {hi[2]/hi[0]:.1f}x spread difference became {lo[2]/lo[0]:.1f}x")
    print(f"    original gaps 0-1 / 0-2: {ghi[0]:.1f} / {ghi[1]:.1f}  "
          f"(ratio {ghi[1]/ghi[0]:.2f})")
    print(f"    t-SNE    gaps 0-1 / 0-2: {glo[0]:.1f} / {glo[1]:.1f}  "
          f"(ratio {glo[1]/glo[0]:.2f})")
    print("    neither relative size NOR relative distance survives the projection")
    fig.tight_layout()
    save(fig, "08-neighbour-embedding")
    return dict(hi=hi, lo=lo, ghi=ghi, glo=glo)


CHEAP = [fig_distributed, fig_hubness, fig_neighbour_embedding]

if __name__ == "__main__":
    print("Part XV — representation learning figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
