"""
Figures for Part XXVI — Graph and Geometric Learning.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about four minutes. Numpy only.
"""
import os
import re
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "gg"))
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


# ---------------------------------------------------------------- a tiny MLP

def mlp_fit(X, y, hidden=96, epochs=300, lr=3e-3, seed=0, classes=2):
    r = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = (X - mu) / sd
    d = A.shape[1]
    W1 = r.standard_normal((d, hidden)) * np.sqrt(2 / d); b1 = np.zeros(hidden)
    W2 = r.standard_normal((hidden, classes)) * np.sqrt(1 / hidden); b2 = np.zeros(classes)
    ps = [W1, b1, W2, b2]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    Y = np.eye(classes)[y]
    n, bs, step = len(A), 128, 0
    for ep in range(epochs):
        perm = r.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, t = A[idx], Y[idx]
            h = np.maximum(x @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(1, keepdims=True)
            P = np.exp(z); P /= P.sum(1, keepdims=True)
            gz = (P - t) / len(idx)
            gW2 = h.T @ gz; gb2 = gz.sum(0)
            gh = gz @ W2.T; gh[h <= 0] = 0
            gW1 = x.T @ gh; gb1 = gh.sum(0)
            gs = [gW1, gb1, gW2, gb2]; step += 1
            for k in range(4):
                ms[k] = 0.9 * ms[k] + 0.1 * gs[k]
                vs[k] = 0.999 * vs[k] + 0.001 * gs[k] ** 2
                ps[k] -= lr * (ms[k] / (1 - 0.9 ** step)) / (np.sqrt(vs[k] / (1 - 0.999 ** step)) + 1e-8)
            W1, b1, W2, b2 = ps

    def predict(Xq):
        a = (Xq - mu) / sd
        h = np.maximum(a @ W1 + b1, 0)
        return np.argmax(h @ W2 + b2, axis=1)
    return predict


# ------------------------------------------------------------ 1. permutation

def _random_graph(n, p, rng):
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, 1); A = A + A.T
    return A


def fig_permutation():
    """A graph has no canonical node ordering. Measure what a model that reads a
    flattened adjacency matrix does when the same graphs arrive relabelled, and
    what a permutation-invariant readout does instead."""
    print("\n[1] the ordering a graph does not have")
    rng = np.random.default_rng(0)
    N, NTRAIN, NTEST = 9, 3000, 1500

    def make(n_graphs, rng):
        As, ys = [], []
        for _ in range(n_graphs):
            p = rng.uniform(0.25, 0.55)
            A = _random_graph(N, p, rng)
            # The label must be permutation-invariant AND not a simple sum over
            # the adjacency entries. Edge count is invariant but a linear readout
            # of the flattened matrix computes it whatever the ordering, so it
            # cannot test whether a model has learned the symmetry at all — the
            # first version of this figure used it and measured nothing.
            # Triangle count is invariant and genuinely structural.
            tri = np.trace(A @ A @ A) / 6.0
            ys.append(int(tri > 9))
            As.append(A)
        return np.array(As), np.array(ys)

    def permute(As, rng):
        out = np.empty_like(As)
        for i, A in enumerate(As):
            q = rng.permutation(N)
            out[i] = A[np.ix_(q, q)]
        return out

    Atr, ytr = make(NTRAIN, rng)
    Ate, yte = make(NTEST, rng)
    Ate_perm = permute(Ate, np.random.default_rng(99))

    flat = lambda A: A.reshape(len(A), -1)
    # An invariant readout must also be able to SEE the property. Sorted degrees
    # cannot detect a triangle; the adjacency spectrum can, and is invariant
    # because a permutation is a similarity transform.
    def inv(A):
        return np.sort(np.linalg.eigvalsh(A), axis=1)

    rows = []
    pred = mlp_fit(flat(Atr), ytr, seed=1)
    rows.append(("dense on adjacency", float(np.mean(pred(flat(Ate)) == yte)),
                 float(np.mean(pred(flat(Ate_perm)) == yte))))

    Atr_aug = np.concatenate([Atr] + [permute(Atr, np.random.default_rng(10 + k))
                                      for k in range(3)])
    ytr_aug = np.tile(ytr, 4)
    pred_a = mlp_fit(flat(Atr_aug), ytr_aug, seed=1, epochs=300)
    rows.append(("dense, permutation-augmented", float(np.mean(pred_a(flat(Ate)) == yte)),
                 float(np.mean(pred_a(flat(Ate_perm)) == yte))))

    pred_i = mlp_fit(inv(Atr), ytr, seed=1)
    rows.append(("permutation-invariant", float(np.mean(pred_i(inv(Ate)) == yte)),
                 float(np.mean(pred_i(inv(Ate_perm)) == yte))))

    for lab, a, b in rows:
        print(f"    {lab:30s} same ordering {a:.3f}   relabelled {b:.3f}   drop {a-b:+.3f}")

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(rows)); w = 0.35
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=BLUE, label="same node ordering")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=ORANGE, label="relabelled nodes")
    ax.axhline(0.5, color=INK, lw=1.1, ls=":")
    ax.text(len(rows) - 0.6, 0.52, "chance", fontsize=8, color=INK, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0].replace(", ", ",\n").replace(" on ", "\non ")
                                          for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
    ax.set_title("The same graphs, with the nodes renumbered", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)
    save(fig, "01-permutation")


# --------------------------------------------------------- 2. over-smoothing

def fig_oversmoothing():
    """Message passing averages over neighbours, and averaging repeatedly is a
    diffusion. Measure the collapse, and measure what it costs a task.

    The generator here has weak community structure by design. A first attempt
    used strong communities, which stay separable however much they are smoothed,
    so the collapse had nothing to destroy — that setting measures the left panel
    and cannot test the right one. Both generators were pre-registered.
    """
    print("\n[2] what repeated averaging does")
    REPS = 5
    depths = list(range(0, 25))
    S_all, A_all = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        N, C = 300, 4
        labels = rng.integers(0, C, N)
        P = np.where(labels[:, None] == labels[None, :], 0.055, 0.030)
        A = (rng.random((N, N)) < P).astype(float)
        A = np.triu(A, 1); A = A + A.T
        A = A + np.eye(N)
        D = A.sum(1)
        S = A / np.sqrt(D[:, None] * D[None, :])
        H = np.eye(C)[labels] * 1.0 + 2.2 * rng.standard_normal((N, C))
        tr = rng.random(N) < 0.35
        sp, ac = [], []
        for d in depths:
            if d > 0:
                H = S @ H
            Z = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
            dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            sp.append(float(dist[~np.eye(N, dtype=bool)].mean()))
            p = mlp_fit(H[tr], labels[tr], hidden=48, epochs=200, seed=2, classes=C)
            ac.append(float(np.mean(p(H[~tr]) == labels[~tr])))
        S_all.append(sp); A_all.append(ac)
    S_all, A_all = np.array(S_all), np.array(A_all)
    sm, ssd = S_all.mean(0), S_all.std(0, ddof=1)
    am, asd = A_all.mean(0), A_all.std(0, ddof=1)
    best = int(np.argmax(am))
    for d in (0, 2, 4, 8, 16, 24):
        print(f"    depth {d:2d}: distance {sm[d]:.4f}   accuracy {am[d]:.3f} "
              f"+/- {asd[d]:.3f}")
    mono = all(sm[i] >= sm[i + 1] - 1e-9 for i in range(1, len(sm) - 1))
    diff = A_all[:, best] - A_all[:, -1]
    se = diff.std(ddof=1) / np.sqrt(REPS)
    print(f"    distance monotonically falling: {mono}   "
          f"({sm[0]:.3f} -> {sm[-1]:.3f})")
    print(f"    accuracy peaks at depth {best} ({am[best]:.3f}); "
          f"falls to {am[-1]:.3f} by depth 24")
    print(f"    paired fall {diff.mean():+.3f} +/- {se:.3f}  "
          f"({'separated' if abs(diff.mean()) > 2 * se else 'NOT separated'}); "
          f"chance is 0.250")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.errorbar(depths, sm, yerr=ssd, fmt="o-", color=PURPLE, lw=1.9, ms=4, capsize=2)
    ax.set_xlabel("message-passing rounds"); ax.set_ylabel("mean pairwise distance")
    ax.set_title("Representations collapse together", fontsize=10.5, color=INK)
    tidy(ax)
    ax = axes[1]
    ax.errorbar(depths, am, yerr=asd, fmt="o-", color=GREEN, lw=1.9, ms=4, capsize=2)
    ax.axhline(0.25, color=INK, lw=1.0, ls=":")
    ax.text(depths[-1], 0.258, "chance", fontsize=8, color=INK, ha="right")
    ax.axvline(best, color=ORANGE, lw=1.2, ls="--")
    ax.text(best + 0.6, am.max() * 0.98, f"best: {best}", fontsize=8, color=ORANGE)
    ax.set_xlabel("message-passing rounds"); ax.set_ylabel("node classification accuracy")
    ax.set_title("And the task goes with them", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "02-oversmoothing")


# ----------------------------------------------------- 3. the WL upper bound

def fig_wl():
    """Message passing computes exactly what 1-dimensional Weisfeiler-Leman
    computes, so a pair of graphs WL cannot separate receives identical
    representations at every depth. Demonstrated on the standard pair."""
    print("\n[3] two graphs message passing cannot tell apart")

    def cycle(n, offset=0):
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
        return A

    def two_triangles():
        A = np.zeros((6, 6))
        for base in (0, 3):
            for i, j in itertools.combinations(range(base, base + 3), 2):
                A[i, j] = A[j, i] = 1.0
        return A

    G1, G2 = two_triangles(), cycle(6)
    assert G1.sum() == G2.sum(), "both must be 2-regular on 6 nodes"
    print(f"    both graphs: 6 nodes, {int(G1.sum()//2)} edges, every degree "
          f"{int(G1.sum(1)[0])}")
    print(f"    isomorphic: False  (one has triangles, the other has none)")

    def mp_readout(A, x, depth, W=None):
        H = x.copy()
        D = A.sum(1) + 1
        S = (A + np.eye(len(A))) / D[:, None]
        for _ in range(depth):
            H = np.tanh(S @ H @ W)
        return H.sum(0)                       # a permutation-invariant readout

    rng = np.random.default_rng(3)
    W = rng.standard_normal((4, 4)) * 0.7
    depths = list(range(1, 9))
    gaps_uniform, gaps_feat = [], []
    x_uniform = np.ones((6, 4))
    xf = np.zeros((6, 4)); xf[:, 0] = 1.0
    # a feature that counts triangles through each node — breaks the tie
    tri1 = np.diag(np.linalg.matrix_power(G1, 3)) / 2
    tri2 = np.diag(np.linalg.matrix_power(G2, 3)) / 2
    x1 = xf.copy(); x1[:, 1] = tri1
    x2 = xf.copy(); x2[:, 1] = tri2
    for d in depths:
        a = mp_readout(G1, x_uniform, d, W); b = mp_readout(G2, x_uniform, d, W)
        gaps_uniform.append(float(np.abs(a - b).max()))
        a2 = mp_readout(G1, x1, d, W); b2 = mp_readout(G2, x2, d, W)
        gaps_feat.append(float(np.abs(a2 - b2).max()))
        print(f"    depth {d}:  uniform features  |difference| {gaps_uniform[-1]:.2e}"
              f"     with a triangle count {gaps_feat[-1]:.4f}")
    print(f"    largest difference under uniform features across all depths: "
          f"{max(gaps_uniform):.2e}")

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.semilogy(depths, np.maximum(gaps_uniform, 1e-18), "o-", color=ORANGE, lw=1.9,
                ms=5, label="uniform node features")
    ax.semilogy(depths, np.maximum(gaps_feat, 1e-18), "o-", color=GREEN, lw=1.9,
                ms=5, label="with a triangle count")
    ax.axhline(1e-15, color=GRAY, lw=1.0, ls=":")
    ax.text(depths[-1], 1.4e-15, "machine precision", fontsize=8, color=GRAY, ha="right")
    ax.set_xlabel("message-passing rounds")
    ax.set_ylabel("difference between the two graphs' readouts")
    ax.set_title("Two 2-regular graphs, six nodes each", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "03-wl-bound")


# ------------------------------------------------ 5. what a symmetry buys

def fig_equivariance():
    """A model built to respect a symmetry does not need to see every
    transformed copy. Measure the sample efficiency against a model that has to
    learn the symmetry from augmentation."""
    print("\n[5] the symmetry, against paying data for it")
    rng = np.random.default_rng(4)
    K = 12                                    # points per cloud

    def make(n, seed):
        r = np.random.default_rng(seed)
        P = r.standard_normal((n, K, 2))
        # label: is the cloud's spread above a threshold — a rotation-INVARIANT property
        y = (np.linalg.norm(P - P.mean(1, keepdims=True), axis=2).mean(1) > 0.95).astype(int)
        return P, y

    def rotate(P, r):
        th = r.uniform(0, 2 * np.pi, len(P))
        c, s = np.cos(th), np.sin(th)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        return np.einsum("nij,nkj->nki", R, P)

    def raw(P):     return P.reshape(len(P), -1)
    def invar(P):   # pairwise distances are rotation invariant
        d = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
        iu = np.triu_indices(K, 1)
        return np.sort(d[:, iu[0], iu[1]], axis=1)

    Pte, yte = make(2000, 999)
    Pte = rotate(Pte, np.random.default_rng(1234))
    sizes = [100, 200, 400, 800, 1600, 3200]
    eq, aug = [], []
    for n in sizes:
        P, y = make(n, 500 + n)
        e = float(np.mean(mlp_fit(invar(P), y, seed=5)(invar(Pte)) == yte))
        Pa = np.concatenate([rotate(P, np.random.default_rng(600 + k)) for k in range(4)])
        ya = np.tile(y, 4)
        a = float(np.mean(mlp_fit(raw(Pa), ya, seed=5)(raw(Pte)) == yte))
        eq.append(e); aug.append(a)
        print(f"    n={n:5d}:  invariant {e:.3f}   augmented {a:.3f}   gap {e-a:+.3f}")
    gaps = np.array(eq) - np.array(aug)
    print(f"    gap at the smallest training set {gaps[0]:+.3f}, "
          f"at the largest {gaps[-1]:+.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.semilogx(sizes, eq, "o-", color=GREEN, lw=1.9, ms=5, label="invariant by construction")
    ax.semilogx(sizes, aug, "o-", color=ORANGE, lw=1.9, ms=5, label="learned from augmentation")
    ax.axhline(0.5, color=INK, lw=1.0, ls=":")
    ax.set_xlabel("training examples"); ax.set_ylabel("accuracy on rotated test data")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Building the symmetry in, against paying data for it",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    tidy(ax)
    save(fig, "05-equivariance")


# ---------------------------------------------------------- 4. over-squashing

def fig_oversquashing():
    """Information from a distant node competes with everything else arriving,
    and the number of competitors grows with distance. Measure the share of what
    reaches a target that came from a chosen source, on a path and on a graph
    with a bottleneck at matched distance.

    A first attempt chose the bottleneck graph's source and target without
    checking their actual separation, so the propagation depth and the distance
    did not match and half the measurements were exactly zero. Distances are now
    computed from the graph.
    """
    print("\n[4] information travelling a distance")

    def shortest_paths(A):
        n = len(A)
        D = np.full((n, n), np.inf); np.fill_diagonal(D, 0)
        D[A > 0] = 1
        for k in range(n):
            D = np.minimum(D, D[:, k][:, None] + D[k][None, :])
        return D

    def path(d):
        n = d + 1
        A = np.zeros((n, n))
        for i in range(d):
            A[i, i + 1] = A[i + 1, i] = 1.0
        return A, 0, n - 1

    def two_trees(k):
        """Two complete binary trees of depth k joined at their roots. Every
        leaf-to-leaf path across the join uses that single edge, and two leaves
        on opposite sides are 2k+1 apart."""
        size = 2 ** k - 1
        n = 2 * size
        A = np.zeros((n, n))
        for base in (0, size):
            for i in range(size):
                for c in (2 * i + 1, 2 * i + 2):
                    if c < size:
                        A[base + i, base + c] = A[base + c, base + i] = 1.0
        A[0, size] = A[size, 0] = 1.0
        return A, n - 1, size - 1

    def share(A, src, dst, steps):
        n = len(A)
        D = A.sum(1) + 1
        S = (A + np.eye(n)) / D[:, None]
        Sd = np.linalg.matrix_power(S, steps)
        return float(Sd[dst, src] / max(Sd[dst].sum(), 1e-300))

    rows = []
    for k in range(1, 6):
        At, s2, t2 = two_trees(k + 1)
        dist = int(shortest_paths(At)[s2, t2])
        Ap, s1, t1 = path(dist)
        rows.append((dist, share(Ap, s1, t1, dist), share(At, s2, t2, dist)))
        print(f"    distance {dist:2d}:  path {rows[-1][1]:.3e}   "
              f"bottlenecked {rows[-1][2]:.3e}   "
              f"ratio {rows[-1][1]/max(rows[-1][2],1e-300):.1f}x")
    ds = [r[0] for r in rows]
    pc = [r[1] for r in rows]; tc = [r[2] for r in rows]
    print(f"    on the path the source's share falls {pc[0]/pc[-1]:.0f}x over this range;")
    print(f"    on the bottlenecked graph {tc[0]/tc[-1]:.0f}x")

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.semilogy(ds, pc, "o-", color=BLUE, lw=1.9, ms=5, label="a path")
    ax.semilogy(ds, tc, "o-", color=ORANGE, lw=1.9, ms=5,
                label="two binary trees joined at the root")
    ax.set_xlabel("distance from source to target")
    ax.set_ylabel("the source's share of what arrives")
    ax.set_title("What reaches the target, and from where", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "04-oversquashing")


if __name__ == "__main__":
    fig_permutation()
    fig_oversmoothing()
    fig_wl()
    fig_oversquashing()
    fig_equivariance()
