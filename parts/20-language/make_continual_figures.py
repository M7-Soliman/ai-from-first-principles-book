"""
Figures 5 to 9 for Part XX — the appended movement G.

Predictions for every one of these were recorded before the
script was run. Figures 1 to 4 live in
make_figures.py and predate that discipline; nothing is invented for them here.

    python3 make_continual_figures.py

Runtime: about five minutes. Numpy only, no autograd.

Scale, stated because it matters: these are small MLPs on synthetic tasks and a
rank-one associative memory standing in for a fact store. They measure the
mechanisms — interference between weight updates, the geometry of averaging, the
ripple from a rank-one edit — rather than reproducing numbers from models a
million times larger. Every caption says which it is.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "lm"))
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
# A small MLP with hand-written backprop, and a family of related tasks.
#
# Every task is a binary classification whose label depends on a different
# random direction in the same input space. That makes the tasks share a domain
# and differ in what matters, which is the situation post-training is actually
# in — not a set of unrelated problems, but the same model asked to weight
# different features.
# ---------------------------------------------------------------------------

def make_task(d, rng, hardness=2.5):
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    return u, hardness


def sample(task, n, rng, d, tid):
    """Inputs carry a one-hot task indicator in their last two coordinates.

    Without it the setup is incoherent rather than hard: both tasks share an
    input distribution and a single output, so an input that is positive under A
    and negative under B has no correct label and no model can satisfy both. A
    first version omitted this and measured an impossibility — joint training
    from scratch reached 0.65, so every "forgetting" number was really a
    statement that the objective had no solution.
    """
    u, hardness = task
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    y = (np.tanh(hardness * (X @ u)) > 0).astype(float)
    oh = np.zeros((n, 2))
    oh[:, tid] = 1.0
    return np.hstack([X, oh]), y


def init_mlp(d, h, rng):
    """d here is the FULL input width, task indicator included."""
    return dict(W1=rng.normal(scale=1 / np.sqrt(d), size=(h, d)),
                b1=np.zeros(h),
                W2=rng.normal(scale=1 / np.sqrt(h), size=h),
                b2=0.0)


def forward(p, X):
    Z = X @ p["W1"].T + p["b1"]
    H = np.tanh(Z)
    logit = H @ p["W2"] + p["b2"]
    return Z, H, logit


def step(p, X, y, lr):
    n = len(X)
    Z, H, logit = forward(p, X)
    prob = 1.0 / (1.0 + np.exp(-logit))
    g = (prob - y) / n
    gW2 = H.T @ g
    gb2 = g.sum()
    dH = np.outer(g, p["W2"])
    dZ = dH * (1.0 - np.tanh(Z) ** 2)
    gW1 = dZ.T @ X
    gb1 = dZ.sum(0)
    p["W2"] -= lr * gW2
    p["b2"] -= lr * gb2
    p["W1"] -= lr * gW1
    p["b1"] -= lr * gb1
    return p


def accuracy(p, X, y):
    _, _, logit = forward(p, X)
    return float(((logit > 0).astype(float) == y).mean())


def flat(p):
    return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(), [p["b2"]]])


def unflat(v, like):
    o = {}
    i = 0
    for k in ("W1", "b1", "W2"):
        n = np.asarray(like[k]).size
        o[k] = v[i:i + n].reshape(np.asarray(like[k]).shape).copy()
        i += n
    o["b2"] = float(v[i])
    return o


def train(p, X, y, lr, steps):
    for _ in range(steps):
        p = step(p, X, y, lr)
    return p


# ---------------------------------------------------------------------------
# Figure 5 — forgetting
# ---------------------------------------------------------------------------

def figure5(d=24, h=64, n=800, lr=1.2, seed=0, trials=8):
    curves_a, curves_b, dists = [], [], []
    recov = []
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        A, B = make_task(d, rng), make_task(d, rng)
        XA, yA = sample(A, n, rng, d, 0)
        XB, yB = sample(B, n, rng, d, 1)
        XAt, yAt = sample(A, 2000, rng, d, 0)
        XBt, yBt = sample(B, 2000, rng, d, 1)

        p = init_mlp(d + 2, h, rng)
        p = train(p, XA, yA, lr, 3000)
        theta0 = flat(p)
        a0 = accuracy(p, XAt, yAt)

        ca, cb, dd = [a0], [accuracy(p, XBt, yBt)], [0.0]
        for k in range(60):
            p = train(p, XB, yB, lr, 25)
            ca.append(accuracy(p, XAt, yAt))
            cb.append(accuracy(p, XBt, yBt))
            dd.append(float(np.linalg.norm(flat(p) - theta0)))
        curves_a.append(ca)
        curves_b.append(cb)
        dists.append(dd)

        # how quickly does a little re-exposure bring task A back?
        rec = [accuracy(p, XAt, yAt)]
        for k in range(12):
            p = train(p, XA, yA, lr, 25)
            rec.append(accuracy(p, XAt, yAt))
        recov.append(rec)

    ca = np.mean(curves_a, 0)
    cb = np.mean(curves_b, 0)
    dd = np.mean(dists, 0)
    rec = np.mean(recov, 0)
    steps = np.arange(len(ca)) * 25

    lost = ca[0] - ca[-1]
    half = next((i for i, v in enumerate(ca) if v <= ca[0] - lost / 2), len(ca) - 1)
    print(f"  task A after training A: {ca[0]:.3f}; after training B: {ca[-1]:.3f}"
          f"   (lost {lost:.3f})")
    print(f"  half the loss arrives by step {steps[half]} of {steps[-1]} "
          f"({steps[half] / steps[-1]:.0%} of B's training)")
    print(f"  task B rises {cb[0]:.3f} -> {cb[-1]:.3f}")
    print(f"  correlation between distance moved and accuracy lost: "
          f"{np.corrcoef(dd, ca[0] - ca)[0, 1]:.3f}")
    back = next((i for i, v in enumerate(rec) if v >= ca[0] - 0.02), None)
    print(f"  re-exposure: {rec[0]:.3f} -> {rec[-1]:.3f} in {(len(rec) - 1) * 25} steps"
          + (f"; back within 2% of the original by step {back * 25}" if back is not None
             else "; did not return within 2%"))
    print(f"  the original training took 3000 steps, so recovery is "
          f"{3000 / max(back * 25, 25):.0f}x faster than learning it new")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.plot(steps, ca, "-", color=BLUE, lw=2.0, label="task A (learned first)")
    ax.plot(steps, cb, "-", color=ORANGE, lw=2.0, label="task B (training now)")
    ax.set_xlabel("steps of training on B")
    ax.set_ylabel("accuracy")
    ax.set_title("Forgetting is fast and early", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.plot(dd, ca, "o-", color=PURPLE, lw=1.7, ms=3.5)
    ax2.set_xlabel("distance moved in parameter space")
    ax2.set_ylabel("task A accuracy")
    ax2.set_title("It is a distance, not a schedule", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("What training on something else costs", fontsize=10.5, y=1.03)
    save(fig, "05-forgetting")
    return steps, ca, cb, dd, rec


# ---------------------------------------------------------------------------
# Figure 6 — replay
# ---------------------------------------------------------------------------

def figure6(d=24, h=64, n=800, lr=1.2, seed=1, trials=8):
    fracs = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    keepA, gotB = np.zeros((trials, len(fracs))), np.zeros((trials, len(fracs)))
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        A, B = make_task(d, rng), make_task(d, rng)
        XA, yA = sample(A, n, rng, d, 0)
        XB, yB = sample(B, n, rng, d, 1)
        XAt, yAt = sample(A, 2000, rng, d, 0)
        XBt, yBt = sample(B, 2000, rng, d, 1)
        base = init_mlp(d + 2, h, rng)
        base = train(base, XA, yA, lr, 3000)
        for j, f in enumerate(fracs):
            p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
            k_replay = int(round(f * n))
            for _ in range(1500):
                if k_replay:
                    idx = rng.integers(0, n, k_replay)
                    Xm = np.vstack([XB, XA[idx]])
                    ym = np.concatenate([yB, yA[idx]])
                else:
                    Xm, ym = XB, yB
                p = step(p, Xm, ym, lr)
            keepA[t, j] = accuracy(p, XAt, yAt)
            gotB[t, j] = accuracy(p, XBt, yBt)
    ka, gb = keepA.mean(0), gotB.mean(0)
    print("  replay fraction   task A kept   task B learned")
    for f, a, b in zip(fracs, ka, gb):
        print(f"  {f:13.0%}   {a:11.3f}   {b:14.3f}")
    gain = ka - ka[0]
    if gain[-1] > 1e-9:
        i5 = int(np.argmax(gain >= 0.5 * gain[-1]))
        print(f"  half the retention available is bought by {fracs[i5]:.0%} replay")
    print(f"  cost to task B across the whole sweep: {gb[0]:.3f} -> {gb[-1]:.3f} "
          f"({gb[0] - gb[-1]:+.3f})")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(fracs * 100, ka, "o-", color=BLUE, lw=2.0, ms=5, label="task A retained")
    ax.plot(fracs * 100, gb, "s-", color=ORANGE, lw=2.0, ms=4.5, label="task B learned")
    ax.set_xlabel("% of the batch replayed from task A")
    ax.set_ylabel("accuracy")
    ax.set_title("A small amount of replay buys most of the retention",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "06-replay")
    return fracs, ka, gb


# ---------------------------------------------------------------------------
# Figure 7 — averaging weights
# ---------------------------------------------------------------------------

def hard_sample(us, n, rng, d, tid):
    """A task that genuinely needs hidden structure: the label is the sign of a
    product of two projections, so no single unit computes it.

    The barrier question needs this. On a task solvable by one direction, a
    64-unit network is so over-parameterised that any two solutions are
    effectively the same function and interpolation is flat whatever the
    initialisation — which is a statement about the task, not about the
    landscape.
    """
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    z = np.tanh(3 * (X @ us[0])) * np.tanh(3 * (X @ us[1]))
    y = (z > 0).astype(float)
    oh = np.zeros((n, 2))
    oh[:, tid] = 1.0
    return np.hstack([X, oh]), y


def figure7(d=12, h=8, n=1500, lr=2.0, steps=6000, seed=2, trials=6):
    alphas = np.linspace(0, 1, 21)
    same = np.zeros((trials, len(alphas)))
    diff = np.zeros((trials, len(alphas)))
    widths = [8, 16, 32, 64, 128]
    bar_w = np.zeros((trials, len(widths)))

    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        us = [make_task(d, rng)[0] for _ in range(2)]
        X, y = hard_sample(us, n, rng, d, 0)
        Xt, yt = hard_sample(us, 3000, rng, d, 0)
        perm = rng.permutation(n)

        # SAME task, two runs. The only difference is where they started.
        shared = init_mlp(d + 2, h, rng)
        p1 = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in shared.items()},
                   X, y, lr, steps)
        p2 = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in shared.items()},
                   X[perm], y[perm], lr, steps)
        q1 = train(init_mlp(d + 2, h, rng), X, y, lr, steps)
        q2 = train(init_mlp(d + 2, h, rng), X, y, lr, steps)
        for j, a in enumerate(alphas):
            same[t, j] = accuracy(unflat((1 - a) * flat(p1) + a * flat(p2), p1), Xt, yt)
            diff[t, j] = accuracy(unflat((1 - a) * flat(q1) + a * flat(q2), q1), Xt, yt)

        for k, w in enumerate(widths):
            r1 = train(init_mlp(d + 2, w, rng), X, y, lr, steps)
            r2 = train(init_mlp(d + 2, w, rng), X, y, lr, steps)
            ends = 0.5 * (accuracy(r1, Xt, yt) + accuracy(r2, Xt, yt))
            mid = accuracy(unflat(0.5 * (flat(r1) + flat(r2)), r1), Xt, yt)
            bar_w[t, k] = ends - mid

    sm, df, bw = same.mean(0), diff.mean(0), bar_w.mean(0)
    mid = len(alphas) // 2
    print(f"  same task, h={h}")
    print(f"    shared init : ends {0.5*(sm[0]+sm[-1]):.3f}, midpoint {sm[mid]:.3f}"
          f"   barrier {0.5*(sm[0]+sm[-1]) - sm[mid]:+.3f}")
    print(f"    distinct    : ends {0.5*(df[0]+df[-1]):.3f}, midpoint {df[mid]:.3f}"
          f"   barrier {0.5*(df[0]+df[-1]) - df[mid]:+.3f}")
    print(f"  barrier against width (distinct initialisations):")
    for w, b in zip(widths, bw):
        print(f"    width {w:4d}: {b:+.3f}")
    print(f"  the barrier shrinks {bw[0]:.3f} -> {bw[-1]:.3f} over a "
          f"{widths[-1] // widths[0]}x width range")

    # merging: two DIFFERENT tasks from one initialisation, which is the soup case
    merge = np.zeros((trials, len(alphas), 2))
    for t in range(trials):
        rng = np.random.default_rng(seed + 100 + t)
        A, B = make_task(24, rng), make_task(24, rng)
        XA, yA = sample(A, 800, rng, 24, 0)
        XB, yB = sample(B, 800, rng, 24, 1)
        XAt, yAt = sample(A, 2000, rng, 24, 0)
        XBt, yBt = sample(B, 2000, rng, 24, 1)
        base = init_mlp(26, 64, rng)
        pa = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()},
                   XA, yA, 1.2, 2500)
        pb = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()},
                   XB, yB, 1.2, 2500)
        for j, a in enumerate(alphas):
            mm = unflat((1 - a) * flat(pa) + a * flat(pb), pa)
            merge[t, j] = (accuracy(mm, XAt, yAt), accuracy(mm, XBt, yBt))
    mg = merge.mean(0)
    print(f"  merging two different fine-tunes of one base model:")
    print(f"    endpoints  A {mg[0,0]:.3f}/{mg[0,1]:.3f}   B {mg[-1,0]:.3f}/{mg[-1,1]:.3f}")
    print(f"    midpoint   A {mg[mid,0]:.3f}/{mg[mid,1]:.3f}"
          f"   (mean {mg[mid].mean():.3f} against {max(mg[0].mean(), mg[-1].mean()):.3f} "
          f"for the better endpoint)")

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(6.9, 2.6))
    ax.plot(alphas, sm, "-", color=BLUE, lw=2.0, label="shared init")
    ax.plot(alphas, df, "-", color=ORANGE, lw=2.0, label="distinct inits")
    ax.set_xlabel(r"interpolation $\alpha$")
    ax.set_ylabel("accuracy")
    ax.set_title("Same task, two runs", fontsize=9.5, pad=7)
    ax.legend(frameon=False, fontsize=7.4)
    tidy(ax)

    ax2.semilogx(widths, bw, "o-", color=ORANGE, lw=1.9, ms=4.5, base=2)
    ax2.axhline(0, color=GRAY, ls="--", lw=1.1)
    ax2.set_xlabel("hidden width")
    ax2.set_ylabel("barrier height")
    ax2.set_title("Width flattens it", fontsize=9.5, pad=7)
    tidy(ax2)

    ax3.plot(alphas, mg[:, 0], "-", color=BLUE, lw=1.9, label="task A")
    ax3.plot(alphas, mg[:, 1], "-", color=ORANGE, lw=1.9, label="task B")
    ax3.plot(alphas, mg.mean(1), "--", color=GRAY, lw=1.4, label="mean")
    ax3.set_xlabel(r"interpolation $\alpha$")
    ax3.set_title("Merging two fine-tunes", fontsize=9.5, pad=7)
    ax3.legend(frameon=False, fontsize=7.4)
    tidy(ax3)
    fig.suptitle("When averaging weights works, and when it destroys both",
                 fontsize=10.4, y=1.04)
    save(fig, "07-averaging")
    return alphas, sm, df, bw, mg


# ---------------------------------------------------------------------------
# Figure 8 — task arithmetic
# ---------------------------------------------------------------------------

def figure8(d=24, h=64, n=800, lr=1.2, seed=3, trials=10):
    """tau = theta_finetuned - theta_pretrained, treated as a vector."""
    cos, add_acc, base_acc, cross_acc, neg_self, neg_other = [], [], [], [], [], []
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        # a shared "pretraining" task, then two fine-tunes from it
        P = make_task(d, rng)
        XP, yP = sample(P, n, rng, d, 0)
        pre = train(init_mlp(d + 2, h, rng), XP, yP, lr, 2000)
        theta0 = flat(pre)

        A, B = make_task(d, rng), make_task(d, rng)
        XA, yA = sample(A, n, rng, d, 0)
        XB, yB = sample(B, n, rng, d, 1)
        XAt, yAt = sample(A, 2000, rng, d, 0)
        XBt, yBt = sample(B, 2000, rng, d, 1)

        pa = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in pre.items()},
                   XA, yA, lr, 2000)
        pb = train({k: (v.copy() if hasattr(v, "copy") else v) for k, v in pre.items()},
                   XB, yB, lr, 2000)
        ta, tb = flat(pa) - theta0, flat(pb) - theta0
        cos.append(float(ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb))))

        both = unflat(theta0 + ta + tb, pre)
        add_acc.append((accuracy(both, XAt, yAt), accuracy(both, XBt, yBt)))
        base_acc.append((accuracy(pa, XAt, yAt), accuracy(pb, XBt, yBt)))
        # what each fine-tune already scores on the OTHER task. Without this,
        # "the sum is good at both" is unfalsifiable: when the two task vectors
        # happen to align, either model alone is already good at both.
        cross_acc.append((accuracy(pb, XAt, yAt), accuracy(pa, XBt, yBt)))

        # negation: remove A, and check B survives
        minus = unflat(flat(pa) - ta, pre)          # = theta0, the control
        minus_b = unflat(flat(pb) - ta, pre)        # remove A from a model holding B
        neg_self.append(accuracy(minus_b, XAt, yAt))
        neg_other.append(accuracy(minus_b, XBt, yBt))

    cos = np.array(cos)
    add_acc = np.array(add_acc)
    base_acc = np.array(base_acc)
    print(f"  cosine between task vectors: mean {cos.mean():+.3f}, "
          f"range [{cos.min():+.3f}, {cos.max():+.3f}]")
    print(f"  each fine-tune on its own task: A {base_acc[:,0].mean():.3f}, "
          f"B {base_acc[:,1].mean():.3f}")
    cross_acc = np.array(cross_acc)
    print(f"  each fine-tune on the OTHER task: A {cross_acc[:,0].mean():.3f}, "
          f"B {cross_acc[:,1].mean():.3f}   (the baseline addition must beat)")
    print(f"  theta0 + tauA + tauB:           A {add_acc[:,0].mean():.3f}, "
          f"B {add_acc[:,1].mean():.3f}")
    print(f"  retained fraction: A {add_acc[:,0].mean() / base_acc[:,0].mean():.3f}, "
          f"B {add_acc[:,1].mean() / base_acc[:,1].mean():.3f}")
    lift = add_acc.mean(1) - cross_acc.mean(1)
    print(f"  lift over the cross-task baseline: {lift.mean():+.3f}")
    r = np.corrcoef(np.abs(cos), add_acc.mean(1))[0, 1]
    r2 = np.corrcoef(np.abs(cos), lift)[0, 1]
    print(f"  correlation of |cosine| with raw accuracy of the sum: {r:+.3f}")
    print(f"  correlation of |cosine| with the LIFT over baseline:  {r2:+.3f}"
          f"   (the second is the one that is about the arithmetic)")
    print(f"  subtracting tauA from a B-model: A falls to "
          f"{np.mean(neg_self):.3f}, B holds at {np.mean(neg_other):.3f}")

    order = np.argsort(np.abs(cos))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.scatter(np.abs(cos)[order], add_acc.mean(1)[order], s=34, color=BLUE,
               zorder=3, label="sum")
    ax.scatter(np.abs(cos)[order], cross_acc.mean(1)[order], s=30, color=PINK,
               zorder=3, marker="s", label="one model, other task")
    ax.axhline(base_acc.mean(), color=GRAY, ls="--", lw=1.2,
               label="each model on its own task")
    ax.set_xlabel(r"$|\cos(\tau_A,\tau_B)|$")
    ax.set_ylabel("mean accuracy")
    ax.set_title("Adding two task vectors", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    labels = ["A model\non A", "B model\non B", "sum\non A", "sum\non B",
              "B −$\\tau_A$\non A", "B −$\\tau_A$\non B"]
    vals = [base_acc[:, 0].mean(), base_acc[:, 1].mean(),
            add_acc[:, 0].mean(), add_acc[:, 1].mean(),
            np.mean(neg_self), np.mean(neg_other)]
    cols = [BLUE, ORANGE, BLUE, ORANGE, PINK, ORANGE]
    ax2.bar(range(6), vals, color=cols, zorder=3)
    ax2.axhline(0.5, color=GRAY, ls="--", lw=1.1)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(labels, fontsize=6.4)
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.35, 1.02)
    ax2.set_title("Adding, and subtracting", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("Behaviour as a direction in weight space", fontsize=10.5, y=1.03)
    save(fig, "08-task-arithmetic")
    return cos, add_acc, base_acc, cross_acc, neg_self, neg_other


# ---------------------------------------------------------------------------
# Figure 9 — editing a fact
# ---------------------------------------------------------------------------

def figure9(dim=128, n_facts=90, seed=4, trials=6):
    """A rank-one edit to a linear associative memory.

    This is the construction the published editing methods use: a linear layer
    is read as a key-value store, and changing one association is a rank-one
    update chosen to map one key to a new value. What is measured is the
    collateral damage — the other associations — as edits accumulate.
    """
    counts = [1, 2, 5, 10, 20, 30, 45, 60, 75, 85]
    keep = np.zeros((trials, len(counts)))
    edited = np.zeros((trials, len(counts)))
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        K = rng.normal(size=(n_facts, dim)) / np.sqrt(dim)
        V = rng.normal(size=(n_facts, dim)) / np.sqrt(dim)
        M0 = np.linalg.lstsq(K, V, rcond=None)[0]
        base = float(np.mean(np.sum((K @ M0 - V) ** 2, 1)))
        # With more facts than dimensions the least-squares fit does not store
        # them at all, and the "collateral damage" measured afterwards is mostly
        # the fit residual. n_facts < dim makes the baseline exact, so every
        # error below is caused by an edit.
        assert base < 1e-18, f"the store does not hold its own facts: {base:.3e}"
        for j, c in enumerate(counts):
            M = M0.copy()
            targets = rng.choice(n_facts, c, replace=False)
            new_v = rng.normal(size=(c, dim)) / np.sqrt(dim)
            for i, idx in enumerate(targets):
                k = K[idx]
                resid = new_v[i] - k @ M
                M = M + np.outer(k / (k @ k), resid)   # exact rank-one edit
            others = np.setdiff1d(np.arange(n_facts), targets)
            err_other = np.mean(np.sum((K[others] @ M - V[others]) ** 2, 1))
            err_edit = np.mean(np.sum((K[targets] @ M - new_v) ** 2, 1))
            keep[t, j] = err_other
            edited[t, j] = err_edit
    ko, ke = keep.mean(0), edited.mean(0)
    scale = float(np.mean(np.sum(V ** 2, 1)))
    print(f"  store of {n_facts} facts in {dim} dimensions; mean squared value "
          f"norm {scale:.3f}")
    print(f"  edits   error on edited facts   error on the rest   as % of the signal")
    for c, a, b in zip(counts, ke, ko):
        print(f"  {c:5d}   {a:20.2e}   {b:17.4f}   {100 * b / scale:16.1f}%")
    ruin = next((c for c, b in zip(counts, ko) if b > scale), None)
    print(f"  the rest is worse than predicting zero at {ruin} edits, "
          f"against a nominal capacity of {dim} directions")
    lin = ko[0] * np.array(counts) / counts[0]
    print(f"  growth against linear-in-edits from the first point: measured "
          f"{ko[-1]:.4f} vs linear {lin[-1]:.4f} ({ko[-1] / lin[-1]:.2f}x)")
    sl = float(np.polyfit(np.log(counts), np.log(np.maximum(ko, 1e-12)), 1)[0])
    print(f"  log-log slope of the damage against edit count: {sl:.2f} "
          f"({'super-linear' if sl > 1.15 else 'linear or sub-linear'})")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.loglog(counts, ko, "o-", color=ORANGE, lw=2.0, ms=5, label="unedited facts")
    ax.loglog(counts, np.maximum(ke, 1e-16), "o-", color=BLUE, lw=2.0, ms=5,
              label="edited facts")
    ax.axhline(scale, color=GRAY, ls="--", lw=1.2)
    ax.text(counts[0], scale * 1.25, "as bad as predicting zero", fontsize=7.6,
            color=GRAY)
    ax.set_xlabel("edits applied in sequence")
    ax.set_ylabel("mean squared error")
    ax.set_title("Each edit is exact; together they are not", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    tidy(ax)
    save(fig, "09-editing")
    return counts, ko, ke, scale


if __name__ == "__main__":
    for i, fn in enumerate([figure5, figure6, figure7, figure8, figure9], 5):
        print(f"Figure {i}")
        fn()
    print("done")
