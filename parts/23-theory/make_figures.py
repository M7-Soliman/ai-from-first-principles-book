"""
Figures for Part XXIII — Learning Theory.

The theory here makes claims that are checkable, so they are checked. A
concentration bound is a statement about a tail probability, and the tail
probability can be simulated. A VC dimension is a statement about which
labellings are realisable, and realisability is a linear program. A
generalisation bound is a number, and so is the gap it bounds.

    python3 make_figures.py

Runtime: about three minutes, all of it in figures 2 and 4.
"""
import os
import re
import itertools

import numpy as np
from scipy.optimize import linprog
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


# ----------------------------------------------------- 1. concentration, exactly

def fig_concentration():
    """Hoeffding and Bernstein are upper bounds on a tail probability. Simulate
    the tail and compare. Both must hold; the question the figure answers is by
    how much they are exceeded, and when the extra term Bernstein carries is
    worth having."""
    print("\n[1] concentration bounds against the tail they bound")
    rng = np.random.default_rng(0)
    N, REPS = 200, 400_000

    eps = np.linspace(0.02, 0.30, 24)
    rows = {}
    for p in (0.5, 0.05):
        draws = rng.binomial(N, p, REPS) / N
        emp = np.array([np.mean(np.abs(draws - p) >= e) for e in eps])
        hoeff = np.minimum(2 * np.exp(-2 * N * eps ** 2), 1.0)
        var = p * (1 - p)
        bern = np.minimum(2 * np.exp(-N * eps ** 2 / (2 * var + 2 * eps / 3)), 1.0)
        rows[p] = (emp, hoeff, bern)
        i = int(np.argmin(abs(eps - 0.10)))
        print(f"    p={p:<5}  at eps=0.10:  empirical {emp[i]:.2e}   "
              f"Hoeffding {hoeff[i]:.2e} ({hoeff[i]/max(emp[i],1e-12):>7.1f}x)   "
              f"Bernstein {bern[i]:.2e} ({bern[i]/max(emp[i],1e-12):>7.1f}x)")
        assert np.all(hoeff >= emp - 1e-9), "Hoeffding violated — impossible"
        assert np.all(bern >= emp - 1e-9), "Bernstein violated — impossible"
    print("    both bounds hold at every point, as they must.")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, p in zip(axes, (0.5, 0.05)):
        emp, hoeff, bern = rows[p]
        ax.semilogy(eps, np.maximum(emp, 1 / REPS), "o-", color=INK, lw=1.6, ms=3.2,
                    label="the actual tail")
        ax.semilogy(eps, hoeff, color=ORANGE, lw=1.9, label="Hoeffding")
        ax.semilogy(eps, bern, color=BLUE, lw=1.9, label="Bernstein")
        ax.axhline(1 / REPS, color=GRAY, lw=0.9, ls=":")
        ax.set_ylim(1 / REPS / 3, 3)
        ax.set_xlabel("deviation $\\varepsilon$")
        ax.set_ylabel("$P(|\\hat{p} - p| \\geq \\varepsilon)$")
        ax.set_title(f"$p = {p}$, $n = {N}$", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)
    save(fig, "01-concentration")
    print("    Hoeffding ignores the variance, so its curve is identical in both")
    print("    panels. Bernstein uses it, which costs at p=0.5 — where the variance")
    print("    is at its largest, Bernstein is the LOOSER of the two — and wins by")
    print("    four orders of magnitude at p=0.05. Neither dominates.")


# ------------------------------------------------------ 2. VC dimension, measured

def _separable(X, y):
    """Is this labelling realisable by a linear threshold function? Exactly,
    by linear programming: find w, b with y_i (w.x_i + b) >= 1."""
    n, d = X.shape
    s = np.where(y > 0, 1.0, -1.0)
    # variables [w (d), b (1)]; constraints -s_i (w.x_i + b) <= -1
    A = -np.column_stack([X * s[:, None], s])
    res = linprog(c=np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n),
                  bounds=[(None, None)] * (d + 1), method="highs")
    return res.status == 0


def fig_vc():
    """VC dimension is defined by which labellings a class can realise, so it is
    checkable by enumeration. Linear classifiers in d dimensions should shatter
    d+1 points and no more. The figure verifies that, and shows the collapse
    past the threshold."""
    print("\n[2] shattering, by enumeration")
    rng = np.random.default_rng(1)
    results = {}
    for d in (2, 3):
        fracs = []
        for m in range(2, 8):
            trials, ok = 40, 0
            for t in range(trials):
                X = rng.standard_normal((m, d))
                allsplit = all(_separable(X, np.array(lab))
                               for lab in itertools.product([0, 1], repeat=m))
                ok += allsplit
            fracs.append(ok / trials)
            print(f"    d={d}  m={m}:  point sets shattered {ok}/{trials} = {ok/trials:.2f}")
        results[d] = fracs
        vc = max([m for m, f in zip(range(2, 8), fracs) if f > 0.5], default=0)
        print(f"    -> largest m always shattered: {vc}   (theory says d+1 = {d+1})")

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ms = list(range(2, 8))
    for d, col in ((2, BLUE), (3, GREEN)):
        ax.plot(ms, results[d], "o-", color=col, lw=1.9, ms=5,
                label=f"$d = {d}$  (VC dimension $d+1 = {d+1}$)")
        ax.axvline(d + 1, color=col, lw=1.0, ls=":")
    ax.set_xlabel("number of points $m$")
    ax.set_ylabel("fraction of point sets shattered")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Where a linear classifier stops being able to fit anything",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "02-shattering")


# ------------------------------------------ 3. the bound against what it bounds

def fig_bound_vs_gap():
    """A generalisation bound is a number. So is the gap. Compare them, first
    where the theory was designed to work and then where it is used."""
    print("\n[3] the bound, against the gap it bounds")
    rng = np.random.default_rng(2)
    DELTA = 0.05

    def vc_bound(vcdim, n, delta=DELTA):
        # The standard VC generalisation bound. The log term goes NEGATIVE once
        # the dimension exceeds twice the sample size, which makes the square
        # root of a negative number — the first version of this figure printed
        # nan for exactly the sizes the figure exists to discuss. Flooring the
        # log at zero keeps it finite and, in that regime, it reduces to
        # sqrt(vcdim/n), which is already greater than one and therefore says
        # nothing about an error rate.
        return np.sqrt((vcdim * (np.maximum(np.log(2 * n / vcdim), 0.0) + 1)
                        + np.log(4 / delta)) / n)

    ns = np.unique(np.round(np.logspace(2, 4.3, 12)).astype(int))
    d = 10
    gaps, bounds = [], []
    for n in ns:
        w = rng.standard_normal(d)
        Xtr = rng.standard_normal((n, d)); ytr = (Xtr @ w > 0).astype(float)
        Xte = rng.standard_normal((20000, d)); yte = (Xte @ w > 0).astype(float)
        # fit by logistic regression on the separable data (perceptron-like)
        ww = np.zeros(d)
        for _ in range(400):
            m = Xtr @ ww
            g = Xtr.T @ ((1 / (1 + np.exp(-m))) - ytr) / n
            ww -= 2.0 * g
        tr = np.mean((Xtr @ ww > 0) != ytr)
        te = np.mean((Xte @ ww > 0) != yte)
        gaps.append(te - tr); bounds.append(vc_bound(d + 1, n))
    for n, g, b in list(zip(ns, gaps, bounds))[::3]:
        print(f"    linear, d={d}:  n={n:<6d} gap {g:+.4f}   VC bound {b:.3f}   "
              f"ratio {b/max(g,1e-6):.0f}x")

    # now the same bound applied where it is usually quoted
    print("    the same bound for networks, using a parameter-count VC estimate:")
    net_rows = []
    for params, n in ((25_000, 50_000), (1_000_000, 50_000), (100_000_000, 1_000_000)):
        b = vc_bound(params, n)
        net_rows.append((params, n, b))
        print(f"      {params:>11,} parameters, n={n:>9,}:  bound {b:6.1f}"
              f"   {'VACUOUS — exceeds 1' if b > 1 else ''}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.loglog(ns, bounds, "o-", color=ORANGE, lw=1.9, ms=4, label="VC bound")
    ax.loglog(ns, np.maximum(gaps, 1e-4), "o-", color=BLUE, lw=1.9, ms=4,
              label="measured gap")
    ax.set_xlabel("training set size $n$")
    ax.set_ylabel("generalisation gap")
    ax.set_title(f"Linear classifier, $d = {d}$", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    labels = [f"{p/1000:.0f}k\nn={n//1000}k" if p < 1e6 else
              f"{p/1e6:.0f}M\nn={n//1000}k" for p, n, _ in net_rows]
    vals = [b for _, _, b in net_rows]
    ax.bar(np.arange(3), vals, 0.55, color=ORANGE)
    ax.axhline(1.0, color=INK, lw=1.4, ls="--")
    ax.text(2.45, 1.25, "an error rate cannot exceed 1", fontsize=8,
            color=INK, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value of the bound")
    ax.set_title("The same bound, at the sizes people use",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "03-bound-vs-gap")


# ------------------------------------------------------------ 4. double descent

def fig_double_descent():
    """The classical curve says test error rises past the interpolation
    threshold. Sweep model size through it and see."""
    print("\n[4] double descent")
    rng = np.random.default_rng(3)
    N, D_IN, NOISE, REPS = 120, 40, 0.35, 24

    w = rng.standard_normal(D_IN) / np.sqrt(D_IN)
    ps = np.unique(np.round(np.logspace(0.5, 3.1, 26)).astype(int))
    curve = []
    for p in ps:
        errs = []
        for r in range(REPS):
            F = rng.standard_normal((D_IN, p)) / np.sqrt(D_IN)
            Xtr = rng.standard_normal((N, D_IN))
            ytr = Xtr @ w + NOISE * rng.standard_normal(N)
            Xte = rng.standard_normal((4000, D_IN))
            yte = Xte @ w + NOISE * rng.standard_normal(4000)
            Ztr = np.tanh(Xtr @ F); Zte = np.tanh(Xte @ F)
            # minimum-norm least squares — the ridgeless interpolant
            beta = np.linalg.pinv(Ztr) @ ytr
            errs.append(np.mean((Zte @ beta - yte) ** 2))
        curve.append(np.median(errs))
    curve = np.array(curve)
    peak = ps[int(np.argmax(curve))]
    tail = curve[ps > 2 * N].min()
    classical = curve[ps < N].min()
    print(f"    interpolation threshold at p = n = {N}")
    print(f"    peak test error at p = {peak}   (value {curve.max():.3f})")
    print(f"    best below the threshold: {classical:.3f} at p = {ps[ps < N][int(np.argmin(curve[ps < N]))]}")
    print(f"    best above it:            {tail:.3f}")
    print(f"    -> the second descent goes {'BELOW' if tail < classical else 'not below'} "
          f"the classical optimum")

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.loglog(ps, curve, "o-", color=BLUE, lw=1.9, ms=4)
    ax.axvline(N, color=ORANGE, lw=1.4, ls="--")
    ax.text(N * 1.1, curve.max() * 0.8, "interpolation\nthreshold  $p = n$",
            fontsize=8, color=ORANGE)
    ax.axhline(classical, color=GREEN, lw=1.1, ls=":")
    ax.text(ps[-1], classical * 1.08, "best classical fit", fontsize=8,
            color=GREEN, ha="right")
    ax.set_xlabel("number of random features $p$")
    ax.set_ylabel("test error")
    ax.set_title("Test error through the interpolation threshold",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "04-double-descent")


# ------------------------------------------------- 5. what the optimiser prefers

def fig_implicit():
    """When infinitely many solutions fit the training data exactly, which one
    you get is decided by the algorithm rather than the objective. Measure the
    norm and the test error of the one gradient descent finds, against other
    exact interpolants of the same data."""
    print("\n[5] which interpolant the optimiser chooses")
    rng = np.random.default_rng(4)
    N, D = 40, 300
    w = np.zeros(D); w[:8] = rng.standard_normal(8)
    X = rng.standard_normal((N, D)); y = X @ w
    Xte = rng.standard_normal((5000, D)); yte = Xte @ w

    # gradient descent from zero
    beta = np.zeros(D)
    for _ in range(60_000):
        beta -= 0.02 * X.T @ (X @ beta - y) / N
    gd_tr = np.mean((X @ beta - y) ** 2)

    minnorm = np.linalg.pinv(X) @ y
    null = np.linalg.svd(X)[2][N:]                    # basis for the null space

    rows = [("gradient descent from 0", beta),
            ("minimum-norm solution", minnorm)]
    for k, scale in ((0, 3.0), (1, 12.0)):
        v = null.T @ rng.standard_normal(null.shape[0])
        rows.append((f"another exact fit ({'small' if k == 0 else 'large'})",
                     minnorm + scale * v / np.linalg.norm(v) * np.linalg.norm(minnorm)))
    out = []
    for name, b in rows:
        tr = np.mean((X @ b - y) ** 2)
        te = np.mean((Xte @ b - yte) ** 2)
        out.append((name, np.linalg.norm(b), tr, te))
        print(f"    {name:28s} norm {np.linalg.norm(b):7.3f}   "
              f"train {tr:.2e}   test {te:8.3f}")
    print(f"    gradient descent and the minimum-norm solution agree to "
          f"{np.linalg.norm(beta - minnorm):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    names = [r[0].replace(" (", "\n(") for r in out]
    ax = axes[0]
    ax.bar(np.arange(len(out)), [r[1] for r in out], 0.55,
           color=[GREEN, GREEN, GRAY, ORANGE])
    ax.set_xticks(np.arange(len(out))); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel("$\\|\\beta\\|_2$")
    ax.set_title("Norm of the solution", fontsize=10.5, color=INK)
    tidy(ax)
    ax = axes[1]
    ax.bar(np.arange(len(out)), [r[3] for r in out], 0.55,
           color=[GREEN, GREEN, GRAY, ORANGE])
    ax.set_xticks(np.arange(len(out))); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel("test error")
    ax.set_title("All four fit the training data exactly",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "05-implicit")


# --------------------------------------------------------- 6. covariate shift

def fig_covariate_shift():
    """Under covariate shift the conditional is unchanged and only the input
    distribution moves. The textbook fix is importance weighting. Measure when
    it is needed — which turns out to depend on whether the model can represent
    the truth."""
    print("\n[6] covariate shift, with and without misspecification")
    rng = np.random.default_rng(5)
    N, REPS = 400, 300

    def run(misspecified, shift):
        errs_u, errs_w = [], []
        for _ in range(REPS):
            xtr = rng.normal(0.0, 1.0, N)
            xte = rng.normal(shift, 1.0, 6000)
            f = lambda x: (x + 0.6 * x ** 2 if misspecified else x)
            ytr = f(xtr) + 0.3 * rng.standard_normal(N)
            yte = f(xte) + 0.3 * rng.standard_normal(len(xte))
            A = np.column_stack([np.ones(N), xtr])          # always fit a LINE
            wts = (np.exp(-(xtr - shift) ** 2 / 2) / np.exp(-xtr ** 2 / 2))
            wts = np.clip(wts, 0, 50)
            bu = np.linalg.lstsq(A, ytr, rcond=None)[0]
            W = np.sqrt(wts)[:, None]
            bw = np.linalg.lstsq(A * W, ytr * np.sqrt(wts), rcond=None)[0]
            B = np.column_stack([np.ones(len(xte)), xte])
            errs_u.append(np.mean((B @ bu - yte) ** 2))
            errs_w.append(np.mean((B @ bw - yte) ** 2))
        return np.median(errs_u), np.median(errs_w)

    shifts = np.linspace(0, 2.5, 9)
    res = {}
    for mis in (False, True):
        u = [run(mis, s) for s in shifts]
        res[mis] = (np.array([a for a, _ in u]), np.array([b for _, b in u]))
        lab = "misspecified" if mis else "well specified"
        for s, (a, b) in list(zip(shifts, u))[::3]:
            print(f"    {lab:14s} shift {s:.2f}:  unweighted {a:7.3f}   weighted {b:7.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, mis in zip(axes, (False, True)):
        u, w = res[mis]
        ax.plot(shifts, u, "o-", color=ORANGE, lw=1.9, ms=4, label="unweighted")
        ax.plot(shifts, w, "o-", color=GREEN, lw=1.9, ms=4, label="importance weighted")
        ax.set_xlabel("shift in the test distribution's mean")
        ax.set_ylabel("test error")
        ax.set_title("Model can represent the truth" if not mis
                     else "Model cannot", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)
    save(fig, "06-covariate-shift")


# --------------------------------------------------------- 7. worst-group risk

def fig_worst_group():
    """Average accuracy hides a group that the objective was happy to sacrifice.
    Build data with a spurious feature that holds for most units and fails for a
    minority, then compare minimising the average against minimising the worst
    group."""
    print("\n[7] average against worst group")
    rng = np.random.default_rng(6)
    N, D = 6000, 20

    def make(n, majority=0.95):
        y = rng.integers(0, 2, n) * 2 - 1
        spur = np.where(rng.random(n) < majority, y, -y)     # agrees with y mostly
        X = rng.standard_normal((n, D)) * 0.4
        X[:, 0] = 2.5 * spur + 0.3 * rng.standard_normal(n)   # easy, spurious
        X[:, 1] = 0.8 * y + 0.9 * rng.standard_normal(n)      # hard, causal
        return X, y, (spur == y)

    Xtr, ytr, gtr = make(N)
    Xte, yte, gte = make(20000, majority=0.5)                 # groups balanced at test

    def fit(group_dro):
        w = np.zeros(D); b = 0.0
        for it in range(4000):
            m = ytr * (Xtr @ w + b)
            l = np.log1p(np.exp(-np.clip(m, -30, 30)))
            if group_dro:
                # weight the two groups by a softmax of their current losses, so
                # whichever group is doing worse gets the gradient
                alpha = np.exp(3 * np.array([l[gtr].mean(), l[~gtr].mean()]))
                alpha /= alpha.sum()
                wt = np.where(gtr, alpha[0] / max(gtr.mean(), 1e-9),
                              alpha[1] / max((~gtr).mean(), 1e-9))
            else:
                wt = np.ones(N)
            s = -ytr / (1 + np.exp(np.clip(m, -30, 30)))
            g = (Xtr * (s * wt)[:, None]).mean(0)
            gb = (s * wt).mean()
            w -= 0.5 * g; b -= 0.5 * gb
        return w, b

    rows = []
    for name, dro in (("minimise the average", False), ("minimise the worst group", True)):
        w, b = fit(dro)
        pred = np.sign(Xte @ w + b)
        acc = np.mean(pred == yte)
        a_maj = np.mean(pred[gte] == yte[gte]); a_min = np.mean(pred[~gte] == yte[~gte])
        rows.append((name, acc, a_maj, a_min))
        print(f"    {name:26s} overall {acc:.3f}   aligned {a_maj:.3f}   "
              f"conflicting {a_min:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(2); wd = 0.26
    ax.bar(x - wd, [r[1] for r in rows], wd, color=GRAY, label="overall")
    ax.bar(x, [r[2] for r in rows], wd, color=GREEN, label="spurious feature agrees")
    ax.bar(x + wd, [r[3] for r in rows], wd, color=ORANGE, label="spurious feature conflicts")
    ax.axhline(0.5, color=INK, lw=1.1, ls=":")
    ax.text(1.42, 0.52, "chance", fontsize=8, color=INK, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
    ax.set_title("What the average conceals", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    tidy(ax)
    save(fig, "07-worst-group")


if __name__ == "__main__":
    fig_concentration()
    fig_vc()
    fig_bound_vs_gap()
    fig_double_descent()
    fig_implicit()
    fig_covariate_shift()
    fig_worst_group()
