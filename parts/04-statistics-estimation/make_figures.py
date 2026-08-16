"""
Figures for Part V — Statistics and Estimation.

Everything is simulated. The coverage figure really does count how many intervals
contain the truth; the p-value histogram is real p-values from real null data; the
double-descent curve is a real sweep of a real model.

    python3 make_figures.py           # writes into ../../book/src/figures/st/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "st"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
LIGHT, INK = "#cbd5e1", "#1e293b"

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

MAX_PT = 900


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


# =============================================================== fig 1 ======
def fig_estimator():
    """An estimator is a random variable: different samples give different answers."""
    rng = np.random.default_rng(0)
    true_mu = 5.0
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))

    ax = axes[0]
    for i in range(6):
        s = rng.normal(true_mu, 2.0, size=12)
        ax.plot(s, np.full(12, i), "o", color=BLUE, alpha=0.45, ms=4)
        ax.plot([s.mean()], [i], "D", color=ORANGE, ms=6, zorder=6)
    ax.axvline(true_mu, color=GREEN, lw=2, zorder=5)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"sample {i+1}" for i in range(6)])
    ax.set_xlabel("value")
    ax.set_title("six samples, six different estimates", fontsize=10, pad=8)
    ax.text(true_mu + 0.15, 5.35, "truth", color=GREEN, fontsize=8.5)
    tidy(ax)

    ax = axes[1]
    ests = np.array([rng.normal(true_mu, 2.0, size=12).mean() for _ in range(20000)])
    ax.hist(ests, bins=60, density=True, color=BLUE, alpha=0.55)
    ax.axvline(true_mu, color=GREEN, lw=2)
    ax.axvline(ests.mean(), color=ORANGE, lw=1.6, ls="--")
    ax.set_xlabel(r"value of $\hat{\mu}$")
    ax.set_yticks([])
    ax.set_title("the sampling distribution of the estimator", fontsize=10, pad=8)
    tidy(ax, grid=False)
    save(fig, "01-estimator")


# =============================================================== fig 3 ======
def fig_bias_variance_quadrants():
    """Four estimators of the same quantity, differing in bias and variance."""
    rng = np.random.default_rng(1)
    truth = 0.0
    n = 4000
    ests = {
        "low bias, low variance": rng.normal(0.0, 0.25, n),
        "low bias, high variance": rng.normal(0.0, 0.95, n),
        "high bias, low variance": rng.normal(1.3, 0.25, n),
        "high bias, high variance": rng.normal(1.3, 0.95, n),
    }
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))
    for ax, (lab, e) in zip(axes, ests.items()):
        ax.hist(e, bins=45, density=True, color=BLUE, alpha=0.55)
        ax.axvline(truth, color=GREEN, lw=2)
        ax.axvline(e.mean(), color=ORANGE, lw=1.6, ls="--")
        mse = np.mean((e - truth) ** 2)
        ax.set_title(f"{lab}\nMSE = {mse:.2f}", fontsize=8.8, pad=5)
        ax.set_xlim(-3.2, 4.4); ax.set_yticks([]); ax.set_xticks([])
        tidy(ax, grid=False)
    save(fig, "03-bias-variance-quadrants")


# =============================================================== fig 2 ======
def fig_root_n():
    """The standard error shrinks like 1/sqrt(n). Measured, not asserted."""
    rng = np.random.default_rng(2)
    ns = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ses = []
    for n in ns:
        ests = rng.normal(0, 1, size=(4000, n)).mean(axis=1)
        ses.append(ests.std())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.85))

    ax = axes[0]
    for n, c in [(4, LIGHT), (16, BLUE), (256, ORANGE)]:
        ests = rng.normal(0, 1, size=(20000, n)).mean(axis=1)
        ax.hist(ests, bins=70, density=True, alpha=0.6, color=c, label=f"n = {n}")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_xlim(-1.8, 1.8); ax.set_yticks([])
    ax.set_xlabel(r"$\bar{x}$")
    ax.set_title("more data, tighter estimates", fontsize=10, pad=8)
    tidy(ax, grid=False)

    ax = axes[1]
    ax.loglog(ns, ses, "o", color=BLUE, ms=6, label="measured")
    ax.loglog(ns, 1 / np.sqrt(ns), color=ORANGE, lw=1.8, ls="--",
              label=r"$1/\sqrt{n}$")
    ax.set_xlabel("sample size $n$"); ax.set_ylabel("standard error")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("four times the data, half the error", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "02-root-n")


# =============================================================== fig 4 ======
def fig_coverage():
    """A 95% interval covers the truth 95% of the time. Counted."""
    rng = np.random.default_rng(3)
    true_mu, n, reps = 5.0, 15, 100
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    covered = 0
    for i in range(reps):
        s = rng.normal(true_mu, 2.0, n)
        se = s.std(ddof=1) / np.sqrt(n)
        lo, hi = s.mean() - 1.96 * se, s.mean() + 1.96 * se
        ok = lo <= true_mu <= hi
        covered += ok
        ax.plot([lo, hi], [i, i], color=(BLUE if ok else PINK), lw=1.0, alpha=0.9)
    ax.axvline(true_mu, color=GREEN, lw=2, zorder=6)
    ax.set_yticks([]); ax.set_xlabel("estimate")
    ax.set_title(f"100 intervals — {covered} contained the truth", fontsize=10, pad=8)
    tidy(ax, grid=False)

    ax = axes[1]
    rates = []
    for n_i in [5, 10, 20, 50, 100]:
        c = 0
        for _ in range(4000):
            s = rng.normal(true_mu, 2.0, n_i)
            se = s.std(ddof=1) / np.sqrt(n_i)
            c += (abs(s.mean() - true_mu) <= 1.96 * se)
        rates.append(c / 4000)
    ax.plot([5, 10, 20, 50, 100], rates, "o-", color=BLUE, lw=2, ms=6)
    ax.axhline(0.95, color=GREEN, lw=1.8, ls="--")
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel("sample size"); ax.set_ylabel("coverage")
    ax.set_title("small samples under-cover with the normal formula",
                 fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "04-coverage")


# =============================================================== fig 5 ======
def fig_bootstrap():
    """Resampling your data approximates resampling the world."""
    rng = np.random.default_rng(4)
    pop = rng.gamma(2.0, 2.0, size=200000)          # skewed, no closed form for median
    sample = rng.choice(pop, 60, replace=False)

    true_dist = np.array([np.median(rng.choice(pop, 60, replace=False))
                          for _ in range(4000)])
    boot = np.array([np.median(rng.choice(sample, 60, replace=True))
                     for _ in range(4000)])

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.hist(true_dist, bins=50, density=True, color=GREEN, alpha=0.45,
            label="true sampling distribution\n(needs the whole population)")
    ax.hist(boot, bins=50, density=True, color=ORANGE, alpha=0.45,
            label="bootstrap\n(needs only your 60 points)")
    ax.legend(fontsize=7.8, frameon=False)
    ax.set_xlabel("sample median"); ax.set_yticks([])
    ax.set_title("the bootstrap recovers the shape without the population",
                 fontsize=9.8, pad=8)
    tidy(ax, grid=False)
    save(fig, "05-bootstrap")


# =============================================================== fig 6 ======
def fig_pvalues():
    """Under the null, p-values are uniform. That IS the definition."""
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.85))

    def pval(a, b):
        na, nb = len(a), len(b)
        se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
        t = (a.mean() - b.mean()) / se
        # two-sided normal approximation
        import math
        return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / np.sqrt(2))))

    null_ps = [pval(rng.normal(0, 1, 30), rng.normal(0, 1, 30)) for _ in range(6000)]
    alt_ps = [pval(rng.normal(0.6, 1, 30), rng.normal(0, 1, 30)) for _ in range(6000)]

    ax = axes[0]
    ax.hist(null_ps, bins=20, density=True, color=BLUE, alpha=0.6)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.3)
    ax.axvspan(0, 0.05, color=PINK, alpha=0.22)
    ax.set_xlabel("p-value"); ax.set_yticks([])
    frac = np.mean(np.array(null_ps) < 0.05)
    ax.set_title(f"no real effect: flat, and {frac:.1%} below 0.05",
                 fontsize=9.4, pad=8)
    tidy(ax, grid=False)

    ax = axes[1]
    ax.hist(alt_ps, bins=20, density=True, color=GREEN, alpha=0.6)
    ax.axvspan(0, 0.05, color=PINK, alpha=0.22)
    ax.set_xlabel("p-value"); ax.set_yticks([])
    frac2 = np.mean(np.array(alt_ps) < 0.05)
    ax.set_title(f"real effect: piles up low, {frac2:.0%} detected",
                 fontsize=9.4, pad=8)
    tidy(ax, grid=False)
    save(fig, "06-pvalues")


# =============================================================== fig 7 ======
def fig_power():
    """Power: the chance of detecting an effect that is really there."""
    rng = np.random.default_rng(6)
    effects = np.linspace(0, 1.2, 13)
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    for n, c in [(10, LIGHT), (30, BLUE), (100, ORANGE)]:
        powers = []
        for d in effects:
            hits = 0
            for _ in range(1500):
                a = rng.normal(d, 1, n); b = rng.normal(0, 1, n)
                se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
                t = (a.mean() - b.mean()) / se
                hits += abs(t) > 1.96
            powers.append(hits / 1500)
        ax.plot(effects, powers, "o-", color=c, lw=2, ms=4, label=f"n = {n}")
    ax.axhline(0.8, color=GREEN, ls="--", lw=1.4)
    ax.text(0.02, 0.83, "conventional 80% target", fontsize=8, color=GREEN)
    ax.set_xlabel("true effect size"); ax.set_ylabel("power")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("small studies miss real effects", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "07-power")


# =============================================================== fig 8 ======
def fig_multiple_comparisons():
    """Test enough hypotheses and something will look significant."""
    rng = np.random.default_rng(7)
    ks = np.arange(1, 61)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.85))

    ax = axes[0]
    prob = 1 - 0.95 ** ks
    ax.plot(ks, prob, color=PINK, lw=2.3)
    ax.axhline(0.05, color=GREEN, ls="--", lw=1.4)
    ax.set_xlabel("number of independent tests")
    ax.set_ylabel("P(at least one false positive)")
    ax.set_title(r"$1 - 0.95^k$ climbs fast", fontsize=10, pad=8)
    ax.text(22, 0.10, "the 5% you thought you had", fontsize=8, color=GREEN)
    tidy(ax)

    ax = axes[1]
    best = []
    for k in ks:
        vals = [max(np.abs(rng.normal(0, 1, k))) for _ in range(800)]
        best.append(np.mean(vals))
    ax.plot(ks, best, color=BLUE, lw=2.3, label="mean best-of-$k$ score")
    ax.axhline(1.96, color=GREEN, ls="--", lw=1.4)
    ax.set_xlabel("number of models tried")
    ax.set_ylabel("best observed statistic")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("the winner looks good even when nothing works",
                 fontsize=9.6, pad=8)
    tidy(ax)
    save(fig, "08-multiple-comparisons")


# =============================================================== fig 9 ======
def fig_test_set_reuse():
    """Reusing a test set for selection makes it optimistic. Simulated."""
    rng = np.random.default_rng(8)
    n_test, trials = 500, 60
    rounds = np.arange(1, 41)
    gaps = []
    for _ in range(trials):
        best_seen, gap_run = -1, []
        for r in rounds:
            # 20 candidate "models", all genuinely 70% accurate
            accs = rng.binomial(n_test, 0.70, size=20) / n_test
            best_seen = max(best_seen, accs.max())
            gap_run.append(best_seen - 0.70)
        gaps.append(gap_run)
    gaps = np.array(gaps).mean(axis=0)

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.plot(rounds * 20, gaps * 100, color=PINK, lw=2.4)
    ax.axhline(0, color=GREEN, lw=1.6, ls="--")
    ax.set_xlabel("models evaluated on the same test set")
    ax.set_ylabel("optimism, percentage points")
    ax.set_title("every model is truly 70% accurate — the best reported is not",
                 fontsize=9.5, pad=8)
    tidy(ax)
    save(fig, "09-test-set-reuse")


# ============================================================== fig 10 ======
def fig_map_vs_mle():
    """A prior pulls the estimate; with enough data the pull becomes negligible."""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7))
    grid = np.linspace(0, 1, 400)
    for ax, n in zip(axes, [4, 20, 200]):
        k = int(round(0.75 * n))
        lik = grid ** k * (1 - grid) ** (n - k)
        lik /= np.trapz(lik, grid)
        prior = grid ** 4 * (1 - grid) ** 4          # believes in 0.5
        prior /= np.trapz(prior, grid)
        post = lik * prior
        post /= np.trapz(post, grid)
        ax.plot(grid, prior, color=GRAY, lw=1.6, ls="--", label="prior")
        ax.plot(grid, lik, color=BLUE, lw=2.0, label="likelihood (MLE)")
        ax.plot(grid, post, color=ORANGE, lw=2.2, label="posterior (MAP)")
        ax.axvline(0.75, color=GREEN, lw=1.3)
        ax.set_yticks([]); ax.set_xlabel(r"$\phi$")
        ax.set_title(f"n = {n}", fontsize=9.6, pad=5)
        if n == 4:
            ax.legend(fontsize=7.2, frameon=False)
        tidy(ax, grid=False)
    save(fig, "10-map-vs-mle")


# ============================================================== fig 11 ======
def fig_fits():
    """Underfit, about right, overfit — the same data, three capacities."""
    rng = np.random.default_rng(9)
    f = lambda x: np.sin(2 * np.pi * x)
    xs = np.sort(rng.uniform(0, 1, 14))
    ys = f(xs) + rng.normal(0, 0.22, 14)
    grid = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    for ax, deg in zip(axes, [1, 4, 13]):
        c = np.polyfit(xs, ys, deg)
        ax.plot(grid, f(grid), color=GREEN, lw=1.6, ls="--", label="truth")
        ax.plot(grid, np.polyval(c, grid), color=ORANGE, lw=2.1, label="fit")
        ax.plot(xs, ys, "o", color=BLUE, ms=5, zorder=6)
        ax.set_ylim(-2.0, 2.0); ax.set_xticks([]); ax.set_yticks([])
        name = {1: "underfit (degree 1)", 4: "about right (degree 4)",
                13: "overfit (degree 13)"}[deg]
        ax.set_title(name, fontsize=9.4, pad=5)
        if deg == 1:
            ax.legend(fontsize=7.5, frameon=False, loc="lower left")
        tidy(ax, grid=False)
    save(fig, "11-fits")


# ============================================================== fig 12 ======
def fig_bias_variance_curve():
    """The decomposition, measured: bias falls, variance rises, MSE is a U."""
    rng = np.random.default_rng(10)
    f = lambda x: np.sin(2 * np.pi * x)
    grid = np.linspace(0.05, 0.95, 60)
    truth = f(grid)
    degs = np.arange(0, 12)
    bias2, var, mse = [], [], []
    for d in degs:
        preds = []
        for _ in range(300):
            xs = np.sort(rng.uniform(0, 1, 16))
            ys = f(xs) + rng.normal(0, 0.25, 16)
            with np.errstate(all="ignore"):
                c = np.polyfit(xs, ys, d)
            preds.append(np.polyval(c, grid))
        P = np.array(preds)
        mean_pred = P.mean(axis=0)
        b2 = np.mean((mean_pred - truth) ** 2)
        v = np.mean(P.var(axis=0))
        bias2.append(b2); var.append(v); mse.append(b2 + v)

    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    ax.semilogy(degs, np.maximum(bias2, 1e-6), "o-", color=BLUE, lw=2, ms=4,
                label=r"bias$^2$")
    ax.semilogy(degs, np.maximum(var, 1e-6), "o-", color=ORANGE, lw=2, ms=4,
                label="variance")
    ax.semilogy(degs, np.maximum(mse, 1e-6), "o-", color=INK, lw=2.4, ms=4,
                label="total error")
    ax.set_xlabel("polynomial degree (capacity)")
    ax.set_ylabel("error")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("measured on 300 resamples per degree", fontsize=9.8, pad=8)
    tidy(ax)
    save(fig, "12-bias-variance-curve")


# ============================================================== fig 13 ======
def fig_double_descent():
    """The classical U-curve is not the whole story. Real sweep, real model."""
    rng = np.random.default_rng(11)
    n_train, n_test, d_in = 40, 400, 20
    w = rng.normal(size=d_in)
    noise = 1.0
    Xtr = rng.normal(size=(n_train, d_in)); ytr = Xtr @ w + rng.normal(0, noise, n_train)
    Xte = rng.normal(size=(n_test, d_in));  yte = Xte @ w + rng.normal(0, noise, n_test)

    # Add width=40 (= n_train) explicitly so the peak at the interpolation threshold
    # lands on a grid point and the caption's "here 40" is literally true.
    widths = np.unique(np.sort(
        np.append(np.round(np.logspace(0, 2.7, 34)).astype(int), [n_train])
    ))
    errs = []
    for p in widths:
        runs = []
        for _ in range(20):
            W = rng.normal(size=(d_in, p)) / np.sqrt(d_in)
            Ztr, Zte = np.tanh(Xtr @ W), np.tanh(Xte @ W)
            # TRUE minimum-norm interpolant. pinv's DEFAULT rcond truncates small
            # singular values -- that is implicit regularisation, and it erases the
            # interpolation peak entirely. Verified: with the default the maximum sits
            # at p=1 (plain underfitting); with rcond=1e-15 and p=n_train=40 the peak
            # lands exactly at width 40.
            beta = np.linalg.pinv(Ztr, rcond=1e-15) @ ytr
            runs.append(np.mean((Zte @ beta - yte) ** 2))
        errs.append(np.median(runs))

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.loglog(widths, errs, "o-", color=BLUE, lw=2, ms=4)
    ax.axvline(n_train, color=PINK, lw=1.8, ls="--")
    ax.text(n_train * 1.12, max(errs) * 0.5, "features = samples",
            color=PINK, fontsize=8.5)
    ax.set_xlabel("number of random features (capacity)")
    ax.set_ylabel("test error")
    ax.set_title("error peaks where capacity meets data, then falls again",
                 fontsize=9.7, pad=8)
    tidy(ax)
    save(fig, "13-double-descent")


# A fig_cross_validation() lived here: a five-by-twenty grid of coloured
# rectangles showing which fold is held out. It computed nothing, which is why
# it was never placed in a section, and it emitted an orphan SVG on every run.
# CONVENTIONS §7 forbids the schematic; cross-validation is taught in Part VI §10.


if __name__ == "__main__":
    print("generating Part V figures ->", OUT)
    for fn in (fig_estimator, fig_bias_variance_quadrants, fig_root_n, fig_coverage,
               fig_bootstrap, fig_pvalues, fig_power, fig_multiple_comparisons,
               fig_test_set_reuse, fig_map_vs_mle, fig_fits, fig_bias_variance_curve,
               fig_double_descent):
        fn()
    print("done.")
