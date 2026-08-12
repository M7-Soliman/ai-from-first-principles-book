"""
Figures for Part XXII — Causal Inference.

Every figure here simulates a system whose true causal effect is set by the
script. That is the whole reason this part can be measured at all: in a real
study the answer is exactly what you do not have, so a method that recovers a
known answer is the only evidence that the method works.

    python3 make_figures.py

Runtime: about a minute. No training.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "ci"))
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


def ols(X, y):
    """Least squares with an intercept. Returns the coefficient vector."""
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def logistic(X, t, iters=25):
    """Logistic regression by Newton–Raphson, returning fitted probabilities.

    Written out rather than imported because the doubly robust figure turns on
    whether the propensity model is ACTUALLY correct when it is supposed to be.
    The first version of that figure fitted a least-squares line and pushed it
    through a logistic, which is not logistic regression, and it made the
    estimator look like it had failed a property it in fact has.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X).reshape(len(X), -1)])
    t = np.asarray(t, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        z = X.T @ (t - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, z)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


# ------------------------------------------------------------ 1. confounding

def fig_confounding():
    """The naive regression of Y on X answers 'how does Y differ across the X we
    observed'. The causal question is 'how would Y change if we set X'. With a
    common cause of both, those are different numbers, and the gap is exactly
    the confounding path. Backdoor adjustment closes it."""
    print("\n[1] confounding, and what adjustment recovers")
    TAU = 1.0                       # the true causal effect of X on Y
    N = 40_000
    rng = np.random.default_rng(0)

    strengths = np.linspace(0, 3.0, 13)
    naive, adjusted = [], []
    for b in strengths:
        U = rng.standard_normal(N)                        # the confounder
        X = 1.2 * U + rng.standard_normal(N)              # U -> X
        Y = TAU * X - b * U + rng.standard_normal(N)      # X -> Y, U -> Y
        naive.append(ols(X, Y)[1])
        adjusted.append(ols(np.column_stack([X, U]), Y)[1])
    naive, adjusted = np.array(naive), np.array(adjusted)

    for b, n, a in list(zip(strengths, naive, adjusted))[::4]:
        print(f"    confounding {b:4.2f}   naive {n:+.3f}   adjusted {a:+.3f}   true {TAU:+.2f}")
    flip = strengths[np.argmax(naive < 0)] if (naive < 0).any() else None
    if flip is not None:
        print(f"    the naive estimate changes SIGN at confounding strength {flip:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(strengths, naive, "o-", color=ORANGE, lw=1.9, ms=4,
            label="naive: regress $Y$ on $X$")
    ax.plot(strengths, adjusted, "o-", color=GREEN, lw=1.9, ms=4,
            label="adjusted: condition on $U$")
    ax.axhline(TAU, color=INK, lw=1.3, ls="--")
    ax.text(strengths[-1], TAU + 0.08, "the true effect", fontsize=8,
            color=INK, ha="right")
    ax.axhline(0, color=GRAY, lw=1.0, ls=":")
    ax.set_xlabel("strength of the confounder's effect on $Y$")
    ax.set_ylabel("estimated effect of $X$ on $Y$")
    ax.set_title("One of these answers the question", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)

    # the same data, shown as the picture that makes it obvious
    ax = axes[1]
    b = 2.6
    U = rng.standard_normal(4000)
    X = 1.2 * U + rng.standard_normal(4000)
    Y = TAU * X - b * U + rng.standard_normal(4000)
    q = np.quantile(U, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(U, q)
    cols = [BLUE, CYAN, GRAY, PINK, PURPLE]
    for s in range(5):
        m = strata == s
        ax.scatter(X[m], Y[m], s=2.5, color=cols[s], alpha=0.45, linewidths=0)
        c = ols(X[m], Y[m])
        xs = np.linspace(X[m].min(), X[m].max(), 2)
        ax.plot(xs, c[0] + c[1] * xs, color=cols[s], lw=1.5)
    c = ols(X, Y)
    xs = np.linspace(X.min(), X.max(), 2)
    ax.plot(xs, c[0] + c[1] * xs, color=ORANGE, lw=2.6, ls="--",
            label="all the data at once")
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
    ax.set_title("Within each stratum of $U$, and across them",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)
    save(fig, "01-confounding")
    print("    within every stratum of the confounder the slope is positive;")
    print("    pooled, it is negative. Same data, opposite conclusion.")


# --------------------------------------------------------- 2. collider bias

def fig_collider():
    """Conditioning is not a neutral act. Conditioning on a common EFFECT of two
    independent causes makes them dependent — so 'controlling for more things'
    can manufacture an association that does not exist."""
    print("\n[2] the association conditioning creates")
    N = 40_000
    rng = np.random.default_rng(1)

    strengths = np.linspace(0, 2.5, 13)
    uncond, cond = [], []
    for w in strengths:
        X = rng.standard_normal(N)
        Y = rng.standard_normal(N)                 # independent of X, by construction
        C = w * X + w * Y + rng.standard_normal(N)  # the collider
        uncond.append(np.corrcoef(X, Y)[0, 1])
        sel = C > np.quantile(C, 0.7)              # condition: keep the top 30%
        cond.append(np.corrcoef(X[sel], Y[sel])[0, 1])
    uncond, cond = np.array(uncond), np.array(cond)
    for w, u, c in list(zip(strengths, uncond, cond))[::4]:
        print(f"    collider strength {w:4.2f}   corr(X,Y) {u:+.3f}   "
              f"given C {c:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    w = 1.6
    X = rng.standard_normal(6000)
    Y = rng.standard_normal(6000)
    C = w * X + w * Y + rng.standard_normal(6000)
    sel = C > np.quantile(C, 0.7)
    ax.scatter(X[~sel], Y[~sel], s=2.5, color=LIGHT, alpha=0.6, linewidths=0,
               label="not selected")
    ax.scatter(X[sel], Y[sel], s=3.5, color=PURPLE, alpha=0.6, linewidths=0,
               label="selected on $C$")
    c = ols(X[sel], Y[sel]); xs = np.linspace(X.min(), X.max(), 2)
    ax.plot(xs, c[0] + c[1] * xs, color=PURPLE, lw=2.2)
    c = ols(X, Y)
    ax.plot(xs, c[0] + c[1] * xs, color=INK, lw=1.6, ls="--",
            label="all the data")
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
    ax.set_title("$X$ and $Y$ are independent", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right", markerscale=2.5)
    tidy(ax)

    ax = axes[1]
    ax.plot(strengths, uncond, "o-", color=GREEN, lw=1.9, ms=4,
            label="$\\mathrm{corr}(X, Y)$")
    ax.plot(strengths, cond, "o-", color=PURPLE, lw=1.9, ms=4,
            label="$\\mathrm{corr}(X, Y \\mid C)$")
    ax.axhline(0, color=INK, lw=1.3, ls="--")
    ax.text(strengths[-1], 0.03, "the truth: no association", fontsize=8,
            color=INK, ha="right")
    ax.set_xlabel("how strongly $X$ and $Y$ drive the collider")
    ax.set_ylabel("measured correlation")
    ax.set_title("What conditioning manufactures", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)
    save(fig, "02-collider")
    print("    there is no association between X and Y. Conditioning on their")
    print("    common effect produces one, and it grows with the collider's strength.")


# ------------------------------------------------- 3. propensity and overlap

def fig_overlap():
    """Adjustment only works where both treated and untreated units exist. As
    the propensity score approaches 0 or 1 the comparison is being made on
    nobody, and the estimator's variance says so before its bias does."""
    print("\n[3] overlap, and where adjustment stops working")
    TAU = 2.0
    N = 6000
    REPS = 200
    rng = np.random.default_rng(2)

    seps = np.linspace(0.4, 4.0, 10)
    bias, sd, ess = [], [], []
    for a in seps:
        est = []
        for r in range(REPS):
            U = rng.standard_normal(N)
            p = 1 / (1 + np.exp(-a * U))               # propensity
            T = rng.random(N) < p
            Y = TAU * T + 2.0 * U + rng.standard_normal(N)
            w = np.where(T, 1 / p, 1 / (1 - p))        # inverse probability
            e = (np.sum(w[T] * Y[T]) / np.sum(w[T])
                 - np.sum(w[~T] * Y[~T]) / np.sum(w[~T]))
            est.append(e)
        est = np.array(est)
        bias.append(est.mean() - TAU); sd.append(est.std())
        ess.append((w.sum() ** 2) / np.sum(w ** 2) / N)
    for a, b, s, e in list(zip(seps, bias, sd, ess))[::3]:
        print(f"    separation {a:4.2f}   bias {b:+.3f}   sd {s:.3f}   "
              f"effective sample {e:.2f} of 1.00")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for a, col, lab in ((0.6, GREEN, "good overlap"), (3.6, ORANGE, "poor overlap")):
        U = rng.standard_normal(20000)
        p = 1 / (1 + np.exp(-a * U))
        T = rng.random(20000) < p
        ax.hist(p[T], bins=40, range=(0, 1), histtype="step", lw=1.8, color=col,
                density=True, label=f"treated, {lab}")
        ax.hist(p[~T], bins=40, range=(0, 1), histtype="step", lw=1.1, color=col,
                density=True, ls=":", label=f"control, {lab}")
    ax.set_xlabel("propensity score"); ax.set_ylabel("density")
    ax.set_title("Who there is someone to compare with", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.2, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(seps, sd, "o-", color=PURPLE, lw=1.9, ms=4, label="standard deviation")
    ax.plot(seps, np.abs(bias), "o-", color=ORANGE, lw=1.9, ms=4, label="|bias|")
    ax.set_xlabel("how strongly the confounder drives treatment")
    ax.set_ylabel("error in the estimated effect")
    ax.set_title("Variance goes first", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)
    save(fig, "03-overlap")


# ------------------------------------------------------- 4. doubly robust

def fig_doubly_robust():
    """The property that gives the estimator its name, tested directly: get
    either the outcome model or the treatment model right and the estimate
    should survive. Get both wrong and it should not.

    The right panel is the caveat that turned up while building the left one.
    Inverse-probability weights blow up when a propensity approaches 0 or 1, so
    everyone clips them — and clipping is a bias. The two panels use the same
    estimator and differ only in whether positivity holds."""
    print("\n[4] doubly robust, all four arms")
    TAU = 2.0

    def run(out_ok, ps_ok, N=8000, REPS=300, clip=0.01, bounded=True, seed=11):
        r = np.random.default_rng(seed)
        est = []
        for _ in range(REPS):
            U = r.standard_normal(N)
            # V is non-monotone in U, so a model linear in U cannot represent it
            # at all — which is what makes the "wrong" arms genuinely wrong.
            V = U ** 2 - 1.0
            if bounded:
                V = np.clip(V, -1.0, 3.0)          # keeps the propensity off 0 and 1
            p = 1 / (1 + np.exp(-(0.6 if bounded else 0.8) * V))
            T = r.random(N) < p
            Y = TAU * T + 1.5 * V + r.standard_normal(N)
            Zo = V if out_ok else U
            Zp = V if ps_ok else U
            ph = np.clip(logistic(Zp, T), clip, 1 - clip)
            c1 = ols(Zo[T], Y[T]); c0 = ols(Zo[~T], Y[~T])
            m1 = c1[0] + c1[1] * Zo
            m0 = c0[0] + c0[1] * Zo
            aipw = (m1 - m0
                    + T * (Y - m1) / ph
                    - (~T) * (Y - m0) / (1 - ph))
            est.append(aipw.mean())
        return np.array(est)

    arms = [("both correct", True, True), ("outcome only", True, False),
            ("treatment only", False, True), ("both wrong", False, False)]
    res = {}
    for name, o, ps in arms:
        e = run(o, ps)
        res[name] = e
        print(f"    {name:16s} estimate {e.mean():+.4f}  (true {TAU:+.2f})  "
              f"bias {e.mean()-TAU:+.4f}   sd {e.std():.4f}")

    # and what clipping costs when positivity does NOT hold
    print("    with positivity violated, sweeping the weight clip:")
    clips = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    cb, cs = [], []
    for c in clips:
        e = run(False, True, N=8000, REPS=200, clip=c, bounded=False, seed=13)
        cb.append(e.mean() - TAU); cs.append(e.std())
        print(f"      clip {c:<6.3f} bias {cb[-1]:+.4f}   sd {cs[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    names = [a[0] for a in arms]
    means = [res[n].mean() for n in names]
    sds = [res[n].std() for n in names]
    ax.bar(np.arange(4), means, 0.55, yerr=sds, capsize=4,
           color=[GREEN, GREEN, GREEN, ORANGE],
           error_kw=dict(lw=1.1, ecolor=GRAY))
    ax.axhline(TAU, color=INK, lw=1.4, ls="--")
    ax.text(3.45, TAU + 0.06, "the true effect", fontsize=8, color=INK, ha="right")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    ax.set_ylabel("estimated effect")
    ax.set_title("Which model has to be right", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.plot(clips, np.abs(cb), "o-", color=ORANGE, lw=1.9, ms=4, label="|bias|")
    ax.plot(clips, cs, "o-", color=PURPLE, lw=1.9, ms=4, label="standard deviation")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("weight clip (looser $\\rightarrow$)")
    ax.set_ylabel("error in the estimated effect")
    ax.set_title("What the variance fix costs", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "04-doubly-robust")
    print("    the property holds when positivity does. The standard remedy for")
    print("    the variance — clipping the weights — is itself a bias, and the")
    print("    two move in opposite directions.")


# ------------------------------------------------ 5. instrumental variables

def fig_iv():
    """When the confounder is unobserved, adjustment is unavailable. An
    instrument buys the effect back — but only in proportion to how strongly it
    moves the treatment, and the failure mode at the weak end is not bias, it is
    instability: the estimate swings by several times the effect from one
    sample to the next."""
    print("\n[5] instruments, and what a weak one costs")
    TAU = 1.0
    N = 3000
    REPS = 300
    rng = np.random.default_rng(4)

    strengths = np.linspace(0.05, 1.5, 12)
    iv_m, iv_s, ols_m = [], [], []
    for g in strengths:
        ivs, olss = [], []
        for _ in range(REPS):
            U = rng.standard_normal(N)                  # unobserved confounder
            Z = rng.standard_normal(N)                  # the instrument
            X = g * Z + 1.0 * U + rng.standard_normal(N)
            Y = TAU * X + 1.5 * U + rng.standard_normal(N)
            # two-stage least squares
            a = ols(Z, X); Xhat = a[0] + a[1] * Z
            ivs.append(ols(Xhat, Y)[1])
            olss.append(ols(X, Y)[1])
        iv_m.append(np.mean(ivs)); iv_s.append(np.std(ivs))
        ols_m.append(np.mean(olss))
    iv_m, iv_s, ols_m = map(np.array, (iv_m, iv_s, ols_m))
    for g, m, s in list(zip(strengths, iv_m, iv_s))[::3]:
        print(f"    instrument strength {g:4.2f}   2SLS {m:+.3f} ± {s:.3f}   "
              f"(true {TAU:+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(strengths, iv_m, "o-", color=BLUE, lw=1.9, ms=4, label="two-stage least squares")
    ax.fill_between(strengths, iv_m - iv_s, iv_m + iv_s, color=BLUE, alpha=0.16)
    ax.plot(strengths, ols_m, "o-", color=ORANGE, lw=1.6, ms=3.5,
            label="ordinary least squares")
    ax.axhline(TAU, color=INK, lw=1.3, ls="--")
    ax.set_xlabel("instrument strength (effect of $Z$ on $X$)")
    ax.set_ylabel("estimated effect of $X$ on $Y$")
    ax.set_title("What the instrument buys back", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    tidy(ax)

    ax = axes[1]
    ax.semilogy(strengths, iv_s, "o-", color=PURPLE, lw=1.9, ms=4)
    ax.set_xlabel("instrument strength")
    ax.set_ylabel("spread of the estimate")
    ax.set_title("And what a weak one costs", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "05-instruments")


# ------------------------------------------------- 6. invariant prediction

def fig_invariance():
    """A predictor that uses a cause transfers to a new environment. One that
    uses an effect of the target does not, however much better it looked on the
    data you had. This is the machine-learning consequence of the whole part."""
    print("\n[6] which predictor survives a new environment")
    rng = np.random.default_rng(5)
    N = 4000

    def env(strength):
        """Xc causes Y; Xs is caused BY Y, with an environment-specific sign."""
        Xc = rng.standard_normal(N)
        Y = 1.0 * Xc + 0.5 * rng.standard_normal(N)
        Xs = strength * Y + 0.5 * rng.standard_normal(N)
        return Xc, Xs, Y

    train = [env(2.0), env(1.5)]                      # spurious feature helps
    Xc = np.concatenate([e[0] for e in train])
    Xs = np.concatenate([e[1] for e in train])
    Y = np.concatenate([e[2] for e in train])

    c_causal = ols(Xc, Y)
    c_spur = ols(Xs, Y)
    c_both = ols(np.column_stack([Xc, Xs]), Y)
    print(f"    coefficients when both are offered: causal {c_both[1]:+.3f}, "
          f"spurious {c_both[2]:+.3f}")

    tests = [("train-like  (+2.0)", 2.0), ("weaker      (+0.5)", 0.5),
             ("absent      ( 0.0)", 0.0), ("reversed    (-2.0)", -2.0)]
    rows = []
    for name, s in tests:
        a, b, y = env(s)
        e_c = np.mean((c_causal[0] + c_causal[1] * a - y) ** 2)
        e_s = np.mean((c_spur[0] + c_spur[1] * b - y) ** 2)
        e_b = np.mean((c_both[0] + c_both[1] * a + c_both[2] * b - y) ** 2)
        rows.append((name, e_c, e_s, e_b))
        print(f"    {name}   causal-only {e_c:6.3f}   spurious-only {e_s:6.3f}   "
              f"both {e_b:6.3f}")

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(rows)); w = 0.26
    ax.bar(x - w, [r[1] for r in rows], w, color=GREEN, label="uses the cause")
    ax.bar(x, [r[3] for r in rows], w, color=BLUE, label="uses both")
    ax.bar(x + w, [r[2] for r in rows], w, color=ORANGE, label="uses the effect")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0].split("(")[0].strip() + "\n" + "(" + r[0].split("(")[1]
                        for r in rows], fontsize=8)
    ax.set_ylabel("test error")
    ax.set_xlabel("environment, by the strength of the spurious relationship")
    ax.set_title("The same three models, four environments", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "06-invariance")


if __name__ == "__main__":
    fig_confounding()
    fig_collider()
    fig_overlap()
    fig_doubly_robust()
    fig_iv()
    fig_invariance()
