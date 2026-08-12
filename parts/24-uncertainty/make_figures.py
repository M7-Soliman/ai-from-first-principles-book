"""
Figures for Part XXIV — Uncertainty Quantification.

An uncertainty estimate makes a checkable promise: that intervals of nominal
coverage 90% contain the truth 90% of the time. So every figure here is a
coverage measurement, run on data whose true generating process the script sets.

    python3 make_figures.py

Runtime: about two minutes.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "uq"))
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


# ------------------------------------------------- 1. the two kinds, separated

def fig_decomposition():
    """Aleatoric uncertainty is in the data and does not fall with more of it.
    Epistemic uncertainty is in the model and does. Generate data whose noise
    level the script sets, fit an ensemble, and separate the two by the law of
    total variance."""
    print("\n[1] aleatoric against epistemic")
    rng = np.random.default_rng(0)
    TRUE_NOISE = 0.40
    D = 6

    def basis(x):
        return np.column_stack([np.ones(len(x))] + [x ** k for k in range(1, D)])

    w_true = rng.standard_normal(D) * 0.6
    xs = np.linspace(-1, 1, 400)
    Bte = basis(xs)
    f_true = Bte @ w_true

    ns = np.unique(np.round(np.logspace(1.1, 3.4, 12)).astype(int))
    ale, epi = [], []
    for n in ns:
        preds = []
        for m in range(40):                      # an ensemble over resamples
            x = rng.uniform(-1, 1, n)
            y = basis(x) @ w_true + TRUE_NOISE * rng.standard_normal(n)
            B = basis(x)
            w = np.linalg.lstsq(B.T @ B + 1e-6 * np.eye(D), B.T @ y, rcond=None)[0]
            preds.append(Bte @ w)
        preds = np.array(preds)
        # law of total variance: within-member noise + between-member spread
        epi.append(float(preds.var(axis=0).mean()))
        # the ground-truth noise variance the data was generated with — not recovered
        # from the ensemble's residuals, since these members report no per-model noise
        ale.append(TRUE_NOISE ** 2)
        print(f"    n={n:<5d}  epistemic {epi[-1]:.4f}   aleatoric {ale[-1]:.4f}")
    ale, epi = np.array(ale), np.array(epi)
    # the first points sit at n ~ D where the design matrix is near-singular, so
    # the epistemic variance there is enormous for a numerical reason rather than
    # an interesting one. Quote the ratio from where the fit is well conditioned.
    i0 = int(np.argmax(ns >= 4 * D))
    print(f"    epistemic falls {epi[i0]/epi[-1]:.0f}x from n={ns[i0]} to n={ns[-1]}"
          f" (and is {epi[0]:.0f} at n={ns[0]}, where n is barely above the {D} basis"
          f" functions and the fit is near-singular);")
    print(f"    aleatoric is set by the generating process and does not move.")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.loglog(ns, epi, "o-", color=BLUE, lw=1.9, ms=4, label="epistemic (model)")
    ax.loglog(ns, ale, "o-", color=ORANGE, lw=1.9, ms=4, label="aleatoric (data)")
    ax.set_xlabel("training points")
    ax.set_ylabel("variance")
    ax.set_title("Only one of them is reducible", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for n, col, lab in ((25, ORANGE, "n = 25"), (400, BLUE, "n = 400")):
        preds = []
        for m in range(40):
            x = rng.uniform(-1, 1, n)
            y = basis(x) @ w_true + TRUE_NOISE * rng.standard_normal(n)
            B = basis(x)
            w = np.linalg.lstsq(B.T @ B + 1e-6 * np.eye(D), B.T @ y, rcond=None)[0]
            preds.append(Bte @ w)
        preds = np.array(preds)
        mu, sd = preds.mean(0), preds.std(0)
        ax.plot(xs, mu, color=col, lw=1.7, label=lab)
        ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.18)
    ax.plot(xs, f_true, color=INK, lw=1.5, ls="--", label="the truth")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Model uncertainty shrinking", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "01-decomposition")


# ------------------------------------------------------- 2. a Gaussian process

def _rbf(a, b, ell=0.35, amp=1.0):
    d = a[:, None] - b[None, :]
    return amp ** 2 * np.exp(-0.5 * (d / ell) ** 2)


def fig_gp():
    """A Gaussian process gives a posterior over functions with a closed form,
    so its intervals can be checked against the coverage they promise — both
    where there is data and where there is not."""
    print("\n[2] the Gaussian process posterior, and its coverage")
    rng = np.random.default_rng(1)
    NOISE = 0.25

    def truth(x):
        return np.sin(3 * x) + 0.4 * np.cos(7 * x)

    xtr = np.concatenate([rng.uniform(-2.5, -0.6, 14), rng.uniform(0.8, 2.5, 14)])
    ytr = truth(xtr) + NOISE * rng.standard_normal(len(xtr))
    xs = np.linspace(-3.2, 3.2, 400)

    K = _rbf(xtr, xtr) + NOISE ** 2 * np.eye(len(xtr))
    Ks = _rbf(xs, xtr)
    Kss = _rbf(xs, xs)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    mu = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    var = np.clip(np.diag(Kss) - np.sum(v ** 2, axis=0), 1e-12, None)
    sd = np.sqrt(var)

    gap = (xs > -0.6) & (xs < 0.8)
    inside = (xs >= -2.5) & (xs <= 2.5) & ~gap
    print(f"    posterior sd where there is data: {sd[inside].mean():.3f}")
    print(f"    in the interpolation gap:         {sd[gap].mean():.3f}")
    print(f"    beyond the data entirely:         {sd[(xs < -2.7) | (xs > 2.7)].mean():.3f}")

    # coverage of the 95% predictive interval, over fresh draws
    cov = []
    for _ in range(400):
        xq = rng.uniform(-2.5, 2.5, 200)
        yq = truth(xq) + NOISE * rng.standard_normal(200)
        Kq = _rbf(xq, xtr)
        m = Kq @ alpha
        vq = np.linalg.solve(L, Kq.T)
        s = np.sqrt(np.clip(np.diag(_rbf(xq, xq)) - np.sum(vq ** 2, axis=0), 1e-12, None)
                    + NOISE ** 2)
        cov.append(np.mean(np.abs(yq - m) <= 1.96 * s))
    print(f"    measured coverage of the nominal 95% interval: {np.mean(cov):.3f}")

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.fill_between(xs, mu - 1.96 * np.sqrt(var + NOISE ** 2),
                    mu + 1.96 * np.sqrt(var + NOISE ** 2),
                    color=BLUE, alpha=0.15, label="95% predictive interval")
    ax.plot(xs, mu, color=BLUE, lw=1.9, label="posterior mean")
    ax.plot(xs, truth(xs), color=INK, lw=1.4, ls="--", label="the truth")
    ax.plot(xtr, ytr, "o", color=ORANGE, ms=4.5, label="observations")
    ax.axvspan(-0.6, 0.8, color=GRAY, alpha=0.07)
    ax.text(0.1, ax.get_ylim()[1] * 0.86, "no data", fontsize=8, color=GRAY, ha="center")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("The interval widens exactly where the evidence stops",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    tidy(ax)
    save(fig, "02-gaussian-process")


# ------------------------------------- 3. what the cheap approximations deliver

def fig_approximations():
    """Ensembles, MC dropout and a point estimate all claim to say something
    about uncertainty. Measure what each actually delivers, on a problem where
    the exact posterior is available for comparison — and query partly outside
    the training range, because that is where the claim is load-bearing."""
    print("\n[3] three cheap approximations, measured")
    rng = np.random.default_rng(2)
    NOISE, N, D = 0.3, 60, 8

    def basis(x):
        return np.column_stack([np.ones(len(x))] + [x ** k for k in range(1, D)])

    w_true = rng.standard_normal(D) * 0.5
    xtr = rng.uniform(-1, 1, N)
    ytr = basis(xtr) @ w_true + NOISE * rng.standard_normal(N)
    B = basis(xtr)
    PRIOR = 1.0

    # the exact posterior for this linear-Gaussian model
    A = B.T @ B / NOISE ** 2 + np.eye(D) / PRIOR ** 2
    Sig = np.linalg.inv(A)
    mu_w = Sig @ (B.T @ ytr) / NOISE ** 2

    def coverage(mu_f, sd_f, xq, yq):
        return float(np.mean(np.abs(yq - mu_f) <= 1.96 * np.sqrt(sd_f ** 2 + NOISE ** 2)))

    rows = []
    trials = 300
    accum = {k: [] for k in ("exact", "ensemble", "dropout", "single")}
    for t in range(trials):
        # query BEYOND the training range as well as inside it. Restricted to the
        # region that is densely observed, ignoring epistemic uncertainty barely
        # costs anything, and the figure would say nothing.
        xq = rng.uniform(-1.4, 1.4, 300)
        yq = basis(xq) @ w_true + NOISE * rng.standard_normal(300)
        Bq = basis(xq)

        mu_f = Bq @ mu_w
        sd_f = np.sqrt(np.einsum("ij,jk,ik->i", Bq, Sig, Bq))
        accum["exact"].append(coverage(mu_f, sd_f, xq, yq))

        # bootstrap ensemble: refit the same linear model on bootstrap resamples of the
        # training set (not a deep ensemble — that would retrain from different random
        # initialisations; see Part XXIV §19)
        preds = []
        for m in range(12):
            idx = rng.integers(0, N, N)
            Bi = B[idx]
            w = np.linalg.solve(Bi.T @ Bi + 1e-3 * np.eye(D), Bi.T @ ytr[idx])
            preds.append(Bq @ w)
        preds = np.array(preds)
        accum["ensemble"].append(coverage(preds.mean(0), preds.std(0), xq, yq))

        # MC dropout: randomly zero half the basis functions at prediction time
        preds = []
        w_map = np.linalg.solve(B.T @ B + 1e-3 * np.eye(D), B.T @ ytr)
        for m in range(40):
            mask = (rng.random(D) > 0.5) / 0.5
            preds.append(Bq @ (w_map * mask))
        preds = np.array(preds)
        accum["dropout"].append(coverage(preds.mean(0), preds.std(0), xq, yq))

        # a single point estimate, claiming only the noise
        accum["single"].append(coverage(Bq @ w_map, np.zeros(len(xq)), xq, yq))

    names = [("exact posterior", "exact"), ("bootstrap ensemble", "ensemble"),
             ("MC dropout", "dropout"), ("point estimate", "single")]
    for lab, k in names:
        v = np.mean(accum[k])
        rows.append((lab, v))
        print(f"    {lab:18s} coverage of the nominal 0.95 interval: {v:.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    cols = [GREEN, BLUE, PURPLE, ORANGE]
    ax.bar(np.arange(len(rows)), [r[1] for r in rows], 0.55, color=cols)
    ax.axhline(0.95, color=INK, lw=1.4, ls="--")
    ax.text(len(rows) - 0.55, 0.962, "nominal 95%", fontsize=8, color=INK, ha="right")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r[0].replace(" ", "\n") for r in rows], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("measured coverage")
    ax.set_title("What each method's 95% interval actually contains",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "03-approximations")


# ----------------------------------------------------- 4. conformal prediction

def fig_conformal():
    """Split conformal makes a coverage guarantee that holds for any model and
    any distribution, requiring only exchangeability. Test it with a
    deliberately terrible model, on a distribution nothing was tuned for."""
    print("\n[4] conformal coverage, with a bad model on purpose")
    rng = np.random.default_rng(3)

    def truth(x):
        return np.sin(4 * x) + 0.5 * x ** 2

    def run(alpha, model_quality, n_cal=500, trials=400):
        covs, widths = [], []
        for _ in range(trials):
            xtr = rng.uniform(-2, 2, 300)
            ytr = truth(xtr) + 0.3 * rng.standard_normal(300)
            deg = 6 if model_quality == "good" else 0     # constant predictor
            c = np.polyfit(xtr, ytr, deg)
            f = lambda x: np.polyval(c, x)

            xc = rng.uniform(-2, 2, n_cal)
            yc = truth(xc) + 0.3 * rng.standard_normal(n_cal)
            scores = np.abs(yc - f(xc))
            k = int(np.ceil((n_cal + 1) * (1 - alpha)))
            q = np.sort(scores)[min(k, n_cal) - 1]

            xq = rng.uniform(-2, 2, 400)
            yq = truth(xq) + 0.3 * rng.standard_normal(400)
            covs.append(np.mean(np.abs(yq - f(xq)) <= q))
            widths.append(2 * q)
        return float(np.mean(covs)), float(np.mean(widths))

    alphas = [0.30, 0.20, 0.10, 0.05, 0.02]
    res = {}
    for quality in ("good", "bad"):
        rows = [run(a, quality) for a in alphas]
        res[quality] = rows
        for a, (c, w) in zip(alphas, rows):
            print(f"    {quality:4s} model, nominal {1-a:.2f}:  coverage {c:.3f}   "
                  f"mean width {w:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    nom = [1 - a for a in alphas]
    for quality, col, lab in (("good", GREEN, "a good model"), ("bad", ORANGE, "a constant predictor")):
        ax.plot(nom, [c for c, _ in res[quality]], "o-", color=col, lw=1.9, ms=5, label=lab)
    ax.plot([0.65, 1.0], [0.65, 1.0], color=INK, lw=1.2, ls="--", label="nominal")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("measured coverage")
    ax.set_title("The guarantee does not depend on the model",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    for quality, col, lab in (("good", GREEN, "a good model"), ("bad", ORANGE, "a constant predictor")):
        ax.plot(nom, [w for _, w in res[quality]], "o-", color=col, lw=1.9, ms=5, label=lab)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("mean interval width")
    ax.set_title("The model decides the width, not the validity",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)
    save(fig, "04-conformal")
    print("    both models are covered at the rate they promised. The bad one pays")
    print("    for it in width, which is the whole trade.")


# ------------------------------------------------- 5. where the guarantee ends

def fig_shift():
    """Conformal coverage requires exchangeability between calibration and test.
    Break it and measure how fast the guarantee degrades — the point being that
    it does not fail gracefully."""
    print("\n[5] coverage when exchangeability breaks")
    rng = np.random.default_rng(4)

    def truth(x):
        return np.sin(4 * x) + 0.5 * x ** 2

    shifts = np.linspace(0.0, 1.6, 9)
    alpha = 0.10
    covs, widths = [], []
    for s in shifts:
        c_list = []
        for _ in range(300):
            xtr = rng.normal(0, 0.7, 400)
            ytr = truth(xtr) + 0.3 * rng.standard_normal(400)
            co = np.polyfit(xtr, ytr, 6)
            f = lambda x: np.polyval(co, x)
            xc = rng.normal(0, 0.7, 500)
            yc = truth(xc) + 0.3 * rng.standard_normal(500)
            sc = np.abs(yc - f(xc))
            k = int(np.ceil(501 * (1 - alpha)))
            q = np.sort(sc)[min(k, 500) - 1]
            xq = rng.normal(s, 0.7, 400)               # the test set moves
            yq = truth(xq) + 0.3 * rng.standard_normal(400)
            c_list.append(np.mean(np.abs(yq - f(xq)) <= q))
        covs.append(float(np.mean(c_list)))
        print(f"    shift {s:.2f}:  coverage {covs[-1]:.3f}   (promised {1-alpha:.2f})")

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(shifts, covs, "o-", color=PURPLE, lw=1.9, ms=5)
    ax.axhline(1 - alpha, color=INK, lw=1.4, ls="--")
    ax.text(shifts[-1], 1 - alpha + 0.012, "the promise", fontsize=8, color=INK, ha="right")
    ax.set_xlabel("shift between calibration and test")
    ax.set_ylabel("measured coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("Exchangeability is the whole assumption", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "05-shift")


# -------------------------------------------------- 6. what a decision needs

def fig_decision():
    """A probability is not a decision. Exact arithmetic: the optimal threshold
    under asymmetric costs, and how much a perfectly calibrated model can lose
    by using 0.5 instead."""
    print("\n[6] from a probability to a decision  (exact)")
    rng = np.random.default_rng(5)
    N = 400_000
    BASE = 0.08

    p = rng.beta(1.2, 6.0, N)                    # a spread of calibrated risks
    p = p / p.mean() * BASE
    p = np.clip(p, 0, 1)
    y = rng.random(N) < p                        # calibrated by construction

    ratios = np.array([1, 2, 5, 10, 20, 50])
    rows = []
    for r in ratios:                             # cost of a miss / cost of a false alarm
        c_fp, c_fn = 1.0, float(r)
        thr_opt = c_fp / (c_fp + c_fn)
        def cost(t):
            act = p >= t
            return float(np.mean(c_fp * (act & ~y) + c_fn * (~act & y)))
        rows.append((r, thr_opt, cost(thr_opt), cost(0.5)))
        print(f"    cost ratio {r:>2}:1   optimal threshold {thr_opt:.3f}   "
              f"cost {cost(thr_opt):.4f}   at 0.5 {cost(0.5):.4f}   "
              f"({cost(0.5)/max(cost(thr_opt),1e-9):.1f}x)")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ts = np.linspace(0.005, 0.6, 120)
    for r, col in ((1, GRAY), (5, BLUE), (20, ORANGE)):
        c_fp, c_fn = 1.0, float(r)
        cs = [float(np.mean(c_fp * ((p >= t) & ~y) + c_fn * ((p < t) & y))) for t in ts]
        ax.plot(ts, cs, color=col, lw=1.9, label=f"miss costs {r}x")
        ax.axvline(1 / (1 + r), color=col, lw=1.0, ls=":")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("expected cost")
    ax.set_title("Where the threshold belongs", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogx(ratios, [r[3] / r[2] for r in rows], "o-", color=PURPLE, lw=1.9, ms=5)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.set_xlabel("cost of a miss, relative to a false alarm")
    ax.set_ylabel("cost at 0.5, relative to optimal")
    ax.set_title("What a perfectly calibrated model still loses",
                 fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "06-decision")
    print("    the model is calibrated throughout. Every one of these losses comes")
    print("    from the threshold, which is a decision rather than a probability.")


if __name__ == "__main__":
    fig_decomposition()
    fig_gp()
    fig_approximations()
    fig_conformal()
    fig_shift()
    fig_decision()
