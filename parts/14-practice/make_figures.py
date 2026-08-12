"""
Figures for Part XIV — Practice and Diagnosis.

Every claim this part makes that can be measured, is measured here. The
gradient-check epsilon curve, the grid-against-random comparison, the
best-of-k inflation and the successive-halving win are all computed; the
calibration and learning-curve figures train small models.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the two that train

Runtime: about a minute for --cheap, five for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "pd"))
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


# ----------------------------------------------- 1. the metric and the threshold

def fig_metrics():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 300, 5700              # 5% positive — an imbalanced problem
    s_pos = rng.normal(1.4, 1.0, n_pos)
    s_neg = rng.normal(0.0, 1.0, n_neg)
    scores = np.concatenate([s_pos, s_neg])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    ths = np.linspace(scores.min(), scores.max(), 400)
    prec, rec, f1, acc = [], [], [], []
    for t in ths:
        pred = scores >= t
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum(~pred & (y == 1)))
        tn = float(np.sum(~pred & (y == 0)))
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        acc.append((tp + tn) / len(y))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    order = np.argsort(rec)
    ax.plot(np.array(rec)[order], np.array(prec)[order], color=BLUE, lw=2.0)
    ax.axhline(n_pos / len(y), color=GRAY, ls=":", lw=1.2)
    ax.text(0.02, n_pos / len(y) + 0.03, "a constant predictor", fontsize=7.5, color=GRAY)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    tidy(ax)

    ax = axes[1]
    ax.plot(ths, acc, color=ORANGE, lw=1.9, label="accuracy")
    ax.plot(ths, f1, color=BLUE, lw=1.9, label="$F_1$")
    ax.plot(ths, prec, color=GREEN, lw=1.3, alpha=0.8, label="precision")
    ax.plot(ths, rec, color=PURPLE, lw=1.3, alpha=0.8, label="recall")
    best_f1 = ths[int(np.argmax(f1))]
    best_acc = ths[int(np.argmax(acc))]
    ax.axvline(best_f1, color=BLUE, ls="--", lw=1.1)
    ax.axvline(best_acc, color=ORANGE, ls="--", lw=1.1)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("value")
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    tidy(ax)

    print(f"    base rate {n_pos/len(y):.3f} — so a constant 'no' scores "
          f"{1 - n_pos/len(y):.3f} accuracy and is useless")
    print(f"    accuracy peaks at threshold {best_acc:+.2f}, F1 at {best_f1:+.2f}")
    i_acc, i_f1 = int(np.argmax(acc)), int(np.argmax(f1))
    print(f"    at the accuracy optimum: recall {rec[i_acc]:.3f}")
    print(f"    at the F1 optimum:       recall {rec[i_f1]:.3f} — the SAME model, "
          f"and the metric chooses the operating point")
    fig.tight_layout()
    save(fig, "01-metrics")


# ------------------------------------------------------- 2. gradient checking

def fig_gradient_check():
    def f(x):
        return np.sin(x) * np.exp(x / 3.0)

    def df(x):
        return np.cos(x) * np.exp(x / 3.0) + np.sin(x) * np.exp(x / 3.0) / 3.0

    x0 = 1.3
    truth = df(x0)
    eps = np.array([10.0 ** -k for k in range(0, 17)])
    fwd, cen, cs = [], [], []
    for e in eps:
        fwd.append(abs(((f(x0 + e) - f(x0)) / e) - truth) / abs(truth))
        cen.append(abs(((f(x0 + e / 2) - f(x0 - e / 2)) / e) - truth) / abs(truth))
        cs.append(abs((np.imag(f(complex(x0, e))) / e) - truth) / abs(truth))
    FLOOR = 1e-18
    cs_true_floor = min(cs)   # the honest floor: machine precision, sometimes exactly 0.0
    fwd = np.maximum(fwd, FLOOR)
    cen = np.maximum(cen, FLOOR)
    cs = np.maximum(cs, FLOOR)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(eps, fwd, "o-", color=ORANGE, ms=4, lw=1.8, label="forward difference")
    ax.loglog(eps, cen, "o-", color=BLUE, ms=4, lw=1.8, label="central difference")
    ax.loglog(eps, cs, "o-", color=GREEN, ms=4, lw=1.8, label="complex step")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("relative error")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    # the extreme range, where finite differencing has nothing left
    ex = np.array([1e-8, 1e-20, 1e-50, 1e-100, 1e-150])
    ex_cen, ex_cs = [], []
    for e in ex:
        ex_cen.append(max(abs(((f(x0 + e / 2) - f(x0 - e / 2)) / e) - truth) / abs(truth), FLOOR))
        ex_cs.append(max(abs((np.imag(f(complex(x0, e))) / e) - truth) / abs(truth), FLOOR))
    ax = axes[1]
    idx = np.arange(len(ex))
    ax.bar(idx - 0.2, ex_cen, 0.4, color=BLUE, label="central difference")
    ax.bar(idx + 0.2, ex_cs, 0.4, color=GREEN, label="complex step")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$10^{{{int(np.log10(e))}}}$" for e in ex], fontsize=8)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("relative error")
    ax.set_title("where subtraction has nothing left", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    central difference is best near eps = 1e-5 ({min(cen):.1e}) and gets "
          f"WORSE on both sides — Part III §3's U-curve")
    print(f"    complex step reaches machine precision ({cs_true_floor:.1e}, exactly 0.0 at "
          f"several eps) and stays there down to 1e-150 — the plot clamps at {FLOOR:.0e} only so "
          f"a log axis can show the exact zeros; that clamp is a display floor, not an accuracy")
    print(f"    because it never subtracts two nearly equal numbers")
    fig.tight_layout()
    save(fig, "03-gradient-check")


# ---------------------------------------------------------- 3. grid vs random

def fig_search():
    rng = np.random.default_rng(0)

    def obj(v):
        return -((v[0] - 0.37) ** 2 * 12 + 0.03 * sum((vi - 0.5) ** 2 for vi in v[1:]))

    dims = [1, 2, 4, 6, 8]
    budget = 64
    gres, rres = [], []
    for D in dims:
        n = max(2, int(round(budget ** (1.0 / D))))
        axes_ = [np.linspace(0.05, 0.95, n)] * D
        grid = np.array(np.meshgrid(*axes_)).reshape(D, -1).T
        gres.append(max(obj(p) for p in grid))
        rres.append(np.mean([max(obj(p) for p in rng.random((budget, D)))
                             for _ in range(300)]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(dims, gres, "o-", color=ORANGE, ms=5, lw=1.9, label="grid search")
    ax.plot(dims, rres, "o-", color=BLUE, ms=5, lw=1.9, label="random search")
    ax.set_xlabel("number of hyperparameters (only one matters)")
    ax.set_ylabel("best objective found (higher is better)")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    D = 2
    n = 8
    g = np.array(np.meshgrid(np.linspace(0.05, 0.95, n),
                             np.linspace(0.05, 0.95, n))).reshape(2, -1).T
    r = rng.random((n * n, 2))
    ax.plot(g[:, 0], g[:, 1], "o", color=ORANGE, ms=4, label=f"grid: {n} distinct $x_1$")
    ax.plot(r[:, 0], r[:, 1], "o", color=BLUE, ms=4, alpha=0.75,
            label=f"random: {n*n} distinct $x_1$")
    ax.axvline(0.37, color=GRAY, ls=":", lw=1.3)
    ax.text(0.39, 0.02, "the value that matters", fontsize=7.5, color=GRAY)
    ax.set_xlabel("$x_1$ — influential")
    ax.set_ylabel("$x_2$ — not")
    ax.legend(fontsize=7.5, frameon=False)
    tidy(ax)

    for D, gv, rv in zip(dims, gres, rres):
        print(f"    D={D}: grid {gv:+.4f}, random {rv:+.4f}"
              + ("   <- random" if rv > gv else ""))
    print(f"    at D=8 random is better by a factor of {gres[-1]/rres[-1]:.0f} in objective")
    print(f"    the reason is in the right panel: 64 grid points give only {n} distinct")
    print(f"    values of the one hyperparameter that matters; 64 random points give 64")
    fig.tight_layout()
    save(fig, "05-search")


# ------------------------------------------------------- 4. best-of-k inflation

def fig_selection_bias():
    rng = np.random.default_rng(1)
    TRUE, SD = 0.80, 0.02
    ks = [1, 2, 5, 10, 20, 50, 100, 200]
    rep, fresh = [], []
    for k in ks:
        s, h = [], []
        for _ in range(6000):
            val = TRUE + rng.normal(0, SD, k)
            s.append(val.max())
            h.append(TRUE + rng.normal(0, SD))
        rep.append(np.mean(s))
        fresh.append(np.mean(h))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.semilogx(ks, rep, "o-", color=PINK, ms=5, lw=1.9, base=2,
                label="best validation score reported")
    ax.semilogx(ks, fresh, "o-", color=BLUE, ms=5, lw=1.9, base=2,
                label="fresh estimate of the winner")
    ax.axhline(TRUE, color=GRAY, ls="--", lw=1.3)
    ax.text(1.1, TRUE - 0.008, "the truth — every configuration is identical",
            fontsize=7.5, color=GRAY)
    ax.set_xlabel("configurations tried")
    ax.set_ylabel("score")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogx(ks, np.array(rep) - TRUE, "o-", color=ORANGE, ms=5, lw=1.9, base=2)
    ax.set_xlabel("configurations tried")
    ax.set_ylabel("inflation of the reported score")
    ax.set_title("bought with nothing but search", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    for k, r_, f_ in zip(ks, rep, fresh):
        print(f"    k={k:3d} | reported {r_:.4f} (+{r_-TRUE:.4f}) | fresh {f_:.4f}")
    print(f"    every configuration has the SAME true score of {TRUE}. The gain is")
    print(f"    entirely selection on validation noise — Part V §14, in the place it")
    print(f"    costs the most.")
    fig.tight_layout()
    save(fig, "07-selection-bias")


# ------------------------------------------------------ 5. successive halving

def fig_halving():
    def run(n_arms, total_budget, mode, seed):
        r = np.random.default_rng(seed)
        quality = r.random(n_arms)

        def observe(a, epochs):
            return quality[a] + r.normal(0, 0.35 / np.sqrt(epochs))

        if mode == "random":
            e = max(1, total_budget // n_arms)
            obs = [observe(a, e) for a in range(n_arms)]
            return quality[int(np.argmax(obs))], quality.max()
        alive = list(range(n_arms))
        spent, e = 0, 1
        while len(alive) > 1 and spent < total_budget:
            spent += len(alive) * e
            obs = {a: observe(a, e) for a in alive}
            alive = sorted(alive, key=lambda a: -obs[a])[:max(1, len(alive) // 2)]
            e *= 2
        return quality[alive[0]], quality.max()

    budgets = [32, 64, 128, 256, 512]
    n_arms = 64
    res = {}
    for mode in ("random", "halving"):
        reg = []
        for B in budgets:
            got = [run(n_arms, B, mode, s) for s in range(500)]
            reg.append(np.mean([best - g for g, best in got]))
        res[mode] = reg

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.loglog(budgets, res["random"], "o-", color=ORANGE, ms=5, lw=1.9, base=2,
              label="random search")
    ax.loglog(budgets, res["halving"], "o-", color=BLUE, ms=5, lw=1.9, base=2,
              label="successive halving")
    ax.set_xlabel(f"total budget (epoch-equivalents), {n_arms} configurations")
    ax.set_ylabel("regret: how far the chosen arm is from the best")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    for B, a, b in zip(budgets, res["random"], res["halving"]):
        print(f"    budget {B:4d} | random regret {a:.4f} | halving {b:.4f} | "
              f"{a/max(b,1e-9):.2f}x better")
    print("    halving spends its budget where the evidence points, so it can afford")
    print("    to start MORE configurations for the same total cost")
    fig.tight_layout()
    save(fig, "06-halving")


CHEAP = [fig_metrics, fig_gradient_check, fig_search, fig_selection_bias, fig_halving]

if __name__ == "__main__":
    print("Part XIV — practice figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
