"""
Figures for Part XXI — Alignment, Interpretability and Consequence.

The ones here need no training. Two of them are exact arithmetic, because the
question they answer is not empirical: how much evidence a clean evaluation
actually supplies, and how many attempts a rare success needs.

The Goodhart figure is a simulation, and it exists to replace a slogan with a
condition. "Optimising a proxy destroys the true objective" is not true in
general. It is true when the gap between proxy and objective has heavier tails
than the objective itself, and the figure measures both cases side by side.

    python3 make_figures.py

Runtime: a few seconds.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "al"))
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


# --------------------------------------------- 5. what a clean evaluation proves

def fig_rare():
    """Exact arithmetic. If you test n times and see zero failures, what have
    you established about the failure rate? Much less than people assume, and
    the shortfall is the reason 'we evaluated it and it was fine' is not an
    argument about rare harms."""
    print("\n[5] what a clean evaluation proves  (exact)")

    # (a) upper confidence bound on p after zero failures in n trials.
    #     P(0 failures) = (1-p)^n = 0.05  ->  p = 1 - 0.05**(1/n)
    n = np.unique(np.round(np.logspace(1, 6, 220)).astype(int))
    p_upper = 1 - 0.05 ** (1.0 / n)
    rule_of_three = 3.0 / n

    for k in (100, 1000, 10000, 100000):
        i = int(np.argmin(abs(n - k)))
        print(f"    {n[i]:>7d} clean tests  ->  failure rate could still be "
              f"{p_upper[i]*100:.3f}%   (rule of three: {rule_of_three[i]*100:.3f}%)")

    # (b) the other direction: trials needed to SEE a failure that is really there
    print("    to see a failure at least once with probability 0.95:")
    for p in (1e-2, 1e-3, 1e-4, 1e-5):
        need = int(np.ceil(np.log(0.05) / np.log(1 - p)))
        print(f"      rate {p:>8.0e}  needs {need:>8,d} attempts")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    ax.loglog(n, p_upper, color=BLUE, lw=2.0, label="95% upper bound on the rate")
    ax.loglog(n, rule_of_three, color=ORANGE, lw=1.3, ls="--",
              label="the rule of three, $3/n$")
    for k, lab in ((1000, None), (100000, None)):
        i = int(np.argmin(abs(n - k)))
        ax.plot([n[i]], [p_upper[i]], "o", color=INK, ms=4.5, zorder=5)
        ax.annotate(f"{n[i]:,} clean tests\nstill allows {p_upper[i]*100:.2f}%",
                    (n[i], p_upper[i]), textcoords="offset points",
                    xytext=(8, 10), fontsize=8, color=INK)
    ax.set_xlabel("evaluations run, all of them clean")
    ax.set_ylabel("failure rate not yet ruled out")
    ax.set_title("What zero observed failures establishes", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)

    ax = axes[1]
    rates = np.logspace(-6, -1, 200)
    need = np.log(0.05) / np.log1p(-rates)
    ax.loglog(rates, need, color=GREEN, lw=2.0)
    for p in (1e-3, 1e-5):
        v = np.log(0.05) / np.log1p(-p)
        ax.plot([p], [v], "o", color=INK, ms=4.5, zorder=5)
        ax.annotate(f"1 in {int(round(1/p)):,}\nneeds {int(round(v)):,}",
                    (p, v), textcoords="offset points", xytext=(9, -4),
                    fontsize=8, color=INK)
    ax.set_xlabel("true failure rate")
    ax.set_ylabel("attempts to see it once, 95% of the time")
    ax.set_title("What it costs to observe a rare failure", fontsize=10.5, color=INK)
    tidy(ax)

    save(fig, "05-rare-events")
    print("    the two panels are the same fact from both ends. A clean run of")
    print("    1,000 evaluations is consistent with a failure rate of 0.3%, which")
    print("    at a million uses is three thousand failures.")


# --------------------------------------------- 4. when optimising a proxy hurts

def fig_goodhart():
    """'Optimising a proxy destroys the true objective' is not true in general.
    It is true when the proxy's error has heavier tails than the objective. This
    measures both cases with everything else held identical: same correlation
    between proxy and objective, same selection procedure, same sample sizes."""
    print("\n[4] Goodhart, and the condition for it  (simulation)")
    rng = np.random.default_rng(0)
    N = 400_000
    ns = np.unique(np.round(np.logspace(0, 3.7, 26)).astype(int))

    def sweep(err_sampler, label):
        """Best-of-n selection on the proxy; report the TRUE reward obtained."""
        true = rng.standard_normal(N)
        err = err_sampler(N)
        # scale the error so both regimes have the SAME proxy/true correlation.
        err = err / err.std()
        proxy = true + err
        r = np.corrcoef(true, proxy)[0, 1]
        out = []
        for n in ns:
            k = N // n
            t = true[:k * n].reshape(k, n)
            p = proxy[:k * n].reshape(k, n)
            pick = p.argmax(1)
            out.append(t[np.arange(k), pick].mean())
        print(f"    {label:22s} corr(proxy, true) = {r:.3f}")
        return np.array(out), r

    light, r1 = sweep(lambda m: rng.standard_normal(m), "light-tailed error")
    heavy, r2 = sweep(lambda m: rng.standard_t(1.5, m), "heavy-tailed error")

    for lab, y in (("light", light), ("heavy", heavy)):
        i = int(y.argmax())
        print(f"      {lab:5s}: best true reward {y[i]:.3f} at n = {ns[i]}, "
              f"and {y[-1]:.3f} at n = {ns[-1]}  "
              f"({'still climbing' if i == len(ns)-1 else 'past its peak'})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    ax.semilogx(ns, light, "o-", color=BLUE, lw=1.9, ms=4,
                label=f"light-tailed error (r = {r1:.2f})")
    ax.semilogx(ns, heavy, "o-", color=ORANGE, lw=1.9, ms=4,
                label=f"heavy-tailed error (r = {r2:.2f})")
    ax.axhline(0, color=GRAY, lw=1.0, ls=":")
    i = int(heavy.argmax())
    ax.annotate("the proxy keeps improving\nthroughout, in both curves",
                (ns[i], heavy[i]), textcoords="offset points", xytext=(-6, -42),
                fontsize=8, color=GRAY, ha="center")
    ax.set_xlabel("selection pressure (best of $n$ on the proxy)")
    ax.set_ylabel("true reward obtained")
    ax.set_title("Same correlation, opposite outcomes", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)

    # what the tails look like — the whole explanation in one panel
    ax = axes[1]
    g = rng.standard_normal(200_000); g /= g.std()
    t = rng.standard_t(1.5, 200_000); t /= t.std()
    qs = np.linspace(0.5, 0.99999, 400)
    ax.semilogy(qs, np.quantile(np.abs(g), qs), color=BLUE, lw=2.0,
                label="light-tailed error")
    ax.semilogy(qs, np.quantile(np.abs(t), qs), color=ORANGE, lw=2.0,
                label="heavy-tailed error")
    ax.set_xlabel("quantile")
    ax.set_ylabel("size of the proxy's error")
    ax.set_title("Why: the tail is where selection looks", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax)

    save(fig, "04-goodhart")
    print("    both errors have the same variance and give the same correlation.")
    print("    Only the tail differs, and the tail is the whole outcome: hard")
    print("    selection samples from it, so it is the error that gets selected.")


if __name__ == "__main__":
    fig_goodhart()
    fig_rare()
