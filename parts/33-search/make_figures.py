"""
Figures for Part XXXIII — Search and Planning.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about six minutes. Numpy and matplotlib only.

Unlike Part XXXII, almost nothing here is a benchmark. These are exact
algorithms on constructed problems, so the optimal cost, the node counts and the
true minimax values are known rather than estimated, and the numbers are
properties of the algorithms rather than of this machine. The one exception is
Figure 8, which simulates a population of candidates and a verifier; its
assumptions are stated in the caption and the crossover it reports is derived
from the simulation rather than set by hand.
"""
import heapq
import os
import re
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "se"))
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
# grid world: 4-connected, unit cost, start at one corner and goal at the other
# ---------------------------------------------------------------------------

def make_grid(n, density, rng):
    """A grid with random blocked cells, guaranteed to have a start-goal path."""
    while True:
        free = rng.random((n, n)) > density
        free[0, 0] = free[n - 1, n - 1] = True
        d = bfs_dist(free, (n - 1, n - 1))
        if np.isfinite(d[0, 0]):
            return free, d


def bfs_dist(free, source):
    """Exact cost-to-go from every free cell, by breadth-first search."""
    n = free.shape[0]
    d = np.full((n, n), np.inf)
    d[source] = 0
    q = deque([source])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and free[rr, cc] and not np.isfinite(d[rr, cc]):
                d[rr, cc] = d[r, c] + 1
                q.append((rr, cc))
    return d


def astar(free, start, goal, h):
    """A* with reopening. Returns (cost, expansions, reexpansions).

    `expansions` counts every time a node is taken off the open list and its
    successors are generated, so a node reopened once contributes twice. That is
    the quantity that matters: it is the work done.
    """
    n = free.shape[0]
    g = np.full((n, n), np.inf)
    g[start] = 0.0
    closed = np.zeros((n, n), dtype=bool)
    tie = 0
    open_ = [(h[start], tie, start)]
    expansions = reexpansions = 0
    while open_:
        f, _, u = heapq.heappop(open_)
        if f > g[u] + h[u] + 1e-12:      # a stale entry, superseded by a better g
            continue
        if closed[u]:
            reexpansions += 1
        closed[u] = True
        expansions += 1
        if u == goal:
            return g[u], expansions, reexpansions
        r, c = u
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and free[rr, cc]:
                ng = g[u] + 1.0
                if ng < g[rr, cc] - 1e-12:
                    g[rr, cc] = ng
                    tie += 1
                    heapq.heappush(open_, (ng + h[rr, cc], tie, (rr, cc)))
    return np.inf, expansions, reexpansions


def manhattan(n, goal):
    r = np.abs(np.arange(n)[:, None] - goal[0])
    c = np.abs(np.arange(n)[None, :] - goal[1])
    return (r + c).astype(float)


# ---------------------------------------------------------------------------
# Figure 1 — what a heuristic buys
# ---------------------------------------------------------------------------

def figure1(n=81, density=0.25, instances=14, seed=0):
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.0, 1.0, 11)
    exp = np.zeros((instances, len(alphas)))
    cost = np.zeros((instances, len(alphas)))

    for i in range(instances):
        free, hstar = make_grid(n, density, rng)
        hs = np.where(np.isfinite(hstar), hstar, 0.0)
        for j, a in enumerate(alphas):
            c, e, _ = astar(free, (0, 0), (n - 1, n - 1), a * hs)
            exp[i, j] = e
            cost[i, j] = c

    mean_exp = exp.mean(0)
    same = np.all(np.abs(cost - cost[:, :1]) < 1e-9)
    mono = np.mean([np.all(np.diff(exp[i]) <= 0) for i in range(instances)])
    free_cells = int(np.isfinite(hstar).sum())

    print(f"  grid {n}x{n}, about {free_cells} reachable cells, {instances} instances")
    print(f"  full curve: " + "  ".join(f"{a:.1f}:{e:.0f}" for a, e in zip(alphas, mean_exp)))
    print(f"  alpha=0 (Dijkstra): {mean_exp[0]:8.0f} expansions")
    print(f"  alpha=0.5:          {mean_exp[5]:8.0f} expansions"
          f"  ({mean_exp[0] / mean_exp[5]:.1f}x less)")
    print(f"  alpha=1 (perfect):  {mean_exp[-1]:8.0f} expansions"
          f"  ({mean_exp[0] / mean_exp[-1]:.1f}x less)")
    print(f"  every heuristic returned the same cost: {same}")
    print(f"  fraction of instances monotone in alpha: {mono:.2f}")
    half = (mean_exp[0] - mean_exp[5]) / (mean_exp[0] - mean_exp[-1])
    print(f"  half a perfect heuristic buys {half * 100:.0f}% of the saving")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for i in range(instances):
        ax.plot(alphas, exp[i], color=LIGHT, lw=0.8, zorder=1)
    ax.plot(alphas, mean_exp, "o-", color=BLUE, lw=2.0, ms=4.5, zorder=3)
    ax.set_xlabel(r"heuristic strength $\alpha$   ($h=\alpha h^*$)")
    ax.set_ylabel("nodes expanded")
    ax.set_title("Work", fontsize=10, pad=8)
    tidy(ax)

    ax2.plot(alphas, cost.mean(0), "o-", color=GREEN, lw=2.0, ms=4.5)
    ax2.set_ylim(cost.mean() - 12, cost.mean() + 12)
    ax2.set_xlabel(r"heuristic strength $\alpha$")
    ax2.set_ylabel("path cost")
    ax2.set_title("Quality — unchanged", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("What an admissible heuristic buys, and what it does not",
                 fontsize=10.5, y=1.03)
    save(fig, "01-heuristic")
    return alphas, mean_exp


# ---------------------------------------------------------------------------
# Figure 2 — the price of an inadmissible heuristic
# ---------------------------------------------------------------------------

def figure2(n=81, density=0.25, instances=14, seed=1):
    rng = np.random.default_rng(seed)
    ws = np.array([1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0])
    exp = np.zeros((instances, len(ws)))
    ratio = np.zeros((instances, len(ws)))

    for i in range(instances):
        free, hstar = make_grid(n, density, rng)
        base = manhattan(n, (n - 1, n - 1))          # admissible on a 4-connected grid
        opt = hstar[0, 0]
        for j, w in enumerate(ws):
            c, e, _ = astar(free, (0, 0), (n - 1, n - 1), w * base)
            exp[i, j] = e
            ratio[i, j] = c / opt

    me, mr = exp.mean(0), ratio.mean(0)
    worst = ratio.max(0)
    print(f"  weight  expansions   mean cost / optimal   worst seen   bound")
    for w, e, r, x in zip(ws, me, mr, worst):
        print(f"  {w:5.2f}   {e:8.0f}      {r:.4f}              {x:.4f}       {w:.2f}")
    print(f"  at w=2 the guarantee allows 2.00x optimal; the worst path seen was "
          f"{worst[4]:.3f}x")
    print(f"  the bound is never approached: largest ratio anywhere = "
          f"{ratio.max():.3f} at w={ws[np.unravel_index(ratio.argmax(), ratio.shape)[1]]:.1f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax.semilogy(ws, me, "o-", color=ORANGE, lw=2.0, ms=4.5)
    ax.set_xlabel("weight $w$")
    ax.set_ylabel("nodes expanded")
    ax.set_title("Work falls fast", fontsize=10, pad=8)
    tidy(ax)

    ax2.plot(ws, ws, "--", color=GRAY, lw=1.2, label="guarantee: $w\\times$ optimal")
    ax2.plot(ws, worst, "s-", color=PURPLE, lw=1.4, ms=4, label="worst path seen")
    ax2.plot(ws, mr, "o-", color=ORANGE, lw=2.0, ms=4.5, label="mean path")
    ax2.set_xlabel("weight $w$")
    ax2.set_ylabel("cost / optimal")
    ax2.set_title("The bound is loose", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax2)
    fig.suptitle("Weighted A*: what the guarantee allows, and what happens",
                 fontsize=10.5, y=1.03)
    save(fig, "02-weighted")
    return ws, mr, worst


# ---------------------------------------------------------------------------
# Figure 3 — admissible is not consistent
# ---------------------------------------------------------------------------

def figure3(n=61, density=0.25, instances=14, seed=2):
    """h_inc keeps h* on a random subset of cells and drops to zero elsewhere.

    That is admissible everywhere — it never exceeds h* — but it is wildly
    inconsistent: stepping from a kept cell to a dropped neighbour changes h by
    much more than the unit edge cost. The comparison heuristic alpha*h* is
    scaled so that its *mean* over reachable cells matches, which is the whole
    point: the two carry the same amount of information on average, and only one
    of them is consistent.
    """
    rng = np.random.default_rng(seed)
    keeps = np.array([0.2, 0.4, 0.6, 0.8])
    inc_exp = np.zeros((instances, len(keeps)))
    inc_re = np.zeros((instances, len(keeps)))
    con_exp = np.zeros((instances, len(keeps)))
    con_re = np.zeros((instances, len(keeps)))

    for i in range(instances):
        free, hstar = make_grid(n, density, rng)
        hs = np.where(np.isfinite(hstar), hstar, 0.0)
        for j, k in enumerate(keeps):
            mask = rng.random((n, n)) < k
            h_inc = np.where(mask, hs, 0.0)
            _, e, r = astar(free, (0, 0), (n - 1, n - 1), h_inc)
            inc_exp[i, j], inc_re[i, j] = e, r
            # matched on mean informativeness, consistent by construction
            reach = np.isfinite(hstar)
            alpha = h_inc[reach].mean() / hs[reach].mean()
            _, e2, r2 = astar(free, (0, 0), (n - 1, n - 1), alpha * hs)
            con_exp[i, j] = e2
            con_re[i, j] = r2

    ie, ir, ce = inc_exp.mean(0), inc_re.mean(0), con_exp.mean(0)
    print(f"  kept   inconsistent: expansions / of which re-expansions   consistent, equal mean")
    for k, a, r, c in zip(keeps, ie, ir, ce):
        print(f"  {k:.1f}    {a:8.0f} / {r:6.0f} ({r / a * 100:4.1f}%)"
              f"              {c:8.0f}   ({a / c:.2f}x)")
    print(f"  re-expansions by the consistent heuristic, over every run: "
          f"{int(con_re.sum())}  (theory says zero)")
    j = int(np.argmax(ie / ce))
    print(f"  worst case: keeping {keeps[j]:.0%} of h* costs {ie[j] / ce[j]:.2f}x the "
          f"work of a uniformly weaker heuristic carrying the same information")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(keeps))
    ax.bar(x - 0.19, ce, 0.36, color=GREEN, label="consistent, same mean $h$", zorder=3)
    ax.bar(x + 0.19, ie - ir, 0.36, color=ORANGE, label="inconsistent: first expansions", zorder=3)
    ax.bar(x + 0.19, ir, 0.36, bottom=ie - ir, color=PINK,
           label="inconsistent: re-expansions", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.0%}" for k in keeps])
    ax.set_xlabel("fraction of cells where the heuristic knows $h^*$ (zero elsewhere)")
    ax.set_ylabel("nodes expanded")
    ax.set_title("Admissible is not enough: what inconsistency costs", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "03-consistency")
    return keeps, ie, ir, ce


# ---------------------------------------------------------------------------
# Figure 4 — beam search
# ---------------------------------------------------------------------------

def trellis(V, L, rho, rng):
    """A scoring model and a true quality, correlated at rho.

    The model is what the search maximises; the quality is what we actually
    want. They are two random fields with correlation rho, which is the cleanest
    way to hold `model error` fixed while the search error varies.
    """
    z1 = rng.normal(size=(L, V, V))
    z2 = rng.normal(size=(L, V, V))
    model = z1
    quality = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    return model, quality


def viterbi(score, V, L):
    """Exact argmax over the V^L sequences, by dynamic programming.

    The first step reads score[0][0, v], the same convention the beam uses: the
    sequence starts in state 0. Getting this wrong makes the exact optimum
    incomparable with the beam's answer, which is a quiet way to ruin the figure.
    """
    dp = score[0][0].copy()
    bp = np.zeros((L, V), dtype=int)
    for t in range(1, L):
        cand = dp[:, None] + score[t]
        bp[t] = cand.argmax(0)
        dp = cand.max(0)
    seq, v = [], int(dp.argmax())
    for t in range(L - 1, 0, -1):
        seq.append(v)
        v = bp[t, v]
    seq.append(v)
    return dp.max(), list(reversed(seq))


def beam(score, V, L, width):
    beams = [([v], score[0][0, v]) for v in range(V)]
    beams.sort(key=lambda b: -b[1])
    beams = beams[:width]
    for t in range(1, L):
        cand = []
        for seq, s in beams:
            prev = seq[-1]
            for v in range(V):
                cand.append((seq + [v], s + score[t][prev, v]))
        cand.sort(key=lambda b: -b[1])
        beams = cand[:width]
    return beams[0]


def score_of(field, seq, L):
    tot = field[0][0, seq[0]]
    for t in range(1, L):
        tot += field[t][seq[t - 1], seq[t]]
    return tot


def figure4(V=12, L=10, instances=120, seed=3):
    widths = [1, 2, 4, 8, 16, 32, 64]
    rhos = [0.95, 0.55]
    rng = np.random.default_rng(seed)
    out = {}
    hurt = {}

    for rho in rhos:
        q = np.zeros((instances, len(widths)))
        exact = np.zeros(instances)
        for i in range(instances):
            model, quality = trellis(V, L, rho, rng)
            for j, w in enumerate(widths):
                seq, _ = beam(model, V, L, w)
                q[i, j] = score_of(quality, seq, L)
            _, mseq = viterbi(model, V, L)
            exact[i] = score_of(quality, mseq, L)
        out[rho] = (q.mean(0), exact.mean())
        worse = (np.diff(q, axis=1) < -1e-9)
        hurt[rho] = worse.mean(0)
        print(f"  correlation {rho}:  quality by beam width "
              + " ".join(f"{v:+.2f}" for v in q.mean(0)))
        print(f"     exact model argmax scores {exact.mean():+.2f} in quality"
              f"  (beam 64 reaches {q.mean(0)[-1]:+.2f})")
        print(f"     widening hurt on "
              + ", ".join(f"{w1}->{w2}: {f:.0%}"
                          for w1, w2, f in zip(widths, widths[1:], hurt[rho])))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for rho, c in zip(rhos, (BLUE, ORANGE)):
        m, ex = out[rho]
        ax.semilogx(widths, m, "o-", color=c, lw=2.0, ms=4.5, base=2,
                    label=f"model–quality corr. {rho}")
        ax.axhline(ex, color=c, ls=":", lw=1.1)
    ax.set_xlabel("beam width")
    ax.set_ylabel("true quality of returned sequence")
    ax.set_title("Searching harder, on average", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    xs = np.arange(len(widths) - 1)
    for k, (rho, c) in enumerate(zip(rhos, (BLUE, ORANGE))):
        ax2.bar(xs + (k - 0.5) * 0.38, hurt[rho] * 100, 0.36, color=c, zorder=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{a}→{b}" for a, b in zip(widths, widths[1:])], fontsize=7.5)
    ax2.set_xlabel("beam width doubled")
    ax2.set_ylabel("% of instances made worse")
    ax2.set_title("On single instances", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("A wider beam is not a better answer", fontsize=10.5, y=1.03)
    save(fig, "04-beam")
    return widths, out, hurt


# ---------------------------------------------------------------------------
# Figure 5 — Monte Carlo tree search
# ---------------------------------------------------------------------------

def uct_run(means, budget, c, rng):
    """UCT on a depth-two tree. means[a][b] is the payoff of leaf b under a.

    Single-agent, so an internal node's value is the max over its children, and
    the trap is deliberate: the best action has the best child but the worst
    average child, so anything that averages picks the wrong arm.
    """
    A, B = means.shape
    na = np.zeros(A)
    wa = np.zeros(A)
    nb = np.zeros((A, B))
    wb = np.zeros((A, B))
    for t in range(budget):
        if (na == 0).any():
            a = int(np.flatnonzero(na == 0)[0])
        else:
            a = int(np.argmax(wa / na + c * np.sqrt(np.log(t) / na)))
        if (nb[a] == 0).any():
            b = int(np.flatnonzero(nb[a] == 0)[0])
        else:
            b = int(np.argmax(wb[a] / nb[a] + c * np.sqrt(np.log(na[a]) / nb[a])))
        r = float(rng.random() < means[a, b])
        na[a] += 1
        wa[a] += r
        nb[a, b] += 1
        wb[a, b] += r
    return int(np.argmax(na))         # the standard choice: the most-visited root arm


def figure5(runs=200, seed=4):
    means = np.array([
        [0.80, 0.15, 0.15],           # best child, worst average — the trap
        [0.62, 0.60, 0.58],
        [0.55, 0.52, 0.50],
        [0.40, 0.38, 0.35],
    ])
    best = int(np.argmax(means.max(1)))
    budgets = [64, 256, 1024]
    cs = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5])
    rng = np.random.default_rng(seed)
    acc = np.zeros((len(budgets), len(cs)))

    for i, N in enumerate(budgets):
        for j, c in enumerate(cs):
            hits = sum(uct_run(means, N, c, rng) == best for _ in range(runs))
            acc[i, j] = hits / runs

    print(f"  root values (max over children): "
          + " ".join(f"{v:.2f}" for v in means.max(1))
          + f"   -> arm {best} is optimal")
    print(f"  root means (average of children): "
          + " ".join(f"{v:.2f}" for v in means.mean(1))
          + "   -> averaging prefers a different arm")
    for i, N in enumerate(budgets):
        j = int(np.argmax(acc[i]))
        print(f"  budget {N:5d}: best c = {cs[j]:.2f} at {acc[i, j]:.0%};"
              f"  at c=0.05 {acc[i, 0]:.0%}, at c=2.5 {acc[i, -1]:.0%}")
    argbest = [cs[int(np.argmax(acc[i]))] for i in range(len(budgets))]
    print(f"  the best exploration constant moves {argbest[0]:.2f} -> {argbest[-1]:.2f} "
          f"as the budget grows {budgets[0]}x{budgets[-1] // budgets[0]}")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (N, c) in enumerate(zip(budgets, (BLUE, ORANGE, GREEN))):
        ax.semilogx(cs, acc[i] * 100, "o-", color=c, lw=1.9, ms=4.5,
                    label=f"{N} simulations")
        j = int(np.argmax(acc[i]))
        ax.plot([cs[j]], [acc[i, j] * 100], "*", color=c, ms=13, zorder=5)
    ax.set_xlabel("exploration constant $c$")
    ax.set_ylabel("% of runs choosing the optimal action")
    ax.set_title("The exploration constant is not a property of the algorithm",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    tidy(ax)
    save(fig, "06-mcts")
    return cs, budgets, acc


# ---------------------------------------------------------------------------
# Figure 6 — branch and bound
# ---------------------------------------------------------------------------

def knapsack_bnb(v, w, cap, tight, cap_nodes=60_000):
    """0/1 knapsack, depth first, items pre-sorted by value density.

    tight=True  uses the fractional (LP) relaxation as the bound.
    tight=False uses the sum of all remaining values, which is valid and useless.
    """
    n = len(v)
    order = np.argsort(-(v / w))
    v, w = v[order], w[order]
    suffix = np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])
    best = 0.0
    nodes = 0
    stack = [(0, 0.0, 0.0)]
    while stack:
        i, val, wt = stack.pop()
        nodes += 1
        if nodes >= cap_nodes:
            return best, nodes, True
        if i == n:
            best = max(best, val)
            continue
        if tight:
            b, room, j = val, cap - wt, i
            while j < n and w[j] <= room:
                b += v[j]
                room -= w[j]
                j += 1
            if j < n:
                b += v[j] * room / w[j]
        else:
            b = val + suffix[i]
        if b <= best + 1e-9:
            continue
        best = max(best, val)
        if wt + w[i] <= cap:
            stack.append((i + 1, val + v[i], wt + w[i]))
        stack.append((i + 1, val, wt))
    return best, nodes, False


def figure6(instances=8, seed=5):
    sizes = [10, 14, 18, 22, 26, 30]
    rng = np.random.default_rng(seed)
    res = {True: [], False: []}
    censored = {True: [], False: []}
    for n in sizes:
        for tight in (True, False):
            nodes, cens = [], 0
            for _ in range(instances):
                w = rng.integers(10, 100, n).astype(float)
                v = w * rng.uniform(0.8, 1.2, n)     # weakly correlated: the hard family
                cap = 0.5 * w.sum()
                _, nd, c = knapsack_bnb(v, w, cap, tight)
                nodes.append(nd)
                cens += c
            res[tight].append(np.mean(nodes))
            censored[tight].append(cens)

    print(f"  n     LP bound      sum bound     ratio    (censored runs at 60k nodes)")
    for k, n in enumerate(sizes):
        t, l = res[True][k], res[False][k]
        print(f"  {n:2d}   {t:9.0f}    {l:9.0f}    {l / t:7.1f}x"
              f"     LP {censored[True][k]}, sum {censored[False][k]}")
    print(f"  the full tree at n={sizes[-1]} has 2^{sizes[-1]} = "
          f"{2 ** sizes[-1]:.3g} leaves; the LP bound visits {res[True][-1]:.0f} nodes")
    print(f"  a valid-but-loose bound is not a weaker version of a good one: it is a "
          f"different algorithm, {res[False][-1] / res[True][-1]:.0f}x more work at n={sizes[-1]}")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.semilogy(sizes, [2.0 ** n for n in sizes], ":", color=GRAY, lw=1.3,
                label="$2^n$ — no pruning")
    ax.semilogy(sizes, res[False], "s-", color=ORANGE, lw=1.9, ms=4.5,
                label="bound = value of all remaining items")
    ax.semilogy(sizes, res[True], "o-", color=BLUE, lw=1.9, ms=4.5,
                label="bound = fractional relaxation")
    for k, n in enumerate(sizes):
        if censored[False][k]:
            ax.plot([n], [res[False][k]], "x", color=INK, ms=8, mew=1.6, zorder=5)
    ax.set_xlabel("number of items $n$")
    ax.set_ylabel("nodes explored")
    ax.set_title("Branch and bound is only as good as its bound", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "05-bnb")
    return sizes, res, censored


# ---------------------------------------------------------------------------
# Figure 7 — ordering, in a game tree
# ---------------------------------------------------------------------------

def build_tree(b, d, rng):
    """Leaf values, plus the exact negamax value of every node, bottom up."""
    levels = [rng.normal(size=b ** d)]
    for k in range(d):
        prev = levels[-1].reshape(-1, b)
        levels.append((-prev).max(1))       # negamax: value to the player to move
    return levels


def alphabeta(levels, b, d, node, depth, alpha, beta, order):
    """Returns (value, leaves evaluated). `order` is 'rand', 'best' or 'worst'."""
    if depth == d:
        return levels[0][node], 1
    children = [node * b + i for i in range(b)]
    vals = [-levels[d - depth - 1][c] for c in children]   # value to the player here
    if order == "best":
        idx = np.argsort(-np.array(vals))
    elif order == "worst":
        idx = np.argsort(np.array(vals))
    else:
        idx = np.arange(b)
    total = 0
    best = -np.inf
    for i in idx:
        v, lv = alphabeta(levels, b, d, children[i], depth + 1, -beta, -alpha, order)
        total += lv
        best = max(best, -v)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best, total


def figure7(b=3, trials=12, seed=6):
    depths = [2, 3, 4, 5, 6, 7, 8]
    rng = np.random.default_rng(seed)
    rows = {o: [] for o in ("rand", "best", "worst")}
    for d in depths:
        acc = {o: [] for o in rows}
        for _ in range(trials):
            # The leaf values are drawn independently, so the natural child order
            # already *is* a random move ordering. Permuting as well would compare
            # three orderings on three different trees, which measures nothing.
            levels = build_tree(b, d, rng)
            for o in rows:
                v, n = alphabeta(levels, b, d, 0, 0, -np.inf, np.inf, o)
                assert abs(v - levels[d][0]) < 1e-9, "alpha-beta changed the value"
                acc[o].append(n)
        for o in rows:
            rows[o].append(np.mean(acc[o]))

    full = [float(b ** d) for d in depths]
    sqrt = [float(b ** (d / 2)) for d in depths]
    print(f"  branching {b}, exact minimax leaves = {b}^d")
    print(f"  d   full      random     best-first   sqrt(full)   worst-first")
    for k, d in enumerate(depths):
        print(f"  {d}  {full[k]:8.0f}  {rows['rand'][k]:9.0f}   {rows['best'][k]:9.0f}"
              f"    {sqrt[k]:9.0f}    {rows['worst'][k]:9.0f}")
    k = -1
    print(f"  at depth {depths[k]}: perfect ordering evaluates {rows['best'][k]:.0f} leaves "
          f"against {full[k]:.0f}, a factor of {full[k] / rows['best'][k]:.0f}")
    print(f"  and against the square root {sqrt[k]:.0f} it is "
          f"{rows['best'][k] / sqrt[k]:.2f}x")
    print(f"  worst-first ordering reaches {rows['worst'][k] / full[k] * 100:.1f}% of the "
          f"full tree")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.semilogy(depths, full, "k:", lw=1.4, label="minimax, no pruning")
    ax.semilogy(depths, rows["worst"], "^-", color=PINK, lw=1.7, ms=4.5,
                label="alpha–beta, worst move first")
    ax.semilogy(depths, rows["rand"], "s-", color=ORANGE, lw=1.9, ms=4.5,
                label="alpha–beta, random order")
    ax.semilogy(depths, rows["best"], "o-", color=BLUE, lw=1.9, ms=4.5,
                label="alpha–beta, best move first")
    ax.semilogy(depths, sqrt, "--", color=GRAY, lw=1.3, label=r"$\sqrt{b^{\,d}}$")
    ax.set_xlabel("search depth $d$")
    ax.set_ylabel("leaves evaluated")
    ax.set_title("Move ordering is worth more than any other single change",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax)
    save(fig, "07-alphabeta")
    return depths, rows, full, sqrt


# ---------------------------------------------------------------------------
# Figure 8 — searching against a verifier
# ---------------------------------------------------------------------------

def figure8(problems=4000, seed=7):
    """Sample k candidates, keep the one a verifier likes best.

    The population is the interesting part. Each problem is solved by a sample
    with probability p. When a sample is wrong it is either an ordinary error or
    — with probability lam — the *attractive* error: the same wrong answer the
    model keeps producing, which is exactly the one a verifier trained on the
    same distribution finds convincing. The verifier scores correct answers
    around 1.0, ordinary errors around 0.0, and the attractive error around 1.35,
    all with spread sigma.

    What is swept is lam, the share of errors that are attractive. The verifier's
    accuracy is then *measured* as the fraction of correct/incorrect pairs it
    ranks the right way round. Nothing below is set by hand except the
    population; the crossover is an output.

    Sweeping the verifier's noise instead does not work, and it took a run to see
    why: accuracy is not monotone in sigma. A very sharp verifier is confidently
    wrong about the attractive error, so accuracy rises, peaks and falls again,
    two different noise levels land on the same accuracy with very different
    solve rates, and the curve stops being a function of the quantity on its own
    x-axis. Sweeping lam keeps accuracy monotone.
    """
    rng = np.random.default_rng(seed)
    p, sg = 0.34, 0.55
    ks = [1, 2, 4, 8, 16, 32]
    lams = np.linspace(0.0, 1.0, 11)
    kmax = max(ks)

    acc_meas = []
    solve = np.zeros((len(lams), len(ks)))
    for si, lam in enumerate(lams):
        correct = rng.random((problems, kmax)) < p
        attract = (~correct) & (rng.random((problems, kmax)) < lam)
        mu = np.where(correct, 1.0, np.where(attract, 1.35, 0.0))
        score = mu + rng.normal(scale=sg, size=mu.shape)

        c = score[correct]
        w = score[~correct]
        cs = rng.choice(c, 200_000)
        ws = rng.choice(w, 200_000)
        acc_meas.append(float((cs > ws).mean()))     # measured, not assumed

        for kj, k in enumerate(ks):
            pick = score[:, :k].argmax(1)
            solve[si, kj] = correct[np.arange(problems), pick].mean()

    acc_meas = np.array(acc_meas)
    order = np.argsort(acc_meas)
    a, s = acc_meas[order], solve[order]
    lam_sorted = lams[order]

    print(f"  base solve rate p = {p:.2f}, verifier spread {sg}")
    print(f"  attractive   verifier acc   " + "  ".join(f"k={k:<3d}" for k in ks))
    for i in range(len(a)):
        print(f"     {lam_sorted[i]:.1f}          {a[i]:.3f}      "
              + "  ".join(f"{v:.3f}" for v in s[i]))
    mono = bool(np.all(np.diff(acc_meas) <= 0))
    print(f"  accuracy is monotone in the swept parameter: {mono}")

    cross = {}
    for kj, k in enumerate(ks[1:], 1):
        y = s[:, kj] - p
        idx = np.flatnonzero(np.diff(np.sign(y)) != 0)
        if len(idx):
            i = idx[0]
            t = -y[i] / (y[i + 1] - y[i])
            cross[k] = a[i] + t * (a[i + 1] - a[i])
    for k, x in cross.items():
        print(f"  at k={k:2d}, selection beats taking the first sample only above "
              f"verifier accuracy {x:.3f}")
    print(f"  a verifier at chance would be accuracy 0.500; the break-even is well above "
          f"it, and it rises with k")
    oracle = 1 - (1 - p) ** np.array(ks)
    print(f"  a perfect selector would reach " + " ".join(f"{v:.3f}" for v in oracle))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    for kj, (k, c) in enumerate(zip(ks, (GRAY, CYAN, BLUE, PURPLE, ORANGE, PINK))):
        if k == 1:
            continue
        ax.plot(a, s[:, kj] * 100, "o-", color=c, lw=1.7, ms=3.8, label=f"$k={k}$")
    ax.axhline(p * 100, color=INK, ls="--", lw=1.2)
    ax.text(a.min() + 0.005, p * 100 + 1.2, "take the first sample", fontsize=7.6, color=INK)
    ax.set_xlabel("measured verifier accuracy")
    ax.set_ylabel("% solved")
    ax.set_title("More candidates, one verifier", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper left")
    tidy(ax)

    kk = sorted(cross)
    ax2.plot(kk, [cross[k] for k in kk], "o-", color=ORANGE, lw=2.0, ms=5)
    ax2.axhline(0.5, color=GRAY, ls="--", lw=1.2)
    ax2.text(kk[0], 0.512, "chance", fontsize=7.8, color=GRAY)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("candidates $k$")
    ax2.set_ylabel("accuracy needed to break even")
    ax2.set_title("How good a verifier has to be", fontsize=10, pad=8)
    tidy(ax2)
    fig.suptitle("A weak verifier is worse than no verifier", fontsize=10.5, y=1.03)
    save(fig, "08-verifier")
    return a, s, cross


if __name__ == "__main__":
    for i, fn in enumerate([figure1, figure2, figure3, figure4,
                            figure5, figure6, figure7, figure8], 1):
        print(f"Figure {i}")
        fn()
    print("done")
