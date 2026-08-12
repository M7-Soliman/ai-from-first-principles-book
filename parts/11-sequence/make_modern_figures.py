"""
Figures 10 to 14 for Part XI — the appended movement G.

Predictions for every one of these were recorded before the
script was run. Figures 1 to 9 live in
make_figures.py and predate that discipline; nothing is invented for them here.

    python3 make_modern_figures.py

Runtime: about four minutes. Numpy only.

Two of these count arithmetic rather than timing it. A FLOP count is a property
of the algorithm and transfers; a wall-clock number on this machine would not,
and Part XXXII already makes that argument at length. Where wall clock is the
only honest measure — the scan, whose whole point is latency rather than work —
it is measured and labelled as belonging to one machine.
"""
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "sq"))
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
# Figure 10 — the trade nobody states
# ---------------------------------------------------------------------------

def figure10(d=512):
    """Arithmetic per layer, counted rather than timed.

    Softmax attention: QK^T is 2 L^2 d, the weighted sum of values another
    2 L^2 d, and the three projections 6 L d^2.
    Linear attention with feature dimension d: the running state is d x d, so
    per token it is 2 d^2 to update and 2 d^2 to read, over L tokens, plus the
    same projections.
    """
    Ls = np.array([2 ** k for k in range(6, 19)])
    attn = 4 * Ls ** 2 * d + 6 * Ls * d ** 2
    lin = 4 * Ls * d ** 2 + 6 * Ls * d ** 2
    cross = Ls[np.argmax(attn >= lin)]
    # 4L^2 d + 6L d^2 = 10L d^2  ->  4L^2 d = 4L d^2  ->  L = d. The projection
    # terms cancel, so the crossover is exactly the model dimension. An earlier
    # version compared the quadratic term alone against the full linear cost and
    # got 2.5d, which is the comparison people make in their heads and is wrong.
    exact = float(d)

    print(f"  model dimension {d}")
    print(f"  crossover measured at L = {cross}, closed form gives L = d = {exact:.0f}")
    print(f"    check: at L=d the two costs are "
          f"{4 * d ** 3 + 6 * d ** 3:.4e} and {10 * d ** 3:.4e}")
    for dd in (768, 2048, 4096, 8192):
        print(f"    at model dimension {dd:5d} the crossover is L = {dd}")
    for L in (512, 2048, 8192, 131072):
        i = int(np.argmin(np.abs(Ls - L)))
        print(f"  L={Ls[i]:7d}: attention {attn[i]:.3e}  linear {lin[i]:.3e}"
              f"   ratio {attn[i] / lin[i]:7.2f}x")

    # generation: cost of the next token given a context of length L
    gen_attn = 4 * Ls * d + 6 * d ** 2
    gen_lin = np.full_like(Ls, 4 * d ** 2 + 6 * d ** 2, dtype=float)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.loglog(Ls, attn, "o-", color=ORANGE, lw=1.9, ms=4, label="softmax attention")
    ax.loglog(Ls, lin, "o-", color=BLUE, lw=1.9, ms=4, label="linear / recurrent")
    ax.axvline(cross, color=GRAY, ls="--", lw=1.1)
    ax.text(cross * 1.15, attn[0] * 3, f"crossover\nL = d = {cross}", fontsize=7.6, color=GRAY)
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("FLOPs per layer, one forward pass")
    ax.set_title("Training a whole sequence", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.loglog(Ls, gen_attn, "o-", color=ORANGE, lw=1.9, ms=4, label="softmax attention")
    ax2.loglog(Ls, gen_lin, "o-", color=BLUE, lw=1.9, ms=4, label="linear / recurrent")
    ax2.set_xlabel("context already generated")
    ax2.set_ylabel("FLOPs for the next token")
    ax2.set_title("Generating one more token", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8)
    tidy(ax2)
    fig.suptitle("Where the quadratic cost is, and where it is not",
                 fontsize=10.5, y=1.03)
    save(fig, "10-cost-crossover")
    return Ls, attn, lin, cross


# ---------------------------------------------------------------------------
# Figure 11 — what removing the softmax costs
# ---------------------------------------------------------------------------

def peak_weight_softmax(L, d, rng, trials=200):
    """The largest attention weight achievable when one key is the target.

    The query is set to the target key, which is the best a softmax attention
    head can do at selecting it, so this is an upper bound on how sharply the
    layer can point at one position.
    """
    peaks = []
    for _ in range(trials):
        K = rng.normal(size=(L, d))        # O(1) entries, as after a projection
        q = K[0]                           # the best query for selecting key 0
        s = K @ q / np.sqrt(d)             # the standard 1/sqrt(d) scaling
        w = np.exp(s - s.max())
        w /= w.sum()
        peaks.append(w[0])
    return float(np.mean(peaks))


def peak_weight_linear(L, d, rng, trials=200, feat=None):
    """The same, for linear attention with a positive feature map.

    phi(x) = elu(x) + 1 is the standard choice; the weights are then
    phi(q).phi(k) normalised, with no exponential to sharpen them.
    """
    m = feat or d
    peaks = []
    phi = lambda x: np.where(x > 0, x + 1.0, np.exp(x))
    for _ in range(trials):
        K = rng.normal(size=(L, m))
        q = K[0]
        s = phi(K) @ phi(q)
        peaks.append(float(s[0] / s.sum()))
    return float(np.mean(peaks))


def figure11(d=64, seed=0):
    Ls = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    rng = np.random.default_rng(seed)
    soft = np.array([peak_weight_softmax(L, d, rng) for L in Ls])
    lin = np.array([peak_weight_linear(L, d, rng) for L in Ls])
    unif = 1.0 / Ls

    print(f"  dimension {d}; peak weight on the target position")
    print(f"     L   softmax   linear   uniform   softmax/linear")
    for L, s, l, u in zip(Ls, soft, lin, unif):
        print(f"  {L:5d}   {s:.4f}   {l:.4f}   {u:.4f}   {s / l:8.2f}x")
    print(f"  the gap grows with length: {soft[0] / lin[0]:.2f}x at L={Ls[0]}"
          f" -> {soft[-1] / lin[-1]:.2f}x at L={Ls[-1]}")

    # what feature dimension does linear attention need to reach softmax at L=512?
    L0 = 512
    dims = [16, 32, 64, 128, 256, 512, 1024]
    got = [peak_weight_linear(L0, m, rng, feat=m) for m in dims]
    soft_d = [peak_weight_softmax(L0, m, rng) for m in dims]
    print(f"  at L={L0}, against dimension:")
    for m, g, sd in zip(dims, got, soft_d):
        print(f"    dim {m:5d}: linear {g:.4f}   softmax {sd:.4f}   ratio {sd / g:8.1f}x")
    hit = next((m for m, g, sd in zip(dims, got, soft_d) if g >= 0.5 * sd), None)
    print(f"  dimension at which linear reaches half of softmax's peak: "
          f"{hit if hit else 'none up to ' + str(dims[-1])}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.loglog(Ls, soft, "o-", color=ORANGE, lw=1.9, ms=4.5, label="softmax")
    ax.loglog(Ls, lin, "o-", color=BLUE, lw=1.9, ms=4.5, label="linear")
    ax.loglog(Ls, unif, ":", color=GRAY, lw=1.3, label="uniform, $1/L$")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("weight on the target position")
    ax.set_title("How sharply a layer can point", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.loglog(dims, got, "o-", color=BLUE, lw=1.9, ms=4.5, base=2,
               label="linear attention")
    ax2.loglog(dims, soft_d, "o-", color=ORANGE, lw=1.9, ms=4.5, base=2,
               label="softmax")
    ax2.set_xlabel("feature dimension")
    ax2.set_ylabel(f"weight on the target, $L={L0}$")
    ax2.set_title("What it takes to catch up", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8)
    tidy(ax2)
    fig.suptitle("The softmax is not decoration", fontsize=10.5, y=1.03)
    save(fig, "11-linear-attention")
    return Ls, soft, lin


# ---------------------------------------------------------------------------
# Figure 12 — the parallel scan
# ---------------------------------------------------------------------------

def scan_sequential(a, b):
    """h[t] = a[t] * h[t-1] + b[t], the obvious way. Depth L, work L."""
    h = np.zeros_like(b)
    prev = 0.0
    for t in range(len(b)):
        prev = a[t] * prev + b[t]
        h[t] = prev
    return h


def scan_parallel(a, b):
    """The same recurrence by a Hillis-Steele associative scan.

    The composition (a1,b1) then (a2,b2) is (a1 a2, a2 b1 + b2), which is
    associative — that is the entire reason a recurrence can be parallelised, and
    it is why the nonlinearity had to go.
    """
    a, b = a.copy(), b.copy()
    L = len(b)
    step = 1
    while step < L:
        a_s = np.concatenate([np.ones(step), a[:-step]])
        b_s = np.concatenate([np.zeros(step), b[:-step]])
        b = a * b_s + b
        a = a * a_s
        step *= 2
    return b


def figure12(seed=1):
    rng = np.random.default_rng(seed)
    Ls = [2 ** k for k in range(4, 21)]
    seq_t, par_t, depth_seq, depth_par, work_seq, work_par = [], [], [], [], [], []

    for L in Ls:
        a = rng.uniform(0.85, 0.99, L)
        b = rng.normal(size=L) * 0.1
        if L <= 2 ** 16:
            h1 = scan_sequential(a, b)
            h2 = scan_parallel(a, b)
            assert np.allclose(h1, h2, atol=1e-8), f"scan disagrees at L={L}"
        t0 = time.perf_counter()
        if L <= 2 ** 18:
            scan_sequential(a, b)
        seq_t.append(time.perf_counter() - t0 if L <= 2 ** 18 else np.nan)
        t0 = time.perf_counter()
        scan_parallel(a, b)
        par_t.append(time.perf_counter() - t0)
        depth_seq.append(L)
        depth_par.append(int(np.ceil(np.log2(L))))
        work_seq.append(2 * L)
        work_par.append(3 * L * int(np.ceil(np.log2(L))))

    print("  the two scans agree exactly wherever both were run")
    print(f"     L      depth seq   depth par   work seq   work par   work ratio")
    for i, L in enumerate(Ls):
        if L in (16, 256, 4096, 65536, 2 ** 20):
            print(f"  {L:7d}  {depth_seq[i]:10d}  {depth_par[i]:10d}"
                  f"  {work_seq[i]:9d}  {work_par[i]:9d}"
                  f"   {work_par[i] / work_seq[i]:8.1f}x")
    ok = [i for i in range(len(Ls)) if np.isfinite(seq_t[i])]
    j = next((i for i in ok if par_t[i] < seq_t[i]), None)
    print(f"  wall-clock crossover at L = {Ls[j] if j is not None else 'never'}"
          f"  (this machine; the depth argument is what transfers)")
    print(f"  at L={Ls[ok[-1]]}: sequential {seq_t[ok[-1]] * 1e3:.1f} ms, "
          f"scan {par_t[ok[-1]] * 1e3:.1f} ms")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.loglog(Ls, depth_seq, "o-", color=ORANGE, lw=1.9, ms=4, label="sequential: depth $L$")
    ax.loglog(Ls, depth_par, "o-", color=BLUE, lw=1.9, ms=4, label=r"scan: depth $\log_2 L$")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("sequential steps")
    ax.set_title("Depth — what parallel hardware pays for", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.loglog(Ls, work_seq, "o-", color=ORANGE, lw=1.9, ms=4, label="sequential")
    ax2.loglog(Ls, work_par, "o-", color=BLUE, lw=1.9, ms=4, label="scan")
    ax2.set_xlabel("sequence length $L$")
    ax2.set_ylabel("operations")
    ax2.set_title("Work — what it costs instead", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8)
    tidy(ax2)
    fig.suptitle("A recurrence in logarithmic depth, and what it is paid for with",
                 fontsize=10.5, y=1.03)
    save(fig, "12-parallel-scan")
    return Ls, depth_par, work_seq, work_par


# ---------------------------------------------------------------------------
# Figure 13 — what a fixed state cannot do
# ---------------------------------------------------------------------------

def recall_fixed_state(L, state, rng, trials=400, vocab=64):
    """Associative recall with a linear outer-product memory.

    Keys and values are random unit vectors; the memory is the sum of their
    outer products, read back by projecting on the query key. This is exactly
    what a linear-attention state is, and its capacity is what is being measured
    — no training is involved, so nothing here depends on an optimiser.
    """
    hits = 0
    for _ in range(trials):
        K = rng.normal(size=(L, state)) / np.sqrt(state)
        V = rng.normal(size=(L, vocab)) / np.sqrt(vocab)
        M = K.T @ V                               # the whole state, state x vocab
        j = rng.integers(L)
        out = K[j] @ M
        best = int(np.argmax(V @ out))
        hits += int(best == j)
    return hits / trials


def figure13(seed=2):
    Ls = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    states = [16, 32, 64, 128]
    rng = np.random.default_rng(seed)
    rows = {}
    for s in states:
        rows[s] = np.array([recall_fixed_state(L, s, rng) for L in Ls])
        half = next((L for L, a in zip(Ls, rows[s]) if a < 0.5), None)
        print(f"  state {s:4d}: accuracy " + " ".join(f"{a:.2f}" for a in rows[s])
              + f"   falls below 50% at L={half}")
    halves = []
    for s in states:
        h = next((L for L, a in zip(Ls, rows[s]) if a < 0.5), Ls[-1])
        halves.append(h)
    print(f"  the length at which recall breaks scales with the state: "
          + " ".join(f"{s}->{h}" for s, h in zip(states, halves)))
    ratios = [halves[i + 1] / halves[i] for i in range(len(halves) - 1)]
    print(f"  doubling the state multiplies the usable length by "
          + " ".join(f"{r:.1f}" for r in ratios)
          + "  (attention has no such limit: it stores every key)")
    for s in states:
        a = rows[s]
        hi = next((i for i, v in enumerate(a) if v < 0.9), None)
        lo = next((i for i, v in enumerate(a) if v < 0.1), None)
        if hi is None or lo is None:
            print(f"  state {s:4d}: transition not bracketed within L<={Ls[-1]} "
                  f"— censored, not measured")
        else:
            print(f"  state {s:4d}: 90% -> 10% spans {lo - hi} doublings of length "
                  f"(L={Ls[hi]} to L={Ls[lo]}) — a smooth sigmoid in log length, "
                  f"not a cliff")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for s, c in zip(states, (PINK, ORANGE, GREEN, BLUE)):
        ax.semilogx(Ls, rows[s] * 100, "o-", color=c, lw=1.9, ms=4.5, base=2,
                    label=f"state {s}")
    ax.axhline(100, color=GRAY, ls="--", lw=1.1)
    ax.text(Ls[0], 92, "attention: exact at every length", fontsize=7.8, color=GRAY)
    ax.set_xlabel("number of stored pairs")
    ax.set_ylabel("% recalled correctly")
    ax.set_title("A fixed state has a capacity, and it is reached",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "13-recall")
    return Ls, rows, halves


# ---------------------------------------------------------------------------
# Figure 14 — decay, and making it selective
# ---------------------------------------------------------------------------

def figure14(seed=3, trials=4000):
    """Selective copying: recover the value of the last *marked* token.

    An earlier version asked one scalar state to hold a distant landmark and the
    most recent token at the same time. No scalar can do that, selective or not,
    so it measured the impossibility rather than the mechanism. The task below is
    the one selectivity actually solves: most tokens are noise to be ignored, a
    few are marked, and what matters is the last marked value. A fixed decay has
    to average everything it sees; an input-dependent one can hold through the
    noise and overwrite on a mark.
    """
    rng = np.random.default_rng(seed)
    L, p_mark = 96, 0.06
    lams = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98])

    print("  lambda   measured half-life   1/(1-lambda)   ln2/(1-lambda)")
    for lam in lams[1:]:
        h, v = 0, 1.0
        while v >= 0.5 and h < 100000:
            v *= lam
            h += 1
        print(f"  {lam:6.3f}   {h:16d}   {1 / (1 - lam):12.1f}   "
              f"{np.log(2) / (1 - lam):14.1f}")

    def make(n):
        x = rng.normal(size=(n, L))
        mark = rng.random((n, L)) < p_mark
        mark[:, 0] = True                       # guarantee at least one
        last = np.zeros(n)
        for i in range(n):
            last[i] = x[i, np.flatnonzero(mark[i])[-1]]
        return x, mark, last

    X, M, target = make(trials)

    def run_fixed(lam):
        h = np.zeros(trials)
        for t in range(L):
            h = lam * h + (1 - lam) * X[:, t]
        return h

    def run_selective(gate_noise=0.0):
        """lambda_t = 0 on a mark (overwrite), 1 otherwise (hold). With
        gate_noise the gate is wrong on that fraction of tokens."""
        h = np.zeros(trials)
        flip = rng.random((trials, L)) < gate_noise
        g = np.where(M ^ flip, 0.0, 1.0)
        for t in range(L):
            h = g[:, t] * h + (1 - g[:, t]) * X[:, t]
        return h

    def corr(h):
        return float(np.corrcoef(h, target)[0, 1])

    fixed = np.array([corr(run_fixed(l)) for l in lams])
    best_i = int(np.argmax(fixed))
    print("  lambda   correlation with the last marked value")
    for l, c in zip(lams, fixed):
        print(f"  {l:6.3f}   {c:.3f}")
    print(f"  best fixed decay: lambda={lams[best_i]} at {fixed[best_i]:.3f}")

    noises = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4])
    sel = np.array([corr(run_selective(g)) for g in noises])
    for g, c in zip(noises, sel):
        print(f"  selective, gate wrong on {g:4.0%} of tokens: {c:.3f}")
    tol = next((g for g, c in zip(noises, sel) if c < fixed[best_i]), None)
    print(f"  selectivity beats every fixed decay until the gate is wrong on "
          f"{tol if tol is not None else '>40%'} of tokens")
    print(f"  a perfect gate reaches {sel[0]:.3f} against {fixed[best_i]:.3f} — "
          f"the mechanism is what buys this, not the capacity")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    xs = 1 / (1 - lams[1:])
    hl = []
    for lam in lams[1:]:
        h, v = 0, 1.0
        while v >= 0.5 and h < 100000:
            v *= lam
            h += 1
        hl.append(h)
    ax.loglog(xs, hl, "o-", color=BLUE, lw=1.9, ms=4.5, label="measured half-life")
    ax.loglog(xs, np.log(2) * xs, "--", color=GRAY, lw=1.3,
              label=r"$\ln 2/(1-\lambda)$")
    ax.set_xlabel(r"$1/(1-\lambda)$")
    ax.set_ylabel("half-life, steps")
    ax.set_title("What a fixed decay remembers", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.plot(lams, fixed, "o-", color=ORANGE, lw=1.9, ms=4.5, label="fixed decay")
    ax2.axhline(sel[0], color=BLUE, ls="--", lw=1.4, label="selective, perfect gate")
    for g, c in zip(noises[1:], sel[1:]):
        ax2.plot([0.99], [c], "v", color=BLUE, ms=5, alpha=0.55)
    ax2.text(0.72, sel[-1] + 0.03, "gate degraded to 40% wrong", fontsize=6.8,
             color=BLUE)
    ax2.set_xlabel(r"decay $\lambda$")
    ax2.set_ylabel("correlation with the last marked value")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Selective copying", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8, loc="center left")
    tidy(ax2)
    fig.suptitle("Why the decay had to become a function of the input",
                 fontsize=10.5, y=1.03)
    save(fig, "14-selective")
    return lams, fixed, sel, noises


if __name__ == "__main__":
    for i, fn in enumerate([figure10, figure11, figure12, figure13, figure14], 10):
        print(f"Figure {i}")
        fn()
    print("done")
