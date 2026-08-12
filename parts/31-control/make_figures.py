"""
Figures for Part XXXI — Agents, Control and Embodiment.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about three minutes. Numpy and matplotlib only.

Every system here is written down, so the optimal controller, the true state and
the true failure rate are known rather than estimated. That is what lets each
error be attributed to the thing that caused it.
"""
import os
import re
import time
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "ct"))
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


# ============================================================ FIGURE 1
# What feedback is for.

def first_order(target, gain_true, gain_model, kp, steps=300, dt=0.05, tau=1.0):
    """dx/dt = (-x + gain_true * u) / tau. Open loop uses the modelled gain to
    pick u; closed loop measures x and corrects."""
    x_ol = x_cl = 0.0
    u_ol = target / gain_model                    # what the model says is needed
    for _ in range(steps):
        x_ol += dt * (-x_ol + gain_true * u_ol) / tau
        u_cl = kp * (target - x_cl)
        x_cl += dt * (-x_cl + gain_true * u_cl) / tau
    return x_ol, x_cl


def figure1():
    target = 1.0
    errs = np.linspace(-0.4, 0.4, 41)             # relative model error
    ol, cl = [], {}
    for e in errs:
        o, _ = first_order(target, 1.0, 1.0 + e, 0.0)
        ol.append(abs(o - target))
    for kp in (1.0, 5.0, 20.0):
        row = []
        for e in errs:
            _, c = first_order(target, 1.0 + e, 1.0, kp)
            row.append(abs(c - target))
        cl[kp] = np.array(row)
    ol = np.array(ol)
    i20 = int(np.argmin(np.abs(errs - 0.2)))
    print(f"  at 20% model error: open loop {ol[i20]:.3f}, "
          + ", ".join(f"kp={k:g} {v[i20]:.4f}" for k, v in cl.items()))
    for k, v in cl.items():
        print(f"  kp={k:<4g} worst error over the range {v.max():.4f}"
              f"   (theory 1/(1+kp) = {1/(1+k):.4f} at zero model error)")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(errs * 100, ol, color=ORANGE, lw=2.2, label="open loop")
    for (k, v), c in zip(cl.items(), (CYAN, BLUE, GREEN)):
        ax.plot(errs * 100, v, color=c, lw=1.9, label=f"feedback, $k_p$ = {k:g}")
    ax.set_xlabel("model error (%)"); ax.set_ylabel("steady-state tracking error")
    ax.set_title("What feedback is for, and what it does not fix",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "01-feedback")
    return errs, ol, cl


# ============================================================ FIGURE 2
# The integral term.

def pi_control(target, gain, kp, ki, steps=1200, dt=0.02, tau=1.0, lag=0.15):
    """A first-order plant behind a first-order actuator lag. With no lag at all
    a PI loop on a first-order plant is unconditionally stable, so there is no
    margin to lose and F2.2 cannot be tested — every real actuator has one."""
    x, v, acc = 0.0, 0.0, 0.0
    hist = []
    for _ in range(steps):
        e = target - x
        acc += e * dt
        u = kp * e + ki * acc
        v += dt * (-v + u) / lag                  # the actuator cannot move instantly
        x += dt * (-x + gain * v) / tau
        hist.append(x)
        if not np.isfinite(x) or abs(x) > 1e6:
            return np.array(hist), False
    return np.array(hist), True


def figure2():
    target, gain, kp = 1.0, 1.0, 8.0
    kis = np.linspace(0.0, 400.0, 161)
    final, stable, osc = [], [], []
    for ki in kis:
        h, ok = pi_control(target, gain, kp, ki)
        stable.append(ok)
        final.append(abs(h[-1] - target) if ok else np.nan)
        osc.append(float(h.max() - target) if ok else np.nan)
    stable = np.array(stable)
    thresh = float(kis[np.argmax(~stable)]) if (~stable).any() else None
    print(f"  ki=0    steady-state error {final[0]:.4f}  (theory {1/(1+kp*gain):.4f})")
    for ki in (5.0, 40.0, 150.0):
        j = int(np.argmin(np.abs(kis - ki)))
        print(f"  ki={ki:<5g} steady-state error {final[j]:.2e}   overshoot {osc[j]:.3f}")
    print(f"  diverges above ki = {thresh:.1f}" if thresh else "  stable throughout")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ki, c in zip((0.0, 5.0, 40.0), (ORANGE, BLUE, GREEN)):
        h, _ = pi_control(target, gain, kp, ki)
        a1.plot(np.arange(len(h)) * 0.02, h, color=c, lw=1.8, label=f"$k_i$ = {ki:g}")
    a1.axhline(target, color=GRAY, ls="--", lw=1.0)
    a1.set_xlabel("time (s)"); a1.set_ylabel("output")
    a1.set_title("Integral action removes the offset", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8)
    tidy(a1)
    a2.semilogy(kis, np.maximum(final, 1e-16), color=BLUE, lw=2.0)
    if thresh:
        a2.axvline(thresh, color=ORANGE, lw=1.4, ls="--")
        a2.text(thresh * 0.97, 1e-6, f"diverges\nabove {thresh:.0f}", fontsize=8,
                color=ORANGE, ha="right")
    a2.set_xlabel("integral gain $k_i$"); a2.set_ylabel("steady-state error")
    a2.set_title("And costs stability margin", fontsize=10, pad=8)
    tidy(a2)
    fig.tight_layout()
    save(fig, "02-integral")
    return final, thresh


# ============================================================ FIGURE 3
# LQR is optimal.

def dlqr(A, B, Q, R, iters=800):
    P = np.array(Q, float)
    for _ in range(iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K, P


def rollout_cost(A, B, Q, R, K, x0, steps=200):
    x = np.array(x0, float)
    c = 0.0
    for _ in range(steps):
        u = -K @ x
        c += x @ Q @ x + u @ R @ u
        x = A @ x + B @ u
        if not np.isfinite(x).all() or np.abs(x).max() > 1e8:
            return np.inf
    return float(c)


def integrator_chain(n, dt=0.1):
    """n cascaded integrators: position, velocity, acceleration, ... The natural
    plant for this comparison, because a person hand-tuning it would reach for
    PD feedback on the first two states and that is a controller we can search
    over honestly. A random A with a random B admits no stabilising gain of the
    hand-tuned form at all, which makes the comparison empty rather than
    informative — that was the first version."""
    A = np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = dt
    B = np.zeros((n, 1)); B[-1, 0] = dt
    return A, B


def figure3():
    rows = []
    for n in (2, 3, 4, 5):
        A, B = integrator_chain(n)
        Q, R = np.eye(n), np.array([[0.1]])
        K, _ = dlqr(A, B, Q, R)
        x0 = np.zeros(n); x0[0] = 1.0
        c_lqr = rollout_cost(A, B, Q, R, K, x0, steps=400)
        # hand-tuned PD: feedback on position and velocity only, both gains swept
        best, best_kd = np.inf, None
        for kp in np.logspace(-1, 2.5, 90):
            for kd in np.logspace(-1, 2.5, 90):
                g = np.zeros((1, n)); g[0, 0] = kp
                if n > 1:
                    g[0, 1] = kd
                c = rollout_cost(A, B, Q, R, g, x0, steps=400)
                if c < best:
                    best, best_kd = c, (kp, kd)
        ratio = best / c_lqr if np.isfinite(best) else np.inf
        rows.append((n, c_lqr, best, ratio))
        print(f"  {n} integrators:  LQR {c_lqr:9.3f}   best tuned PD "
              + (f"{best:9.3f}   ratio {ratio:.3f}   at kp={best_kd[0]:.2f} kd={best_kd[1]:.2f}"
                 if np.isfinite(best) else "  no stabilising PD gain"))
    finite = [r for r in rows if np.isfinite(r[3])]
    print(f"  any tuned controller beating LQR: "
          f"{sum(1 for r in finite if r[3] < 1 - 1e-9)}  (must be zero)")
    if len(finite) > 1:
        print(f"  the gap widens with order: "
              + " -> ".join(f"{r[3]:.2f}" for r in finite))

    rows = np.array(rows, float)
    cap = 4.0
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(len(rows))
    shown = np.where(np.isfinite(rows[:, 3]), np.minimum(rows[:, 3], cap), cap)
    ax.bar(x, shown, 0.55, color=BLUE)
    for i, v in enumerate(rows[:, 3]):
        if np.isfinite(v):
            ax.text(i, min(v, cap) + 0.06, f"{v:.2f}", ha="center", fontsize=8.5)
        else:
            ax.text(i, cap * 0.5, "never\nstabilises", ha="center", fontsize=8,
                    color="white", fontweight="bold")
    ax.axhline(1.0, color=ORANGE, lw=1.6, ls="--")
    ax.text(len(rows) - 0.4, 1.06, "LQR", color=ORANGE, fontsize=8.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)} integrators" for v in rows[:, 0]], fontsize=8.5)
    ax.set_ylabel("cost of the best tuned PD, relative to LQR")
    ax.set_ylim(0.9, cap + 0.3)
    ax.set_title("How much a hand-tuned controller leaves on the table",
                 fontsize=10.5, pad=10)
    tidy(ax)
    save(fig, "03-lqr")
    return rows


# ============================================================ FIGURE 4
# ILLUSTRATIVE.
#
# Clipped LQR never drove the state outside its bound on any plant tried here, so
# there was nothing for the constrained planner to be better at, and comparing
# cost instead measured the coarseness of the search rather than the idea. Drawn
# as a schematic with NO numbers on either axis.

def figure4():
    t = np.linspace(0, 1, 300)
    limit = 1.0
    # a trajectory that overshoots the bound, and one that is turned away from it
    unaware = 1.35 * np.sin(np.pi * t) ** 1.4
    aware = np.minimum(unaware, limit - 0.06 * np.sin(np.pi * t) ** 0.5)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axhspan(limit, 1.6, color=ORANGE, alpha=0.10, zorder=0)
    ax.axhline(limit, color=ORANGE, lw=1.6, ls="--", zorder=1)
    ax.text(0.015, limit + 0.045, "the constraint", color=ORANGE, fontsize=9)
    ax.plot(t, unaware, color=ORANGE, lw=2.2, label="a controller with no constraint in it")
    ax.plot(t, aware, color=GREEN, lw=2.2, label="a planner that has one")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.6)
    ax.set_xlabel("time"); ax.set_ylabel("state")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    ax.set_title("Optimising a cost, against respecting a limit", fontsize=10.5, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "04-mpc")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


# ============================================================ FIGURE 5
# ILLUSTRATIVE.
#
# Two attempts. The first used a contracting closed loop, which pulls deviations
# back so nothing compounds (exponent 0.03). The second removed the contraction
# and measured 0.56 against 0.52 — both a random walk, with the two arms barely
# separated. The second attempt was already reaching for the predicted answer,
# so the figure is drawn as a schematic with NO numbers on either axis.

def figure5():
    H = np.logspace(0.3, 2.1, 200)
    fast = H ** 2.0
    slow = H ** 1.0
    fast = fast / fast[0]; slow = slow / slow[0]

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.loglog(H, fast, color=ORANGE, lw=2.2, label="cloned from the expert's states")
    ax.loglog(H, slow, color=GREEN, lw=2.2, label="trained on its own states")
    ax.set_xlabel("horizon"); ax.set_ylabel("deviation from the expert")
    ax.set_xticks([]); ax.set_yticks([])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Why a cloned policy drifts", fontsize=10.5, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "05-imitation")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


# ============================================================ FIGURE 6
# How far a learned model can be trusted.

def figure6():
    rows = {}
    for lam, name in ((1.05, "unstable, $\\lambda$ = 1.05"),
                      (1.00, "marginal, $\\lambda$ = 1.00"),
                      (0.95, "stable, $\\lambda$ = 0.95")):
        err = 0.02                                  # one-step model error
        Hs = np.arange(1, 61)
        d = []
        for H in Hs:
            r = np.random.default_rng(7)
            devs = []
            for _ in range(300):
                x_true = x_mod = 1.0
                for _ in range(H):
                    x_true = lam * x_true
                    x_mod = (lam + err * r.choice([-1.0, 1.0])) * x_mod
                # ABSOLUTE divergence. The relative version is scale-free — it
                # depends only on err/lambda, so it is identical for a stable and
                # an unstable system and cannot test F6.2 at all. That was the
                # first version and all three curves lay on top of each other.
                devs.append(abs(x_mod - x_true))
            d.append(np.mean(devs))
        rows[name] = np.array(d)
        g = np.polyfit(Hs[10:], np.log(np.maximum(d[10:], 1e-30)), 1)[0]
        print(f"  {name}: divergence {d[0]:.4f} at H=1 -> {d[-1]:.4g} at H=60"
              f"   log-slope {g:+.4f} per step  ({np.exp(g):.4f}x per step)")
    Hs = np.arange(1, 61)

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for (name, d), c in zip(rows.items(), (ORANGE, BLUE, GREEN)):
        ax.semilogy(Hs, np.maximum(d, 1e-12), color=c, lw=2.0, label=name)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.text(1.5, 1.25, "as large as the initial state", fontsize=8, color=GRAY)
    ax.set_xlabel("rollout length"); ax.set_ylabel("divergence from the true state")
    ax.set_title("The same one-step model error, three systems",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax)
    save(fig, "06-rollout")
    return Hs, rows


# ============================================================ FIGURE 7
# Planning, and dimension.

def grid_plan(d, res=12):
    """Cost of a full grid search over d joints at `res` resolution — counted
    rather than run, because running it is the point."""
    return float(res) ** d


def rrt(d, seed=0, max_iter=20000, step=0.25):
    """A basic RRT in the unit cube with a slab obstacle in the middle."""
    r = np.random.default_rng(seed)
    start = np.full(d, 0.1); goal = np.full(d, 0.9)

    def free(p):
        return not (0.45 < p[0] < 0.55 and np.all(p[1:] < 0.8))

    nodes = [start]
    for it in range(max_iter):
        q = goal if r.random() < 0.05 else r.random(d)
        arr = np.array(nodes)
        i = int(np.argmin(((arr - q) ** 2).sum(1)))
        direction = q - arr[i]
        n = np.linalg.norm(direction)
        if n < 1e-9:
            continue
        new = arr[i] + step * direction / n
        if np.any(new < 0) or np.any(new > 1) or not free(new):
            continue
        nodes.append(new)
        if np.linalg.norm(new - goal) < step:
            return it + 1, len(nodes)
    return None, len(nodes)


def figure7():
    dims = [2, 3, 4, 5, 6, 7, 8]
    grid, rrt_it, rrt_t = [], [], []
    for d in dims:
        grid.append(grid_plan(d))
        its, ts = [], []
        for s in range(12):
            t0 = time.perf_counter()
            it, _ = rrt(d, seed=s)
            ts.append(time.perf_counter() - t0)
            its.append(it if it else np.nan)
        rrt_it.append(np.nanmean(its)); rrt_t.append(np.mean(ts))
        print(f"  d={d}: grid cells {grid[-1]:.2e}   RRT samples {rrt_it[-1]:8.0f}"
              f"   time {rrt_t[-1]*1e3:7.1f} ms")
    gs = np.polyfit(dims, np.log(grid), 1)[0]
    rs = np.polyfit(dims, np.log(np.maximum(rrt_it, 1)), 1)[0]
    print(f"  log-slope per dimension: grid {gs:+.3f}, RRT {rs:+.3f}"
          f"  -> {np.exp(gs):.1f}x against {np.exp(rs):.2f}x per joint")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.semilogy(dims, grid, "o-", color=ORANGE, lw=1.9, ms=5,
                label=f"grid cells to search  ($\\times${np.exp(gs):.0f}/joint)")
    ax.semilogy(dims, rrt_it, "o-", color=GREEN, lw=1.9, ms=5,
                label=f"samples a planner needs  ($\\times${np.exp(rs):.2f}/joint)")
    ax.axhline(1e9, color=GRAY, ls="--", lw=1.0)
    ax.text(2.1, 1.6e9, "a second of compute, roughly", fontsize=8, color=GRAY)
    ax.set_xlabel("degrees of freedom"); ax.set_ylabel("work")
    ax.set_title("Where exhaustive planning stops", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "07-planning")
    return dims, grid, rrt_it, gs, rs


# ============================================================ FIGURE 8
# Validating a rare failure.

def figure8():
    rates = np.logspace(-9, -4, 40)                # failures per mile
    # miles for a 95% upper bound at the target, seeing zero failures
    zero_fail = 3.0 / rates
    # miles to *estimate* the rate to within 20% with 95% confidence:
    # need about (1.96/0.2)^2 ~ 96 events
    estimate = 96.0 / rates
    for rt in (1e-5, 1e-6, 1e-8):
        # Evaluate the closed form exactly at the target rate, not at the nearest
        # point on the plotting grid — `rates` is log-spaced for the curve, and
        # its nearest sample to 1e-8 is off by ~6%, which once leaked into the
        # caption as 2.8e8 / 9.1e9 instead of the true 3.0e8 / 9.6e9.
        zf, est = 3.0 / rt, 96.0 / rt
        print(f"  target 1 per {1/rt:.0e} miles:  "
              f"{zf:.2e} miles to bound it with zero failures,  "
              f"{est:.2e} miles to estimate it")
    us_annual = 3.2e12
    est_1e8 = 96.0 / 1e-8
    print(f"  for reference, all US driving is about {us_annual:.1e} miles a year;"
          f"  the estimate at 1e-8 is {est_1e8/us_annual*1000:.2f} thousandths of that")
    # rule of three: what zero failures in N miles actually bounds
    Ns = np.logspace(4, 9, 40)
    bound = 3.0 / Ns
    print(f"  zero failures in 1e6 miles bounds the rate only below "
          f"{3.0/1e6:.1e} per mile — one per {1e6/3:.0e} miles")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.loglog(1 / rates, zero_fail, color=GREEN, lw=2.0, label="bound it (zero failures)")
    a1.loglog(1 / rates, estimate, color=ORANGE, lw=2.0, label="estimate it to 20%")
    a1.axhline(us_annual, color=GRAY, ls="--", lw=1.0)
    a1.text(2e4, us_annual * 1.5, "all US driving, one year", fontsize=7.6, color=GRAY)
    a1.set_xlabel("target: one failure per this many miles")
    a1.set_ylabel("miles of evidence needed")
    a1.set_title("The arithmetic of a safety case", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a1)
    a2.loglog(Ns, bound, color=BLUE, lw=2.0)
    a2.set_xlabel("miles driven with zero failures")
    a2.set_ylabel("95% upper bound on the rate")
    a2.set_title("What zero failures establishes", fontsize=10, pad=8)
    tidy(a2)
    fig.tight_layout()
    save(fig, "08-validation")
    return rates, zero_fail, estimate


if __name__ == "__main__":
    for i, fn in enumerate([figure1, figure2, figure3, figure4,
                            figure5, figure6, figure7, figure8], 1):
        print(f"Figure {i}")
        fn()
    print("done")
