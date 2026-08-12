"""
Figures for Part XVIII — Reinforcement Learning.

Every claim this part makes that can be measured, is measured here — and almost
all of it can be, because the results that matter in reinforcement learning
appear on small MDPs where the true values are computable in closed form.

Four of these show a method FAILING, which is the point: the maximisation bias
of Q-learning, the divergence of the deadly triad, the variance of REINFORCE
without a baseline, and a policy that scores well on its reward while doing the
wrong thing.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train a network

Runtime: about a minute for --cheap, ten for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "rl"))
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


# ------------------------------------------------------- a random MDP, exactly

def random_mdp(n_states=40, n_actions=4, seed=0, sparsity=5):
    """A random MDP with a sparse transition kernel. Everything about it is
    computable exactly, which is what makes the next few figures checkable."""
    r = np.random.default_rng(seed)
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            nxt = r.choice(n_states, sparsity, replace=False)
            w = r.dirichlet(np.ones(sparsity))
            P[s, a, nxt] = w
    R = r.normal(0, 1, (n_states, n_actions))
    return P, R


def value_iteration(P, R, gamma, iters=2000, tol=1e-14, record=False):
    """Exact value iteration. Returns V* and, optionally, the error at each
    sweep against the converged answer."""
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    hist = []
    for i in range(iters):
        Q = R + gamma * P @ V
        Vn = Q.max(1)
        hist.append(Vn.copy())
        if np.abs(Vn - V).max() < tol:
            V = Vn
            break
        V = Vn
    if record:
        return V, np.array(hist)
    return V


def policy_values(P, R, pi, gamma):
    """Exact V^pi by solving the linear system (I - gamma P_pi) V = R_pi."""
    n_s = P.shape[0]
    Ppi = np.einsum("sa,san->sn", pi, P)
    Rpi = (pi * R).sum(1)
    return np.linalg.solve(np.eye(n_s) - gamma * Ppi, Rpi)


# ------------------------------------------------------- 1. the contraction

def fig_contraction():
    """The Bellman operator is a gamma-contraction in the max norm. That is not
    a heuristic — the error falls by exactly gamma per sweep, and the figure is
    that statement checked to machine precision."""
    P, R = random_mdp(seed=0)
    gammas = [0.5, 0.8, 0.9, 0.99]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ratios = {}
    for g, c in zip(gammas, (GREEN, BLUE, PURPLE, ORANGE)):
        Vstar, hist = value_iteration(P, R, g, iters=3000, record=True)
        err = np.abs(hist - Vstar).max(1)
        err = err[err > 1e-15]
        ax.semilogy(err, color=c, lw=1.8, label=f"$\\gamma = {g}$")
        r = err[1:] / err[:-1]
        mid = r[len(r) // 4: 3 * len(r) // 4]
        ratios[g] = float(np.median(mid))
    ax.set_xlabel("sweep"); ax.set_ylabel(r"$\|V_k - V^*\|_\infty$")
    ax.set_title("Error falls geometrically", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(gammas, [ratios[g] for g in gammas], "o", color=BLUE, ms=8,
            label="measured ratio per sweep")
    ax.plot([0.4, 1.02], [0.4, 1.02], color=GRAY, lw=1.3, ls="--",
            label=r"$\gamma$, the theory")
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("error ratio per sweep")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_title("And the rate is exactly $\\gamma$", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print("    gamma | measured contraction rate | error")
    for g in gammas:
        print(f"     {g:.2f} | {ratios[g]:.6f}                  | {abs(ratios[g]-g):.2e}")
    worst = max(abs(ratios[g] - g) for g in gammas)
    print(f"    the operator contracts at exactly gamma, to within {worst:.1e}")
    print(f"    sweeps to reach 1e-6 at gamma=0.99: "
          f"{int(np.ceil(np.log(1e-6) / np.log(0.99)))}, and at gamma=0.9: "
          f"{int(np.ceil(np.log(1e-6) / np.log(0.9)))}")
    fig.tight_layout()
    save(fig, "01-contraction")
    return ratios


# ------------------------------------------------- 2. the maximisation bias

def fig_max_bias():
    """Q-learning's target takes a max over noisy estimates, and the max of
    noisy estimates is biased upward (Part V section 14, arriving in a new
    place). Double Q-learning removes it, and both are measured here against a
    known answer."""
    rng = np.random.default_rng(0)
    TRUE = 0.0                       # every action has true value 0
    n_actions_list = [2, 4, 8, 16, 32]
    n_trials, n_samples = 4000, 8

    single, double = [], []
    for K in n_actions_list:
        s, d = [], []
        for _ in range(n_trials):
            a = rng.normal(TRUE, 1.0, (K, n_samples))
            b = rng.normal(TRUE, 1.0, (K, n_samples))
            # single estimator: pick the max and report its own estimate
            qa = a.mean(1)
            s.append(qa.max())
            # double estimator: pick with one, evaluate with the other
            qb = b.mean(1)
            d.append(qb[int(np.argmax(qa))])
        single.append(float(np.mean(s))); double.append(float(np.mean(d)))
        print(f"    {K:2d} actions: single estimator {single[-1]:+.4f}   "
              f"double {double[-1]:+.4f}   (truth {TRUE:+.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(n_actions_list, single, "o-", color=ORANGE, lw=1.9, ms=5,
            label="single estimator (Q-learning)")
    ax.plot(n_actions_list, double, "s-", color=GREEN, lw=1.9, ms=5,
            label="double estimator")
    ax.axhline(TRUE, color=GRAY, lw=1.3, ls="--", label="the truth")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("actions to choose among"); ax.set_ylabel("estimated max value")
    ax.set_title("The max of noisy estimates is biased up",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    # and in an actual Q-learning run on a two-state MDP built to expose it
    def run(double_q, episodes=3000, seed=0):
        r = np.random.default_rng(seed)
        # state 0: LEFT -> state 1 (8 actions, reward N(-0.1,1)), RIGHT -> end, 0
        nA = 8
        QA = {0: np.zeros(2), 1: np.zeros(nA)}
        QB = {0: np.zeros(2), 1: np.zeros(nA)}
        left_frac = []
        left_count = 0
        for ep in range(episodes):
            Q = {k: (QA[k] + QB[k]) / 2 for k in QA} if double_q else QA
            a = int(r.integers(2)) if r.random() < 0.1 else int(np.argmax(Q[0]))
            left_count += (a == 0)
            left_frac.append(left_count / (ep + 1))
            if a == 1:
                tgt = 0.0
                if double_q and r.random() < 0.5:
                    QB[0][a] += 0.1 * (tgt - QB[0][a])
                else:
                    QA[0][a] += 0.1 * (tgt - QA[0][a])
                continue
            a2 = int(r.integers(nA))
            rew = r.normal(-0.1, 1.0)
            if double_q:
                if r.random() < 0.5:
                    QA[1][a2] += 0.1 * (rew - QA[1][a2])
                    QA[0][a] += 0.1 * (0 + 1.0 * QB[1][int(np.argmax(QA[1]))] - QA[0][a])
                else:
                    QB[1][a2] += 0.1 * (rew - QB[1][a2])
                    QB[0][a] += 0.1 * (0 + 1.0 * QA[1][int(np.argmax(QB[1]))] - QB[0][a])
            else:
                QA[1][a2] += 0.1 * (rew - QA[1][a2])
                QA[0][a] += 0.1 * (0 + 1.0 * QA[1].max() - QA[0][a])
        return np.array(left_frac)

    sq = np.mean([run(False, seed=s) for s in range(30)], 0)
    dq = np.mean([run(True, seed=s) for s in range(30)], 0)

    ax = axes[1]
    ax.plot(sq * 100, color=ORANGE, lw=1.9, label="Q-learning")
    ax.plot(dq * 100, color=GREEN, lw=1.9, label="double Q-learning")
    ax.axhline(5, color=GRAY, lw=1.3, ls="--", label="optimal (5%, the $\\epsilon$ floor)")
    ax.set_xlabel("episode"); ax.set_ylabel("% choosing the bad action")
    ax.set_title("What it costs in a real run", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.4, frameon=False)
    tidy(ax)

    print(f"    in the two-state MDP, the LEFT action has true value -0.1 and is wrong.")
    print(f"    after 3000 episodes Q-learning still takes it {sq[-1]*100:.1f}% of the time;")
    print(f"    double Q-learning {dq[-1]*100:.1f}%, against an optimal 5%.")
    fig.tight_layout()
    save(fig, "03-max-bias")
    return dict(single=single, double=double, sq=float(sq[-1]), dq=float(dq[-1]))


# --------------------------------------------------------- 3. the deadly triad

def fig_deadly_triad():
    """Baird's counterexample. Off-policy sampling, bootstrapping and linear
    function approximation together diverge — and removing any ONE of the three
    makes it converge. Measured, on the smallest system that shows it."""
    n_states, n_feat = 7, 8
    # the classic feature matrix: state i has feature 2i set, plus a shared one
    X = np.zeros((n_states, n_feat))
    for i in range(6):
        X[i, i] = 2.0
        X[i, 7] = 1.0
    X[6, 6] = 1.0
    X[6, 7] = 2.0
    gamma = 0.99

    def run(bootstrap=True, off_policy=True, tabular=False, steps=1500,
            alpha=0.01, seed=0):
        r = np.random.default_rng(seed)
        w = np.ones(n_feat if not tabular else n_states)
        w[6 if not tabular else 6] = 10.0
        norms = []
        for t in range(steps):
            # ON-POLICY means the STATE distribution is the target policy's too,
            # not merely that the importance ratio is 1. Under the target policy
            # every transition lands in state 6, so that is where the visits go.
            s = int(r.integers(n_states)) if off_policy else 6
            # behaviour policy: mostly "dashed" (to a random upper state);
            # target policy: always "solid" (to state 6)
            if off_policy:
                took_solid = r.random() < 1.0 / 7.0
                rho = (1.0 / (1.0 / 7.0)) if took_solid else 0.0
                s2 = 6 if took_solid else int(r.integers(6))
            else:
                rho = 1.0
                s2 = 6
            feat = X[s] if not tabular else np.eye(n_states)[s]
            feat2 = X[s2] if not tabular else np.eye(n_states)[s2]
            target = 0.0 + (gamma * feat2 @ w if bootstrap else 0.0)
            w = w + alpha * rho * (target - feat @ w) * feat
            norms.append(float(np.linalg.norm(w)))
        return np.array(norms)

    combos = [("all three", dict()),
              ("no bootstrapping", dict(bootstrap=False)),
              ("on-policy", dict(off_policy=False)),
              ("tabular (no approximation)", dict(tabular=True))]
    res = {}
    for name, kw in combos:
        res[name] = run(**kw)
        n = res[name]
        print(f"    {name:28s} ||w|| {n[0]:8.2f} -> {n[-1]:12.4g}   "
              f"{'DIVERGES' if n[-1] > 100 * n[0] else 'converges'}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for (name, _), c in zip(combos, (ORANGE, BLUE, GREEN, PURPLE)):
        ax.semilogy(res[name], color=c, lw=1.9, label=name)
    ax.set_xlabel("update"); ax.set_ylabel(r"$\|\mathbf{w}\|$")
    ax.set_title("Remove any one and it converges", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.4, frameon=False)
    tidy(ax)

    ax = axes[1]
    names = [c[0] for c in combos]
    finals = [res[n][-1] for n in names]
    ax.barh(range(4), finals, color=[ORANGE, BLUE, GREEN, PURPLE], height=0.55)
    ax.set_xscale("log")
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlabel(r"final $\|\mathbf{w}\|$")
    ax.set_title("After 1,500 updates", fontsize=10.5, color=INK)
    tidy(ax)

    print("    all three ingredients are needed. This is not instability that better")
    print("    tuning fixes — the iteration has no fixed point to converge to.")
    fig.tight_layout()
    save(fig, "04-deadly-triad")
    return {k: float(v[-1]) for k, v in res.items()}


# ------------------------------------------------------------ 4. TD against MC

def fig_td_vs_mc():
    """Bias and variance of the two estimators, against values computed
    exactly by solving the linear system."""
    P, R = random_mdp(n_states=12, n_actions=1, seed=3, sparsity=3)
    gamma = 0.9
    pi = np.ones((12, 1))
    Vtrue = policy_values(P, R, pi, gamma)

    def episode(rng, s, horizon=60):
        traj = []
        for _ in range(horizon):
            rew = R[s, 0] + rng.normal(0, 0.5)
            s2 = int(rng.choice(12, p=P[s, 0]))
            traj.append((s, rew, s2))
            s = s2
        return traj

    def run(mode, n_ep, alpha=0.05, seed=0):
        rng = np.random.default_rng(seed)
        V = np.zeros(12)
        for _ in range(n_ep):
            s0 = int(rng.integers(12))
            traj = episode(rng, s0)
            if mode == "td":
                for (s, rew, s2) in traj:
                    V[s] += alpha * (rew + gamma * V[s2] - V[s])
            else:
                G = 0.0
                for (s, rew, s2) in reversed(traj):
                    G = rew + gamma * G
                    V[s] += alpha * (G - V[s])
        return V

    eps = [10, 30, 100, 300, 1000]
    out = {}
    for mode in ("td", "mc"):
        errs = []
        for n in eps:
            runs = np.array([run(mode, n, seed=s) for s in range(20)])
            rmse = np.sqrt(((runs - Vtrue) ** 2).mean(1))
            errs.append((rmse.mean(), runs.std(0).mean()))
        out[mode] = errs
        for n, (e, v) in zip(eps, errs):
            print(f"    {mode.upper()} {n:5d} episodes: RMSE {e:.4f}   "
                  f"across-run spread {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(eps, [e for e, _ in out["td"]], "o-", color=BLUE, lw=1.9, ms=5,
            label="TD(0)")
    ax.plot(eps, [e for e, _ in out["mc"]], "s-", color=ORANGE, lw=1.9, ms=5,
            label="Monte Carlo")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("episodes"); ax.set_ylabel(r"RMSE against exact $V^\pi$")
    ax.set_title("Error against the true values", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(eps, [v for _, v in out["td"]], "o-", color=BLUE, lw=1.9, ms=5,
            label="TD(0)")
    ax.plot(eps, [v for _, v in out["mc"]], "s-", color=ORANGE, lw=1.9, ms=5,
            label="Monte Carlo")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("episodes"); ax.set_ylabel("spread across runs")
    ax.set_title("Where the difference comes from", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    r10 = out["mc"][0][1] / out["td"][0][1]
    print(f"    at 10 episodes Monte Carlo's spread is {r10:.1f}x TD's — that is the")
    print("    variance of a whole return against the variance of one step.")
    fig.tight_layout()
    save(fig, "02-td-vs-mc")
    return out


# ------------------------------------------------------------ 5. exploration

def fig_exploration():
    """Four ways to choose an arm, and the regret each accumulates. The point
    is not that one wins — it is that epsilon-greedy's regret grows linearly
    while the others' grows logarithmically."""
    K, T, RUNS = 10, 20000, 80
    rng_master = np.random.default_rng(0)
    means = rng_master.uniform(0, 1, K)
    best = means.max()

    def run(policy, seed):
        r = np.random.default_rng(seed)
        n = np.zeros(K); s = np.zeros(K)
        regret = np.empty(T)
        tot = 0.0
        for t in range(T):
            if policy == "greedy":
                a = int(np.argmax(np.where(n > 0, s / np.maximum(n, 1), np.inf)))
            elif policy == "eps":
                a = int(r.integers(K)) if r.random() < 0.1 else \
                    int(np.argmax(np.where(n > 0, s / np.maximum(n, 1), 0)))
            elif policy == "ucb":
                if (n == 0).any():
                    a = int(np.argmin(n))
                else:
                    a = int(np.argmax(s / n + np.sqrt(2 * np.log(t + 1) / n)))
            else:   # thompson, Beta posterior on a Bernoulli arm
                a = int(np.argmax(r.beta(1 + s, 1 + (n - s))))
            rew = float(r.random() < means[a])
            n[a] += 1; s[a] += rew
            tot += best - means[a]
            regret[t] = tot
        return regret

    pols = [("greedy", GRAY), ("eps", ORANGE), ("ucb", BLUE), ("thompson", GREEN)]
    names = {"greedy": "greedy", "eps": r"$\epsilon$-greedy (0.1)",
             "ucb": "UCB", "thompson": "Thompson sampling"}
    res = {}
    for p, _ in pols:
        res[p] = np.mean([run(p, s) for s in range(RUNS)], 0)
        print(f"    {names[p]:24s} regret at T={T}: {res[p][-1]:7.1f}   "
              f"(last 5000 steps: {res[p][-1]-res[p][-5001]:6.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for p, c in pols:
        ax.plot(res[p], color=c, lw=1.9, label=names[p])
    ax.set_xlabel("step"); ax.set_ylabel("cumulative regret")
    ax.set_title("Regret over time", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    ax = axes[1]
    for p, c in pols:
        ax.plot(np.arange(1, T + 1), res[p] / np.arange(1, T + 1),
                color=c, lw=1.9, label=names[p])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("regret per step")
    ax.set_title("Falling to zero, or not", fontsize=10.5, color=INK)
    tidy(ax)

    print("    epsilon-greedy keeps exploring at a fixed rate forever, so its regret")
    print("    per step approaches epsilon times the average gap rather than zero.")
    fig.tight_layout()
    save(fig, "06-exploration")
    return {k: float(v[-1]) for k, v in res.items()}


# --------------------------------------------- 6. policy gradient variance

def fig_baseline():
    """REINFORCE is unbiased and its variance is enormous. A baseline removes
    much of it and changes the expectation by exactly nothing — which is the
    part of the claim worth checking rather than trusting."""
    K = 6
    rng = np.random.default_rng(0)
    rewards = rng.normal(0, 1, K) + 5.0        # a large constant offset
    theta = rng.normal(0, 0.4, K)

    def softmax(z):
        e = np.exp(z - z.max()); return e / e.sum()

    pi = softmax(theta)
    # the exact gradient: sum_a pi(a) r(a) grad log pi(a)
    grad_exact = np.zeros(K)
    for a in range(K):
        gl = -pi.copy(); gl[a] += 1.0
        grad_exact += pi[a] * rewards[a] * gl

    def estimate(n, baseline, seed):
        r = np.random.default_rng(seed)
        acts = r.choice(K, n, p=pi)
        rew = rewards[acts] + r.normal(0, 0.5, n)
        b = rew.mean() if baseline == "mean" else 0.0
        G = np.zeros((n, K))
        G[np.arange(n), acts] = 1.0
        G -= pi[None, :]
        return (G * (rew - b)[:, None])

    ns = [8, 32, 128, 512, 2048]
    out = {}
    for baseline in ("none", "mean"):
        bias, var = [], []
        for n in ns:
            ests = np.array([estimate(n, baseline, s).mean(0) for s in range(400)])
            bias.append(float(np.linalg.norm(ests.mean(0) - grad_exact)))
            var.append(float(ests.var(0).sum()))
        out[baseline] = (bias, var)
        for n, bi, v in zip(ns, bias, var):
            print(f"    baseline={baseline:5s} n={n:5d}: bias {bi:.5f}   "
                  f"variance {v:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for b, c, lab in (("none", ORANGE, "no baseline"), ("mean", GREEN, "mean baseline")):
        ax.plot(ns, out[b][1], "o-", color=c, lw=1.9, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("samples per estimate"); ax.set_ylabel("variance of the estimate")
    ax.set_title("A baseline cuts the variance", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for b, c, lab in (("none", ORANGE, "no baseline"), ("mean", GREEN, "mean baseline")):
        ax.plot(ns, out[b][0], "o-", color=c, lw=1.9, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("samples per estimate")
    ax.set_ylabel("bias against the exact gradient")
    ax.set_title("And leaves the expectation alone", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ratio = out["none"][1][0] / out["mean"][1][0]
    print(f"    at 8 samples the baseline cuts variance by {ratio:.0f}x.")
    print(f"    the rewards here sit around {rewards.mean():.1f}; subtracting a constant")
    print("    that does not depend on the action cannot change the expectation,")
    print("    because the expected score function is zero.")
    fig.tight_layout()
    save(fig, "05-baseline")
    return out


# ------------------------------------------------------- 7. reward hacking

def fig_reward_hacking():
    """A policy that maximises its reward and does the wrong thing. The reward
    is a proxy for what was wanted, and the optimal policy for the proxy is not
    the optimal policy for the goal."""
    H, W = 7, 7
    GOAL = (6, 6)
    COINS = [(1, 5), (2, 4), (1, 3), (3, 5), (0, 4)]     # a loop near the start
    gamma = 0.97

    def build(proxy):
        n = H * W
        idx = lambda r, c: r * W + c
        P = np.zeros((n, 4, n)); R = np.zeros((n, 4))
        for r in range(H):
            for c in range(W):
                s = idx(r, c)
                for a, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    r2, c2 = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
                    if (r, c) == GOAL:
                        P[s, a, s] = 1.0
                        continue
                    P[s, a, idx(r2, c2)] = 1.0
                    if (r2, c2) == GOAL:
                        R[s, a] += 1.0            # goal reward, proxy or true
                    if proxy and (r2, c2) in COINS:
                        R[s, a] += 0.30            # the proxy pays for coins
                    R[s, a] -= 0.01                # a small step cost
        return P, R

    Pp, Rp = build(proxy=True)
    Pt, Rt = build(proxy=False)

    def greedy_policy(P, R, g):
        V = value_iteration(P, R, g)
        Q = R + g * P @ V
        return Q.argmax(1)

    def rollout(pi, P, Rtrue, Rproxy, steps=120, start=0):
        s, tp, tt, reached = start, 0.0, 0.0, False
        for t in range(steps):
            a = int(pi[s])
            tp += (gamma ** t) * Rproxy[s, a]
            tt += (gamma ** t) * Rtrue[s, a]
            s = int(np.argmax(P[s, a]))
            if s == GOAL[0] * W + GOAL[1]:
                reached = True
        return tp, tt, reached

    pi_proxy = greedy_policy(Pp, Rp, gamma)
    pi_true = greedy_policy(Pt, Rt, gamma)

    rows = []
    for name, pi in (("trained on the proxy", pi_proxy),
                     ("trained on the goal", pi_true)):
        tp, tt, reached = rollout(pi, Pp, Rt, Rp)
        rows.append((name, tp, tt, reached))
        print(f"    {name:22s} proxy return {tp:6.3f}   true return {tt:6.3f}   "
              f"reaches the goal: {reached}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [r[1] for r in rows], w, color=ORANGE, label="proxy return")
    ax.bar(x + w/2, [r[2] for r in rows], w, color=GREEN, label="true return")
    ax.set_xticks(x)
    ax.set_xticklabels(["trained on\nthe proxy", "trained on\nthe goal"], fontsize=8.5)
    ax.set_ylabel("discounted return")
    ax.set_title("Scoring well, doing the wrong thing", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    grid = np.zeros((H, W))
    for (r, c) in COINS:
        grid[r, c] = 0.5
    grid[GOAL] = 1.0
    ax.imshow(grid, cmap="YlGn", vmin=0, vmax=1.2)
    s = 0
    seen = [(0, 0)]
    for t in range(40):
        a = int(pi_proxy[s]); s = int(np.argmax(Pp[s, a]))
        seen.append((s // W, s % W))
    ys, xs = zip(*seen)
    ax.plot(xs, ys, "-o", color=ORANGE, lw=1.6, ms=3, alpha=0.8)
    ax.set_title("Where the proxy policy goes", fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    print(f"    the proxy policy earns {rows[0][1]/max(rows[1][1],1e-9):.1f}x the proxy return")
    print(f"    of the policy that actually solves the task, and reaches the goal: "
          f"{rows[0][3]}.")
    print("    nothing is broken. It is the optimal policy for the reward it was given.")
    fig.tight_layout()
    save(fig, "07-reward-hacking")
    return rows



CHEAP = [fig_contraction, fig_max_bias, fig_deadly_triad, fig_td_vs_mc,
         fig_exploration, fig_baseline, fig_reward_hacking]

if __name__ == "__main__":
    print("Part XVIII — reinforcement learning figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        try:
            from train_figures import EXPENSIVE       # noqa: E402
        except ImportError:
            EXPENSIVE = []                             # nothing expensive yet — see train_figures.py
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
