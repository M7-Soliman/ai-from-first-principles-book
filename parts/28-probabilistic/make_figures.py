"""
Figures for Part XXVIII — Probabilistic Models and Inference.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about four minutes. Numpy and matplotlib only.

Everything approximate here is checked against something exact: enumeration for
the HMM, the analytic posterior for the Kalman filter, the closed form for
mean-field on a Gaussian, and the true covariance for the samplers. That is what
makes each error a measurement rather than an impression.
"""
import os
import re
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "pb"))
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


def partial_corr(a, b, given, bins=12):
    """Correlation between a and b within narrow strata of `given` — a
    conditional dependence measured rather than asserted."""
    q = np.quantile(given, np.linspace(0, 1, bins + 1))
    out, w = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (given >= lo) & (given <= hi)
        if m.sum() > 30:
            out.append(np.corrcoef(a[m], b[m])[0, 1])
            w.append(m.sum())
    return float(np.average(out, weights=w))


# ============================================================ FIGURE 1
# Structure is a claim about independence, and conditioning can create it.

def figure1():
    r = np.random.default_rng(0)
    n = 200_000

    # chain X -> Y -> Z
    X = r.standard_normal(n)
    Y = X + r.standard_normal(n)
    Z = Y + r.standard_normal(n)
    chain_marg = np.corrcoef(X, Z)[0, 1]
    chain_cond = partial_corr(X, Z, Y)

    # collider X -> Y <- Z, with X and Z independent
    A = r.standard_normal(n)
    B = r.standard_normal(n)
    C = A + B + 0.5 * r.standard_normal(n)
    D = C + r.standard_normal(n)              # a descendant of the collider
    coll_marg = np.corrcoef(A, B)[0, 1]
    coll_cond = partial_corr(A, B, C)
    desc_cond = partial_corr(A, B, D)

    print(f"  chain    X,Z  marginal {chain_marg:+.3f}  given Y {chain_cond:+.3f}")
    print(f"  collider A,B  marginal {coll_marg:+.3f}  given C {coll_cond:+.3f}"
          f"  given descendant D {desc_cond:+.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    labels = ["chain\n$X \\to Y \\to Z$", "collider\n$A \\to C \\leftarrow B$",
              "collider's\ndescendant"]
    marg = [chain_marg, coll_marg, coll_marg]
    cond = [chain_cond, coll_cond, desc_cond]
    x = np.arange(3)
    ax.bar(x - 0.19, marg, 0.36, color=BLUE, label="unconditional")
    ax.bar(x + 0.19, cond, 0.36, color=ORANGE,
           label="conditioned (third bar: on the collider's descendant)")
    ax.axhline(0, color=GRAY, lw=0.9)
    for i, (m, c) in enumerate(zip(marg, cond)):
        ax.text(i - 0.19, m + 0.02, f"{m:+.2f}", ha="center", fontsize=8)
        ax.text(i + 0.19, c - 0.05 if c < 0 else c + 0.02, f"{c:+.2f}",
                ha="center", va="top" if c < 0 else "bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("correlation")
    ax.set_title("Conditioning removes dependence, and creates it", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(ax)
    save(fig, "01-dseparation")
    return chain_marg, chain_cond, coll_marg, coll_cond, desc_cond


# ============================================================ FIGURE 2
# Exact inference, checked against enumeration and against the analytic filter.

def hmm_random(K, M, seed):
    r = np.random.default_rng(seed)
    pi = r.dirichlet(np.ones(K))
    A = r.dirichlet(np.ones(K), size=K)
    B = r.dirichlet(np.ones(M) * 0.7, size=K)
    return pi, A, B


def forward_backward(pi, A, B, obs):
    K, T = len(pi), len(obs)
    al = np.zeros((T, K)); sc = np.zeros(T)
    al[0] = pi * B[:, obs[0]]; sc[0] = al[0].sum(); al[0] /= sc[0]
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[:, obs[t]]
        sc[t] = al[t].sum(); al[t] /= sc[t]
    be = np.zeros((T, K)); be[-1] = 1.0
    for t in range(T - 2, -1, -1):
        be[t] = A @ (B[:, obs[t + 1]] * be[t + 1]) / sc[t + 1]
    g = al * be
    return g / g.sum(1, keepdims=True), float(np.log(sc).sum())


def enumerate_posterior(pi, A, B, obs):
    K, T = len(pi), len(obs)
    g = np.zeros((T, K)); tot = 0.0
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * B[path[0], obs[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
        tot += p
        for t, s in enumerate(path):
            g[t, s] += p
    return g / tot, float(np.log(tot))


def kalman(y, F, H, Q, R, m0, P0):
    m, P = m0, P0
    ms, Ps = [], []
    for yt in y:
        m = F * m
        P = F * P * F + Q
        S = H * P * H + R
        K = P * H / S
        m = m + K * (yt - H * m)
        P = (1 - K * H) * P
        ms.append(m); Ps.append(P)
    return np.array(ms), np.array(Ps)


def figure2():
    # (a) forward-backward against enumeration
    errs = []
    for T in range(2, 10):
        pi, A, B = hmm_random(3, 4, 1)
        obs = np.random.default_rng(T).integers(0, 4, T)
        g1, ll1 = forward_backward(pi, A, B, obs)
        g2, ll2 = enumerate_posterior(pi, A, B, obs)
        errs.append(max(np.abs(g1 - g2).max(), abs(ll1 - ll2)))
    print(f"  forward-backward vs enumeration: worst error {max(errs):.2e}")

    # (b) cost: enumeration is K^T, forward-backward is T*K^2
    Ts = np.arange(2, 26)
    K = 3
    enum_ops = K ** Ts.astype(float) * Ts
    fb_ops = Ts * K * K
    cross = Ts[np.argmax(enum_ops > fb_ops)]
    print(f"  enumeration exceeds forward-backward from T = {cross}; "
          f"at T=25 the ratio is {enum_ops[-1]/fb_ops[-1]:.1e}")

    # (c) Kalman against the analytic posterior for a static scalar state
    r = np.random.default_rng(3)
    truth, R = 1.7, 0.6
    y = truth + np.sqrt(R) * r.standard_normal(60)
    P0 = 1e6
    ms, Ps = kalman(y, 1.0, 1.0, 0.0, R, 0.0, P0)
    n = np.arange(1, 61)
    # The analytic posterior for the SAME prior the filter was given. Comparing
    # against the flat-prior formula instead leaves a 2e-6 residual that is the
    # prior's, not the filter's — checked and corrected before anything was read.
    prec = 1 / P0 + n / R
    post_var = 1 / prec
    post_mean = (np.cumsum(y) / R) / prec
    print(f"  Kalman vs analytic posterior: mean {np.abs(ms-post_mean).max():.2e}, "
          f"variance {np.abs(Ps-post_var).max():.2e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.semilogy(Ts, enum_ops, "o-", color=ORANGE, lw=1.8, ms=3.5,
                label="enumerate every path, $K^T$")
    a1.semilogy(Ts, fb_ops, "o-", color=GREEN, lw=1.8, ms=3.5,
                label="forward-backward, $TK^2$")
    a1.set_xlabel("sequence length $T$   ($K = 3$ states)")
    a1.set_ylabel("operations")
    a1.set_title("Same answer, different cost", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a1)

    a2.plot(n, np.abs(ms - post_mean) + 1e-18, "o-", color=BLUE, lw=1.6, ms=3,
            label="posterior mean")
    a2.plot(n, np.abs(Ps - post_var) + 1e-18, "o-", color=PURPLE, lw=1.6, ms=3,
            label="posterior variance")
    a2.axhline(np.finfo(float).eps, color=GRAY, ls="--", lw=1.0)
    a2.text(2, np.finfo(float).eps * 1.5, " float64 eps", fontsize=8, color=GRAY)
    a2.set_yscale("log"); a2.set_ylim(1e-19, 1e-8)
    a2.set_xlabel("observations")
    a2.set_ylabel("|filter $-$ analytic posterior|")
    a2.set_title("The filter is the posterior", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(a2)
    fig.tight_layout()
    save(fig, "02-exact")
    return max(errs), cross, float(np.abs(ms - post_mean).max()), float(np.abs(Ps - post_var).max())


# ============================================================ FIGURE 3
# EM: monotone, and not global.

def em_gmm(x, K, seed, iters=200):
    r = np.random.default_rng(seed)
    mu = r.choice(x, K, replace=False).astype(float)
    var = np.full(K, float(x.var()))
    w = np.full(K, 1.0 / K)
    lls = []
    for _ in range(iters):
        p = w * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        tot = p.sum(1)
        lls.append(float(np.log(np.maximum(tot, 1e-300)).sum()))
        g = p / tot[:, None]
        Nk = g.sum(0) + 1e-12
        w = Nk / len(x)
        mu = (g * x[:, None]).sum(0) / Nk
        var = np.maximum((g * (x[:, None] - mu) ** 2).sum(0) / Nk, 1e-6)
    return np.array(lls), mu


def figure3():
    """Panel 1: the likelihood never decreases. Panel 2: how often EM finds the
    components that generated the data.

    The recovery criterion is in PARAMETER space — both fitted means within a
    quarter of the separation of the two true means — rather than a threshold on
    the log-likelihood. A likelihood threshold is not scale-free: across this sweep
    the spread between restarts runs from 0.2 to 380 nats, so any fixed cutoff
    measures the cutoff rather than the optimisation. That was the first version
    and it was replaced before any outcome was read.
    """
    seps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    fails, identifiable, curves = [], [], None
    n_restart = 200
    mono = 0
    for sep in seps:
        r = np.random.default_rng(int(sep * 10))
        x = np.concatenate([r.normal(0, 1, 600), r.normal(sep, 1, 600)])
        truth = np.array([0.0, sep])
        bad, all_lls, finals, mus = 0, [], [], []
        for s in range(n_restart):
            lls, mu = em_gmm(x, 2, seed=s)
            assert np.all(np.diff(lls) > -1e-6), "EM decreased the likelihood"
            mono += 1
            all_lls.append(lls); finals.append(lls[-1]); mus.append(np.sort(mu))
            if np.abs(np.sort(mu) - truth).max() > sep / 4:
                bad += 1
        f = bad / n_restart
        # Does the BEST restart recover them? If not, the components are not
        # identifiable at this separation and the failure is estimation, not
        # optimisation. Separating the two is the point of the panel.
        best_ok = np.abs(mus[int(np.argmax(finals))] - truth).max() <= sep / 4
        fails.append(f); identifiable.append(bool(best_ok))
        if sep == 2.0:
            curves = all_lls
        se = np.sqrt(max(f * (1 - f), 1e-6) / n_restart) * 100
        print(f"  separation {sep:.1f} sigma: {f*100:5.1f}% +/- {se:.1f} of {n_restart}"
              f" restarts miss the true components   "
              f"{'best restart finds them' if best_ok else 'BEST RESTART ALSO MISSES — not identifiable'}")
    print(f"  the log-likelihood never decreased on any of {mono} runs")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for lls in curves[:14]:
        a1.plot(lls[:60], color=BLUE, lw=0.9, alpha=0.55)
    a1.set_xlabel("EM iteration"); a1.set_ylabel("log-likelihood")
    a1.set_title("Never decreases", fontsize=10, pad=8)
    tidy(a1)
    fa = np.array(fails) * 100
    se = np.sqrt(np.maximum(np.array(fails) * (1 - np.array(fails)), 1e-6) / n_restart) * 100
    ident = np.array(identifiable)
    a2.errorbar(seps, fa, yerr=2 * se, fmt="-", color=ORANGE, lw=1.9,
                capsize=3, ecolor=GRAY, elinewidth=1)
    a2.plot(np.array(seps)[ident], fa[ident], "o", color=ORANGE, ms=6,
            label="components identifiable")
    a2.plot(np.array(seps)[~ident], fa[~ident], "o", color="white", ms=6,
            markeredgecolor=ORANGE, markeredgewidth=1.6,
            label="not identifiable at all")
    a2.legend(frameon=False, fontsize=7.6, loc="upper center")
    a2.set_xlabel("separation between components (in $\\sigma$)")
    a2.set_ylabel("% of restarts missing the true components")
    a2.set_title("And is not global", fontsize=10, pad=8)
    a2.set_ylim(-3, max(fa) + 12)
    tidy(a2)
    fig.tight_layout()
    save(fig, "03-em")
    return seps, fails, identifiable


# ============================================================ FIGURE 4
# Mean-field, against a closed form.

def meanfield_gaussian(rho, iters=500):
    """Coordinate-ascent mean-field for a 2-D Gaussian posterior with unit
    marginals and correlation rho. Returns the fitted marginal sd."""
    P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))     # precision
    # CAVI for a Gaussian: q_i is Gaussian with precision P_ii
    return np.array([1 / np.sqrt(P[0, 0]), 1 / np.sqrt(P[1, 1])])


def meanfield_numeric(rho, steps=4000, seed=0):
    """The same thing found by gradient ascent on the ELBO, so the closed form is
    checked rather than assumed."""
    r = np.random.default_rng(seed)
    S = np.array([[1.0, rho], [rho, 1.0]])
    P = np.linalg.inv(S)
    logs = np.zeros(2)                       # log sd of each factor
    m = np.zeros(2)
    lr = 0.02
    for t in range(steps):
        sd = np.exp(logs)
        eps = r.standard_normal((256, 2))
        z = m + sd * eps
        # d/dparams of E_q[log p(z)] + H[q], reparameterised
        gz = -(z @ P)
        gm = gz.mean(0)
        gs = (gz * eps * sd).mean(0) + 1.0   # +1 from the entropy term
        m += lr * gm
        logs += lr * gs
    return np.exp(logs)


def figure4():
    rhos = np.linspace(0.0, 0.95, 20)
    closed = np.array([meanfield_gaussian(r)[0] for r in rhos])
    theory = np.sqrt(1 - rhos ** 2)
    check = [meanfield_numeric(r) for r in (0.0, 0.5, 0.9)]
    print(f"  closed form vs sqrt(1-rho^2): max deviation {np.abs(closed-theory).max():.2e}")
    for r, c in zip((0.0, 0.5, 0.9), check):
        print(f"  ELBO by gradient ascent at rho={r}: sd {c[0]:.4f}, {c[1]:.4f} "
              f"(closed form {meanfield_gaussian(r)[0]:.4f})")
    print(f"  at rho=0.9 mean-field reports sd {closed[np.argmin(abs(rhos-0.9))]:.3f} "
          f"against a true 1.000")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    a1.plot(rhos, theory, color=GRAY, lw=3.0, alpha=0.45, label="$\\sqrt{1-\\rho^2}$, predicted")
    a1.plot(rhos, closed, color=BLUE, lw=1.9, label="mean-field, measured")
    a1.plot([0.0, 0.5, 0.9], [c[0] for c in check], "o", color=ORANGE, ms=6,
            label="ELBO by gradient ascent")
    a1.axhline(1.0, color=GREEN, lw=1.4, ls="--",
               label="true marginal — and what the\nforward KL recovers")
    a1.set_xlabel("posterior correlation $\\rho$")
    a1.set_ylabel("marginal standard deviation")
    a1.set_ylim(0, 1.15)
    a1.set_title("Mean-field is confidently narrow", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=7.6, loc="lower left")
    tidy(a1)

    # the picture: true contours against the mean-field fit at rho = 0.9
    rho = 0.9
    S = np.array([[1.0, rho], [rho, 1.0]])
    g = np.linspace(-3.2, 3.2, 240)
    Xg, Yg = np.meshgrid(g, g)
    pts = np.stack([Xg, Yg], -1)
    Pi = np.linalg.inv(S)
    dens = np.exp(-0.5 * np.einsum("...i,ij,...j->...", pts, Pi, pts))
    sd = meanfield_gaussian(rho)
    q = np.exp(-0.5 * (Xg ** 2 / sd[0] ** 2 + Yg ** 2 / sd[1] ** 2))
    a2.contour(Xg, Yg, dens, levels=[0.14, 0.4, 0.8], colors=GREEN, linewidths=1.6)
    a2.contour(Xg, Yg, q, levels=[0.14, 0.4, 0.8], colors=BLUE, linewidths=1.6,
               linestyles="--")
    a2.set_title("$\\rho = 0.9$:  true (solid), mean-field (dashed)", fontsize=10, pad=8)
    a2.set_xlabel("$z_1$"); a2.set_ylabel("$z_2$")
    a2.set_aspect("equal")
    tidy(a2, grid=False)
    fig.tight_layout()
    save(fig, "04-meanfield")
    return rhos, closed, theory


# ============================================================ FIGURE 5
# What a gradient buys a sampler.

def ess(x):
    """Effective sample size from the initial-positive-sequence estimator."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair <= 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def rw_metropolis(logp, x0, step, n):
    x = x0.copy(); lp = logp(x)
    out = np.empty((n, len(x0))); acc = 0
    r = np.random.default_rng(0)
    for i in range(n):
        y = x + step * r.standard_normal(len(x0))
        ly = logp(y)
        if np.log(r.random()) < ly - lp:
            x, lp = y, ly; acc += 1
        out[i] = x
    return out, acc / n


def hmc(logp, gradp, x0, step, L, n, seed=0):
    r = np.random.default_rng(seed)
    x = x0.copy(); out = np.empty((n, len(x0))); acc = 0; grads = 0
    for i in range(n):
        p = r.standard_normal(len(x0))
        H0 = -logp(x) + 0.5 * p @ p
        xn, pn = x.copy(), p.copy()
        pn += 0.5 * step * gradp(xn); grads += 1
        for _ in range(L):
            xn += step * pn
            pn += step * gradp(xn); grads += 1
        pn -= 0.5 * step * gradp(xn)
        H1 = -logp(xn) + 0.5 * pn @ pn
        if np.log(r.random()) < H0 - H1:
            x = xn; acc += 1
        out[i] = x
    return out, acc / n, grads


def figure5():
    conds = [1, 4, 16, 64, 256]
    rw_e, hmc_e = [], []
    for c in conds:
        sd = np.array([1.0, 1.0 / np.sqrt(c)])
        logp = lambda z: -0.5 * np.sum((z / sd) ** 2)
        gradp = lambda z: -z / sd ** 2
        n = 20000
        s_rw, a_rw = rw_metropolis(logp, np.zeros(2), 2.4 * sd.min(), n)
        e_rw = ess(s_rw[:, 0]) / n                       # one density eval per step
        L = max(4, int(3 * np.sqrt(c)))
        s_h, a_h, g = hmc(logp, gradp, np.zeros(2), 0.7 * sd.min(), L, n // 20)
        e_h = ess(s_h[:, 0]) / g
        rw_e.append(e_rw); hmc_e.append(e_h)
        print(f"  cond {c:4d}:  RW ESS/eval {e_rw:.2e} (acc {a_rw:.2f})   "
              f"HMC ESS/grad {e_h:.2e} (acc {a_h:.2f})   ratio {e_h/e_rw:.1f}x")
    sl = lambda y: np.polyfit(np.log(conds), np.log(y), 1)[0]
    print(f"  slopes — RW {sl(rw_e):.2f} (predicted -1), HMC {sl(hmc_e):.2f} (predicted -0.5)")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.loglog(conds, rw_e, "o-", color=ORANGE, lw=1.9, ms=5,
              label=f"random-walk Metropolis  (slope {sl(rw_e):.2f})")
    ax.loglog(conds, hmc_e, "o-", color=GREEN, lw=1.9, ms=5,
              label=f"Hamiltonian Monte Carlo  (slope {sl(hmc_e):.2f})")
    ax.set_xlabel("condition number of the target")
    ax.set_ylabel("effective samples per density or gradient evaluation")
    ax.set_title("What the gradient is worth as the target gets harder",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(ax)
    save(fig, "05-samplers")
    return conds, rw_e, hmc_e, sl(rw_e), sl(hmc_e)


# ============================================================ FIGURE 6
# A diagnostic that passes while the chain is wrong.

def rhat(chains):
    """Gelman-Rubin, on an (m, n) array of scalar draws."""
    m, n = chains.shape
    W = chains.var(1, ddof=1).mean()
    B = n * chains.mean(1).var(ddof=1)
    V = (n - 1) / n * W + B / n
    return float(np.sqrt(V / W))


def bimodal_logp(z, sep):
    a = -0.5 * (z - sep) ** 2
    b = -0.5 * (z + sep) ** 2
    return np.logaddexp(a, b)


def figure6():
    # The first version used sep = 4.0 with a proposal of 0.8, and the chains
    # crossed between modes often enough that R-hat reported 1.20 in both arms —
    # the target was not actually well separated, so nothing was trapped and the
    # experiment could not test the claim. Corrected before any outcome was read.
    sep = 9.0
    n, m = 20000, 4
    res = {}
    for name, starts in (("all chains in one mode", [sep] * m),
                         ("chains dispersed across both", [sep, sep, -sep, -sep])):
        chains = []
        for j, x0 in enumerate(starts):
            r = np.random.default_rng(100 + j)
            x = float(x0); lp = bimodal_logp(x, sep)
            out = np.empty(n)
            for i in range(n):
                y = x + 0.8 * r.standard_normal()
                ly = bimodal_logp(y, sep)
                if np.log(r.random()) < ly - lp:
                    x, lp = y, ly
                out[i] = x
            chains.append(out)
        C = np.array(chains)
        share = float((C > 0).mean())
        res[name] = (rhat(C), share, C)
        print(f"  {name:32s} R-hat {res[name][0]:.4f}   mass found in the right mode "
              f"{share*100:.1f}%  (truth 50%)")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    grid = np.linspace(-15, 15, 500)
    dens = np.exp(bimodal_logp(grid, sep)); dens /= dens.sum() * (grid[1] - grid[0])
    for ax, (name, (rh, share, C)) in zip(axes, res.items()):
        ax.hist(C.ravel(), bins=110, range=(-15, 15), density=True, color=BLUE, alpha=0.75)
        ax.plot(grid, dens, color=GREEN, lw=1.8)
        ax.set_title(f"{name}\n$\\hat{{R}}$ = {rh:.3f}", fontsize=9.5, pad=7)
        ax.set_xlabel("$z$")
        tidy(ax, grid=False)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "06-diagnostics")
    return {k: (v[0], v[1]) for k, v in res.items()}


# ============================================================ FIGURE 7
# The reverse SDE and the probability-flow ODE.

def figure7():
    """A Gaussian mixture target whose exact score is available at every noise
    level, so the two integrators are compared against the truth rather than
    against a trained network."""
    mus, ws = np.array([-2.0, 2.0]), np.array([0.5, 0.5])
    s0 = 0.5
    # T must be long enough that the forward process has actually mixed, or the
    # terminal distribution is still a mixture and initialising both integrators
    # from a single Gaussian starts them from the wrong place. At T = 1,
    # exp(-T) = 0.37 and the two components are still 0.74 apart; at T = 5 they
    # are 0.013 apart. Caught and corrected before any outcome was read.
    T = 5.0

    def marg(t):                       # variance-preserving-style schedule
        return np.exp(-t), 1 - np.exp(-2 * t)

    def score(x, t):
        a, v = marg(t)
        var = a ** 2 * s0 ** 2 + v
        d = x[:, None] - a * mus[None, :]
        w = ws * np.exp(-0.5 * d ** 2 / var)
        w /= w.sum(1, keepdims=True)
        return -(w * d).sum(1) / var

    def sample_target(n, r):
        k = r.choice(2, n, p=ws)
        return mus[k] + s0 * r.standard_normal(n)

    true = sample_target(200_000, np.random.default_rng(0))
    rows = []
    for steps in (5, 10, 20, 50, 100):
        dt = T / steps
        r = np.random.default_rng(1)
        n = 40_000
        x_ode = r.standard_normal(n)          # p_T is N(0,1) to 1e-4 at T = 5
        x_sde = x_ode.copy()
        for i in range(steps):
            t = T - i * dt
            s_o, s_s = score(x_ode, t), score(x_sde, t)
            # probability-flow ODE:   dx = [f - 0.5 g^2 score] dt   with f=-x, g^2=2
            x_ode = x_ode - dt * (-x_ode - s_o)
            # reverse SDE:            dx = [f - g^2 score] dt + g dW
            x_sde = x_sde - dt * (-x_sde - 2 * s_s) + np.sqrt(2 * dt) * r.standard_normal(n)
        rows.append((steps, x_ode.mean(), x_ode.var(), x_sde.mean(), x_sde.var()))
        print(f"  steps {steps:3d}:  ODE mean {x_ode.mean():+.3f} var {x_ode.var():.3f}   "
              f"SDE mean {x_sde.mean():+.3f} var {x_sde.var():.3f}")
    print(f"  target: mean {true.mean():+.3f} var {true.var():.3f}")

    # determinism: repeat from one starting point
    r = np.random.default_rng(7)
    x0 = r.standard_normal(1)
    ends_o, ends_s = [], []
    for rep in range(24):
        dt, steps = T / 100, 100
        xo, xs = x0.copy(), x0.copy()
        rr = np.random.default_rng(1000 + rep)
        for i in range(steps):
            t = T - i * dt
            xo = xo - dt * (-xo - score(xo, t))
            xs = xs - dt * (-xs - 2 * score(xs, t)) + np.sqrt(2 * dt) * rr.standard_normal(1)
        ends_o.append(float(xo[0])); ends_s.append(float(xs[0]))
    print(f"  from one start, 24 runs: ODE spread {np.std(ends_o):.2e}, "
          f"SDE spread {np.std(ends_s):.3f}")

    rows = np.array(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.plot(rows[:, 0], rows[:, 2], "o-", color=BLUE, lw=1.8, ms=4.5, label="probability-flow ODE")
    a1.plot(rows[:, 0], rows[:, 4], "o-", color=ORANGE, lw=1.8, ms=4.5, label="reverse SDE")
    a1.axhline(true.var(), color=GREEN, lw=1.5, ls="--", label="target")
    a1.set_xscale("log"); a1.set_xlabel("integration steps"); a1.set_ylabel("variance of samples")
    a1.set_title("Same marginals, different integrators", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(a1)

    a2.scatter(np.zeros(24) + 1, ends_o, s=18, color=BLUE, label="ODE")
    a2.scatter(np.zeros(24) + 2, ends_s, s=18, color=ORANGE, label="SDE")
    a2.set_xlim(0.4, 2.6); a2.set_xticks([1, 2]); a2.set_xticklabels(["ODE", "SDE"])
    a2.set_ylabel("sample value")
    a2.set_title("24 runs from one starting point", fontsize=10, pad=8)
    tidy(a2)
    fig.tight_layout()
    save(fig, "07-sde")
    return rows, float(true.var()), float(np.std(ends_o)), float(np.std(ends_s))


# ============================================================ FIGURE 8
# Where KL fails and transport does not.

def figure8():
    seps = np.linspace(0.0, 4.0, 41)
    kl, w1, kl_ov, w1_ov = [], [], [], []
    n = 4000
    r = np.random.default_rng(0)
    base = np.sort(r.uniform(0, 1, n))                # uniform on [0,1]
    for s in seps:
        # disjoint case: supports [0,1] and [s, s+1] once s >= 1
        q = base + s
        w1.append(float(np.abs(np.sort(base) - np.sort(q)).mean()))
        kl.append(np.inf if s >= 1.0 else float(-np.log(max(1 - s, 1e-12))))
        # overlapping case: two Gaussians, both full support
        w1_ov.append(float(s))                        # W1 between N(0,1), N(s,1)
        kl_ov.append(float(0.5 * s ** 2))             # KL between them
    finite = np.isfinite(kl)
    print(f"  disjoint supports: W1 is finite at every separation "
          f"(max {max(w1):.2f}); KL is infinite from separation {seps[~finite][0]:.1f}")
    sl = np.polyfit(seps[seps > 1.2], np.array(w1)[seps > 1.2], 1)[0]
    print(f"  W1 grows linearly with separation, slope {sl:.4f} (predicted 1.0)")
    print(f"  overlapping supports: both finite; KL is quadratic, W1 linear")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.plot(seps, w1, color=GREEN, lw=2.0, label="$W_1$")
    kl_plot = np.array(kl, float)
    a1.plot(seps[finite], kl_plot[finite], color=ORANGE, lw=2.0, label="KL")
    a1.axvline(1.0, color=GRAY, ls="--", lw=1.0)
    a1.text(1.06, 2.3, "supports separate;\nKL is $\\infty$ beyond here",
            fontsize=8, color=GRAY)
    a1.set_ylim(0, 4.2)
    a1.set_xlabel("separation"); a1.set_ylabel("divergence / distance")
    a1.set_title("Disjoint supports", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(a1)

    a2.plot(seps, w1_ov, color=GREEN, lw=2.0, label="$W_1$  (linear)")
    a2.plot(seps, kl_ov, color=ORANGE, lw=2.0, label="KL  (quadratic)")
    a2.set_xlabel("separation"); a2.set_ylabel("divergence / distance")
    a2.set_title("Overlapping supports", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(a2)
    fig.tight_layout()
    save(fig, "08-transport")
    return sl, max(w1)


if __name__ == "__main__":
    print("Figure 1 — d-separation, measured")
    figure1()
    print("Figure 2 — exact inference against exact answers")
    figure2()
    print("Figure 3 — EM")
    figure3()
    print("Figure 4 — mean-field against its closed form")
    figure4()
    print("Figure 5 — samplers")
    figure5()
    print("Figure 6 — a diagnostic that passes")
    figure6()
    print("Figure 7 — the SDE and the ODE")
    figure7()
    print("Figure 8 — KL against transport")
    figure8()
    print("done")
