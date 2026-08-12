"""
Reference solutions for the Part XVIII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ---------------------------------------------------------------- exact MDPs

def bellman_backup(P, R, V, gamma):
    return (R + gamma * (P @ V)).max(1)


def value_iteration(P, R, gamma, tol=1e-12, max_iters=10000):
    V = np.zeros(P.shape[0])
    for k in range(1, max_iters + 1):
        Vn = bellman_backup(P, R, V, gamma)
        if np.abs(Vn - V).max() < tol:
            return Vn, k
        V = Vn
    return V, max_iters


def greedy_policy(P, R, V, gamma):
    return (R + gamma * (P @ V)).argmax(1)


def policy_value_exact(P, R, pi, gamma):
    n = P.shape[0]
    Ppi = np.einsum("sa,san->sn", pi, P)
    Rpi = (np.asarray(pi) * R).sum(1)
    return np.linalg.solve(np.eye(n) - gamma * Ppi, Rpi)


def _onehot(idx, n_a):
    pi = np.zeros((len(idx), n_a))
    pi[np.arange(len(idx)), idx] = 1.0
    return pi


def policy_iteration(P, R, gamma, max_iters=1000):
    n_s, n_a, _ = P.shape
    pi = np.zeros(n_s, dtype=int)
    for k in range(1, max_iters + 1):
        V = policy_value_exact(P, R, _onehot(pi, n_a), gamma)
        pi2 = greedy_policy(P, R, V, gamma)
        if np.array_equal(pi2, pi):
            return pi, V, k
        pi = pi2
    return pi, policy_value_exact(P, R, _onehot(pi, n_a), gamma), max_iters


def contraction_rate(P, R, gamma, n_sweeps=200):
    Vstar, _ = value_iteration(P, R, gamma)
    V = np.zeros(P.shape[0])
    errs = []
    for _ in range(n_sweeps):
        V = bellman_backup(P, R, V, gamma)
        e = np.abs(V - Vstar).max()
        if e < 1e-14:
            break
        errs.append(e)
    errs = np.array(errs)
    if len(errs) < 4:
        return float(gamma)
    r = errs[1:] / errs[:-1]
    return float(np.median(r[len(r) // 4: 3 * len(r) // 4]))


def effective_horizon(gamma):
    return 1.0 / (1.0 - gamma)


def sweeps_for_accuracy(gamma, eps):
    return int(np.ceil(np.log(eps) / np.log(gamma)))


# ------------------------------------------------------------ sampled updates

def td0_update(V, s, r, s2, alpha, gamma):
    V = np.asarray(V, dtype=float)
    V[s] += alpha * (r + gamma * V[s2] - V[s])
    return V


def mc_returns(rewards, gamma):
    out = np.zeros(len(rewards))
    G = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def n_step_return(rewards, V_bootstrap, gamma, n):
    n = min(n, len(rewards))
    G = sum((gamma ** k) * rewards[k] for k in range(n))
    return float(G + (gamma ** n) * V_bootstrap)


def q_learning_target(r, Q_next, gamma):
    return float(r + gamma * np.max(Q_next))


def sarsa_target(r, Q_next, a_next, gamma):
    return float(r + gamma * Q_next[a_next])


def double_q_target(r, QA_next, QB_next, gamma):
    return float(r + gamma * QB_next[int(np.argmax(QA_next))])


def max_estimator_bias(n_actions, n_samples, n_trials, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sd, (n_trials, n_actions, n_samples))
    return float(x.mean(2).max(1).mean())


# --------------------------------------------------------- policy gradients

def softmax_policy(theta):
    z = np.asarray(theta, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def score_function(theta, a):
    pi = softmax_policy(theta)
    g = -pi.copy()
    g[a] += 1.0
    return g


def exact_policy_gradient(theta, rewards):
    pi = softmax_policy(theta)
    g = np.zeros_like(pi)
    for a in range(len(pi)):
        g += pi[a] * rewards[a] * score_function(theta, a)
    return g


def reinforce_estimate(theta, rewards, n, baseline=None, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    pi = softmax_policy(theta)
    acts = rng.choice(len(pi), n, p=pi)
    rew = np.asarray(rewards, float)[acts]
    if noise:
        rew = rew + rng.normal(0.0, noise, n)
    b = rew.mean() if baseline == "mean" else (0.0 if baseline is None else baseline)
    G = np.zeros((n, len(pi)))
    G[np.arange(n), acts] = 1.0
    G -= pi[None, :]
    return (G * (rew - b)[:, None]).mean(0)


def advantage_from_values(rewards, values, gamma):
    r = np.asarray(rewards, float)
    v = np.asarray(values, float)
    return r + gamma * v[1:] - v[:-1]


def gae(rewards, values, gamma, lam):
    d = advantage_from_values(rewards, values, gamma)
    out = np.zeros_like(d)
    acc = 0.0
    for t in range(len(d) - 1, -1, -1):
        acc = d[t] + gamma * lam * acc
        out[t] = acc
    return out


def ppo_objective(ratio, advantage, eps=0.2):
    r = np.asarray(ratio, float)
    a = np.asarray(advantage, float)
    return np.minimum(r * a, np.clip(r, 1 - eps, 1 + eps) * a)


# --------------------------------------------------------------- exploration

def ucb_action(counts, values, t, c=2.0):
    counts = np.asarray(counts, float)
    if (counts == 0).any():
        return int(np.argmin(counts))
    return int(np.argmax(np.asarray(values, float)
                         + np.sqrt(c * np.log(max(t, 1)) / counts)))


def thompson_action(successes, failures, rng):
    s = np.asarray(successes, float)
    f = np.asarray(failures, float)
    return int(np.argmax(rng.beta(1.0 + s, 1.0 + f)))


def cumulative_regret(chosen_means, best_mean):
    return np.cumsum(best_mean - np.asarray(chosen_means, float))


def regret_growth(regret, frac=0.25):
    regret = np.asarray(regret, float)
    k = max(1, int(len(regret) * frac))
    return float((regret[-1] - regret[-k - 1]) / k)


# ------------------------------------------------------------ reward design

def potential_shaping(Phi, s, s2, gamma):
    return float(gamma * Phi[s2] - Phi[s])


def shaped_rewards(P, R, Phi, gamma):
    Phi = np.asarray(Phi, float)
    return R + gamma * (P @ Phi) - Phi[:, None]


def policies_agree(pi1, pi2, P, R, gamma, tol=1e-9):
    n_a = P.shape[1]
    v1 = policy_value_exact(P, R, _onehot(np.asarray(pi1, int), n_a), gamma)
    v2 = policy_value_exact(P, R, _onehot(np.asarray(pi2, int), n_a), gamma)
    return bool(np.abs(v1 - v2).max() < tol)


# --------------------------------------------------------------------- patch

for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
