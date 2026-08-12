"""
Tests for the Part XVIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several tests assert that a method FAILS, because those are the part's results:
the maximisation bias, the divergence of the deadly triad, and epsilon-greedy's
regret that does not stop growing.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def random_mdp(n_s=25, n_a=4, seed=0, sparsity=4):
    r = np.random.default_rng(seed)
    P = np.zeros((n_s, n_a, n_s))
    for s in range(n_s):
        for a in range(n_a):
            nxt = r.choice(n_s, sparsity, replace=False)
            P[s, a, nxt] = r.dirichlet(np.ones(sparsity))
    return P, r.normal(0, 1, (n_s, n_a))


# ---------------------------------------------------------------- exact MDPs

def test_value_iteration_reaches_the_bellman_fixed_point():
    P, R = random_mdp(seed=0)
    V, n = attempt(drills.value_iteration, P, R, 0.9)
    assert np.abs(attempt(drills.bellman_backup, P, R, V, 0.9) - V).max() < 1e-9
    assert n > 1


def test_value_iteration_is_independent_of_where_it_starts():
    """The contraction gives a UNIQUE fixed point, reached from anywhere."""
    P, R = random_mdp(seed=1)
    V1, _ = attempt(drills.value_iteration, P, R, 0.9)

    def from_start(V0):
        V = V0.copy()
        for _ in range(4000):
            Vn = attempt(drills.bellman_backup, P, R, V, 0.9)
            if np.abs(Vn - V).max() < 1e-12:
                return Vn
            V = Vn
        return V
    assert np.abs(from_start(np.full(25, 500.0)) - V1).max() < 1e-6
    assert np.abs(from_start(np.full(25, -500.0)) - V1).max() < 1e-6


def test_policy_iteration_agrees_with_value_iteration_and_terminates():
    P, R = random_mdp(seed=2)
    V_vi, n_vi = attempt(drills.value_iteration, P, R, 0.9)
    pi, V_pi, n_pi = attempt(drills.policy_iteration, P, R, 0.9)
    assert np.abs(V_pi - V_vi).max() < 1e-6
    assert n_pi < 20, f"policy iteration should terminate quickly, took {n_pi}"
    assert n_pi < n_vi, "and in far fewer iterations than value iteration"


def test_exact_evaluation_matches_iterating():
    P, R = random_mdp(seed=3)
    r = np.random.default_rng(0)
    pi = r.dirichlet(np.ones(4), 25)
    V = attempt(drills.policy_value_exact, P, R, pi, 0.9)
    Ppi = np.einsum("sa,san->sn", pi, P)
    Rpi = (pi * R).sum(1)
    U = np.zeros(25)
    for _ in range(500):
        U = Rpi + 0.9 * Ppi @ U
    assert np.abs(U - V).max() < 1e-8


def test_the_contraction_rate_is_exactly_gamma():
    """Section 12, as an assertion."""
    P, R = random_mdp(seed=4)
    for g in (0.5, 0.8, 0.95):
        rate = attempt(drills.contraction_rate, P, R, g, n_sweeps=400)
        assert rate == pytest.approx(g, abs=2e-3), f"gamma {g}: measured {rate:.5f}"


def test_horizon_and_sweep_count():
    assert attempt(drills.effective_horizon, 0.99) == pytest.approx(100.0)
    assert attempt(drills.sweeps_for_accuracy, 0.9, 1e-6) == 132
    assert attempt(drills.sweeps_for_accuracy, 0.99, 1e-6) == 1375
    assert attempt(drills.sweeps_for_accuracy, 0.99, 1e-6) > \
        10 * attempt(drills.sweeps_for_accuracy, 0.9, 1e-6), \
        "a long horizon is expensive in a way you can compute in advance"


# ------------------------------------------------------------ sampled updates

def test_returns_and_n_step_interpolate():
    rew = np.array([1.0, 2.0, 3.0])
    G = attempt(drills.mc_returns, rew, 0.5)
    assert G[2] == pytest.approx(3.0)
    assert G[1] == pytest.approx(2.0 + 0.5 * 3.0)
    assert G[0] == pytest.approx(1.0 + 0.5 * (2.0 + 0.5 * 3.0))
    assert attempt(drills.n_step_return, rew, 0.0, 0.5, 3) == pytest.approx(G[0])
    assert attempt(drills.n_step_return, rew, 7.0, 0.5, 1) == pytest.approx(1.0 + 0.5 * 7.0)


def test_td_converges_to_the_exact_values():
    P, R = random_mdp(n_s=8, n_a=1, seed=5, sparsity=3)
    gamma = 0.9
    Vtrue = attempt(drills.policy_value_exact, P, R, np.ones((8, 1)), gamma)
    rng = np.random.default_rng(0)
    V = np.zeros(8)
    s = 0
    for t in range(400000):
        rew = R[s, 0]
        s2 = int(rng.choice(8, p=P[s, 0]))
        V = attempt(drills.td0_update, V, s, rew, s2, 0.02 / (1 + t / 20000), gamma)
        s = s2
    assert np.abs(V - Vtrue).max() < 0.15, f"max error {np.abs(V - Vtrue).max():.3f}"


def test_q_learning_and_sarsa_targets_differ_by_one_symbol():
    Qn = np.array([1.0, 5.0, 2.0])
    assert attempt(drills.q_learning_target, 1.0, Qn, 0.9) == pytest.approx(1 + 0.9 * 5.0)
    assert attempt(drills.sarsa_target, 1.0, Qn, 0, 0.9) == pytest.approx(1 + 0.9 * 1.0)
    assert attempt(drills.sarsa_target, 1.0, Qn, 1, 0.9) == \
        attempt(drills.q_learning_target, 1.0, Qn, 0.9), \
        "they agree exactly when the greedy action is the one taken"


def test_the_max_estimator_is_biased_upward():
    """Section 19: every true value is zero and the estimate is not."""
    prev = None
    for K in (2, 8, 32):
        b = attempt(drills.max_estimator_bias, K, 8, 4000, seed=0)
        assert b > 0.05, f"{K} actions: bias {b:.4f}"
        if prev is not None:
            assert b > prev, "the bias must grow with the number of actions"
        prev = b


def test_the_double_estimator_removes_the_bias():
    rng = np.random.default_rng(0)
    single, double = [], []
    for _ in range(4000):
        a = rng.normal(0, 1, (8, 8)).mean(1)
        b = rng.normal(0, 1, (8, 8)).mean(1)
        single.append(a.max())
        double.append(attempt(drills.double_q_target, 0.0, a, b, 1.0))
    assert np.mean(single) > 0.3
    assert abs(np.mean(double)) < 0.05, f"double estimator {np.mean(double):+.4f}"


# --------------------------------------------------------- policy gradients

def test_the_score_function_has_zero_mean():
    """The three-line identity that makes a baseline free."""
    theta = np.array([0.4, -1.2, 0.9, 0.1])
    pi = attempt(drills.softmax_policy, theta)
    assert pi.sum() == pytest.approx(1.0)
    total = sum(pi[a] * attempt(drills.score_function, theta, a) for a in range(4))
    assert np.abs(total).max() < 1e-12


def test_softmax_is_stable():
    big = np.array([900.0, 901.0, 899.0])
    p = attempt(drills.softmax_policy, big)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


def test_reinforce_is_unbiased_with_and_without_a_baseline():
    theta = np.array([0.3, -0.5, 0.8])
    rewards = np.array([5.2, 4.8, 5.5])
    exact = attempt(drills.exact_policy_gradient, theta, rewards)
    for baseline in (None, "mean"):
        ests = np.array([attempt(drills.reinforce_estimate, theta, rewards, 2000,
                                 baseline=baseline, seed=s, noise=0.5)
                         for s in range(60)])
        assert np.abs(ests.mean(0) - exact).max() < 0.05, \
            f"baseline={baseline} is biased"


def test_a_baseline_cuts_the_variance_substantially():
    """Section 27, as an assertion. The rewards sit near 5, so most of the
    signal without a baseline is a constant that carries no information."""
    theta = np.array([0.3, -0.5, 0.8])
    rewards = np.array([5.2, 4.8, 5.5])
    def var(baseline):
        ests = np.array([attempt(drills.reinforce_estimate, theta, rewards, 16,
                                 baseline=baseline, seed=s, noise=0.5)
                         for s in range(400)])
        return ests.var(0).sum()
    assert var(None) > 10 * var("mean"), \
        f"no baseline {var(None):.4f} against mean baseline {var('mean'):.4f}"


def test_gae_interpolates_between_td_and_monte_carlo():
    rew = np.array([1.0, 0.0, 2.0])
    V = np.array([0.5, 0.4, 0.3, 0.2])
    g = 0.9
    a0 = attempt(drills.gae, rew, V, g, 0.0)
    a1 = attempt(drills.gae, rew, V, g, 1.0)
    one_step = attempt(drills.advantage_from_values, rew, V, g)
    assert np.allclose(a0, one_step), "lambda=0 is the one-step advantage"
    assert a1[0] == pytest.approx(sum(g ** k * one_step[k] for k in range(3)))


def test_ppo_clipping_stops_rewarding_a_large_step():
    A_pos, A_neg = 1.0, -1.0
    inside = attempt(drills.ppo_objective, np.array([1.0]), np.array([A_pos]))[0]
    far = attempt(drills.ppo_objective, np.array([5.0]), np.array([A_pos]))[0]
    assert far == pytest.approx(1.2 * A_pos), "a positive advantage is capped at 1+eps"
    assert far > inside
    farther = attempt(drills.ppo_objective, np.array([50.0]), np.array([A_pos]))[0]
    assert farther == pytest.approx(far), \
        "and going further buys nothing at all — the gradient vanishes"
    neg = attempt(drills.ppo_objective, np.array([0.01]), np.array([A_neg]))[0]
    assert neg == pytest.approx(0.8 * A_neg)


# --------------------------------------------------------------- exploration

def test_ucb_tries_everything_before_committing():
    counts = np.array([0.0, 5.0, 5.0])
    assert attempt(drills.ucb_action, counts, np.array([0.0, 1.0, 1.0]), 10) == 0
    counts = np.array([1.0, 100.0])
    a = attempt(drills.ucb_action, counts, np.array([0.5, 0.55]), 200)
    assert a == 0, "the under-tried arm gets the larger bonus"


def test_epsilon_greedy_regret_does_not_stop_growing():
    """Section 35: the property that separates it from UCB and Thompson."""
    K, T = 5, 30000
    rng = np.random.default_rng(0)
    means = np.array([0.2, 0.4, 0.5, 0.55, 0.8])
    best = means.max()

    def run(policy):
        r = np.random.default_rng(1)
        n = np.zeros(K); s = np.zeros(K); chosen = np.empty(T)
        for t in range(T):
            if policy == "eps":
                a = int(r.integers(K)) if r.random() < 0.1 else \
                    int(np.argmax(np.where(n > 0, s / np.maximum(n, 1), 0.0)))
            else:
                a = attempt(drills.ucb_action, n, np.where(n > 0, s / np.maximum(n, 1), 0.0), t + 1)
            s[a] += float(r.random() < means[a]); n[a] += 1
            chosen[t] = means[a]
        return attempt(drills.cumulative_regret, chosen, best)

    g_eps = attempt(drills.regret_growth, run("eps"))
    g_ucb = attempt(drills.regret_growth, run("ucb"))
    assert g_eps > 0.01, f"epsilon-greedy keeps paying: {g_eps:.4f} per step"
    assert g_ucb < g_eps / 3, f"UCB decelerates: {g_ucb:.4f} against {g_eps:.4f}"


# ------------------------------------------------------------ reward design

def test_potential_shaping_leaves_the_optimal_policy_unchanged():
    """The one shaping form with a guarantee, checked rather than trusted."""
    P, R = random_mdp(seed=7)
    gamma = 0.9
    rng = np.random.default_rng(0)
    Phi = rng.normal(0, 5.0, 25)          # an arbitrary, large potential
    Rs = attempt(drills.shaped_rewards, P, R, Phi, gamma)
    V1, _ = attempt(drills.value_iteration, P, R, gamma)
    V2, _ = attempt(drills.value_iteration, P, Rs, gamma)
    pi1 = attempt(drills.greedy_policy, P, R, V1, gamma)
    pi2 = attempt(drills.greedy_policy, P, Rs, V2, gamma)
    assert attempt(drills.policies_agree, pi1, pi2, P, R, gamma, 1e-8), \
        "potential-based shaping must not change what is optimal"
    assert np.abs(V2 - (V1 - Phi)).max() < 1e-6, \
        "and the values shift by exactly the potential"


def test_a_non_potential_bonus_can_change_the_optimal_policy():
    """The contrast that makes the guarantee worth having."""
    P, R = random_mdp(seed=8)
    gamma = 0.9
    rng = np.random.default_rng(1)
    bonus = rng.normal(0, 2.0, (25, 4))    # depends on the ACTION: not a potential
    V1, _ = attempt(drills.value_iteration, P, R, gamma)
    V2, _ = attempt(drills.value_iteration, P, R + bonus, gamma)
    pi1 = attempt(drills.greedy_policy, P, R, V1, gamma)
    pi2 = attempt(drills.greedy_policy, P, R + bonus, V2, gamma)
    assert not attempt(drills.policies_agree, pi1, pi2, P, R, gamma, 1e-8), \
        "an arbitrary bonus is not safe, which is why the potential form matters"
