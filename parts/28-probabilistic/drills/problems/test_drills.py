"""
Tests for the Part XXVIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Most of these compare an approximation against the exact answer it approximates,
which this subject uniquely allows. Several assert a NEGATIVE result, because the
part's content includes what these methods provably get wrong: that mean field is
overconfident by a computable factor, that an unbiased estimator can be useless,
and that a clean convergence diagnostic is not evidence of a correct answer.
"""
import itertools

import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def chain_model():
    """A -> B -> C, all binary."""
    parents = [(), (0,), (1,)]
    tables = [{(): np.array([0.6, 0.4])},
              {(0,): np.array([0.8, 0.2]), (1,): np.array([0.3, 0.7])},
              {(0,): np.array([0.9, 0.1]), (1,): np.array([0.2, 0.8])}]
    return parents, tables


def collider_model():
    """A -> C <- B, all binary, A and B independent."""
    parents = [(), (), (0, 1)]
    tables = [{(): np.array([0.5, 0.5])},
              {(): np.array([0.5, 0.5])},
              {(0, 0): np.array([0.95, 0.05]), (0, 1): np.array([0.4, 0.6]),
               (1, 0): np.array([0.4, 0.6]), (1, 1): np.array([0.05, 0.95])}]
    return parents, tables


# ------------------------------------------------------------------ structure

def test_a_joint_from_a_graph_sums_to_one():
    p, t = chain_model()
    J = attempt(drills.joint_from_graph, p, t, 2)
    assert J.shape == (2, 2, 2)
    assert J.sum() == pytest.approx(1.0)


def test_the_chain_makes_its_ends_dependent():
    p, t = chain_model()
    J = attempt(drills.joint_from_graph, p, t, 2)
    AC = attempt(drills.marginal, J, (0, 2))
    A = AC.sum(1); C = AC.sum(0)
    assert not np.allclose(AC, np.outer(A, C), atol=1e-3)


def test_conditioning_on_the_middle_of_a_chain_separates_the_ends():
    p, t = chain_model()
    J = attempt(drills.joint_from_graph, p, t, 2)
    for b in (0, 1):
        s = J[:, b, :]
        s = s / s.sum()
        assert np.allclose(s, np.outer(s.sum(1), s.sum(0)), atol=1e-9)


def test_the_collider_leaves_its_parents_independent():
    p, t = collider_model()
    J = attempt(drills.joint_from_graph, p, t, 2)
    AB = attempt(drills.marginal, J, (0, 1))
    assert np.allclose(AB, np.outer(AB.sum(1), AB.sum(0)), atol=1e-12)


def test_conditioning_on_a_collider_makes_them_dependent():
    """The one that inverts. Independent causes become dependent once their
    common effect is observed, which is why adjusting for more variables is not
    safer."""
    p, t = collider_model()
    J = attempt(drills.joint_from_graph, p, t, 2)
    s = J[:, :, 1] / J[:, :, 1].sum()
    assert not np.allclose(s, np.outer(s.sum(1), s.sum(0)), atol=1e-3)


def test_d_separation_agrees_with_the_chain():
    p, _ = chain_model()
    assert not attempt(drills.d_separated, p, 0, 2, [])
    assert attempt(drills.d_separated, p, 0, 2, [1])


def test_d_separation_agrees_with_the_collider():
    p, _ = collider_model()
    assert attempt(drills.d_separated, p, 0, 1, [])
    assert not attempt(drills.d_separated, p, 0, 1, [2])


def test_conditioning_on_a_colliders_descendant_also_opens_the_path():
    parents = [(), (), (0, 1), (2,)]          # A -> C <- B,  C -> D
    assert attempt(drills.d_separated, parents, 0, 1, [])
    assert not attempt(drills.d_separated, parents, 0, 1, [3])


def test_the_measured_correlation_matches_what_the_graph_says():
    r = np.random.default_rng(0)
    n = 60000
    A = r.standard_normal(n); B = r.standard_normal(n)
    C = A + B + 0.4 * r.standard_normal(n)
    assert abs(np.corrcoef(A, B)[0, 1]) < 0.02
    assert attempt(drills.partial_corr, A, B, C) < -0.4


# -------------------------------------------------------------------- chains

def hmm():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    return pi, A, B


def test_forward_backward_matches_enumeration():
    pi, A, B = hmm()
    obs = [0, 1, 1, 0, 1]
    g1, ll1 = attempt(drills.forward_backward, pi, A, B, obs)
    g2, ll2 = attempt(drills.enumerate_posterior, pi, A, B, obs)
    assert np.abs(g1 - g2).max() < 1e-12
    assert ll1 == pytest.approx(ll2, abs=1e-12)


def test_the_posterior_is_a_distribution_at_every_step():
    pi, A, B = hmm()
    g, _ = attempt(drills.forward_backward, pi, A, B, [0, 1, 0, 0, 1, 1])
    assert np.allclose(g.sum(1), 1.0)


def test_scaling_survives_a_long_sequence():
    """Unscaled, the forward recursion is a product of hundreds of
    probabilities and underflows to zero."""
    pi, A, B = hmm()
    r = np.random.default_rng(1)
    obs = r.integers(0, 2, 600)
    g, ll = attempt(drills.forward_backward, pi, A, B, obs)
    assert np.isfinite(ll) and ll < 0
    assert np.all(np.isfinite(g))


def test_the_best_path_is_not_the_sequence_of_best_states():
    """Found by search, because it is harder to construct than it looks. The
    per-state answer here is not merely worse than the best path — it uses a
    transition the model gives probability zero, so it is a sequence the model
    says cannot happen."""
    pi = np.array([0.091, 0.748, 0.160])
    A = np.array([[0.463, 0.091, 0.446],
                  [0.592, 0.406, 0.002],
                  [0.000, 0.155, 0.845]])
    A = A / A.sum(1, keepdims=True)
    B = np.array([[0.613, 0.387], [0.709, 0.291], [0.150, 0.850]])
    obs = [0, 1, 1, 0]
    g, _ = attempt(drills.forward_backward, pi, A, B, obs)
    path, _ = attempt(drills.viterbi, pi, A, B, obs)
    marg = g.argmax(1)
    assert not np.array_equal(path, marg)
    assert all(A[path[t], path[t + 1]] > 0 for t in range(len(obs) - 1))
    assert any(A[marg[t], marg[t + 1]] == 0 for t in range(len(obs) - 1))


def test_the_kalman_filter_is_the_posterior():
    r = np.random.default_rng(3)
    truth, R, P0 = 1.7, 0.6, 1e6
    y = truth + np.sqrt(R) * r.standard_normal(50)
    ms, Ps = attempt(drills.kalman_scalar, y, 1.0, 1.0, 0.0, R, 0.0, P0)
    n = np.arange(1, 51)
    prec = 1 / P0 + n / R
    assert np.abs(Ps - 1 / prec).max() < 1e-9
    assert np.abs(ms - (np.cumsum(y) / R) / prec).max() < 1e-9


def test_the_filters_uncertainty_shrinks_as_one_over_n():
    r = np.random.default_rng(4)
    y = r.standard_normal(200)
    _, Ps = attempt(drills.kalman_scalar, y, 1.0, 1.0, 0.0, 1.0, 0.0, 1e6)
    assert Ps[99] / Ps[199] == pytest.approx(2.0, rel=0.02)


# ------------------------------------------------------------------------ EM

def test_em_never_decreases_the_likelihood():
    r = np.random.default_rng(5)
    x = np.concatenate([r.normal(0, 1, 400), r.normal(4, 1, 400)])
    for seed in range(12):
        lls, *_ = attempt(drills.em_gmm, x, 2, seed)
        assert np.all(np.diff(lls) > -1e-8)


def test_em_recovers_well_separated_components_from_a_good_start():
    r = np.random.default_rng(6)
    x = np.concatenate([r.normal(0, 1, 800), r.normal(6, 1, 800)])
    best = None
    for seed in range(20):
        lls, mu, *_ = attempt(drills.em_gmm, x, 2, seed)
        if best is None or lls[-1] > best[0]:
            best = (lls[-1], np.sort(mu))
    assert np.abs(best[1] - np.array([0.0, 6.0])).max() < 0.3


def test_some_restarts_miss_them_anyway():
    """The measurement of Figure 3: separated components make distinct basins,
    and a restart that lands both means in one cluster cannot leave."""
    r = np.random.default_rng(7)
    sep = 6.0
    x = np.concatenate([r.normal(0, 1, 800), r.normal(sep, 1, 800)])
    miss = 0
    for seed in range(60):
        _, mu, *_ = attempt(drills.em_gmm, x, 2, seed)
        if np.abs(np.sort(mu) - np.array([0.0, sep])).max() > sep / 4:
            miss += 1
    assert miss > 0


# ---------------------------------------------------------------- variational

def test_mean_field_underestimates_the_variance_by_a_known_factor():
    """Exactly sqrt(1 - rho^2). Not approximately, and not because the optimiser
    stopped early."""
    for rho in (0.0, 0.3, 0.6, 0.9, 0.95):
        sd = attempt(drills.meanfield_gaussian_sd, rho)
        assert sd[0] == pytest.approx(np.sqrt(1 - rho ** 2), abs=1e-12)
        assert sd[1] == pytest.approx(np.sqrt(1 - rho ** 2), abs=1e-12)


def test_maximising_the_elbo_finds_the_same_answer():
    """Which is what makes the previous test a statement about the objective
    rather than about the optimiser."""
    rho = 0.9
    S = np.array([[1.0, rho], [rho, 1.0]])
    best = attempt(drills.meanfield_gaussian_sd, rho)[0]
    here = attempt(drills.elbo_gaussian, np.zeros(2), np.log([best, best]), S)
    for delta in (-0.3, -0.1, 0.1, 0.3):
        other = attempt(drills.elbo_gaussian, np.zeros(2),
                        np.log([best, best]) + delta, S)
        assert other < here


def test_the_elbo_is_a_lower_bound_on_the_evidence():
    rho = 0.7
    S = np.array([[1.0, rho], [rho, 1.0]])
    for sd in (0.2, 0.5, np.sqrt(1 - rho ** 2), 1.0, 2.0):
        assert attempt(drills.elbo_gaussian, np.zeros(2), np.log([sd, sd]), S) <= 1e-9


# --------------------------------------------------------------- monte carlo

def test_importance_weights_are_normalised():
    r = np.random.default_rng(8)
    s = r.standard_normal(500)
    w = attempt(drills.importance_weights,
                lambda x: -0.5 * x ** 2, lambda x: -0.5 * (x - 0.3) ** 2, s)
    assert w.sum() == pytest.approx(1.0)


def test_the_effective_sample_size_collapses_with_dimension():
    """Unbiased the whole way down, which is why unbiasedness is not the
    property to check."""
    out = []
    for d in (1, 5, 20, 50):
        r = np.random.default_rng(9)
        s = r.standard_normal((4000, d)) + 0.7
        logp = lambda x: -0.5 * np.sum(x ** 2)
        logq = lambda x: -0.5 * np.sum((x - 0.7) ** 2)
        w = attempt(drills.importance_weights, logp, logq, list(s))
        out.append(attempt(drills.ess_weights, w))
    assert out[0] > out[1] > out[2] > out[3]
    assert out[-1] < 0.01 * out[0]


def test_metropolis_recovers_a_gaussian():
    logp = lambda x: -0.5 * float(x @ x)
    s, acc = attempt(drills.metropolis, logp, np.zeros(1), 2.0, 40000)
    assert 0.2 < acc < 0.7
    assert abs(s.mean()) < 0.05
    assert s.var() == pytest.approx(1.0, rel=0.1)


def test_leapfrog_conserves_energy():
    """The check that catches a sign error. A wrong sign still produces samples."""
    logp = lambda x: -0.5 * float(x @ x)
    gradp = lambda x: -x
    x = np.array([1.3]); p = np.array([-0.4])
    H0 = -logp(x) + 0.5 * float(p @ p)
    xn, pn = attempt(drills.leapfrog, gradp, x, p, 0.05, 100)
    H1 = -logp(xn) + 0.5 * float(pn @ pn)
    assert abs(H1 - H0) < 1e-3


def test_hmc_beats_metropolis_on_an_ill_conditioned_target():
    sd = np.array([1.0, 0.05])
    logp = lambda z: -0.5 * float(np.sum((z / sd) ** 2))
    gradp = lambda z: -z / sd ** 2
    s_rw, _ = attempt(drills.metropolis, logp, np.zeros(2), 2.4 * sd.min(), 20000)
    s_h, acc = attempt(drills.hmc, logp, gradp, np.zeros(2), 0.7 * sd.min(), 60, 1000)
    assert acc > 0.6
    assert attempt(drills.ess_chain, s_h[:, 0]) / 1000 > \
           attempt(drills.ess_chain, s_rw[:, 0]) / 20000


def test_effective_sample_size_is_far_below_the_draw_count():
    logp = lambda x: -0.5 * float(x @ x)
    s, _ = attempt(drills.metropolis, logp, np.zeros(1), 0.15, 20000)
    assert attempt(drills.ess_chain, s[:, 0]) < 0.05 * len(s)


def test_rhat_is_one_for_chains_that_agree():
    r = np.random.default_rng(10)
    assert attempt(drills.rhat, r.standard_normal((4, 5000))) == pytest.approx(1.0, abs=0.02)


def test_rhat_passes_on_chains_that_all_missed_the_same_mode():
    """R-hat tests whether chains agree. Chains that started together and never
    left agree perfectly, and half the distribution leaves no trace."""
    r = np.random.default_rng(11)
    trapped = r.standard_normal((4, 5000)) + 9.0        # every chain in one mode
    assert attempt(drills.rhat, trapped) < 1.01
    dispersed = np.vstack([r.standard_normal((2, 5000)) + 9.0,
                           r.standard_normal((2, 5000)) - 9.0])
    assert attempt(drills.rhat, dispersed) > 2.0        # the correct answer looks broken


# ------------------------------------------------------------ continuous time

def test_the_simulator_matches_the_exact_marginal():
    x0 = np.full(40000, 2.0)
    f = lambda x, t: -x
    g = lambda t: np.sqrt(2.0)
    x = attempt(drills.euler_maruyama, f, g, x0, 1.0, 400)
    m, v = attempt(drills.ou_marginal, 2.0, 1.0)
    assert x.mean() == pytest.approx(m, abs=0.03)
    assert x.var() == pytest.approx(v, rel=0.05)


def test_the_noise_scales_as_the_square_root_of_the_step():
    """Get this exponent wrong and the process looks plausible with the wrong
    variance."""
    f = lambda x, t: 0.0 * x
    g = lambda t: 1.0
    vs = [attempt(drills.euler_maruyama, f, g, np.zeros(40000), T, 200).var()
          for T in (0.25, 1.0, 4.0)]
    assert vs[1] / vs[0] == pytest.approx(4.0, rel=0.08)
    assert vs[2] / vs[1] == pytest.approx(4.0, rel=0.08)


# ------------------------------------------------------------------ transport

def test_wasserstein_grows_linearly_where_kl_becomes_infinite():
    base = np.linspace(0, 1, 2000)
    for s in (1.5, 2.5, 3.5):
        assert attempt(drills.wasserstein1, base, base + s) == pytest.approx(s, abs=1e-9)


def test_kl_between_gaussians_is_quadratic_in_the_separation():
    a = attempt(drills.kl_gaussian, 0.0, 1.0, 1.0, 1.0)
    b = attempt(drills.kl_gaussian, 0.0, 1.0, 2.0, 1.0)
    assert b / a == pytest.approx(4.0, rel=1e-9)


def test_sinkhorn_returns_a_coupling_with_the_right_marginals():
    n = 20
    a = np.full(n, 1.0 / n); b = np.full(n, 1.0 / n)
    x = np.linspace(0, 1, n); y = np.linspace(0.5, 1.5, n)
    C = (x[:, None] - y[None, :]) ** 2
    P = attempt(drills.sinkhorn, a, b, C, 0.01, 2000)
    assert np.allclose(P.sum(1), a, atol=1e-6)
    assert np.allclose(P.sum(0), b, atol=1e-6)


def test_less_regularisation_sharpens_the_coupling():
    n = 20
    a = np.full(n, 1.0 / n); b = np.full(n, 1.0 / n)
    x = np.linspace(0, 1, n)
    C = (x[:, None] - x[None, :]) ** 2
    ent = []
    for eps in (0.5, 0.05, 0.005):
        P = attempt(drills.sinkhorn, a, b, C, eps, 3000)
        ent.append(float(-(P * np.log(P + 1e-300)).sum()))
    assert ent[0] > ent[1] > ent[2]
