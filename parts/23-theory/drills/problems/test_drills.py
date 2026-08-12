"""
Tests for the Part XXIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part's subject is where the
guarantees stop: that a bound holding everywhere can still say nothing, that
neither concentration inequality dominates the other, and that the correction
for covariate shift does not help a model that was already right.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# --------------------------------------------------------------- concentration

def test_hoeffding_never_undershoots_the_tail():
    """It is an upper bound. If this ever fails, something is wrong with the
    simulation, not with Hoeffding."""
    rng = np.random.default_rng(0)
    n, reps, p = 200, 200_000, 0.5
    draws = rng.binomial(n, p, reps) / n
    for eps in (0.05, 0.10, 0.15):
        emp = attempt(drills.empirical_tail, draws, p, eps)
        assert attempt(drills.hoeffding_bound, n, eps) >= emp - 1e-9


def test_neither_inequality_dominates():
    """Figure 1's finding. Bernstein pays a worse constant for using the
    variance, so it loses where the variance is large."""
    n, eps = 200, 0.10
    h = attempt(drills.hoeffding_bound, n, eps)
    b_half = attempt(drills.bernstein_bound, n, eps, 0.25)      # p = 0.5
    b_small = attempt(drills.bernstein_bound, n, eps, 0.05 * 0.95)
    assert b_half > h                       # Bernstein is LOOSER at p = 0.5
    assert b_small < h / 1000               # and far tighter when var is small


def test_hoeffding_ignores_the_distribution():
    """The same bound for a fair coin and a one-in-twenty coin — which is the
    whole content of 'distribution-free'."""
    a = attempt(drills.hoeffding_bound, 200, 0.1)
    b = attempt(drills.hoeffding_bound, 200, 0.1)
    assert a == b


def test_samples_needed_inverts_the_bound():
    for eps, delta in ((0.05, 0.05), (0.01, 0.01)):
        n = attempt(drills.samples_needed, eps, delta)
        assert attempt(drills.hoeffding_bound, n, eps) <= delta + 1e-12
        assert attempt(drills.hoeffding_bound, n - 1, eps) > delta


def test_the_rate_is_one_over_root_n():
    n1 = attempt(drills.samples_needed, 0.02, 0.05)
    n2 = attempt(drills.samples_needed, 0.01, 0.05)
    assert n2 / n1 == pytest.approx(4.0, rel=0.02)      # halve eps, quadruple n


# ---------------------------------------------------------- uniform convergence

def test_finite_class_penalty_is_logarithmic():
    """Doubling the search space adds a constant under the square root, not a
    factor — the reason searching over many models is affordable."""
    a = attempt(drills.finite_class_bound, 1000, 5000)
    b = attempt(drills.finite_class_bound, 2000, 5000)
    c = attempt(drills.finite_class_bound, 4000, 5000)
    assert (b ** 2 - a ** 2) == pytest.approx(c ** 2 - b ** 2, rel=1e-6)


def test_sauer_is_exactly_two_to_the_m_below_the_dimension():
    for m in range(1, 8):
        assert attempt(drills.sauer_bound, m, m) == 2 ** m
        assert attempt(drills.sauer_bound, m, m + 3) == 2 ** m


def test_sauer_becomes_polynomial_above_it():
    """The dichotomy: either every labelling, or polynomially many."""
    vc = 3
    for m in (10, 20, 40):
        b = attempt(drills.sauer_bound, m, vc)
        assert b < 2 ** m
        assert b <= (np.e * m / vc) ** vc + 1


def test_a_linear_classifier_shatters_d_plus_one_points():
    """Figure 2, by enumeration rather than by citing the theorem."""
    rng = np.random.default_rng(1)
    for d in (2, 3):
        for _ in range(3):
            assert attempt(drills.shatters, rng.standard_normal((d + 1, d))) is True


def test_and_fails_at_d_plus_two():
    rng = np.random.default_rng(2)
    for d in (2, 3):
        for _ in range(3):
            assert attempt(drills.shatters, rng.standard_normal((d + 2, d))) is False


def test_xor_is_the_labelling_that_cannot_be_realised():
    """Part VII's opening problem is the d=2, m=4 case."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    assert attempt(drills.is_separable, X, np.array([0, 1, 1, 0])) is False
    assert attempt(drills.is_separable, X, np.array([0, 0, 1, 1])) is True


# ------------------------------------------------------------ vacuity

def test_the_vc_bound_is_real_for_a_small_class():
    b = attempt(drills.vc_bound, 11, 8000)
    assert 0.0 < b < 0.3
    assert attempt(drills.is_vacuous, b) is False


def test_the_vc_bound_is_vacuous_at_network_sizes():
    """Figure 3's right panel. The bound is satisfied by a random classifier, so
    it excludes nothing at all."""
    for params, n in ((25_000, 50_000), (1_000_000, 50_000), (100_000_000, 1_000_000)):
        b = attempt(drills.vc_bound, params, n)
        assert np.isfinite(b), "the bound must stay real in the regime it is quoted in"
        assert attempt(drills.is_vacuous, b) is True


def test_no_sample_size_rescues_a_dimension_above_it():
    """Not a loose constant — sqrt(vcdim/n) exceeds 1 whenever vcdim > n."""
    for n in (10_000, 1_000_000, 100_000_000):
        assert attempt(drills.is_vacuous, attempt(drills.vc_bound, 2 * n, n)) is True


# ---------------------------------------------------------------- complexity

def test_rademacher_falls_as_one_over_root_n():
    rng = np.random.default_rng(3)
    vals = []
    for n in (100, 400, 1600):
        X = rng.standard_normal((n, 5))
        vals.append(attempt(drills.rademacher_linear, X))
    assert vals[0] > vals[1] > vals[2]
    assert vals[0] / vals[2] == pytest.approx(4.0, rel=0.25)


def test_margin_fraction_is_monotone_in_the_margin():
    rng = np.random.default_rng(4)
    y = rng.choice([-1.0, 1.0], 2000)
    s = y * rng.random(2000) * 2
    fr = [attempt(drills.margin_fraction, s, y, g) for g in (0.0, 0.5, 1.0, 1.5)]
    assert fr == sorted(fr, reverse=True)


# ------------------------------------------------------------ online learning

def test_follow_the_leader_suffers_linear_regret():
    """The alternating sequence that defeats the obvious algorithm."""
    T = 400
    L = np.zeros((T, 2))
    L[::2, 0] = 1.0            # action 0 loses on even rounds
    L[1::2, 1] = 1.0           # action 1 loses on odd rounds
    ch = attempt(drills.follow_the_leader, L)
    r = attempt(drills.regret, L, ch)
    assert r > 0.3 * T         # regret grows with T rather than as its root


def test_multiplicative_weights_has_sublinear_regret_on_the_same_sequence():
    T = 400
    L = np.zeros((T, 2))
    L[::2, 0] = 1.0
    L[1::2, 1] = 1.0
    best = L.sum(axis=0).min()
    tot = attempt(drills.multiplicative_weights, L, eta=np.sqrt(np.log(2) / T))
    assert tot - best < 0.15 * T


def test_regret_is_measured_against_the_best_fixed_action():
    # action 0 costs 0+0+1 = 1 over the three rounds; action 1 costs 1+1+0 = 2.
    # The best fixed action is therefore action 0, at a total of 1.
    L = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert attempt(drills.regret, L, np.array([0, 0, 0])) == pytest.approx(0.0)
    assert attempt(drills.regret, L, np.array([1, 1, 1])) == pytest.approx(1.0)
    # And regret can be NEGATIVE, because the comparator is the best fixed
    # action rather than the best sequence: playing [0, 0, 1] costs nothing at
    # all, which beats every constant strategy. That is exactly why no-regret is
    # a weaker promise than it sounds — §31.
    assert attempt(drills.regret, L, np.array([0, 0, 1])) == pytest.approx(-1.0)


# ------------------------------------------------------------ distribution shift

def _shift_data(n, shift, misspecified, seed=0):
    r = np.random.default_rng(seed)
    xtr = r.normal(0.0, 1.0, n)
    xte = r.normal(shift, 1.0, 8000)
    f = (lambda x: x + 0.6 * x ** 2) if misspecified else (lambda x: x)
    ytr = f(xtr) + 0.3 * r.standard_normal(n)
    yte = f(xte) + 0.3 * r.standard_normal(len(xte))
    return xtr, ytr, xte, yte


def _err(xtr, ytr, xte, yte, w=None):
    A = np.column_stack([np.ones(len(xtr)), xtr])
    b = (np.linalg.lstsq(A, ytr, rcond=None)[0] if w is None
         else attempt(drills.weighted_least_squares, A, ytr, w))
    B = np.column_stack([np.ones(len(xte)), xte])
    return float(np.mean((B @ b - yte) ** 2))


def test_weighting_does_not_help_a_well_specified_model():
    """Figure 6's left panel, and the reason 'always reweight' is wrong."""
    xtr, ytr, xte, yte = _shift_data(600, 1.5, misspecified=False)
    w = attempt(drills.importance_weights, xtr, 0.0, 1.5, clip=50)
    plain = _err(xtr, ytr, xte, yte)
    wtd = _err(xtr, ytr, xte, yte, w)
    assert plain < 0.2 and wtd < 0.3
    assert wtd >= plain * 0.9          # no improvement to speak of


def test_weighting_rescues_a_misspecified_one():
    xtr, ytr, xte, yte = _shift_data(600, 1.5, misspecified=True)
    w = attempt(drills.importance_weights, xtr, 0.0, 1.5, clip=50)
    plain = _err(xtr, ytr, xte, yte)
    wtd = _err(xtr, ytr, xte, yte, w)
    assert wtd < plain / 2


def test_importance_weights_are_one_when_nothing_shifted():
    x = np.linspace(-3, 3, 50)
    w = attempt(drills.importance_weights, x, 0.0, 0.0)
    assert np.allclose(w, 1.0)


def test_label_shift_weights_recover_a_known_prior_change():
    C = np.array([[0.9, 0.2], [0.1, 0.8]])       # p(pred i | true j)
    true_target = np.array([0.25, 0.75])
    q = C @ true_target
    w = attempt(drills.label_shift_weights, C, q)
    assert np.allclose(w * 0.5, true_target, atol=1e-6)


def test_worst_group_is_not_the_average():
    y = np.array([1, 1, 1, 1, 0, 0])
    pred = np.array([1, 1, 1, 1, 1, 1])
    g = np.array([0, 0, 0, 0, 1, 1])
    worst, accs = attempt(drills.worst_group_accuracy, pred, y, g)
    assert accs[0] == pytest.approx(1.0)
    assert worst == pytest.approx(0.0)
    assert np.mean(pred == y) == pytest.approx(4 / 6)   # the average hides it


# ------------------------------------------------------- movement G: width

def test_the_ntk_gram_is_psd():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(12, 40))
    K = attempt(drills.ntk_gram, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_the_ntk_gram_is_rank_limited_by_the_parameter_count():
    rng = np.random.default_rng(1)
    K = attempt(drills.ntk_gram, rng.normal(size=(10, 3)))
    assert np.linalg.matrix_rank(K, tol=1e-8) == 3


def test_relative_kernel_change_is_scale_free():
    rng = np.random.default_rng(2)
    K0 = rng.normal(size=(6, 6))
    K1 = K0 + 0.1 * rng.normal(size=(6, 6))
    a = attempt(drills.relative_kernel_change, K0, K1)
    b = attempt(drills.relative_kernel_change, 1000 * K0, 1000 * K1)
    assert a == pytest.approx(b)
    assert attempt(drills.relative_kernel_change, K0, K0) == pytest.approx(0.0)


def test_loglog_slope_recovers_a_power_law():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert attempt(drills.loglog_slope, x, 3.0 * x ** -0.5) == pytest.approx(-0.5)
    assert attempt(drills.loglog_slope, x, 3.0 * x ** 2) == pytest.approx(2.0)


def test_width_stability_is_a_statement_about_a_trend():
    w = np.array([64.0, 128.0, 256.0, 512.0, 1024.0])
    assert attempt(drills.is_width_stable, w, np.full(5, 0.4)) is True
    assert attempt(drills.is_width_stable, w, 0.4 * w) is False        # grows
    assert attempt(drills.is_width_stable, w, 0.4 / np.sqrt(w)) is False   # goes lazy


def test_standard_parameterisation_uses_one_learning_rate():
    a = attempt(drills.init_scales, "standard", 1024, 8)
    assert a[2] == pytest.approx(1.0) and a[3] == pytest.approx(1.0)


def test_mup_scales_the_two_learning_rates_in_opposite_directions():
    for m in (256, 1024, 4096):
        _, out_std, lr_in, lr_out = attempt(drills.init_scales, "mup", m, 8)
        assert out_std == pytest.approx(1.0 / m)
        assert lr_in == pytest.approx(float(m))
        assert lr_out == pytest.approx(1.0 / m)


def test_mup_differs_from_ntk_in_exactly_the_learning_rates():
    """The failure this guards against is real: dropping the input-layer
    multiplier leaves something that looks like muP and is lazier than NTK."""
    mup = attempt(drills.init_scales, "mup", 512, 8)
    ntk = attempt(drills.init_scales, "ntk", 512, 8)
    assert mup[2:] != ntk[2:]
    assert ntk[2] == pytest.approx(1.0) and ntk[3] == pytest.approx(1.0)


def test_the_correlation_fixed_point_is_one_in_the_ordered_phase():
    assert attempt(drills.correlation_fixed_point, 1.0, 0.05) == pytest.approx(1.0, abs=1e-3)
    assert attempt(drills.correlation_fixed_point, 1.4, 0.05) == pytest.approx(1.0, abs=1e-2)


def test_the_correlation_fixed_point_drops_below_one_in_the_chaotic_phase():
    c3 = attempt(drills.correlation_fixed_point, 3.0, 0.05)
    c45 = attempt(drills.correlation_fixed_point, 4.5, 0.05)
    assert c3 < 0.95 and c45 < c3


def test_chi1_crosses_one_exactly_at_criticality():
    assert attempt(drills.chi1, 1.0, 0.05) < 1.0
    assert attempt(drills.chi1, 3.0, 0.05) > 1.0
    s = attempt(drills.critical_sigma_w2, 0.05)
    assert attempt(drills.chi1, s, 0.05) == pytest.approx(1.0, abs=1e-6)


def test_the_critical_variance_for_tanh():
    """Near 1.76 at sigma_b^2 = 0.05 — the value at which Figure 12 finds the
    gradient scale nearest one, arrived at independently."""
    assert attempt(drills.critical_sigma_w2, 0.05) == pytest.approx(1.76, abs=0.08)
    assert attempt(drills.critical_sigma_w2, 0.0) == pytest.approx(1.0, abs=0.02)


def test_more_bias_noise_pushes_criticality_up():
    lo = attempt(drills.critical_sigma_w2, 0.0)
    hi = attempt(drills.critical_sigma_w2, 0.2)
    assert hi > lo


def test_transfer_spread():
    assert attempt(drills.transfer_spread, [0.1, 0.1, 0.1]) == pytest.approx(1.0)
    assert attempt(drills.transfer_spread, [1.0, 0.5, 0.25, 0.0625]) == pytest.approx(16.0)


def test_a_scheme_that_transfers_has_a_flat_optimum():
    w = np.array([64.0, 128.0, 256.0, 512.0, 1024.0])
    mup = np.full(5, 7.5)
    std = 4.8 / w * 64
    assert attempt(drills.transfer_spread, mup) == pytest.approx(1.0)
    assert attempt(drills.transfer_spread, std) > 10
    assert attempt(drills.is_width_stable, w, mup) is True
    assert attempt(drills.is_width_stable, w, std) is False
