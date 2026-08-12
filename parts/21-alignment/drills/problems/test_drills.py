"""
Tests for the Part XXI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part's subject is what evidence
does NOT establish: that a clean evaluation leaves a large failure rate open,
that a probe direction can be removed at no cost, that a component essential to
a computation can show a patching effect of exactly zero, and that the three
standard fairness criteria cannot all hold at once.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------ what evidence you have

def test_zero_failure_bound_matches_the_exact_expression():
    for n in (10, 100, 1021, 30000):
        got = attempt(drills.zero_failure_bound, n)
        assert got == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)


def test_a_thousand_clean_tests_leave_a_third_of_a_percent_open():
    """The number from Figure 5, and the reason 'we tested it' is weak."""
    b = attempt(drills.zero_failure_bound, 1021)
    assert 0.0029 < b < 0.0030
    # which at a million uses is thousands of failures
    assert b * 1_000_000 > 2900


def test_rule_of_three_approximates_the_exact_bound():
    for n in (100, 1000, 10000):
        exact = attempt(drills.zero_failure_bound, n)
        approx = attempt(drills.rule_of_three, n)
        assert approx == pytest.approx(exact, rel=0.02)


def test_trials_needed_is_about_three_over_p():
    for p in (1e-2, 1e-3, 1e-4):
        n = attempt(drills.trials_needed, p)
        assert n == pytest.approx(3 / p, rel=0.02)
    assert attempt(drills.trials_needed, 1e-3) == 2995


def test_trials_needed_and_attack_success_are_inverses():
    p = 3e-4
    n = attempt(drills.trials_needed, p, 0.95)
    assert attempt(drills.attack_success, p, n) >= 0.95
    assert attempt(drills.attack_success, p, n - 1) < 0.95


def test_attack_success_rises_with_attempts():
    p = 0.02
    vals = [attempt(drills.attack_success, p, n) for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(p)
    assert vals[-1] > 0.99


def test_binomial_interval_contains_the_estimate_and_agrees_at_zero():
    lo, hi = attempt(drills.binomial_interval, 7, 100)
    assert 0.0 <= lo <= 0.07 <= hi <= 1.0
    lo0, hi0 = attempt(drills.binomial_interval, 0, 500)
    assert lo0 == pytest.approx(0.0, abs=1e-9)
    assert hi0 == pytest.approx(attempt(drills.zero_failure_bound, 500), rel=0.05)


# ------------------------------------------------------------- superposition

def test_orthogonal_directions_are_bounded_by_dimension():
    assert attempt(drills.max_orthogonal, 5) == 5
    assert attempt(drills.max_orthogonal, 768) == 768


def test_features_kept_counts_surviving_columns():
    W = np.zeros((5, 20))
    W[:, :5] = np.eye(5)
    W[:, 5:] = 0.01 * np.random.default_rng(0).standard_normal((5, 15))
    assert attempt(drills.features_kept, W) == 5


def test_overlap_is_zero_for_orthogonal_and_positive_for_superposed():
    ortho = np.eye(5)
    assert attempt(drills.mean_overlap, ortho) == pytest.approx(0.0, abs=1e-9)
    r = np.random.default_rng(0)
    packed = r.standard_normal((5, 20))
    assert attempt(drills.mean_overlap, packed) > 0.1


def test_more_features_than_dimensions_forces_overlap():
    """The claim §3 rests on: 20 directions in 5 dimensions cannot be
    orthogonal, so superposition necessarily costs interference."""
    r = np.random.default_rng(1)
    for trial in range(5):
        W = r.standard_normal((5, 20))
        assert attempt(drills.mean_overlap, W) > 0


# ------------------------------------------------------- dictionary recovery

def test_best_cosine_is_one_when_the_dictionary_contains_the_truth():
    r = np.random.default_rng(0)
    T = r.standard_normal((6, 4))
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    got = attempt(drills.best_cosine, T, T)
    assert np.allclose(got, 1.0, atol=1e-6)


def test_recovery_ignores_sign():
    """Feature directions are unsigned: -v represents the same direction."""
    r = np.random.default_rng(0)
    T = r.standard_normal((6, 4))
    assert attempt(drills.recovered_count, T, -T) == 6


def test_a_narrow_dictionary_cannot_recover_everything():
    """PCA's ceiling in Figure 1: 5 components cannot cover 20 features."""
    r = np.random.default_rng(2)
    T = r.standard_normal((20, 5))
    found = r.standard_normal((5, 5))
    assert attempt(drills.recovered_count, T, found) < 20


# ---------------------------------------------------- probing against ablation

def test_ablation_removes_the_direction_and_nothing_else():
    r = np.random.default_rng(0)
    H = r.standard_normal((200, 16))
    d = r.standard_normal(16)
    Hp = attempt(drills.ablate_direction, H, d)
    assert np.allclose(Hp @ (d / np.linalg.norm(d)), 0, atol=1e-8)
    # the component along any orthogonal direction is untouched
    e = r.standard_normal(16)
    e = e - (e @ d) / (d @ d) * d
    e = e / np.linalg.norm(e)
    assert np.allclose(Hp @ e, H @ e, atol=1e-8)


def test_ablation_is_idempotent():
    r = np.random.default_rng(1)
    H = r.standard_normal((50, 8))
    d = r.standard_normal(8)
    once = attempt(drills.ablate_direction, H, d)
    twice = attempt(drills.ablate_direction, once, d)
    assert np.allclose(once, twice, atol=1e-9)


def test_causal_effect_arithmetic_on_figure_2s_reported_numbers():
    """This checks `causal_effect`'s arithmetic against Figure 2's reported
    numbers. It does NOT retrain the model, so it cannot catch the model or
    the experiment design being wrong — only that the subtraction is done
    right on whatever numbers are fed in. (This is exactly the gap that let
    a previous version of this test go on passing after the figure's central
    claim stopped reproducing: it hardcoded stale numbers and never ran the
    model to check them. The only way to verify the claim itself is to
    regenerate the figure: `python3 train_figures.py` from
    parts/21-alignment/, which prints every intermediate number.)

    Figure 2: the unused direction costs a negligible +0.001 (control-adjusted,
    not positive, once the random-direction drop of -0.003 is subtracted); the
    direction the network's own output layer actually reads for the task costs
    -0.249, control-adjusted -0.246."""
    raw, adj = attempt(drills.causal_effect, 0.753, 0.754, 0.750)
    assert raw == pytest.approx(-0.001, abs=1e-9)
    assert adj <= 0
    raw_u, adj_u = attempt(drills.causal_effect, 0.753, 0.504, 0.750)
    assert raw_u == pytest.approx(0.249, abs=1e-9)
    assert adj_u == pytest.approx(0.246, abs=1e-9)


# ------------------------------------------------------------------ patching

def test_restoration_is_zero_and_one_at_the_endpoints():
    assert attempt(drills.restoration, 0.10, 0.10, 0.90) == pytest.approx(0.0)
    assert attempt(drills.restoration, 0.90, 0.10, 0.90) == pytest.approx(1.0)
    assert attempt(drills.restoration, 0.50, 0.10, 0.90) == pytest.approx(0.5)


def test_restoration_survives_an_uninformative_pair():
    """If corrupting did not change the answer there is nothing to restore, and
    the right behaviour is 0.0 rather than a division by zero."""
    assert attempt(drills.restoration, 0.7, 0.7, 0.7) == 0.0


def test_an_essential_input_can_show_zero_effect():
    """§10's point, as arithmetic: b is necessary to the computation but is
    identical in both runs, so patching it restores nothing."""
    acc_corrupt, acc_clean = 0.0, 1.0
    patched_b = acc_corrupt           # splicing an identical value changes nothing
    assert attempt(drills.restoration, patched_b, acc_corrupt, acc_clean) == 0.0


def test_steering_vector_points_from_negative_to_positive():
    r = np.random.default_rng(0)
    base = r.standard_normal((100, 12))
    axis = np.zeros(12); axis[3] = 1.0
    v = attempt(drills.steering_vector, base + 2 * axis, base - 2 * axis)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    assert v @ axis > 0.9


# ---------------------------------------------------------------- Goodhart

def test_selection_on_a_proxy_beats_no_selection():
    r = np.random.default_rng(0)
    true = r.standard_normal(60000)
    proxy = true + r.standard_normal(60000)
    assert attempt(drills.best_of_n_true, true, proxy, 16) > true.mean() + 0.5


def test_a_heavy_tail_destroys_the_true_reward_under_pressure():
    """Figure 4, as a test. Same correlation, opposite outcome at large n."""
    r = np.random.default_rng(0)
    N = 300000
    true = r.standard_normal(N)
    light = r.standard_normal(N); light /= light.std()
    heavy = r.standard_t(1.5, N); heavy /= heavy.std()
    # the two proxies have the same correlation with the truth
    c_l = np.corrcoef(true, true + light)[0, 1]
    c_h = np.corrcoef(true, true + heavy)[0, 1]
    assert abs(c_l - c_h) < 0.02

    l_lo = attempt(drills.best_of_n_true, true, true + light, 8)
    l_hi = attempt(drills.best_of_n_true, true, true + light, 1000)
    h_lo = attempt(drills.best_of_n_true, true, true + heavy, 8)
    h_hi = attempt(drills.best_of_n_true, true, true + heavy, 1000)
    assert l_hi > l_lo               # light-tailed: more pressure still helps
    assert h_hi < h_lo               # heavy-tailed: it does not
    assert l_hi > 2 * h_hi           # and the two end up far apart


def test_tail_ratio_separates_the_two_regimes():
    r = np.random.default_rng(0)
    g = r.standard_normal(400000)
    t = r.standard_t(1.5, 400000)
    assert attempt(drills.tail_ratio, t) > 2 * attempt(drills.tail_ratio, g)


def test_ensembling_shrinks_the_scale_but_not_the_shape():
    """A negative result, and the reason §18 is careful about ensembles.

    Averaging independent proxies reduces the SIZE of the error — the standard
    deviation falls by roughly the usual factor. It does not reduce the tail
    ratio, because a tail index below 2 is preserved under averaging. So an
    ensemble does not convert the heavy-tailed regime into the light-tailed
    one; it only moves along it.
    """
    r = np.random.default_rng(0)
    parts = [r.standard_t(1.5, 200000) for _ in range(4)]
    ens = attempt(drills.ensemble_proxy, parts)
    assert ens.std() < 0.75 * parts[0].std()          # the scale does fall
    solo_tail = attempt(drills.tail_ratio, parts[0])
    ens_tail = attempt(drills.tail_ratio, ens)
    assert ens_tail > 0.9 * solo_tail                 # the shape does not


# --------------------------------------------------------- fairness criteria

def _biased_data(seed=0, n=40000):
    """A PERFECTLY calibrated score, by construction: draw the probability, then
    draw the outcome from it. The groups differ in base rate, which is the
    condition under which the three criteria provably cannot all hold."""
    r = np.random.default_rng(seed)
    group = r.integers(0, 2, n)
    score = np.where(group == 0, r.beta(2, 5, n), r.beta(5, 3, n))
    y = (r.random(n) < score).astype(int)
    return score, y, group


def test_demographic_parity_reports_a_rate_per_group():
    score, y, g = _biased_data()
    rates = attempt(drills.demographic_parity, (score > 0.5).astype(int), g)
    assert set(rates) == {0, 1}
    assert all(0 <= v <= 1 for v in rates.values())


def test_equalised_odds_handles_a_group_with_no_positives():
    y_pred = np.array([0, 1, 0, 1])
    y_true = np.array([0, 0, 0, 0])
    g = np.array([0, 0, 1, 1])
    out = attempt(drills.equalised_odds, y_pred, y_true, g)
    assert np.isnan(out[0][0]) and np.isnan(out[1][0])   # tpr undefined
    assert not np.isnan(out[0][1])                       # fpr is fine


def test_the_three_criteria_cannot_all_hold():
    """A theorem, not a limitation of the method: with different base rates, a
    calibrated score cannot also equalise both odds and selection rates."""
    score, y, g = _biased_data()
    pred = (score > 0.5).astype(int)
    dp = attempt(drills.demographic_parity, pred, g)
    eo = attempt(drills.equalised_odds, pred, y, g)
    cal = attempt(drills.calibration_by_group, score, y, g, 5)

    calibrated = max(cal.values()) < 0.05
    parity = abs(dp[0] - dp[1]) < 0.02
    odds = (abs(eo[0][0] - eo[1][0]) < 0.02 and abs(eo[0][1] - eo[1][1]) < 0.02)
    assert calibrated                       # the score is well calibrated ...
    assert not parity                       # ... and cannot also be these two
    assert sum([calibrated, parity, odds]) < 3
