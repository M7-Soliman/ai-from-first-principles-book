"""
Tests for the Part XXV drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part's subject includes what does
not work: that permutation importance calls two essential correlated features
unimportant, that a random split on ordered rows reports a model that does not
exist, and that averaging has a floor no number of trees reaches.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------ splitting

def test_sse_is_zero_for_a_constant():
    assert attempt(drills.sse, np.full(50, 3.0)) == pytest.approx(0.0)


def test_best_split_finds_a_planted_threshold():
    r = np.random.default_rng(0)
    x = r.uniform(-1, 1, 4000)
    y = (x > 0.25).astype(float) * 3.0 + 0.05 * r.standard_normal(4000)
    gain, thr = attempt(drills.best_split, x, y)
    assert gain > 0
    assert thr == pytest.approx(0.25, abs=0.02)


def test_best_split_declines_when_there_is_nothing():
    r = np.random.default_rng(1)
    x = r.uniform(0, 1, 500)
    y = np.zeros(500)
    gain, thr = attempt(drills.best_split, x, y)
    assert gain == 0.0 and thr is None


def test_the_gain_identity_matches_impurity_reduction():
    r = np.random.default_rng(2)
    y = r.standard_normal(200)
    k = 73
    direct = attempt(drills.sse, y) - attempt(drills.sse, y[:k]) - attempt(drills.sse, y[k:])
    viasum = attempt(drills.split_gain, y[:k].sum(), k, y[k:].sum(), len(y) - k)
    assert float(viasum) == pytest.approx(direct, rel=1e-9)


def test_histogram_split_agrees_with_the_exact_one():
    """The approximation should land in the same place when the bins are fine."""
    r = np.random.default_rng(3)
    x = r.uniform(-1, 1, 5000)
    y = (x > 0.4).astype(float) * 2.0 + 0.05 * r.standard_normal(5000)
    _, thr = attempt(drills.best_split, x, y)
    edges = attempt(drills.quantile_bins, x, 128)
    xb = np.searchsorted(edges, x, side="right")
    _, b = attempt(drills.histogram_best_split, xb, y, len(edges) + 1)
    assert b is not None
    approx_thr = edges[min(b, len(edges) - 1)]
    assert abs(approx_thr - thr) < 0.05


# ------------------------------------------------------------------ ensembles

def test_averaging_variance_falls_to_a_floor_and_no_further():
    s2, rho = 4.0, 0.3
    v10 = attempt(drills.averaging_variance, s2, rho, 10)
    v1000 = attempt(drills.averaging_variance, s2, rho, 1000)
    floor = attempt(drills.variance_floor, s2, rho)
    assert v1000 < v10
    assert v1000 > floor
    assert v1000 == pytest.approx(floor, abs=0.01)


def test_independent_predictors_average_to_nothing():
    assert attempt(drills.averaging_variance, 1.0, 0.0, 10000) == pytest.approx(1e-4)


def test_correlated_predictors_cannot_be_averaged_away():
    """The floor is the whole reason decorrelation is attempted at all."""
    assert attempt(drills.variance_floor, 1.0, 0.5) == pytest.approx(0.5)
    assert attempt(drills.averaging_variance, 1.0, 0.5, 10**6) > 0.499


def test_a_bootstrap_leaves_about_a_third_out():
    fr = [attempt(drills.oob_mask, 2000, seed=s).mean() for s in range(10)]
    assert np.mean(fr) == pytest.approx(1 / np.e, abs=0.02)


def test_permutation_importance_finds_a_feature_that_matters():
    r = np.random.default_rng(4)
    X = r.standard_normal((3000, 4))
    y = 3.0 * X[:, 1]
    predict = lambda A: 3.0 * A[:, 1]
    assert attempt(drills.permutation_importance, predict, X, y, 1) > 5.0
    assert attempt(drills.permutation_importance, predict, X, y, 0) == pytest.approx(0, abs=0.2)


def test_permutation_importance_misses_duplicated_signal():
    """Two columns carrying the same thing: permuting either alone costs
    nothing, and both look unimportant while the pair is essential."""
    r = np.random.default_rng(5)
    z = r.standard_normal(3000)
    X = np.column_stack([z, z.copy(), r.standard_normal(3000)])
    y = 3.0 * z
    predict = lambda A: 1.5 * A[:, 0] + 1.5 * A[:, 1]
    i0 = attempt(drills.permutation_importance, predict, X, y, 0)
    i1 = attempt(drills.permutation_importance, predict, X, y, 1)
    both = float(np.mean((predict(np.column_stack(
        [r.standard_normal(3000), r.standard_normal(3000), X[:, 2]])) - y) ** 2))
    assert i0 < 0.35 * both and i1 < 0.35 * both


# ------------------------------------------------------------------- boosting

def test_the_squared_loss_gradient_is_the_residual():
    y = np.array([1.0, 2.0, 3.0]); F = np.array([0.5, 2.5, 3.0])
    assert np.allclose(attempt(drills.negative_gradient, y, F, "ls"), y - F)
    assert np.allclose(attempt(drills.hessian, y, F, "ls"), 1.0)


def test_the_logistic_gradient_is_the_calibration_error():
    y = np.array([1.0, 0.0]); F = np.array([0.0, 0.0])
    g = attempt(drills.negative_gradient, y, F, "logistic")
    assert np.allclose(g, [0.5, -0.5])
    assert np.allclose(attempt(drills.hessian, y, F, "logistic"), 0.25)


def test_the_leaf_value_is_the_mean_for_squared_loss():
    g = np.array([1.0, 2.0, 3.0]); h = np.ones(3)
    assert attempt(drills.leaf_value, g, h, 0.0) == pytest.approx(2.0)


def test_the_penalty_shrinks_the_leaf_toward_zero():
    g = np.array([4.0, 4.0]); h = np.ones(2)
    a = attempt(drills.leaf_value, g, h, 0.0)
    b = attempt(drills.leaf_value, g, h, 6.0)
    assert abs(b) < abs(a)


def test_a_split_that_does_not_pay_for_itself_scores_negative():
    g = np.array([1.0] * 10); h = np.ones(10)
    # a split of an entirely homogeneous node, charged a per-leaf penalty
    val = attempt(drills.structured_gain, g[:5], h[:5], g[5:], h[5:], 0.0, 5.0)
    assert val < 0


# --------------------------------------------------------------- calibration

def test_ece_is_zero_for_a_calibrated_model():
    r = np.random.default_rng(6)
    p = r.uniform(0, 1, 400_000)
    y = (r.random(400_000) < p).astype(float)
    assert attempt(drills.expected_calibration_error, p, y) < 0.005


def test_ece_catches_a_systematically_overconfident_model():
    r = np.random.default_rng(7)
    p_true = r.uniform(0, 1, 200_000)
    y = (r.random(200_000) < p_true).astype(float)
    p_bad = np.clip(p_true * 1.4, 0, 1)
    assert attempt(drills.expected_calibration_error, p_bad, y) > 0.08


def test_isotonic_is_monotone_and_changes_no_ranking():
    """The property that makes a calibrator safe to apply to a deployed
    ranker."""
    r = np.random.default_rng(8)
    s = r.uniform(0, 1, 4000)
    y = (r.random(4000) < np.clip(s ** 2, 0, 1)).astype(float)
    xs, vs = attempt(drills.isotonic_fit, s, y)
    assert np.all(np.diff(vs) >= -1e-12)
    cal = np.interp(s, xs, vs)
    assert attempt(drills.auc, cal, y) == pytest.approx(attempt(drills.auc, s, y), abs=0.01)


# ------------------------------------------------------------- ordered rows

def test_lag_matrix_never_uses_the_future():
    y = np.arange(20.0)
    X, t = attempt(drills.lag_matrix, y, 3, horizon=1)
    # every lag must be strictly less than its target
    assert np.all(X.max(axis=1) < t)


def test_lag_matrix_respects_the_horizon():
    y = np.arange(30.0)
    X, t = attempt(drills.lag_matrix, y, 4, horizon=5)
    assert np.all(t - X.max(axis=1) >= 5)


def test_rolling_origin_never_trains_on_the_future():
    for tr, te in attempt(drills.rolling_origin_splits, 200, 4, 60, 1):
        assert tr.max() < te.min()


def test_a_random_split_is_optimistic_on_a_series():
    """Figure 8, in miniature: nearest-neighbour interpolation is trivial when
    the split leaves both neighbours in training, and that is not the task."""
    r = np.random.default_rng(9)
    y = np.cumsum(r.standard_normal(3000) * 0.2)
    X, t = attempt(drills.lag_matrix, y, 4, 1)
    n = len(t)
    fit = lambda tr: np.linalg.lstsq(
        np.column_stack([X[tr], np.ones(len(tr))]), t[tr], rcond=None)[0]
    err = lambda w, te: float(np.mean(
        (np.column_stack([X[te], np.ones(len(te))]) @ w - t[te]) ** 2))
    perm = r.permutation(n); cut = int(0.75 * n)
    rand = err(fit(perm[:cut]), perm[cut:])
    chrono = err(fit(np.arange(cut)), np.arange(cut, n))
    assert rand < chrono          # the random split reports the better number


def test_seasonal_naive_repeats_the_season():
    y = np.array([1., 2., 3., 1., 2., 3., 1., 2., 3.])
    p = attempt(drills.seasonal_naive, y, 3)
    assert np.allclose(p[3:], y[3:])


def test_pinball_is_minimised_at_the_right_quantile():
    r = np.random.default_rng(10)
    y = r.standard_normal(200_000)
    for tau in (0.1, 0.5, 0.9):
        true_q = np.quantile(y, tau)
        best = attempt(drills.pinball_loss, y, np.full_like(y, true_q), tau)
        for off in (-0.4, -0.15, 0.15, 0.4):
            assert attempt(drills.pinball_loss, y, np.full_like(y, true_q + off), tau) >= best
