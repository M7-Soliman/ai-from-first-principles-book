"""
Tests for the Part XXIV drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part is about what an interval
does not promise: that conformal coverage is marginal rather than conditional,
that it collapses when exchangeability breaks, and that a perfectly calibrated
model still loses money at the wrong threshold.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------ coverage

def test_coverage_counts_what_falls_inside():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, 5.0, 2.0])
    hi = np.array([1.0, 1.5, 6.0, 4.0])
    assert attempt(drills.coverage, lo, hi, y) == pytest.approx(0.75)


def test_a_gaussian_interval_covers_at_its_nominal_rate():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    lo, hi = attempt(drills.gaussian_interval, np.zeros(200_000), np.ones(200_000), 0.05)
    assert attempt(drills.coverage, lo, hi, y) == pytest.approx(0.95, abs=0.003)


def test_marginal_coverage_hides_a_group():
    """The whole of §26, as arithmetic: 0.90 overall, 0.80 for half of it."""
    y = np.concatenate([np.zeros(1000), np.zeros(1000)])
    lo = np.full(2000, -1.0); hi = np.full(2000, 1.0)
    y[:200] = 5.0            # group 0 fails 20% of the time
    g = np.concatenate([np.zeros(1000, int), np.ones(1000, int)])
    per = attempt(drills.coverage_by_group, lo, hi, y, g)
    assert per[0] == pytest.approx(0.80)
    assert per[1] == pytest.approx(1.00)
    assert attempt(drills.coverage, lo, hi, y) == pytest.approx(0.90)


# --------------------------------------------------------- the two kinds

def test_the_decomposition_separates_two_known_quantities():
    rng = np.random.default_rng(1)
    K, N = 30, 500
    true_ale = 0.25
    means = rng.standard_normal((K, N)) * 0.4          # spread between members
    vars_ = np.full((K, N), true_ale)
    ale, epi = attempt(drills.decompose_variance, means, vars_)
    assert np.allclose(ale, true_ale)
    assert epi.mean() == pytest.approx(0.16, rel=0.15)


def test_epistemic_falls_with_agreement_and_aleatoric_does_not():
    N = 200
    tight = np.tile(np.zeros(N), (20, 1)) + 0.01 * np.random.default_rng(2).standard_normal((20, N))
    wide = np.tile(np.zeros(N), (20, 1)) + 1.00 * np.random.default_rng(3).standard_normal((20, N))
    v = np.full((20, N), 0.3)
    a1, e1 = attempt(drills.decompose_variance, tight, v)
    a2, e2 = attempt(drills.decompose_variance, wide, v)
    assert e1.mean() < e2.mean() / 100
    assert a1.mean() == pytest.approx(a2.mean())        # unmoved, as it must be


# ------------------------------------------------------ Gaussian processes

def test_the_posterior_interpolates_the_observations():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.sin(x)
    mu, var = attempt(drills.gp_posterior, x, y, x, 1.0, 1.0, 1e-4)
    assert np.allclose(mu, y, atol=1e-3)
    assert np.all(var < 1e-3)                            # certain where observed


def test_the_variance_grows_away_from_the_data():
    """Figure 2's property, as a test."""
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    _, near = attempt(drills.gp_posterior, x, y, np.array([0.0]), 0.5, 1.0, 0.1)
    _, far = attempt(drills.gp_posterior, x, y, np.array([8.0]), 0.5, 1.0, 0.1)
    assert far[0] > 10 * near[0]


def test_the_posterior_variance_ignores_the_observed_values():
    """It depends on WHERE you observed, not what you saw — §9."""
    x = np.array([-1.0, 0.0, 1.0])
    q = np.array([0.4, 2.0])
    _, v1 = attempt(drills.gp_posterior, x, np.array([0.0, 1.0, 0.0]), q, 0.7, 1.0, 0.1)
    _, v2 = attempt(drills.gp_posterior, x, np.array([9.0, -3.0, 4.0]), q, 0.7, 1.0, 0.1)
    assert np.allclose(v1, v2)


def test_marginal_likelihood_prefers_the_length_scale_that_generated_the_data():
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(-3, 3, 60))
    K = attempt(drills.rbf_kernel, x, x, 0.8, 1.0) + 1e-6 * np.eye(60)
    y = np.linalg.cholesky(K) @ rng.standard_normal(60)
    lmls = {ell: attempt(drills.log_marginal_likelihood, x, y, ell, 1.0, 0.05)
            for ell in (0.1, 0.3, 0.8, 2.0, 6.0)}
    assert max(lmls, key=lmls.get) in (0.3, 0.8)


# ---------------------------------------------------- conformal prediction

def test_conformal_covers_at_the_nominal_rate():
    """Figure 4. The model here is a constant, and it makes no difference."""
    rng = np.random.default_rng(5)
    covs = []
    for _ in range(300):
        cal = np.abs(rng.standard_normal(500))
        q = attempt(drills.conformal_quantile, cal, 0.10)
        test = np.abs(rng.standard_normal(400))
        covs.append(np.mean(test <= q))
    assert np.mean(covs) == pytest.approx(0.90, abs=0.01)


def test_the_finite_sample_correction_matters():
    """Using n instead of n+1 under-covers. Small, systematic, and free to fix."""
    rng = np.random.default_rng(6)
    n = 20
    right, wrong = [], []
    for _ in range(4000):
        cal = rng.standard_normal(n)
        q_right = attempt(drills.conformal_quantile, cal, 0.10)
        s = np.sort(cal)
        q_wrong = s[int(np.ceil(n * 0.90)) - 1]
        t = rng.standard_normal()
        right.append(t <= q_right); wrong.append(t <= q_wrong)
    assert np.mean(right) >= 0.90 - 0.015
    assert np.mean(wrong) < np.mean(right)


def test_conformal_returns_infinity_when_the_sample_is_too_small():
    """At n = 5 you cannot promise 99% coverage, and the honest output says so."""
    assert attempt(drills.conformal_quantile, np.arange(5.0), 0.01) == float("inf")


def test_the_guarantee_does_not_depend_on_the_model():
    """A useless model is covered at the same rate and pays in width."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-2, 2, 800)
    y = np.sin(3 * x) + 0.2 * rng.standard_normal(800)
    good = np.polyval(np.polyfit(x, y, 7), x)
    bad = np.full_like(y, y.mean())
    out = {}
    for name, pred in (("good", good), ("bad", bad)):
        q = attempt(drills.conformal_quantile, np.abs(y - pred)[:400], 0.10)
        lo, hi = attempt(drills.conformal_interval, pred[400:], q)
        out[name] = (attempt(drills.coverage, lo, hi, y[400:]),
                     attempt(drills.interval_width, lo, hi))
    assert out["good"][0] == pytest.approx(0.90, abs=0.05)
    assert out["bad"][0] == pytest.approx(0.90, abs=0.05)
    assert out["bad"][1] > 2 * out["good"][1]        # the width is where it shows


def test_normalised_scores_make_the_interval_adapt():
    y = np.array([0.0, 0.0]); pred = np.array([1.0, 1.0])
    s = attempt(drills.normalised_scores, y, pred, np.array([1.0, 4.0]))
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.25)


def test_a_prediction_set_can_be_empty_and_that_is_informative():
    # deliberately off the boundary: 1 - 0.95 evaluates to 0.05000000000000004,
    # which is above a threshold of 0.05, and a test sitting exactly on the edge
    # measures floating point rather than the method.
    probs = np.array([[0.4, 0.35, 0.25], [0.97, 0.02, 0.01]])
    tight = attempt(drills.conformal_set, probs, 0.05)
    assert tight[0].sum() == 0            # nothing scored well enough
    assert tight[1].sum() == 1
    loose = attempt(drills.conformal_set, probs, 0.8)
    assert loose[0].sum() == 3


# ---------------------------------------------------------------- decisions

def test_the_optimal_threshold_is_the_cost_ratio():
    assert attempt(drills.optimal_threshold, 1.0, 1.0) == pytest.approx(0.5)
    assert attempt(drills.optimal_threshold, 1.0, 9.0) == pytest.approx(0.1)
    assert attempt(drills.optimal_threshold, 3.0, 1.0) == pytest.approx(0.75)


def test_a_calibrated_model_still_loses_at_the_default_threshold():
    """Figure 6. Nothing is wrong with the probabilities."""
    rng = np.random.default_rng(8)
    p = rng.beta(1.2, 6.0, 200_000)
    y = rng.random(200_000) < p                  # calibrated by construction
    c_fp, c_fn = 1.0, 20.0
    t_opt = attempt(drills.optimal_threshold, c_fp, c_fn)
    at_opt = attempt(drills.expected_cost, p, y, t_opt, c_fp, c_fn)
    at_half = attempt(drills.expected_cost, p, y, 0.5, c_fp, c_fn)
    assert at_half > 1.5 * at_opt


def test_the_optimal_threshold_is_actually_optimal():
    rng = np.random.default_rng(9)
    p = rng.beta(2.0, 4.0, 100_000)
    y = rng.random(100_000) < p
    c_fp, c_fn = 1.0, 4.0
    t = attempt(drills.optimal_threshold, c_fp, c_fn)
    best = attempt(drills.expected_cost, p, y, t, c_fp, c_fn)
    for other in np.linspace(0.02, 0.98, 40):
        assert attempt(drills.expected_cost, p, y, other, c_fp, c_fn) >= best - 1e-3


def test_information_is_worth_nothing_if_it_never_changes_the_action():
    """§36's point, and the reason feature importance is the wrong instrument."""
    U = np.array([[0.0, -1.0], [-100.0, -100.0]])   # action 1 is never worth taking
    p0 = np.array([0.5, 0.5])
    ps = np.array([0.5, 0.5])
    pos = np.array([[0.9, 0.1], [0.1, 0.9]])        # highly informative signal
    assert attempt(drills.value_of_information, p0, ps, pos, U) == pytest.approx(0.0, abs=1e-9)


def test_information_is_worth_something_when_it_does():
    U = np.array([[0.0, -1.0], [-0.45, -0.45]])
    p0 = np.array([0.5, 0.5])
    ps = np.array([0.5, 0.5])
    pos = np.array([[0.95, 0.05], [0.05, 0.95]])
    assert attempt(drills.value_of_information, p0, ps, pos, U) > 0.0


def test_abstention_is_a_band_and_can_be_empty():
    lo, hi = attempt(drills.abstain_band, 1.0, 1.0, 0.2)
    assert 0 < lo < hi < 1
    lo2, hi2 = attempt(drills.abstain_band, 1.0, 1.0, 0.9)
    assert np.isnan(lo2) and np.isnan(hi2)     # too expensive to ever be worth it
