"""
Tests for the Part V drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Most assertions are about SIMULATED FREQUENCIES, so they carry tolerances
sized to the number of trials. Where a claim in the book is exact -- the MSE
identity, the n-1 correction -- the tolerance is tight.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


rng = np.random.default_rng(0)


# ------------------------------------------------------------ estimators ----

def test_sample_variance_matches_numpy():
    x = rng.normal(size=50)
    assert np.isclose(attempt(drills.sample_variance, x, 1), np.var(x, ddof=1))
    assert np.isclose(attempt(drills.sample_variance, x, 0), np.var(x, ddof=0))


def test_n_minus_one_is_unbiased_and_n_is_not():
    """The whole point of the correction, measured by simulation."""
    r = np.random.default_rng(1)
    biased, unbiased = [], []
    for _ in range(4000):
        s = r.normal(0.0, 1.0, 5)          # small n: the bias is visible
        biased.append(attempt(drills.sample_variance, s, 0))
        unbiased.append(attempt(drills.sample_variance, s, 1))
    assert abs(np.mean(unbiased) - 1.0) < 0.05, "ddof=1 should be unbiased"
    assert np.mean(biased) < 0.9, "ddof=0 should be biased LOW"
    # and the bias should be almost exactly the factor (n-1)/n = 0.8
    assert abs(np.mean(biased) / np.mean(unbiased) - 0.8) < 0.03


def test_standard_error_follows_root_n():
    x = rng.normal(size=100)
    se = attempt(drills.standard_error, x)
    assert np.isclose(se, np.std(x, ddof=1) / 10.0)


def test_standard_error_is_not_the_data_spread():
    x = rng.normal(0, 10, 400)
    se = attempt(drills.standard_error, x)
    assert 0.3 < se < 0.7, "SE ~ 10/20 = 0.5; the DATA spread is 10"


def test_four_times_the_data_halves_the_error():
    r = np.random.default_rng(2)
    se_n = attempt(drills.standard_error, r.normal(size=250))
    se_4n = attempt(drills.standard_error, r.normal(size=1000))
    assert 1.7 < se_n / se_4n < 2.3


def test_mse_identity_holds_exactly():
    est = rng.normal(0.3, 0.5, 5000)       # biased by 0.3, sd 0.5
    b2, v, mse = attempt(drills.mse_decomposition, est, 0.0)
    assert np.isclose(b2 + v, mse, rtol=1e-9), "MSE = Var + Bias^2 is an identity"
    assert abs(b2 - 0.09) < 0.02 and abs(v - 0.25) < 0.02


def test_shrinkage_beats_the_sample_mean_when_data_is_scarce():
    """Part V section 5's Try it, as a simulation."""
    r = np.random.default_rng(3)
    truth, n = 0.1, 4
    plain, shrunk = [], []
    for _ in range(4000):
        s = r.normal(truth, 1.0, n)
        plain.append(np.mean(s))
        shrunk.append(attempt(drills.shrinkage_estimator, s, 0.0, 0.9))
    mse_plain = np.mean((np.array(plain) - truth) ** 2)
    mse_shrunk = np.mean((np.array(shrunk) - truth) ** 2)
    assert mse_shrunk < mse_plain, "with n=4 and truth near 0, shrinking wins"


# ------------------------------------------------------------- intervals ----

def test_confidence_interval_is_centred_and_about_two_se_wide():
    x = rng.normal(5.0, 2.0, 200)
    lo, hi = attempt(drills.confidence_interval, x, 0.95)
    se = attempt(drills.standard_error, x)
    assert np.isclose((lo + hi) / 2, np.mean(x))
    assert np.isclose(hi - lo, 2 * 1.96 * se, rtol=0.02)


def test_ninety_nine_is_wider_than_ninety_five():
    x = rng.normal(size=100)
    l95, h95 = attempt(drills.confidence_interval, x, 0.95)
    l99, h99 = attempt(drills.confidence_interval, x, 0.99)
    assert (h99 - l99) > (h95 - l95)


def test_coverage_is_the_stated_level():
    c = attempt(drills.coverage, 0.0, 1.0, 60, 0.95, 3000, 0)
    assert 0.93 < c < 0.97, f"a 95% procedure should cover ~95% of the time, got {c}"


def test_coverage_tracks_the_level_you_ask_for():
    c90 = attempt(drills.coverage, 0.0, 1.0, 60, 0.90, 3000, 1)
    c99 = attempt(drills.coverage, 0.0, 1.0, 60, 0.99, 3000, 1)
    assert c90 < c99
    assert 0.87 < c90 < 0.93 and 0.975 < c99 < 1.0


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    data = rng.normal(3.0, 1.0, 120)
    a = attempt(drills.bootstrap_ci, data, np.mean, 1500, 0.05, 7)
    b = attempt(drills.bootstrap_ci, data, np.mean, 1500, 0.05, 7)
    assert a == b, "same seed must give the same interval"
    assert a[0] < np.mean(data) < a[1]


def test_bootstrap_matches_the_formula_for_a_mean():
    """Where a formula exists, the bootstrap should agree with it."""
    data = rng.normal(0.0, 1.0, 400)
    blo, bhi = attempt(drills.bootstrap_ci, data, np.mean, 3000, 0.05, 5)
    flo, fhi = attempt(drills.confidence_interval, data, 0.95)
    assert abs(blo - flo) < 0.03 and abs(bhi - fhi) < 0.03


def test_bootstrap_works_for_a_median_which_has_no_formula():
    data = rng.exponential(2.0, 200)        # skewed on purpose
    lo, hi = attempt(drills.bootstrap_ci, data, np.median, 2000, 0.05, 11)
    assert lo < np.median(data) < hi
    assert hi - lo > 0


# --------------------------------------------------------------- testing ----

def test_pvalue_is_large_when_there_is_no_difference():
    r = np.random.default_rng(4)
    p = attempt(drills.two_sample_pvalue, r.normal(size=200), r.normal(size=200))
    assert p > 0.05


def test_pvalue_is_tiny_for_a_large_difference():
    r = np.random.default_rng(5)
    p = attempt(drills.two_sample_pvalue, r.normal(0, 1, 200), r.normal(1.5, 1, 200))
    assert p < 1e-4


def test_pvalues_are_uniform_under_the_null():
    p = attempt(drills.pvalues_under_null, 30, 4000, 0)
    assert 0.03 < np.mean(p < 0.05) < 0.07, "about 5% below 0.05 BY CONSTRUCTION"
    assert 0.45 < np.mean(p) < 0.55, "uniform on [0,1] has mean 0.5"
    assert 0.08 < np.mean(p < 0.10) < 0.12


def test_power_rises_with_effect_and_with_n():
    small = attempt(drills.power, 0.2, 30, 1500, 0.05, 0)
    big = attempt(drills.power, 1.0, 30, 1500, 0.05, 0)
    more_n = attempt(drills.power, 0.2, 200, 1500, 0.05, 0)
    assert small < big and small < more_n
    assert big > 0.9, "a one-sd effect at n=30 should be found nearly always"


def test_underpowered_study_misses_a_real_effect():
    """Figure 7's claim: n=10 misses a half-sd effect most of the time."""
    p = attempt(drills.power, 0.5, 10, 3000, 0.05, 1)
    assert p < 0.5, f"expected under 50% power, got {p}"


def test_bonferroni_divides_the_threshold():
    p = np.array([0.001, 0.02, 0.04, 0.5, 0.9])
    keep = attempt(drills.bonferroni, p, 0.05)
    assert list(keep) == [True, False, False, False, False]   # threshold 0.01


def test_best_of_k_is_biased_upward():
    b1 = attempt(drills.best_of_k_bias, 1, 3000, 200, 0)
    b40 = attempt(drills.best_of_k_bias, 40, 3000, 200, 0)
    assert abs(b1) < 0.005, "one candidate cannot be selected for luck"
    assert b40 > 2 * abs(b1) and b40 > 0.02, \
        "the winner of a 40-way search overstates itself"
    b200 = attempt(drills.best_of_k_bias, 200, 3000, 200, 0)
    assert b200 > b40, "more configurations, more optimism"


def test_pairing_gives_a_tighter_interval():
    """The same data, analysed two ways. Pairing cancels shared difficulty."""
    r = np.random.default_rng(6)
    difficulty = r.normal(0, 1.0, 300)      # some examples are simply hard
    a = difficulty + r.normal(0.15, 0.2, 300)
    b = difficulty + r.normal(0.00, 0.2, 300)
    plo, phi = attempt(drills.paired_difference_ci, a, b)
    ulo, uhi = attempt(drills.unpaired_difference_ci, a, b)
    assert (phi - plo) < 0.3 * (uhi - ulo), "pairing should shrink it a lot"
    assert plo > 0, "paired analysis detects the 0.15 difference"
    assert ulo < 0 < uhi, "unpaired analysis cannot -- same data, weaker method"


# ---------------------------------------------------------- estimation -----

def test_map_reduces_to_mle_under_a_uniform_prior():
    assert np.isclose(attempt(drills.map_bernoulli, 7, 10, 1.0, 1.0), 0.7)


def test_prior_pulls_toward_a_half_and_weakens_with_data():
    near = attempt(drills.map_bernoulli, 3, 4, 5.0, 5.0)     # scarce data
    far = attempt(drills.map_bernoulli, 300, 400, 5.0, 5.0)  # plenty
    assert abs(near - 0.5) < abs(0.75 - 0.5), "the prior should pull it in"
    assert abs(far - 0.75) < 0.02, "data overwhelms the prior"


def test_ridge_shrinks_weights_toward_zero():
    X = rng.normal(size=(60, 5))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 3.0]) + rng.normal(0, 0.1, 60)
    w0 = attempt(drills.ridge_weights, X, y, 0.0)
    w_big = attempt(drills.ridge_weights, X, y, 500.0)
    assert np.linalg.norm(w_big) < np.linalg.norm(w0)
    assert np.allclose(w0, [2, -1, 0.5, 0, 3], atol=0.1), "lam=0 is least squares"


def test_ridge_stabilises_a_collinear_problem():
    """Part III section 7: regularization lifts the smallest eigenvalue."""
    base = rng.normal(size=(50, 1))
    X = np.hstack([base, base + 1e-6 * rng.normal(size=(50, 1))])
    y = (base.ravel() * 3 + rng.normal(0, 0.01, 50))
    w0 = attempt(drills.ridge_weights, X, y, 0.0)
    wr = attempt(drills.ridge_weights, X, y, 1.0)
    assert np.linalg.norm(wr) < np.linalg.norm(w0), \
        "unregularized coefficients blow up on near-duplicate columns"


def test_bias_falls_variance_rises_and_total_is_a_u():
    degrees = [1, 2, 3, 5, 9, 13]
    b2, var, total = attempt(drills.bias_variance_curve, degrees,
                             n_train=25, n_sets=150, noise=0.35, seed=0)
    assert b2[0] > b2[3], "bias must fall as capacity grows"
    assert var[-1] > var[0], "variance must rise as capacity grows"
    assert np.argmin(total) not in (0, len(degrees) - 1), \
        "total error should be minimised strictly in between"
