"""
Tests for the Part XIV drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several tests assert that an instrument FAILS: the complex step on a
non-analytic function, a comparison with one run per arm, a grid that cannot
spend its budget. Those are the point of the part.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------- metrics ----

def test_degenerate_classifiers():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, dtype=float)
    p, r = attempt(drills.precision_recall, y, s, 100.0)      # flags nobody
    assert p == pytest.approx(1.0) and r == pytest.approx(0.0)
    p, r = attempt(drills.precision_recall, y, s, -1.0)       # flags everybody
    assert p == pytest.approx(0.2) and r == pytest.approx(1.0)


def test_f1_is_near_zero_when_either_is():
    assert attempt(drills.f1, 1.0, 0.0) == pytest.approx(0.0)
    assert attempt(drills.f1, 0.5, 0.5) == pytest.approx(0.5)
    # harmonic, not arithmetic: 1.0 and 0.02 must NOT average to 0.51
    assert attempt(drills.f1, 1.0, 0.02) < 0.05


def test_accuracy_and_f1_choose_different_thresholds():
    """Section 3's figure, as an assertion."""
    r = np.random.default_rng(0)
    n_pos, n_neg = 200, 3800
    s = np.concatenate([r.normal(1.4, 1.0, n_pos), r.normal(0.0, 1.0, n_neg)])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    ths = np.linspace(s.min(), s.max(), 300)
    accs, f1s, recs = [], [], []
    for t in ths:
        tp, fp, fn, tn = attempt(drills.confusion_counts, y, s, t)
        p, rc = attempt(drills.precision_recall, y, s, t)
        accs.append((tp + tn) / len(y))
        f1s.append(attempt(drills.f1, p, rc))
        recs.append(rc)
    i_acc, i_f1 = int(np.argmax(accs)), int(np.argmax(f1s))
    assert ths[i_acc] > ths[i_f1], "accuracy should prefer a stricter threshold"
    assert recs[i_f1] > 2 * recs[i_acc], \
        f"recall at the two optima: {recs[i_acc]:.3f} vs {recs[i_f1]:.3f}"


def test_coverage_trades_against_accuracy():
    r = np.random.default_rng(1)
    n = 2000
    conf = r.random(n)
    y = np.zeros(n, int)
    pred = (r.random(n) > conf * 0.9).astype(int)     # wrong more often when unsure
    cov_hi, acc_hi = attempt(drills.coverage_accuracy, y, pred, conf, 0.8)
    cov_lo, acc_lo = attempt(drills.coverage_accuracy, y, pred, conf, 0.0)
    assert cov_lo > cov_hi
    assert acc_hi > acc_lo, "refusing the unsure cases must raise accuracy"
    cov0, acc0 = attempt(drills.coverage_accuracy, y, pred, conf, 2.0)
    assert cov0 == 0.0 and acc0 == 1.0, "answering nothing is perfectly accurate"


# --------------------------------------------------------------- calibration ----

def test_ece_is_zero_for_a_calibrated_predictor():
    r = np.random.default_rng(2)
    conf = r.uniform(0.5, 1.0, 40000)
    correct = (r.random(40000) < conf).astype(float)   # correct exactly as often as claimed
    assert attempt(drills.expected_calibration_error, conf, correct, 10) < 0.02


def test_ece_detects_overconfidence():
    r = np.random.default_rng(3)
    conf = r.uniform(0.85, 1.0, 20000)
    correct = (r.random(20000) < 0.6).astype(float)    # claims ~0.9, delivers 0.6
    assert attempt(drills.expected_calibration_error, conf, correct, 10) > 0.25


def test_temperature_scaling_fixes_calibration_and_not_accuracy():
    """Section 5: the whole point is that predictions are unchanged."""
    r = np.random.default_rng(4)
    n, k = 4000, 3
    y = r.integers(0, k, n)
    logits = r.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.1
    logits *= 6.0                                     # deliberately overconfident

    def stats(T):
        p = attempt(drills.softmax_T, logits, T)
        pred = p.argmax(1)
        return p.max(1), (pred == y).astype(float), pred

    c0, k0, pred0 = stats(1.0)
    T = attempt(drills.fit_temperature, logits, y)
    c1, k1, pred1 = stats(T)
    assert T > 1.5, f"an overconfident model needs T > 1, got {T:.3f}"
    assert np.array_equal(pred0, pred1), "temperature must not change any prediction"
    assert k1.mean() == pytest.approx(k0.mean())
    e0 = attempt(drills.expected_calibration_error, c0, k0)
    e1 = attempt(drills.expected_calibration_error, c1, k1)
    assert e1 < e0 / 2, f"calibration error {e0:.3f} -> {e1:.3f}"


# ---------------------------------------------------------- gradient checking ----

def _f(x):
    return np.sin(x) * np.exp(x / 3.0)


def _df(x):
    return np.cos(x) * np.exp(x / 3.0) + np.sin(x) * np.exp(x / 3.0) / 3.0


def test_central_difference_has_a_u_curve():
    x0 = 1.3
    truth = _df(x0)
    errs = {e: abs(attempt(drills.central_difference, _f, x0, e) - truth) / abs(truth)
            for e in (1e-1, 1e-5, 1e-12)}
    assert errs[1e-5] < errs[1e-1], "too large: truncation error"
    assert errs[1e-5] < errs[1e-12], "too small: cancellation — Part III §3"


def test_complex_step_has_no_u_curve():
    """It still has TRUNCATION error at large eps — the O(eps^2) term. What it
    lacks is the cancellation error at small eps, so the curve falls
    monotonically to machine precision instead of turning back up."""
    x0 = 1.3
    truth = _df(x0)
    eps = [1e-3, 1e-5, 1e-8, 1e-20, 1e-100, 1e-150]
    errs = [abs(attempt(drills.complex_step, _f, x0, e) - truth) / abs(truth)
            for e in eps]
    for a, b in zip(errs, errs[1:]):
        assert b <= a + 1e-15, f"error must not rise as eps shrinks: {a:.2e} -> {b:.2e}"
    assert errs[0] > 1e-8, "at eps=1e-3 the O(eps^2) truncation term is still visible"
    for e, err in zip(eps[2:], errs[2:]):
        assert err < 1e-14, f"complex step failed at eps={e:g}: {err:.2e}"


def test_finite_differences_fail_where_the_complex_step_does_not():
    x0 = 1.3
    truth = _df(x0)
    e = 1e-30
    fd = abs(attempt(drills.central_difference, _f, x0, e) - truth) / abs(truth)
    cs = abs(attempt(drills.complex_step, _f, x0, e) - truth) / abs(truth)
    assert fd > 0.5, "the subtraction should have underflowed to zero"
    assert cs < 1e-12


def test_the_complex_step_needs_an_analytic_function():
    """Its blind spot, asserted: abs() is not differentiable in the complex sense."""
    truth = 1.0                                        # d|x|/dx at x = 2.0
    got = attempt(drills.complex_step, np.abs, 2.0, 1e-20)
    assert abs(got - truth) > 0.5, \
        f"complex step should NOT work on abs(), got {got}"


def test_projected_check_catches_a_wrong_jacobian():
    A = np.random.default_rng(5).normal(size=(4, 5))

    def g(v):
        return np.tanh(A @ v)

    def good(v):
        return (1 - np.tanh(A @ v) ** 2)[:, None] * A

    def bad(v):
        return A                                       # forgot the tanh derivative

    x = np.random.default_rng(6).normal(size=5)
    assert attempt(drills.projected_check, g, good, x, seed=0) < 1e-5
    assert attempt(drills.projected_check, g, bad, x, seed=0) > 1e-2


# ------------------------------------------------------------------ diagnosis ----

def test_the_diagnostic_table():
    assert attempt(drills.diagnose, 0.02, 0.03, 0.05) == "working"
    assert attempt(drills.diagnose, 0.02, 0.40, 0.05) == "overfitting-or-broken-eval"
    assert attempt(drills.diagnose, 0.40, 0.42, 0.05) == "underfitting-or-bug"
    assert attempt(drills.diagnose, 0.40, 0.10, 0.05) == "measurement-error"


def test_update_ratio():
    p = [np.ones(10) * 2.0]
    u = [np.ones(10) * 0.02]
    assert attempt(drills.update_ratio, p, u)[0] == pytest.approx(0.01)


def test_worst_mistakes_finds_the_confident_errors():
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.02, 0.98], [0.7, 0.3]])
    y = np.array([0, 0, 0, 0])
    idx = attempt(drills.worst_mistakes, y, probs, k=2)
    assert idx[0] == 2, "the most confidently wrong example must come first"
    assert set(idx.tolist()) == {2, 1}


# --------------------------------------------------------------------- search ----

def test_log_uniform_spreads_across_decades():
    s = attempt(drills.log_uniform, 1e-5, 1e-1, 40000, seed=0)
    top = float(np.mean(s > 1e-2))
    assert 0.2 < top < 0.3, f"a log-uniform sample puts ~1/4 in the top decade, got {top:.3f}"
    lin = np.random.default_rng(0).uniform(1e-5, 1e-1, 40000)
    assert float(np.mean(lin > 1e-2)) > 0.85, "uniform crowds the top decade"


def test_grid_tests_few_distinct_values_and_random_tests_many():
    """Section 16's argument, as two numbers."""
    D, budget = 2, 64
    g = attempt(drills.grid_points, [(0.0, 1.0)] * D, budget)
    r = np.random.default_rng(0).random((len(g), D))
    dg = attempt(drills.distinct_values_per_axis, g)
    dr = attempt(drills.distinct_values_per_axis, r)
    assert dg[0] == 8, f"a 64-point 2-D grid gives 8 distinct values per axis, got {dg[0]}"
    assert dr[0] == len(r), "random gives one distinct value per trial"


def test_grid_cannot_spend_an_arbitrary_budget():
    g = attempt(drills.grid_points, [(0.0, 1.0)] * 3, 60)
    assert len(g) == 27, f"3 dimensions, 60 budget -> 3^3 = 27 points, got {len(g)}"


def test_successive_halving_finds_a_good_arm():
    quality = np.linspace(0, 1, 32)
    rng = np.random.default_rng(7)

    def observe(a, epochs):
        return quality[a] + rng.normal(0, 0.3 / np.sqrt(epochs))

    picks = [quality[attempt(drills.successive_halving, observe, 32, 128)]
             for _ in range(40)]
    assert np.mean(picks) > 0.8, f"mean quality of the chosen arm {np.mean(picks):.3f}"


def test_best_of_k_inflation_grows_with_k():
    infl = [attempt(drills.best_of_k_inflation, 0.8, 0.02, k, trials=3000, seed=0)
            for k in (1, 10, 100)]
    assert infl[0] < 0.005
    assert infl[1] > 0.02
    assert infl[2] > infl[1], "more search means more inflation"


# ------------------------------------------------------------------ reporting ----

def test_one_run_each_cannot_establish_a_difference():
    """Section 19, asserted as a failure."""
    assert attempt(drills.difference_is_detectable, [0.847], [0.851]) is False


def test_a_real_difference_is_detected_and_a_fake_one_is_not():
    r = np.random.default_rng(8)
    a = 0.80 + r.normal(0, 0.005, 10)
    b_same = 0.80 + r.normal(0, 0.005, 10)
    b_diff = 0.86 + r.normal(0, 0.005, 10)
    assert attempt(drills.difference_is_detectable, a, b_same) is False
    assert attempt(drills.difference_is_detectable, a, b_diff) is True


def test_smallest_detectable_difference_shrinks_like_root_n():
    d4 = attempt(drills.smallest_detectable_difference, 0.01, 4)
    d16 = attempt(drills.smallest_detectable_difference, 0.01, 16)
    assert d16 == pytest.approx(d4 / 2, rel=1e-9)
    assert attempt(drills.smallest_detectable_difference, 0.01, 1) == float("inf")
