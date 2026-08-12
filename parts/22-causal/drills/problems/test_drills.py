"""
Tests for the Part XXII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because that is what this part is about: that
a correctly computed regression reports the wrong sign, that conditioning on a
collider manufactures an association between independent variables, and that
adjusting for a descendant of the treatment makes the answer worse rather than
better.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# a graph is node -> set of its DIRECT CAUSES
CONFOUND = {"X": {"U"}, "Y": {"X", "U"}, "U": set()}
COLLIDE = {"X": set(), "Y": set(), "C": {"X", "Y"}}
MEDIATE = {"X": set(), "M": {"X"}, "Y": {"M"}}
CHAIN_D = {"X": set(), "C": {"X", "Y"}, "D": {"C"}, "Y": set()}


# ------------------------------------------------------------ the graph tools

def test_descendants_follow_arrows_forward():
    assert attempt(drills.descendants, MEDIATE, "X") == {"M", "Y"}
    assert attempt(drills.descendants, MEDIATE, "Y") == set()


def test_collider_is_identified_on_a_path():
    p = ["X", "C", "Y"]
    assert attempt(drills.is_collider, COLLIDE, p, 1) is True
    p2 = ["X", "M", "Y"]
    assert attempt(drills.is_collider, MEDIATE, p2, 1) is False


def test_fork_is_open_and_closes_under_conditioning():
    assert attempt(drills.d_separated, CONFOUND, "X", "Y", set()) is False
    # X and Y are still connected directly, so conditioning on U cannot separate
    g = {"X": {"U"}, "Y": {"U"}, "U": set()}
    assert attempt(drills.d_separated, g, "X", "Y", set()) is False
    assert attempt(drills.d_separated, g, "X", "Y", {"U"}) is True


def test_collider_is_closed_and_opens_under_conditioning():
    """Figure 2, as a graph statement. This is the rule that reverses."""
    assert attempt(drills.d_separated, COLLIDE, "X", "Y", set()) is True
    assert attempt(drills.d_separated, COLLIDE, "X", "Y", {"C"}) is False


def test_conditioning_on_a_colliders_descendant_also_opens_it():
    assert attempt(drills.d_separated, CHAIN_D, "X", "Y", set()) is True
    assert attempt(drills.d_separated, CHAIN_D, "X", "Y", {"D"}) is False


def test_chain_closes_under_conditioning_on_the_mediator():
    assert attempt(drills.d_separated, MEDIATE, "X", "Y", set()) is False
    assert attempt(drills.d_separated, MEDIATE, "X", "Y", {"M"}) is True


def test_backdoor_accepts_the_confounder():
    assert attempt(drills.satisfies_backdoor, CONFOUND, "X", "Y", {"U"}) is True
    assert attempt(drills.satisfies_backdoor, CONFOUND, "X", "Y", set()) is False


def test_backdoor_rejects_a_descendant_of_the_treatment():
    """§12's first condition, which is the formal version of 'do not adjust for
    everything you have'."""
    assert attempt(drills.satisfies_backdoor, MEDIATE, "X", "Y", {"M"}) is False


def test_backdoor_is_satisfied_by_the_empty_set_when_nothing_confounds():
    assert attempt(drills.satisfies_backdoor, MEDIATE, "X", "Y", set()) is True


# ------------------------------------------------------------- the estimators

def _confounded(n=60000, tau=1.0, b=2.6, seed=0):
    r = np.random.default_rng(seed)
    u = r.standard_normal(n)
    t = r.random(n) < 1 / (1 + np.exp(-1.5 * u))
    y = tau * t - b * u + r.standard_normal(n)
    return y, t, u


def test_the_naive_difference_gets_the_sign_wrong():
    """Figure 1's headline, as a test: the estimate is computed correctly and
    reports the opposite of the truth."""
    y, t, u = _confounded()
    naive = attempt(drills.naive_difference, y, t)
    assert naive < 0                       # the true effect is +1.0


def test_adjustment_recovers_the_true_effect():
    y, t, u = _confounded()
    adj = attempt(drills.backdoor_adjust, y, t, u, 20)
    assert adj == pytest.approx(1.0, abs=0.08)


def test_more_data_does_not_fix_confounding():
    """It is bias, not variance — the whole reason Part XIV's diagnostics are
    silent about it."""
    small = attempt(drills.naive_difference, *_confounded(4000)[:2])
    big = attempt(drills.naive_difference, *_confounded(200000)[:2])
    assert abs(small - 1.0) > 1.0 and abs(big - 1.0) > 1.0
    assert abs(big - small) < 0.25          # it converged — to the wrong number


def test_propensity_recovers_the_assignment_rule():
    r = np.random.default_rng(1)
    u = r.standard_normal(40000)
    true_e = 1 / (1 + np.exp(-1.5 * u))
    t = r.random(40000) < true_e
    e = attempt(drills.fit_propensity, u, t)
    assert np.mean(np.abs(e - true_e)) < 0.02


def test_ipw_matches_adjustment_on_the_same_data():
    y, t, u = _confounded()
    e = attempt(drills.fit_propensity, u, t)
    assert attempt(drills.ipw_estimate, y, t, e) == pytest.approx(1.0, abs=0.08)


def test_aipw_is_right_when_only_the_propensity_is():
    """The doubly robust property, with a deliberately useless outcome model."""
    y, t, u = _confounded()
    e = attempt(drills.fit_propensity, u, t)
    m1 = np.zeros_like(y)                   # wrong, and not even a function of u
    m0 = np.zeros_like(y)
    assert attempt(drills.aipw_estimate, y, t, e, m1, m0) == pytest.approx(1.0, abs=0.1)


def test_effective_sample_size_is_one_for_equal_weights():
    assert attempt(drills.effective_sample_size, np.ones(500)) == pytest.approx(1.0)


def test_effective_sample_size_collapses_when_one_weight_dominates():
    w = np.ones(1000); w[0] = 1e5
    assert attempt(drills.effective_sample_size, w) < 0.01


def test_overlap_is_empty_when_the_groups_separate():
    e = np.concatenate([np.full(50, 0.05), np.full(50, 0.95)])
    t = np.concatenate([np.zeros(50, bool), np.ones(50, bool)])
    lo, hi = attempt(drills.overlap_range, e, t)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------- the designs

def _iv(n=40000, tau=1.0, g=0.9, seed=2):
    r = np.random.default_rng(seed)
    u = r.standard_normal(n)
    z = r.standard_normal(n)
    x = g * z + 1.0 * u + r.standard_normal(n)
    y = tau * x + 1.5 * u + r.standard_normal(n)
    return y, x, z


def test_two_stage_recovers_the_effect_with_the_confounder_unobserved():
    y, x, z = _iv()
    assert attempt(drills.two_stage_least_squares, y, x, z) == pytest.approx(1.0, abs=0.06)


def test_ordinary_least_squares_is_biased_on_the_same_data():
    y, x, z = _iv()
    import numpy.linalg as la
    A = np.column_stack([np.ones(len(x)), x])
    ols = la.lstsq(A, y, rcond=None)[0][1]
    assert ols > 1.2                        # the confounder inflates it


def test_a_weak_instrument_is_unusable_and_says_so():
    """Not a wider interval around the right answer — an estimate with no
    information in it. Compare the strength statistic in each case."""
    strong = attempt(drills.first_stage_strength, *_iv(g=0.9)[1:])
    weak = attempt(drills.first_stage_strength, *_iv(g=0.02)[1:])
    assert strong > 1000 and weak < 30
    ests = [attempt(drills.two_stage_least_squares, *_iv(g=0.02, seed=s))
            for s in range(12)]
    assert np.std(ests) > 0.5               # wildly unstable across samples


def test_difference_in_differences_removes_a_fixed_group_gap():
    r = np.random.default_rng(3)
    n = 20000
    tr = r.random(n) < 0.5
    af = r.random(n) < 0.5
    # a large fixed difference between groups, a common trend, and a +2 effect
    y = 5.0 * tr + 3.0 * af + 2.0 * (tr & af) + r.standard_normal(n)
    assert attempt(drills.did_estimate, y, tr, af) == pytest.approx(2.0, abs=0.1)


def test_regression_discontinuity_finds_the_jump():
    r = np.random.default_rng(4)
    n = 40000
    run = r.uniform(-1, 1, n)
    y = 1.5 * run + 2.0 * (run >= 0) + 0.3 * r.standard_normal(n)
    assert attempt(drills.rdd_estimate, y, run, 0.0, 0.3) == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------- policies and transfer

def test_off_policy_value_recovers_a_known_policy_value():
    r = np.random.default_rng(5)
    n = 200000
    # two actions; action 1 pays 1.0, action 0 pays 0.0
    logged_p1 = 0.3
    a = r.random(n) < logged_p1
    rew = a.astype(float) + 0.05 * r.standard_normal(n)
    logged = np.where(a, logged_p1, 1 - logged_p1)
    target_p1 = 0.8
    target = np.where(a, target_p1, 1 - target_p1)
    v = attempt(drills.off_policy_value, rew, a, logged, target)
    assert v == pytest.approx(0.8, abs=0.02)


def test_off_policy_estimate_fails_without_support():
    """If the logging policy never took an action, no reweighting recovers its
    value. The estimate is not merely uncertain — it is about a different
    policy."""
    r = np.random.default_rng(6)
    n = 20000
    a = np.zeros(n, bool)                   # action 1 never taken
    rew = np.zeros(n)
    logged = np.ones(n)
    target = np.full(n, 0.1)                # target mostly wants action 1
    v = attempt(drills.off_policy_value, rew, a, logged, target)
    assert v == pytest.approx(0.0, abs=1e-9)   # says zero, knows nothing


def test_self_normalising_trades_bias_for_variance():
    r = np.random.default_rng(7)
    vals_ips, vals_sn = [], []
    for s in range(40):
        rr = np.random.default_rng(100 + s)
        n = 2000
        a = rr.random(n) < 0.1
        rew = a.astype(float)
        logged = np.where(a, 0.1, 0.9)
        target = np.where(a, 0.9, 0.1)
        vals_ips.append(attempt(drills.off_policy_value, rew, a, logged, target))
        vals_sn.append(attempt(drills.self_normalised_value, rew, a, logged, target))
    assert np.std(vals_sn) < np.std(vals_ips)
