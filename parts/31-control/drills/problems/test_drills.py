"""
Tests for the Part XXXI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert an EXACT optimum, because the plants here are written down and
LQR is optimal by construction. Several assert a NEGATIVE result, because this
part's content includes what control provably cannot do: remove an offset
without integral action, respect a constraint it does not contain, or establish
a rare failure rate from a clean record.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------- feedback

def test_open_loop_error_tracks_model_error():
    for e in (0.1, 0.2, 0.4):
        y = attempt(drills.open_loop, 1.0, 1.0 + e, 1.0)
        assert y == pytest.approx(1.0 + e, abs=0.02)


def test_feedback_makes_the_plant_gain_nearly_irrelevant():
    ys = [attempt(drills.proportional, 1.0, g, 20.0) for g in (0.6, 1.0, 1.6)]
    assert max(ys) - min(ys) < 0.05


def test_a_proportional_loop_always_misses_by_a_known_amount():
    """Not approximately — the offset is 1/(1+kp*g) and no gain removes it."""
    for g, kp in ((1.0, 1.0), (1.0, 9.0), (2.0, 4.0)):
        y = attempt(drills.proportional, 1.0, g, kp)
        assert 1.0 - y == pytest.approx(attempt(drills.proportional_offset, g, kp), abs=1e-3)


def test_low_gain_feedback_can_be_worse_than_none():
    """The finding of Figure 1. The instinct that feedback is monotonically
    helpful is the instinct that produces this."""
    ol = abs(attempt(drills.open_loop, 1.0, 1.2, 1.0) - 1.0)      # 20% model error
    cl = abs(attempt(drills.proportional, 1.0, 1.2, 1.0) - 1.0)   # cautious gain
    hi = abs(attempt(drills.proportional, 1.0, 1.2, 20.0) - 1.0)
    assert cl > ol
    assert hi < ol


def test_integral_action_removes_the_offset_exactly():
    h0, ok0 = attempt(drills.pi_control, 1.0, 1.0, 4.0, 0.0)
    h1, ok1 = attempt(drills.pi_control, 1.0, 1.0, 4.0, 5.0)
    assert ok0 and ok1
    assert abs(h0[-1] - 1.0) > 0.1
    assert abs(h1[-1] - 1.0) < 1e-6


def test_too_much_integral_gain_diverges():
    _, ok = attempt(drills.pi_control, 1.0, 1.0, 8.0, 10.0)
    assert ok
    _, bad = attempt(drills.pi_control, 1.0, 1.0, 8.0, 400.0)
    assert not bad


def test_delay_lowers_the_gain_you_are_allowed():
    """Latency is a stability question. A gain that is fine with no dead time
    destabilises the same plant once the measurement is stale."""
    _, ok = attempt(drills.pi_control, 1.0, 1.0, 8.0, 20.0, delay=0)
    assert ok
    _, bad = attempt(drills.pi_control, 1.0, 1.0, 8.0, 20.0, delay=40)
    assert not bad


# -------------------------------------------------------------------- optimal

def test_lqr_stabilises_an_integrator_chain():
    for n in (2, 3, 4):
        A, B = attempt(drills.integrator_chain, n)
        K, _ = attempt(drills.dlqr, A, B, np.eye(n), np.array([[0.1]]))
        poles = attempt(drills.closed_loop_poles, A, B, K)
        assert np.abs(poles).max() < 1.0


def test_nothing_beats_lqr():
    """It cannot be beaten for the cost it was given. If a search finds
    something cheaper, the rollout is too short or the cost is not the same one."""
    n = 3
    A, B = attempt(drills.integrator_chain, n)
    Q, R = np.eye(n), np.array([[0.1]])
    K, _ = attempt(drills.dlqr, A, B, Q, R)
    x0 = np.zeros(n); x0[0] = 1.0
    best = attempt(drills.rollout_cost, A, B, Q, R, K, x0)
    r = np.random.default_rng(0)
    for _ in range(300):
        Kp = K * (1 + 0.5 * r.standard_normal(K.shape))
        assert attempt(drills.rollout_cost, A, B, Q, R, Kp, x0) >= best - 1e-6


def test_hand_tuning_falls_off_a_cliff():
    """Fine on two states, hopeless on five — which is the argument for writing
    down a cost instead of a gain."""
    ratios = {}
    for n in (2, 5):
        A, B = attempt(drills.integrator_chain, n)
        Q, R = np.eye(n), np.array([[0.1]])
        K, _ = attempt(drills.dlqr, A, B, Q, R)
        x0 = np.zeros(n); x0[0] = 1.0
        c_lqr = attempt(drills.rollout_cost, A, B, Q, R, K, x0)
        best = np.inf
        for kp in np.logspace(-1, 2.5, 40):
            for kd in np.logspace(-1, 2.5, 40):
                g = np.zeros((1, n)); g[0, 0] = kp; g[0, 1] = kd
                best = min(best, attempt(drills.rollout_cost, A, B, Q, R, g, x0))
        ratios[n] = best / c_lqr
    assert ratios[2] < 1.05                    # a tuned PD is fine here
    assert ratios[5] > 1000                    # and is hopeless here
    # note: a finer grid finds no stabilising PD at all on five integrators.
    # A coarse one finds a technically-bounded gain costing 10^12 times LQR,
    # which is the same conclusion arriving as a number instead of a failure.


def test_an_unstable_gain_does_not_look_cheap():
    """A destabilising gain must report infinity rather than a finite number
    from a rollout that ended before the divergence showed. Note that ZERO gain
    on an integrator chain is marginally stable, not divergent — it sits where
    it started and costs a finite amount, which is a trap when constructing
    this test."""
    A, B = attempt(drills.integrator_chain, 2)
    Q, R = np.eye(2), np.array([[0.1]])
    assert np.isfinite(attempt(drills.rollout_cost, A, B, Q, R,
                               np.zeros((1, 2)), np.array([1.0, 0.0])))
    positive_feedback = np.array([[-50.0, -50.0]])
    assert attempt(drills.rollout_cost, A, B, Q, R, positive_feedback,
                   np.array([1.0, 0.0])) == np.inf


# ----------------------------------------------------------------- transforms

def test_a_transform_and_its_inverse_compose_to_the_identity():
    T = attempt(drills.transform, attempt(drills.rot_z, 0.7), [1.0, -2.0, 0.5])
    assert np.allclose(T @ attempt(drills.transform_inverse, T), np.eye(4), atol=1e-12)


def test_the_inverse_is_not_the_transpose():
    """The most common bug in robotics code, and it produces poses that are
    plausible and wrong."""
    T = attempt(drills.transform, attempt(drills.rot_z, 0.7), [1.0, -2.0, 0.5])
    assert not np.allclose(attempt(drills.transform_inverse, T), T.T, atol=1e-6)


def test_transforms_compose_in_order():
    Ta = attempt(drills.transform, attempt(drills.rot_z, 0.5), [1.0, 0.0, 0.0])
    Tb = attempt(drills.transform, attempt(drills.rot_z, -0.2), [0.0, 2.0, 0.0])
    assert not np.allclose(Ta @ Tb, Tb @ Ta, atol=1e-6)


def test_a_quaternion_and_its_negation_are_the_same_rotation():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(attempt(drills.quat_to_matrix, q),
                       attempt(drills.quat_to_matrix, -q), atol=1e-12)


def test_a_quaternion_gives_a_proper_rotation():
    r = np.random.default_rng(0)
    q = r.standard_normal(4)
    R = attempt(drills.quat_to_matrix, q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_gimbal_lock_costs_a_degree_of_freedom():
    """At a pitch of 90 degrees, roll and yaw do the same thing."""
    a = attempt(drills.euler_to_matrix, 0.3, np.pi / 2, 0.0)
    b = attempt(drills.euler_to_matrix, 0.0, np.pi / 2, -0.3)
    assert np.allclose(a, b, atol=1e-9)
    # away from the singularity they are different rotations
    c = attempt(drills.euler_to_matrix, 0.3, 0.4, 0.0)
    d = attempt(drills.euler_to_matrix, 0.0, 0.4, -0.3)
    assert not np.allclose(c, d, atol=1e-3)


# ----------------------------------------------------------------- kinematics

def test_forward_kinematics_reaches_where_it_should():
    assert np.allclose(attempt(drills.two_link_fk, [0.0, 0.0]), [2.0, 0.0])
    assert np.allclose(attempt(drills.two_link_fk, [np.pi / 2, 0.0]), [0.0, 2.0], atol=1e-12)


def test_the_jacobian_loses_rank_at_full_extension():
    J = attempt(drills.two_link_jacobian, [0.4, 0.0])
    assert abs(np.linalg.det(J)) < 1e-12
    J2 = attempt(drills.two_link_jacobian, [0.4, 1.0])
    assert abs(np.linalg.det(J2)) > 0.1


def test_damping_bounds_the_joint_velocity_near_a_singularity():
    J = attempt(drills.two_link_jacobian, [0.4, 1e-4])     # nearly extended
    v = np.array([0.0, 0.01])
    tiny = attempt(drills.damped_pinv, J, 1e-9) @ v
    damped = attempt(drills.damped_pinv, J, 0.05) @ v
    assert np.abs(tiny).max() > 10 * np.abs(damped).max()


# ------------------------------------------------------------------ estimation

def test_a_bias_grows_faster_than_noise():
    """Which is why a bias must be estimated rather than assumed small."""
    biased = attempt(drills.dead_reckon, 1.0, 0.05, 0.0, 400)
    noisy = attempt(drills.dead_reckon, 1.0, 0.0, 0.5, 400, seed=1)
    assert biased[-1] / max(biased[99], 1e-12) == pytest.approx(4.0, rel=0.05)
    assert noisy[-1] < biased[-1]


# -------------------------------------------------------------------- planning

def test_grid_search_is_exponential_in_the_joints():
    a = attempt(drills.grid_cells, 3)
    b = attempt(drills.grid_cells, 6)
    assert b / a == pytest.approx(12.0 ** 3)
    assert attempt(drills.grid_cells, 8) > 1e8


def test_model_error_compounds_on_an_unstable_plant_and_decays_on_a_stable_one():
    """The measurement of Figure 6, and the reason to use ABSOLUTE divergence:
    the relative version depends only on the ratio of model error to gain and is
    identical for both."""
    up = [attempt(drills.rollout_divergence, 1.05, 0.02, H) for H in (1, 30, 60)]
    dn = [attempt(drills.rollout_divergence, 0.95, 0.02, H) for H in (1, 30, 60)]
    assert up[0] < up[1] < up[2]
    assert dn[0] > dn[1] > dn[2]
    assert up[-1] > 100 * dn[-1]


# ------------------------------------------------------------------ validation

def test_the_rule_of_three():
    assert attempt(drills.rule_of_three, 1000) == pytest.approx(3.0 / 1000, rel=0.01)
    assert attempt(drills.rule_of_three, 1e6) == pytest.approx(3e-6, rel=0.01)


def test_a_clean_record_is_weak_evidence():
    """A million failure-free miles bounds the rate only below one per three
    hundred thousand."""
    bound = attempt(drills.rule_of_three, 1e6)
    assert 1.0 / bound < 4e5


def test_bounding_needs_the_reciprocal_of_the_rate():
    for rate in (1e-5, 1e-6, 1e-8):
        assert attempt(drills.miles_to_bound, rate) == pytest.approx(3.0 / rate, rel=0.01)


def test_estimating_costs_far_more_than_bounding():
    for rate in (1e-6, 1e-8):
        b = attempt(drills.miles_to_bound, rate)
        e = attempt(drills.miles_to_estimate, rate)
        assert e / b > 25
    assert attempt(drills.miles_to_estimate, 1e-8) > 5e9


def test_the_bound_is_only_an_upper_one():
    """Zero failures says nothing about how good the system is, only how bad it
    is not."""
    a = attempt(drills.rule_of_three, 1e6)
    b = attempt(drills.rule_of_three, 1e7)
    assert b < a                                # more miles, tighter bound
    assert a > 0 and b > 0                      # and never zero
