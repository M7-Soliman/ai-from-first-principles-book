"""
Tests for the Part III drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Two themes are graded beyond correctness:
  * stability -- inputs chosen so the textbook formula demonstrably breaks
  * conditioning -- solvers raced as kappa grows, with the normal equations
    expected to fail first
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


# --------------------------------------------------------------- stability --

def test_machine_epsilon_float64():
    eps = attempt(drills.machine_epsilon, np.float64)
    assert np.isclose(eps, np.finfo(np.float64).eps, rtol=2.0)
    assert np.float64(1.0) + eps != np.float64(1.0)
    assert np.float64(1.0) + eps / 2 == np.float64(1.0)


def test_machine_epsilon_float32_is_much_larger():
    e64 = attempt(drills.machine_epsilon, np.float64)
    e32 = attempt(drills.machine_epsilon, np.float32)
    assert e32 > e64 * 1e6, "float32 has ~7 digits, float64 ~16"


def test_logsumexp_matches_naive_when_safe():
    z = rng.normal(size=8)
    assert np.isclose(attempt(drills.logsumexp, z), np.log(np.exp(z).sum()))


def test_logsumexp_survives_overflow():
    v = attempt(drills.logsumexp, np.array([1000.0, 1000.0]))
    assert np.isfinite(v) and np.isclose(v, 1000.0 + np.log(2))


def test_logsumexp_survives_underflow():
    v = attempt(drills.logsumexp, np.array([-1000.0, -1000.0]))
    assert np.isfinite(v) and np.isclose(v, -1000.0 + np.log(2))


def test_log_softmax_no_negative_infinity():
    out = attempt(drills.log_softmax, np.array([0.0, 800.0]))
    assert np.all(np.isfinite(out)), "must not exp() then log()"


def test_log_softmax_sums_to_one_in_probability_space():
    z = rng.normal(size=6)
    assert np.isclose(np.exp(attempt(drills.log_softmax, z)).sum(), 1.0)


def test_quadratic_roots_easy_case():
    r = attempt(drills.quadratic_roots, 1.0, -3.0, 2.0)   # roots 1 and 2
    assert np.allclose(np.sort(r), [1.0, 2.0])


def test_quadratic_roots_survives_cancellation():
    """Drill C2: the textbook formula loses the small root entirely here."""
    b = 1e8
    r = attempt(drills.quadratic_roots, 1.0, b, 1.0)
    small = min(r, key=abs)
    assert np.isclose(small, -1.0 / b, rtol=1e-6), \
        "small root lost to cancellation -- rearrange the formula"


def test_quadratic_roots_product_identity():
    a, b, c = 1.0, 1e7, 3.0
    r1, r2 = attempt(drills.quadratic_roots, a, b, c)
    assert np.isclose(r1 * r2, c / a, rtol=1e-8)


def test_kahan_beats_naive():
    xs = np.full(200000, 0.1)
    exact = 20000.0
    naive = 0.0
    for x in xs:
        naive += x
    k = attempt(drills.kahan_sum, xs)
    assert abs(k - exact) <= abs(naive - exact) + 1e-12


def test_welford_matches_numpy():
    xs = rng.normal(size=500) * 3 + 7
    assert np.isclose(attempt(drills.welford_var, xs), np.var(xs))


def test_welford_survives_large_offset():
    """Drill C3: the E[X^2]-E[X]^2 identity returns 0 or negative here."""
    xs = np.array([1e8, 1e8 + 1.0, 1e8 + 2.0])
    v = attempt(drills.welford_var, xs)
    assert v > 0, "variance must be positive"
    assert np.isclose(v, np.var(xs), rtol=1e-6)


# ------------------------------------------------------------ conditioning --

def test_make_matrix_hits_target_condition_number():
    for k in (1e2, 1e6, 1e10):
        A = attempt(drills.make_matrix, 40, 5, k, np.random.default_rng(1))
        assert A.shape == (40, 5)
        assert np.isclose(np.linalg.cond(A), k, rtol=0.05)


@pytest.mark.parametrize("solver", ["solve_normal", "solve_qr", "solve_svd"])
def test_solvers_agree_when_well_conditioned(solver):
    A = attempt(drills.make_matrix, 50, 4, 10.0, np.random.default_rng(2))
    x = rng.normal(size=4)
    b = A @ x
    got = attempt(getattr(drills, solver), A, b)
    assert np.allclose(got, x, atol=1e-8)


def test_qr_beats_normal_equations_when_ill_conditioned():
    """The whole argument of section 20, as an assertion."""
    A = attempt(drills.make_matrix, 60, 5, 1e9, np.random.default_rng(3))
    x = rng.normal(size=5)
    b = A @ x
    e_norm = np.linalg.norm(attempt(drills.solve_normal, A, b) - x)
    e_qr = np.linalg.norm(attempt(drills.solve_qr, A, b) - x)
    assert e_qr < e_norm, "QR should beat the normal equations at kappa=1e9"


def test_svd_handles_rank_deficiency():
    A = rng.normal(size=(20, 4))
    A[:, 3] = A[:, 0]                      # exactly duplicated column
    b = rng.normal(size=20)
    out = attempt(drills.solve_svd, A, b)
    assert np.all(np.isfinite(out)), "truncate small singular values, don't divide"


# -------------------------------------------------------------- optimisers --

def test_gd_converges_and_returns_full_path():
    grad = lambda p: np.array([2 * p[0], 4 * p[1]])
    path = attempt(drills.gd, grad, np.array([2.0, 1.0]), 0.1, 100)
    assert len(path) == 101, "path must include the starting point"
    assert np.allclose(path[-1], [0.0, 0.0], atol=1e-3)


def test_gd_diverges_past_the_stability_limit():
    """Section 9: the limit is 2/lambda_max. Here lambda_max = 20."""
    grad = lambda p: np.array([20.0 * p[0]])
    ok = attempt(drills.gd, grad, np.array([1.0]), 0.09, 40)     # under 0.1
    bad = attempt(drills.gd, grad, np.array([1.0]), 0.11, 40)    # over 0.1
    assert abs(ok[-1][0]) < 1.0
    assert abs(bad[-1][0]) > 1.0


def test_momentum_beats_plain_on_a_valley():
    grad = lambda p: np.array([2 * p[0], 44.0 * p[1]])
    x0, lr, n = np.array([2.4, 0.9]), 0.03, 60
    plain = attempt(drills.gd, grad, x0, lr, n)
    mom = attempt(drills.gd_momentum, grad, x0, lr, n, 0.85)
    assert np.linalg.norm(mom[-1]) < np.linalg.norm(plain[-1])


def test_newton_converges_in_one_step_on_a_quadratic():
    A = np.array([[1.0, 0.0], [0.0, 14.0]])
    path = attempt(drills.newton, lambda p: A @ p, lambda p: A,
                   np.array([2.2, 1.0]), 3)
    assert np.allclose(path[1], [0.0, 0.0], atol=1e-10), \
        "Newton should land exactly on the minimum in one step"


def test_newton_is_conditioning_independent():
    for k in (1.0, 100.0, 10000.0):
        A = np.diag([1.0, k])
        path = attempt(drills.newton, lambda p: A @ p, lambda p: A,
                       np.array([1.0, 1.0]), 2)
        assert np.allclose(path[1], [0.0, 0.0], atol=1e-9)


def test_rmsprop_beats_gd_on_skewed_curvature():
    A = np.diag([0.4, 30.0])
    grad = lambda p: A @ p
    x0 = np.array([2.4, 0.9])
    g = attempt(drills.gd, grad, x0, 0.03, 90)
    r = attempt(drills.rmsprop, grad, x0, 0.12, 90)
    obj = lambda p: 0.5 * (0.4 * p[0] ** 2 + 30 * p[1] ** 2)
    assert obj(r[-1]) < obj(g[-1])


def test_adam_bias_correction_is_present():
    """Without correction the first step is ~(1-b1)/sqrt(1-b2) times too small."""
    A = np.eye(1)
    path = attempt(drills.adam, lambda p: A @ p, np.array([1.0]), 0.1, 1)
    first = abs(path[1][0] - path[0][0])
    assert first > 0.05, ("first step far too small -- bias correction missing "
                          f"(got {first:.4f}, expected ~0.1)")


def test_adam_converges():
    A = np.diag([1.0, 20.0])
    path = attempt(drills.adam, lambda p: A @ p, np.array([1.5, 1.0]), 0.1, 400)
    assert np.linalg.norm(path[-1]) < 0.1


def test_line_search_accepts_a_decreasing_step():
    f = lambda x: float((x ** 2).sum())
    gf = lambda x: 2 * x
    x = np.array([2.0])
    a = attempt(drills.backtracking_line_search, f, gf, x, -gf(x))
    assert a > 0
    assert f(x + a * (-gf(x))) < f(x)


def test_line_search_shrinks_when_the_first_step_overshoots():
    f = lambda x: float((x ** 2).sum())
    gf = lambda x: 2 * x
    x = np.array([1.0])
    a = attempt(drills.backtracking_line_search, f, gf, x, -gf(x), alpha0=10.0)
    assert a < 10.0, "an overshooting initial step must be backtracked"
