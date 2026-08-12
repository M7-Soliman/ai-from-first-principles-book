"""
Tests for the Part IX drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

The load-bearing tests are the ones that assert a *number the book quotes*:
the 1/sqrt(b) exponent, the tenfold amplification of the Newton step, the
machine-noise flatness of the ReLU hyperbola, conjugate gradients terminating
in n steps at kappa = 10 and failing to at kappa = 1000. Where the book says
an agreement is only approximate, the test asserts that too.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def quadratic(lam):
    """f = 0.5 * sum(lam_i theta_i^2); gradient is lam * theta."""
    lam = np.asarray(lam, float)
    return lambda th: lam * np.asarray(th, float)


def spd(n, kappa, seed=2):
    r = np.random.default_rng(seed)
    Q = np.linalg.qr(r.normal(size=(n, n)))[0]
    A = Q @ np.diag(np.logspace(0, np.log10(kappa), n)) @ Q.T
    return A, r.normal(size=n)


def regression_data(n=4000, d=20, seed=7, cond=1.0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    if cond != 1.0:
        S = np.diag(np.logspace(0, np.log10(cond) / 2, d))
        Q = np.linalg.qr(r.normal(size=(d, d)))[0]
        X = X @ (Q @ S @ Q.T)
    w = r.normal(size=d)
    return X, X @ w + r.normal(size=n)


# --------------------------------------------------------- instruments ----

def test_gHg_matches_exact_quadratic():
    """On a quadratic, H is constant and g'Hg is exact."""
    lam = np.array([1.0, 7.0, 40.0])
    th = np.array([0.6, -1.3, 0.2])
    got = attempt(drills.gHg, quadratic(lam), th)
    g = lam * th
    assert got == pytest.approx(float(g @ (lam * g)), rel=1e-4)


def test_gHg_no_hessian_needed_on_a_nonquadratic():
    """A quartic: compare against an exact Hessian computed by hand."""
    def grad(t):
        t = np.asarray(t, float)
        return 4 * t ** 3
    th = np.array([0.5, -0.9])
    g = 4 * th ** 3
    H = np.diag(12 * th ** 2)
    got = attempt(drills.gHg, grad, th)
    assert got == pytest.approx(float(g @ H @ g), rel=1e-3)


def test_step_ceiling_matches_the_formula():
    lam = np.array([1.0, 25.0])
    th = np.array([2.0, 0.5])
    got = attempt(drills.step_ceiling, quadratic(lam), th)
    g = lam * th
    assert got == pytest.approx(2 * (g @ g) / (g @ (lam * g)), rel=1e-3)


def test_step_ceiling_is_infinite_under_negative_curvature():
    """g'Hg < 0 means curvature is helping; there is no upper bound."""
    lam = np.array([-3.0, -1.0])
    got = attempt(drills.step_ceiling, quadratic(lam), np.array([1.0, 1.0]))
    assert np.isinf(got)


def test_gradient_error_falls_like_one_over_sqrt_b():
    X, y = regression_data()
    sizes = np.array([16, 64, 256, 1024])
    err = attempt(drills.gradient_error_vs_batch, X, y, sizes, 120,
                  np.random.default_rng(0))
    slope = np.polyfit(np.log(sizes), np.log(err), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.06), f"exponent was {slope:.3f}"


def test_newton_step_is_noisier_than_the_gradient_at_every_batch_size():
    """Section 5's measurement: even an exact H amplifies the gradient's error."""
    X, y = regression_data(cond=1e4)
    sizes = np.array([32, 128, 512, 2048])
    rng = np.random.default_rng(1)
    eg = attempt(drills.gradient_error_vs_batch, X, y, sizes, 120, rng)
    rng = np.random.default_rng(1)
    eh = attempt(drills.newton_step_error_vs_batch, X, y, sizes, 120, rng)
    assert np.all(eh > eg), f"gradient {eg}, newton {eh}"
    assert np.mean(eh / eg) > 2.0, f"amplification only {np.mean(eh / eg):.2f}x"


# ------------------------------------------------------------ symmetry ----

def test_relu_hyperbola_is_flat_to_machine_noise():
    """Not 'nearly equal' -- exactly equal. Section 7."""
    r = np.random.default_rng(3)
    x = np.linspace(0.2, 2.0, 60)
    y = 1.7 * np.maximum(1.1 * x, 0) + r.normal(scale=0.05, size=60)
    spread = attempt(drills.hyperbola_spread, x, y, 1.1 * 1.7,
                     np.linspace(0.4, 3.5, 120))
    assert spread < 1e-12, f"spread {spread:.2e} is too large to be rounding"


def test_relu_scaling_loss_is_invariant_under_the_symmetry():
    r = np.random.default_rng(4)
    x = np.linspace(0.3, 1.8, 40)
    y = r.normal(size=40)
    base = attempt(drills.relu_scaling_loss, x, y, 1.3, 0.9)
    for a in (0.25, 2.0, 7.5):
        assert drills.relu_scaling_loss(x, y, 1.3 * a, 0.9 / a) == pytest.approx(base)


# --------------------------------------------------------------- cliffs ----

def test_one_unclipped_step_destroys_the_run():
    L, w0, lr = 6, 1.45, 0.02
    j0 = float(attempt(drills.deep_product_loss, w0, L))
    g0 = float(attempt(drills.deep_product_grad, w0, L))
    w_plain = w0 - lr * g0
    assert float(drills.deep_product_loss(w_plain, L)) / j0 > 1e5


def test_clipping_keeps_you_on_the_cliff():
    L, w0, lr = 6, 1.45, 0.02
    j0 = float(attempt(drills.deep_product_loss, w0, L))
    g0 = float(attempt(drills.deep_product_grad, w0, L))
    w_clip = attempt(drills.clipped_step, np.array([w0]), np.array([g0]), lr, 1.0)
    j1 = float(attempt(drills.deep_product_loss, float(w_clip[0]), L))
    assert j1 < j0, "a clipped step should still go downhill"
    assert abs(float(w_clip[0]) - w0) <= lr * 1.0 + 1e-12


def test_clipping_preserves_direction():
    g = np.array([3.0, -4.0, 12.0])
    w = np.zeros(3)
    step = attempt(drills.clipped_step, w, g, 1.0, 1.0)
    cos = -step @ g / (np.linalg.norm(step) * np.linalg.norm(g))
    assert cos == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------- momentum, Nesterov ----

def test_nesterov_reaches_the_optimum_sooner():
    lam = np.array([1.0, 22.0])
    kw = dict(lr=0.03, alpha=0.9, steps=120)
    pm = attempt(drills.momentum_path, quadratic(lam), [-9.0, 3.2],
                 nesterov=False, **kw)
    pn = attempt(drills.momentum_path, quadratic(lam), [-9.0, 3.2],
                 nesterov=True, **kw)
    im = attempt(drills.iterations_to_reach, pm, 0.01)
    inn = attempt(drills.iterations_to_reach, pn, 0.01)
    assert inn is not None and im is not None
    assert inn < im, f"nesterov {inn}, momentum {im}"
    assert np.linalg.norm(pn[-1]) < np.linalg.norm(pm[-1])


def test_momentum_path_starts_at_theta0_and_has_the_right_length():
    p = attempt(drills.momentum_path, quadratic([1.0]), [2.0], lr=0.1,
                alpha=0.9, steps=5)
    assert p.shape == (6, 1)
    assert p[0] == pytest.approx([2.0])


# ------------------------------------------------------- initialisation ----

def test_fan_in_holds_the_forward_variance_and_loses_the_backward():
    widths = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    f, b = attempt(drills.variance_profile, widths, "fan_in", 512,
                   np.random.default_rng(1))
    assert f[-1] / f[0] == pytest.approx(1.0, rel=0.35)
    assert b[0] / b[-1] < 0.02, f"backward ratio {b[0] / b[-1]:.4f}"


def test_glorot_is_the_compromise():
    """Its worst drift is far smaller than either single-sided rule's."""
    widths = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    rng = lambda: np.random.default_rng(1)
    out = {}
    for rule in ("fan_in", "fan_out", "glorot"):
        f, b = attempt(drills.variance_profile, widths, rule, 512, rng())
        fr, br = f[-1] / f[0], b[0] / b[-1]
        out[rule] = max(fr, 1 / fr, br, 1 / br)
    assert out["glorot"] < out["fan_in"] / 3
    assert out["glorot"] < out["fan_out"] / 3


def test_output_bias_reproduces_the_class_frequencies():
    labels = np.array([0] * 97 + [1] * 3)
    b = attempt(drills.output_bias_for_marginal, labels, 2)
    p = np.exp(b) / np.exp(b).sum()
    assert p == pytest.approx([0.97, 0.03], abs=1e-9)


# --------------------------------------------------------- second order ----

def test_damped_newton_becomes_gradient_descent():
    """As alpha grows the step tends to -g/alpha. Drill C14."""
    A, b = spd(6, 100.0)
    g = b
    for alpha in (1e4, 1e6):
        step = attempt(drills.damped_newton_step, A, g, alpha)
        assert step == pytest.approx(-g / alpha, rel=0.02)


def test_damped_newton_solves_the_undamped_problem_at_alpha_zero():
    A, b = spd(5, 30.0)
    step = attempt(drills.damped_newton_step, A, b, 0.0)
    assert step == pytest.approx(-np.linalg.solve(A, b), rel=1e-8)


def test_cg_terminates_in_n_steps_when_well_conditioned():
    n = 12
    A, b = spd(n, 10.0)
    res = attempt(drills.conjugate_gradient_residuals, A, b, 2 * n)
    assert res[n] < 1e-10 * res[0], f"residual at n was {res[n]:.2e}"


def test_cg_finite_termination_is_lost_when_ill_conditioned():
    """The theorem is about exact arithmetic. Section 20's pitfall."""
    n = 12
    A, b = spd(n, 1000.0)
    res = attempt(drills.conjugate_gradient_residuals, A, b, 2 * n)
    assert res[n] > 1e-6 * res[0], "at kappa=1000 CG should NOT have terminated"
    assert res[-1] < 1e-10 * res[0], "but it should get there with more steps"


def test_cg_beats_steepest_descent():
    n = 12
    A, b = spd(n, 10.0)
    cg = attempt(drills.conjugate_gradient_residuals, A, b, 2 * n)
    sd = attempt(drills.steepest_descent_residuals, A, b, 2 * n)
    assert cg[-1] < sd[-1] / 1e6


# ------------------------------------------------ meta-algorithms ----------

def test_batchnorm_standardises_each_unit():
    r = np.random.default_rng(0)
    H = r.normal(size=(64, 5)) * np.array([1.0, 3.0, 0.2, 7.0, 0.5]) + 4.0
    out, _ = attempt(drills.batchnorm_forward, H, 1.0, 0.0)
    assert out.mean(axis=0) == pytest.approx(np.zeros(5), abs=1e-8)
    assert out.std(axis=0) == pytest.approx(np.ones(5), abs=1e-4)


def test_batchnorm_gamma_beta_restore_arbitrary_statistics():
    r = np.random.default_rng(1)
    H = r.normal(size=(64, 3))
    gamma, beta = np.array([2.0, 0.5, 3.0]), np.array([-1.0, 4.0, 0.0])
    out, _ = attempt(drills.batchnorm_forward, H, gamma, beta)
    assert out.mean(axis=0) == pytest.approx(beta, abs=1e-8)
    assert out.std(axis=0) == pytest.approx(gamma, abs=1e-4)


def test_batchnorm_gradient_cannot_ask_to_shift_the_mean():
    """The point of differentiating THROUGH mu: a request to move the mean
    of H produces no gradient, because the normalisation would undo it."""
    r = np.random.default_rng(2)
    H = r.normal(size=(48, 4))
    gamma, beta = np.ones(4), np.zeros(4)
    _, cache = attempt(drills.batchnorm_forward, H, gamma, beta)
    dout = np.ones((48, 4))          # "increase every output" == "raise the mean"
    dH, _, _ = attempt(drills.batchnorm_backward, dout, cache)
    assert np.abs(dH).max() < 1e-8, f"leaked gradient of size {np.abs(dH).max():.2e}"


def test_batchnorm_backward_matches_numerical_gradient():
    r = np.random.default_rng(3)
    H = r.normal(size=(20, 3))
    gamma, beta = r.normal(size=3), r.normal(size=3)
    dout = r.normal(size=(20, 3))

    def f(Hx):
        out, _ = drills.batchnorm_forward(Hx, gamma, beta)
        return float((out * dout).sum())

    _, cache = attempt(drills.batchnorm_forward, H, gamma, beta)
    dH, dgamma, dbeta = attempt(drills.batchnorm_backward, dout, cache)
    num = np.zeros_like(H)
    eps = 1e-6
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            Hp = H.copy(); Hp[i, j] += eps
            Hm = H.copy(); Hm[i, j] -= eps
            num[i, j] = (f(Hp) - f(Hm)) / (2 * eps)
    assert dH == pytest.approx(num, abs=1e-6)
    assert dbeta == pytest.approx(dout.sum(axis=0), rel=1e-8)


def test_coordinate_descent_cost_grows_like_one_over_alpha():
    counts = [attempt(drills.coordinate_descent_sweeps, a)
              for a in (0.1, 0.01, 0.001)]
    assert counts[0] < counts[1] < counts[2]
    for a, b in zip(counts, counts[1:]):
        assert 5 < b / a < 20, f"ratio {b / a:.1f} is not roughly tenfold"


def test_polyak_beats_the_iterate_on_a_noisy_quadratic():
    lam = np.array([1.0, 45.0])
    rng = np.random.default_rng(0)
    p = np.array([-8.0, 3.0])
    path = [p.copy()]
    for _ in range(800):
        p = p - 0.043 * (lam * p + rng.normal(scale=4.0, size=2))
        path.append(p.copy())
    path = np.array(path)
    avg = attempt(drills.polyak_average, path)
    raw = np.linalg.norm(path[-150:], axis=1).mean()
    run = np.linalg.norm(avg[-150:], axis=1).mean()
    assert run < raw / 3, f"raw {raw:.3f}, averaged {run:.3f}"


def test_polyak_ema_forgets():
    """An EMA tracks a moved optimum; a plain mean does not."""
    path = np.concatenate([np.zeros((400, 1)), np.full((400, 1), 10.0)])
    mean = attempt(drills.polyak_average, path)
    ema = attempt(drills.polyak_average, path, alpha=0.98)
    assert abs(float(ema[-1]) - 10.0) < 0.5
    assert abs(float(mean[-1]) - 5.0) < 0.5


def test_blurring_a_quadratic_only_shifts_it():
    """Section 28's second failure mode, as arithmetic.

    The grid must be far wider than 5*sigma or the kernel's edge padding
    dominates and the measurement is meaningless -- which is exactly how
    Figure 11 was got wrong the first time.
    """
    th = np.linspace(-24, 24, 9601)
    dx = th[1] - th[0]
    v = -0.5 * th ** 2
    b = attempt(drills.gaussian_blur, v, dx, 2.5)
    mid = np.abs(th) < 6
    diff = (b - v)[mid]
    assert diff.std() < 1e-9, f"blur was not a constant shift (std {diff.std():.2e})"
    # 1e-3 rather than machine precision: the kernel is truncated at 5 sigma.
    # It is still tight enough to reject the ~-2.58 a too-narrow grid gives.
    assert diff.mean() == pytest.approx(-0.5 * 2.5 ** 2, abs=1e-3)
    c2 = np.gradient(np.gradient(b, dx), dx)[len(b) // 2]
    assert c2 == pytest.approx(-1.0, abs=1e-4), "blurring must not change the curvature"


def test_continuation_finds_the_global_minimum_where_descent_does_not():
    th = np.linspace(-24, 24, 9601)
    dx = th[1] - th[0]
    J = 0.10 * th ** 2 + 1.5 * np.cos(2.2 * th) + 0.6 * np.sin(0.9 * th) + 0.35 * th
    start = 5.5
    stuck = attempt(drills.local_descent, J, th, start)
    cur = start
    for s in (3.0, 1.6, 0.8, 0.0):
        v = attempt(drills.gaussian_blur, J, dx, s)
        cur = th[attempt(drills.local_descent, v, th, cur)]
    final = drills.local_descent(J, th, cur)
    assert J[final] < J[stuck] - 1.0
    assert final == int(np.argmin(J)), "continuation should reach the global minimum"
