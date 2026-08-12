"""
Tests for the Part VII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

The decisive tests are the gradient checks. An analytic gradient that matches
central differences to 1e-6 is almost certainly right; one that does not is
certainly wrong, and no amount of reasoning about the algebra substitutes.
"""
import math

import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{getattr(fn, '__name__', fn)} not implemented yet")


def V(x):
    try:
        return drills.Value(x)
    except NotImplementedError:
        pytest.skip("Value not implemented yet")


def run(thunk):
    """Evaluate a thunk, skipping if any Value operation is still a stub.

    Value's __init__ works before the operators do, so V() alone cannot tell
    us whether the engine is finished -- the NotImplementedError surfaces
    later, inside __mul__ or backward(). Every Value test therefore runs its
    body through here.
    """
    try:
        return thunk()
    except NotImplementedError as e:
        pytest.skip(f"{e}")


rng = np.random.default_rng(0)


def make_params(d=3, h=4, seed=0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 0.6, (h, d)), "b1": r.normal(0, 0.1, h),
            "W2": r.normal(0, 0.6, (1, h)), "b2": r.normal(0, 0.1, 1)}


# --------------------------------------------------------- the primitives ---

def test_relu_and_its_derivative():
    z = np.array([-2.0, -0.5, 0.5, 3.0])
    assert np.allclose(attempt(drills.relu, z), [0, 0, 0.5, 3.0])
    assert np.allclose(attempt(drills.relu_grad, z), [0, 0, 1, 1])


def test_sigmoid_is_stable_and_correct():
    assert np.isclose(attempt(drills.sigmoid, np.array([0.0]))[0], 0.5)
    out = attempt(drills.sigmoid, np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(out)), "naive 1/(1+exp(-z)) overflows at -800"
    assert out[0] >= 0.0 and out[1] <= 1.0


def test_sigmoid_derivative_peaks_at_a_quarter():
    z = np.linspace(-6, 6, 400)
    g = attempt(drills.sigmoid_grad, z)
    assert np.isclose(g.max(), 0.25, atol=1e-4), "the 0.25 ceiling of §18"
    assert np.isclose(attempt(drills.sigmoid_grad, np.array([0.0]))[0], 0.25)


def test_init_scale_has_the_factor_of_two_for_relu():
    lin = attempt(drills.init_scale, 100, "linear")
    rel = attempt(drills.init_scale, 100, "relu")
    assert np.isclose(lin, 0.1)
    assert np.isclose(rel / lin, math.sqrt(2.0), rtol=1e-6), \
        "ReLU zeroes half the units, so the variance needs doubling"


# ----------------------------------------------------- forward / backward ---

def test_forward_shapes_and_range():
    p = make_params()
    z1, a1, z2, y_hat = attempt(drills.forward, p, np.array([1.0, 2.0, -1.0]))
    assert z1.shape == (4,) and a1.shape == (4,)
    assert np.all(a1 >= 0), "a1 is post-ReLU"
    assert isinstance(z2, float) and 0.0 < y_hat < 1.0


def test_backward_shapes_match_the_parameters():
    """The invariant of §15: dL/dW has the shape of W."""
    p = make_params()
    x, y = np.array([0.5, -1.0, 2.0]), 1.0
    g = attempt(drills.backward, p, x, y, attempt(drills.forward, p, x))
    for k in p:
        assert g[k].shape == np.shape(p[k]), f"{k}: {g[k].shape} vs {np.shape(p[k])}"


def test_backward_matches_numerical_gradient():
    """The decisive test. If this passes, the four equations are right."""
    p = make_params(seed=3)
    x, y = np.array([0.7, -0.4, 1.1]), 1.0

    def loss_of(params):
        y_hat = attempt(drills.forward, params, x)[3]
        y_hat = min(max(y_hat, 1e-12), 1 - 1e-12)
        return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    analytic = attempt(drills.backward, p, x, y, attempt(drills.forward, p, x))
    numeric = attempt(drills.numerical_gradient, loss_of,
                      {k: np.array(v, dtype=float) for k, v in p.items()}, 1e-6)
    for k in p:
        assert np.allclose(analytic[k], numeric[k], atol=1e-5), \
            f"gradient mismatch on {k}"


def test_backward_works_for_the_zero_target_too():
    p = make_params(seed=4)
    x, y = np.array([-1.2, 0.3, 0.8]), 0.0

    def loss_of(params):
        y_hat = attempt(drills.forward, params, x)[3]
        y_hat = min(max(y_hat, 1e-12), 1 - 1e-12)
        return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    analytic = attempt(drills.backward, p, x, y, attempt(drills.forward, p, x))
    numeric = attempt(drills.numerical_gradient, loss_of,
                      {k: np.array(v, dtype=float) for k, v in p.items()}, 1e-6)
    for k in p:
        assert np.allclose(analytic[k], numeric[k], atol=1e-5)


def test_output_delta_is_prediction_minus_truth():
    """Equation 1: the sigma' factor cancels, leaving y_hat - y."""
    p = make_params(seed=5)
    x, y = np.array([0.2, 0.4, -0.6]), 1.0
    cache = attempt(drills.forward, p, x)
    g = attempt(drills.backward, p, x, y, cache)
    assert np.allclose(g["b2"].ravel()[0], cache[3] - y, atol=1e-9)


# ------------------------------------------------------------ the engine ----

def test_value_add_and_mul():
    def body():
        a, b = V(2.0), V(-3.0)
        c = a * b + a
        c.backward()
        return a, b, c
    a, b, c = run(body)
    assert np.isclose(c.data, -4.0)
    assert np.isclose(a.grad, -2.0)      # d(ab + a)/da = b + 1
    assert np.isclose(b.grad, 2.0)       # d(ab + a)/db = a


def test_value_reuse_accumulates():
    """The smallest case that distinguishes += from =. y = x*x gives 2x."""
    def body():
        x = V(3.0)
        y = x * x
        y.backward()
        return x
    x = run(body)
    assert np.isclose(x.grad, 6.0), \
        "got x instead of 2x -- your _backward assigns where it must accumulate"


def test_value_reuse_deeper():
    def body():
        x = V(2.0)
        y = (x + x) * x       # 2x^2 -> dy/dx = 4x = 8
        y.backward()
        return x
    assert np.isclose(run(body).grad, 8.0)


def test_value_pow_neg_sub_div():
    def body():
        a, b = V(4.0), V(2.0)
        c = a / b - b ** 2
        c.backward()
        return a, b, c
    a, b, c = run(body)
    assert np.isclose(c.data, -2.0)
    assert np.isclose(a.grad, 0.5)                  # d(a/b)/da = 1/b
    assert np.isclose(b.grad, -a.data / b.data ** 2 - 2 * b.data)


def test_value_relu_tanh_exp_log():
    for fn, x0, expect_d in [
        ("relu", 2.0, 1.0),
        ("relu", -2.0, 0.0),
        ("tanh", 0.5, 1 - math.tanh(0.5) ** 2),
        ("exp", 1.0, math.e),
        ("log", 4.0, 0.25),
    ]:
        def body(fn=fn, x0=x0):
            x = V(x0)
            getattr(x, fn)().backward()
            return x
        assert np.isclose(run(body).grad, expect_d, atol=1e-9), f"{fn} at {x0}"


def test_value_against_numerical_gradient_on_a_messy_expression():
    """Ten random expressions, each reusing a value. This is the project's
    stage-1 deliverable in miniature."""
    def f(vals):
        a, b, c = vals
        return (a * b + c.relu()) * (a + c) + (b ** 2) / (a + 3.0)

    r = np.random.default_rng(11)
    for _ in range(10):
        x0 = r.normal(size=3)

        def body(x0=x0):
            vs = [V(v) for v in x0]
            f(vs).backward()
            return np.array([v.grad for v in vs])
        analytic = run(body)

        h, numeric = 1e-6, np.zeros(3)
        for i in range(3):
            hi, lo = x0.copy(), x0.copy()
            hi[i] += h
            lo[i] -= h
            numeric[i] = run(lambda hi=hi, lo=lo: (f([V(v) for v in hi]).data
                                                   - f([V(v) for v in lo]).data) / (2 * h))
        assert np.allclose(analytic, numeric, atol=1e-4), \
            f"analytic {analytic} vs numeric {numeric}"


def test_value_topological_order_is_respected():
    """A node's gradient is only final once every consumer has contributed."""
    def body():
        x = V(1.5)
        shared = x * 2.0
        out = shared * shared + shared  # uses `shared` three times
        out.backward()
        return x, shared
    x, shared = run(body)
    s = shared.data
    assert np.isclose(shared.grad, 2 * s + 1)
    assert np.isclose(x.grad, (2 * s + 1) * 2.0)


# --------------------------------------------------------- the pathologies --

def test_sigmoid_gradients_vanish_with_depth():
    """Figure 10: with sigmoids the early layers receive essentially nothing."""
    g = attempt(drills.gradient_by_layer, 20, "sigmoid", 16, 0)
    assert len(g) == 20
    assert g[0] < g[-1] / 1000.0, \
        f"input layer {g[0]:.3e} vs output layer {g[-1]:.3e} -- expected collapse"


def test_relu_gradients_survive_depth():
    g = attempt(drills.gradient_by_layer, 20, "relu", 16, 0)
    ratio = g[-1] / max(g[0], 1e-30)
    assert ratio < 1e3, \
        f"ReLU with matched init should hold roughly constant, got {ratio:.1e}"


def test_relu_beats_sigmoid_at_depth():
    s = attempt(drills.gradient_by_layer, 20, "sigmoid", 16, 1)
    r = attempt(drills.gradient_by_layer, 20, "relu", 16, 1)
    assert (r[-1] / max(r[0], 1e-30)) < (s[-1] / max(s[0], 1e-30))


def test_count_dead_units():
    X = rng.normal(size=(50, 3))
    W = rng.normal(size=(6, 3))
    b = np.zeros(6)
    assert 0 <= attempt(drills.count_dead_units, W, b, X) <= 6
    b_dead = np.full(6, -50.0)          # every unit driven far negative
    assert attempt(drills.count_dead_units, W, b_dead, X) == 6
    b_alive = np.full(6, 50.0)
    assert attempt(drills.count_dead_units, W, b_alive, X) == 0


def test_xor_is_learned():
    """The four points no line separates."""
    params, loss = attempt(drills.train_xor, 4, 4000, 0.5, 0)
    assert loss < 0.05, f"final loss {loss:.4f} -- XOR should be solvable"
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])
    preds = [attempt(drills.forward, params, xi)[3] for xi in X]
    assert all((p > 0.5) == (t > 0.5) for p, t in zip(preds, y)), \
        f"predictions {np.round(preds, 3)} do not match {y}"


def test_xor_needs_the_hidden_layer():
    """A linear model cannot do it, at any setting of its parameters."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])
    Xb = np.hstack([X, np.ones((4, 1))])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    preds = Xb @ w
    assert not all((p > 0.5) == (t > 0.5) for p, t in zip(preds, y)), \
        "if a line separated XOR, Part VII would not exist"
