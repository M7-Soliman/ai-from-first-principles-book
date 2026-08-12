"""
Tests for the Part XI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Where this part claims an identity, the test demands exactness — truncation at
full length equals the full gradient; norm clipping rotates by exactly zero;
the leaky weights sum to one. Where the book reports a measurement, the test
asserts the inequality that measurement established.
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


def toy(T=7, n_in=3, n_h=5, n_out=4, seed=1):
    r = np.random.default_rng(seed)
    return dict(
        x=r.normal(size=(T, n_in)), y=r.integers(0, n_out, T),
        U=r.normal(size=(n_h, n_in)) * 0.5, W=r.normal(size=(n_h, n_h)) * 0.5,
        V=r.normal(size=(n_out, n_h)) * 0.5, b=r.normal(size=n_h) * 0.1,
        c=r.normal(size=n_out) * 0.1, h0=np.zeros(n_h))


# --------------------------------------------------------------- forward ----

def test_forward_shapes_and_a_hand_computed_step():
    t = toy()
    H, O = attempt(drills.rnn_forward, t["x"], t["U"], t["W"], t["V"],
                   t["b"], t["c"], t["h0"])
    assert H.shape == (7, 5) and O.shape == (7, 4)
    h1 = np.tanh(t["b"] + t["W"] @ t["h0"] + t["U"] @ t["x"][0])
    assert H[0] == pytest.approx(h1, abs=1e-12)
    assert O[0] == pytest.approx(t["c"] + t["V"] @ h1, abs=1e-12)


def test_softmax_is_stable_and_normalised():
    z = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]])
    p = attempt(drills.softmax_rows, z)
    assert np.isfinite(p).all(), "overflowed — subtract the row max first"
    assert p.sum(1) == pytest.approx([1.0, 1.0])
    assert p[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_loss_matches_the_definition():
    t = toy()
    _, O = attempt(drills.rnn_forward, t["x"], t["U"], t["W"], t["V"],
                   t["b"], t["c"], t["h0"])
    L = attempt(drills.sequence_loss, O, t["y"])
    P = drills.softmax_rows(O)
    assert L == pytest.approx(float(-np.log(P[np.arange(7), t["y"]]).mean()))


# -------------------------------------------------------------- backward ----

def test_bptt_matches_finite_differences():
    t = toy()
    loss, g = attempt(drills.bptt, t["x"], t["y"], t["U"], t["W"], t["V"],
                      t["b"], t["c"], t["h0"])

    def J(**over):
        p = dict(U=t["U"], W=t["W"], V=t["V"], b=t["b"], c=t["c"])
        p.update(over)
        _, O = drills.rnn_forward(t["x"], p["U"], p["W"], p["V"], p["b"],
                                  p["c"], t["h0"])
        return drills.sequence_loss(O, t["y"])

    assert loss == pytest.approx(J())
    eps = 1e-6
    for name, arr, grad in (("U", t["U"], g["dU"]), ("W", t["W"], g["dW"]),
                            ("V", t["V"], g["dV"]), ("b", t["b"], g["db"]),
                            ("c", t["c"], g["dc"])):
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in arr.shape)
            up, dn = arr.copy(), arr.copy()
            up[i] += eps
            dn[i] -= eps
            num = (J(**{name: up}) - J(**{name: dn})) / (2 * eps)
            assert grad[i] == pytest.approx(num, abs=2e-6), f"d{name} at {i}"


def test_dW_accumulates_over_every_step():
    """The classic bug is computing dW at one step instead of summing."""
    t = toy(T=12)
    _, g_full = attempt(drills.bptt, t["x"], t["y"], t["U"], t["W"], t["V"],
                        t["b"], t["c"], t["h0"])
    _, g_one = drills.bptt(t["x"][:1], t["y"][:1], t["U"], t["W"], t["V"],
                           t["b"], t["c"], t["h0"])
    assert np.linalg.norm(g_full["dW"]) > 2 * np.linalg.norm(g_one["dW"])


def test_truncation_at_full_length_is_exact():
    t = toy(T=9)
    _, a = attempt(drills.bptt, t["x"], t["y"], t["U"], t["W"], t["V"],
                   t["b"], t["c"], t["h0"], k=None)
    _, b = drills.bptt(t["x"], t["y"], t["U"], t["W"], t["V"], t["b"],
                       t["c"], t["h0"], k=9)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-12), key


def test_truncation_is_biased_and_the_direction_recovers_first():
    t = toy(T=25, n_h=8)
    _, full = attempt(drills.bptt, t["x"], t["y"], t["U"], t["W"], t["V"],
                      t["b"], t["c"], t["h0"])
    prev_cos = -2.0
    for k in (1, 2, 5, 10, 25):
        _, g = drills.bptt(t["x"], t["y"], t["U"], t["W"], t["V"], t["b"],
                           t["c"], t["h0"], k=k)
        a, f = g["dW"].ravel(), full["dW"].ravel()
        cos = float(a @ f / (np.linalg.norm(a) * np.linalg.norm(f)))
        assert cos >= prev_cos - 1e-9, f"cosine went backwards at k={k}"
        prev_cos = cos
    assert prev_cos == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------- gradients over time ----

def test_linear_jacobian_product_is_the_spectral_radius_to_the_t():
    r = np.random.default_rng(2)
    Q = np.linalg.qr(r.normal(size=(6, 6)))[0]
    for rho in (0.8, 1.0, 1.2):
        lam = np.linspace(0.3 * rho, rho, 6)
        W = Q @ np.diag(lam) @ Q.T
        norms = attempt(drills.jacobian_product_norm, W, 30, "linear")
        assert len(norms) == 31
        assert norms[30] == pytest.approx(rho ** 30, rel=1e-6)


def test_tanh_still_vanishes_and_can_still_explode():
    d = 24
    r = np.random.default_rng(3)
    small = r.normal(size=(d, d)) * (0.4 / np.sqrt(d))
    big = r.normal(size=(d, d)) * (3.0 / np.sqrt(d))
    a = attempt(drills.jacobian_product_norm, small, 40, "tanh")
    b = attempt(drills.jacobian_product_norm, big, 40, "tanh")
    assert a[-1] < 1e-6, f"should vanish, got {a[-1]:.2e}"
    assert b[-1] > a[-1] * 1e6, "a bounded nonlinearity does not bound the Jacobian"


def test_scale_to_radius():
    r = np.random.default_rng(4)
    W = r.normal(size=(7, 7))
    for rho in (0.5, 0.95, 1.5):
        assert attempt(drills.spectral_radius,
                       attempt(drills.scale_to_radius, W, rho)) == pytest.approx(rho)


# --------------------------------------------------------------- clipping ----

def test_norm_clipping_rotates_by_exactly_zero():
    g = np.array([3.0, -4.0, 12.0])
    assert attempt(drills.clip_angle, g, 1.0, "norm") == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(attempt(drills.clip_by_norm, g, 1.0)) == pytest.approx(1.0)


def test_norm_clipping_leaves_small_gradients_alone():
    g = np.array([0.1, 0.2])
    assert attempt(drills.clip_by_norm, g, 10.0) == pytest.approx(g)


def test_elementwise_clipping_rotates():
    g = np.array([10.0, 0.1, 0.05])
    ang = attempt(drills.clip_angle, g, 1.0, "elementwise")
    assert ang > 0.05, f"element-wise clipping should rotate, got {ang:.4f} rad"


# ------------------------------------------------------------ leaky units ----

def test_leaky_weights_sum_to_one_and_decay_geometrically():
    w = attempt(drills.leaky_weights, 0.9, 400)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert w[1] / w[0] == pytest.approx(0.9)


def test_effective_memory():
    for a, want in ((0.5, 2.0), (0.9, 10.0), (0.98, 50.0), (0.99, 100.0)):
        assert attempt(drills.effective_memory, a) == pytest.approx(want)


# ------------------------------------------------------------------- LSTM ----

def _lstm_params(n_in, n_h, seed=5, forget_bias=0.0):
    r = np.random.default_rng(seed)
    p = {}
    for k in "fgoc":
        p["W" + k] = r.normal(size=(n_h, n_in)) * 0.3
        p["U" + k] = r.normal(size=(n_h, n_h)) * 0.3
        p["b" + k] = np.zeros(n_h)
    p["bf"] = np.full(n_h, forget_bias)
    return p


def test_sigmoid_is_stable():
    z = np.array([-1000.0, 0.0, 1000.0])
    s = attempt(drills.sigmoid, z)
    assert np.isfinite(s).all()
    assert s == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)


def test_forget_bias_sets_the_gate_at_initialisation():
    """Section 20's mechanism, in one line."""
    p0 = _lstm_params(2, 6, forget_bias=0.0)
    p1 = _lstm_params(2, 6, forget_bias=1.0)
    x, h, s = np.zeros(2), np.zeros(6), np.zeros(6)
    _, _, g0 = attempt(drills.lstm_step, x, h, s, p0)
    _, _, g1 = attempt(drills.lstm_step, x, h, s, p1)
    assert g0["f"].mean() == pytest.approx(0.5, abs=1e-9)
    assert g1["f"].mean() == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-9)


def test_a_closed_forget_gate_erases_the_cell():
    p = _lstm_params(2, 6, forget_bias=-30.0)      # gate ~ 0
    for k in "goc":
        p["W" + k][:] = 0.0
        p["U" + k][:] = 0.0
    x, h = np.zeros(2), np.zeros(6)
    s = np.ones(6) * 5.0
    _, s_new, g = attempt(drills.lstm_step, x, h, s, p)
    assert g["f"].mean() < 1e-10
    assert np.abs(s_new).max() < 1e-8, "a closed forget gate must clear the cell"


def test_lstm_run_shapes():
    p = _lstm_params(3, 4)
    H, S, G = attempt(drills.lstm_run, np.zeros((20, 3)), p, np.zeros(4), np.zeros(4))
    assert H.shape == (20, 4) and S.shape == (20, 4) and len(G) == 20


def test_open_forget_gate_holds_only_when_the_input_gate_is_shut():
    """Section 19's finding: an open forget gate makes the cell INTEGRATE.
    It holds exactly only when nothing is being added."""
    p = _lstm_params(3, 4, forget_bias=30.0)          # forget ~ 1
    X = np.zeros((20, 3))

    # input gate open: the cell drifts, because it keeps accumulating
    _, S_open, _ = attempt(drills.lstm_run, X, p, np.zeros(4), np.ones(4))
    drift = float(np.abs(S_open[-1] - 1.0).max())

    # input gate shut: the cell holds its initial value exactly
    p2 = dict(p)
    p2["bg"] = np.full(4, -30.0)
    _, S_shut, _ = drills.lstm_run(X, p2, np.zeros(4), np.ones(4))
    assert S_shut[-1] == pytest.approx(np.ones(4), abs=1e-9), "nothing added, so nothing changes"
    assert drift > 0.1, ("with the input gate open the cell should integrate, "
                         f"but it moved only {drift:.3f}")


def test_retention_arithmetic():
    """Section 19's comparison: 0.905 per step against 0.5 per step."""
    a = attempt(drills.retention, 0.905, 29)
    b = attempt(drills.retention, 0.5, 29)
    assert a == pytest.approx(0.0549, abs=5e-4)
    assert b < 2e-9
    assert a / b > 1e6


# ------------------------------------------------------------- the counts ----

def test_parameter_counts():
    assert attempt(drills.rnn_params, 1, 64, 10) == 64 * 1 + 64 * 64 + 10 * 64 + 64 + 10
    assert attempt(drills.table_params, 10, 20) == pytest.approx(1e20)
    assert attempt(drills.unshared_params, 1, 64, 10, 20) == \
        20 * drills.rnn_params(1, 64, 10)


def test_sharing_saves_exactly_tau():
    for tau in (5, 20, 100):
        assert (attempt(drills.unshared_params, 3, 32, 8, tau)
                / attempt(drills.rnn_params, 3, 32, 8)) == pytest.approx(tau)


# ------------------------------------------- movement G: what came after

def test_the_cost_crossover_is_the_model_dimension():
    for d in (256, 512, 4096):
        L = attempt(drills.cost_crossover, d)
        assert L == pytest.approx(d)
        a = attempt(drills.attention_flops, L, d)
        b = attempt(drills.linear_flops, L, d)
        assert a == pytest.approx(b, rel=1e-9)


def test_attention_is_cheaper_below_the_crossover():
    d = 1024
    assert attempt(drills.attention_flops, d // 4, d) < attempt(drills.linear_flops, d // 4, d)
    assert attempt(drills.attention_flops, d * 4, d) > attempt(drills.linear_flops, d * 4, d)


def test_generation_cost_grows_for_attention_and_not_for_a_state():
    d = 512
    a = [attempt(drills.generation_flops, L, d, "attention") for L in (1024, 8192, 65536)]
    r = [attempt(drills.generation_flops, L, d, "recurrent") for L in (1024, 8192, 65536)]
    assert a[0] < a[1] < a[2]
    assert r[0] == r[1] == r[2]
    assert a[2] / r[2] > 40


def test_linear_attention_two_ways_agree_exactly():
    """Reassociating a sum is an identity, not an approximation. If these
    disagree the usual cause is the normaliser, which is its own running sum."""
    rng = np.random.default_rng(0)
    Q, K, V = (rng.normal(size=(24, 8)), rng.normal(size=(24, 8)), rng.normal(size=(24, 5)))
    a = attempt(drills.linear_attention_quadratic, Q, K, V)
    b = attempt(drills.linear_attention_recurrent, Q, K, V)
    assert np.allclose(a, b, atol=1e-10)


def test_linear_attention_rows_are_a_convex_combination_of_values():
    rng = np.random.default_rng(1)
    Q, K = rng.normal(size=(16, 6)), rng.normal(size=(16, 6))
    V = rng.normal(size=(16, 4))
    out = attempt(drills.linear_attention_quadratic, Q, K, V)
    assert out.shape == (16, 4)
    assert (out.max() <= V.max() + 1e-9) and (out.min() >= V.min() - 1e-9)


def test_the_two_scans_agree():
    rng = np.random.default_rng(2)
    for L in (1, 2, 7, 64, 257):
        a = rng.uniform(0.7, 0.99, L)
        b = rng.normal(size=L)
        assert np.allclose(attempt(drills.scan_sequential, a, b),
                           attempt(drills.scan_associative, a, b), atol=1e-9)


def test_the_scan_reduces_to_a_cumulative_sum():
    b = np.arange(1.0, 9.0)
    a = np.ones(8)
    assert np.allclose(attempt(drills.scan_associative, a, b), np.cumsum(b))


def test_scan_depth_is_logarithmic():
    assert attempt(drills.scan_depth, 1024, True) == 10
    assert attempt(drills.scan_depth, 1024, False) == 1024
    assert attempt(drills.scan_depth, 1 << 20, True) == 20


def test_discretisation_matches_the_matrix_exponential():
    A = np.array([-1.0, -0.25, -4.0])
    B = np.array([1.0, 2.0, 0.5])
    Abar, Bbar = attempt(drills.discretise, A, B, 0.1)
    assert np.allclose(Abar, np.exp(0.1 * A))
    assert np.allclose(Bbar, (np.exp(0.1 * A) - 1) / A * B)


def test_a_small_step_makes_the_discrete_system_approach_the_continuous_one():
    A, B = np.array([-2.0]), np.array([1.0])
    for dt in (1e-2, 1e-3, 1e-4):
        Abar, Bbar = attempt(drills.discretise, A, B, dt)
        tol = (dt * abs(A[0])) ** 2        # the second-order term, not dt^2
        assert Abar[0] == pytest.approx(1 + dt * A[0], abs=tol)
        assert Bbar[0] == pytest.approx(dt * B[0], abs=tol)


def test_the_ssm_kernel_is_a_decaying_convolution():
    Abar, Bbar, C = np.array([0.5]), np.array([1.0]), np.array([2.0])
    k = attempt(drills.ssm_kernel, Abar, Bbar, C, 5)
    assert np.allclose(k, [2.0, 1.0, 0.5, 0.25, 0.125])


def test_the_ssm_kernel_convolution_equals_the_recurrence():
    rng = np.random.default_rng(3)
    Abar, Bbar, C = np.array([0.8]), np.array([0.5]), np.array([1.5])
    L = 32
    x = rng.normal(size=L)
    k = attempt(drills.ssm_kernel, Abar, Bbar, C, L)
    conv = np.array([np.dot(k[:t + 1][::-1], x[:t + 1]) for t in range(L)])
    h, out = 0.0, []
    for t in range(L):
        h = Abar[0] * h + Bbar[0] * x[t]
        out.append(C[0] * h)
    assert np.allclose(conv, out, atol=1e-9)


def test_the_half_life_is_exact():
    for lam in (0.5, 0.9, 0.98, 0.999):
        h = attempt(drills.half_life, lam)
        assert lam ** h == pytest.approx(0.5)
        assert h < 1.0 / (1 - lam)          # not 1/(1-lambda)


def test_the_familiar_formula_is_a_limit_not_an_identity():
    """ln2/(1-lam) is 5% high at lam=0.9 and converges as lam -> 1. Asserting
    equality is the natural mistake; asserting convergence is the claim."""
    errs = [abs(attempt(drills.half_life, lam) - np.log(2) / (1 - lam))
            / attempt(drills.half_life, lam) for lam in (0.9, 0.99, 0.999)]
    assert errs[0] > 0.04
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.001


def test_a_perfect_gate_recovers_the_last_marked_value():
    rng = np.random.default_rng(4)
    L = 64
    x = rng.normal(size=L)
    mark = np.zeros(L, dtype=bool)
    mark[[3, 11, 40]] = True
    gate = np.where(mark, 0.0, 1.0)
    h = attempt(drills.selective_scan, x, gate)
    assert h[-1] == pytest.approx(x[40])


def test_no_fixed_decay_recovers_it():
    rng = np.random.default_rng(5)
    L = 64
    x = rng.normal(size=L)
    mark = np.zeros(L, dtype=bool)
    mark[[3, 11, 40]] = True
    for lam in (0.5, 0.9, 0.99):
        h = attempt(drills.selective_scan, x, np.full(L, lam))
        assert abs(h[-1] - x[40]) > 1e-3


def test_the_hybrid_cache_scales_with_attention_layers_not_depth():
    a = attempt(drills.hybrid_cache, 32, 4, 512, 8192)
    b = attempt(drills.hybrid_cache, 64, 4, 512, 8192)
    c = attempt(drills.hybrid_cache, 32, 32, 512, 8192)
    assert a == b
    assert c == 8 * a
