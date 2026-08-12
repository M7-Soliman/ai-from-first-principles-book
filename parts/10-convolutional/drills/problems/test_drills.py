"""
Tests for the Part X drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Where this part claims an exact result, the test demands exactness rather than a
tolerance: convolution equals a matrix multiplication, the input gradient equals
a flipped full convolution, the separable ratio is an identity, and the
checkerboard condition is arithmetic. A tolerance in those places would hide the
point.
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


def rand(*shape, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    return r.normal(size=shape)


# ----------------------------------------------------------- the operation ----

def test_output_size_matches_the_special_cases():
    assert attempt(drills.output_size, 32, 3) == 30                  # valid
    assert attempt(drills.output_size, 32, 3, padding=1) == 32       # same
    assert attempt(drills.output_size, 32, 5, padding=4) == 36       # full
    assert attempt(drills.output_size, 32, 5, padding=2, stride=2) == 16
    assert attempt(drills.output_size, 32, 3, padding=2, dilation=2) == 32


def test_conv_is_a_matrix_multiplication():
    K, x = rand(3, 3, seed=1), rand(8, 8, seed=2)
    M, (oh, ow) = attempt(drills.conv_matrix, K, 8, 8)
    assert M.shape == (oh * ow, 64)
    via_mat = (M @ x.ravel()).reshape(oh, ow)
    via_conv = attempt(drills.conv2d_naive, x[None, None], K[None, None])[0, 0]
    assert via_mat == pytest.approx(via_conv, abs=1e-12)


def test_the_matrix_has_few_distinct_values():
    """324 nonzeros taking 9 values: sparsity and sharing, separately."""
    K = rand(3, 3, seed=3)
    M, _ = attempt(drills.conv_matrix, K, 8, 8)
    nz = M[M != 0]
    assert M.size == 2304
    assert len(nz) == 324
    assert len(np.unique(np.round(nz, 12))) == 9


def test_im2col_agrees_with_the_loop_everywhere():
    for stride, padding in ((1, 0), (1, 1), (2, 0), (2, 1), (3, 2)):
        x, k = rand(4, 3, 12, 12, seed=4), rand(6, 3, 3, 3, seed=5)
        a = attempt(drills.conv2d_naive, x, k, stride=stride, padding=padding)
        b = attempt(drills.conv2d_im2col, x, k, stride=stride, padding=padding)
        assert a.shape == b.shape, f"shape mismatch at stride={stride} pad={padding}"
        assert a == pytest.approx(b, abs=1e-10), f"stride={stride} pad={padding}"


def test_im2col_buffer_is_k_squared_times_the_input():
    x = rand(2, 3, 10, 10, seed=6)
    cols = attempt(drills.im2col, x, 3, 3)
    assert cols.shape == (2, 3 * 9, 8 * 8)
    assert cols.size / x.size == pytest.approx(9 * 64 / 100, rel=1e-9)


# -------------------------------------------------------------- backward ----

def test_input_gradient_is_a_flipped_full_convolution():
    """Section 16's exact claim, so demand exactness."""
    x, k = rand(2, 3, 9, 9, seed=7), rand(5, 3, 3, 3, seed=8)
    z = attempt(drills.conv2d_naive, x, k)
    g = rand(*z.shape, seed=9)
    dx = attempt(drills.conv_backward_input, g, k, x.shape)
    # the same thing as a full convolution with the flipped, transposed kernel
    k_flip = k[:, :, ::-1, ::-1].transpose(1, 0, 2, 3).copy()
    dx2 = attempt(drills.conv2d_naive, g, k_flip, padding=2)
    assert dx == pytest.approx(dx2, abs=1e-10)


def test_gradients_match_finite_differences():
    for stride, padding in ((1, 0), (1, 1), (2, 1)):
        x = rand(2, 2, 7, 7, seed=10)
        k = rand(3, 2, 3, 3, seed=11)
        z = attempt(drills.conv2d_naive, x, k, stride=stride, padding=padding)
        g = rand(*z.shape, seed=12)

        def J(xx, kk):
            return float((drills.conv2d_naive(xx, kk, stride=stride,
                                              padding=padding) * g).sum())

        dx = attempt(drills.conv_backward_input, g, k, x.shape,
                     stride=stride, padding=padding)
        dk = attempt(drills.conv_backward_kernel, g, x, k.shape,
                     stride=stride, padding=padding)
        eps = 1e-6
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in x.shape)
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            num = (J(xp, k) - J(xm, k)) / (2 * eps)
            assert dx[i] == pytest.approx(num, abs=1e-5), f"dx at {i}, stride={stride}"
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in k.shape)
            kp, km = k.copy(), k.copy()
            kp[i] += eps
            km[i] -= eps
            num = (J(x, kp) - J(x, km)) / (2 * eps)
            assert dk[i] == pytest.approx(num, abs=1e-5), f"dk at {i}, stride={stride}"


# -------------------------------------------------------------- pooling ----

def test_max_pool_shapes_and_values():
    x = np.arange(32, dtype=float).reshape(1, 2, 4, 4)
    out = attempt(drills.max_pool, x, 2)
    assert out.shape == (1, 2, 2, 2)
    assert out[0, 0].tolist() == [[5.0, 7.0], [13.0, 15.0]]


def test_global_avg_pool():
    x = rand(3, 5, 7, 7, seed=13)
    assert attempt(drills.global_avg_pool, x) == pytest.approx(x.mean((2, 3)))


# ---------------------------------------------- equivariance and invariance ----

def _stack(k1, k2):
    def f(x):
        h = np.maximum(drills.conv2d_naive(x, k1, padding=1), 0)
        return np.maximum(drills.conv2d_naive(h, k2, padding=1), 0)
    return f


def test_translation_equivariance_is_exact():
    k1, k2 = rand(4, 1, 3, 3, seed=14), rand(4, 4, 3, 3, seed=15)
    x = rand(1, 1, 32, 32, seed=16)
    attempt(drills.conv2d_naive, x, k1, padding=1)      # skip early if missing
    f = _stack(k1, k2)
    for dy, dx in ((1, 0), (0, 1), (5, 7), (11, -6)):
        err = attempt(drills.equivariance_error, f, x, dy, dx,
                      max(abs(dy), abs(dx)) + 4)
        assert err < 1e-9, f"shift ({dy},{dx}) gave {err:.2e}"


def test_rotation_equivariance_fails():
    k1, k2 = rand(4, 1, 3, 3, seed=17), rand(4, 4, 3, 3, seed=18)
    x = rand(1, 1, 32, 32, seed=19)
    attempt(drills.conv2d_naive, x, k1, padding=1)
    f = _stack(k1, k2)
    base = f(x)
    rot = f(np.rot90(x, 1, (2, 3)).copy())
    err = np.abs(rot - np.rot90(base, 1, (2, 3)))[:, :, 4:-4, 4:-4].max()
    assert err > 0.1 * np.abs(base).max(), "rotation should NOT be equivariant"


def test_pooling_buys_invariance_and_convolution_does_not():
    k1 = rand(4, 1, 3, 3, seed=20)
    x = rand(8, 1, 32, 32, seed=21)
    attempt(drills.conv2d_naive, x, k1, padding=1)

    def conv_only(t):
        return np.maximum(drills.conv2d_naive(t, k1, padding=1), 0)

    def pooled(t):
        return drills.max_pool(conv_only(t), 4)

    def globally(t):
        return drills.global_avg_pool(conv_only(t))

    e_conv = attempt(drills.invariance_error, conv_only, x, 0, 1)
    e_pool = attempt(drills.invariance_error, pooled, x, 0, 1)
    e_glob = attempt(drills.invariance_error, globally, x, 0, 1)
    assert e_conv > 3 * e_pool, f"conv {e_conv:.3f}, pooled {e_pool:.3f}"
    assert e_glob < 0.1 * e_pool, f"global {e_glob:.4f}, pooled {e_pool:.3f}"


def test_effective_receptive_field_is_smaller_than_the_support():
    """The gradient of a stack of 3x3 layers is centre-weighted."""
    S, depth = 81, 12
    ks = [rand(1, 1, 3, 3, seed=100 + i) for i in range(depth)]

    def grad_fn(_):
        # the composed linear response: convolve a delta forward through the stack
        g = np.zeros((1, 1, S, S))
        g[0, 0, S // 2, S // 2] = 1.0
        for k in ks:
            g = drills.conv2d_naive(g, k[:, :, ::-1, ::-1].copy(), padding=1)
        return g

    attempt(drills.conv2d_naive, np.zeros((1, 1, 5, 5)), ks[0], padding=1)
    support, eff = attempt(drills.receptive_field, grad_fn, (1, 1, S, S),
                           (S // 2, S // 2))
    assert support == 1 + 2 * depth, f"support {support}, expected {1 + 2 * depth}"
    assert eff < support, f"effective {eff} should be under the support {support}"


# --------------------------------------------------------------- counting ----

def test_parameter_counts():
    assert attempt(drills.conv_params, 64, 128, 3) == 73856
    assert attempt(drills.conv_params, 64, 128, 3, bias=False) == 73728
    assert attempt(drills.conv_params, 64, 128, 3, groups=4, bias=False) == 73728 // 4
    assert attempt(drills.conv_params, 64, 64, 3, groups=64, bias=False) == 64 * 9


def test_separable_ratio_is_an_identity():
    for c_in, c_out, k in ((64, 64, 3), (128, 256, 3), (256, 512, 3), (32, 64, 5)):
        std = attempt(drills.conv_params, c_in, c_out, k, bias=False)
        dw = attempt(drills.conv_params, c_in, c_in, k, groups=c_in, bias=False)
        pw = attempt(drills.conv_params, c_in, c_out, 1, bias=False)
        assert (dw + pw) / std == pytest.approx(
            attempt(drills.separable_ratio, c_out, k), abs=1e-12)


def test_macs_do_not_depend_on_parameters_alone():
    """The same layer costs different compute at different resolutions."""
    p = attempt(drills.conv_params, 64, 128, 3, bias=False)
    a = attempt(drills.conv_macs, 64, 128, 3, 56, 56)
    b = attempt(drills.conv_macs, 64, 128, 3, 112, 112)
    assert b == 4 * a
    assert p == 73728 and a == 73728 * 56 * 56


# ------------------------------------------------------------ transposed ----

def test_transpose_undoes_the_downsampling_shape():
    x = rand(2, 3, 8, 8, seed=22)
    k = rand(3, 5, 4, 4, seed=23)
    out = attempt(drills.conv_transpose2d, x, k, stride=2, padding=1)
    assert out.shape == (2, 5, 16, 16)


def test_transpose_is_the_adjoint_of_convolution():
    """<conv(x), g> == <x, conv_transpose(g)> -- the defining property."""
    x = rand(1, 2, 9, 9, seed=24)
    k = rand(3, 2, 3, 3, seed=25)
    z = attempt(drills.conv2d_naive, x, k)
    g = rand(*z.shape, seed=26)
    lhs = float((z * g).sum())
    back = attempt(drills.conv_transpose2d, g, k.transpose(0, 1, 2, 3))
    rhs = float((x * back).sum())
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_checkerboard_condition_is_exact():
    for k, s in ((2, 2), (4, 2), (4, 4), (6, 3)):
        vals = attempt(drills.checkerboard_values, k, s)
        assert len(vals) == 1, f"k={k} s={s} should be uniform, got {vals}"
    for k, s in ((3, 2), (5, 3), (6, 4)):
        vals = attempt(drills.checkerboard_values, k, s)
        assert len(vals) > 1, f"k={k} s={s} should checkerboard, got {vals}"


# ------------------------------------------------------------------ Gabor ----

def test_quadrature_pair_is_orthogonal():
    s0 = attempt(drills.gabor, 65, phi=0.0, tau=np.pi / 6)
    s1 = attempt(drills.gabor, 65, phi=np.pi / 2, tau=np.pi / 6)
    corr = (s0 * s1).sum() / np.sqrt((s0 * s0).sum() * (s1 * s1).sum())
    assert abs(corr) < 1e-10, f"correlation {corr:.2e}"


def test_complex_cell_is_invariant_and_simple_cells_are_not():
    S, tau, f = 65, np.pi / 6, 0.8
    s0 = attempt(drills.gabor, S, phi=0.0, tau=tau, f=f)
    s1 = attempt(drills.gabor, S, phi=np.pi / 2, tau=tau, f=f)
    yy, xx = np.mgrid[:S, :S] - (S - 1) / 2.0

    def stim(shift):
        xr = ((xx - shift * np.cos(tau)) * np.cos(tau)
              + (yy - shift * np.sin(tau)) * np.sin(tau))
        return np.cos(f * xr)

    shifts = np.arange(0, 8.01, 0.5)
    simple = np.array([(s0 * stim(d)).sum() for d in shifts])
    cplx = np.array([attempt(drills.complex_cell, s0, s1, stim(d)) for d in shifts])
    spread = lambda v: (v.max() - v.min()) / np.abs(v).max()
    assert spread(simple) > 1.5, "the simple cell should sweep its whole range"
    assert spread(cplx) < 1e-6, f"the complex cell varied by {spread(cplx):.2e}"


def test_complex_cell_ignores_a_sign_flip():
    s0 = attempt(drills.gabor, 41, phi=0.0)
    s1 = attempt(drills.gabor, 41, phi=np.pi / 2)
    img = rand(41, 41, seed=27)
    a = attempt(drills.complex_cell, s0, s1, img)
    b = attempt(drills.complex_cell, s0, s1, -img)
    assert a == pytest.approx(b, rel=1e-12)
