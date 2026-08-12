"""
Stage 3 tests — the array engine.

    python3 -m pytest tests/test_tensor.py -v

The deliverable is that the array engine gives IDENTICAL gradients to the
scalar one, plus a measured speedup you report yourself.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor import Tensor, unbroadcast   # noqa: E402


def run(thunk):
    try:
        return thunk()
    except NotImplementedError as e:
        pytest.skip(str(e))


def test_unbroadcast_sums_over_the_broadcast_axis():
    g = np.ones((32, 3))
    out = run(lambda: unbroadcast(g, (3,)))
    assert out.shape == (3,)
    assert np.allclose(out, 32.0), "a bias contributed to every row"


def test_unbroadcast_is_identity_when_shapes_match():
    g = np.arange(6.0).reshape(2, 3)
    assert np.allclose(run(lambda: unbroadcast(g, (2, 3))), g)


def test_add_with_broadcasting_gives_the_right_bias_gradient():
    def body():
        W = Tensor(np.zeros((4, 3)))
        b = Tensor(np.zeros(3))
        out = W + b
        out.grad = np.ones_like(out.data)
        out.backward()
        return b
    b = run(body)
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, 4.0)


def test_matmul_backward_matches_numerical():
    rng = np.random.default_rng(0)
    A0, B0 = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))

    def loss_of(A, B):
        return float((np.asarray(A) @ np.asarray(B)).sum())

    def body():
        A, B = Tensor(A0), Tensor(B0)
        out = (A @ B).sum()
        out.backward()
        return A, B
    A, B = run(body)

    h = 1e-6
    for i in range(3):
        for j in range(4):
            hi, lo = A0.copy(), A0.copy()
            hi[i, j] += h
            lo[i, j] -= h
            num = (loss_of(hi, B0) - loss_of(lo, B0)) / (2 * h)
            assert np.isclose(A.grad[i, j], num, atol=1e-4)


def test_log_softmax_is_stable():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    out = run(lambda: Tensor(z).log_softmax())
    assert np.all(np.isfinite(out.data)), "log(softmax(z)) in two steps gives nan here"
    assert np.isclose(np.exp(out.data).sum(), 1.0, atol=1e-9)
