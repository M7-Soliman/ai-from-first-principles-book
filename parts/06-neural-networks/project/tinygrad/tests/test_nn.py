"""
Stage 2 tests — the network library.

    python3 -m pytest tests/test_nn.py -v

These skip until stage 1 is finished, since everything here stands on Value.

The deliverable is the last test: XOR trained to near-zero loss from several
random seeds. Four points, no line separates them, and your own engine solves
it.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import Value                                    # noqa: E402
import nn as tnn                                            # noqa: E402


def run(thunk):
    try:
        return thunk()
    except NotImplementedError as e:
        pytest.skip(str(e))


# ---------------------------------------------------------------- pieces ----

def test_neuron_produces_a_value_and_owns_parameters():
    def body():
        n = tnn.Neuron(3, nonlin="relu")
        out = n([Value(1.0), Value(-2.0), Value(0.5)])
        return n, out
    n, out = run(body)
    assert isinstance(out, Value)
    ps = run(lambda: n.parameters())
    assert len(ps) == 4, "three weights and one bias"
    assert all(isinstance(p, Value) for p in ps)


def test_neuron_weights_are_not_all_zero():
    """Zero initialisation makes every unit identical forever -- §11."""
    n = run(lambda: tnn.Neuron(8, nonlin="relu"))
    ws = run(lambda: n.parameters())[:8]
    assert any(abs(w.data) > 1e-9 for w in ws), \
        "symmetry must be broken by randomness"


def test_layer_shape():
    def body():
        layer = tnn.Layer(3, 4)
        return layer, layer([Value(1.0), Value(0.0), Value(-1.0)])
    layer, out = run(body)
    assert len(out) == 4
    assert len(run(lambda: layer.parameters())) == 4 * (3 + 1)


def test_mlp_parameter_count():
    mlp = run(lambda: tnn.MLP([2, 8, 8, 1]))
    n = len(run(lambda: mlp.parameters()))
    expected = 8 * (2 + 1) + 8 * (8 + 1) + 1 * (8 + 1)
    assert n == expected, f"expected {expected} parameters, got {n}"


def test_mlp_forward_runs():
    mlp = run(lambda: tnn.MLP([2, 4, 1]))
    out = run(lambda: mlp([Value(0.5), Value(-0.5)]))
    # a single output may be returned bare or in a list of one
    v = out[0] if isinstance(out, (list, tuple)) else out
    assert isinstance(v, Value)


def test_zero_grad_clears_everything():
    def body():
        mlp = tnn.MLP([2, 4, 1])
        out = mlp([Value(1.0), Value(1.0)])
        v = out[0] if isinstance(out, (list, tuple)) else out
        v.backward()
        before = [p.grad for p in mlp.parameters()]
        mlp.zero_grad()
        return mlp, before
    mlp, before = run(body)
    assert any(g != 0.0 for g in before), "backward should have set gradients"
    assert all(p.grad == 0.0 for p in mlp.parameters()), "zero_grad must clear them"


# ---------------------------------------------------------------- losses ----

def test_squared_error_is_zero_when_perfect():
    v = run(lambda: tnn.squared_error([Value(1.0), Value(2.0)], [1.0, 2.0]))
    assert math.isclose(v.data, 0.0, abs_tol=1e-12)


def test_bce_with_logits_is_stable_at_extremes():
    """A confidently wrong prediction must give a large FINITE loss, not inf.

    Going via sigmoid then log underflows and gives inf here. The softplus
    form does not."""
    hi = run(lambda: tnn.binary_cross_entropy_with_logits(Value(-800.0), 1.0))
    lo = run(lambda: tnn.binary_cross_entropy_with_logits(Value(800.0), 0.0))
    for v in (hi, lo):
        assert math.isfinite(v.data), "log of an underflowed sigmoid gives inf"
    assert hi.data > 100.0, "confidently wrong should be punished hard"


@pytest.mark.parametrize("z0,y", [(-3.0, 0.0), (-3.0, 1.0), (-0.5, 1.0),
                                  (0.5, 0.0), (3.0, 1.0), (0.0, 1.0)])
def test_bce_gradient_is_prediction_minus_truth(z0, y):
    """Part VII §10: the sigma' factor cancels exactly, leaving a - y.

    The (0.0, 1.0) case is the one that catches a real bug. The stable form
    branches on the sign of z, and if the |z| term and the max(z, 0) term use
    DIFFERENT predicates -- one `>= 0`, the other `> 0` -- they disagree at
    exactly zero and the gradient comes out wrong by 1. Use one predicate for
    both.
    """
    def body():
        z = Value(z0)
        tnn.binary_cross_entropy_with_logits(z, y).backward()
        return z
    z = run(body)
    expected = 1.0 / (1.0 + math.exp(-z0)) - y
    assert math.isclose(z.grad, expected, abs_tol=1e-6), \
        f"at z={z0}, y={y}: expected sigmoid(z)-y = {expected:.4f}, got {z.grad:.4f}"


# ------------------------------------------------------------- optimiser ----

def test_sgd_moves_parameters_downhill():
    def body():
        p = Value(5.0)
        opt = tnn.SGD([p], lr=0.1, momentum=0.0)
        for _ in range(50):
            opt.zero_grad()
            (p * p).backward()               # minimise p^2
            opt.step()
        return p
    p = run(body)
    assert abs(p.data) < 1.0, f"should have descended toward 0, ended at {p.data}"


def test_momentum_moves_further_on_a_consistent_gradient():
    def run_with(beta):
        p = Value(5.0)
        opt = tnn.SGD([p], lr=0.01, momentum=beta)
        for _ in range(20):
            opt.zero_grad()
            (p * p).backward()
            opt.step()
        return abs(p.data)
    plain = run(lambda: run_with(0.0))
    momentum = run(lambda: run_with(0.9))
    assert momentum < plain, "momentum should accelerate a consistent direction"


# ------------------------------------------------------------ deliverable ---

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_xor_trains_to_near_zero_loss(seed):
    """The stage-2 deliverable, from several seeds. No line separates these
    four points; a hidden layer does."""
    import random

    def body():
        random.seed(seed)
        X, y = tnn.xor_data()
        model = tnn.MLP([2, 8, 8, 1])
        opt = tnn.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loss = None
        for _ in range(2000):
            opt.zero_grad()
            total = None
            for xi, yi in zip(X, y):
                out = model([Value(v) for v in xi])
                z = out[0] if isinstance(out, (list, tuple)) else out
                li = tnn.binary_cross_entropy_with_logits(z, yi)
                total = li if total is None else total + li
            loss = total * (1.0 / len(X))
            loss.backward()
            opt.step()
        return model, loss

    model, loss = run(body)
    assert loss.data < 0.05, f"final loss {loss.data:.4f} -- XOR is solvable"

    X, y = tnn.xor_data()
    for xi, yi in zip(X, y):
        out = run(lambda xi=xi: model([Value(v) for v in xi]))
        z = out[0] if isinstance(out, (list, tuple)) else out
        assert (z.data > 0) == (yi > 0.5), \
            f"input {xi} predicted the wrong side"
