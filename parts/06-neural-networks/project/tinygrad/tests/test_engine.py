"""
Stage 1 tests — the scalar engine.

    python3 -m pytest tests/test_engine.py -v

Unimplemented pieces SKIP, so this is green from the first minute and fills
in as you go.

The decisive test is test_gradient_check_on_reused_values. An engine can look
completely correct on expressions where nothing is reused and still be wrong
by a factor of two the moment anything shares a value -- which in a real
network is constantly.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import Value, numerical_gradient   # noqa: E402


def run(thunk):
    """Evaluate a thunk, skipping if any operation is still a stub."""
    try:
        return thunk()
    except NotImplementedError as e:
        pytest.skip(str(e))


# ------------------------------------------------------------- basic ops ----

def test_add():
    def body():
        a, b = Value(2.0), Value(3.0)
        c = a + b
        c.backward()
        return a, b, c
    a, b, c = run(body)
    assert c.data == 5.0
    assert a.grad == 1.0 and b.grad == 1.0


def test_mul():
    def body():
        a, b = Value(2.0), Value(3.0)
        c = a * b
        c.backward()
        return a, b, c
    a, b, c = run(body)
    assert c.data == 6.0
    assert a.grad == 3.0 and b.grad == 2.0


def test_pow():
    def body():
        a = Value(3.0)
        c = a ** 3
        c.backward()
        return a, c
    a, c = run(body)
    assert c.data == 27.0
    assert math.isclose(a.grad, 27.0)          # 3x^2 at x=3


def test_scalars_on_either_side():
    def body():
        a = Value(4.0)
        c = 2.0 * a + 1.0 - a / 2.0
        c.backward()
        return a, c
    a, c = run(body)
    assert math.isclose(c.data, 7.0)
    assert math.isclose(a.grad, 1.5)           # 2 - 0.5


# ---------------------------------------------------------- the key case ----

def test_reuse_gives_2x_not_x():
    """y = x*x. With += you get 2x; with = you get x, exactly half."""
    def body():
        x = Value(3.0)
        y = x * x
        y.backward()
        return x
    x = run(body)
    assert math.isclose(x.grad, 6.0), (
        f"got {x.grad}, expected 6. If you got 3, a _backward is assigning "
        "where it must accumulate.")


def test_reuse_through_a_longer_chain():
    def body():
        x = Value(2.0)
        s = x + x            # 2x
        y = s * x            # 2x^2  ->  dy/dx = 4x = 8
        y.backward()
        return x
    assert math.isclose(run(body).grad, 8.0)


def test_shared_subexpression_used_three_times():
    def body():
        x = Value(1.5)
        s = x * 2.0
        out = s * s + s
        out.backward()
        return x, s
    x, s = run(body)
    assert math.isclose(s.grad, 2 * s.data + 1)
    assert math.isclose(x.grad, (2 * s.data + 1) * 2.0)


# ------------------------------------------------------------ activations ---

@pytest.mark.parametrize("name,x0,expected", [
    ("relu", 2.0, 1.0),
    ("relu", -2.0, 0.0),
    ("tanh", 0.5, 1 - math.tanh(0.5) ** 2),
    ("exp", 1.0, math.e),
    ("log", 4.0, 0.25),
])
def test_activation_derivatives(name, x0, expected):
    def body():
        x = Value(x0)
        getattr(x, name)().backward()
        return x
    assert math.isclose(run(body).grad, expected, rel_tol=1e-9)


def test_sigmoid_built_from_primitives():
    """You never implemented sigmoid -- but you can build it, and its
    derivative should come out sigma*(1-sigma), peaking at 0.25."""
    def body():
        x = Value(0.0)
        s = 1.0 / (1.0 + (-x).exp())
        s.backward()
        return x, s
    x, s = run(body)
    assert math.isclose(s.data, 0.5)
    assert math.isclose(x.grad, 0.25, rel_tol=1e-9), "the 0.25 ceiling of §18"


# --------------------------------------------------------- gradient check ---

def test_numerical_gradient_on_a_known_function():
    def f(xs):
        return xs[0] ** 2 + 3.0 * xs[1]
    g = run(lambda: numerical_gradient(f, [2.0, 5.0]))
    assert math.isclose(g[0], 4.0, abs_tol=1e-4)
    assert math.isclose(g[1], 3.0, abs_tol=1e-4)


def test_gradient_check_on_reused_values():
    """THE stage-1 deliverable: ten random expressions, every one of which
    reuses at least one value."""
    import random
    rnd = random.Random(0)

    def expr(vals):
        a, b, c = vals
        # a appears three times, c twice -- reuse everywhere
        return (a * b + c.relu()) * (a + c) + (b ** 2) / (a + 4.0)

    for _ in range(10):
        x0 = [rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-2, 2)]

        def body():
            vs = [Value(v) for v in x0]
            expr(vs).backward()
            return [v.grad for v in vs]
        analytic = run(body)

        def scalar_f(xs):
            return expr([Value(v) for v in xs]).data
        numeric = run(lambda: numerical_gradient(scalar_f, list(x0)))

        for a_, n_ in zip(analytic, numeric):
            assert math.isclose(a_, n_, abs_tol=1e-4), \
                f"analytic {analytic} vs numeric {numeric}"


def test_graph_records_parents():
    def body():
        a, b = Value(1.0), Value(2.0)
        return a, b, a * b
    a, b, c = run(body)
    assert a in c._prev and b in c._prev, "an op must record its children"
    assert Value(3.0)._prev == set(), "a leaf has no parents"
