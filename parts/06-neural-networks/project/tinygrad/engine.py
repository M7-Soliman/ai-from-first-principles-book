"""
Stage 1 — the scalar autograd engine.

    python3 -m pytest tests/test_engine.py -v

One value at a time. It is unbearably slow and it is the clearest possible
version of the idea. Stage 3 makes it fast; nothing about the structure
changes when it does.

Three ideas and no others:
  1. wrap a number in an object that records what produced it
  2. each operation stores a small function saying how to send a gradient
     backward through itself
  3. backward() sorts the graph topologically and walks it in reverse

The one rule that matters: every _backward ACCUMULATES with +=. A value used
twice must receive both contributions. `y = x * x` is the smallest case that
catches an assignment, and it is in the tests.
"""
import math


class Value:
    """A scalar that remembers how it was computed."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # set by whichever op made this
        self._prev = set(_children)
        self._op = _op                     # for drawing the graph, stage 5

    # ---------------------------------------------------------- core ops --

    def __add__(self, other):
        """out = self + other

        Both parents receive the incoming gradient unchanged: d(a+b)/da = 1.
        """
        raise NotImplementedError("stage 1: Value.__add__")

    def __mul__(self, other):
        """out = self * other

        Each parent receives the OTHER parent's value times the incoming
        gradient. Write it out for y = x * x and you will see why += matters.
        """
        raise NotImplementedError("stage 1: Value.__mul__")

    def __pow__(self, other):
        """out = self ** other, for a numeric exponent only.

        Local derivative is the power rule from Part 0 §16.
        """
        raise NotImplementedError("stage 1: Value.__pow__")

    # -------------------------------------------------------- activations --

    def relu(self):
        """max(0, x). Derivative is 1 where active, 0 otherwise."""
        raise NotImplementedError("stage 1: Value.relu")

    def tanh(self):
        """Derivative is 1 - tanh(x)^2 -- expressible in its own output."""
        raise NotImplementedError("stage 1: Value.tanh")

    def exp(self):
        """Its own derivative, which is the whole point of e (Part 0 §9)."""
        raise NotImplementedError("stage 1: Value.exp")

    def log(self):
        """Natural log. Derivative 1/x, defined only for x > 0."""
        raise NotImplementedError("stage 1: Value.log")

    # ----------------------------------------------------------- backward --

    def backward(self):
        """Seed dL/dL = 1 and walk the graph in reverse topological order.

        Topological order matters: a node's gradient is only complete once
        every consumer of it has contributed. Build the order with a
        depth-first post-order traversal, then reverse it.
        """
        raise NotImplementedError("stage 1: Value.backward")

    # --------------------------- free once the four core ops work ---------

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ------------------------------------------------------------- utilities ----

def numerical_gradient(f, xs, h=1e-6):
    """Central-difference gradient of f at the list of floats `xs`.

    Your gradient check. Part III §3 explains the precision floor: shrinking
    h improves the truncation error until cancellation in the numerator takes
    over, so 1e-6 is about the sweet spot in float64 and float32 has no
    digits to spare at all.
    """
    raise NotImplementedError("stage 1: numerical_gradient")


def draw_ascii(root, max_nodes=40):
    """Print the graph reachable from `root`. Optional, and useful.

    Handy for stage 5 and for convincing yourself the topological order is
    what you think it is.
    """
    raise NotImplementedError("optional: draw_ascii")
