"""
Part VII drills — implementation problems.

    python3 -m pytest test_drills.py -v

Rules:
  * NumPy only. NOT allowed: torch, tensorflow, jax, autograd, sklearn.
    The whole point is that you write the differentiation.
  * Weight convention throughout: W is (out x in) and z = W a + b, matching
    Part VII §14 and Part I §8. Get this wrong and every shape test fails.

The centrepiece is `Value` -- a scalar automatic differentiation engine in
about forty lines. It is also stage 1 of the project, so a working one here
is a head start there.
"""
import numpy as np


# --------------------------------------------------------- the primitives ---

def relu(z):
    """max(0, z), elementwise.  (§7)"""
    raise NotImplementedError("TODO: relu")


def relu_grad(z):
    """Derivative of ReLU: 1 where z > 0, else 0.  (§7)

    Exactly 1 where active -- it neither shrinks nor amplifies the gradient,
    which is the entire reason it replaced the sigmoid.
    """
    raise NotImplementedError("TODO: relu_grad")


def sigmoid(z):
    """Stable logistic sigmoid.  (§6)

    Must not overflow at z = -800: branch on the sign so the exponent is
    never positive.
    """
    raise NotImplementedError("TODO: sigmoid")


def sigmoid_grad(z):
    """sigma(z) * (1 - sigma(z)).  (§7, Part 0 §18)

    Peaks at exactly 0.25. That ceiling, compounded over depth, is §18.
    """
    raise NotImplementedError("TODO: sigmoid_grad")


def init_scale(fan_in, nonlinearity="relu"):
    """Standard deviation for the initial weights.  (§11)

        sqrt(1 / fan_in)   for a linear layer
        sqrt(2 / fan_in)   for ReLU

    The factor of two is because ReLU zeroes roughly half the units, halving
    the variance. That is the whole derivation of He initialisation.
    """
    raise NotImplementedError("TODO: init_scale")


# ----------------------------------------------------- forward / backward ---

def forward(params, x):
    """Two-layer forward pass. Return (z1, a1, z2, y_hat).  (§15)

    `params` is a dict with W1 (h x d), b1 (h,), W2 (1 x h), b2 (scalar).

        z1 = W1 x + b1      a1 = relu(z1)
        z2 = W2 a1 + b2     y_hat = sigmoid(z2)

    Return z2 and y_hat as plain floats.
    """
    raise NotImplementedError("TODO: forward")


def backward(params, x, y, cache):
    """The four equations of §14. Return a dict of gradients.  (drill C4, §14)

    `cache` is the (z1, a1, z2, y_hat) that forward returned. Keys of the
    result: 'W1', 'b1', 'W2', 'b2', with shapes matching params exactly.

        1.  delta2 = y_hat - y                      (sigmoid + cross-entropy)
        2.  delta1 = (W2' delta2) * relu_grad(z1)
        3.  dL/dW  = delta a_prev'                  (an outer product)
        4.  dL/db  = delta

    The shape invariants of §15 catch nearly every bug here: delta has the
    shape of z, and dL/dW has the shape of W.
    """
    raise NotImplementedError("TODO: backward")


def numerical_gradient(f, params, h=1e-5):
    """Central-difference gradient of a scalar f(params).  (Part 0 drill C15)

    Perturb every entry of every array in `params` by +/-h and use
    (f(+h) - f(-h)) / 2h. Return a dict shaped like params.

    Use float64 -- Part III §3 explains why the central difference has a
    precision floor and why float32 does not have the digits to spare.
    """
    raise NotImplementedError("TODO: numerical_gradient")


# ------------------------------------------------------------ the engine ----

class Value:
    """A scalar that remembers how it was computed.  (drill C12, §21)

    Three ideas, and nothing else:
      1. wrap a number in an object that records what produced it
      2. each operation stores a small function saying how to send a
         gradient backward through itself
      3. backward() topologically sorts the graph and walks it in reverse

    Implement: __add__, __mul__, __pow__ (integer/float exponent), __neg__,
    __sub__, __truediv__, relu(), tanh(), exp(), log(), and backward().

    ACCUMULATE with += in every _backward, never =. A value used twice must
    receive both contributions -- y = x * x is the smallest case, and gives
    x instead of 2x if you assign.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        raise NotImplementedError("TODO: Value.__add__")

    def __mul__(self, other):
        raise NotImplementedError("TODO: Value.__mul__")

    def __pow__(self, other):
        raise NotImplementedError("TODO: Value.__pow__")

    def relu(self):
        raise NotImplementedError("TODO: Value.relu")

    def tanh(self):
        raise NotImplementedError("TODO: Value.tanh")

    def exp(self):
        raise NotImplementedError("TODO: Value.exp")

    def log(self):
        raise NotImplementedError("TODO: Value.log")

    def backward(self):
        """Topological sort, seed dL/dL = 1, then walk in reverse."""
        raise NotImplementedError("TODO: Value.backward")

    # --- these are free once the four above work -------------------------
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


# --------------------------------------------------------- the pathologies --

def gradient_by_layer(depth, activation="sigmoid", width=16, seed=0):
    """Gradient magnitude reaching each layer of a deep network.  (C8, §18)

    Build `depth` layers of the given activation, run one forward and one
    backward pass, and return an array of length `depth`: the norm of the
    gradient arriving at each layer, ordered INPUT layer first.

    Initialise with init_scale for the activation. With sigmoids the early
    entries are smaller by orders of magnitude; with ReLU they hold roughly
    constant. That contrast is Figure 10.
    """
    raise NotImplementedError("TODO: gradient_by_layer")


def count_dead_units(W, b, X):
    """How many units are inactive for EVERY row of X.  (drill C10, §19)

    A unit whose pre-activation is negative on every input has output zero
    and gradient zero forever -- it cannot recover. Return an int.
    """
    raise NotImplementedError("TODO: count_dead_units")


def train_xor(hidden=4, steps=4000, lr=0.5, seed=0):
    """Train a two-layer network on XOR. Return (params, final_loss).  (C5, §1)

    Inputs are the four corners, targets are 0,1,1,0. Use ReLU hidden units,
    a sigmoid output and cross-entropy. Should reach a loss well under 0.05,
    which no linear model can do at all.
    """
    raise NotImplementedError("TODO: train_xor")
