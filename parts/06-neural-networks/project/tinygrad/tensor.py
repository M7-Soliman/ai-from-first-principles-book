"""
Stage 3 — the array engine.

    python3 -m pytest tests/test_tensor.py -v

The structure is IDENTICAL to stage 1. Only the local derivatives change from
numbers to matrix expressions. If you find yourself redesigning, stop -- the
design was right and you are only vectorising it.

Two things are genuinely new and both are where people get it wrong:

  * matmul's backward, which you derived in Part VII §14 -- dL/dW is an outer
    product, delta a', and dL/da_prev pushes back through W transposed
  * broadcasting's backward, which must SUM over the axes that were
    broadcast. A bias of shape (n,) added to a (batch, n) activation receives
    the gradient summed over the batch, because it contributed to every row
"""
import numpy as np


class Tensor:
    """Same three ideas as Value, with an ndarray inside."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """Remember to sum over broadcast axes in the backward pass."""
        raise NotImplementedError("stage 3: Tensor.__add__")

    def __mul__(self, other):
        raise NotImplementedError("stage 3: Tensor.__mul__")

    def __matmul__(self, other):
        """Part VII §14, equations 2 and 3, in matrix form."""
        raise NotImplementedError("stage 3: Tensor.__matmul__")

    def relu(self):
        raise NotImplementedError("stage 3: Tensor.relu")

    def log_softmax(self, axis=-1):
        """Computed DIRECTLY -- never log(softmax(x)).

        z - logsumexp(z), per Part III §4. The fused form cannot take the log
        of an underflowed probability, which is the whole reason frameworks
        ship it fused (Part II §25).
        """
        raise NotImplementedError("stage 3: Tensor.log_softmax")

    def sum(self, axis=None):
        raise NotImplementedError("stage 3: Tensor.sum")

    def backward(self):
        raise NotImplementedError("stage 3: Tensor.backward")

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing whatever broadcasting did.

    THE function people get wrong in stage 3. If a (3,) bias was broadcast
    against a (32, 3) activation, its gradient arrives as (32, 3) and must be
    summed over axis 0 to become (3,) again.
    """
    raise NotImplementedError("stage 3: unbroadcast")
