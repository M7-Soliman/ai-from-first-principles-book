"""
Stage 2 — the network library, built on stage 1's Value.

    python3 -m pytest tests/test_nn.py -v

Nothing here knows any calculus. Every derivative is already inside Value;
this file only composes. That is the point of stage 1 — once the engine is
right, a network is bookkeeping.
"""
import random

from engine import Value


class Module:
    """Base class, so every component can hand over its parameters.

    Part IV §8 explains why a real framework needs a whole registration
    mechanism for this. Yours can be three lines, because you are not trying
    to survive arbitrary user code.
    """

    def parameters(self):
        return []

    def zero_grad(self):
        """Clear every parameter's gradient.

        Necessary because Value._backward ACCUMULATES with +=. Without this
        you step on the sum of every batch so far and the loss diverges in
        about fifty steps. Part IV §13 has the same line for the same reason.
        """
        raise NotImplementedError("stage 2: Module.zero_grad")


class Neuron(Module):
    """w . x + b, optionally followed by a nonlinearity.

    Initialise the weights with Part VII §11's scale -- sqrt(2/fan_in) for
    ReLU. Zero will not do: every unit would compute the same thing, receive
    the same gradient, and stay identical forever.
    """

    def __init__(self, n_in, nonlin="relu"):
        raise NotImplementedError("stage 2: Neuron.__init__")

    def __call__(self, x):
        raise NotImplementedError("stage 2: Neuron.__call__")

    def parameters(self):
        raise NotImplementedError("stage 2: Neuron.parameters")


class Layer(Module):
    """A row of Neurons sharing an input."""

    def __init__(self, n_in, n_out, nonlin="relu"):
        raise NotImplementedError("stage 2: Layer.__init__")

    def __call__(self, x):
        raise NotImplementedError("stage 2: Layer.__call__")

    def parameters(self):
        raise NotImplementedError("stage 2: Layer.parameters")


class MLP(Module):
    """A stack of Layers. The last one is linear -- it emits logits.

    sizes=[2, 8, 8, 1] means two inputs, two hidden layers of eight, one
    output. Keeping the final layer linear is Part VII §6: the loss applies
    the sigmoid or softmax itself, fused, for the stability reason in
    Part II §25.
    """

    def __init__(self, sizes):
        raise NotImplementedError("stage 2: MLP.__init__")

    def __call__(self, x):
        raise NotImplementedError("stage 2: MLP.__call__")

    def parameters(self):
        raise NotImplementedError("stage 2: MLP.parameters")


# ---------------------------------------------------------------- losses ----

def squared_error(pred, target):
    """(pred - target)^2, summed over a batch and averaged.

    Maximum likelihood under Gaussian noise -- Part II §26.
    """
    raise NotImplementedError("stage 2: squared_error")


def binary_cross_entropy_with_logits(logit, target):
    """Cross-entropy from a RAW logit, never from a probability.

    Compute it as log(1 + exp(-|z|)) + max(z, 0) - z*target, which is the
    stable softplus form of Part III §4. Going via sigmoid then log is the
    log-of-zero failure that Part II §25's pitfall is about, and your engine
    will happily produce inf and then nan.
    """
    raise NotImplementedError("stage 2: binary_cross_entropy_with_logits")


# ------------------------------------------------------------- optimiser ----

class SGD:
    """Stochastic gradient descent with momentum. Part III §§14-15.

    Keep a velocity per parameter:

        v <- beta*v + grad
        p <- p - lr*v

    Along a consistent direction the effective step is 1/(1-beta) times a
    single gradient step -- ten times at beta=0.9. That is why momentum is
    close to free.
    """

    def __init__(self, params, lr=0.05, momentum=0.9):
        raise NotImplementedError("stage 2: SGD.__init__")

    def step(self):
        raise NotImplementedError("stage 2: SGD.step")

    def zero_grad(self):
        raise NotImplementedError("stage 2: SGD.zero_grad")


# ----------------------------------------------------------------- data -----

def xor_data():
    """The four points no line separates. Part VII §1."""
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]
    return X, y


def two_moons(n=200, noise=0.15, seed=0):
    """Two interleaving half-circles. Optional, for the stage-2 deliverable.

    Not linearly separable, and a two-hidden-layer MLP handles it easily.
    Plot the decision boundary once it trains -- that plot is the deliverable.
    """
    raise NotImplementedError("stage 2: two_moons")
