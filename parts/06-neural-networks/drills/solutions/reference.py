"""
Reference solutions for the Part VII drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# --------------------------------------------------------- the primitives ---

def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_grad(z):
    return (np.asarray(z, dtype=float) > 0).astype(float)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def init_scale(fan_in, nonlinearity="relu"):
    gain = 2.0 if nonlinearity == "relu" else 1.0
    return math.sqrt(gain / fan_in)


# ----------------------------------------------------- forward / backward ---

def forward(params, x):
    x = np.asarray(x, dtype=float)
    z1 = params["W1"] @ x + params["b1"]
    a1 = relu(z1)
    z2 = float((params["W2"] @ a1 + params["b2"]).item())
    y_hat = float(sigmoid(np.array([z2]))[0])
    return z1, a1, z2, y_hat


def backward(params, x, y, cache):
    x = np.asarray(x, dtype=float)
    z1, a1, z2, y_hat = cache

    d2 = y_hat - y                              # eq 1: the sigma' cancelled
    dW2 = np.outer(np.array([d2]), a1)          # eq 3
    db2 = np.array([d2])                        # eq 4

    d1 = (params["W2"].T @ np.array([d2])).ravel() * relu_grad(z1)   # eq 2
    dW1 = np.outer(d1, x)                       # eq 3
    db1 = d1                                    # eq 4

    return {"W1": dW1, "b1": db1,
            "W2": dW2.reshape(params["W2"].shape),
            "b2": db2.reshape(np.shape(params["b2"]))}


def numerical_gradient(f, params, h=1e-5):
    grads = {}
    for k, v in params.items():
        arr = np.array(v, dtype=float)
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = arr[i]
            arr[i] = orig + h
            params[k] = arr
            fp = f(params)
            arr[i] = orig - h
            params[k] = arr
            fm = f(params)
            arr[i] = orig
            params[k] = arr
            g[i] = (fp - fm) / (2 * h)
            it.iternext()
        grads[k] = g
    return grads


# ------------------------------------------------------------ the engine ----

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad            # += : a value may be used twice
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only numeric powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad        # d/dx e^x = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            order.append(v)
        build(self)

        self.grad = 1.0                      # dL/dL
        for v in reversed(order):
            v._backward()

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
    rng = np.random.default_rng(seed)
    act = sigmoid if activation == "sigmoid" else relu
    act_grad = sigmoid_grad if activation == "sigmoid" else relu_grad

    Ws = [rng.normal(0, init_scale(width, activation), size=(width, width))
          for _ in range(depth)]
    a = rng.normal(size=width)

    zs, acts = [], [a]
    for W in Ws:
        z = W @ acts[-1]
        zs.append(z)
        acts.append(act(z))

    delta = np.ones(width)                    # seed the backward pass
    norms = []
    for k in range(depth - 1, -1, -1):
        delta = delta * act_grad(zs[k])
        norms.append(float(np.linalg.norm(delta)))
        delta = Ws[k].T @ delta
    return np.array(norms[::-1])              # input layer first


def count_dead_units(W, b, X):
    Z = np.asarray(X, dtype=float) @ np.asarray(W, dtype=float).T + np.asarray(b, dtype=float)
    return int(np.sum(np.all(Z <= 0, axis=0)))


def train_xor(hidden=4, steps=4000, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])

    params = {
        "W1": rng.normal(0, init_scale(2, "relu"), size=(hidden, 2)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, init_scale(hidden, "linear"), size=(1, hidden)),
        "b2": np.zeros(1),
    }

    loss = None
    for _ in range(steps):
        grads = {k: np.zeros_like(v) for k, v in params.items()}
        loss = 0.0
        for xi, yi in zip(X, y):
            cache = forward(params, xi)
            y_hat = min(max(cache[3], 1e-12), 1 - 1e-12)
            loss += -(yi * math.log(y_hat) + (1 - yi) * math.log(1 - y_hat))
            g = backward(params, xi, yi, cache)
            for k in grads:
                grads[k] += g[k]
        loss /= len(X)
        for k in params:
            params[k] = params[k] - lr * grads[k] / len(X)
    return params, float(loss)


import drills  # noqa: E402

for _name, _obj in list(globals().items()):
    if (callable(_obj) or isinstance(_obj, type)) and hasattr(drills, _name):
        setattr(drills, _name, _obj)
