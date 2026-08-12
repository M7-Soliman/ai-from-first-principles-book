"""
Tests for the Part IV drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Two things are graded beyond correctness:
  * the SCALING relationships the figures measure, not their constants
  * whether an implementation leaks the graph -- several tests check that a
    computation left nothing attached behind it
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        name = getattr(fn, "__name__", str(fn))
        pytest.skip(f"{name} not implemented yet")


def attempt_cls(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{cls.__name__} not implemented yet")


torch.manual_seed(0)


def tiny_model(d_in=8, d_out=3):
    return nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, d_out))


def tiny_batch(n=12, d_in=8, d_out=3):
    return torch.randn(n, d_in), torch.randint(0, d_out, (n,))


# ------------------------------------------------------------ the object ----

def test_describe_reports_five_attributes():
    t = torch.randn(2, 3)
    d = attempt(drills.describe, t)
    assert set(d) == {"shape", "dtype", "device", "requires_grad", "has_grad"}
    assert d["shape"] == (2, 3) and isinstance(d["shape"], tuple)
    assert d["dtype"] == torch.float32, "PyTorch's default is 32-bit, not 64"
    assert d["requires_grad"] is False and d["has_grad"] is False


def test_describe_tracks_grad_state():
    t = torch.randn(3, requires_grad=True)
    (t * 2).sum().backward()
    d = attempt(drills.describe, t)
    assert d["requires_grad"] is True and d["has_grad"] is True


def test_to_float32_converts_floats_only():
    x64 = torch.randn(4, dtype=torch.float64)
    assert attempt(drills.to_float32, x64).dtype == torch.float32
    labels = torch.randint(0, 3, (4,))
    out = attempt(drills.to_float32, labels)
    assert out.dtype == labels.dtype, "labels are indices; converting breaks cross_entropy"


def test_to_float32_makes_a_batch_usable():
    layer = nn.Linear(4, 2)
    x64 = torch.randn(5, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        layer(x64)                       # the dtype error of section 16
    layer(attempt(drills.to_float32, x64))   # must now work


# ------------------------------------------------------------- the graph ----

def _tiny_loss():
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x, y = torch.tensor(2.0), torch.tensor(1.0)
    return (torch.sigmoid(w * x + b) - y) ** 2


def test_graph_node_types_finds_the_chain():
    names = attempt(drills.graph_node_types, _tiny_loss())
    assert names == sorted(names), "return them sorted"
    joined = " ".join(names)
    for expected in ("AccumulateGrad", "Mul", "Add", "Sigmoid", "Pow"):
        assert expected in joined, f"missing {expected} in {names}"


def test_count_leaves_is_one_per_parameter():
    assert attempt(drills.count_leaves, _tiny_loss()) == 2   # w and b

    model = tiny_model()
    xb, yb = tiny_batch()
    loss = F.cross_entropy(model(xb), yb)
    n_param_tensors = len(list(model.parameters()))          # 4: two W, two b
    assert attempt(drills.count_leaves, loss) == n_param_tensors


def test_retained_bytes_has_a_linear_activation_term():
    """The INCREMENT per extra example is constant, so doubling the batch
    doubles the increment. This is the activation term, and it is exactly
    linear -- unlike the total, which the parameter term dampens."""
    model = nn.Sequential(*[m for _ in range(4)
                            for m in (nn.Linear(32, 32), nn.ReLU())])

    def at(b):
        return attempt(drills.retained_activation_bytes, model(torch.randn(b, 32)))

    b32, b64, b128 = at(32), at(64), at(128)
    assert b32 > 0 and b64 > b32 < b128
    step_lo, step_hi = b64 - b32, b128 - b64
    ratio = step_hi / step_lo
    assert 1.8 < ratio < 2.2, \
        f"doubling the batch should double the increment, got {ratio:.2f}"


def test_retained_bytes_has_a_constant_term_too():
    """The graph also saves the weight matrices, which do not scale with batch.
    So the total is NOT proportional to batch size -- Figure 2's flat region."""
    model = nn.Sequential(*[m for _ in range(4)
                            for m in (nn.Linear(32, 32), nn.ReLU())])

    def at(b):
        return attempt(drills.retained_activation_bytes, model(torch.randn(b, 32)))

    small, large = at(8), at(64)
    ratio = large / small
    assert ratio < 7.0, (
        f"an 8x batch gave {ratio:.1f}x memory. If you got ~8x you are missing "
        "the saved weight matrices; if you got ~1x you fell into the id() "
        "liveness trap and under-counted.")


def test_retained_bytes_scales_with_depth():
    def bytes_at(depth):
        model = nn.Sequential(*[m for _ in range(depth)
                                for m in (nn.Linear(32, 32), nn.ReLU())])
        return attempt(drills.retained_activation_bytes, model(torch.randn(16, 32)))
    assert bytes_at(8) > 1.8 * bytes_at(4), "activations grow with depth too"


def test_retained_bytes_counts_each_tensor_once():
    """A tensor saved by two nodes must not be counted twice."""
    x = torch.randn(4, 4, requires_grad=True)
    once = attempt(drills.retained_activation_bytes, torch.sigmoid(x))
    assert once < x.numel() * x.element_size() * 3, "looks like double counting"


def test_no_grad_region_builds_no_graph():
    model = tiny_model()
    with torch.no_grad():
        out = model(torch.randn(4, 8))
    assert out.grad_fn is None
    assert attempt(drills.retained_activation_bytes, out) == 0


def test_leaf_and_grad_asymmetry():
    x = torch.tensor(2.0, requires_grad=True)
    y = x * 3
    x_leaf, y_leaf, x_has, y_has = attempt(drills.is_leaf_and_grad, x, y)
    assert (x_leaf, y_leaf) == (True, False)
    assert (x_has, y_has) == (True, False), "only leaves receive a .grad"


# ------------------------------------------------------------- the model ----

def test_count_parameters():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    assert attempt(drills.count_parameters, model) == (10 * 20 + 20) + (20 * 5 + 5)


def test_count_parameters_ignores_buffers():
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    # BatchNorm has 4 weights + 4 biases learnable; running stats are buffers
    assert attempt(drills.count_parameters, m) == (4 * 4 + 4) + (4 + 4)


def test_parameter_names():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert attempt(drills.parameter_names, model) == \
        ["0.bias", "0.weight", "2.bias", "2.weight"]


def test_registration_bug_is_real():
    """The plain-list model runs and reports zero parameters."""
    broken = attempt_cls(drills.BrokenMLP, width=16, depth=4)
    out = broken(torch.randn(2, 16))
    assert out.shape == (2, 16), "it must still run -- that is what makes it nasty"
    assert attempt(drills.count_parameters, broken) == 0, \
        "a plain Python list registers nothing"


def test_modulelist_fixes_it():
    fixed = attempt_cls(drills.FixedMLP, width=16, depth=4)
    assert attempt(drills.count_parameters, fixed) == 4 * (16 * 16 + 16)
    assert fixed(torch.randn(2, 16)).shape == (2, 16)


def test_only_the_fixed_model_learns():
    torch.manual_seed(1)
    broken = attempt_cls(drills.BrokenMLP, width=8, depth=3)
    fixed = attempt_cls(drills.FixedMLP, width=8, depth=3)
    x = torch.randn(16, 8)
    for m in (broken, fixed):
        before = [p.detach().clone() for p in m.parameters()]
        opt = torch.optim.SGD(m.parameters(), lr=0.1) if before else None
        if opt is None:
            continue                       # broken has no parameters at all
        opt.zero_grad(); m(x).pow(2).sum().backward(); opt.step()
        moved = any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
        assert moved, "the registered model must actually update"


# -------------------------------------------------------------- the loop ----

def test_train_step_returns_a_float_not_a_tensor():
    model, opt = tiny_model(), None
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    xb, yb = tiny_batch()
    loss = attempt(drills.train_step, model, opt, xb, yb)
    assert isinstance(loss, float), "return loss.item(); returning the tensor leaks the graph"


def test_train_step_actually_decreases_loss():
    torch.manual_seed(2)
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    xb, yb = tiny_batch(n=32)
    first = attempt(drills.train_step, model, opt, xb, yb)
    for _ in range(30):
        last = attempt(drills.train_step, model, opt, xb, yb)
    assert last < first


def test_train_step_zeroes_gradients():
    """Two identical steps from the same state must give the same gradient."""
    torch.manual_seed(3)
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr 0: state cannot move
    xb, yb = tiny_batch()
    attempt(drills.train_step, model, opt, xb, yb)
    g1 = model[0].weight.grad.clone()
    attempt(drills.train_step, model, opt, xb, yb)
    g2 = model[0].weight.grad.clone()
    assert torch.allclose(g1, g2), "gradients accumulated -- zero_grad is missing"


def test_gradient_accumulation_matches_one_big_batch():
    torch.manual_seed(4)
    model = tiny_model()
    X, Y = tiny_batch(n=32)
    chunks = [(X[i:i + 8], Y[i:i + 8]) for i in range(0, 32, 8)]

    accum = attempt(drills.accumulate_gradients, model, chunks)

    model.zero_grad()
    F.cross_entropy(model(X), Y).backward()
    whole = torch.cat([p.grad.flatten() for p in model.parameters()])

    assert torch.allclose(accum, whole, atol=1e-5), \
        "divide each loss by the number of batches -- each is already a mean"


def test_mode_spread_train_moves_eval_does_not():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
    tr, ev = attempt(drills.mode_output_spread, model, torch.randn(1, 8), 200)
    assert tr > 1e-3, "dropout must be active in train mode"
    assert ev < 1e-5, "eval mode must be deterministic"


def test_evaluate_is_correct_and_leaves_no_graph():
    torch.manual_seed(6)
    model = tiny_model()
    batches = [tiny_batch(n=8) for _ in range(3)]
    acc = attempt(drills.evaluate, model, batches)
    assert 0.0 <= acc <= 1.0

    expected_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in batches:
            expected_correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    assert abs(acc - expected_correct / total) < 1e-6


def test_evaluate_uses_eval_mode():
    """With dropout present, two evaluations must agree exactly."""
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 3))
    model.train()                                   # deliberately left in train mode
    batches = [tiny_batch(n=16) for _ in range(2)]
    a = attempt(drills.evaluate, model, batches)
    b = attempt(drills.evaluate, model, batches)
    assert a == b, "call model.eval() -- otherwise dropout randomises the score"


# ------------------------------------------------------------ checkpoints ----

def test_checkpoint_round_trip_restores_weights_and_optimiser():
    torch.manual_seed(8)
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    xb, yb = tiny_batch()
    for _ in range(3):
        opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        attempt(drills.save_checkpoint, path, model, opt, epoch=7)

        fresh = tiny_model()
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1, momentum=0.9)
        epoch = attempt(drills.load_checkpoint, path, fresh, fresh_opt)

        assert epoch == 7
        for a, b in zip(model.parameters(), fresh.parameters()):
            assert torch.allclose(a, b)
        assert fresh_opt.state_dict()["state"], \
            "momentum buffers were not restored -- the loss will bump on resume"


def test_loaded_model_is_in_eval_mode():
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        attempt(drills.save_checkpoint, path, model, opt, epoch=0)
        fresh = tiny_model()
        fresh.train()
        attempt(drills.load_checkpoint, path, fresh,
                torch.optim.SGD(fresh.parameters(), lr=0.1))
        assert not fresh.training, "a freshly loaded module defaults to TRAIN mode"
