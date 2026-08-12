"""
Reference solutions for the Part IV drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ------------------------------------------------------------ the object ----

def describe(t):
    return {
        "shape": tuple(t.shape),
        "dtype": t.dtype,
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
        "has_grad": t.grad is not None,
    }


def to_float32(x):
    # only floating types convert; integer labels must survive untouched
    return x.float() if x.is_floating_point() else x


# ------------------------------------------------------------- the graph ----

def _nodes(root):
    """Every distinct grad_fn node reachable from root, as a LIST.

    A list rather than a generator on purpose. PyTorch hands out a fresh
    Python wrapper on each .grad_fn access and frees it when unreferenced, so
    CPython recycles the id -- and a `seen` set of ids then marks unvisited
    nodes as already seen. Holding every node in `out` is what makes id()
    trustworthy here.
    """
    out, seen, stack = [], set(), [root]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        out.append(fn)                     # keeps the wrapper alive
        stack.extend(n for n, _ in fn.next_functions)
    return out


def graph_node_types(loss):
    return sorted({type(fn).__name__ for fn in _nodes(loss.grad_fn)})


def count_leaves(loss):
    return sum(1 for fn in _nodes(loss.grad_fn)
               if type(fn).__name__ == "AccumulateGrad")


def retained_activation_bytes(output):
    if output.grad_fn is None:
        return 0
    keep, seen, total = [], set(), 0
    for fn in _nodes(output.grad_fn):
        for attr in dir(fn):
            if not attr.startswith("_saved_"):
                continue
            v = getattr(fn, attr, None)
            if torch.is_tensor(v):
                keep.append(v)             # same liveness trap, for tensors
                if id(v) not in seen:      # a tensor saved twice counts once
                    seen.add(id(v))
                    total += v.numel() * v.element_size()
    return total


def is_leaf_and_grad(x, y):
    y.backward()
    return (bool(x.is_leaf), bool(y.is_leaf),
            x.grad is not None, y.grad is not None)


# ------------------------------------------------------------- the model ----

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parameter_names(model):
    return sorted(n for n, _ in model.named_parameters())


class BrokenMLP(nn.Module):
    def __init__(self, width=16, depth=4):
        super().__init__()
        # a plain list -- __setattr__ sees a list, not a Module, so nothing
        # is registered and parameters() will be empty
        self.layers = [nn.Linear(width, width) for _ in range(depth)]

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class FixedMLP(nn.Module):
    def __init__(self, width=16, depth=4):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


# -------------------------------------------------------------- the loop ----

def train_step(model, opt, xb, yb):
    opt.zero_grad()
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    return loss.item()          # .item(), so the graph can be freed


def accumulate_gradients(model, batches):
    model.zero_grad()
    for xb, yb in batches:
        # each loss is already a mean over its batch, so scale before summing
        (F.cross_entropy(model(xb), yb) / len(batches)).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def mode_output_spread(model, x, n=200):
    with torch.no_grad():
        model.train()
        tr = torch.stack([model(x).squeeze() for _ in range(n)])
        model.eval()
        ev = torch.stack([model(x).squeeze() for _ in range(n)])
    return float(tr.std()), float(ev.std())


def evaluate(model, batches):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in batches:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += yb.numel()
    return correct / total


# ------------------------------------------------------------ checkpoints ----

def save_checkpoint(path, model, opt, epoch):
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict()}, path)


def load_checkpoint(path, model, opt):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    model.eval()                 # a fresh module defaults to TRAIN mode
    return ckpt["epoch"]


import drills  # noqa: E402

for _name, _obj in list(globals().items()):
    if (callable(_obj) or isinstance(_obj, type)) and hasattr(drills, _name):
        setattr(drills, _name, _obj)
