"""
Stage 3 — the checks that catch a bug before the run finishes.

    python3 instrument.py

Section 17's list, made executable. Run these on any new model before you
spend a night tuning something that was never connected.
"""
import torch
import torch.nn.functional as F


def report_parameters(model, expected=None):
    """Count them, and compare against what the architecture implies.

    Catches bug 04 in one line -- an unregistered layer reports far fewer
    parameters than the code appears to build.
    """
    n = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n:,}")
    if expected is not None and n != expected:
        print(f"  !! expected {expected:,} -- something is not registered")
    return n


def report_gradients(model):
    """None, exactly zero, and nan are three DIFFERENT failures.

    None  -> the parameter is not in the graph (unregistered, detached, frozen)
    0.0   -> it is in the graph and the path is broken, or the unit is dead
    nan   -> an overflow upstream
    """
    issues = []
    for name, p in model.named_parameters():
        if p.grad is None:
            issues.append((name, "None -- not in the graph"))
        elif torch.isnan(p.grad).any():
            issues.append((name, "nan -- overflow upstream"))
        elif float(p.grad.norm()) == 0.0:
            issues.append((name, "exactly 0 -- broken path or dead unit"))
    total = sum(float(p.grad.norm()) for p in model.parameters()
                if p.grad is not None)
    print(f"  gradient norm: {total:.4f}")
    for name, why in issues:
        print(f"  !! {name}: {why}")
    return total


def check_modes(model, x, n=50):
    """Two evaluations of the same input must agree exactly. Catches bug 07."""
    model.eval()
    with torch.no_grad():
        outs = torch.stack([model(x) for _ in range(n)])
    # std ACROSS PASSES (dim 0), not across every element -- the logits differ
    # from each other even when every pass is identical, and averaging that in
    # gives a false alarm on a perfectly deterministic model.
    spread = float(outs.std(dim=0).max())
    print(f"  eval-mode spread over {n} passes: {spread:.2e}")
    if spread > 1e-6:
        print("  !! not deterministic in eval mode -- is model.eval() called?")
    return spread


def overfit_tiny_sample(model_fn, X, y, n=10, steps=400, lr=1e-2):
    """The single most useful test available, and almost nobody runs it first.

    A model that cannot drive the loss to near zero on ten examples has a
    STRUCTURAL bug -- a shape error, a detached gradient, a label
    misalignment -- and no amount of tuning will help. Part VII §20.
    """
    model = model_fn()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xb, yb = X[:n], y[:n]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    final = float(loss)
    print(f"  overfit {n} examples: final loss {final:.4f}")
    if final > 0.05:
        print("  !! cannot memorise ten examples -- the bug is structural, "
              "not a generalisation problem")
    return final


if __name__ == "__main__":
    from baseline import MLP, make_data, splits

    X, y = make_data()
    (Xtr, ytr), _, _ = splits(X, y)

    print("clean baseline:")
    model = MLP()
    report_parameters(model, expected=5634)
    F.cross_entropy(model(Xtr[:64]), ytr[:64]).backward()
    report_gradients(model)
    check_modes(model, Xtr[:1])
    overfit_tiny_sample(MLP, Xtr, ytr)
