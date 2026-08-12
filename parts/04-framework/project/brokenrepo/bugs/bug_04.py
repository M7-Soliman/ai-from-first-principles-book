"""bug_04 -- one fault. Diagnose from symptoms before reading.

    python3 diagnose.py 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# An instrumentation seam. Stage 3 of the project fills this in; diagnose.py
# uses it to sample things that are only visible DURING an epoch -- a retained
# graph, for instance, is freed the moment train() returns.
BATCH_PROBE = None


# ------------------------------------------------------------------ data ----

def make_data(n=3000, d=20, seed=0):
    """A learnable rule with a little noise, so the task is neither trivial
    nor impossible. Labels depend on the first three features only."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    logit = X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2]
    y = (logit + 0.5 * torch.randn(n, generator=g) > 0).long()
    return X, y


def splits(X, y, seed=0):
    """Train / validation / test. The test set is touched ONCE, at the end --
    Part VI §9, and Part V §15 on what repeated looks cost you."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_tr, n_va = int(0.7 * len(X)), int(0.15 * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# ----------------------------------------------------------------- model ----

class MLP(nn.Module):
    """A Module subclass rather than a Sequential, so the registration of
    Part IV §8 is visible and breakable."""

    def __init__(self, d_in=20, hidden=64, n_classes=2, depth=2, p_drop=0.3):
        super().__init__()                       # required -- §8
        self.layers = []
        prev = d_in
        for _ in range(depth):
            self.layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.drop = nn.Dropout(p_drop)           # behaves differently in the two modes
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(self.drop(x))           # raw logits -- §25 of Part II


# ------------------------------------------------------------------ loop ----

def evaluate(model, X, y):
    """eval() for the layers, no_grad for the graph. Both, and they are
    independent -- Part IV §14."""
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def train(epochs=8, batch_size=64, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    X, y = make_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits(X, y)

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=batch_size, shuffle=True)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n:,}")           # §17's first check

    history = []
    grad_norms = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()                   # 1  gradients accumulate
            logits = model(xb)                # 2
            loss = F.cross_entropy(logits, yb)  # 3  takes LOGITS
            loss.backward()                   # 4
            opt.step()                        # 5
            running += loss.item()            # .item() -- §3, or you leak
            if BATCH_PROBE is not None:
                BATCH_PROBE(locals())
        gn = sum(float(p.grad.norm()) for p in model.parameters()
                 if p.grad is not None)
        grad_norms.append(gn)
        val = evaluate(model, Xva, yva)
        history.append((running / len(loader), val))
        if verbose:
            print(f"epoch {epoch}  loss {history[-1][0]:.4f}  val {val:.3f}")

    train.last_grad_norms = grad_norms        # read by diagnose.py
    return model, history, (Xte, yte)


if __name__ == "__main__":
    model, history, (Xte, yte) = train()
    print(f"\ntest accuracy (touched once): {evaluate(model, Xte, yte):.3f}")
