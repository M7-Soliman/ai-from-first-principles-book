"""
The Part XIV figures that require training. Imported by make_figures.py.

    python3 train_figures.py

Two questions that need a real model: whether its confidences mean anything,
and how far a learning curve can be trusted to extrapolate.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, plt, BLUE, ORANGE, GREEN,           # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK)

torch.set_num_threads(4)


def spiral(n, classes=3, noise=1.0, seed=0):
    r = np.random.default_rng(seed)
    X = np.zeros((n * classes, 2), dtype=np.float32)
    y = np.zeros(n * classes, dtype=np.int64)
    for c in range(classes):
        t = np.linspace(0.2, 3.4, n) + r.normal(0, noise, n)
        rad = np.linspace(0.15, 1.0, n)
        ang = t + c * 2 * np.pi / classes
        X[c * n:(c + 1) * n] = np.stack([rad * np.sin(ang), rad * np.cos(ang)], 1)
        y[c * n:(c + 1) * n] = c
    return torch.tensor(X), torch.tensor(y)


def mlp(width=128, depth=3, classes=3):
    layers, d = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers += [nn.Linear(d, classes)]
    return nn.Sequential(*layers)


def train(m, X, y, steps=1500, lr=3e-3, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        i = torch.randint(0, len(X), (min(bs, len(X)),))
        opt.zero_grad()
        F.cross_entropy(m(X[i]), y[i]).backward()
        opt.step()
    return m


# ------------------------------------------------------------- 6. calibration

def _bins(conf, correct, n_bins=12):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum() < 5:
            continue
        xs.append(float(conf[m].mean()))
        ys.append(float(correct[m].mean()))
        ws.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ws)


def _ece(conf, correct, n_bins=12):
    xs, ys, ws = _bins(conf, correct, n_bins)
    if len(xs) == 0:
        return 0.0
    return float(np.sum(ws * np.abs(xs - ys)) / np.sum(ws))


def fig_calibration():
    # a genuinely hard version of the task: at low noise the model is right
    # 99.9% of the time and there is nothing to calibrate.
    Xtr, ytr = spiral(120, seed=1)
    Xva, yva = spiral(120, seed=2)
    Xte, yte = spiral(600, seed=3)

    torch.manual_seed(0)
    m = train(mlp(width=256, depth=4), Xtr, ytr, steps=4000, seed=0)
    m.eval()
    with torch.no_grad():
        logits_te = m(Xte)
        logits_va = m(Xva)

    def stats(logits, y, T=1.0):
        p = F.softmax(logits / T, dim=1)
        conf, pred = p.max(1)
        return conf.numpy(), (pred == y).numpy().astype(float)

    conf0, corr0 = stats(logits_te, yte)
    # fit a single temperature on the VALIDATION set, as one must
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_va / logT.exp(), yva)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(logT.exp())
    conf1, corr1 = stats(logits_te, yte, T)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.3, label="perfect calibration")
    for (c, k, col, lab) in ((conf0, corr0, ORANGE, "as trained"),
                             (conf1, corr1, BLUE, f"temperature scaled ($T={T:.2f}$)")):
        xs, ys, _ = _bins(c, k)
        ax.plot(xs, ys, "o-", color=col, ms=5, lw=1.9, label=lab)
    ax.set_xlabel("confidence the model reports")
    ax.set_ylabel("accuracy it actually achieves")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.hist(conf0, bins=24, color=ORANGE, alpha=0.65, label="as trained")
    ax.hist(conf1, bins=24, color=BLUE, alpha=0.6, label="temperature scaled")
    ax.set_xlabel("reported confidence")
    ax.set_ylabel("examples")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    e0, e1 = _ece(conf0, corr0), _ece(conf1, corr1)
    print(f"    accuracy {corr0.mean():.3f} -> {corr1.mean():.3f}: UNCHANGED, because a "
          f"single temperature cannot reorder the predictions")
    print(f"    mean reported confidence {conf0.mean():.3f} against accuracy "
          f"{corr0.mean():.3f} — overconfident by {conf0.mean()-corr0.mean():+.3f}")
    print(f"    expected calibration error {e0:.4f} -> {e1:.4f} at T = {T:.3f}")
    print(f"    one parameter, fitted on validation data, and the confidences become usable")
    fig.tight_layout()
    save(fig, "02-calibration")


# ----------------------------------------------- 7. learning curves, extrapolated

def fig_learning_curve():
    Xte, yte = spiral(800, seed=7)
    sizes = np.array([15, 30, 60, 120, 240, 480, 960, 1920])
    errs = []
    for n in sizes:
        acc = []
        for s in range(4):
            Xtr, ytr = spiral(int(n), seed=100 + s)
            torch.manual_seed(s)
            m = train(mlp(width=128, depth=3), Xtr, ytr, steps=2500, seed=s)
            m.eval()
            with torch.no_grad():
                acc.append(float((m(Xte).argmax(1) == yte).float().mean()))
        errs.append(1 - np.mean(acc))
        print(f"    {int(n)*3:5d} training points -> test error {errs[-1]:.4f}")
    errs = np.array(errs)

    # fit the power law on the FIRST FOUR points only, then check the prediction
    k = 4
    lg = np.polyfit(np.log(sizes[:k]), np.log(errs[:k]), 1)
    pred = np.exp(np.polyval(lg, np.log(sizes)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.loglog(sizes * 3, errs, "o-", color=BLUE, ms=5, lw=1.9, label="measured")
    ax.loglog(sizes * 3, pred, "--", color=PINK, lw=1.6,
              label=f"power law fitted to the first {k} points")
    ax.axvline(sizes[k - 1] * 3, color=GRAY, ls=":", lw=1.2)
    ax.text(sizes[k - 1] * 3 * 1.1, errs.max(), "fitted up to here",
            fontsize=7.5, color=GRAY)
    ax.set_xlabel("training points")
    ax.set_ylabel("test error")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogx(sizes * 3, errs / pred, "o-", color=PURPLE, ms=5, lw=1.9)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.3)
    ax.set_xlabel("training points")
    ax.set_ylabel("measured error ÷ extrapolated")
    ax.set_title("where the extrapolation stops holding", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    print(f"    fitted slope on the first {k} points: {lg[0]:.3f}")
    print(f"    at {sizes[-1]*3} points the extrapolation predicts {pred[-1]:.4f} "
          f"and the measurement is {errs[-1]:.4f} — a factor of {errs[-1]/pred[-1]:.2f}")
    ratio = errs[-1] / pred[-1]
    print(f"    the extrapolation is {'OPTIMISTIC' if ratio > 1 else 'PESSIMISTIC'} by a "
          f"factor of {max(ratio, 1/ratio):.2f} at the far end")
    fig.tight_layout()
    save(fig, "04-learning-curve")


EXPENSIVE = [fig_calibration, fig_learning_curve]

if __name__ == "__main__":
    print("Part XIV — training figures")
    for f in EXPENSIVE:
        f()
