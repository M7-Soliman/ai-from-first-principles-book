"""
The Part XIII figures that require training. Imported by make_figures.py.

    python3 train_figures.py

One question that can only be answered by training: how many copies of a
sequence it takes before a model reproduces it verbatim.

A second figure on repeated data was attempted and dropped: on a rule simple
enough to train here, 125 examples already suffice, so the curve is flat and
says nothing. Reporting a flat curve as evidence about data scaling would have
been worse than omitting it. Part XVII measures it properly.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, plt, BLUE, ORANGE, GREEN,          # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK)

torch.set_num_threads(4)


class Tiny(nn.Module):
    """A small causal transformer — enough to memorise, small enough to be honest."""

    def __init__(self, vocab, d=96, n_layer=2, n_head=4, T=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T))
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return self.head(self.nf(self.enc(x, mask=m)))


def _train(m, X, steps=None, epochs=None, lr=3e-3, bs=64, seed=0):
    """Either a fixed step budget (steps=) or a fixed number of passes (epochs=).

    The distinction matters enormously for fig_memorisation: sampling randomly
    for a fixed number of STEPS shows every sequence the same number of times on
    average, whatever its duplication count, which destroys the thing being
    measured. Duplication only maps to exposure under a fixed number of PASSES.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)

    def step(b):
        logits = m(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    if epochs is not None:
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), bs):
                step(X[perm[i:i + bs]])
    else:
        for _ in range(steps):
            step(X[torch.randint(0, len(X), (bs,))])
    return m


# ------------------------------------------------- 5. memorisation and duplicates

def fig_memorisation():
    """How many copies does it take before the model reproduces a sequence?

    Canaries are random sequences inserted a controlled number of times into an
    otherwise structured corpus. Extraction: prompt with the first half and ask
    whether the model completes the second half exactly.
    """
    vocab, T = 40, 32
    EPOCHS = 4
    rng = np.random.default_rng(0)
    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    REPS = 3          # canaries per count, averaged — a single one is noisy

    # background corpus with real structure, so the model has something else to learn
    n_bg = 6000
    bg = np.zeros((n_bg, T), dtype=np.int64)
    for i in range(n_bg):
        a = rng.integers(2, vocab, T // 2)
        bg[i] = np.concatenate([a, a])                      # a learnable rule

    canaries = rng.integers(2, vocab, (len(counts), REPS, T))   # no structure at all
    # a sequence appearing c times is seen c * EPOCHS times in total
    rows = [bg]
    for c, grp in zip(counts, canaries):
        for seq in grp:
            rows.append(np.tile(seq, (c, 1)))
    X = torch.tensor(np.concatenate(rows))

    torch.manual_seed(0)
    m = Tiny(vocab, T=T)
    _train(m, X, epochs=EPOCHS, seed=0)      # passes, not steps
    m.eval()

    half = T // 2
    exact, nll = [], []
    with torch.no_grad():
        for grp in canaries:
            es, ns = [], []
            for seq in grp:
                sq = torch.tensor(seq)[None]
                logits = m(sq[:, :-1])
                ls = F.cross_entropy(logits[0], sq[0, 1:], reduction="none")
                ns.append(float(ls[half - 1:].mean()))       # loss on the second half
                cur = sq[:, :half].clone()                    # greedy completion
                for _ in range(T - half):
                    nxt = m(cur)[0, -1].argmax()
                    cur = torch.cat([cur, nxt.view(1, 1)], 1)
                es.append(float((cur[0, half:] == sq[0, half:]).float().mean()))
            exact.append(float(np.mean(es)))
            nll.append(float(np.mean(ns)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.semilogx(counts, exact, "o-", color=PINK, ms=6, lw=1.9, base=2)
    ax.axhline(1 / (vocab - 2), color=GRAY, ls=":", lw=1.2)
    ax.text(1.1, 1 / (vocab - 2) + 0.03, "chance", fontsize=7.5, color=GRAY)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("copies of the sequence in the training set")
    ax.set_ylabel("fraction of the held-out half reproduced exactly")
    tidy(ax)

    ax = axes[1]
    ax.semilogx(counts, nll, "o-", color=BLUE, ms=6, lw=1.9, base=2)
    ax.axhline(np.log(vocab - 2), color=GRAY, ls=":", lw=1.2)
    ax.text(1.1, np.log(vocab - 2) * 1.02, "no information", fontsize=7.5, color=GRAY)
    ax.set_xlabel("copies of the sequence in the training set")
    ax.set_ylabel("loss on the unseen half (nats)")
    tidy(ax)

    for c, e, l in zip(counts, exact, nll):
        print(f"    {c:3d} copies | reproduced {e:.2f} | loss {l:.3f} "
              f"(chance {np.log(vocab-2):.3f})")
    print(f"    corpus {len(X)} sequences, {EPOCHS} passes — so a sequence with c "
          f"copies is seen c x {EPOCHS} times")
    print(f"    even 128 copies is {100*128/len(X):.2f}% of the corpus")
    fig.tight_layout()
    save(fig, "06-memorisation")


# ------------------------------------------------ 6. repeating data against fresh

EXPENSIVE = [fig_memorisation]

if __name__ == "__main__":
    print("Part XIII — training figures")
    for f in EXPENSIVE:
        f()
