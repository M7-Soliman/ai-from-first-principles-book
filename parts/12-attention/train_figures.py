"""
The Part XII figures that require training. Imported by make_figures.py.

    python3 train_figures.py        # just these

Everything here trains a transformer small enough to read completely — a few
thousand parameters, on synthetic tasks whose structure is known exactly, so
that when a head does something we can say what.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, noaxes, plt, BLUE, ORANGE, GREEN,   # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK)

torch.set_num_threads(4)


# ------------------------------------------------------------ a tiny transformer

class Block(nn.Module):
    def __init__(self, d, n_head, pre_norm=True, mult=4):
        super().__init__()
        self.pre_norm = pre_norm
        self.n_head = n_head
        self.d = d
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                nn.Linear(mult * d, d))
        self.last_attn = None

    def attn(self, x):
        B, T, d = x.shape
        h = self.n_head
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, h, d // h).transpose(1, 2)
        k = k.view(B, T, h, d // h).transpose(1, 2)
        v = v.view(B, T, h, d // h).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / (d // h) ** 0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
        a = F.softmax(s, dim=-1)
        self.last_attn = a.detach()
        return self.proj((a @ v).transpose(1, 2).contiguous().view(B, T, d))

    def forward(self, x):
        if self.pre_norm:
            x = x + self.attn(self.n1(x))
            return x + self.ff(self.n2(x))
        x = self.n1(x + self.attn(x))
        return self.n2(x + self.ff(x))


class Tiny(nn.Module):
    def __init__(self, vocab, d=64, n_layer=2, n_head=2, T=64, pre_norm=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, n_head, pre_norm) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def copy_task(n, T, vocab=12, seed=0, fixed_offset=False):
    """A repeated block at a RANDOM offset — the canonical induction-head task.

    The offset varies per sequence, which is the whole point. With a fixed
    offset a one-layer model solves the task from the positional embedding
    alone ("attend to position t - 16") and no induction head is needed. Making
    the offset random removes that shortcut: the only remaining strategy is to
    find the earlier occurrence of the CURRENT token and copy what followed it,
    and that composition needs two layers.
    """
    r = np.random.default_rng(seed)
    out = np.empty((n, T), dtype=np.int64)
    for i in range(n):
        if fixed_offset:
            start = T // 2
        else:
            start = int(r.integers(T // 4, T // 2 + 1))
        blk = r.integers(2, vocab, start)
        reps = int(np.ceil(T / start))         # tile until it is long enough
        out[i] = np.tile(blk, reps)[:T]
    return torch.tensor(out)


def train(model, X, steps=800, lr=3e-3, bs=64, warmup=0, log_every=0, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for s in range(steps):
        if warmup:
            for gp in opt.param_groups:
                gp["lr"] = lr * min(1.0, (s + 1) / warmup)
        idx = torch.randint(0, len(X), (bs,))
        b = X[idx]
        logits = model(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and s % log_every == 0:
            hist.append((s, float(loss)))
    return hist


def per_position_loss(model, X):
    with torch.no_grad():
        logits = model(X[:, :-1])
        ls = F.cross_entropy(logits.transpose(1, 2), X[:, 1:], reduction="none")
        return ls.mean(0).numpy()


# --------------------------------------------- 8. the induction head, and the elbow

def fig_induction():
    T, vocab = 32, 12
    X = copy_task(3000, T, vocab, seed=1)

    Xfix = copy_task(3000, T, vocab, seed=1, fixed_offset=True)
    curves, fixed_curves = {}, {}
    for n_layer in (1, 2):
        torch.manual_seed(0)
        m = Tiny(vocab, d=64, n_layer=n_layer, n_head=2, T=T)
        curves[n_layer] = train(m, X, steps=2500, log_every=20, seed=0)
        if n_layer == 2:
            two_layer, loss2 = m, per_position_loss(m, X)
        else:
            loss1 = per_position_loss(m, X)
        # the control: the SAME model on a fixed offset, where position suffices
        torch.manual_seed(0)
        mf = Tiny(vocab, d=64, n_layer=n_layer, n_head=2, T=T)
        fixed_curves[n_layer] = train(mf, Xfix, steps=2500, log_every=20, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    ax = axes[0]
    for nl, c in ((1, ORANGE), (2, BLUE)):
        st, l = zip(*curves[nl])
        ax.plot(st, l, color=c, lw=1.8,
                label=f"{nl} layer" + ("s" if nl > 1 else "") + ", random offset")
        st, l = zip(*fixed_curves[nl])
        ax.plot(st, l, color=c, lw=1.2, ls=":", alpha=0.8,
                label=f"{nl} layer" + ("s" if nl > 1 else "") + ", fixed offset")
    ax.axhline(np.log(vocab - 2), color=GRAY, ls=":", lw=1.2)
    ax.text(20, np.log(vocab - 2) * 1.03, "chance", fontsize=7.5, color=GRAY)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(loss1, "o-", color=ORANGE, ms=3, lw=1.6, label="1 layer")
    ax.plot(loss2, "o-", color=BLUE, ms=3, lw=1.6, label="2 layers")
    ax.axvline(T // 2 - 1, color=PINK, ls="--", lw=1.3)
    ax.text(T // 2 + 0.5, ax.get_ylim()[1] * 0.9, "the repeat begins",
            fontsize=7.5, color=PINK)
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("loss at that position")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[2]
    with torch.no_grad():
        two_layer(X[:1, :-1])
    A = two_layer.blocks[-1].last_attn[0].mean(0).numpy()
    im = ax.imshow(A, cmap="magma", vmin=0)
    ax.set_xlabel("attending to position")
    ax.set_ylabel("from position")
    ax.set_title("layer 2, averaged over heads", fontsize=9, color=INK, pad=6)
    ax.tick_params(labelsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    print(f"    RANDOM offset: 1 layer plateaus at {curves[1][-1][1]:.3f}, "
          f"2 layers reach {curves[2][-1][1]:.3f}")
    print(f"    FIXED  offset: 1 layer {fixed_curves[1][-1][1]:.3f}, "
          f"2 layers {fixed_curves[2][-1][1]:.3f}  <- position alone suffices here")
    print(f"    so the one-layer gap is specific to the offset being unpredictable")

    # the induction signature: for each position in the repeat, does attention
    # land on the token AFTER the previous occurrence of the current token?
    with torch.no_grad():
        Xs = X[:200]
        two_layer(Xs[:, :-1])
        A_all = two_layer.blocks[-1].last_attn.mean(1)          # (B, T-1, T-1)
    hits, base = [], []
    for b in range(len(Xs)):
        seq = Xs[b].tolist()
        for t in range(2, T - 1):
            prev = [j for j in range(t) if seq[j] == seq[t]]
            if not prev:
                continue
            tgt = prev[-1] + 1
            if tgt >= t:
                continue
            hits.append(float(A_all[b, t, tgt]))
            base.append(1.0 / (t + 1))
    hit, bs = float(np.mean(hits)), float(np.mean(base))
    print(f"    INDUCTION SIGNATURE — layer-2 attention onto 'the token after the "
          f"previous occurrence of the current token': {hit:.3f}")
    print(f"    against {bs:.3f} for uniform attention over the same prefix — "
          f"{hit/max(bs,1e-9):.1f}x")
    fig.tight_layout()
    save(fig, "10-induction")


# ------------------------------------------------ 9. pre-norm, post-norm, warmup

def fig_norm_placement():
    T, vocab = 32, 12
    X = copy_task(2000, T, vocab, seed=2)

    # gradient norm at initialisation, by depth
    depths = [1, 2, 4, 8, 16]
    gpre, gpost = [], []
    for L in depths:
        for pre, store in ((True, gpre), (False, gpost)):
            torch.manual_seed(0)
            m = Tiny(vocab, d=64, n_layer=L, n_head=2, T=T, pre_norm=pre)
            b = X[:32]
            logits = m(b[:, :-1])
            F.cross_entropy(logits.reshape(-1, vocab), b[:, 1:].reshape(-1)).backward()
            g = m.blocks[0].qkv.weight.grad
            store.append(float(g.norm()))

    # does post-norm need warmup?
    runs = {}
    DEPTH, LR = 12, 1e-2          # deep enough, and a rate post-norm cannot take cold
    for pre in (True, False):
        for wu in (0, 100):
            torch.manual_seed(0)
            m = Tiny(vocab, d=64, n_layer=DEPTH, n_head=2, T=T, pre_norm=pre)
            runs[(pre, wu)] = train(m, X, steps=500, lr=LR, warmup=wu,
                                    log_every=10, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.semilogy(depths, gpre, "o-", color=BLUE, ms=5, lw=1.9, label="pre-norm")
    ax.semilogy(depths, gpost, "o-", color=ORANGE, ms=5, lw=1.9, label="post-norm")
    ax.set_xlabel("depth (blocks)")
    ax.set_ylabel("gradient norm at layer 1, at init")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for (pre, wu), c, ls in (((True, 0), BLUE, "-"), ((True, 100), BLUE, "--"),
                             ((False, 0), ORANGE, "-"), ((False, 100), ORANGE, "--")):
        s, l = zip(*runs[(pre, wu)])
        ax.plot(s, l, color=c, lw=1.7, ls=ls,
                label=("pre" if pre else "post") + ("-norm, warmup" if wu else "-norm, none"))
    ax.set_xlabel("training step  (12 blocks, lr $10^{-2}$)")
    ax.set_ylabel("loss")
    ax.legend(fontsize=7.5, frameon=False)
    tidy(ax)

    print(f"    gradient at layer 1, depth {depths[-1]}: pre-norm {gpre[-1]:.3e}, "
          f"post-norm {gpost[-1]:.3e} — ratio {gpost[-1]/max(gpre[-1],1e-30):.2f}")
    for (pre, wu), h in runs.items():
        print(f"    {'pre ' if pre else 'post'}-norm, warmup {wu:3d}: "
              f"final loss {h[-1][1]:.3f}")
    fig.tight_layout()
    save(fig, "07-norm-placement")


# --------------------------------------------------------- 10. rank collapse

def fig_rank_collapse():
    """Attention without skip connections collapses to rank one with depth."""
    torch.manual_seed(0)
    n, d = 64, 32
    depths = list(range(1, 33))

    def run(skip, seed=0):
        torch.manual_seed(seed)
        X = torch.randn(n, d)
        ranks, ents = [], []
        for L in depths:
            Wq, Wk, Wv = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
            q, k, v = X @ Wq, X @ Wk, X @ Wv
            A = F.softmax(q @ k.T / d ** 0.5, -1)
            Y = A @ v
            X = X + Y if skip else Y
            Xc = X - X.mean(0, keepdim=True)
            sv = torch.linalg.svdvals(Xc)
            p = sv / sv.sum()
            ents.append(float(torch.exp(-(p * (p + 1e-30).log()).sum())))
            ranks.append(float(sv[0] / sv.sum()))
        return ranks, ents

    r_no, e_no = run(False)
    r_sk, e_sk = run(True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(depths, e_no, "o-", color=ORANGE, ms=3.5, lw=1.8, label="attention only")
    ax.plot(depths, e_sk, "o-", color=BLUE, ms=3.5, lw=1.8, label="with skip connections")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.2)
    ax.text(1, 1.15, "rank one", fontsize=7.5, color=GRAY)
    ax.set_xlabel("depth")
    ax.set_ylabel("effective rank of the representation")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.semilogy(depths, r_no, "o-", color=ORANGE, ms=3.5, lw=1.8, label="attention only")
    ax.semilogy(depths, r_sk, "o-", color=BLUE, ms=3.5, lw=1.8, label="with skip connections")
    ax.set_xlabel("depth")
    ax.set_ylabel("share of the spectrum in the top direction")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    effective rank after {depths[-1]} layers: attention alone "
          f"{e_no[-1]:.2f}, with skips {e_sk[-1]:.2f} (started near {e_no[0]:.1f})")
    print(f"    top singular direction holds {100*r_no[-1]:.1f}% of the spectrum "
          f"without skips, {100*r_sk[-1]:.1f}% with")
    fig.tight_layout()
    save(fig, "08-rank-collapse")


# --------------------------------------------------- 11. in-context learning

def fig_in_context():
    """A model trained on many short tasks learns to infer the rule from the prompt.

    Each sequence is drawn with its OWN random permutation of the vocabulary, so
    the model never sees the same rule twice and cannot memorise any of them.

    The measurement that matters conditions on whether the rule for the current
    query has been DEMONSTRATED yet. Averaging over all positions dilutes the
    effect with queries the model has been given no information about, and no
    amount of in-context learning can help there.
    """
    vocab, T, n_seq = 10, 40, 8000

    def make(n, seed):
        r = np.random.default_rng(seed)
        out = np.empty((n, T), dtype=np.int64)
        for i in range(n):
            perm = r.permutation(vocab)          # a fresh rule per sequence
            a = r.integers(0, vocab, T // 2)
            seq = np.empty(T, dtype=np.int64)
            seq[0::2] = a
            seq[1::2] = perm[a]                  # the answer follows each query
            out[i] = seq
        return torch.tensor(out)

    X = make(n_seq, 3)
    torch.manual_seed(0)
    m = Tiny(vocab, d=96, n_layer=2, n_head=4, T=T)
    train(m, X, steps=3000, lr=3e-3, seed=0)

    Xt = make(1000, 99)                          # rules never seen in training
    with torch.no_grad():
        pred = m(Xt[:, :-1]).argmax(-1)
        corr = (pred == Xt[:, 1:]).float().numpy()

    # bucket each answer position by how many times its query appeared earlier
    seen_acc, unseen_acc, by_shown = {}, [], {}
    for b in range(len(Xt)):
        seq = Xt[b].tolist()
        for j in range(0, T - 1, 2):             # position j predicts the answer
            q = seq[j]
            shown = sum(1 for jj in range(0, j, 2) if seq[jj] == q)
            ok = corr[b, j]
            by_shown.setdefault(min(shown, 5), []).append(ok)
            (unseen_acc if shown == 0 else seen_acc.setdefault(1, [])).append(ok)

    shown_keys = sorted(by_shown)
    shown_vals = [float(np.mean(by_shown[k])) for k in shown_keys]
    a_unseen = float(np.mean(unseen_acc))
    a_seen = float(np.mean(seen_acc[1]))

    # and the position curve, for the aggregate view
    pos_acc = [float(corr[:, j].mean()) for j in range(0, T - 1, 2)]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.bar(shown_keys, shown_vals, 0.65, color=BLUE)
    ax.axhline(1 / vocab, color=GRAY, ls=":", lw=1.2)
    ax.text(0.6, 1 / vocab + 0.03, "chance", fontsize=7.5, color=GRAY)
    ax.set_xticks(shown_keys)
    ax.set_xticklabels([str(k) if k < 5 else "5+" for k in shown_keys])
    ax.set_xlabel("times this query's answer was shown earlier")
    ax.set_ylabel("accuracy on the answer")
    ax.set_ylim(0, 1.02)
    tidy(ax)

    ax = axes[1]
    ax.plot(np.arange(len(pos_acc)) + 1, pos_acc, "o-", color=GREEN, ms=4, lw=1.8)
    ax.axhline(1 / vocab, color=GRAY, ls=":", lw=1.2)
    ax.text(1, 1 / vocab + 0.03, "chance", fontsize=7.5, color=GRAY)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("query number within the sequence")
    ax.set_ylabel("accuracy, averaged over all queries")
    ax.set_title("the aggregate view dilutes it", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    print(f"    rules at test time were never trained on; no weight changed during "
          f"any of this")
    print(f"    query never demonstrated: accuracy {a_unseen:.3f} "
          f"(chance {1/vocab:.3f}) — correctly, it cannot know")
    print(f"    query demonstrated at least once: {a_seen:.3f}")
    print(f"    by number of demonstrations: " +
          ", ".join(f"{k if k<5 else '5+'}:{v:.2f}" for k, v in zip(shown_keys, shown_vals)))
    fig.tight_layout()
    save(fig, "11-in-context")


EXPENSIVE = [fig_induction, fig_norm_placement, fig_rank_collapse, fig_in_context]

if __name__ == "__main__":
    print("Part XII — training figures")
    for f in EXPENSIVE:
        f()
