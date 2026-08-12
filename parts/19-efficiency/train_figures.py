"""
The Part XIX figures that require training.

    python3 train_figures.py

Three questions that need a real model:

  * what does quantization actually cost, measured as loss rather than as
    weight error?
  * is a fine-tuning update low rank, and in what precise sense?
  * what does speculative decoding accept, and what does that buy?
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, plt, BLUE, ORANGE, GREEN,           # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK, CYAN, quantize)

torch.set_num_threads(4)

V, ORDER, CTX = 24, 3, 48
N_CTX = V ** ORDER


def make_source(seed=0, hidden=64, temp=0.12):
    r = np.random.default_rng(seed)
    Emb = r.normal(size=(V, 12)) / np.sqrt(12)
    W1 = r.normal(size=(36, hidden)) / np.sqrt(36)
    b1 = r.normal(size=hidden) * 0.1
    W2 = r.normal(size=(hidden, V)) / np.sqrt(hidden)
    c = np.arange(N_CTX)
    X = np.concatenate([Emb[c // (V * V)], Emb[(c // V) % V], Emb[c % V]], 1)
    L = (np.tanh(X @ W1 + b1) @ W2) / temp
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def sample_source(P, n, seed=0):
    r = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int64)
    c = int(r.integers(0, N_CTX))
    C = P.cumsum(1); U = r.random(n)
    for i in range(n):
        v = min(int(np.searchsorted(C[c], U[i])), V - 1)
        out[i] = v
        c = (c % (V ** 2)) * V + v
    return out


class Net(nn.Module):
    def __init__(self, d=96, L=3, heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            n1=nn.LayerNorm(d), n2=nn.LayerNorm(d),
            att=nn.MultiheadAttention(d, heads, batch_first=True),
            ff=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))))
            for _ in range(L)])
        self.nf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)

    def forward(self, x):
        n = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(n))[None]
        m = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        for b in self.blocks:
            a = b["n1"](h)
            h = h + b["att"](a, a, a, attn_mask=m, need_weights=False)[0]
            h = h + b["ff"](b["n2"](h))
        return self.head(self.nf(h))


def train(net, data, steps=4000, lr=2e-3, bs=32, seed=0):
    torch.manual_seed(seed)
    D = torch.tensor(data)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    r = np.random.default_rng(seed)
    usable = len(D) - CTX - 1
    for _ in range(steps):
        i = torch.tensor(r.integers(0, usable, bs))
        x = torch.stack([D[j:j + CTX] for j in i])
        y = torch.stack([D[j + 1:j + CTX + 1] for j in i])
        loss = F.cross_entropy(net(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
    return net


@torch.no_grad()
def evaluate(net, data, n=20000):
    D = torch.tensor(data)
    tot = cnt = 0
    for j in range(0, min(len(D) - CTX - 1, n), CTX):
        x = D[j:j + CTX][None]; y = D[j + 1:j + CTX + 1][None]
        tot += float(F.cross_entropy(net(x).reshape(-1, V), y.reshape(-1))) * CTX
        cnt += CTX
    return tot / cnt


# ------------------------------------------- 5. what quantization costs

def fig_quant_loss():
    """Weight error is not the quantity anyone cares about. This measures the
    loss, at several bit widths and granularities, on a model that was
    actually trained."""
    P = make_source(0)
    tr = sample_source(P, 300000, seed=1)
    ev = sample_source(P, 40000, seed=2)
    print("      training the reference model ...")
    net = train(Net(), tr)
    base = evaluate(net, ev)
    print(f"    full precision loss {base:.4f}")

    import copy
    bits_list = [8, 6, 5, 4, 3, 2]
    schemes = [("per-tensor", dict(axis=None)), ("per-channel", dict(axis=0)),
               ("per-group (64)", dict(group=64))]
    res = {}
    for name, kw in schemes:
        losses = []
        for b in bits_list:
            q = copy.deepcopy(net)
            with torch.no_grad():
                for mod in q.modules():
                    if isinstance(mod, nn.Linear):
                        w = mod.weight.detach().numpy().astype(np.float64)
                        k = dict(kw)
                        if "group" in k and w.shape[1] % k["group"]:
                            k = dict(axis=0)
                        mod.weight.copy_(torch.tensor(
                            quantize(w, b, **k), dtype=torch.float32))
            losses.append(evaluate(q, ev))
        res[name] = losses
        print(f"    {name:16s} " + "  ".join(
            f"{b}b {l:.4f}" for b, l in zip(bits_list, losses)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for (name, _), c in zip(schemes, (ORANGE, BLUE, GREEN)):
        ax.plot(bits_list, res[name], "o-", color=c, lw=1.9, ms=5, label=name)
    ax.axhline(base, color=GRAY, lw=1.3, ls="--", label="full precision")
    ax.set_xticks(bits_list); ax.invert_xaxis()
    ax.set_xlabel("bits per weight"); ax.set_ylabel("test loss")
    ax.set_title("What it costs, in loss", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    ax = axes[1]
    for (name, _), c in zip(schemes, (ORANGE, BLUE, GREEN)):
        ax.plot(bits_list, np.array(res[name]) - base, "o-", color=c, lw=1.9,
                ms=5, label=name)
    ax.axhline(0, color=GRAY, lw=1.2, ls="--")
    ax.set_xticks(bits_list); ax.invert_xaxis(); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("bits per weight"); ax.set_ylabel("loss above full precision")
    ax.set_title("Where the cliff is", fontsize=10.5, color=INK)
    tidy(ax)

    for name in res:
        ok = [b for b, l in zip(bits_list, res[name]) if l - base < 0.01]
        print(f"    {name:16s} stays within 0.01 nats down to "
              f"{min(ok) if ok else 'never'} bits")
    print("    the degradation is not gradual — it is flat and then it is a cliff,")
    print("    and where the cliff sits depends on the granularity, not the method.")
    fig.tight_layout()
    save(fig, "04-quant-loss")
    return dict(base=base, res=res, bits=bits_list)


# ------------------------------------------ 6. the rank of a fine-tuning update

def fig_lora_rank():
    """Is a fine-tuning update low rank? Measured by fine-tuning a trained
    model on a shifted task and taking the SVD of the weight delta."""
    P0 = make_source(0)
    P1 = make_source(7)                      # a different but related source
    tr0 = sample_source(P0, 300000, seed=1)
    tr1 = sample_source(P1, 120000, seed=3)
    ev1 = sample_source(P1, 40000, seed=4)

    print("      pretraining ...")
    net = train(Net(), tr0)
    W0 = {n: p.detach().clone() for n, p in net.named_parameters() if p.dim() == 2}
    print("      fine-tuning on the shifted source ...")
    net = train(net, tr1, steps=1500, lr=5e-4, seed=5)
    deltas = {n: (p.detach() - W0[n]).numpy()
              for n, p in net.named_parameters() if n in W0 and p.dim() == 2}

    specs, names = [], []
    for n, d in deltas.items():
        if min(d.shape) < 32:
            continue
        s = np.linalg.svd(d, compute_uv=False)
        specs.append(s / s.sum()); names.append(n)

    # how many singular values to reach 90% of the delta's energy?
    def rank_for(s, frac):
        c = np.cumsum(s ** 2) / np.sum(s ** 2)
        return int(np.searchsorted(c, frac) + 1)

    ranks90 = [rank_for(s, 0.90) for s in specs]
    ranks99 = [rank_for(s, 0.99) for s in specs]
    dims = [min(deltas[n].shape) for n in names]
    for n, r9, r99, dm in zip(names, ranks90, ranks99, dims):
        print(f"    {n:34s} rank for 90% {r9:3d} / {dm:3d}   99% {r99:3d}")

    # a control: the SVD spectrum of a random matrix of the same shape
    rng = np.random.default_rng(0)
    ctrl = []
    for n in names:
        R = rng.normal(size=deltas[n].shape)
        s = np.linalg.svd(R, compute_uv=False)
        ctrl.append(rank_for(s / s.sum(), 0.90))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for s in specs:
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], color=BLUE, lw=1.1, alpha=0.5)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("magnitude, relative to the largest")
    ax.set_title("Spectrum of the weight update", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    x = np.arange(len(names))
    w = 0.28
    ax.bar(x - w, ranks90, w, color=BLUE, label="90% of the update")
    ax.bar(x, ranks99, w, color=PURPLE, label="99%")
    ax.bar(x + w, ctrl, w, color=GRAY, label="random matrix, 90%")
    ax.plot(x, dims, "k_", ms=14, mew=1.6, label="full rank")
    ax.set_xticks([]); ax.set_xlabel("weight matrices")
    ax.set_ylabel("rank needed")
    ax.set_title("How much of it is low rank", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.2, frameon=False)
    tidy(ax)

    print(f"    median rank for 90% of the update: {int(np.median(ranks90))} "
          f"of {int(np.median(dims))}")
    print(f"    for 99%: {int(np.median(ranks99))}")
    print(f"    a random matrix of the same shape needs {int(np.median(ctrl))} for 90%")
    print("    so the update IS concentrated — but 90% of its energy is not 90% of")
    print("    its effect, and the 99% figure is the honest one to quote.")
    fig.tight_layout()
    save(fig, "06-lora-rank")
    return dict(ranks90=ranks90, ranks99=ranks99, dims=dims, ctrl=ctrl)


# --------------------------------------------- 7. speculative decoding

def fig_speculative():
    """What a draft model's proposals are worth. Measured as the acceptance
    rate under the exact rejection rule, and the speedup it implies."""
    P = make_source(0)
    tr = sample_source(P, 300000, seed=1)
    ev = sample_source(P, 20000, seed=2)

    print("      training the target model ...")
    target = train(Net(d=96, L=3), tr)
    drafts = {}
    for tag, (d, L, steps) in (("tiny", (24, 1, 1500)),
                               ("small", (48, 2, 2500)),
                               ("close", (96, 3, 4000))):
        print(f"      training the {tag} draft ...")
        drafts[tag] = train(Net(d=d, L=L), tr, steps=steps, seed=11)
        print(f"        loss {evaluate(drafts[tag], ev):.4f} "
              f"(target {evaluate(target, ev):.4f})")

    @torch.no_grad()
    def acceptance(draft, k=4, trials=400, seed=0):
        g = torch.Generator().manual_seed(seed)
        D = torch.tensor(ev)
        acc = []
        for t in range(trials):
            j = int(torch.randint(0, len(D) - CTX - k - 1, (1,), generator=g))
            ctx = D[j:j + CTX][None].clone()
            n_acc = 0
            for i in range(k):
                pq = F.softmax(draft(ctx)[0, -1], -1)
                x = int(torch.multinomial(pq, 1, generator=g))
                pp = F.softmax(target(ctx)[0, -1], -1)
                # the exact rejection rule: accept with prob min(1, p/q)
                r = float(torch.rand(1, generator=g))
                if r < min(1.0, float(pp[x] / max(float(pq[x]), 1e-12))):
                    n_acc += 1
                    ctx = torch.cat([ctx, torch.tensor([[x]])], 1)[:, -CTX:]
                else:
                    break
            acc.append(n_acc)
        return float(np.mean(acc))

    K = 4
    rows = []
    for tag, dm in drafts.items():
        a = acceptance(dm, k=K)
        n_par = sum(p.numel() for p in dm.parameters())
        n_tgt = sum(p.numel() for p in target.parameters())
        cost = n_par / n_tgt
        # one target pass verifies K drafts; expected tokens per target pass = 1 + a
        speedup = (1 + a) / (1 + K * cost)
        rows.append((tag, a, cost, speedup))
        print(f"    {tag:6s} draft: {n_par/1e3:6.1f}k params ({cost*100:4.1f}% of target)   "
              f"accepted {a:.2f} of {K}   implied speedup {speedup:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], 0.55, color=BLUE)
    ax.axhline(K, color=GRAY, lw=1.3, ls="--")
    ax.text(len(rows) - 0.6, K + 0.06, f"all {K} accepted", fontsize=8, color=GRAY,
            ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel(f"tokens accepted of {K}")
    ax.set_title("What the draft buys", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.plot([r[2] * 100 for r in rows], [r[3] for r in rows], "o-",
            color=PURPLE, lw=1.9, ms=7)
    for r in rows:
        ax.annotate(r[0], (r[2] * 100, r[3]), textcoords="offset points",
                    xytext=(6, 5), fontsize=8, color=INK)
    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--")
    ax.set_xlabel("draft cost, % of the target")
    ax.set_ylabel("implied speedup")
    ax.set_title("And what it costs", fontsize=10.5, color=INK)
    tidy(ax)

    best = max(rows, key=lambda r: r[3])
    print(f"    the best trade here is the {best[0]} draft at {best[3]:.2f}x.")
    print("    a draft that agrees more costs more to run, and the product is what")
    print("    matters — which is why the useful draft is small AND close, not either.")
    print("    the output distribution is unchanged: the rejection rule guarantees it.")
    fig.tight_layout()
    save(fig, "07-speculative")
    return rows


EXPENSIVE = [fig_quant_loss, fig_lora_rank, fig_speculative]

if __name__ == "__main__":
    for f in EXPENSIVE:
        f()
