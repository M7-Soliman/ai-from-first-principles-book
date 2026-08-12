"""
The Part XX figures that require training a model.

    python3 train_figures.py

Three questions, all asked on a synthetic language whose rules are known, so
"did it learn the rule or memorise the examples" has an answer:

  * does in-context learning use the demonstrations' label mapping, or only
    their format?
  * does producing intermediate steps help on a task that needs them?
  * how many repetitions does it take before a sequence is memorised verbatim?
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, plt, BLUE, ORANGE, GREEN,           # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK, CYAN)

torch.set_num_threads(4)


class LM(nn.Module):
    def __init__(self, V, d=128, L=4, heads=4, ctx=128):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(ctx, d)
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


def run_training(model, batches, steps, lr=2e-3, seed=0, mask_fn=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for i in range(steps):
        x, y = batches(i)
        logits = model(x)
        if mask_fn is None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   y[:, 1:].reshape(-1))
        else:
            loss = mask_fn(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
    return model


# ------------------------------------------------- 3. in-context learning

def fig_icl():
    """Does a model trained on many small classification tasks learn to use the
    demonstrations' LABEL MAPPING, or only their format? Measured by permuting
    the labels at test time, which is the experiment that settles it."""
    N_SYM, N_CLASS, N_SHOT = 40, 4, 8
    SEP, ARROW = N_SYM, N_SYM + 1
    LAB0 = N_SYM + 2
    V = LAB0 + N_CLASS
    CTX = (N_SHOT + 1) * 4

    def make_task(rng):
        """A fresh random mapping from symbol to class, unseen at training."""
        return rng.integers(0, N_CLASS, N_SYM)

    def make_example(rng, mapping, permute=None, random_labels=False):
        # the QUERY must appear among the demonstrations, or the task is
        # unanswerable: with 40 symbols and 8 shots the query would otherwise
        # be undemonstrated four times in five, and the ceiling would be chance
        # plus a fifth. The first version of this figure did exactly that and
        # measured nothing.
        xs = rng.integers(0, N_SYM, N_SHOT + 1)
        xs[-1] = xs[int(rng.integers(0, N_SHOT))]
        seq, tgt = [], []
        for i, x in enumerate(xs):
            lab = int(mapping[x])
            if i < N_SHOT:
                if random_labels:
                    lab = int(rng.integers(0, N_CLASS))
                elif permute is not None:
                    lab = int(permute[lab])
            seq += [int(x), ARROW, LAB0 + lab, SEP]
            tgt.append(len(seq) - 2)          # position of the label token
        return seq, tgt, int(mapping[xs[-1]])

    def batch(rng, bs=64, **kw):
        S, P, Y = [], [], []
        for _ in range(bs):
            m = make_task(rng)
            s, t, y = make_example(rng, m, **kw)
            S.append(s); P.append(t[-1]); Y.append(y)
        return torch.tensor(S), torch.tensor(P), torch.tensor(Y)

    torch.manual_seed(0)                  # the caption quotes these numbers
    r = np.random.default_rng(0)
    model = LM(V, ctx=CTX)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    STEPS = 6000
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    print("      training on many fresh tasks ...")
    for step in range(STEPS):
        S, P, Y = batch(r)
        logits = model(S)
        pred = logits[torch.arange(len(S)), P - 1]     # predict the label token
        loss = F.cross_entropy(pred, Y + LAB0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 2000 == 0:
            print(f"        step {step:5d}  loss {loss.item():.4f}")

    @torch.no_grad()
    def evaluate(n=3000, **kw):
        rr = np.random.default_rng(1234)
        ok = 0
        for _ in range(n // 64):
            S, P, Y = batch(rr, 64, **kw)
            pred = model(S)[torch.arange(len(S)), P - 1].argmax(-1) - LAB0
            ok += int((pred == Y).sum())
        return ok / ((n // 64) * 64)

    perm = np.array([(c + 1) % N_CLASS for c in range(N_CLASS)])
    conds = [("correct labels", dict()),
             ("permuted labels", dict(permute=perm)),
             ("random labels", dict(random_labels=True))]
    res = {}
    for name, kw in conds:
        res[name] = evaluate(**kw)
        print(f"    {name:18s} accuracy {res[name]:.3f}")
    # and with fewer demonstrations
    shots = []
    print("    (chance is %.3f)" % (1 / N_CLASS))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x, [res[n] for n, _ in conds], 0.55,
           color=[GREEN, ORANGE, GRAY])
    ax.axhline(1 / N_CLASS, color=GRAY, lw=1.3, ls="--")
    ax.text(2.4, 1 / N_CLASS + 0.02, "chance", fontsize=8, color=GRAY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in conds], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy on the held-out query")
    ax.set_title("What the demonstrations supply", fontsize=10.5, color=INK)
    tidy(ax)

    # accuracy against number of demonstrations
    @torch.no_grad()
    def eval_shots(k, n=1600):
        rr = np.random.default_rng(7)
        ok = tot = 0
        for _ in range(n // 64):
            S, P, Y = [], [], []
            for _ in range(64):
                m = make_task(rr)
                xs = rr.integers(0, N_SYM, k + 1)
                if k > 0:
                    xs[-1] = xs[int(rr.integers(0, k))]
                seq = []
                for i, xv in enumerate(xs):
                    seq += [int(xv), ARROW, LAB0 + int(m[xv]), SEP]
                S.append(seq); P.append(len(seq) - 2); Y.append(int(m[xs[-1]]))
            S = torch.tensor(S); P = torch.tensor(P); Y = torch.tensor(Y)
            pred = model(S)[torch.arange(len(S)), P - 1].argmax(-1) - LAB0
            ok += int((pred == Y).sum()); tot += len(S)
        return ok / tot

    ks = [0, 1, 2, 4, 8]
    accs = [eval_shots(k) for k in ks]
    for k, a in zip(ks, accs):
        print(f"    {k:2d} demonstrations: accuracy {a:.3f}")
    ax = axes[1]
    ax.plot(ks, accs, "o-", color=BLUE, lw=1.9, ms=5)
    ax.axhline(1 / N_CLASS, color=GRAY, lw=1.3, ls="--")
    ax.set_xlabel("demonstrations in the prompt")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Finding the relevant one gets harder", fontsize=10.5, color=INK)
    tidy(ax)

    print("    the mapping is fresh at test time, so nothing about it is in the")
    print("    weights. Permuting the labels drops accuracy BELOW chance, because the")
    print("    model systematically returns the permuted answer — which is what says")
    print("    it is reading the mapping rather than the format.")
    print("    note the second panel falls rather than rises. In this construction the")
    print("    query is always demonstrated, so one shot is a trivial copy and more")
    print("    shots make it a retrieval problem. It measures how hard the relevant")
    print("    demonstration is to find, not how many demonstrations help.")
    fig.tight_layout()
    save(fig, "03-in-context")
    return dict(res=res, ks=ks, accs=accs)


# ------------------------------------------------------ 4. chain of thought

def fig_cot():
    """A task that needs several steps. Train two models on identical data,
    one shown the intermediate steps and one only the answer, and measure."""
    MOD = 10
    N_STEP_MAX = 6
    V = MOD + 3
    EQ, SEP, PAD = MOD, MOD + 1, MOD + 2

    def make(rng, n_steps, show_work):
        xs = rng.integers(0, MOD, n_steps + 1)
        run = int(xs[0])
        steps = []
        for v in xs[1:]:
            run = (run + int(v)) % MOD
            steps.append(run)
        seq = list(map(int, xs)) + [EQ]
        if show_work:
            seq += steps[:-1]
        seq += [steps[-1]]
        return seq, len(seq) - 1, steps[-1]

    def batch(rng, bs, n_steps, show_work):
        S, P, Y = [], [], []
        for _ in range(bs):
            s, p, y = make(rng, n_steps, show_work)
            S.append(s); P.append(p); Y.append(y)
        L = max(len(s) for s in S)
        S = [s + [PAD] * (L - len(s)) for s in S]
        return torch.tensor(S), torch.tensor(P), torch.tensor(Y)

    def train_one(show_work, steps=5000, seed=0):
        torch.manual_seed(seed)
        r = np.random.default_rng(seed)
        m = LM(V, d=128, L=4, ctx=3 * N_STEP_MAX + 8)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for i in range(steps):
            ns = int(r.integers(2, N_STEP_MAX + 1))
            S, P, Y = batch(r, 64, ns, show_work)
            logits = m(S)
            if show_work:
                # supervise every intermediate step too
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                       S[:, 1:].reshape(-1))
            else:
                loss = F.cross_entropy(logits[torch.arange(len(S)), P - 1], Y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
        return m

    @torch.no_grad()
    def acc(m, n_steps, show_work, n=1280):
        rr = np.random.default_rng(99)
        ok = tot = 0
        for _ in range(n // 64):
            S, P, Y = batch(rr, 64, n_steps, show_work)
            if show_work:
                # generate the work, then read the final answer
                cur = S[:, :n_steps + 2].clone()
                for _ in range(n_steps):
                    nxt = m(cur)[:, -1].argmax(-1, keepdim=True)
                    cur = torch.cat([cur, nxt], 1)
                pred = cur[:, -1]
            else:
                pred = m(S)[torch.arange(len(S)), P - 1].argmax(-1)
            ok += int((pred == Y).sum()); tot += len(S)
        return ok / tot

    print("      training the direct-answer model ...")
    m_direct = train_one(False)
    print("      training the show-the-work model ...")
    m_work = train_one(True)

    ns = [2, 3, 4, 5, 6]
    a_direct = [acc(m_direct, n, False) for n in ns]
    a_work = [acc(m_work, n, True) for n in ns]
    for n, d, w in zip(ns, a_direct, a_work):
        print(f"    {n} steps: direct answer {d:.3f}   with the work shown {w:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(ns, a_direct, "o-", color=ORANGE, lw=1.9, ms=5, label="answer only")
    ax.plot(ns, a_work, "s-", color=GREEN, lw=1.9, ms=5, label="steps then answer")
    ax.axhline(1 / MOD, color=GRAY, lw=1.3, ls="--", label="chance")
    ax.set_xlabel("steps the problem requires"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Producing the steps is the computation",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(ns, np.array(a_work) - np.array(a_direct), "o-", color=PURPLE,
            lw=1.9, ms=5)
    ax.axhline(0, color=GRAY, lw=1.2, ls="--")
    ax.set_xlabel("steps the problem requires")
    ax.set_ylabel("accuracy gained by writing the steps")
    ax.set_title("The gap persists — it does not widen with depth",
                 fontsize=10.5, color=INK)
    tidy(ax)

    print("    both models saw identical problems and identical answers. The only")
    print("    difference is whether the intermediate values were written down —")
    print("    which gives the model more forward passes to compute with.")
    fig.tight_layout()
    save(fig, "04-chain-of-thought")
    return dict(ns=ns, direct=a_direct, work=a_work)


EXPENSIVE = [fig_icl, fig_cot]

if __name__ == "__main__":
    for f in EXPENSIVE:
        f()
