"""
The Part XXI figures that require training a model.

    python3 train_figures.py

Four questions, all asked on systems whose ground truth is known by
construction, because interpretability claims are exactly the claims that
cannot be checked when the ground truth is not known:

  * can a sparse autoencoder recover features from superposition, and does
    anything simpler? (Part XV measured superposition itself; this scores the
    instrument against a known answer, which that part did not do.)
  * does a probe finding a feature mean the network uses it?
  * where in the network does a specific piece of information live?
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


# ---------------------------------- the model the figures are measured on

class ToyModel(nn.Module):
    """The standard toy model of superposition: project m features down to n
    dimensions and try to reconstruct them. With n < m this is impossible in
    general, so what the model does with the room it has is the question."""

    def __init__(self, m, n):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, m))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(m))

    def forward(self, x):
        h = x @ self.W.T
        return F.relu(h @ self.W + self.b)


def measure_superposition():
    """Sweep sparsity and measure how many of the m features the model keeps.
    Dense inputs: it keeps n of them and discards the rest. Sparse inputs: it
    keeps nearly all m, in overlapping directions that interfere.

    Prints only — see the note below on why there is no figure. \u00a73
    quotes Part XV's own run (16 kept at 99% sparsity, mean interference
    0.345), not this one — this is a second, independent re-run of the
    same experiment, and the drills (W5, C4) quote ITS numbers, close but
    not identical (17 and 0.354 at the same sparsity)."""
    print("\n[*] superposition, a second run of Part XV \u00a733's experiment,")
    print("    for the numbers this part's drills quote (no figure: Part XV")
    print("    Figure 10 already shows this)")
    M, N = 20, 5
    STEPS, BS = 6000, 1024
    imp = torch.tensor(0.85 ** np.arange(M), dtype=torch.float32)
    sparsities = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]

    kept, mats = [], {}
    for S in sparsities:
        torch.manual_seed(0)
        model = ToyModel(M, N)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        g = torch.Generator().manual_seed(1)
        for _ in range(STEPS):
            x = torch.rand(BS, M, generator=g)
            x = x * (torch.rand(BS, M, generator=g) > S)
            loss = (imp * (x - model(x)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        W = model.W.detach()
        norms = W.norm(dim=0)                       # per-feature representation
        n_kept = int((norms > 0.5 * norms.max()).sum())
        kept.append(n_kept)
        # how non-orthogonal the stored directions are
        Wn = W / (norms.clamp_min(1e-8))
        G = (Wn.T @ Wn).numpy()
        off = np.abs(G[~np.eye(M, dtype=bool)]).mean()
        mats[S] = G
        print(f"    sparsity {S:<6.3f}  features kept {n_kept:2d} of {M}"
              f"   (n = {N} dimensions)   mean |overlap| {off:.3f}")

    # NO FIGURE HERE. Part XV Figure 10 is this same experiment — twenty
    # features, five dimensions, the same sparsity sweep — and shipping it
    # twice would be padding. The numbers are printed because this part's
    # drills (W5, C4) quote them (§3 quotes Part XV's own, different, run of
    # the same experiment — 16 and 0.345 — not this one), and because
    # fig_dictionary needs the model.
    print("    (no figure written: Part XV Figure 10 already shows this)")
    return M, N


# ------------------------------------------------ 1. recovering the features

def fig_dictionary(M=20, N=5):
    """Train a toy model in the superposition regime, then ask whether a sparse
    autoencoder can recover the features from its activations — and whether
    anything simpler can. PCA is the control, and it cannot: a 5-dimensional
    space has 5 principal components and there are 20 features."""
    print("\n[1] recovering features from superposition")
    STEPS, BS, S = 6000, 1024, 0.99
    imp = torch.tensor(0.85 ** np.arange(M), dtype=torch.float32)

    torch.manual_seed(0)
    model = ToyModel(M, N)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        x = torch.rand(BS, M, generator=g)
        x = x * (torch.rand(BS, M, generator=g) > S)
        loss = (imp * (x - model(x)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    W = model.W.detach()
    true_dirs = (W / W.norm(dim=0).clamp_min(1e-8)).T          # M x N, unit rows

    with torch.no_grad():
        x = torch.rand(40000, M, generator=g)
        x = x * (torch.rand(40000, M, generator=g) > S)
        H = x @ W.T                                            # activations

    def recovery(dirs):
        """For each true feature, the best cosine match among the found ones."""
        d = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = (true_dirs @ d.T).abs().max(dim=1).values
        return sim.numpy()

    # --- control: PCA
    Hc = H - H.mean(0)
    U, Sv, Vt = torch.linalg.svd(Hc, full_matrices=False)
    pca_rec = recovery(Vt)

    # --- sparse autoencoder
    results = {}
    for DICT in (10, 20, 40, 80):
        torch.manual_seed(0)
        enc = nn.Linear(N, DICT)
        dec = nn.Linear(DICT, N, bias=False)
        with torch.no_grad():
            dec.weight.copy_(enc.weight.T)
        o = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, 4000)
        for i in range(4000):
            idx = torch.randint(0, len(H), (512,), generator=g)
            h = H[idx]
            a = F.relu(enc(h))
            rec = dec(a)
            l = ((rec - h) ** 2).sum(1).mean() + 3e-2 * a.abs().sum(1).mean()
            o.zero_grad(); l.backward()
            with torch.no_grad():                       # unit-norm dictionary
                dec.weight.div_(dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
            o.step(); sc.step()
        rec = recovery(dec.weight.T.detach())
        results[DICT] = rec
        print(f"    SAE, {DICT:3d} dictionary entries: "
              f"{int((rec > 0.9).sum()):2d} of {M} features recovered "
              f"(cosine > 0.9), mean best cosine {rec.mean():.3f}")
    print(f"    PCA, {N:3d} components          : "
          f"{int((pca_rec > 0.9).sum()):2d} of {M} features recovered, "
          f"mean best cosine {pca_rec.mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ds = sorted(results)
    ax.plot(ds, [int((results[d] > 0.9).sum()) for d in ds], "o-",
            color=GREEN, lw=1.9, ms=5, label="sparse autoencoder")
    ax.axhline(int((pca_rec > 0.9).sum()), color=ORANGE, lw=1.4, ls="--",
               label=f"PCA ({N} components)")
    ax.axhline(M, color=GRAY, lw=1.0, ls=":")
    ax.text(ds[-1], M - 1.4, f"all {M} features", fontsize=8, color=GRAY, ha="right")
    ax.set_xlabel("dictionary size")
    ax.set_ylabel("features recovered (cosine > 0.9)")
    ax.set_ylim(0, M + 1)
    ax.set_yticks(range(0, M + 1, 5))          # a count, so integers only
    ax.set_title("What a dictionary recovers", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    tidy(ax)

    ax = axes[1]
    best = results[max(results)]
    order = np.argsort(-best)
    ax.bar(np.arange(M), best[order], 0.8, color=GREEN, label="sparse autoencoder")
    ax.bar(np.arange(M), pca_rec[order], 0.45, color=ORANGE, label="PCA")
    ax.axhline(0.9, color=INK, lw=1.0, ls="--")
    ax.set_xlabel("feature, ordered by how well it was recovered")
    ax.set_xticks(range(0, M, 4))              # a feature index, so integers only
    ax.set_ylabel("best cosine with a found direction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature by feature", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    tidy(ax)
    save(fig, "01-dictionary")


# --------------------------------------- 2. a probe is not a causal statement

def fig_probe_causal():
    """The most important negative result in interpretability, on a network
    where the ground truth is known: train on task A, then probe for a property
    B that the network provably does not use. The probe succeeds. Ablating the
    direction it found changes nothing. Reading a feature out is not evidence
    that the network reads it."""
    print("\n[2] probing against causal effect")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    D, H = 24, 96
    N_TRAIN, N_TEST = 40000, 8000

    def gen(n):
        x = rng.standard_normal((n, D)).astype(np.float32)
        # the task depends ONLY on the first block ...
        y = (np.sin(2.0 * x[:, :6] @ rng_fixed_A).sum(1) > 0).astype(np.int64)
        # ... and this property depends only on the second, which the task
        # never consults. It is present in the input, so it reaches the hidden
        # layer, but nothing in the objective asks the model to compute it.
        z = (x[:, 6:12] @ rng_fixed_B > 0).astype(np.int64).ravel()
        return torch.tensor(x), torch.tensor(y), torch.tensor(z)

    global rng_fixed_A, rng_fixed_B
    rng_fixed_A = rng.standard_normal((6, 6)).astype(np.float32)
    rng_fixed_B = rng.standard_normal((6, 1)).astype(np.float32)

    Xtr, Ytr, Ztr = gen(N_TRAIN)
    Xte, Yte, Zte = gen(N_TEST)

    net = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
    head = nn.Linear(H, 2)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for i in range(3000):
        idx = torch.randint(0, N_TRAIN, (512,))
        loss = F.cross_entropy(head(net(Xtr[idx])), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    with torch.no_grad():
        Htr, Hte = net(Xtr), net(Xte)
        task_acc = (head(Hte).argmax(1) == Yte).float().mean().item()
    print(f"    task accuracy                         {task_acc:.3f}")

    def probe(feat_tr, tgt_tr, feat_te, tgt_te):
        p = nn.Linear(feat_tr.shape[1], 2)
        o = torch.optim.AdamW(p.parameters(), lr=3e-2, weight_decay=1e-4)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(o, 1500)
        for i in range(1500):
            idx = torch.randint(0, len(feat_tr), (512,))
            l = F.cross_entropy(p(feat_tr[idx]), tgt_tr[idx])
            o.zero_grad(); l.backward(); o.step(); s.step()
        with torch.no_grad():
            acc = (p(feat_te).argmax(1) == tgt_te).float().mean().item()
        return acc, p

    z_acc, zp = probe(Htr, Ztr, Hte, Zte)
    y_acc, yp = probe(Htr, Ytr, Hte, Yte)
    print(f"    probe for the UNUSED property         {z_acc:.3f}")
    print(f"    probe for the property the task uses  {y_acc:.3f}")

    def ablate(direction):
        """Project the hidden activations off a given direction and
        re-measure the task. If the network was using it, this hurts."""
        d = direction / direction.norm()
        with torch.no_grad():
            Hp = Hte - (Hte @ d)[:, None] * d[None]
            return (head(Hp).argmax(1) == Yte).float().mean().item()

    dz = (zp.weight[1] - zp.weight[0]).detach()
    # For the unused property there is no readout to appeal to, so the best
    # available direction is the one a freshly trained probe finds. For the
    # property the task IS about, there is something better than a second
    # probe: head's own weights ARE the direction the network reads to
    # decide, so ablate that directly. (An earlier version used a second
    # probe here too, for symmetry with dz — but in a 96-dimensional,
    # redundantly-encoded hidden layer a probe's direction need not coincide
    # with head's, so removing it left the network's actual decision
    # boundary untouched and the reported cost of ablating the "used"
    # direction did not reproduce. head.weight[1] - head.weight[0] is
    # provably the right direction: projecting it out zeroes the logit
    # difference head computes for every input, leaving only the bias term.)
    dy = (head.weight[1] - head.weight[0]).detach()
    acc_z = ablate(dz)
    acc_y = ablate(dy)
    rnd = np.mean([ablate(torch.randn(H)) for _ in range(20)])
    print(f"    task after ablating the unused direction   {acc_z:.3f}"
          f"   (change {acc_z - task_acc:+.3f})")
    print(f"    task after ablating a random direction     {rnd:.3f}"
          f"   (change {rnd - task_acc:+.3f})")
    print(f"    task after ablating the USED direction     {acc_y:.3f}"
          f"   (change {acc_y - task_acc:+.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.bar([0, 1], [z_acc, y_acc], 0.55, color=[PURPLE, GREEN])
    ax.axhline(0.5, color=GRAY, lw=1.3, ls="--")
    ax.text(1.42, 0.52, "chance", fontsize=8, color=GRAY, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["a property the task\nnever uses",
                        "the property the\ntask is about"], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("probe accuracy")
    ax.set_title("Both are readable", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    vals = [task_acc, acc_z, rnd, acc_y]
    # not LIGHT for the control bar — that is the gridline colour, and the bar
    # vanishes against it in print.
    ax.bar(np.arange(4), vals, 0.55, color=[GRAY, PURPLE, CYAN, GREEN])
    ax.axhline(task_acc, color=INK, lw=1.0, ls=":")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["intact", "unused\ndirection\nremoved",
                        "random\ndirection\nremoved", "used\ndirection\nremoved"],
                       fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("task accuracy")
    ax.set_title("Only one of them matters", fontsize=10.5, color=INK)
    tidy(ax)
    save(fig, "02-probe-causal")
    print("    a probe measures what is PRESENT. Ablation measures what is USED.")
    print("    They are different questions and the answers here differ.")


# ------------------------------------------------- 3. where the information is

def fig_patching():
    """Activation patching on a task whose internal structure is known: the
    answer needs two of the three inputs combined in a specific order. Run the
    model on a corrupted input, splice in one activation from the clean run,
    and measure how much of the correct answer comes back. That map is a causal
    claim, unlike a correlational one."""
    print("\n[3] activation patching")
    torch.manual_seed(0)
    V, MOD, L, D = 10, 10, 4, 64
    NPOS = 4          # a, b, c, and the answer position

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(NPOS, D)
            self.layers = nn.ModuleList([nn.ModuleDict(dict(
                att=nn.MultiheadAttention(D, 4, batch_first=True),
                ff=nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)),
                n1=nn.LayerNorm(D), n2=nn.LayerNorm(D))) for _ in range(L)])
            self.head = nn.Linear(D, MOD)

        def forward(self, x, patch=None):
            """patch = (layer, position, vector) spliced in after that layer."""
            h = self.emb(x) + self.pos(torch.arange(x.shape[1]))[None]
            m = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)
            for li, b in enumerate(self.layers):
                a = b["n1"](h)
                h = h + b["att"](a, a, a, attn_mask=m, need_weights=False)[0]
                h = h + b["ff"](b["n2"](h))
                if patch is not None and patch[0] == li:
                    h = h.clone()
                    h[:, patch[1]] = patch[2]
            return self.head(h[:, -1])

        def acts(self, x):
            out = []
            h = self.emb(x) + self.pos(torch.arange(x.shape[1]))[None]
            m = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)
            for b in self.layers:
                a = b["n1"](h)
                h = h + b["att"](a, a, a, attn_mask=m, need_weights=False)[0]
                h = h + b["ff"](b["n2"](h))
                out.append(h.clone())
            return out

    g = np.random.default_rng(0)

    def batch(n):
        # the answer is (a + b) mod 10. c is present and irrelevant, which is
        # the control: patching it should do nothing.
        a = g.integers(0, V, n); b = g.integers(0, V, n); c = g.integers(0, V, n)
        x = np.stack([a, b, c, np.zeros(n, int)], 1)
        return torch.tensor(x), torch.tensor((a + b) % MOD)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    for i in range(4000):
        x, y = batch(512)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    x, y = batch(4000)
    with torch.no_grad():
        acc = (net(x).argmax(1) == y).float().mean().item()
    print(f"    task accuracy {acc:.3f}  (chance {1/MOD:.2f})")

    # clean and corrupted pairs differing ONLY in a
    n = 2000
    a = g.integers(0, V, n); b = g.integers(0, V, n); c = g.integers(0, V, n)
    a2 = (a + g.integers(1, V, n)) % V                 # a changed, b and c not
    clean = torch.tensor(np.stack([a, b, c, np.zeros(n, int)], 1))
    corrupt = torch.tensor(np.stack([a2, b, c, np.zeros(n, int)], 1))
    y_clean = torch.tensor((a + b) % MOD)

    with torch.no_grad():
        clean_acts = net.acts(clean)
        base = (net(corrupt).argmax(1) == y_clean).float().mean().item()
        full = (net(clean).argmax(1) == y_clean).float().mean().item()
    print(f"    corrupted run recovers the clean answer {base:.3f}; clean run {full:.3f}")

    grid = np.zeros((L, NPOS))
    with torch.no_grad():
        for li in range(L):
            for pos in range(NPOS):
                out = net(corrupt, patch=(li, pos, clean_acts[li][:, pos]))
                r = (out.argmax(1) == y_clean).float().mean().item()
                grid[li, pos] = (r - base) / max(full - base, 1e-9)
    np.set_printoptions(precision=3, suppress=True)
    for li in range(L):
        print("    layer %d  " % li +
              "  ".join(f"{p}:{grid[li, i]:+.2f}"
                        for i, p in enumerate(["a", "b", "c", "ans"])))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(NPOS))
    ax.set_xticklabels(["$a$\n(corrupted)", "$b$", "$c$\n(irrelevant)", "answer"],
                       fontsize=8.5)
    ax.set_yticks(range(L)); ax.set_yticklabels([f"layer {i}" for i in range(L)],
                                                fontsize=8.5)
    for li in range(L):
        for pi in range(NPOS):
            ax.text(pi, li, f"{grid[li, pi]:+.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(grid[li, pi]) > 0.55 else INK)
    ax.set_title("Fraction of the answer restored", fontsize=10.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1]
    for pi, (lab, col) in enumerate(zip(["$a$ (corrupted)", "$b$", "$c$ (irrelevant)",
                                         "answer position"],
                                        [ORANGE, BLUE, GRAY, GREEN])):
        ax.plot(range(L), grid[:, pi], "o-", color=col, lw=1.8, ms=5, label=lab)
    ax.axhline(0, color=GRAY, lw=1.0, ls=":")
    ax.set_xticks(range(L)); ax.set_xlabel("layer patched")
    ax.set_ylabel("fraction of the clean answer restored")
    ax.set_title("Where the corrupted input still matters", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    save(fig, "03-patching")


if __name__ == "__main__":
    M, N = measure_superposition()
    fig_dictionary(M, N)
    fig_probe_causal()
    fig_patching()
