"""
The Part XV figures that require training. Imported by make_figures.py.

    python3 train_figures.py

Six questions that need a real model:

  * does a linear autoencoder recover PCA, or only PCA's subspace?
  * does a denoising autoencoder really learn the score of the data density?
  * what stops a contrastive objective from collapsing?
  * how many features can a layer hold in superposition, and at what sparsity?
  * does a probe's accuracy establish that a representation encodes anything?
  * what does self-supervised pretraining buy, as a function of label budget?

The answers to the second and fifth are the reason this part exists in the
shape it does.
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
DEV = "cpu"


# ------------------------------------------------- 4. linear autoencoder = PCA

def fig_linear_ae():
    """ch14 says an undercomplete linear autoencoder 'learns to span the same
    subspace as PCA'. That sentence is exact and the exactness matters: it
    recovers the SUBSPACE, and not the axes. Both halves measured."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, D, k = 4000, 12, 3

    # data with a clear three-dimensional signal and decaying spectrum
    A = rng.normal(size=(k, D))
    scales = np.array([4.0, 2.0, 1.0])
    Z = rng.normal(size=(n, k)) * scales
    X = Z @ A + rng.normal(scale=0.35, size=(n, D))
    X = X - X.mean(0)
    Xt = torch.tensor(X, dtype=torch.float32)

    # PCA by SVD — the reference answer
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Vpca = Vt[:k]                                   # k x D, the principal axes

    enc = nn.Linear(D, k, bias=False)
    dec = nn.Linear(k, D, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
    losses = []
    for step in range(4000):
        idx = torch.randint(0, n, (256,))
        xb = Xt[idx]
        loss = F.mse_loss(dec(enc(xb)), xb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                losses.append(F.mse_loss(dec(enc(Xt)), Xt).item())

    W = enc.weight.detach().numpy()                 # k x D

    def principal_angles(Aa, Bb):
        Qa = np.linalg.qr(Aa.T)[0]
        Qb = np.linalg.qr(Bb.T)[0]
        s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        return np.degrees(np.arccos(np.clip(s, -1, 1)))

    ang = principal_angles(W, Vpca)
    # axis-by-axis agreement: |cos| between the i-th AE row and the i-th PC
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    axis_cos = np.abs(np.einsum("ij,ij->i", Wn, Vpca))

    # the reconstruction error PCA achieves, versus what the AE achieves
    Xr_pca = X @ Vpca.T @ Vpca
    mse_pca = float(((X - Xr_pca) ** 2).mean())
    with torch.no_grad():
        mse_ae = float(F.mse_loss(dec(enc(Xt)), Xt).item())

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))

    ax = axes[0]
    ax.plot(np.arange(len(losses)) * 20, losses, color=BLUE, lw=1.6,
            label="autoencoder")
    ax.axhline(mse_pca, color=ORANGE, lw=1.6, ls="--",
               label=f"PCA, rank {k}")
    ax.set_xlabel("step"); ax.set_ylabel("reconstruction MSE")
    ax.set_title("It reaches PCA's error", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.bar(np.arange(k) + 1, ang, color=GREEN, width=0.55)
    ax.set_ylim(0, 90)
    ax.set_xticks(np.arange(k) + 1)
    ax.set_xlabel("principal angle index")
    ax.set_ylabel("angle (degrees)")
    ax.set_title("Same subspace: angles $\\approx 0$", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[2]
    ax.bar(np.arange(k) + 1, axis_cos, color=PURPLE, width=0.55)
    ax.axhline(1.0, color=GRAY, lw=1.0, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(k) + 1)
    ax.set_xlabel("component")
    ax.set_ylabel(r"$|\cos|$ with the matching PC")
    ax.set_title("Different axes", fontsize=10.5, color=INK)
    tidy(ax)

    print(f"    PCA rank-{k} reconstruction MSE {mse_pca:.5f}; "
          f"autoencoder {mse_ae:.5f} (ratio {mse_ae/mse_pca:.4f})")
    print(f"    principal angles between the two subspaces (degrees): "
          f"{', '.join(f'{a:.2f}' for a in ang)}")
    print(f"    per-axis |cos| with the matching principal component: "
          f"{', '.join(f'{c:.3f}' for c in axis_cos)}")
    print("    the SUBSPACE is recovered to a fraction of a degree; the AXES are not")
    print("    the loss cannot tell the difference: every rotation of the code has")
    print("    identical reconstruction error, so nothing selects the principal axes")
    fig.tight_layout()
    save(fig, "02-linear-ae")


# --------------------------------------------- 5. denoising learns the score

def fig_denoising_score():
    """The load-bearing figure of the part, and of Part XVI. A denoising
    autoencoder trained with Gaussian corruption has r(x) - x proportional to
    the score of the SMOOTHED density. We compare against the analytic score of
    a Gaussian mixture, where both are known in closed form."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # a two-component Gaussian mixture in 2-D: analytic density and score
    mus = np.array([[-1.4, -0.6], [1.3, 0.8]])
    sds = np.array([0.55, 0.40])
    ws = np.array([0.55, 0.45])

    def mixture_logp_score(P, extra_var=0.0):
        """log p and its gradient for the mixture, optionally convolved with
        N(0, extra_var I) — which is exactly what corruption does to it."""
        comps = []
        for m, sd, w in zip(mus, sds, ws):
            v = sd ** 2 + extra_var
            d2 = ((P - m) ** 2).sum(1)
            logk = np.log(w) - np.log(2 * np.pi * v) - d2 / (2 * v)
            comps.append((logk, (m - P) / v))
        L = np.stack([c[0] for c in comps])            # 2 x N
        M = L.max(0)
        r = np.exp(L - M)
        post = r / r.sum(0)                            # responsibilities
        score = sum(post[i][:, None] * comps[i][1] for i in range(len(comps)))
        logp = M + np.log(r.sum(0))
        return logp, score

    def sample(n, seed):
        r = np.random.default_rng(seed)
        c = r.choice(len(ws), size=n, p=ws)
        return (mus[c] + r.normal(size=(n, 2)) * sds[c][:, None]).astype(np.float32)

    SIGMA = 0.35
    Xtr = sample(20000, 0)
    Xt = torch.tensor(Xtr)

    net = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
                        nn.Linear(256, 256), nn.Tanh(),
                        nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(6000):
        idx = torch.randint(0, len(Xt), (512,))
        x = Xt[idx]
        xt = x + SIGMA * torch.randn_like(x)           # corruption
        loss = F.mse_loss(net(xt), x)                  # predict the CLEAN point
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"      step {step:5d}  denoising MSE {loss.item():.4f}")

    # evaluate on a grid, in the region the data actually occupies
    g = np.linspace(-3.0, 3.0, 22)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    with torch.no_grad():
        R = net(torch.tensor(P, dtype=torch.float32)).numpy()
    learned = (R - P) / SIGMA ** 2                     # the claimed score estimate

    logp_s, true_s = mixture_logp_score(P, extra_var=SIGMA ** 2)   # smoothed
    logp_0, true_0 = mixture_logp_score(P, extra_var=0.0)          # unsmoothed

    dens = np.exp(logp_s)
    keep = dens > np.quantile(dens, 0.45)              # where there is data

    def agree(a, b, m):
        ca = (a[m] * b[m]).sum(1) / (np.linalg.norm(a[m], axis=1) *
                                     np.linalg.norm(b[m], axis=1) + 1e-12)
        sc = np.linalg.norm(a[m], axis=1).mean() / np.linalg.norm(b[m], axis=1).mean()
        return float(ca.mean()), float(sc)

    cos_s, scale_s = agree(learned, true_s, keep)
    cos_0, scale_0 = agree(learned, true_0, keep)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    sub = slice(None, None, 1)

    def field(ax, V, title, colour):
        ax.contourf(GX, GY, np.exp(logp_s).reshape(GX.shape), levels=12,
                    cmap="Blues", alpha=0.55)
        ax.quiver(P[sub, 0], P[sub, 1], V[sub, 0], V[sub, 1],
                  color=colour, width=0.005, scale=90)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

    field(axes[0], true_s, "The true score $\\nabla_x \\log p_\\sigma$", INK)
    field(axes[1], learned, r"$(r(\mathbf{x}) - \mathbf{x})/\sigma^2$", ORANGE)

    ax = axes[2]
    ax.scatter(np.linalg.norm(true_s[keep], axis=1),
               np.linalg.norm(learned[keep], axis=1), s=7, color=PURPLE, alpha=0.6)
    lim = max(np.linalg.norm(true_s[keep], axis=1).max(),
              np.linalg.norm(learned[keep], axis=1).max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GRAY, lw=1.1, ls="--")
    ax.set_xlabel("true score magnitude")
    ax.set_ylabel("learned magnitude")
    ax.set_title(f"cosine {cos_s:.3f}", fontsize=10.5, color=INK)
    tidy(ax)

    print(f"    against the SMOOTHED score (sigma={SIGMA}): "
          f"mean cosine {cos_s:.4f}, magnitude ratio {scale_s:.3f}")
    print(f"    against the UNSMOOTHED score:              "
          f"mean cosine {cos_0:.4f}, magnitude ratio {scale_0:.3f}")
    print("    the network was trained only to remove noise. Nothing in its loss")
    print("    mentions a density, a gradient, or a log. The score is what denoising IS.")
    fig.tight_layout()
    save(fig, "03-denoising-score")
    return dict(cos_s=cos_s, cos_0=cos_0, scale_s=scale_s, sigma=SIGMA)


# ------------------------------------------------ a generator with known factors

FACT_D, N_CLASS, N_NUIS = 64, 8, 8


def factor_data(n, seed=0):
    """Each example is built from TWO independent factors along random
    orthogonal directions: a 'class' we want kept and a 'nuisance' we want
    discarded. Because we built it, we know exactly what a representation
    should and should not retain.

    The factors are then pushed through a FIXED random nonlinearity before we
    ever see them. This matters: pixels are a nonlinear function of the
    underlying factors, and a generator that skips this step makes the class
    linearly decodable from the raw input, which quietly removes most of what
    representation learning is for.

    The nuisance is deliberately the LOUDER of the two, as it usually is."""
    r = np.random.default_rng(seed)
    B = np.linalg.qr(r.normal(size=(FACT_D, FACT_D)))[0]
    Ccode = r.normal(size=(N_CLASS, FACT_D)) @ B * 1.0
    Ncode = r.normal(size=(N_NUIS, FACT_D)) @ B * 2.5      # nuisance dominates
    M1 = r.normal(size=(FACT_D, 96)) / np.sqrt(FACT_D)
    M2 = r.normal(size=(96, FACT_D)) / np.sqrt(96)
    return Ccode, Ncode, (M1, M2)


OBS_FREQ = 2.0


def observe(z, obs):
    M1, M2 = obs
    return np.sin(OBS_FREQ * (z @ M1)) @ M2


def make_batch(Ccode, Ncode, n, rng, cls=None, nui=None, obs=None):
    c = rng.integers(0, N_CLASS, n) if cls is None else cls
    u = rng.integers(0, N_NUIS, n) if nui is None else nui
    z = Ccode[c] + Ncode[u] + rng.normal(scale=0.45, size=(n, FACT_D))
    x = observe(z, obs) if obs is not None else z
    return x.astype(np.float32), c, u


def encoder():
    return nn.Sequential(nn.Linear(FACT_D, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 32))


def eff_rank(H):
    """Effective rank: exp of the entropy of the normalised eigenspectrum.
    A collapsed representation has rank near 1 whatever its dimension."""
    H = H - H.mean(0)
    s = np.linalg.svd(H, compute_uv=False) ** 2
    p = s / max(s.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_probe(H, y, Hte, yte):
    """Multinomial logistic regression, the standard linear probe."""
    from sklearn.linear_model import LogisticRegression
    # multi_class="multinomial" was deprecated in sklearn 1.5 and removed in
    # 1.7; multinomial is the default for >2 classes since 1.5, so dropping
    # the argument keeps the same behaviour on both old and new sklearn.
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(H, y)
    return float(m.score(Hte, yte))


def train_contrastive(mode, steps=1500, seed=0, aug="nuisance", tau=0.2,
                      probe_every=25, track_probe=False):
    """mode: 'infonce' | 'positives-only' | 'stopgrad'
    aug: which factor the two views are allowed to disagree on."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    Ccode, Ncode, OBS = factor_data(0)
    net = encoder()
    pred = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
    params = list(net.parameters()) + (list(pred.parameters())
                                       if mode == "stopgrad" else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    ranks, probes, steps_at = [], [], []
    B = 256
    for step in range(steps):
        c = rng.integers(0, N_CLASS, B)
        u1 = rng.integers(0, N_NUIS, B)
        if aug == "nuisance":            # views share class, differ in nuisance
            u2 = rng.integers(0, N_NUIS, B)
            c1, c2 = c, c
        elif aug == "class":             # views share nuisance, differ in class
            u2 = u1
            c1 = c
            c2 = rng.integers(0, N_CLASS, B)
        else:                            # views differ in neither
            u2, c1, c2 = u1, c, c
        x1 = torch.tensor(make_batch(Ccode, Ncode, B, rng, c1, u1, OBS)[0])
        x2 = torch.tensor(make_batch(Ccode, Ncode, B, rng, c2, u2, OBS)[0])
        z1, z2 = net(x1), net(x2)
        z1n, z2n = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        if mode == "infonce":
            logits = z1n @ z2n.T / tau
            loss = F.cross_entropy(logits, torch.arange(B))
        elif mode == "positives-only":
            loss = -(z1n * z2n).sum(1).mean()
        else:                            # stopgrad: predictor on one side only
            p1, p2 = F.normalize(pred(z1), dim=1), F.normalize(pred(z2), dim=1)
            loss = -0.5 * ((p1 * z2n.detach()).sum(1).mean() +
                           (p2 * z1n.detach()).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % probe_every == 0:
            with torch.no_grad():
                xe, ce, _ = make_batch(Ccode, Ncode, 1200, rng, obs=OBS)
                xf, cf, _ = make_batch(Ccode, Ncode, 1200, rng, obs=OBS)
                He = net(torch.tensor(xe)).numpy()
                Hf = net(torch.tensor(xf)).numpy()
            ranks.append(eff_rank(He))
            steps_at.append(step)
            probes.append(linear_probe(He, ce, Hf, cf) if track_probe else np.nan)
    with torch.no_grad():
        xa, ca, ua = make_batch(Ccode, Ncode, 3000, rng, obs=OBS)
        xb, cb, ub = make_batch(Ccode, Ncode, 3000, rng, obs=OBS)
        Ha, Hb = net(torch.tensor(xa)).numpy(), net(torch.tensor(xb)).numpy()
    return dict(ranks=ranks, probes=probes, steps_at=steps_at, net=net,
                Ccode=Ccode, Ncode=Ncode, obs=OBS,
                Ha=Ha, ca=ca, ua=ua, Hb=Hb, cb=cb, ub=ub)


# --------------------------------------------------------- 6. collapse

def fig_collapse():
    """Why a contrastive objective needs something pushing outward — and the
    discovery that a linear probe is a LAGGING indicator of collapse, still
    reporting perfect accuracy on a representation that is already 99% gone."""
    STEPS = 20000
    runs = {}
    for mode in ("infonce", "positives-only", "stopgrad"):
        print(f"      training {mode} for {STEPS} steps ...")
        runs[mode] = train_contrastive(mode, steps=STEPS, probe_every=1000,
                                       track_probe=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    names = {"infonce": "InfoNCE (negatives)",
             "positives-only": "positives only",
             "stopgrad": "stop-gradient + predictor"}
    cols = {"infonce": BLUE, "positives-only": ORANGE, "stopgrad": GREEN}

    ax = axes[0]
    for m, r in runs.items():
        ax.plot(r["steps_at"], r["ranks"], color=cols[m], lw=1.8, label=names[m])
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("effective rank of the code")
    ax.set_title("Collapse (32 dimensions available)", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for m, r in runs.items():
        ax.plot(r["steps_at"], r["probes"], color=cols[m], lw=1.8, label=names[m])
    ax.axhline(1.0 / N_CLASS, color=GRAY, lw=1.1, ls="--")
    ax.text(STEPS * 0.98, 1.0 / N_CLASS + 0.03, "chance", fontsize=8,
            color=GRAY, ha="right")
    ax.set_xlabel("step"); ax.set_ylabel("linear probe accuracy")
    ax.set_ylim(0, 1.06)
    ax.set_title("The probe notices much later", fontsize=10.5, color=INK)
    tidy(ax)

    po = runs["positives-only"]
    for m in ("infonce", "stopgrad", "positives-only"):
        print(f"    {names[m]:28s} final rank {runs[m]['ranks'][-1]:7.3f}   "
              f"final probe {runs[m]['probes'][-1]:.3f}")
    print(f"    chance is {1.0/N_CLASS:.3f}")
    print("    the lag, for positives-only:")
    for st, rk, pr in zip(po["steps_at"], po["ranks"], po["probes"]):
        if st % 2000 == 0:
            print(f"      step {st:6d}   effective rank {rk:7.3f}   probe {pr:.3f}")
    print("    negatives are one way to prevent collapse and not the only way;")
    print("    and a probe alone would not have told you it was happening.")
    fig.tight_layout()
    save(fig, "05-collapse")
    return dict(ranks={m: runs[m]["ranks"][-1] for m in runs},
                probes={m: runs[m]["probes"][-1] for m in runs},
                lag=list(zip(po["steps_at"], po["ranks"], po["probes"])))


# --------------------------------------------------- 7. augmentation specifies

def fig_augmentation():
    """The augmentation is not a trick to get more data. It is the statement of
    what the representation is required to ignore, and it is obeyed exactly."""
    out = {}
    for aug in ("nuisance", "class", "none"):
        print(f"      training with aug={aug} ...")
        r = train_contrastive("infonce", aug=aug)
        out[aug] = dict(
            cls=linear_probe(r["Ha"], r["ca"], r["Hb"], r["cb"]),
            nui=linear_probe(r["Ha"], r["ua"], r["Hb"], r["ub"]),
            rank=r["ranks"][-1])

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    labels = ["views differ\nin nuisance", "views differ\nin class",
              "views differ\nin neither"]
    x = np.arange(3); w = 0.36
    keys = ["nuisance", "class", "none"]
    ax.bar(x - w/2, [out[k]["cls"] for k in keys], w, color=BLUE,
           label="class decodable")
    ax.bar(x + w/2, [out[k]["nui"] for k in keys], w, color=ORANGE,
           label="nuisance decodable")
    ax.axhline(1.0 / N_CLASS, color=GRAY, lw=1.1, ls="--",
               label=f"chance ({1.0/N_CLASS:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("linear probe accuracy")
    ax.set_ylim(0, 1.24)
    ax.set_title("What the augmentation discards, the representation discards",
                 fontsize=10.5, color=INK, pad=26)
    ax.legend(fontsize=8.2, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), columnspacing=1.4, handlelength=1.5)
    tidy(ax)

    for k, lab in zip(keys, ["differ in nuisance", "differ in class",
                             "differ in neither"]):
        print(f"    {lab:20s}  class {out[k]['cls']:.3f}   "
              f"nuisance {out[k]['nui']:.3f}   rank {out[k]['rank']:.1f}")
    print(f"    chance {1.0/N_CLASS:.3f}")
    print("    the factor the two views were allowed to disagree on is the factor")
    print("    that gets thrown away. Nothing else in the objective mentions it.")
    fig.tight_layout()
    save(fig, "04-augmentation")
    return out


# ------------------------------------------------------------ 8. superposition

def fig_superposition():
    """More features than dimensions. A layer with m dimensions asked to
    represent n > m features will refuse when the features are dense and pack
    them at non-orthogonal angles when they are sparse."""
    torch.manual_seed(0)
    n_feat, m_dim = 20, 5
    sparsities = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    imp = torch.tensor((0.85 ** np.arange(n_feat)), dtype=torch.float32)

    results = []
    for S in sparsities:
        torch.manual_seed(0)
        W = nn.Parameter(torch.randn(m_dim, n_feat) * 0.2)
        b = nn.Parameter(torch.zeros(n_feat))
        opt = torch.optim.Adam([W, b], lr=2e-3)
        for step in range(6000):
            h = torch.rand(1024, n_feat)
            mask = (torch.rand(1024, n_feat) > S).float()
            h = h * mask                              # sparse non-negative features
            out = F.relu(h @ W.T @ W + b)
            loss = (imp * (h - out) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        Wd = W.detach().numpy()
        norms = np.linalg.norm(Wd, axis=0)            # per-feature norm
        rep = int((norms > 0.35).sum())               # features actually stored
        Wn = Wd / (norms + 1e-9)
        G = Wn.T @ Wn
        off = np.abs(G[~np.eye(n_feat, dtype=bool)])
        results.append(dict(S=S, rep=rep, norms=norms, interf=float(off.mean())))
        print(f"      sparsity {S:.2f}: {rep:2d}/{n_feat} features stored in "
              f"{m_dim} dimensions, mean |interference| {off.mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot([r["S"] for r in results], [r["rep"] for r in results],
            "o-", color=BLUE, lw=1.9, ms=5)
    ax.axhline(m_dim, color=GRAY, lw=1.2, ls="--")
    ax.text(0.02, m_dim + 0.4, f"{m_dim} dimensions available", fontsize=8, color=GRAY)
    ax.set_xlabel("feature sparsity")
    ax.set_ylabel("features actually stored")
    ax.set_ylim(0, n_feat + 1)
    ax.set_title("Sparsity buys room that does not exist",
                 fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    for r, c in zip([results[0], results[3], results[-1]], (GRAY, BLUE, ORANGE)):
        ax.plot(np.arange(1, n_feat + 1), r["norms"], "o-", ms=3.5, lw=1.5,
                color=c, label=f"sparsity {r['S']:.2f}")
    ax.set_xlabel("feature, by importance")
    ax.set_ylabel("stored magnitude")
    ax.set_title("Which features survive", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    dense, sparse = results[0], results[-1]
    print(f"    dense (sparsity {dense['S']:.2f}): {dense['rep']} features in "
          f"{m_dim} dimensions — it stores the top few ORTHOGONALLY and drops the rest")
    print(f"    sparse (sparsity {sparse['S']:.2f}): {sparse['rep']} features in "
          f"{m_dim} dimensions — {sparse['rep']/m_dim:.1f}x overcapacity, "
          f"at mean interference {sparse['interf']:.3f}")
    print("    superposition is a bet: features that rarely co-occur can share")
    print("    directions, because the collision rarely happens.")
    fig.tight_layout()
    save(fig, "10-superposition")
    return results


# ------------------------------------------------------- 9. probe selectivity

def fig_probe():
    """What a probe's accuracy does and does not establish.

    The control task of Hewitt & Liang: every example IDENTITY gets a fixed
    random label, consistent between train and test. The mapping is perfectly
    learnable and means nothing. A probe that scores well on it is memorising
    identities, not reading a property off the representation.
    """
    rng = np.random.default_rng(0)
    Ccode, Ncode, ctrl_obs = factor_data(0)
    ctrl_map = rng.integers(0, N_CLASS, N_CLASS * N_NUIS)   # identity -> junk label

    def batch(n, seed):
        r = np.random.default_rng(seed)
        X, c, u = make_batch(Ccode, Ncode, n, r, obs=ctrl_obs)
        return X, c, ctrl_map[c * N_NUIS + u]

    Xtr, ctr, jtr = batch(3000, 1)
    Xte, cte, jte = batch(3000, 2)

    torch.manual_seed(0)
    net_rand = encoder()                                    # untrained
    print("      pretraining an encoder with InfoNCE for the 'trained' arm ...")
    trained = train_contrastive("infonce", steps=4000)["net"]

    def feats(net, X):
        with torch.no_grad():
            return net(torch.tensor(X)).numpy()

    reps = {
        "raw input": (Xtr, Xte),
        "untrained net": (feats(net_rand, Xtr), feats(net_rand, Xte)),
        "contrastive": (feats(trained, Xtr), feats(trained, Xte)),
    }

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    def mk(kind):
        return (LogisticRegression(max_iter=4000) if kind == "linear"
                else MLPClassifier((256, 256), max_iter=1200, random_state=0))

    res = {}
    for name, (Ftr, Fte) in reps.items():
        res[name] = {}
        for kind in ("linear", "MLP"):
            m = mk(kind); m.fit(Ftr, ctr); real = float(m.score(Fte, cte))
            m2 = mk(kind); m2.fit(Ftr, jtr); ctrl = float(m2.score(Fte, jte))
            res[name][kind] = dict(real=real, ctrl=ctrl, sel=real - ctrl)
            print(f"      {name:15s} {kind:6s}  real {real:.3f}   "
                  f"control {ctrl:.3f}   selectivity {real-ctrl:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    keys = list(reps); x = np.arange(len(keys)); w = 0.36
    for ax, kind in zip(axes, ("linear", "MLP")):
        ax.bar(x - w/2, [res[k][kind]["real"] for k in keys], w,
               color=BLUE, label="real task")
        ax.bar(x + w/2, [res[k][kind]["ctrl"] for k in keys], w,
               color=PINK, label="control task")
        ax.axhline(1.0 / N_CLASS, color=GRAY, lw=1.1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8.2)
        ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
        ax.set_title(f"{kind} probe", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        tidy(ax)

    print("    selectivity = real minus control, and it is the quantity that means")
    print("    something. Accuracy alone cannot distinguish a representation that")
    print("    encodes the property from a probe strong enough to learn it anyway.")
    fig.tight_layout()
    save(fig, "09-probe")
    return res


# ------------------------------------------------------ 10. label efficiency

def fig_label_efficiency():
    """What self-supervision buys, stated as the thing anyone actually cares
    about: how many labels you need to reach a given accuracy."""
    print("      pretraining with InfoNCE (no labels used) ...")
    pre = train_contrastive("infonce", steps=4000)
    net_ssl, Ccode, Ncode, Obs = pre["net"], pre["Ccode"], pre["Ncode"], pre["obs"]

    from sklearn.linear_model import LogisticRegression
    budgets = [8, 16, 32, 64, 128, 256, 512]
    ssl_m, sup_m, ssl_s, sup_s = [], [], [], []

    rr = np.random.default_rng(77)
    Xv, cv, _ = make_batch(Ccode, Ncode, 3000, rr, obs=Obs)
    with torch.no_grad():
        Hv = net_ssl(torch.tensor(Xv)).numpy()

    for nb in budgets:
        a_ssl, a_sup = [], []
        for trial in range(5):
            r = np.random.default_rng(1000 + 17 * trial)
            Xs, cs, _ = make_batch(Ccode, Ncode, nb, r, obs=Obs)
            if len(np.unique(cs)) < 2:
                continue
            with torch.no_grad():
                Hs = net_ssl(torch.tensor(Xs)).numpy()
            m = LogisticRegression(max_iter=4000).fit(Hs, cs)
            a_ssl.append(float(m.score(Hv, cv)))

            torch.manual_seed(trial)
            sup = nn.Sequential(encoder(), nn.Linear(32, N_CLASS))
            opt = torch.optim.Adam(sup.parameters(), lr=3e-3)
            xs, ys = torch.tensor(Xs), torch.tensor(cs)
            for _ in range(600):
                loss = F.cross_entropy(sup(xs), ys)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                a_sup.append(float((sup(torch.tensor(Xv)).argmax(1).numpy() == cv).mean()))
        ssl_m.append(np.mean(a_ssl)); ssl_s.append(np.std(a_ssl))
        sup_m.append(np.mean(a_sup)); sup_s.append(np.std(a_sup))
        print(f"      {nb:4d} labels ({nb/N_CLASS:.0f} per class): "
              f"frozen encoder + linear probe {ssl_m[-1]:.3f}±{ssl_s[-1]:.3f}   "
              f"supervised from scratch {sup_m[-1]:.3f}±{sup_s[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.errorbar(budgets, ssl_m, yerr=ssl_s, fmt="o-", color=GREEN, lw=1.9,
                ms=4, capsize=2, label="pretrained encoder + linear probe")
    ax.errorbar(budgets, sup_m, yerr=sup_s, fmt="s--", color=ORANGE, lw=1.7,
                ms=4, capsize=2, label="supervised from scratch")
    ax.axhline(1.0 / N_CLASS, color=GRAY, lw=1.0, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("labelled examples")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy against label budget", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    tidy(ax)

    ax = axes[1]
    ax.plot(budgets, np.array(ssl_m) - np.array(sup_m), "o-", color=PURPLE,
            lw=1.9, ms=4)
    ax.axhline(0, color=GRAY, lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("labelled examples")
    ax.set_ylabel("accuracy gained by pretraining")
    ax.set_title("Where the benefit lives", fontsize=10.5, color=INK)
    tidy(ax)

    # how many labels supervised needs to match SSL at the smallest budget
    tgt = ssl_m[0]
    need = next((b for b, a in zip(budgets, sup_m) if a >= tgt), None)
    print(f"    the frozen encoder reaches {tgt:.3f} with {budgets[0]} labels;")
    print(f"    supervised training needs {need} labels to match it "
          f"({need/budgets[0]:.0f}x more)" if need else
          "    supervised training does not match it within the budgets tested")
    fig.tight_layout()
    save(fig, "06-label-efficiency")
    return dict(budgets=budgets, ssl=ssl_m, sup=sup_m)


EXPENSIVE = [fig_linear_ae, fig_denoising_score, fig_collapse, fig_augmentation,
             fig_superposition, fig_probe, fig_label_efficiency]



if __name__ == "__main__":
    for f in EXPENSIVE:
        f()
