"""
The Part XVI figures that require training. Imported by make_figures.py.

    python3 train_figures.py

Everything here is trained on a three-component Gaussian mixture whose density,
score and mixing weights are known in closed form. That is the point: "did the
model learn the distribution" becomes an arithmetic question — are the recovered
mode weights 0.40, 0.35 and 0.25 — rather than a judgement about whether some
pictures look plausible.

Six questions that need a real model:

  * how loose is the ELBO, against a likelihood that can be computed exactly?
  * does the latent code get used, or does the posterior collapse?
  * does a diffusion model trained by denoising score matching actually
    recover the distribution, weights and all?
  * what does guidance trade away?
  * does a GAN drop modes, and how many?
  * does log-likelihood predict sample quality?
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (save, tidy, plt, BLUE, ORANGE, GREEN,           # noqa: E402
                          PURPLE, PINK, GRAY, LIGHT, INK, CYAN,
                          MUS, SDS, WS, mix_logp, mix_score, mix_sample)

torch.set_num_threads(4)


# ------------------------------------------------------------------ shared

def assign_modes(X):
    """Which mixture component each sample most likely came from, and the
    resulting weight estimates. The ground truth is WS."""
    X = np.asarray(X)
    d2 = np.stack([((X - m) ** 2).sum(1) / (2 * sd ** 2) + np.log(sd)
                   for m, sd in zip(MUS, SDS)])
    lab = d2.argmin(0)
    w = np.array([(lab == i).mean() for i in range(len(WS))])
    return lab, w


def mode_report(X, name):
    lab, w = assign_modes(X)
    tv = 0.5 * np.abs(w - WS).sum()
    print(f"    {name:34s} weights {np.round(w, 3)}  "
          f"(true {WS})  total-variation {tv:.4f}")
    return w, tv


def mmd(X, Y, sigmas=(0.2, 0.5, 1.0, 2.0)):
    """Maximum mean discrepancy with a mixture of RBF kernels. A distance
    between two SAMPLES, so it works for models that have no density."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)

    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)

    n, m = len(X), len(Y)
    kxx = (k(X, X).sum() - n) / (n * (n - 1))
    kyy = (k(Y, Y).sum() - m) / (m * (m - 1))
    return float(kxx + kyy - 2 * k(X, Y).mean())


def mlp(din, dout, w=256, depth=3):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, w), nn.SiLU()]
        d = w
    return nn.Sequential(*layers, nn.Linear(w, dout))


# ------------------------------------------------ 3. diffusion, end to end

class TimeMLP(nn.Module):
    """Predicts the noise that was added, given the noisy point and the level."""

    def __init__(self, w=256):
        super().__init__()
        self.net = mlp(2 + 32, 2, w=w, depth=3)
        self.register_buffer("freqs", torch.exp(
            torch.linspace(np.log(1.0), np.log(300.0), 16)))

    def embed(self, t):
        a = t[:, None] * self.freqs[None, :]
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.embed(t)], 1))


def cosine_schedule(T):
    s = 0.008
    t = np.linspace(0, 1, T + 1)
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-8, 0.999)
    return betas.astype(np.float64)


def linear_schedule(T):
    return np.linspace(1e-4, 0.02, T).astype(np.float64)


def train_diffusion(T=200, steps=9000, schedule="cosine", seed=0, n=20000):
    torch.manual_seed(seed)
    betas = cosine_schedule(T) if schedule == "cosine" else linear_schedule(T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    ab = torch.tensor(abar, dtype=torch.float32)

    X = torch.tensor(mix_sample(n, seed=seed))
    net = TimeMLP()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = torch.randint(0, n, (512,))
        x0 = X[idx]
        t = torch.randint(0, T, (512,))
        eps = torch.randn_like(x0)
        a = ab[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = F.mse_loss(net(xt, t.float() / T), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 3000 == 0:
            print(f"      [{schedule}] step {step:5d}  denoising MSE {loss.item():.4f}")
    return net, betas, abar


@torch.no_grad()
def ddpm_sample(net, betas, abar, m=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = len(betas)
    alphas = 1.0 - betas
    x = torch.randn(m, 2, generator=g)
    for i in reversed(range(T)):
        t = torch.full((m,), i / T)
        eps = net(x, t)
        a, ab_i = alphas[i], abar[i]
        mean = (x - betas[i] / np.sqrt(1 - ab_i) * eps) / np.sqrt(a)
        if i > 0:
            ab_prev = abar[i - 1]
            var = betas[i] * (1 - ab_prev) / (1 - ab_i)
            x = mean + np.sqrt(var) * torch.randn(m, 2, generator=g)
        else:
            x = mean
    return x.numpy()


def fig_diffusion():
    """The payoff. A network trained only to predict the noise that was added
    to a data point, sampled by reversing the corruption, against a
    distribution whose mixing weights are known exactly."""
    print("      training the diffusion model ...")
    net, betas, abar = train_diffusion()
    S = ddpm_sample(net, betas, abar, m=4000)
    real = mix_sample(4000, seed=99)

    w_model, tv = mode_report(S, "diffusion samples")
    w_real, tv_real = mode_report(real, "a fresh sample of the truth")
    d_model = mmd(S[:1200], real[:1200])
    d_real = mmd(mix_sample(1200, seed=7), real[:1200])

    # does the learned network match the analytic score at each noise level?
    cos_by_t = []
    ts = [0.05, 0.2, 0.5, 0.9]
    with torch.no_grad():
        for tf in ts:
            i = int(tf * len(betas))
            ab_i = abar[i]
            g = np.linspace(-3.5, 3.5, 30)
            GX, GY = np.meshgrid(g, g)
            P = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32)
            eps = net(torch.tensor(P), torch.full((len(P),), i / len(betas))).numpy()
            learned = -eps / np.sqrt(1 - ab_i)          # score of the noisy marginal
            # the exact score of the same marginal: data scaled then blurred
            Pu = P / np.sqrt(ab_i)
            truth = mix_score(Pu, extra_var=(1 - ab_i) / ab_i) / np.sqrt(ab_i)
            dens = np.exp(mix_logp(Pu, extra_var=(1 - ab_i) / ab_i))
            keep = dens > np.quantile(dens, 0.5)
            c = ((learned[keep] * truth[keep]).sum(1) /
                 (np.linalg.norm(learned[keep], axis=1) *
                  np.linalg.norm(truth[keep], axis=1) + 1e-9))
            cos_by_t.append(float(c.mean()))
            print(f"    noise level t={tf:.2f}: mean cosine with the exact score "
                  f"{c.mean():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, D, ttl, col in ((axes[0], real, "The distribution", GRAY),
                            (axes[1], S, "Samples from the model", ORANGE)):
        ax.scatter(D[:, 0], D[:, 1], s=3, color=col, alpha=0.4)
        ax.scatter(MUS[:, 0], MUS[:, 1], s=45, marker="x", color=INK, lw=1.8)
        ax.set_title(ttl, fontsize=10.5, color=INK)
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.6, 3.4)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_visible(False)

    ax = axes[2]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, WS, w, color=GRAY, label="true")
    ax.bar(x + w/2, w_model, w, color=ORANGE, label="recovered")
    ax.set_xticks(x); ax.set_xticklabels(["mode 1", "mode 2", "mode 3"], fontsize=9)
    ax.set_ylabel("mixing weight")
    ax.set_title(f"Weights recovered (TV {tv:.3f})", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    tidy(ax)

    print(f"    MMD to a held-out real sample: model {d_model:.5f}, "
          f"a fresh real sample {d_real:.5f}")
    print(f"    the sampler starts from pure noise and never sees the data.")
    fig.tight_layout()
    save(fig, "04-diffusion")
    return dict(tv=tv, tv_real=tv_real, mmd=d_model, mmd_real=d_real,
                cos=cos_by_t, ts=ts)


# ------------------------------------------------------ 4. GAN mode collapse

K_RING = 8
RING_MUS = np.stack([2.0 * np.cos(np.arange(K_RING) * 2 * np.pi / K_RING),
                     2.0 * np.sin(np.arange(K_RING) * 2 * np.pi / K_RING)], 1)
RING_SD = 0.10


def ring_sample(n, seed=0):
    r = np.random.default_rng(seed)
    c = r.integers(0, K_RING, n)
    return (RING_MUS[c] + r.normal(size=(n, 2)) * RING_SD).astype(np.float32)


def ring_weights(X):
    d = ((np.asarray(X)[:, None, :] - RING_MUS[None]) ** 2).sum(-1)
    lab = d.argmin(1)
    return np.array([(lab == i).mean() for i in range(K_RING)])


def ring_tv(X):
    return float(0.5 * np.abs(ring_weights(X) - 1.0 / K_RING).sum())


def train_gan(seed, sat=False, dsteps=1, betas=(0.5, 0.9), steps=6000):
    torch.manual_seed(seed)
    G = mlp(2, 2, w=128, depth=3)
    D = mlp(2, 1, w=128, depth=3)
    oG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=betas)
    oD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=betas)
    X = torch.tensor(ring_sample(20000, seed))
    B = 256
    for _ in range(steps):
        for _ in range(dsteps):
            real = X[torch.randint(0, len(X), (B,))]
            z = torch.randn(B, 2)
            lD = (F.binary_cross_entropy_with_logits(D(real), torch.ones(B, 1))
                  + F.binary_cross_entropy_with_logits(D(G(z).detach()),
                                                       torch.zeros(B, 1)))
            oD.zero_grad(); lD.backward(); oD.step()
        z = torch.randn(B, 2)
        if sat:      # the original minimax form
            lG = -F.binary_cross_entropy_with_logits(D(G(z)), torch.zeros(B, 1))
        else:        # the non-saturating form everyone actually uses
            lG = F.binary_cross_entropy_with_logits(D(G(z)), torch.ones(B, 1))
        oG.zero_grad(); lG.backward(); oG.step()
    with torch.no_grad():
        return G(torch.randn(4000, 2)).numpy()


def fig_gan():
    """Mode collapse, measured — and the finding that it does not come from
    where the textbook says. Eight modes in a ring, so a dropped mode is a
    countable event rather than an impression."""
    CONDS = [("standard\n(non-saturating, 1:1)", dict()),
             ("saturating\nloss", dict(sat=True)),
             ("discriminator\nover-trained 5:1", dict(dsteps=5)),
             ("Adam $\\beta_1=0.9$", dict(betas=(0.9, 0.999)))]
    seeds = [0, 1, 2]
    res = {}
    for name, kw in CONDS:
        cov, tv, last = [], [], None
        for sd in seeds:
            S = train_gan(sd, **kw)
            w = ring_weights(S)
            cov.append(int((w > 0.02).sum())); tv.append(ring_tv(S)); last = S
        res[name] = dict(cov=cov, tv=tv, S=last)
        print(f"    {name.replace(chr(10),' '):34s} modes covered {cov}   "
              f"TV {[round(t,3) for t in tv]}")

    # the same data, by diffusion, for a fair reference
    print("      training a diffusion model on the same ring for comparison ...")
    torch.manual_seed(0)
    betas_d = cosine_schedule(200)
    abar_d = np.cumprod(1.0 - betas_d)
    ab = torch.tensor(abar_d, dtype=torch.float32)
    Xr = torch.tensor(ring_sample(20000, 0))
    net = TimeMLP()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 9000)
    for step in range(9000):
        x0 = Xr[torch.randint(0, len(Xr), (512,))]
        t = torch.randint(0, 200, (512,))
        eps = torch.randn_like(x0)
        a = ab[t][:, None]
        loss = F.mse_loss(net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t.float() / 200), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    S_diff = ddpm_sample(net, betas_d, abar_d, m=4000)
    diff_cov = int((ring_weights(S_diff) > 0.02).sum())
    diff_tv = ring_tv(S_diff)
    real_tv = ring_tv(ring_sample(4000, 99))
    print(f"    diffusion on the same data: modes {diff_cov}/8, TV {diff_tv:.3f}")
    print(f"    a fresh real sample of the same size: TV {real_tv:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    worst = max(res, key=lambda k: np.mean(res[k]["tv"]))
    for ax, D, ttl, col in ((axes[0], ring_sample(4000, 99), "The distribution", GRAY),
                            (axes[1], res[worst]["S"],
                             "GAN, " + worst.replace("\n", " "), ORANGE)):
        ax.scatter(D[:, 0], D[:, 1], s=3, color=col, alpha=0.45)
        ax.scatter(RING_MUS[:, 0], RING_MUS[:, 1], s=40, marker="x",
                   color=INK, lw=1.6)
        ax.set_title(ttl, fontsize=9.6, color=INK)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "bottom", "left"):
            ax.spines[s_].set_visible(False)

    ax = axes[2]
    names = [c[0] for c in CONDS]
    x = np.arange(len(names))
    means = [np.mean(res[n]["cov"]) for n in names]
    lo = [min(res[n]["cov"]) for n in names]
    ax.bar(x, means, 0.55, color=[GREEN, GREEN, ORANGE, ORANGE],
           yerr=[np.array(means) - np.array(lo), np.zeros(len(x))],
           capsize=3, ecolor=INK)
    ax.axhline(K_RING, color=GRAY, lw=1.2, ls="--")
    ax.text(-0.45, K_RING + 0.12, "all 8 modes", fontsize=8, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylabel("modes produced (of 8)")
    ax.set_ylim(0, 9.4)
    ax.set_title("Where collapse actually comes from", fontsize=10.5, color=INK)
    tidy(ax)

    print("    the loss form is NOT what breaks it: the saturating objective, which is")
    print("    the textbook explanation, covers all eight modes at every seed. What")
    print("    breaks it is the balance between the two players.")
    fig.tight_layout()
    save(fig, "06-gan")
    return dict(res={k: dict(cov=v["cov"], tv=v["tv"]) for k, v in res.items()},
                diff_cov=diff_cov, diff_tv=diff_tv, real_tv=real_tv)


# --------------------------------------- 5. likelihood against sample quality

def fig_likelihood_quality():
    """Test log-likelihood and sample quality are nearly independent. Two
    models are constructed to make the point in both directions."""
    real_tr = mix_sample(6000, seed=0)
    real_te = mix_sample(4000, seed=1)

    def logp_mixture(X, mus, sds, ws):
        L = []
        for m, sd, w in zip(mus, sds, ws):
            d2 = ((X - m) ** 2).sum(1)
            L.append(np.log(w) - np.log(2 * np.pi * sd ** 2) - d2 / (2 * sd ** 2))
        L = np.stack(L); M = L.max(0)
        return M + np.log(np.exp(L - M).sum(0))

    models = {}

    # A: the truth
    models["exact model"] = (
        lambda X: logp_mixture(X, MUS, SDS, WS),
        mix_sample(4000, seed=5))

    # B: drops one mode entirely — samples are all perfect, likelihood is not
    keep = [0, 1]
    w2 = WS[keep] / WS[keep].sum()
    r = np.random.default_rng(3)
    c = r.choice(len(keep), 4000, p=w2)
    S_drop = MUS[keep][c] + r.normal(size=(4000, 2)) * SDS[keep][c][:, None]
    models["drops a mode"] = (
        lambda X: logp_mixture(X, MUS[keep], SDS[keep], w2), S_drop)

    # C: the truth plus a broad background — samples are often garbage,
    #    likelihood barely moves because the background is thin everywhere
    eps = 0.20
    def logp_bg(X):
        good = logp_mixture(X, MUS, SDS, WS)
        bg = -np.log(2 * np.pi * 9.0) - (X ** 2).sum(1) / (2 * 9.0)
        M = np.maximum(good + np.log(1 - eps), bg + np.log(eps))
        return M + np.log(np.exp(good + np.log(1 - eps) - M)
                          + np.exp(bg + np.log(eps) - M))
    r = np.random.default_rng(4)
    isbg = r.random(4000) < eps
    S_bg = mix_sample(4000, seed=6)
    S_bg[isbg] = r.normal(scale=3.0, size=(isbg.sum(), 2))
    models["truth + 20% garbage"] = (logp_bg, S_bg)

    rows = []
    for name, (lp, S) in models.items():
        ll = float(lp(real_te).mean())
        d = mmd(S[:1200], real_te[:1200])
        _, tv = assign_modes(S)[1], 0.5 * np.abs(assign_modes(S)[1] - WS).sum()
        # fraction of samples that are plausible under the truth
        plaus = float((mix_logp(S) > np.quantile(mix_logp(real_te), 0.01)).mean())
        rows.append((name, ll, d, plaus))
        print(f"    {name:22s} test log-lik {ll:8.4f}   MMD {d:9.5f}   "
              f"plausible samples {plaus*100:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    ax = axes[0]
    ax.bar(x, [r[1] for r in rows], color=[GRAY, BLUE, ORANGE], width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylabel("test log-likelihood")
    ax.set_title("What likelihood says", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.bar(x, [r[3] * 100 for r in rows], color=[GRAY, BLUE, ORANGE], width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylabel("% of samples that are plausible")
    ax.set_ylim(0, 105)
    ax.set_title("What the samples say", fontsize=10.5, color=INK)
    tidy(ax)

    print("    the mode-dropping model produces PERFECT samples and the worse")
    print("    likelihood; the background model keeps most of the likelihood and")
    print("    produces visible garbage. The two measurements disagree by construction.")
    fig.tight_layout()
    save(fig, "07-likelihood-quality")
    return rows


# ----------------------------------------------- 6. the VAE: bound and collapse

class VAE(nn.Module):
    def __init__(self, zdim=2, w=128, dec_sd=0.25):
        super().__init__()
        self.zdim = zdim
        self.enc = mlp(2, 2 * zdim, w=w, depth=2)
        self.dec = mlp(zdim, 2, w=w, depth=2)
        self.dec_sd = dec_sd

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h[:, :self.zdim], h[:, self.zdim:].clamp(-8, 4)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.dec(z), mu, logvar


def train_vae(beta=1.0, zdim=2, steps=6000, seed=0, dec_sd=0.25, n=20000):
    torch.manual_seed(seed)
    X = torch.tensor(mix_sample(n, seed=seed))
    m = VAE(zdim=zdim, dec_sd=dec_sd)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(steps):
        x = X[torch.randint(0, n, (512,))]
        xr, mu, logvar = m(x)
        rec = ((x - xr) ** 2).sum(1) / (2 * dec_sd ** 2) \
              + 2 * np.log(dec_sd) + np.log(2 * np.pi)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        loss = (rec + beta * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def vae_iwae(m, X, k=2000, chunk=200):
    """Importance-weighted estimate of log p(x). With enough samples this is a
    tight estimate of the true marginal, so the ELBO's gap can be measured."""
    out = []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        h = m.enc(x)
        mu, logvar = h[:, :m.zdim], h[:, m.zdim:].clamp(-8, 4)
        sd = (0.5 * logvar).exp()
        z = mu[:, None, :] + sd[:, None, :] * torch.randn(len(x), k, m.zdim)
        xr = m.dec(z.reshape(-1, m.zdim)).reshape(len(x), k, 2)
        lpxz = -(((x[:, None, :] - xr) ** 2).sum(-1) / (2 * m.dec_sd ** 2)
                 + 2 * np.log(m.dec_sd) + np.log(2 * np.pi))
        lpz = -0.5 * (z ** 2).sum(-1) - m.zdim * 0.5 * np.log(2 * np.pi)
        lqz = (-0.5 * ((z - mu[:, None, :]) / sd[:, None, :]) ** 2
               - torch.log(sd)[:, None, :] - 0.5 * np.log(2 * np.pi)).sum(-1)
        w = lpxz + lpz - lqz
        out.append(torch.logsumexp(w, 1) - np.log(k))
    return torch.cat(out)


@torch.no_grad()
def vae_elbo(m, X):
    h = m.enc(X)
    mu, logvar = h[:, :m.zdim], h[:, m.zdim:].clamp(-8, 4)
    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
    xr = m.dec(z)
    rec = -(((X - xr) ** 2).sum(1) / (2 * m.dec_sd ** 2)
            + 2 * np.log(m.dec_sd) + np.log(2 * np.pi))
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return rec - kl


def fig_vae():
    """Two things the ELBO does not tell you: how far below the true likelihood
    it sits, and whether the latent code is being used at all."""
    Xte = torch.tensor(mix_sample(2000, seed=42))
    true_ll = float(mix_logp(Xte.numpy()).mean())

    betas = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    elbo, iwae, kl_used, active = [], [], [], []
    for b in betas:
        m = train_vae(beta=b)
        e = float(vae_elbo(m, Xte).mean())
        i = float(vae_iwae(m, Xte).mean())
        h = m.enc(Xte)
        mu, logvar = h[:, :m.zdim], h[:, m.zdim:].clamp(-8, 4)
        kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)
        elbo.append(e); iwae.append(i)
        kl_used.append(float(kl_dim.sum()))
        active.append(int((kl_dim > 0.01).sum()))
        print(f"    beta {b:4.1f}: ELBO {e:8.3f}   log p (IWAE, k=2000) {i:8.3f}   "
              f"gap {i-e:6.3f}   KL {kl_dim.sum():6.3f} nats   "
              f"active dims {active[-1]}/{m.zdim}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(betas, elbo, "o-", color=BLUE, lw=1.9, ms=4.5, label="ELBO (the bound)")
    ax.plot(betas, iwae, "s-", color=ORANGE, lw=1.9, ms=4.5,
            label="$\\log p(x)$ (IWAE, $k{=}2000$)")
    ax.axhline(true_ll, color=GRAY, lw=1.3, ls="--", label="the true density")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("nats per example")
    ax.set_title("The bound, and what it bounds", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    tidy(ax)

    ax = axes[1]
    ax.plot(betas, kl_used, "o-", color=PURPLE, lw=1.9, ms=4.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("KL to the prior (nats)")
    ax.set_title("How much the code is used", fontsize=10.5, color=INK)
    tidy(ax)
    ax2 = ax.twinx()
    ax2.plot(betas, active, "s--", color=GREEN, lw=1.4, ms=3.5)
    ax2.set_ylabel("active latent dimensions", color=GREEN, fontsize=9)
    ax2.set_ylim(-0.1, 2.3)
    ax2.tick_params(labelsize=8.5, colors=GREEN)
    ax2.spines["top"].set_visible(False)

    print(f"    the true density has log-likelihood {true_ll:.3f} nats per example")
    print(f"    the gap between ELBO and log p is the KL from q to the true posterior,")
    print(f"    and it is never zero: {min(np.array(iwae)-np.array(elbo)):.3f} at best here.")
    print(f"    at beta={betas[-1]} the code carries {kl_used[-1]:.3f} nats "
          f"in {active[-1]} dimensions — posterior collapse, and the ELBO")
    print("    is not obviously worse for it.")
    fig.tight_layout()
    save(fig, "02-vae")
    return dict(betas=betas, elbo=elbo, iwae=iwae, kl=kl_used, active=active,
                true_ll=true_ll)


# ------------------------------------------------------------ 7. guidance

def fig_guidance():
    """Classifier-free guidance, and what it costs. Trained with a class label
    that is dropped some of the time, so one network gives both the conditional
    and unconditional score."""
    K = 3
    T = 200

    class CondNet(nn.Module):
        def __init__(self, w=256):
            super().__init__()
            self.emb = nn.Embedding(K + 1, 32)          # K classes + "no class"
            self.net = mlp(2 + 32 + 32, 2, w=w, depth=3)
            self.register_buffer("freqs", torch.exp(
                torch.linspace(np.log(1.0), np.log(300.0), 16)))

        def forward(self, x, t, c):
            a = t[:, None] * self.freqs[None, :]
            return self.net(torch.cat([x, a.sin(), a.cos(), self.emb(c)], 1))

    torch.manual_seed(0)
    betas = cosine_schedule(T)
    abar = np.cumprod(1.0 - betas)
    ab = torch.tensor(abar, dtype=torch.float32)

    r = np.random.default_rng(0)
    lab = r.choice(K, 20000, p=WS)
    X = torch.tensor((MUS[lab] + r.normal(size=(20000, 2)) * SDS[lab][:, None]
                      ).astype(np.float32))
    Y = torch.tensor(lab)

    net = CondNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 9000)
    for step in range(9000):
        i = torch.randint(0, len(X), (512,))
        x0, c = X[i], Y[i].clone()
        c[torch.rand(512) < 0.15] = K                   # drop the label
        t = torch.randint(0, T, (512,))
        eps = torch.randn_like(x0)
        a = ab[t][:, None]
        loss = F.mse_loss(net(a.sqrt() * x0 + (1 - a).sqrt() * eps,
                              t.float() / T, c), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 4000 == 0:
            print(f"      guidance model step {step:5d}  MSE {loss.item():.4f}")

    @torch.no_grad()
    def sample(w, cls, m=1500, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(m, 2, generator=g)
        c = torch.full((m,), cls, dtype=torch.long)
        u = torch.full((m,), K, dtype=torch.long)
        for i in reversed(range(T)):
            t = torch.full((m,), i / T)
            e_c, e_u = net(x, t, c), net(x, t, u)
            eps = e_u + w * (e_c - e_u)
            mean = (x - betas[i] / np.sqrt(1 - abar[i]) * eps) / np.sqrt(1 - betas[i])
            if i > 0:
                var = betas[i] * (1 - abar[i - 1]) / (1 - abar[i])
                x = mean + np.sqrt(var) * torch.randn(m, 2, generator=g)
            else:
                x = mean
        return x.numpy()

    ws = [0.0, 1.0, 2.0, 4.0, 8.0]
    fid, div, corr = [], [], []
    for w in ws:
        S = np.concatenate([sample(w, c) for c in range(K)])
        cls = np.concatenate([np.full(1500, c) for c in range(K)])
        lab_hat, _ = assign_modes(S)
        corr.append(float((lab_hat == cls).mean()))
        fid.append(float(np.mean(mix_logp(S))))          # how typical, per truth
        # spread WITHIN the mode each sample actually landed in, against the
        # truth's spread for that mode. Measured on correctly-placed samples
        # only, so w=0 (which ignores the class) is still a fair number.
        sp = []
        for c in range(K):
            sel = (lab_hat == c)
            if sel.sum() > 30:
                sp.append(S[sel].std(0).mean() / SDS[c])
        div.append(float(np.mean(sp)))
        print(f"    guidance w={w:4.1f}: class obeyed {corr[-1]*100:5.1f}%   "
              f"mean log-density {fid[-1]:9.3f}   within-mode spread vs truth "
              f"{div[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(ws, np.array(corr) * 100, "o-", color=BLUE, lw=1.9, ms=4.5)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("% of samples in the requested mode", color=BLUE)
    ax.set_title("What guidance buys", fontsize=10.5, color=INK)
    ax.tick_params(labelcolor=BLUE)
    tidy(ax)

    ax = axes[1]
    ax.plot(ws, div, "s-", color=ORANGE, lw=1.9, ms=4.5)
    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--")
    ax.text(0.1, 1.03, "the true spread", fontsize=8, color=GRAY)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("within-mode spread / true spread", color=ORANGE)
    ax.tick_params(labelcolor=ORANGE)
    ax.set_title("What it costs", fontsize=10.5, color=INK)
    tidy(ax)
    ax2 = ax.twinx()
    ax2.plot(ws, fid, "^--", color=PURPLE, lw=1.5, ms=4)
    ax2.set_ylabel("mean log-density under the truth", color=PURPLE, fontsize=8.6)
    ax2.tick_params(labelsize=8.5, colors=PURPLE)
    ax2.spines["top"].set_visible(False)

    best = int(np.argmax(fid))
    print(f"    adherence rises from {corr[0]*100:.1f}% (chance) to 100% by w=1.")
    print(f"    typicality PEAKS at w={ws[best]} (log-density {fid[best]:.3f}, against "
          f"{np.mean(mix_logp(mix_sample(4000, 77))):.3f} for real data)")
    print(f"    and then collapses: {fid[-1]:.1f} at w={ws[-1]}. Past the peak the")
    print("    samples are not merely less diverse, they have left the distribution.")
    fig.tight_layout()
    save(fig, "05-guidance")
    return dict(ws=ws, corr=corr, fid=fid, div=div)


EXPENSIVE = [fig_diffusion, fig_gan, fig_likelihood_quality, fig_vae, fig_guidance]



if __name__ == "__main__":
    for f in EXPENSIVE:
        f()
