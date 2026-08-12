"""
Reference solutions for the Part XVI drills.

Importing this module monkey-patches the implementations into `drills`, so the
same test file runs against either.
"""
import numpy as np

import drills

# NumPy 2.0 renamed trapz to trapezoid and deprecated the old name; fall back
# to trapz on older NumPy where trapezoid does not exist yet.
_trapezoid = getattr(np, "trapezoid", np.trapz)


def _lse(L, axis=0):
    M = np.max(L, axis=axis, keepdims=True)
    return (M + np.log(np.exp(L - M).sum(axis=axis, keepdims=True))).squeeze(axis)


# --------------------------------------------------------- densities & scores

def gaussian_logpdf(X, mu, sd):
    X = np.atleast_2d(X); mu = np.asarray(mu, float)
    d = X.shape[1]
    d2 = ((X - mu) ** 2).sum(1)
    return -0.5 * d * np.log(2 * np.pi * sd ** 2) - d2 / (2 * sd ** 2)


def mixture_logpdf(X, mus, sds, weights, extra_var=0.0):
    X = np.atleast_2d(X)
    d = X.shape[1]
    L = []
    for m, sd, w in zip(np.asarray(mus, float), np.asarray(sds, float),
                        np.asarray(weights, float)):
        v = sd ** 2 + extra_var
        d2 = ((X - m) ** 2).sum(1)
        L.append(np.log(w) - 0.5 * d * np.log(2 * np.pi * v) - d2 / (2 * v))
    return _lse(np.stack(L), axis=0)


def mixture_score(X, mus, sds, weights, extra_var=0.0):
    X = np.atleast_2d(X)
    d = X.shape[1]
    logs, scores = [], []
    for m, sd, w in zip(np.asarray(mus, float), np.asarray(sds, float),
                        np.asarray(weights, float)):
        v = sd ** 2 + extra_var
        d2 = ((X - m) ** 2).sum(1)
        logs.append(np.log(w) - 0.5 * d * np.log(2 * np.pi * v) - d2 / (2 * v))
        scores.append((m - X) / v)
    L = np.stack(logs)
    R = np.exp(L - L.max(0))
    post = R / R.sum(0)
    return sum(post[i][:, None] * scores[i] for i in range(len(logs)))


def energy_to_density(E, grid, dx):
    E = np.asarray(E, float)
    u = np.exp(-(E - E.min()))
    Z = _trapezoid(u, dx=dx)
    return u / Z, float(Z)


# --------------------------------------------------------------- divergences

def kl_divergence(p, q, dx=1.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    if np.any(q[m] <= 0):
        return float("inf")
    return float(np.sum(p[m] * np.log(p[m] / q[m])) * dx)


def fit_gaussian_forward_kl(grid, p, dx):
    grid = np.asarray(grid, float); p = np.asarray(p, float)
    w = p * dx
    w = w / w.sum()
    mu = float((w * grid).sum())
    sd = float(np.sqrt((w * (grid - mu) ** 2).sum()))
    return mu, sd


def fit_gaussian_reverse_kl(grid, p, dx, n_steps=400):
    grid = np.asarray(grid, float); p = np.asarray(p, float)
    best, arg = np.inf, (0.0, 1.0)
    mus = np.linspace(grid.min(), grid.max(), 120)
    # np.ptp(grid), not grid.ptp(): the method was removed from ndarray in
    # NumPy 2.0 and the function form works under both.
    sds = np.exp(np.linspace(np.log(0.05), np.log(np.ptp(grid) / 2), 60))
    for mu in mus:
        q = np.exp(gaussian_logpdf(grid[:, None], np.array([mu]), 1.0))
        for sd in sds:
            q = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            v = kl_divergence(q, p, dx)
            if v < best:
                best, arg = v, (float(mu), float(sd))
    return arg


# ------------------------------------------------------------ autoregressive

def chain_rule_logprob(seq, cond_logprobs):
    seq = np.asarray(seq, int)
    return float(np.asarray(cond_logprobs)[np.arange(len(seq)), seq].sum())


def causal_mask(n):
    return np.triu(np.ones((n, n), dtype=bool), 1)


def sequential_dependency_depth(n):
    return int(n), 1


def top_p_filter(logits, p=0.9):
    logits = np.asarray(logits, float)
    order = np.argsort(-logits)
    s = logits[order]
    pr = np.exp(s - s.max()); pr /= pr.sum()
    c = np.cumsum(pr)
    keep = c < p
    keep[0] = True
    if not keep.all():
        keep[np.argmax(c >= p)] = True
    out = np.full_like(logits, -np.inf)
    out[order[keep]] = logits[order[keep]]
    return out


# ------------------------------------------------------------------ the ELBO

def gaussian_kl_to_standard_normal(mu, logvar):
    mu, logvar = np.asarray(mu, float), np.asarray(logvar, float)
    return -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).sum(-1)


def elbo_terms(x, z, mu_q, logvar_q, x_recon, dec_sd):
    x, x_recon = np.asarray(x, float), np.asarray(x_recon, float)
    d = x.shape[1]
    rec = -(((x - x_recon) ** 2).sum(1) / (2 * dec_sd ** 2)
            + d * np.log(dec_sd) + 0.5 * d * np.log(2 * np.pi))
    return rec, gaussian_kl_to_standard_normal(mu_q, logvar_q)


def iwae_bound(log_p_xz, log_q_z):
    w = np.asarray(log_p_xz, float) - np.asarray(log_q_z, float)
    k = w.shape[1]
    return _lse(w, axis=1) - np.log(k)


def reparameterise(mu, logvar, eps):
    return np.asarray(mu) + np.exp(np.asarray(logvar) / 2.0) * np.asarray(eps)


# ------------------------------------------------------- diffusion machinery

def cosine_alpha_bar(T, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return (f / f[0])[1:]


def betas_from_alpha_bar(abar):
    abar = np.asarray(abar, float)
    prev = np.concatenate([[1.0], abar[:-1]])
    return np.clip(1 - abar / prev, 1e-8, 0.999)


def forward_diffuse(x0, abar_t, eps):
    return np.sqrt(abar_t) * np.asarray(x0) + np.sqrt(1 - abar_t) * np.asarray(eps)


def eps_to_score(eps_pred, abar_t):
    return -np.asarray(eps_pred) / np.sqrt(1 - abar_t)


def eps_to_x0(xt, eps_pred, abar_t):
    return (np.asarray(xt) - np.sqrt(1 - abar_t) * np.asarray(eps_pred)) / np.sqrt(abar_t)


def ddpm_posterior(xt, eps_pred, beta_t, abar_t, abar_prev):
    mean = (np.asarray(xt) - beta_t / np.sqrt(1 - abar_t) * np.asarray(eps_pred)) \
        / np.sqrt(1 - beta_t)
    var = beta_t * (1 - abar_prev) / (1 - abar_t)
    return mean, float(var)


def guided_eps(eps_cond, eps_uncond, w):
    eps_cond, eps_uncond = np.asarray(eps_cond), np.asarray(eps_uncond)
    return eps_uncond + w * (eps_cond - eps_uncond)


def langevin_step(x, score, eta, noise):
    return np.asarray(x) + eta / 2.0 * np.asarray(score) + np.sqrt(eta) * np.asarray(noise)


# ------------------------------------------------------------------- the GAN

def optimal_discriminator(p_data, p_model):
    p_data, p_model = np.asarray(p_data, float), np.asarray(p_model, float)
    den = p_data + p_model
    return np.where(den > 0, p_data / np.where(den > 0, den, 1.0), 0.5)


def jensen_shannon(p, q, dx=1.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, dx) + 0.5 * kl_divergence(q, m, dx)


def generator_losses(d_fake):
    d = np.clip(np.asarray(d_fake, float), 1e-12, 1 - 1e-12)
    return float(np.mean(np.log(1 - d))), float(-np.mean(np.log(d)))


# ---------------------------------------------------------------- evaluation

def assign_to_modes(X, mus):
    X, mus = np.atleast_2d(X), np.atleast_2d(mus)
    return ((X[:, None, :] - mus[None]) ** 2).sum(-1).argmin(1)


def mode_weights(X, mus):
    lab = assign_to_modes(X, mus)
    return np.array([(lab == i).mean() for i in range(len(np.atleast_2d(mus)))])


def total_variation(p, q):
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def modes_covered(X, mus, thresh=0.02):
    return int((mode_weights(X, mus) > thresh).sum())


def rbf_mmd(X, Y, sigmas=(0.2, 0.5, 1.0, 2.0)):
    X, Y = np.atleast_2d(X).astype(float), np.atleast_2d(Y).astype(float)

    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)

    n, m = len(X), len(Y)
    kxx = (k(X, X).sum() - k(X, X).trace()) / (n * (n - 1))
    kyy = (k(Y, Y).sum() - k(Y, Y).trace()) / (m * (m - 1))
    return float(kxx + kyy - 2 * k(X, Y).mean())


def _radii(A, k):
    d = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def precision_recall_distributions(gen, real, k=5):
    gen, real = np.atleast_2d(gen).astype(float), np.atleast_2d(real).astype(float)
    rr, rg = _radii(real, k), _radii(gen, k)
    d_gr = np.sqrt(((gen[:, None, :] - real[None, :, :]) ** 2).sum(-1))
    precision = float((d_gr <= rr[None, :]).any(1).mean())
    recall = float((d_gr.T <= rg[None, :]).any(1).mean())
    return precision, recall


def nearest_training_distance(samples, train):
    S, T = np.atleast_2d(samples).astype(float), np.atleast_2d(train).astype(float)
    return np.sqrt(((S[:, None, :] - T[None, :, :]) ** 2).sum(-1)).min(1)


# --------------------------------------------------------------------- patch

for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
