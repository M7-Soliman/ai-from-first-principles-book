"""
Reference solutions for the Part XXVIII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import itertools

import numpy as np

import drills


# ------------------------------------------------------------------ structure

def joint_from_graph(parents, tables, card):
    n = len(parents)
    J = np.zeros([card] * n)
    for cfg in itertools.product(range(card), repeat=n):
        p = 1.0
        for i in range(n):
            key = tuple(cfg[j] for j in parents[i])
            p *= tables[i][key][cfg[i]]
        J[cfg] = p
    return J


def marginal(J, keep):
    axes = tuple(a for a in range(J.ndim) if a not in keep)
    return J.sum(axis=axes)


def is_collider(parents, node, a, b):
    return a in parents[node] and b in parents[node]


def descendants(parents, node):
    kids = {i for i in range(len(parents)) if node in parents[i]}
    out = set(kids)
    for k in kids:
        out |= descendants(parents, k)
    return out


def d_separated(parents, a, b, given):
    """Bayes-ball over every path in the underlying undirected graph."""
    given = set(given)
    n = len(parents)
    adj = {i: set(parents[i]) for i in range(n)}
    for i in range(n):
        for p in parents[i]:
            adj[p].add(i)
    # A collider is unblocked when IT or one of ITS descendants is observed —
    # not when a descendant of the conditioning set is. Getting this backwards
    # gives an implementation that is right on every graph without a collider,
    # which is most of the ones you would test it on.
    opens = {i for i in range(n)
             if i in given or (descendants(parents, i) & given)}

    def walk(path):
        cur = path[-1]
        if cur == b:
            return True                     # an open path exists
        for nxt in adj[cur]:
            if nxt in path:
                continue
            if len(path) >= 2:
                prev = path[-2]
                collider = prev in parents[cur] and nxt in parents[cur]
                if collider and cur not in opens:
                    continue                # blocked: unobserved collider
                if not collider and cur in given:
                    continue                # blocked: observed non-collider
            if walk(path + [nxt]):
                return True
        return False

    return not walk([a])


def partial_corr(a, b, given, bins=12):
    q = np.quantile(given, np.linspace(0, 1, bins + 1))
    out, w = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (given >= lo) & (given <= hi)
        if m.sum() > 30:
            out.append(np.corrcoef(a[m], b[m])[0, 1])
            w.append(m.sum())
    return float(np.average(out, weights=w))


# -------------------------------------------------------------------- chains

def forward_backward(pi, A, B, obs):
    pi = np.asarray(pi, float); A = np.asarray(A, float); B = np.asarray(B, float)
    K, T = len(pi), len(obs)
    al = np.zeros((T, K)); sc = np.zeros(T)
    al[0] = pi * B[:, obs[0]]; sc[0] = al[0].sum(); al[0] /= sc[0]
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[:, obs[t]]
        sc[t] = al[t].sum(); al[t] /= sc[t]
    be = np.zeros((T, K)); be[-1] = 1.0
    for t in range(T - 2, -1, -1):
        be[t] = A @ (B[:, obs[t + 1]] * be[t + 1]) / sc[t + 1]
    g = al * be
    return g / g.sum(1, keepdims=True), float(np.log(sc).sum())


def enumerate_posterior(pi, A, B, obs):
    pi = np.asarray(pi, float); A = np.asarray(A, float); B = np.asarray(B, float)
    K, T = len(pi), len(obs)
    g = np.zeros((T, K)); tot = 0.0
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * B[path[0], obs[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
        tot += p
        for t, s in enumerate(path):
            g[t, s] += p
    return g / tot, float(np.log(tot))


def viterbi(pi, A, B, obs):
    pi = np.asarray(pi, float); A = np.asarray(A, float); B = np.asarray(B, float)
    K, T = len(pi), len(obs)
    with np.errstate(divide="ignore"):
        d = np.log(pi) + np.log(B[:, obs[0]])
    bp = np.zeros((T, K), int)
    for t in range(1, T):
        with np.errstate(divide="ignore"):
            m = d[:, None] + np.log(A)
        bp[t] = m.argmax(0)
        d = m.max(0) + np.log(B[:, obs[t]])
    path = [int(d.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return np.array(path[::-1]), float(d.max())


def kalman_scalar(y, F, H, Q, R, m0, P0):
    m, P = float(m0), float(P0)
    ms, Ps = [], []
    for yt in np.asarray(y, float):
        m = F * m
        P = F * P * F + Q
        S = H * P * H + R
        K = P * H / S
        m = m + K * (yt - H * m)
        P = (1 - K * H) * P
        ms.append(m); Ps.append(P)
    return np.array(ms), np.array(Ps)


# ------------------------------------------------------------------------ EM

def em_gmm(x, K, seed=0, iters=200):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    mu = r.choice(x, K, replace=False).astype(float)
    var = np.full(K, float(x.var()))
    w = np.full(K, 1.0 / K)
    lls = []
    for _ in range(iters):
        p = w * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        tot = p.sum(1)
        lls.append(float(np.log(np.maximum(tot, 1e-300)).sum()))
        g = p / tot[:, None]
        Nk = g.sum(0) + 1e-12
        w = Nk / len(x)
        mu = (g * x[:, None]).sum(0) / Nk
        var = np.maximum((g * (x[:, None] - mu) ** 2).sum(0) / Nk, 1e-6)
    return np.array(lls), mu, var, w


# ---------------------------------------------------------------- variational

def meanfield_gaussian_sd(rho):
    P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    return np.array([1 / np.sqrt(P[0, 0]), 1 / np.sqrt(P[1, 1])])


def elbo_gaussian(m, log_sd, Sigma):
    """ELBO for a factorised Gaussian q against a zero-mean Gaussian p, in closed
    form: E_q[log p] + H[q]."""
    m = np.asarray(m, float); sd = np.exp(np.asarray(log_sd, float))
    P = np.linalg.inv(np.asarray(Sigma, float))
    d = len(m)
    quad = m @ P @ m + np.sum(np.diag(P) * sd ** 2)
    logZ = 0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(Sigma)[1])
    ent = np.sum(np.log(sd)) + 0.5 * d * (1 + np.log(2 * np.pi))
    return float(-0.5 * quad - logZ + ent)


# --------------------------------------------------------------- monte carlo

def importance_weights(logp, logq, samples):
    lw = np.array([logp(s) - logq(s) for s in samples], float)
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


def ess_weights(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def metropolis(logp, x0, step, n, seed=0):
    r = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    lp = logp(x)
    out = np.empty((n, len(x))); acc = 0
    for i in range(n):
        y = x + step * r.standard_normal(len(x))
        ly = logp(y)
        if np.log(r.random()) < ly - lp:
            x, lp = y, ly; acc += 1
        out[i] = x
    return out, acc / n


def leapfrog(gradp, x, p, step, L):
    x = x.copy(); p = p.copy()
    p = p + 0.5 * step * gradp(x)
    for i in range(L):
        x = x + step * p
        p = p + (step if i < L - 1 else 0.5 * step) * gradp(x)
    return x, p


def hmc(logp, gradp, x0, step, L, n, seed=0):
    r = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    out = np.empty((n, len(x))); acc = 0
    for i in range(n):
        p = r.standard_normal(len(x))
        H0 = -logp(x) + 0.5 * p @ p
        xn, pn = leapfrog(gradp, x, p, step, L)
        H1 = -logp(xn) + 0.5 * pn @ pn
        if np.log(r.random()) < H0 - H1:
            x = xn; acc += 1
        out[i] = x
    return out, acc / n


def ess_chain(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair <= 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def rhat(chains):
    C = np.asarray(chains, float)
    m, n = C.shape
    W = C.var(1, ddof=1).mean()
    B = n * C.mean(1).var(ddof=1)
    V = (n - 1) / n * W + B / n
    return float(np.sqrt(V / W))


# ------------------------------------------------------------ continuous time

def euler_maruyama(f, g, x0, T, steps, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x0, float).copy()
    dt = T / steps
    for i in range(steps):
        t = i * dt
        x = x + f(x, t) * dt + g(t) * np.sqrt(dt) * r.standard_normal(x.shape)
    return x


def ou_marginal(x0, t, theta=1.0, sigma=np.sqrt(2.0)):
    """Exact marginal of dx = -theta x dt + sigma dW started at x0."""
    a = np.exp(-theta * t)
    v = sigma ** 2 / (2 * theta) * (1 - a ** 2)
    return a * np.asarray(x0, float), v


# ------------------------------------------------------------------ transport

def wasserstein1(a, b):
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if len(a) != len(b):
        q = np.linspace(0, 1, max(len(a), len(b)))
        a = np.quantile(a, q); b = np.quantile(b, q)
    return float(np.abs(a - b).mean())


def kl_gaussian(m1, s1, m2, s2):
    return float(np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5)


def sinkhorn(a, b, C, eps, iters=500):
    a = np.asarray(a, float); b = np.asarray(b, float)
    K = np.exp(-np.asarray(C, float) / eps)
    u = np.ones_like(a)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    return u[:, None] * K * v[None, :]


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
