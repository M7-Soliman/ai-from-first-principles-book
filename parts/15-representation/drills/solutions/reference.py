"""
Reference solutions for the Part XV drills.

Importing this module monkey-patches the implementations into `drills`, so the
same test file runs against either. Read it after attempting the drills.
"""
import numpy as np

import drills


# ------------------------------------------------------------------- geometry

def cosine_matrix(A, B=None):
    B = A if B is None else B
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    A = A / np.where(na == 0, 1.0, na)
    B = B / np.where(nb == 0, 1.0, nb)
    S = A @ B.T
    S[(na == 0).ravel(), :] = 0.0
    S[:, (nb == 0).ravel()] = 0.0
    return S


def anisotropy(H, n_pairs=100_000, seed=0):
    r = np.random.default_rng(seed)
    n = len(H)
    i = r.integers(0, n, n_pairs)
    j = r.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    A = H[i] / np.linalg.norm(H[i], axis=1, keepdims=True)
    B = H[j] / np.linalg.norm(H[j], axis=1, keepdims=True)
    return float((A * B).sum(1).mean())


def center_and_whiten(H, eps=1e-6):
    Hc = H - H.mean(0)
    C = np.cov(Hc, rowvar=False)
    lam, V = np.linalg.eigh(C)
    Hw = Hc @ V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 0.0) + eps)) @ V.T
    return Hc, Hw


def knn_counts(H, k=10):
    from scipy.spatial.distance import cdist
    D = cdist(H, H)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k, axis=1)[:, :k]
    return np.bincount(nn.ravel(), minlength=len(H))


def hubness_skew(H, k=10):
    c = knn_counts(H, k).astype(float)
    m, s = c.mean(), c.std()
    return float(((c - m) ** 3).mean() / s ** 3)


def mutual_knn(H, k=10):
    from scipy.spatial.distance import cdist
    D = cdist(H, H)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k, axis=1)[:, :k]
    A = np.zeros((len(H), len(H)), dtype=bool)
    rows = np.repeat(np.arange(len(H)), k)
    A[rows, nn.ravel()] = True
    return A & A.T


# ------------------------------------------------------------ subspaces & PCA

def principal_angles(A, B):
    Qa = np.linalg.qr(np.asarray(A, dtype=float).T)[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=float).T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.sort(np.degrees(np.arccos(np.clip(s, -1.0, 1.0))))


def pca_subspace(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k]


def subspace_reconstruction_error(X, W):
    Xc = X - X.mean(0)
    Q = np.linalg.qr(np.asarray(W, dtype=float).T)[0]
    R = Xc @ Q @ Q.T
    return float(((Xc - R) ** 2).mean())


# ------------------------------------------------------ codes and their spread

def _eigs(H):
    Hc = H - H.mean(0)
    s = np.linalg.svd(Hc, compute_uv=False) ** 2
    return s


def effective_rank(H, eps=1e-12):
    s = _eigs(H)
    p = s / max(s.sum(), eps)
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(H):
    s = _eigs(H)
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-30))


# --------------------------------------------------------------- contrastive

def _norm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def info_nce(Z1, Z2, tau=0.2):
    A, B = _norm(Z1), _norm(Z2)
    L = (A @ B.T) / tau
    M = L.max(1, keepdims=True)
    lse = M.ravel() + np.log(np.exp(L - M).sum(1))
    return float((lse - np.diag(L)).mean())


def alignment(Z1, Z2):
    A, B = _norm(Z1), _norm(Z2)
    return float(((A - B) ** 2).sum(1).mean())


def uniformity(Z, t=2.0):
    A = _norm(Z)
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(A), k=1)
    return float(np.log(np.exp(-t * d2[iu]).mean()))


def collapse_report(H):
    s = _eigs(H)
    return dict(effective_rank=effective_rank(H),
                participation_ratio=participation_ratio(H),
                top_eigenvalue_share=float(s.max() / max(s.sum(), 1e-30)))


# ------------------------------------------------------------------- probing

def linear_probe_accuracy(Htr, ytr, Hte, yte, C=1.0):
    """Multinomial logistic regression with an L2 penalty, fitted by L-BFGS.

    This used scikit-learn, which is one import for a model the book teaches how
    to build — and when scikit-learn stopped being installed, four tests failed
    for reasons that had nothing to do with the drill. Parts XXII and XXIII fit
    logistic regressions directly for the same reason.

    The objective matches scikit-learn's exactly, including that it fits RAW
    features rather than standardised ones and that the penalty is on the summed
    rather than averaged loss:

        min_W  0.5 * ||W||^2 + C * sum_i  -log p(y_i | x_i)

    so a larger C means WEAKER regularization. Both of those matter: drill C13
    turns on a probe's regularization suppressing a signal that is present but
    small in absolute magnitude, and standardising the features would rescale
    that signal away and quietly destroy the experiment.
    """
    from scipy.optimize import minimize

    Htr = np.asarray(Htr, float); Hte = np.asarray(Hte, float)
    ytr = np.asarray(ytr).astype(int); yte = np.asarray(yte).astype(int)
    K = int(max(ytr.max(), yte.max())) + 1
    Xtr = np.column_stack([Htr, np.ones(len(Htr))])
    Xte = np.column_stack([Hte, np.ones(len(Hte))])
    d = Xtr.shape[1]
    Y = np.eye(K)[ytr]

    def obj(w):
        W = w.reshape(d, K)
        z = Xtr @ W
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        loss = float(np.sum(lse - (z * Y).sum(axis=1)))
        P = np.exp(z - m); P /= P.sum(axis=1, keepdims=True)
        G = Xtr.T @ (P - Y)
        # the penalty excludes the intercept row, as scikit-learn's does
        Wp = W.copy(); Wp[-1] = 0.0
        return 0.5 * float(np.sum(Wp ** 2)) + C * loss, (Wp + C * G).ravel()

    res = minimize(obj, np.zeros(d * K), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=4000, ftol=1e-12, gtol=1e-10))
    W = res.x.reshape(d, K)
    return float(np.mean(np.argmax(Xte @ W, axis=1) == yte))


def control_labels(identities, n_classes, seed=0):
    identities = np.asarray(identities)
    uniq = np.unique(identities)
    r = np.random.default_rng(seed)
    table = {u: int(v) for u, v in zip(uniq, r.integers(0, n_classes, len(uniq)))}
    return np.array([table[i] for i in identities])


def selectivity(Htr, ytr, Hte, yte, jtr, jte):
    return (linear_probe_accuracy(Htr, ytr, Hte, yte)
            - linear_probe_accuracy(Htr, jtr, Hte, jte))


# ------------------------------------------------------- denoising and score

def gaussian_mixture_score(P, mus, sds, weights, extra_var=0.0):
    P = np.asarray(P, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sds = np.asarray(sds, dtype=float)
    ws = np.asarray(weights, dtype=float)
    d = P.shape[1]
    logs, scores = [], []
    for m, sd, w in zip(mus, sds, ws):
        v = sd ** 2 + extra_var
        d2 = ((P - m) ** 2).sum(1)
        logs.append(np.log(w) - 0.5 * d * np.log(2 * np.pi * v) - d2 / (2 * v))
        scores.append((m - P) / v)
    L = np.stack(logs)
    R = np.exp(L - L.max(0))
    post = R / R.sum(0)
    return sum(post[i][:, None] * scores[i] for i in range(len(ws)))


def score_from_denoiser(reconstruct, X, sigma):
    return (reconstruct(X) - X) / sigma ** 2


def field_agreement(A, B):
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    cos = (A * B).sum(1) / np.maximum(na * nb, 1e-12)
    return float(cos.mean()), float(na.mean() / max(nb.mean(), 1e-12))


# -------------------------------------------------------------- sparse codes

def soft_threshold(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def sparse_code(x, D, lam, n_steps=200, lr=None):
    D = np.asarray(D, dtype=float)
    if lr is None:
        lr = 1.0 / np.linalg.eigvalsh(D @ D.T).max()
    h = np.zeros(D.shape[0])
    for _ in range(n_steps):
        h = soft_threshold(h - lr * (D @ (D.T @ h - x)), lr * lam)
    return h


def dictionary_recovery(D_true, D_learned):
    A = D_true / np.linalg.norm(D_true, axis=1, keepdims=True)
    B = D_learned / np.linalg.norm(D_learned, axis=1, keepdims=True)
    return np.abs(A @ B.T).max(1)


def feature_interference(W):
    n = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    G = n.T @ n
    off = ~np.eye(G.shape[0], dtype=bool)
    return float(np.abs(G[off]).mean())


def features_stored(W, thresh=0.35):
    return int((np.linalg.norm(W, axis=0) > thresh).sum())


# --------------------------------------------------------------------- patch


# ------------------------------------ movement G: learning as compression

def rate_distortion_gaussian(distortion, sigma2=1.0):
    d = np.asarray(distortion, float)
    return np.where(d >= sigma2, 0.0, 0.5 * np.log2(sigma2 / np.maximum(d, 1e-300)))


def distortion_at_rate(rate, sigma2=1.0):
    return sigma2 * 2.0 ** (-2.0 * np.asarray(rate, float))


def entropy_bits(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def uniform_quantise(x, step):
    idx = np.round(np.asarray(x, float) / step)
    return idx * step, idx.astype(np.int64)


def excess_rate(rate, distortion, sigma2=1.0):
    return np.asarray(rate, float) - rate_distortion_gaussian(distortion, sigma2)


def scalar_space_filling_loss():
    return float(0.5 * np.log2(np.pi * np.e / 6.0))


def gaussian_code_bits(residuals):
    r = np.asarray(residuals, float)
    s2 = max(float(np.mean(r ** 2)), 1e-300)
    return float(0.5 * len(r) * np.log2(2 * np.pi * np.e * s2))


def parameter_bits(n_params, n_samples):
    return float(0.5 * n_params * np.log2(n_samples))


def description_length(residuals, n_params):
    return gaussian_code_bits(residuals) + parameter_bits(n_params, len(residuals))


def mutual_information_bits(pxy):
    p = np.asarray(pxy, float)
    p = p / p.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (px @ py)[nz])))


def bottleneck_objective(ixz, izy, beta):
    return float(ixz) - float(beta) * float(izy)


for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
