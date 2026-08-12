"""
Reference solutions for the Part XXX drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import itertools

import numpy as np

import drills


# --------------------------------------------------------------------- mel

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, fs, fmin=0.0, fmax=None):
    fmax = fs / 2 if fmax is None else fmax
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(mid - lo, 1e-12)
        dn = (hi - freqs) / max(hi - mid, 1e-12)
        fb[m] = np.clip(np.minimum(up, dn), 0, None)
    return fb, edges


# ------------------------------------------------------------------- stft

def stft(x, win, hop):
    x = np.asarray(x, float)
    w = np.hanning(win + 1)[:-1]
    frames = 1 + (len(x) - win) // hop
    return np.array([np.fft.rfft(x[i * hop:i * hop + win] * w) for i in range(frames)])


def istft(S, win, hop, length):
    w = np.hanning(win + 1)[:-1]
    out = np.zeros(length); norm = np.zeros(length)
    for i, spec in enumerate(np.asarray(S)):
        seg = np.fft.irfft(spec, win)
        out[i * hop:i * hop + win] += seg * w
        norm[i * hop:i * hop + win] += w ** 2
    return out / np.maximum(norm, 1e-8)


def interior(x, win):
    x = np.asarray(x)
    return x[win:len(x) - win]


def griffin_lim(mag, win, hop, length, iters, seed=0):
    r = np.random.default_rng(seed)
    S = np.asarray(mag, float) * np.exp(1j * r.uniform(-np.pi, np.pi, np.shape(mag)))
    errs = []
    for _ in range(iters):
        x = istft(S, win, hop, length)
        S2 = stft(x, win, hop)
        errs.append(float(np.linalg.norm(np.abs(S2) - mag) / np.linalg.norm(mag)))
        S = mag * np.exp(1j * np.angle(S2))
    return istft(S, win, hop, length), np.array(errs)


# -------------------------------------------------------------------- ctc

def ctc_collapse(path, blank=0):
    out, prev = [], None
    for s in path:
        if s != prev and s != blank:
            out.append(int(s))
        prev = s
    return out


def ctc_expand(target, blank=0):
    out = [blank]
    for s in target:
        out += [int(s), blank]
    return np.array(out)


def ctc_logprob(logits, target, blank=0):
    lp = np.asarray(logits, float)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    ext = ctc_expand(target, blank)
    T, S = len(lp), len(ext)
    a = np.full(S, -np.inf)
    a[0] = lp[0, ext[0]]
    if S > 1:
        a[1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = a.copy()
        a = np.full(S, -np.inf)
        for s in range(S):
            v = prev[s]
            if s > 0:
                v = np.logaddexp(v, prev[s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                v = np.logaddexp(v, prev[s - 2])
            a[s] = v + lp[t, ext[s]]
    return float(np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1])


def ctc_enumerate_logprob(logits, target, blank=0):
    lp = np.asarray(logits, float)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    T, V = lp.shape
    tot = -np.inf
    for path in itertools.product(range(V), repeat=T):
        if ctc_collapse(path, blank) == list(target):
            tot = np.logaddexp(tot, sum(lp[t, s] for t, s in enumerate(path)))
    return float(tot)


def ctc_alignment_count(T, target, V=None, blank=0):
    ext = ctc_expand(target, blank)
    S = len(ext)
    a = [0] * S
    a[0] = 1
    if S > 1:
        a[1] = 1
    for _ in range(1, T):
        prev = list(a)
        a = [0] * S
        for s in range(S):
            v = prev[s]
            if s > 0:
                v += prev[s - 1]
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                v += prev[s - 2]
            a[s] = v
    return int(a[-1] + a[-2]) if S > 1 else int(a[-1])


# ----------------------------------------------------------- quantisation

def kmeans(X, k, iters=40, seed=0):
    X = np.asarray(X, float)
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return C, float(d.min(1).mean())


def scalar_quantise_error(X, levels, seed=0):
    X = np.asarray(X, float)
    return float(sum(kmeans(X[:, j:j + 1], levels, seed=seed)[1]
                     for j in range(X.shape[1])))


def residual_quantise_error(X, stages, k=16, seed=0):
    resid = np.asarray(X, float).copy()
    for s in range(stages):
        C, _ = kmeans(resid, k, seed=seed + s)
        idx = ((resid[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        resid = resid - C[idx]
    return float((resid ** 2).sum(1).mean())


def codebook_usage(X, C):
    idx = ((np.asarray(X, float)[:, None, :] - np.asarray(C, float)[None, :, :]) ** 2).sum(-1).argmin(1)
    return int(len(np.unique(idx)))


# ------------------------------------------------------------ contrastive

def contrastive_loss(A, B, tau):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    S = A @ B.T / tau
    n = len(A)
    def ce(M):
        M = M - M.max(1, keepdims=True)
        return float(-np.mean(np.diag(M) - np.log(np.exp(M).sum(1))))
    return 0.5 * (ce(S) + ce(S.T))


def effective_negatives(sims, tau):
    w = np.exp(np.asarray(sims, float) / tau)
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w + 1e-300)).sum()))


def recall_at_1(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float(((B @ A.T).argmax(1) == np.arange(len(B))).mean())


def modality_similarities(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    n = len(A)
    iu = np.triu_indices(n, 1)
    within = 0.5 * ((A @ A.T)[iu].mean() + (B @ B.T)[iu].mean())
    across = (A @ B.T).mean()
    matched = np.sum(A * B, 1).mean()
    return float(within), float(across), float(matched)


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
