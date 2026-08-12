"""
Reference solutions for the Part XIII drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))

_P = (1 << 31) - 1                      # a Mersenne prime, for the hash family


def _str_hash(x):
    """A deterministic replacement for Python's `hash()`.

    `hash()` on a string is salted per process (PYTHONHASHSEED), so a signature
    built on it is not reproducible across runs. blake2b fixes that: the same
    shingle always maps to the same integer.
    """
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little")


# ----------------------------------------------------------------- the frame ----

def rate_with_interval(draws, z=1.96):
    d = np.asarray(draws, float)
    n = len(d)
    p = float(d.mean())
    return p, float(z * np.sqrt(max(p * (1 - p), 0.0) / n))


def frame_estimate(p_groups, pop_shares, frame_shares, n, seed=0):
    r = np.random.default_rng(seed)
    p_groups = np.asarray(p_groups, float)
    pop_shares = np.asarray(pop_shares, float)
    frame_shares = np.asarray(frame_shares, float)
    truth = float(pop_shares @ p_groups)
    counts = r.multinomial(n, frame_shares / frame_shares.sum())
    draws = np.concatenate([r.random(c) < p for c, p in zip(counts, p_groups)])
    est, hw = rate_with_interval(draws)
    return est, hw, truth


def interval_covers(estimate, half_width, truth):
    return bool(abs(estimate - truth) <= half_width)


# ----------------------------------------------------------------- agreement ----

def confusion(a, b, n_classes):
    a, b = np.asarray(a, int), np.asarray(b, int)
    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(M, (a, b), 1)
    return M


def cohen_kappa(a, b, n_classes=None):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if n_classes is None:
        n_classes = int(max(a.max(), b.max())) + 1
    M = confusion(a, b, n_classes)
    n = M.sum()
    p_o = np.trace(M) / n
    p_e = float((M.sum(1) / n) @ (M.sum(0) / n))
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return float((p_o - p_e) / (1 - p_e))


def kappa_per_class(a, b, n_classes):
    a, b = np.asarray(a, int), np.asarray(b, int)
    return np.array([cohen_kappa((a == c).astype(int), (b == c).astype(int), 2)
                     for c in range(n_classes)])


# --------------------------------------------------------------- label noise ----

def flip_labels(y, p, n_classes=2, seed=0, asymmetric=False):
    r = np.random.default_rng(seed)
    y = np.asarray(y, int).copy()
    if asymmetric:
        cand = np.where(y == 0)[0]
        k = int(round(p * len(y)))
        k = min(k, len(cand))
        if k:
            y[r.choice(cand, k, replace=False)] = 1
        return y
    flip = r.random(len(y)) < p
    if n_classes == 2:
        y[flip] = 1 - y[flip]
    else:
        for i in np.where(flip)[0]:
            choices = [c for c in range(n_classes) if c != y[i]]
            y[i] = choices[r.integers(len(choices))]
    return y


def noise_ceiling(p):
    return 1.0 - p


# ------------------------------------------------------------------ leakage ----

def random_split(n, frac_test=0.3, seed=0):
    r = np.random.default_rng(seed)
    perm = r.permutation(n)
    k = int(round(frac_test * n))
    return perm[k:], perm[:k]


def grouped_split(groups, frac_test=0.3, seed=0):
    groups = np.asarray(groups)
    r = np.random.default_rng(seed)
    uniq = r.permutation(np.unique(groups))
    k = max(1, int(round(frac_test * len(uniq))))
    test_groups = set(uniq[:k].tolist())
    mask = np.array([g in test_groups for g in groups])
    return np.where(~mask)[0], np.where(mask)[0]


def leakage_rate(train_idx, test_idx, groups):
    groups = np.asarray(groups)
    tr = set(groups[np.asarray(train_idx, int)].tolist())
    te = groups[np.asarray(test_idx, int)]
    if len(te) == 0:
        return 0.0
    return float(np.mean([g in tr for g in te]))


def contamination_inflation(scores_all, scores_clean):
    return float(scores_all) - float(scores_clean)


# ------------------------------------------------------------- near-duplicates ----

def shingles(text, k=5):
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def minhash_signature(shingle_set, n_perm, seed=0):
    r = np.random.default_rng(seed)
    A = r.integers(1, _P, n_perm)
    B = r.integers(0, _P, n_perm)
    if not shingle_set:
        return np.full(n_perm, _P, dtype=np.int64)
    h = np.array([_str_hash(s) % _P for s in shingle_set], dtype=np.int64)
    return np.array([int(((A[i] * h + B[i]) % _P).min()) for i in range(n_perm)],
                    dtype=np.int64)


def minhash_similarity(sig_a, sig_b):
    a, b = np.asarray(sig_a), np.asarray(sig_b)
    return float((a == b).mean())


def lsh_bands(signature, n_bands):
    sig = np.asarray(signature)
    rows = len(sig) // n_bands
    return [tuple(sig[i * rows:(i + 1) * rows].tolist()) for i in range(n_bands)]


def lsh_threshold(n_bands, rows_per_band):
    return float((1.0 / n_bands) ** (1.0 / rows_per_band))


def lsh_candidate_probability(similarity, n_bands, rows_per_band):
    s = np.asarray(similarity, float)
    return 1.0 - (1.0 - s ** rows_per_band) ** n_bands


# ------------------------------------------------------------------ collapse ----

def collapse_generations(n, generations, keep=1.0, seed=0):
    r = np.random.default_rng(seed)
    mu, sd = 0.0, 1.0
    out = [1.0]
    for _ in range(generations):
        s = r.normal(mu, sd, n)
        if keep < 1.0:
            k = max(2, int(round(keep * len(s))))
            s = s[np.argsort(np.abs(s - s.mean()))[:k]]
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        out.append(sd)
    return np.array(out)


def mode_count(samples, bins=60, min_prominence=0.15):
    h, _ = np.histogram(np.asarray(samples, float), bins=bins)
    if h.max() == 0:
        return 0
    h = h.astype(float)
    # smooth, or histogram noise reads as modes: a unimodal Gaussian gives 4
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    h = np.convolve(h, k, mode="same")
    h /= h.max()
    peaks = [i for i in range(1, len(h) - 1)
             if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] >= min_prominence]
    # keep a peak only if a deep enough valley separates it from the previous one
    kept = []
    for i in peaks:
        if not kept:
            kept.append(i)
            continue
        valley = h[kept[-1]:i + 1].min()
        if min(h[kept[-1]], h[i]) - valley >= min_prominence:
            kept.append(i)
        elif h[i] > h[kept[-1]]:
            kept[-1] = i
    return max(len(kept), 1)


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
