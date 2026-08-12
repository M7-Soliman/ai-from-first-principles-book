"""
Reference solutions for the Part XXVI drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------- representation

def edge_list(A):
    A = np.asarray(A)
    src, dst = np.nonzero(A)
    return np.column_stack([src, dst]).astype(np.int64)


def permute_graph(X, A, perm):
    perm = np.asarray(perm, int)
    return np.asarray(X)[perm], np.asarray(A)[np.ix_(perm, perm)]


def is_invariant(fn, X, A, perm, tol=1e-8):
    Xp, Ap = permute_graph(X, A, perm)
    return bool(np.max(np.abs(np.asarray(fn(X, A)) - np.asarray(fn(Xp, Ap)))) <= tol)


def is_equivariant(fn, X, A, perm, tol=1e-8):
    perm = np.asarray(perm, int)
    Xp, Ap = permute_graph(X, A, perm)
    return bool(np.max(np.abs(np.asarray(fn(X, A))[perm] - np.asarray(fn(Xp, Ap)))) <= tol)


# --------------------------------------------------------------- message pass

def aggregate(X, edges, n, how="sum"):
    X = np.asarray(X, float)
    edges = np.asarray(edges, int).reshape(-1, 2)
    out = np.zeros((n, X.shape[1]))
    if len(edges) == 0:
        return out
    src, dst = edges[:, 0], edges[:, 1]
    if how == "max":
        out[:] = -np.inf
        np.maximum.at(out, dst, X[src])
        out[~np.isfinite(out)] = 0.0
        return out
    np.add.at(out, dst, X[src])
    if how == "mean":
        deg = np.bincount(dst, minlength=n).astype(float)
        out /= np.maximum(deg, 1.0)[:, None]
    return out


def mp_layer(X, edges, W_self, W_neigh, how="mean"):
    X = np.asarray(X, float)
    m = aggregate(X, edges, len(X), how)
    return np.tanh(X @ np.asarray(W_self, float) + m @ np.asarray(W_neigh, float))


def normalised_adjacency(A):
    A = np.asarray(A, float) + np.eye(len(A))
    d = A.sum(1)
    dm = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return A * dm[:, None] * dm[None, :]


def smooth(A_hat, X, k):
    H = np.asarray(X, float)
    for _ in range(k):
        H = np.asarray(A_hat, float) @ H
    return H


def mean_pairwise_distance(H):
    H = np.asarray(H, float)
    n = len(H)
    if n < 2:
        return 0.0
    d = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=-1)
    return float(d[np.triu_indices(n, 1)].mean())


# ------------------------------------------------------------------ the limit

def wl_refine(A, colours=None, rounds=3):
    A = np.asarray(A)
    n = len(A)
    c = (np.asarray(A > 0).sum(1) if colours is None
         else np.asarray(colours)).astype(np.int64)
    _, c = np.unique(c, return_inverse=True)
    for _ in range(rounds):
        sigs = []
        for i in range(n):
            nb = tuple(sorted(c[j] for j in np.nonzero(A[i])[0]))
            sigs.append((int(c[i]), nb))
        order = {s: k for k, s in enumerate(sorted(set(sigs)))}
        c = np.array([order[s] for s in sigs], np.int64)
    return c


def wl_distinguishable(A1, A2, rounds=3):
    A1 = np.asarray(A1); A2 = np.asarray(A2)
    n1, n2 = len(A1), len(A2)
    A = np.zeros((n1 + n2, n1 + n2), int)
    A[:n1, :n1] = A1
    A[n1:, n1:] = A2
    c = wl_refine(A, rounds=rounds)
    k = int(c.max()) + 1
    return not np.array_equal(np.bincount(c[:n1], minlength=k),
                              np.bincount(c[n1:], minlength=k))


# ------------------------------------------------------------------- distance

def shortest_path_lengths(A, src):
    A = np.asarray(A)
    n = len(A)
    d = np.full(n, -1)
    d[src] = 0
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.nonzero(A[u])[0]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    return d


def receptive_field(A, node, k):
    d = shortest_path_lengths(A, node)
    return (d >= 0) & (d <= k)


def influence_share(A, src, k):
    """Share of node t's k-round representation that came from `src`, under
    row-normalised propagation with self-loops. Returns the vector over t."""
    A = np.asarray(A, float) + np.eye(len(A))
    M = A / A.sum(1, keepdims=True)
    P = np.linalg.matrix_power(M, k)
    return P[:, src]


# --------------------------------------------------------------------- tasks

def homophily(A, y):
    A = np.asarray(A)
    y = np.asarray(y)
    src, dst = np.nonzero(np.triu(A, 1))
    if len(src) == 0:
        return 0.0
    return float((y[src] == y[dst]).mean())


def label_propagation(A, y, labelled, iters=30):
    A = np.asarray(A, float) + np.eye(len(A))
    M = A / A.sum(1, keepdims=True)
    y = np.asarray(y, int)
    labelled = np.asarray(labelled, bool)
    k = int(y.max()) + 1
    F = np.zeros((len(y), k))
    F[labelled, y[labelled]] = 1.0
    clamp = F.copy()
    for _ in range(iters):
        F = M @ F
        F[labelled] = clamp[labelled]
    return F.argmax(1)


def sample_negatives(A, n_neg, seed=0, mode="uniform"):
    A = np.asarray(A)
    n = len(A)
    rng = np.random.default_rng(seed)
    out = []
    if mode == "hard":
        two = (A @ A > 0) & (A == 0)
        np.fill_diagonal(two, False)
        cand = np.column_stack(np.nonzero(np.triu(two, 1)))
        if len(cand) == 0:
            return np.zeros((0, 2), int)
        pick = rng.choice(len(cand), min(n_neg, len(cand)), replace=False)
        return cand[pick]
    while len(out) < n_neg:
        i, j = rng.integers(0, n, 2)
        if i != j and A[i, j] == 0:
            out.append((int(i), int(j)))
    return np.array(out, int)


def remove_edges(A, edges):
    A = np.array(A, copy=True)
    for i, j in np.asarray(edges, int).reshape(-1, 2):
        A[i, j] = 0
        A[j, i] = 0
    return A


def filtered_rank(scores, true_idx, known_true=()):
    scores = np.asarray(scores, float)
    mask = np.ones(len(scores), bool)
    for k in known_true:
        mask[k] = False
    mask[true_idx] = True
    s = scores[mask]
    t = scores[true_idx]
    return int((s > t).sum()) + 1


# ------------------------------------------------------------------- geometry

def pairwise_distance_features(P):
    P = np.asarray(P, float)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return np.sort(d[np.triu_indices(len(P), 1)])


def random_rotation(seed=0):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def equivariant_update(P, V, edges, w):
    P = np.asarray(P, float)
    V = np.array(V, float, copy=True)
    edges = np.asarray(edges, int).reshape(-1, 2)
    w = np.asarray(w, float).reshape(-1, 1)
    rel = P[edges[:, 1]] - P[edges[:, 0]]
    np.add.at(V, edges[:, 1], w * rel)
    return V


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
