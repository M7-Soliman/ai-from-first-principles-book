"""
Reference solutions for the Part XXXIII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import heapq
import itertools
from collections import Counter

import numpy as np

import drills


# --------------------------------------------------------------- heuristics

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def misplaced_tiles(state, goal):
    return sum(1 for s, g in zip(state, goal) if s != g and s != 0)


def is_admissible(h, h_star, tol=1e-9):
    h = np.asarray(h, float)
    h_star = np.asarray(h_star, float)
    return bool(np.all(h <= h_star + tol))


def is_consistent(edges, h, tol=1e-9):
    return all(h[u] <= cost + h[v] + tol for u, v, cost in edges)


def dominates(h1, h2, tol=1e-9):
    return bool(np.all(np.asarray(h1, float) >= np.asarray(h2, float) - tol))


def max_heuristic(*hs):
    out = np.asarray(hs[0], float)
    for h in hs[1:]:
        out = np.maximum(out, np.asarray(h, float))
    return out


def weighted_bound(w, optimal_cost):
    return float(w) * float(optimal_cost)


# ------------------------------------------------------------------------ A*

def astar(neighbours, start, goal, h):
    g = {start: 0.0}
    parent = {start: None}
    tie = itertools.count()
    open_ = [(h(start), next(tie), start)]
    expansions = 0
    while open_:
        f, _, u = heapq.heappop(open_)
        # A cheaper path to u makes this entry stale. Dropping only stale
        # entries — rather than everything already expanded — is what lets an
        # inconsistent heuristic reopen a settled state and stay optimal.
        if f > g[u] + h(u) + 1e-12:
            continue
        expansions += 1
        if u == goal:
            path, node = [], u
            while node is not None:
                path.append(node)
                node = parent[node]
            return g[u], path[::-1], expansions
        for v, cost in neighbours(u):
            ng = g[u] + cost
            if ng < g.get(v, np.inf) - 1e-12:
                g[v] = ng
                parent[v] = u
                heapq.heappush(open_, (ng + h(v), next(tie), v))
    return np.inf, None, expansions


# --------------------------------------------------------------- beam search

def viterbi(scores):
    scores = np.asarray(scores, float)
    L, V = scores.shape[0], scores.shape[1]
    dp = scores[0][0].copy()
    bp = np.zeros((L, V), dtype=int)
    for t in range(1, L):
        cand = dp[:, None] + scores[t]
        bp[t] = cand.argmax(0)
        dp = cand.max(0)
    v = int(dp.argmax())
    seq = [v]
    for t in range(L - 1, 0, -1):
        v = int(bp[t, v])
        seq.append(v)
    return float(dp.max()), seq[::-1]


def beam_search(scores, width):
    scores = np.asarray(scores, float)
    L, V = scores.shape[0], scores.shape[1]
    beams = sorted(((float(scores[0][0, v]), [v]) for v in range(V)),
                   key=lambda b: -b[0])[:width]
    for t in range(1, L):
        cand = [(s + float(scores[t][seq[-1], v]), seq + [v])
                for s, seq in beams for v in range(V)]
        cand.sort(key=lambda b: -b[0])
        beams = cand[:width]
    return beams[0][0], beams[0][1]


# ---------------------------------------------------------- branch and bound

def fractional_bound(values, weights, capacity):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if len(v) == 0 or capacity <= 0:
        return 0.0
    order = np.argsort(-(v / w))
    total, room = 0.0, float(capacity)
    for i in order:
        if w[i] <= room:
            total += v[i]
            room -= w[i]
        else:
            total += v[i] * room / w[i]
            break
    return float(total)


def knapsack_bnb(values, weights, capacity, tight=True):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(-(v / w))
    v, w = v[order], w[order]
    n = len(v)
    suffix = np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])
    best, nodes = 0.0, 0
    stack = [(0, 0.0, 0.0)]
    while stack:
        i, val, wt = stack.pop()
        nodes += 1
        if i == n:
            best = max(best, val)
            continue
        if tight:
            bound = val + fractional_bound(v[i:], w[i:], capacity - wt)
        else:
            bound = val + suffix[i]
        if bound <= best + 1e-9:
            continue
        best = max(best, val)
        if wt + w[i] <= capacity:
            stack.append((i + 1, val + v[i], wt + w[i]))
        stack.append((i + 1, val, wt))
    return float(best), nodes


# ---------------------------------------------------------------- game trees

def _depth(leaves, b):
    n, d = len(leaves), 0
    while n > 1:
        n //= b
        d += 1
    return d


def minimax(leaves, b):
    leaves = np.asarray(leaves, float)
    d = _depth(leaves, b)

    def go(node, depth, maximizing):
        if depth == d:
            return float(leaves[node]), 1
        vals, total = [], 0
        for i in range(b):
            v, c = go(node * b + i, depth + 1, not maximizing)
            vals.append(v)
            total += c
        return (max(vals) if maximizing else min(vals)), total

    return go(0, 0, True)


def alphabeta(leaves, b):
    leaves = np.asarray(leaves, float)
    d = _depth(leaves, b)

    def go(node, depth, alpha, beta, maximizing):
        if depth == d:
            return float(leaves[node]), 1
        total = 0
        if maximizing:
            best = -np.inf
            for i in range(b):
                v, c = go(node * b + i, depth + 1, alpha, beta, False)
                total += c
                best = max(best, v)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = np.inf
            for i in range(b):
                v, c = go(node * b + i, depth + 1, alpha, beta, True)
                total += c
                best = min(best, v)
                beta = min(beta, best)
                if alpha >= beta:
                    break
        return best, total

    return go(0, 0, -np.inf, np.inf, True)


def knuth_moore(b, d):
    return b ** ((d + 1) // 2) + b ** (d // 2) - 1


# ------------------------------------------------------------------- bandits

def ucb_scores(sums, counts, c):
    sums = np.asarray(sums, float)
    counts = np.asarray(counts, float)
    total = max(counts.sum(), 1.0)
    out = np.full(counts.shape, np.inf)
    seen = counts > 0
    out[seen] = (sums[seen] / counts[seen]
                 + c * np.sqrt(np.log(total) / counts[seen]))
    return out


def expected_restarts(p):
    return 1.0 / float(p)


def id_overhead(b):
    return float(b) / (float(b) - 1.0)


# ----------------------------------------------------- sampling and verifiers

def pass_at_k(p, k):
    return 1.0 - (1.0 - float(p)) ** int(k)


def selection_solve_rate(p, k, tpr, fpr):
    p, tpr, fpr, k = float(p), float(tpr), float(fpr), int(k)
    acc_right = p * tpr                       # accepted and correct
    acc_wrong = (1.0 - p) * fpr               # accepted and wrong
    acc = acc_right + acc_wrong
    q = 1.0 - acc                             # a given sample is not accepted
    if acc <= 0.0:
        return p                              # nothing is ever accepted
    if q <= 0.0:
        return acc_right / acc                # something is always accepted
    # Conditioning matters here. Given that nothing was accepted, the first
    # sample is *not* correct with probability p — being rejected is evidence
    # against it. Using p for the fallback makes a perfect verifier look better
    # than pass@k, which is impossible.
    fallback = p * (1.0 - tpr) / q
    return (acc_right / acc) * (1.0 - q ** k) + (q ** k) * fallback


def majority_vote(answers):
    counts = Counter(answers)
    top = max(counts.values())
    for a in answers:                          # earliest occurrence breaks the tie
        if counts[a] == top:
            return a
    return None


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
