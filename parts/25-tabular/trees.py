"""
Decision trees, bagging and gradient boosting, written out.

Everything Part XXV measures is built here rather than imported. That is partly
the book's habit and partly a practical lesson learned the hard way: a figure
that cannot be regenerated without an optional package is a figure whose claim
cannot be checked the day that package stops being installed.

Splits are found on a histogram of at most `n_bins` thresholds per feature,
which is what every production implementation does and is the difference
between a tree that fits in a figure script and one that does not.
"""
import numpy as np


# --------------------------------------------------------------------- binning

def fit_bins(X, n_bins=64):
    """Per-feature bin edges at quantiles of the training data."""
    X = np.asarray(X, float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]


def apply_bins(X, edges):
    X = np.asarray(X, float)
    return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right")
                            for j in range(X.shape[1])]).astype(np.int32)


# ------------------------------------------------------------------ the tree

class Tree:
    """A regression tree on pre-binned features, fitted to arbitrary targets.

    Boosting needs a tree fitted to gradients rather than to labels, so the fit
    takes the target directly and the leaf value is whatever `leaf_value` says.
    """

    def __init__(self, max_depth=6, min_samples=10, n_bins=64,
                 feature_fraction=1.0, lam=0.0, rng=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_bins = n_bins
        self.feature_fraction = feature_fraction
        self.lam = lam                       # L2 on the leaf value, as in XGBoost
        self.rng = rng or np.random.default_rng(0)

    # leaf value for a plain regression tree: the mean
    def _leaf(self, g, h):
        return float(g.sum() / (h.sum() + self.lam)) if h is not None else float(g.mean())

    def fit(self, Xb, g, h=None, n_bin_values=None):
        """Xb is integer-binned. `g` is the target (or the negative gradient);
        `h` the second-order weights, or None for a least-squares tree."""
        self.nbv = n_bin_values if n_bin_values is not None else int(Xb.max()) + 1
        self.nodes = []
        idx = np.arange(len(Xb))
        self._build(Xb, g, h, idx, 0)
        return self

    def _build(self, Xb, g, h, idx, depth):
        node = len(self.nodes)
        self.nodes.append(None)
        gi = g[idx]
        hi = None if h is None else h[idx]
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples:
            self.nodes[node] = ("leaf", self._leaf(gi, hi))
            return node

        n_feat = Xb.shape[1]
        k = max(1, int(round(self.feature_fraction * n_feat)))
        feats = (np.arange(n_feat) if k == n_feat
                 else self.rng.choice(n_feat, k, replace=False))

        best = None
        parent = self._score(gi.sum(), len(gi) if hi is None else hi.sum())
        for j in feats:
            col = Xb[idx, j]
            Gs = np.bincount(col, weights=gi, minlength=self.nbv)
            Hs = (np.bincount(col, minlength=self.nbv) if hi is None
                  else np.bincount(col, weights=hi, minlength=self.nbv))
            GL, HL = np.cumsum(Gs), np.cumsum(Hs)
            GT, HT = GL[-1], HL[-1]
            GR, HR = GT - GL, HT - HL
            valid = (HL >= self.min_samples) & (HR >= self.min_samples)
            if not valid.any():
                continue
            gain = self._score(GL, HL) + self._score(GR, HR) - parent
            gain[~valid] = -np.inf
            b = int(np.argmax(gain))
            if best is None or gain[b] > best[0]:
                best = (float(gain[b]), int(j), b)

        if best is None or best[0] <= 1e-12:
            self.nodes[node] = ("leaf", self._leaf(gi, hi))
            return node

        _, j, thr = best
        left = idx[Xb[idx, j] <= thr]
        right = idx[Xb[idx, j] > thr]
        l = self._build(Xb, g, h, left, depth + 1)
        r = self._build(Xb, g, h, right, depth + 1)
        self.nodes[node] = ("split", j, thr, l, r)
        return node

    def _score(self, G, H):
        """The reduction in loss a node achieves. For a least-squares tree this
        is the familiar sum-of-squares term; the +lam is XGBoost's."""
        return (G ** 2) / (np.asarray(H, float) + self.lam + 1e-12)

    def predict(self, Xb):
        out = np.empty(len(Xb))
        for i in range(len(Xb)):
            n = 0
            while True:
                node = self.nodes[n]
                if node[0] == "leaf":
                    out[i] = node[1]; break
                _, j, thr, l, r = node
                n = l if Xb[i, j] <= thr else r
        return out

    def n_leaves(self):
        return sum(1 for n in self.nodes if n[0] == "leaf")


# ------------------------------------------------------------------- wrappers

class Forest:
    """Bagged trees. `feature_fraction < 1` makes it a random forest."""

    def __init__(self, n_trees=100, max_depth=12, min_samples=2, n_bins=64,
                 feature_fraction=1.0, bootstrap=True, seed=0):
        self.n_trees = n_trees; self.max_depth = max_depth
        self.min_samples = min_samples; self.n_bins = n_bins
        self.feature_fraction = feature_fraction; self.bootstrap = bootstrap
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.edges = fit_bins(X, self.n_bins)
        Xb = apply_bins(X, self.edges)
        nbv = self.n_bins + 1
        self.trees = []
        n = len(X)
        for t in range(self.n_trees):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tr = Tree(self.max_depth, self.min_samples, self.n_bins,
                      self.feature_fraction, 0.0, np.random.default_rng(self.seed + t))
            tr.fit(Xb[idx], np.asarray(y, float)[idx], None, nbv)
            self.trees.append(tr)
        return self

    def predict(self, X):
        Xb = apply_bins(X, self.edges)
        return np.mean([t.predict(Xb) for t in self.trees], axis=0)

    def staged_predict(self, X):
        """The running average as trees are added — what variance reduction
        looks like from the outside."""
        Xb = apply_bins(X, self.edges)
        total = np.zeros(len(X))
        for k, t in enumerate(self.trees, 1):
            total += t.predict(Xb)
            yield total / k


class GBM:
    """Gradient boosting. Squared loss for regression, logistic for binary."""

    def __init__(self, n_trees=200, lr=0.1, max_depth=3, min_samples=10,
                 n_bins=64, lam=1.0, subsample=1.0, objective="ls", seed=0):
        self.n_trees = n_trees; self.lr = lr; self.max_depth = max_depth
        self.min_samples = min_samples; self.n_bins = n_bins; self.lam = lam
        self.subsample = subsample; self.objective = objective; self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, float)
        self.edges = fit_bins(X, self.n_bins)
        Xb = apply_bins(X, self.edges)
        nbv = self.n_bins + 1
        n = len(X)
        if self.objective == "ls":
            self.base = float(y.mean())
        else:
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base = float(np.log(p / (1 - p)))
        F = np.full(n, self.base)
        self.trees = []
        for t in range(self.n_trees):
            if self.objective == "ls":
                g, h = y - F, np.ones(n)
            else:
                p = 1 / (1 + np.exp(-F))
                g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
            idx = (np.arange(n) if self.subsample >= 1.0
                   else rng.choice(n, int(self.subsample * n), replace=False))
            tr = Tree(self.max_depth, self.min_samples, self.n_bins, 1.0,
                      self.lam, np.random.default_rng(self.seed + t))
            tr.fit(Xb[idx], g[idx], h[idx], nbv)
            F += self.lr * tr.predict(Xb)
            self.trees.append(tr)
        return self

    def decision(self, X, n_trees=None):
        Xb = apply_bins(X, self.edges)
        F = np.full(len(X), self.base)
        for t in self.trees[:n_trees or len(self.trees)]:
            F += self.lr * t.predict(Xb)
        return F

    def predict(self, X, n_trees=None):
        F = self.decision(X, n_trees)
        return F if self.objective == "ls" else 1 / (1 + np.exp(-F))

    def staged_decision(self, X):
        Xb = apply_bins(X, self.edges)
        F = np.full(len(X), self.base)
        for t in self.trees:
            F += self.lr * t.predict(Xb)
            yield F.copy()
