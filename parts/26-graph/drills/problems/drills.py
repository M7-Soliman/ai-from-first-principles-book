"""
Part XXVI drills — Graph and Geometric Learning.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail. Reference solutions are in
../solutions/reference.py; read them after you have tried, not before.

Numpy only. Dense adjacency matrices throughout — the graphs here are small
enough, and the sparse machinery would hide the arithmetic that is the point.
"""
import numpy as np


# ------------------------------------------------------------- representation

def edge_list(A):
    """Directed edge list from a dense adjacency. Return an (E, 2) integer
    array of (source, target) pairs — both directions for an undirected graph,
    which is what a message-passing implementation actually consumes."""
    raise NotImplementedError


def permute_graph(X, A, perm):
    """Relabel the nodes. `perm[i]` is the old index of new node i, so the new
    features are X[perm] and the new adjacency is A permuted on both axes.
    Return (X_new, A_new)."""
    raise NotImplementedError


def is_invariant(fn, X, A, perm, tol=1e-8):
    """Does fn(X, A) return the same value after relabelling? Three lines, and
    the cheapest bug-catcher in this part."""
    raise NotImplementedError


def is_equivariant(fn, X, A, perm, tol=1e-8):
    """Does fn's per-node output permute with the nodes?"""
    raise NotImplementedError


# --------------------------------------------------------------- message pass

def aggregate(X, edges, n, how="sum"):
    """Gather each node's incoming messages and combine them. `how` is one of
    "sum", "mean", "max". Return an (n, d) array. Nodes with no incoming edge
    get zeros."""
    raise NotImplementedError


def mp_layer(X, edges, W_self, W_neigh, how="mean"):
    """One message-passing layer: transform the node's own features, add the
    transformed aggregate of its neighbours, squash with tanh."""
    raise NotImplementedError


def normalised_adjacency(A):
    """The symmetric normalisation with self-loops that the standard graph
    convolution uses: D^{-1/2} (A + I) D^{-1/2}."""
    raise NotImplementedError


def smooth(A_hat, X, k):
    """Apply the propagation operator k times. This is what depth does to the
    representations when the transformation is removed."""
    raise NotImplementedError


def mean_pairwise_distance(H):
    """Mean Euclidean distance between distinct rows of H — the statistic that
    goes to zero when representations collapse."""
    raise NotImplementedError


# ------------------------------------------------------------------ the limit

def wl_refine(A, colours=None, rounds=3):
    """Colour refinement. Start from `colours` (degree if None); each round,
    replace a node's colour by a canonical id for the pair (own colour, sorted
    multiset of neighbour colours). Return the final integer colours."""
    raise NotImplementedError


def wl_distinguishable(A1, A2, rounds=3):
    """Can refinement tell the two graphs apart? Refine them together so the
    colour ids are comparable, then compare the colour histograms."""
    raise NotImplementedError


# ------------------------------------------------------------------- distance

def shortest_path_lengths(A, src):
    """BFS distances from src. Unreachable nodes get -1."""
    raise NotImplementedError


def receptive_field(A, node, k):
    """Boolean mask of the nodes a k-layer network can see from `node`."""
    raise NotImplementedError


def influence_share(A, src, k):
    """Share of each node's k-round representation that came from `src`, under
    row-normalised propagation with self-loops. Return the vector over targets.
    This is the over-squashing measurement of Figure 4."""
    raise NotImplementedError


# --------------------------------------------------------------------- tasks

def homophily(A, y):
    """Fraction of undirected edges joining nodes with the same label. The one
    number to compute before choosing an architecture."""
    raise NotImplementedError


def label_propagation(A, y, labelled, iters=30):
    """Spread the known labels along edges with no learning at all, clamping
    the labelled nodes each round. Return predicted labels for every node."""
    raise NotImplementedError


def sample_negatives(A, n_neg, seed=0, mode="uniform"):
    """Non-edges to score a link predictor against. "uniform" draws random
    non-adjacent pairs; "hard" draws pairs at exactly distance two. Return an
    (m, 2) array. The metric depends on which you pick."""
    raise NotImplementedError


def remove_edges(A, edges):
    """A copy of A with the given undirected edges deleted — what has to happen
    to the message-passing graph before predicting them."""
    raise NotImplementedError


def filtered_rank(scores, true_idx, known_true=()):
    """Rank of the true candidate after removing other known-true candidates
    from the comparison set. 1 is best."""
    raise NotImplementedError


# ------------------------------------------------------------------- geometry

def pairwise_distance_features(P):
    """A sorted vector of pairwise distances — invariant to rotation,
    translation, reflection and node ordering, all at once, and free."""
    raise NotImplementedError


def random_rotation(seed=0):
    """A uniformly random 3x3 rotation matrix, determinant +1."""
    raise NotImplementedError


def equivariant_update(P, V, edges, w):
    """Update node vectors as a weighted sum of relative position vectors:
    V[t] += sum over edges (s, t) of w_e * (P[t] - P[s]). Every term rotates
    the same way, so the output does."""
    raise NotImplementedError
