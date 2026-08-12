"""
Tests for the Part XXVI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part's subject includes what
provably cannot work: that refinement cannot separate two regular graphs, that
mean aggregation cannot count, and that repeated propagation destroys the
representations it is meant to build.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def ring(n):
    A = np.zeros((n, n), int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def path(n):
    A = np.zeros((n, n), int)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


def random_graph(n, p, seed=0):
    r = np.random.default_rng(seed)
    A = (r.random((n, n)) < p).astype(int)
    A = np.triu(A, 1)
    return A + A.T


# ------------------------------------------------------------- representation

def test_edge_list_has_both_directions():
    A = path(4)
    E = attempt(drills.edge_list, A)
    assert len(E) == 6
    assert {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)} == {tuple(e) for e in E}


def test_permutation_preserves_the_graph():
    A = random_graph(12, 0.3, 1)
    X = np.random.default_rng(2).standard_normal((12, 3))
    perm = np.random.default_rng(3).permutation(12)
    Xp, Ap = attempt(drills.permute_graph, X, A, perm)
    # the same edges, renamed: degree multiset is unchanged
    assert sorted(Ap.sum(1)) == sorted(A.sum(1))
    assert np.allclose(np.sort(Xp, axis=0), np.sort(X, axis=0))


def test_a_sum_readout_is_invariant_and_a_first_row_readout_is_not():
    A = random_graph(10, 0.3, 4)
    X = np.random.default_rng(5).standard_normal((10, 2))
    perm = np.random.default_rng(6).permutation(10)
    assert attempt(drills.is_invariant, lambda X, A: X.sum(0), X, A, perm)
    assert not attempt(drills.is_invariant, lambda X, A: X[0], X, A, perm)


def test_degree_is_equivariant():
    A = random_graph(10, 0.35, 7)
    X = np.random.default_rng(8).standard_normal((10, 2))
    perm = np.random.default_rng(9).permutation(10)
    assert attempt(drills.is_equivariant, lambda X, A: A.sum(1).astype(float), X, A, perm)


# --------------------------------------------------------------- message pass

def test_sum_counts_and_mean_does_not():
    """A star with two leaves and one with four. Sum tells them apart."""
    for how, same in [("sum", False), ("mean", True), ("max", True)]:
        vals = []
        for k in (2, 4):
            n = k + 1
            A = np.zeros((n, n), int)
            A[0, 1:] = A[1:, 0] = 1
            X = np.ones((n, 1))
            m = attempt(drills.aggregate, X, attempt(drills.edge_list, A), n, how)
            vals.append(float(m[0, 0]))
        assert (vals[0] == pytest.approx(vals[1])) is same, how


def test_aggregate_leaves_isolated_nodes_at_zero():
    A = np.zeros((3, 3), int)
    A[0, 1] = A[1, 0] = 1
    m = attempt(drills.aggregate, np.ones((3, 2)), attempt(drills.edge_list, A), 3, "sum")
    assert np.allclose(m[2], 0.0)


def test_a_message_passing_layer_is_equivariant():
    A = random_graph(14, 0.3, 10)
    r = np.random.default_rng(11)
    X = r.standard_normal((14, 4))
    Ws, Wn = r.standard_normal((4, 4)), r.standard_normal((4, 4))

    def f(X, A):
        return drills.mp_layer(X, drills.edge_list(A), Ws, Wn)

    assert attempt(drills.is_equivariant, f, X, A, r.permutation(14), 1e-8)


def test_normalised_adjacency_is_symmetric_with_unit_spectral_radius():
    A = random_graph(20, 0.2, 12)
    H = attempt(drills.normalised_adjacency, A)
    assert np.allclose(H, H.T)
    assert np.max(np.abs(np.linalg.eigvalsh(H))) == pytest.approx(1.0, abs=1e-8)


def test_propagation_collapses_the_representations():
    """Over-smoothing, in six lines. The distances shrink monotonically and by
    a factor of tens — the operator is a diffusion and it has one fixed
    point."""
    A = random_graph(60, 0.12, 13)
    X = np.random.default_rng(14).standard_normal((60, 8))
    H = attempt(drills.normalised_adjacency, A)
    d = [attempt(drills.mean_pairwise_distance, attempt(drills.smooth, H, X, k))
         for k in (0, 2, 8, 32)]
    assert d[0] > d[1] > d[2] > d[3]
    assert d[-1] < 0.05 * d[0]


def test_what_survives_the_collapse_is_one_direction():
    """The precise statement, which is worth knowing because the loose one is
    wrong for this operator. With the symmetric normalisation the rows do not
    go to a constant — they go to a common direction scaled by the square root
    of degree. Every pair of nodes ends up perfectly aligned, so the only
    information left is degree."""
    A = random_graph(60, 0.12, 13)
    X = np.random.default_rng(14).standard_normal((60, 8))
    H = attempt(drills.normalised_adjacency, A)
    Z = attempt(drills.smooth, H, X, 64)
    U = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    assert np.min(U @ U.T) > 0.9999
    # and the surviving scale is exactly the square root of degree
    scale = np.linalg.norm(Z, axis=1)
    root_deg = np.sqrt(A.sum(1) + 1)
    assert np.corrcoef(scale, root_deg)[0, 1] > 0.999


# ------------------------------------------------------------------ the limit

def test_refinement_separates_a_path_from_a_ring():
    assert attempt(drills.wl_distinguishable, path(6), ring(6), 3)


def test_refinement_cannot_separate_two_regular_graphs():
    """Two 2-regular graphs on six nodes: one hexagon, two triangles. Every
    node has degree two in both, so refinement never splits a colour class.
    This is a proof, not a limitation of the test's budget."""
    two_triangles = np.zeros((6, 6), int)
    for a, b, c in [(0, 1, 2), (3, 4, 5)]:
        for i, j in [(a, b), (b, c), (c, a)]:
            two_triangles[i, j] = two_triangles[j, i] = 1
    assert not attempt(drills.wl_distinguishable, ring(6), two_triangles, 10)


def test_features_break_the_tie_that_structure_cannot():
    two_triangles = np.zeros((6, 6), int)
    for a, b, c in [(0, 1, 2), (3, 4, 5)]:
        for i, j in [(a, b), (b, c), (c, a)]:
            two_triangles[i, j] = two_triangles[j, i] = 1
    assert not attempt(drills.wl_distinguishable, ring(6), two_triangles, 10)
    # a triangle count is exactly what refinement cannot compute
    tri_ring = np.diag(np.linalg.matrix_power(ring(6), 3)) // 2
    tri_tt = np.diag(np.linalg.matrix_power(two_triangles, 3)) // 2
    assert not np.array_equal(np.sort(tri_ring), np.sort(tri_tt))
    # so hand it to the model as a feature: refine the two graphs together,
    # starting from the triangle counts, and they separate at once
    U = np.zeros((12, 12), int)
    U[:6, :6], U[6:, 6:] = ring(6), two_triangles
    c = attempt(drills.wl_refine, U, np.concatenate([tri_ring, tri_tt]), 2)
    assert not np.array_equal(np.sort(c[:6]), np.sort(c[6:]))


def test_refinement_is_permutation_invariant():
    A = random_graph(16, 0.25, 15)
    perm = np.random.default_rng(16).permutation(16)
    _, Ap = attempt(drills.permute_graph, np.zeros((16, 1)), A, perm)
    c1 = attempt(drills.wl_refine, A, None, 3)
    c2 = attempt(drills.wl_refine, Ap, None, 3)
    assert np.array_equal(np.sort(c1), np.sort(c2))


# ------------------------------------------------------------------- distance

def test_shortest_paths_on_a_path():
    d = attempt(drills.shortest_path_lengths, path(7), 0)
    assert list(d) == [0, 1, 2, 3, 4, 5, 6]


def test_unreachable_nodes_are_marked():
    A = np.zeros((4, 4), int)
    A[0, 1] = A[1, 0] = 1
    d = attempt(drills.shortest_path_lengths, A, 0)
    assert d[2] == -1 and d[3] == -1


def test_receptive_field_grows_with_depth():
    A = path(11)
    sizes = [attempt(drills.receptive_field, A, 5, k).sum() for k in (1, 2, 3)]
    assert sizes == [3, 5, 7]


def test_influence_decays_along_a_path():
    """The over-squashing measurement: a source's share of a distant node's
    representation falls by orders of magnitude, so what arrives is not lost
    but diluted below what a fixed-width vector resolves."""
    A = path(9)
    share = attempt(drills.influence_share, A, 0, 8)
    assert share[1] > share[4] > share[8]
    assert share[8] < share[1] / 100


def test_influence_shares_are_a_distribution():
    A = random_graph(12, 0.25, 17)
    tot = sum(attempt(drills.influence_share, A, s, 3) for s in range(12))
    assert np.allclose(tot, 1.0)


# --------------------------------------------------------------------- tasks

def test_homophily_is_one_when_neighbours_agree():
    A = path(8)
    assert attempt(drills.homophily, A, np.zeros(8, int)) == pytest.approx(1.0)


def test_homophily_is_zero_on_a_bipartite_labelling():
    A = ring(8)
    y = np.arange(8) % 2
    assert attempt(drills.homophily, A, y) == pytest.approx(0.0)


def _homophilous_blocks(seed=18, n=40, p_in=0.20, p_out=0.02):
    r = np.random.default_rng(seed)
    y = np.repeat([0, 1], n)
    A = np.zeros((2 * n, 2 * n), int)
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            if r.random() < (p_in if y[i] == y[j] else p_out):
                A[i, j] = A[j, i] = 1
    return A, y, r


def test_label_propagation_beats_chance_on_a_homophilous_graph():
    """No learning at all, and on a homophilous graph it is hard to beat. This
    is the baseline missing from most published comparisons."""
    A, y, r = _homophilous_blocks()
    labelled = np.zeros(80, bool)
    for c in (0, 1):                          # six seeds per class
        labelled[r.choice(np.nonzero(y == c)[0], 6, replace=False)] = True
    pred = attempt(drills.label_propagation, A, y, labelled, 50)
    assert (pred[~labelled] == y[~labelled]).mean() > 0.85


def test_label_propagation_inherits_the_imbalance_of_its_seeds():
    """The failure mode nobody mentions, averaged over twelve graphs so it is a
    measurement rather than an anecdote. Same graphs, same algorithm, same
    number of seeds — six per class, or three and nine. The harmonic solution
    has no notion of class prior, so the majority seed class spreads further,
    and accuracy falls from 0.99 to about 0.81. A propagation baseline needs
    its seed counts reported alongside it."""
    bal, imb = [], []
    for s in range(12):
        A, y, _ = _homophilous_blocks(seed=s)
        r = np.random.default_rng(1000 + s)
        lb = np.zeros(80, bool)
        for c in (0, 1):
            lb[r.choice(np.nonzero(y == c)[0], 6, replace=False)] = True
        li = np.zeros(80, bool)
        li[r.choice(np.nonzero(y == 0)[0], 3, replace=False)] = True
        li[r.choice(np.nonzero(y == 1)[0], 9, replace=False)] = True
        pb = attempt(drills.label_propagation, A, y, lb, 50)
        pi = attempt(drills.label_propagation, A, y, li, 50)
        bal.append((pb[~lb] == y[~lb]).mean())
        imb.append((pi[~li] == y[~li]).mean())
    gap = np.array(bal) - np.array(imb)
    assert np.mean(bal) > 0.95
    assert gap.mean() > 2 * gap.std(ddof=1) / np.sqrt(len(gap))   # separated


def test_hard_negatives_are_at_distance_two():
    A = random_graph(40, 0.08, 19)
    neg = attempt(drills.sample_negatives, A, 20, 0, "hard")
    for i, j in neg:
        assert A[i, j] == 0
        assert attempt(drills.shortest_path_lengths, A, int(i))[j] == 2


def test_uniform_negatives_are_mostly_far_away():
    """Which is why they are easy: a model that knows only which component a
    node is in scores well against them."""
    A = random_graph(60, 0.04, 20)
    neg = attempt(drills.sample_negatives, A, 60, 1, "uniform")
    far = [attempt(drills.shortest_path_lengths, A, int(i))[j] for i, j in neg]
    assert np.mean([d != 2 for d in far]) > 0.7


def test_removing_the_test_edge_changes_the_graph_the_model_reads():
    A = ring(6)
    B = attempt(drills.remove_edges, A, [(0, 1)])
    assert B[0, 1] == 0 and B[1, 0] == 0
    assert A[0, 1] == 1                      # the original is not mutated
    assert attempt(drills.shortest_path_lengths, B, 0)[1] == 5


def test_filtering_stops_penalising_a_model_for_being_right():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert attempt(drills.filtered_rank, scores, 2, ()) == 3
    assert attempt(drills.filtered_rank, scores, 2, (0, 1)) == 1


# ------------------------------------------------------------------- geometry

def test_distance_features_are_invariant_to_rotation_and_ordering():
    r = np.random.default_rng(21)
    P = r.standard_normal((9, 3))
    R = attempt(drills.random_rotation, 3)
    t = r.standard_normal(3)
    f0 = attempt(drills.pairwise_distance_features, P)
    f1 = attempt(drills.pairwise_distance_features, P @ R.T + t)
    f2 = attempt(drills.pairwise_distance_features, P[r.permutation(9)])
    assert np.allclose(f0, f1, atol=1e-10)
    assert np.allclose(f0, f2, atol=1e-10)


def test_distance_features_cannot_see_chirality():
    """E(3) includes reflection, so a distance-only model is built to give a
    molecule and its mirror image the same answer. Declaring too large a group
    makes the distinction unrepresentable."""
    P = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mirror = P * np.array([1., 1, -1])
    assert np.allclose(attempt(drills.pairwise_distance_features, P),
                       attempt(drills.pairwise_distance_features, mirror))


def test_random_rotation_is_a_rotation():
    R = attempt(drills.random_rotation, 5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_the_vector_update_rotates_with_its_input():
    r = np.random.default_rng(22)
    P = r.standard_normal((10, 3))
    V = np.zeros((10, 3))
    A = random_graph(10, 0.3, 23)
    E = attempt(drills.edge_list, A)
    w = r.standard_normal(len(E))            # in practice a function of distances
    R = attempt(drills.random_rotation, 7)
    V0 = attempt(drills.equivariant_update, P, V, E, w)
    V1 = attempt(drills.equivariant_update, P @ R.T, V, E, w)
    assert np.allclose(V0 @ R.T, V1, atol=1e-10)


def test_the_vector_update_ignores_translation():
    r = np.random.default_rng(24)
    P = r.standard_normal((8, 3))
    A = random_graph(8, 0.4, 25)
    E = attempt(drills.edge_list, A)
    w = r.standard_normal(len(E))
    V0 = attempt(drills.equivariant_update, P, np.zeros((8, 3)), E, w)
    V1 = attempt(drills.equivariant_update, P + 17.0, np.zeros((8, 3)), E, w)
    assert np.allclose(V0, V1, atol=1e-10)
