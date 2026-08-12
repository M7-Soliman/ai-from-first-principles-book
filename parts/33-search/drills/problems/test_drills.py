"""
Tests for the Part XXXIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Every assertion here is exact. These are algorithms with provable properties —
alpha-beta returns precisely the minimax value, a valid bound never excludes the
optimum, and the Knuth-Moore count is a closed form rather than an estimate — so
nothing in this file depends on the machine it runs on.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ---------------------------------------------------------------- heuristics

def test_manhattan_is_the_l1_distance():
    assert attempt(drills.manhattan, (0, 0), (3, 4)) == 7
    assert attempt(drills.manhattan, (2, 5), (2, 5)) == 0


def test_misplaced_tiles_ignores_the_blank():
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert attempt(drills.misplaced_tiles, goal, goal) == 0
    # blank moved into position 7 and the 8 displaced: one *tile* is misplaced
    assert attempt(drills.misplaced_tiles, (1, 2, 3, 4, 5, 6, 7, 0, 8), goal) == 1


def test_admissibility_is_about_the_maximum_not_the_mean():
    h_star = np.array([10.0] * 100)
    good = np.full(100, 9.0)
    assert attempt(drills.is_admissible, good, h_star) is True
    bad = good.copy()
    bad[57] = 10.5                       # one overestimate out of a hundred
    assert attempt(drills.is_admissible, bad, h_star) is False
    assert abs(bad.mean() - good.mean()) < 0.02      # and the mean barely moved


def test_consistency_detects_a_triangle_violation():
    edges = [("a", "b", 1.0), ("b", "goal", 1.0)]
    assert attempt(drills.is_consistent, edges, {"a": 2.0, "b": 1.0, "goal": 0.0}) is True
    assert attempt(drills.is_consistent, edges, {"a": 5.0, "b": 1.0, "goal": 0.0}) is False


def test_a_zeroed_heuristic_is_admissible_but_inconsistent():
    # h* on one node, zero on its neighbour: never an overestimate, and the
    # drop across a unit edge is far larger than the edge cost.
    edges = [("a", "b", 1.0), ("b", "goal", 1.0)]
    h = {"a": 2.0, "b": 0.0, "goal": 0.0}
    h_star = [2.0, 1.0, 0.0]
    assert attempt(drills.is_admissible, [h["a"], h["b"], h["goal"]], h_star) is True
    assert attempt(drills.is_consistent, edges, h) is False


def test_dominance():
    assert attempt(drills.dominates, [3, 4, 5], [1, 4, 2]) is True
    assert attempt(drills.dominates, [3, 4, 5], [1, 9, 2]) is False


def test_max_of_admissible_is_admissible_and_dominates_both():
    h_star = np.array([9.0, 9.0, 9.0])
    h1 = np.array([1.0, 8.0, 3.0])
    h2 = np.array([7.0, 2.0, 3.0])
    m = attempt(drills.max_heuristic, h1, h2)
    assert np.allclose(m, [7.0, 8.0, 3.0])
    assert attempt(drills.is_admissible, m, h_star) is True
    assert attempt(drills.dominates, m, h1) is True
    assert attempt(drills.dominates, m, h2) is True


def test_the_weighted_bound():
    assert attempt(drills.weighted_bound, 2.0, 50.0) == pytest.approx(100.0)
    assert attempt(drills.weighted_bound, 1.0, 50.0) == pytest.approx(50.0)


# ------------------------------------------------------------------------ A*

def _grid(blocked=(), n=6):
    """An n x n 4-connected unit-cost grid with some cells removed."""
    blocked = set(blocked)

    def neighbours(s):
        r, c = s
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = (r + dr, c + dc)
            if 0 <= t[0] < n and 0 <= t[1] < n and t not in blocked:
                yield t, 1.0
    return neighbours


def test_astar_finds_the_optimal_cost_on_an_open_grid():
    cost, path, _ = attempt(drills.astar, _grid(), (0, 0), (5, 5), lambda s: 0.0)
    assert cost == pytest.approx(10.0)
    assert path[0] == (0, 0) and path[-1] == (5, 5)
    assert len(path) == 11


def test_astar_path_steps_are_all_legal():
    _, path, _ = attempt(drills.astar, _grid(), (0, 0), (5, 5),
                         lambda s: drills.manhattan(s, (5, 5)))
    for u, v in zip(path, path[1:]):
        assert abs(u[0] - v[0]) + abs(u[1] - v[1]) == 1


def test_every_admissible_heuristic_gives_the_same_cost():
    blocked = [(1, 1), (1, 2), (1, 3), (3, 2), (3, 3), (3, 4)]
    g = _grid(blocked)
    costs = []
    for alpha in (0.0, 0.5, 1.0):
        c, _, _ = attempt(drills.astar, g, (0, 0), (5, 5),
                          lambda s, a=alpha: a * drills.manhattan(s, (5, 5)))
        costs.append(c)
    assert len(set(costs)) == 1


def test_a_better_heuristic_expands_fewer_nodes():
    """The goal is deliberately not the opposite corner. On a full open
    rectangle every cell lies on some monotone shortest path, so Manhattan
    distance prunes nothing at all — which is worth seeing once."""
    g, goal = _grid(n=14), (0, 13)
    _, _, e0 = attempt(drills.astar, g, (0, 0), goal, lambda s: 0.0)
    _, _, e1 = attempt(drills.astar, g, (0, 0), goal,
                       lambda s: drills.manhattan(s, goal))
    assert e1 * 4 < e0


def test_astar_reports_unreachable():
    walled = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
    cost, path, _ = attempt(drills.astar, _grid(walled), (0, 0), (5, 5), lambda s: 0.0)
    assert np.isinf(cost) and path is None


def test_astar_stays_optimal_under_an_inconsistent_heuristic():
    """The heuristic knows the truth on a scattered subset and is zero
    elsewhere: admissible, wildly inconsistent, and A* must still be exact."""
    rng = np.random.default_rng(0)
    known = {(r, c) for r in range(6) for c in range(6) if rng.random() < 0.4}
    g = _grid()

    def h(s):
        return drills.manhattan(s, (5, 5)) if s in known else 0.0

    cost, _, _ = attempt(drills.astar, g, (0, 0), (5, 5), h)
    assert cost == pytest.approx(10.0)


# --------------------------------------------------------------- beam search

def _trellis(seed=0, L=5, V=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(L, V, V))


def test_a_full_width_beam_matches_exact_search():
    s = _trellis()
    exact, eseq = attempt(drills.viterbi, s)
    beam, bseq = attempt(drills.beam_search, s, 4 ** 5)
    assert beam == pytest.approx(exact)
    assert bseq == eseq


def test_a_narrow_beam_never_beats_exact_search_on_the_model():
    for seed in range(5):
        s = _trellis(seed)
        exact, _ = attempt(drills.viterbi, s)
        for width in (1, 2, 3):
            beam, _ = attempt(drills.beam_search, s, width)
            assert beam <= exact + 1e-9


def test_beam_and_viterbi_share_the_start_convention():
    """Both must treat symbol 0 as the start state. If they disagree, every
    comparison drawn between them is measuring the convention."""
    s = np.full((3, 2, 2), -10.0)
    s[0][0, 1] = 0.0                # only starting at symbol 1 is cheap
    s[1][1, 1] = 0.0
    s[2][1, 1] = 0.0
    exact, eseq = attempt(drills.viterbi, s)
    assert eseq[0] == 1 and exact == pytest.approx(0.0)
    beam, bseq = attempt(drills.beam_search, s, 4)
    assert beam == pytest.approx(exact) and bseq == eseq


def test_widening_the_beam_never_lowers_the_model_score():
    s = _trellis(3)
    scores = [attempt(drills.beam_search, s, w)[0] for w in (1, 2, 4, 8)]
    assert all(b >= a - 1e-9 for a, b in zip(scores, scores[1:]))


# ---------------------------------------------------------- branch and bound

def test_the_fractional_bound_by_hand():
    # densities 6, 5, 4; capacity 5 takes items 0 and 1 whole, then 2/3 of item 2
    v, w = [6.0, 10.0, 12.0], [1.0, 2.0, 3.0]
    assert attempt(drills.fractional_bound, v, w, 5.0) == pytest.approx(6.0 + 10.0 + 8.0)
    assert attempt(drills.fractional_bound, v, w, 1.0) == pytest.approx(6.0)


def test_the_fractional_bound_is_an_upper_bound():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 10
        w = rng.integers(1, 20, n).astype(float)
        v = w * rng.uniform(0.8, 1.2, n)
        cap = 0.5 * w.sum()
        best, _ = attempt(drills.knapsack_bnb, v, w, cap, True)
        assert attempt(drills.fractional_bound, v, w, cap) >= best - 1e-9


def test_both_bounds_return_the_same_optimum():
    rng = np.random.default_rng(2)
    for _ in range(10):
        n = 12
        w = rng.integers(1, 20, n).astype(float)
        v = w * rng.uniform(0.8, 1.2, n)
        cap = 0.5 * w.sum()
        a, _ = attempt(drills.knapsack_bnb, v, w, cap, True)
        b, _ = attempt(drills.knapsack_bnb, v, w, cap, False)
        assert a == pytest.approx(b)


def test_bnb_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 12
    w = rng.integers(1, 20, n).astype(float)
    v = rng.uniform(1, 20, n)
    cap = 0.5 * w.sum()
    masks = np.arange(1 << n)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
    feasible = bits @ w <= cap
    brute = (bits @ v)[feasible].max()
    best, _ = attempt(drills.knapsack_bnb, v, w, cap, True)
    assert best == pytest.approx(brute)


def test_the_tight_bound_explores_far_fewer_nodes():
    rng = np.random.default_rng(4)
    tight_total = loose_total = 0
    for _ in range(6):
        n = 16
        w = rng.integers(10, 100, n).astype(float)
        v = w * rng.uniform(0.8, 1.2, n)
        cap = 0.5 * w.sum()
        tight_total += attempt(drills.knapsack_bnb, v, w, cap, True)[1]
        loose_total += attempt(drills.knapsack_bnb, v, w, cap, False)[1]
    assert loose_total > 3 * tight_total


# ---------------------------------------------------------------- game trees

def test_alphabeta_returns_exactly_the_minimax_value():
    rng = np.random.default_rng(5)
    for _ in range(50):
        leaves = rng.normal(size=3 ** 4)
        mv, _ = attempt(drills.minimax, leaves, 3)
        av, _ = attempt(drills.alphabeta, leaves, 3)
        assert av == pytest.approx(mv)


def test_minimax_evaluates_the_whole_tree():
    leaves = np.arange(2 ** 5, dtype=float)
    _, n = attempt(drills.minimax, leaves, 2)
    assert n == 32


def test_alphabeta_never_evaluates_more_than_minimax():
    rng = np.random.default_rng(6)
    for _ in range(30):
        leaves = rng.normal(size=2 ** 6)
        _, mn = attempt(drills.minimax, leaves, 2)
        _, an = attempt(drills.alphabeta, leaves, 2)
        assert an <= mn


def test_alphabeta_prunes_a_lot_on_average():
    rng = np.random.default_rng(7)
    leaves = [attempt(drills.alphabeta, rng.normal(size=3 ** 6), 3)[1] for _ in range(20)]
    assert np.mean(leaves) < 0.6 * 3 ** 6


def test_the_knuth_moore_count():
    assert attempt(drills.knuth_moore, 3, 8) == 161          # 81 + 81 - 1
    assert attempt(drills.knuth_moore, 3, 7) == 27 + 81 - 1
    assert attempt(drills.knuth_moore, 2, 2) == 3


def test_knuth_moore_is_about_twice_the_square_root():
    for b, d in ((3, 8), (4, 8), (5, 6)):
        ratio = attempt(drills.knuth_moore, b, d) / np.sqrt(b ** d)
        assert 1.8 < ratio < 2.1


# ------------------------------------------------------------------- bandits

def test_ucb_pulls_an_untried_arm_first():
    s = attempt(drills.ucb_scores, [5.0, 0.0], [10.0, 0.0], 1.0)
    assert np.isinf(s[1]) and int(np.argmax(s)) == 1


def test_the_ucb_bonus_shrinks_with_visits():
    a = attempt(drills.ucb_scores, [5.0, 50.0], [10.0, 100.0], 1.0)
    means = np.array([0.5, 0.5])
    assert a[0] - means[0] > a[1] - means[1]


def test_ucb_at_zero_exploration_is_greedy():
    s = attempt(drills.ucb_scores, [6.0, 5.0], [10.0, 10.0], 0.0)
    assert int(np.argmax(s)) == 0
    assert s[0] == pytest.approx(0.6)


def test_expected_restarts_and_iterative_deepening_overhead():
    assert attempt(drills.expected_restarts, 0.05) == pytest.approx(20.0)
    assert attempt(drills.id_overhead, 3) == pytest.approx(1.5)
    assert attempt(drills.id_overhead, 10) == pytest.approx(10.0 / 9.0)


# ----------------------------------------------------- sampling and verifiers

def test_pass_at_k():
    assert attempt(drills.pass_at_k, 0.3, 1) == pytest.approx(0.3)
    assert attempt(drills.pass_at_k, 0.3, 8) == pytest.approx(1 - 0.7 ** 8)
    assert attempt(drills.pass_at_k, 0.0, 100) == pytest.approx(0.0)


def test_a_perfect_verifier_reaches_the_ceiling():
    p = 0.3
    for k in (1, 4, 16):
        got = attempt(drills.selection_solve_rate, p, k, 1.0, 0.0)
        assert got == pytest.approx(attempt(drills.pass_at_k, p, k))


def test_an_uninformative_verifier_leaves_you_where_you_started():
    p = 0.3
    for k in (1, 4, 16):
        got = attempt(drills.selection_solve_rate, p, k, 0.5, 0.5)
        assert got == pytest.approx(p)


def test_one_candidate_is_always_the_base_rate():
    """Whatever the verifier does, k=1 cannot change the answer — there is
    nothing to select between. A formula that violates this is conditioning
    wrongly somewhere."""
    for tpr, fpr in ((1.0, 0.0), (0.3, 0.8), (0.6, 0.4), (0.0, 0.0)):
        assert attempt(drills.selection_solve_rate, 0.3, 1, tpr, fpr) == pytest.approx(0.3)


def test_selection_helps_exactly_when_tpr_exceeds_fpr():
    p, k = 0.3, 8
    assert attempt(drills.selection_solve_rate, p, k, 0.9, 0.2) > p
    assert attempt(drills.selection_solve_rate, p, k, 0.2, 0.9) < p


def test_a_bad_verifier_gets_worse_with_more_candidates():
    p = 0.3
    rates = [attempt(drills.selection_solve_rate, p, k, 0.3, 0.8) for k in (1, 2, 4, 8, 16)]
    assert all(b <= a + 1e-12 for a, b in zip(rates, rates[1:]))
    assert rates[-1] < rates[0]


def test_majority_vote_breaks_ties_toward_the_earliest():
    assert attempt(drills.majority_vote, ["a", "b", "a", "c"]) == "a"
    assert attempt(drills.majority_vote, ["b", "a", "a", "b"]) == "b"


def test_majority_vote_amplifies_a_shared_misconception():
    """Independent errors scatter; a shared one wins. Voting removes variance
    and never bias, which is why self-consistency fails on systematic errors."""
    scattered = ["right", "wrong1", "wrong2", "wrong3", "right"]
    assert attempt(drills.majority_vote, scattered) == "right"
    shared = ["right", "same_wrong", "same_wrong", "same_wrong", "right"]
    assert attempt(drills.majority_vote, shared) == "same_wrong"
