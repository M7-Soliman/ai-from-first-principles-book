"""
Tests for the Part XII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Where this part claims exactness, the test demands it: the KV cache and the
online softmax must reproduce the naive computation, not approximate it, and
RoPE's relative property is checked at positions thousands apart.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


rng = np.random.default_rng(0)


def rand(*shape, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    return r.normal(size=shape)


# ------------------------------------------------------------- the operation ----

def test_softmax_is_stable_and_normalised():
    x = np.array([[-1000.0, 0.0, 1000.0], [1.0, 1.0, 1.0]])
    s = attempt(drills.softmax, x)
    assert np.isfinite(s).all()
    assert s.sum(-1) == pytest.approx(np.ones(2))
    assert s[1] == pytest.approx(np.full(3, 1 / 3))


def test_attention_shapes_and_convexity():
    Q, K, V = rand(5, 8, seed=1), rand(7, 8, seed=2), rand(7, 3, seed=3)
    out = attempt(drills.attention, Q, K, V)
    assert out.shape == (5, 3)
    # every output lies inside the convex hull of the values — section 26's point
    assert (out.max() <= V.max() + 1e-9) and (out.min() >= V.min() - 1e-9)


def test_causal_mask_blocks_the_future():
    n, d = 6, 8
    Q, K, V = rand(n, d, seed=4), rand(n, d, seed=5), rand(n, d, seed=6)
    m = attempt(drills.causal_mask, n)
    assert m.shape == (n, n) and not m[0, 0] and m[0, 1] and not m[3, 2]
    out = attempt(drills.attention, Q, K, V, m)
    # position 0 can only see itself, so its output IS v_0
    assert out[0] == pytest.approx(V[0], abs=1e-10)


def test_the_mask_is_verified_by_consequence():
    """Section 21's test: changing token j must not move outputs before j."""
    n, d = 10, 16
    X = rand(n, d, seed=7)
    Wq, Wk, Wv = (rand(d, d, seed=10 + i) / np.sqrt(d) for i in range(3))
    m = attempt(drills.causal_mask, n)

    def run(Xin):
        return drills.attention(Xin @ Wq, Xin @ Wk, Xin @ Wv, m)

    a = run(X)
    X2 = X.copy()
    X2[6] = rand(d, seed=99)
    b = run(X2)
    assert np.abs(a[:6] - b[:6]).max() == 0.0, "outputs before the change must be identical"
    assert np.abs(a[6:] - b[6:]).max() > 1e-6, "outputs from the change onward must move"


def test_entropy_collapses_without_the_scale():
    """Section 4, measured rather than asserted."""
    n = 64
    lo = attempt(drills.attention_entropy, rand(n, 8, seed=20), rand(n, 8, seed=21),
                 scale=False)
    hi = attempt(drills.attention_entropy, rand(n, 1024, seed=22), rand(n, 1024, seed=23),
                 scale=False)
    assert hi < lo, "unscaled entropy must fall as d grows"
    assert hi < 0.5, f"at d=1024 the unscaled softmax should be nearly one-hot, got {hi:.3f}"
    for d in (8, 1024):
        e = attempt(drills.attention_entropy, rand(n, d, seed=24), rand(n, d, seed=25),
                    scale=True)
        assert 3.0 < e < np.log(n), f"scaled entropy at d={d} was {e:.3f}"


# ------------------------------------------------------------------ heads ----

def test_split_and_merge_round_trip():
    X = rand(12, 32, seed=30)
    H = attempt(drills.split_heads, X, 4)
    assert H.shape == (4, 12, 8)
    assert attempt(drills.merge_heads, H) == pytest.approx(X)


def test_split_heads_keeps_features_contiguous():
    """The transpose-before-reshape bug produces the right shape and wrong data."""
    X = np.arange(2 * 6, dtype=float).reshape(2, 6)
    H = attempt(drills.split_heads, X, 3)
    assert H[0, 0] == pytest.approx([0.0, 1.0]), "head 0 gets the FIRST two features"
    assert H[2, 0] == pytest.approx([4.0, 5.0])


def test_multi_head_matches_single_head_when_h_is_one():
    d, n = 16, 7
    X = rand(n, d, seed=40)
    Wq, Wk, Wv = (rand(d, d, seed=41 + i) / np.sqrt(d) for i in range(3))
    Wo = np.eye(d)
    a = attempt(drills.multi_head_attention, X, Wq, Wk, Wv, Wo, 1)
    b = attempt(drills.attention, X @ Wq, X @ Wk, X @ Wv)
    assert a == pytest.approx(b, abs=1e-10)


# --------------------------------------------------------------- position ----

def test_self_attention_is_permutation_equivariant():
    d, n = 16, 9
    X = rand(n, d, seed=50)
    Wq, Wk, Wv = (rand(d, d, seed=51 + i) / np.sqrt(d) for i in range(3))

    def f(Z):
        return drills.attention(Z @ Wq, Z @ Wk, Z @ Wv)

    attempt(drills.attention, X @ Wq, X @ Wk, X @ Wv)
    for s in range(4):
        perm = np.random.default_rng(s).permutation(n)
        assert attempt(drills.is_permutation_equivariant, f, X, perm), \
            "self-attention must be exactly permutation equivariant"


def test_positional_encoding_breaks_the_symmetry():
    d, n = 16, 9
    X = rand(n, d, seed=60)
    pe = attempt(drills.sinusoidal_encoding, n, d)
    assert pe.shape == (n, d) and np.abs(pe).max() <= 1.0 + 1e-9
    Wq, Wk, Wv = (rand(d, d, seed=61 + i) / np.sqrt(d) for i in range(3))

    def f(Z):
        return drills.attention((Z + pe) @ Wq, (Z + pe) @ Wk, (Z + pe) @ Wv)

    perm = np.random.default_rng(1).permutation(n)
    assert not attempt(drills.is_permutation_equivariant, f, X, perm, 1e-8)


def test_rope_depends_only_on_the_offset():
    """Section 11's identity, tested at positions thousands apart."""
    d = 64
    q, k = rand(d, seed=70), rand(d, seed=71)
    base = attempt(drills.rope_score, q, k, 32, 30)
    for m0 in (130, 1030, 10030):
        s = attempt(drills.rope_score, q, k, m0 + 2, m0)
        assert s == pytest.approx(base, abs=1e-8), \
            f"offset 2 at position {m0} gave {s}, at position 30 gave {base}"


def test_rope_changes_with_the_offset():
    d = 64
    q, k = rand(d, seed=72), rand(d, seed=73)
    vals = [attempt(drills.rope_score, q, k, 30 + off, 30) for off in (1, 2, 5, 9)]
    assert len(set(np.round(vals, 6))) == len(vals), "different offsets must differ"


# ------------------------------------------------------------- the counts ----

def test_the_parameter_split_is_one_to_two():
    for d in (512, 768, 4096):
        a = attempt(drills.attention_params, d)
        f = attempt(drills.ffn_params, d)
        assert f == 2 * a
        assert a / (a + f) == pytest.approx(1 / 3)


def test_crossover_is_four_d():
    for d in (64, 256, 1024):
        n = attempt(drills.crossover_length, d)
        assert n == pytest.approx(4 * d)
        fl = attempt(drills.block_flops, int(n), d)
        assert fl["scores"] == pytest.approx(fl["ffn"], rel=1e-9)


def test_kv_cache_size_and_grouping():
    full = attempt(drills.kv_cache_bytes, 131072, 4096, 32)
    assert full / 1e9 == pytest.approx(68.7, abs=0.5)
    gq = attempt(drills.kv_cache_bytes, 131072, 4096, 32, n_kv_head=8, n_head=32)
    assert gq == pytest.approx(full / 4)


# ------------------------------------------------------------- generation ----

def test_the_kv_cache_is_exact():
    """Section 28: an exact optimisation, so demand equality."""
    d = 12
    x0 = rand(1, d, seed=80)
    Wq, Wk, Wv = (rand(d, d, seed=81 + i) / np.sqrt(d) for i in range(3))
    a = attempt(drills.generate_no_cache, x0, Wq, Wk, Wv, 15)
    b = attempt(drills.generate_with_cache, x0, Wq, Wk, Wv, 15)
    assert a.shape == b.shape == (16, d)
    assert np.abs(a - b).max() < 1e-12, "the cache must not change the result"


# ---------------------------------------------------------- online softmax ----

def test_online_softmax_matches_the_one_shot_version():
    """Section 29's whole mathematical content."""
    n, dv = 300, 5
    s = rand(n, seed=90) * 8
    V = rand(n, dv, seed=91)
    ref = attempt(drills.softmax, s) @ V
    for tile in (1, 7, 64, n):
        out = attempt(drills.online_softmax_weighted_sum, s, V, tile)
        assert out == pytest.approx(ref, abs=1e-10), f"tile={tile}"


def test_online_softmax_survives_large_scores():
    n, dv = 128, 3
    s = rand(n, seed=92) * 50 + 900.0
    V = rand(n, dv, seed=93)
    out = attempt(drills.online_softmax_weighted_sum, s, V, 16)
    assert np.isfinite(out).all()
    assert out == pytest.approx(attempt(drills.softmax, s) @ V, abs=1e-10)


# ------------------------------------------------------------------ shapes ----

def test_effective_rank_of_known_matrices():
    X = np.outer(rand(20, seed=94), rand(6, seed=95))       # rank one
    assert attempt(drills.effective_rank, X) == pytest.approx(1.0, abs=1e-6)
    Y = rand(40, 8, seed=96)
    assert attempt(drills.effective_rank, Y) > 5.0


def test_attention_collapses_without_skips():
    """Section 26, and the skip connection is what prevents it."""
    X = rand(64, 32, seed=97)
    deep_no = attempt(drills.attention_only_depth, X, 32, skip=False, seed=1)
    deep_yes = attempt(drills.attention_only_depth, X, 32, skip=True, seed=1)
    r_no = attempt(drills.effective_rank, deep_no)
    r_yes = attempt(drills.effective_rank, deep_yes)
    assert r_no < 1.5, f"attention alone should collapse to rank one, got {r_no:.2f}"
    assert r_yes > 5.0, f"skips should preserve rank, got {r_yes:.2f}"
