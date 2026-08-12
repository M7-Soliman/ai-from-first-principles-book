"""
Tests for the Part XIX drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Two tests assert that a technique does NOT deliver what its FLOP count
promises. Those are the part's results, not oversights.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


N = 7e9
PEAK = 1.0e15
BW = 2.0e12


# -------------------------------------------------------------- the roofline

def test_one_token_is_the_worst_possible_shape():
    """Section 3: the intensity of decoding is 1 FLOP per byte."""
    assert attempt(drills.arithmetic_intensity, N, 1) == pytest.approx(1.0)
    assert attempt(drills.arithmetic_intensity, N, 64) == pytest.approx(64.0)
    assert attempt(drills.arithmetic_intensity, N, 1, bytes_per_weight=1) == \
        pytest.approx(2.0), "fewer bytes per weight raises the intensity"


def test_decode_is_memory_bound_and_prefill_is_not():
    ridge = attempt(drills.ridge_point, PEAK, BW)
    assert ridge == pytest.approx(500.0)
    assert attempt(drills.is_memory_bound, N, 1, PEAK, BW)
    assert attempt(drills.is_memory_bound, N, 64, PEAK, BW)
    assert not attempt(drills.is_memory_bound, N, 4096, PEAK, BW)
    frac = attempt(drills.achievable_flops, 1.0, PEAK, BW) / PEAK
    assert frac == pytest.approx(0.002), f"one token reaches {frac*100:.2f}% of peak"


def test_batching_is_the_whole_game():
    s1 = attempt(drills.batching_speedup, N, 1, BW, PEAK)
    s64 = attempt(drills.batching_speedup, N, 64, BW, PEAK)
    s512 = attempt(drills.batching_speedup, N, 512, BW, PEAK)
    s4096 = attempt(drills.batching_speedup, N, 4096, BW, PEAK)
    assert s1 == pytest.approx(1.0)
    assert s64 == pytest.approx(64.0, rel=1e-6), "linear while memory-bound"
    assert s512 == pytest.approx(500.0, rel=1e-6), "capped at the ridge"
    assert s4096 == pytest.approx(s512), "and flat after it — arithmetic is now the limit"


def test_quantizing_weights_speeds_up_decode_proportionally():
    """Because decode is memory-bound, halving the bytes halves the time."""
    t16 = attempt(drills.decode_time_per_token, N, 2, BW)
    t8 = attempt(drills.decode_time_per_token, N, 1, BW)
    t4 = attempt(drills.decode_time_per_token, N, 0.5, BW)
    assert t16 / t8 == pytest.approx(2.0)
    assert t16 / t4 == pytest.approx(4.0)
    assert attempt(drills.decode_time_per_token, N, 2, BW, batch=32) == \
        pytest.approx(t16 / 32)


# --------------------------------------------------------------- the KV cache

def test_the_cache_overtakes_the_weights():
    L, H, D = 32, 32, 128
    t = attempt(drills.context_where_cache_equals_weights, N, L, H, D)
    assert 20000 < t < 35000, f"crossover at {t:.0f} tokens"
    small = attempt(drills.context_where_cache_equals_weights, N, L, 1, D)
    assert small == pytest.approx(t * 32, rel=1e-9), \
        "multi-query pushes the crossover out by the head count"


def test_the_cache_decides_concurrency():
    L, H, D = 32, 32, 128
    GB = 1024 ** 3
    mh = attempt(drills.max_concurrent_sequences, 80 * GB, N, 65536, L, H, D)
    mq = attempt(drills.max_concurrent_sequences, 80 * GB, N, 65536, L, 1, D)
    assert mh < 5, f"multi-head fits {mh} sequences at 64k context"
    assert mq > 30 * mh, f"multi-query fits {mq}"
    assert attempt(drills.max_concurrent_sequences, 8 * GB, N, 4096, L, H, D) == 0, \
        "the weights alone do not fit"


def test_cache_grows_linearly_in_both_context_and_batch():
    a = attempt(drills.kv_cache_bytes, 1024, 1, 32, 32, 128)
    assert attempt(drills.kv_cache_bytes, 2048, 1, 32, 32, 128) == pytest.approx(2 * a)
    assert attempt(drills.kv_cache_bytes, 1024, 4, 32, 32, 128) == pytest.approx(4 * a)


# ------------------------------------------------------------- quantization

def test_quantization_error_falls_with_bits_and_granularity():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (256, 256))
    prev = None
    for b in (4, 6, 8):
        e = attempt(drills.quantization_error, W,
                    attempt(drills.quantize_symmetric, W, b))
        if prev is not None:
            assert e < prev
        prev = e
    e_tensor = attempt(drills.quantization_error, W,
                       attempt(drills.quantize_symmetric, W, 8))
    e_chan = attempt(drills.quantization_error, W,
                     attempt(drills.quantize_symmetric, W, 8, axis=0))
    e_group = attempt(drills.quantization_error, W,
                      attempt(drills.quantize_symmetric, W, 8, group=64))
    assert e_group < e_chan < e_tensor


def test_outliers_dominate_the_error():
    """Section 10: a handful of columns widen the range for everyone."""
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.02, (256, 256))
    cols = rng.choice(256, 5, replace=False)
    W[:, cols] *= 100.0
    found = attempt(drills.outlier_channels, W, factor=5.0)
    assert set(found.tolist()) == set(cols.tolist())
    # measure on the weights that were actually quantized, which is where the
    # inflated range does its damage
    rest = np.setdiff1d(np.arange(256), found)
    plain_q = attempt(drills.quantize_symmetric, W, 8, axis=0)
    plain = attempt(drills.quantization_error, W[:, rest], plain_q[:, rest])
    mixed = attempt(drills.mixed_precision_error, W, 8, found)
    assert plain > 0.05, f"with outliers in the range, the rest lose precision: {plain:.4f}"
    assert mixed < plain / 5, f"plain {plain:.4f}, scales computed without them {mixed:.4f}"


def test_group_scales_are_real_storage():
    n = 1e9
    plain = attempt(drills.quantized_bytes, n, 4)
    grouped = attempt(drills.quantized_bytes, n, 4, group=128)
    assert grouped > plain
    assert grouped / plain == pytest.approx(1.03125, rel=1e-6), \
        "a 16-bit scale per 128 weights is 0.125 bits each, about 3% on top of 4"


# ---------------------------------------------------------------- sparsity

def test_2of4_keeps_exactly_half():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(64, 128))
    P = attempt(drills.prune_2of4, W)
    assert attempt(drills.sparsity_of, P) == pytest.approx(0.5)
    g = P.reshape(-1, 4)
    assert ((g != 0).sum(1) == 2).all(), "exactly two survive in every group of four"
    kept = np.abs(W.reshape(-1, 4))
    assert all(set(np.argsort(-kept[i])[:2]) == set(np.flatnonzero(g[i] != 0))
               for i in range(0, len(g), 97)), "and they are the two largest"


def test_unstructured_sparsity_can_cost_more_bytes_than_it_saves():
    """Section 19: removed arithmetic is not removed bytes."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=(512, 512))
    dense = attempt(drills.dense_storage_bytes, W)
    for s, expect_bigger in ((0.5, True), (0.6, True), (0.9, False)):
        P = attempt(drills.magnitude_prune, W, s)
        sp = attempt(drills.sparse_storage_bytes, P)
        if expect_bigger:
            assert sp > dense, \
                f"at {s*100:.0f}% sparsity the sparse format is {sp/dense:.2f}x dense"
        else:
            assert sp < dense
    half = attempt(drills.sparse_storage_bytes,
                   attempt(drills.magnitude_prune, W, 0.5))
    assert half / dense > 1.4, "half the weights removed, and it got substantially larger"


def test_magnitude_pruning_removes_the_smallest():
    W = np.array([[0.1, -3.0, 0.05, 2.0]])
    P = attempt(drills.magnitude_prune, W, 0.5)
    assert attempt(drills.sparsity_of, P) == pytest.approx(0.5)
    assert P[0, 1] == -3.0 and P[0, 3] == 2.0
    assert P[0, 0] == 0.0 and P[0, 2] == 0.0


# -------------------------------------------------------------- distillation

def test_temperature_reveals_the_dark_knowledge():
    """Section 26: a confident teacher's extra information is invisible at T=1."""
    logits = np.array([[10.0, 2.0, 1.5, -3.0]])
    h1 = attempt(drills.dark_knowledge, logits, 1.0)
    h4 = attempt(drills.dark_knowledge, logits, 4.0)
    assert h1 < 0.02, f"at T=1 the distribution is nearly one-hot: entropy {h1:.4f}"
    assert h4 > 5 * h1
    p4 = attempt(drills.softmax_T, logits, 4.0)
    assert p4[0, 1] > p4[0, 2] > p4[0, 3], "and it preserves the ordering of the wrong answers"


def test_distillation_loss_is_minimised_by_matching_the_teacher():
    rng = np.random.default_rng(4)
    teacher = rng.normal(size=(32, 6)) * 3
    labels = teacher.argmax(1)
    match = attempt(drills.distillation_loss, teacher.copy(), teacher, labels)
    off = attempt(drills.distillation_loss, rng.normal(size=(32, 6)), teacher, labels)
    assert match < off
    pure_hard = attempt(drills.distillation_loss, teacher.copy(), teacher, labels,
                        alpha=0.0)
    assert pure_hard > 0


# ------------------------------------------------------------------ adapters

def test_lora_is_a_tiny_fraction_and_merges_exactly():
    p, frac = attempt(drills.lora_params, 4096, 4096, 8)
    assert p == 8 * 8192
    assert frac < 0.005, f"rank 8 is {frac*100:.3f}% of the matrix"
    rng = np.random.default_rng(5)
    W0 = rng.normal(size=(64, 48)); A = rng.normal(size=(4, 48)); B = rng.normal(size=(64, 4))
    x = rng.normal(size=(7, 48))
    sep = attempt(drills.lora_forward, x, W0, A, B, scale=0.5)
    merged = x @ attempt(drills.merge_lora, W0, A, B, scale=0.5).T
    assert np.abs(sep - merged).max() < 1e-9, \
        "merged and unmerged must be the same function"


def test_lora_starts_as_the_identity():
    """B is initialised at zero, so the adapted model begins exactly at W0."""
    rng = np.random.default_rng(6)
    W0 = rng.normal(size=(32, 32)); A = rng.normal(size=(4, 32)); B = np.zeros((32, 4))
    assert np.allclose(attempt(drills.merge_lora, W0, A, B), W0)


def test_energy_rank_is_not_effect_rank():
    """Section 33: the 90% figure and the 99% figure are very different."""
    rng = np.random.default_rng(7)
    U = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    Vt = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    s = np.exp(-np.arange(64) / 6.0)
    D = U @ np.diag(s) @ Vt
    r90 = attempt(drills.rank_for_energy, D, 0.9)
    r99 = attempt(drills.rank_for_energy, D, 0.99)
    assert r90 < r99 < 64
    assert r99 >= 2 * r90, f"90% needs {r90}, 99% needs {r99}"


# -------------------------------------------------------------------- serving

def test_speculative_speedup_needs_a_draft_that_is_small_and_close():
    slow_close = attempt(drills.speculative_speedup, 0.9, 4, 0.5)
    fast_close = attempt(drills.speculative_speedup, 0.9, 4, 0.02)
    fast_poor = attempt(drills.speculative_speedup, 0.3, 4, 0.02)
    assert fast_close > slow_close, "a cheaper draft is better at equal acceptance"
    assert fast_close > fast_poor, "and a closer draft is better at equal cost"
    assert attempt(drills.speculative_speedup, 0.0, 4, 0.02) < 1.05, \
        "a draft that is never accepted buys nothing"


def test_paging_recovers_most_of_the_wasted_cache():
    rng = np.random.default_rng(8)
    lens = rng.integers(20, 300, 200)
    naive, paged = attempt(drills.paged_waste, lens, 16, 4096)
    assert naive > 0.9, f"reserving the maximum wastes {naive*100:.1f}%"
    assert paged < 0.1, f"paging wastes {paged*100:.1f}%"


def test_output_tokens_cost_far_more_than_input_tokens():
    """Section 44: the structure of the cost model, and of the pricing."""
    per_in = attempt(drills.request_cost, N, 1000, 0, 32, BW, PEAK) / 1000
    per_out = attempt(drills.request_cost, N, 0, 1000, 32, BW, PEAK) / 1000
    assert per_out > 5 * per_in, f"input {per_in:.3e}s, output {per_out:.3e}s"
    big = attempt(drills.request_cost, N, 0, 1000, 256, BW, PEAK) / 1000
    assert big == pytest.approx(per_out / 8), "and decode is shared across the batch"
