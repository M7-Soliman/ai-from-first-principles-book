"""
Tests for the Part XXXII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

These assert arithmetic rather than timing. The prose's timing drills are
properties of your machine, and a test that asserted one would fail on somebody
else's — which is the same reason Figures 1 to 3 say so in their captions.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------- roofline

def test_the_roofline_has_two_regimes():
    bw, fl = 1e12, 1e14
    assert attempt(drills.roofline, 1.0, bw, fl) == pytest.approx(1e12)      # memory
    assert attempt(drills.roofline, 1e6, bw, fl) == pytest.approx(1e14)      # compute


def test_the_ridge_is_where_they_meet():
    bw, fl = 2e12, 3e14
    r = attempt(drills.ridge_point, bw, fl)
    assert r == pytest.approx(150.0)
    assert attempt(drills.roofline, r, bw, fl) == pytest.approx(fl)


def test_which_side_a_kernel_is_on():
    bw, fl = 2e12, 3e14
    assert attempt(drills.is_memory_bound, 1.0, bw, fl)          # vector add
    assert not attempt(drills.is_memory_bound, 500.0, bw, fl)    # large matmul


def test_a_vector_add_is_hopelessly_memory_bound():
    ai = attempt(drills.arithmetic_intensity, 1.0, 24.0)         # 1 flop, 3 float64
    assert ai < 0.05
    assert attempt(drills.is_memory_bound, ai, 2e12, 3e14)


def test_matmul_intensity_grows_with_size():
    small = attempt(drills.arithmetic_intensity, 2 * 128 ** 3, 3 * 128 ** 2 * 8)
    big = attempt(drills.arithmetic_intensity, 2 * 1024 ** 3, 3 * 1024 ** 2 * 8)
    assert big / small == pytest.approx(8.0)


def test_the_batch_size_decides_the_regime():
    """The same layer, opposite sides of the ridge. This is why generation is
    bandwidth-limited and training is not."""
    bw, fl = 2e12, 3e14
    one = attempt(drills.dense_layer_intensity, 4096, 4096, 1)
    many = attempt(drills.dense_layer_intensity, 4096, 4096, 512)
    assert attempt(drills.is_memory_bound, one, bw, fl)
    assert not attempt(drills.is_memory_bound, many, bw, fl)
    assert many / one > 100


# --------------------------------------------------------------------- fusion

def test_an_unfused_chain_moves_n_times_the_data():
    a = attempt(drills.unfused_bytes, 1000, 1)
    b = attempt(drills.unfused_bytes, 1000, 8)
    assert b / a == pytest.approx(8.0)


def test_a_fused_chain_moves_the_same_data_whatever_its_length():
    a = attempt(drills.fused_bytes, 1000, 1)
    b = attempt(drills.fused_bytes, 1000, 8)
    assert a == b


def test_the_speedup_is_bounded_by_the_chain_length():
    for k in (1, 4, 8):
        ratio = attempt(drills.unfused_bytes, 1000, k) / attempt(drills.fused_bytes, 1000, k)
        assert ratio == pytest.approx(attempt(drills.fusion_speedup_bound, k))


# ---------------------------------------------------------------------- cache

def test_blocking_computes_the_same_product():
    r = np.random.default_rng(0)
    A = r.standard_normal((64, 64)); B = r.standard_normal((64, 64))
    for bs in (8, 16, 32, 64):
        assert np.allclose(attempt(drills.blocked_matmul, A, B, bs), A @ B, atol=1e-10)


def test_a_bigger_block_holds_more_live_data():
    assert attempt(drills.working_set_bytes, 64) == 3 * 64 * 64 * 8
    assert attempt(drills.working_set_bytes, 128) / attempt(drills.working_set_bytes, 64) \
        == pytest.approx(4.0)


def test_the_block_size_can_be_computed_from_the_cache():
    """Rather than swept. A 256 KB cache and a 32 MB cache give different
    answers, and both are arithmetic."""
    small = attempt(drills.best_block_size, 256 * 1024)
    large = attempt(drills.best_block_size, 32 * 1024 * 1024)
    assert large > small
    assert attempt(drills.working_set_bytes, small) <= 256 * 1024
    assert attempt(drills.working_set_bytes, large) <= 32 * 1024 * 1024


# ---------------------------------------------------------------- collectives

def test_naive_allreduce_grows_with_the_worker_count():
    a = attempt(drills.naive_allreduce_volume, 8)
    b = attempt(drills.naive_allreduce_volume, 64)
    assert b / a == pytest.approx(63.0 / 7.0)


def test_ring_volume_is_bounded_by_two_buffers():
    """Whatever the worker count. This is why the ring is the algorithm
    everyone uses."""
    for n in (2, 8, 64, 1024):
        v = attempt(drills.ring_allreduce_volume, n)
        assert v < 2.0
        assert v == pytest.approx(2.0 * (n - 1) / n)


def test_the_ring_wins_on_volume_by_a_growing_factor():
    for n in (8, 64):
        naive = attempt(drills.naive_allreduce_volume, n)
        ring = attempt(drills.ring_allreduce_volume, n)
        assert naive / ring == pytest.approx((n - 1) / (2 * (n - 1) / n))


def test_but_loses_on_latency_for_a_small_message():
    """Which is why a real library switches algorithm by message size, and why a
    benchmark at one size says little about another."""
    bw, lat, n = 1e11, 5e-6, 64
    big = 1e9
    small = 1e3
    ring_big = attempt(drills.collective_time, n, big, bw, lat, "ring")
    naive_big = attempt(drills.collective_time, n, big, bw, lat, "naive")
    assert ring_big < naive_big
    ring_small = attempt(drills.collective_time, n, small, bw, lat, "ring")
    naive_small = attempt(drills.collective_time, n, small, bw, lat, "naive")
    assert ring_small > naive_small


def test_a_synchronous_step_runs_at_the_slowest_worker():
    times = np.full(100, 1.0); times[37] = 3.0
    assert attempt(drills.straggler_step_time, times) == pytest.approx(3.0)


def test_adam_costs_four_times_the_parameters_before_activations():
    p = 7e9
    m = attempt(drills.training_memory, p, 4, 2, 0.0)
    assert m == pytest.approx(4 * p * 4)


# -------------------------------------------------------------------- privacy

def test_clipping_bounds_every_gradient():
    r = np.random.default_rng(0)
    G = r.standard_normal((200, 10)) * 5.0
    Gc = attempt(drills.clip_gradients, G, 1.0)
    assert np.linalg.norm(Gc, axis=1).max() <= 1.0 + 1e-9


def test_clipping_leaves_small_gradients_alone():
    G = np.array([[0.1, 0.0], [0.2, 0.0]])
    assert np.allclose(attempt(drills.clip_gradients, G, 1.0), G)


def test_the_private_step_is_random_and_the_clipped_sum_is_not():
    """Clipping without noise bounds influence but does not hide it — both parts
    are the mechanism."""
    r = np.random.default_rng(1)
    G = r.standard_normal((50, 8))
    a = attempt(drills.privatise, G, 1.0, 0.5, seed=0)
    b = attempt(drills.privatise, G, 1.0, 0.5, seed=1)
    assert not np.allclose(a, b)
    c = attempt(drills.privatise, G, 1.0, 0.0, seed=0)
    d = attempt(drills.privatise, G, 1.0, 0.0, seed=1)
    assert np.allclose(c, d)


def test_budgets_add_across_steps():
    """Reporting a per-step epsilon without the total is how a guarantee gets
    overstated."""
    assert attempt(drills.compose_budget, 0.01, 1000) == pytest.approx(10.0)
    assert attempt(drills.compose_budget, 0.01, 1) == pytest.approx(0.01)


def test_a_person_with_many_records_is_protected_less():
    assert attempt(drills.group_privacy, 1.0, 1000) == pytest.approx(1000.0)


def test_a_perfect_attack_and_a_useless_one():
    scores = np.arange(100.0)
    labels = (np.arange(100) >= 50).astype(int)
    assert attempt(drills.auc, scores, labels) == pytest.approx(1.0)
    r = np.random.default_rng(0)
    assert attempt(drills.auc, r.standard_normal(2000),
                   (np.arange(2000) % 2)) == pytest.approx(0.5, abs=0.05)


# ------------------------------------------------------------------ adversary

def test_the_margin_is_the_minimum_perturbation():
    w = np.array([3.0, 4.0])                       # norm 5
    X = np.array([[5.0, 0.0], [1.0, 0.0]])
    m = attempt(drills.margin_to_boundary, X, w)
    assert m[0] == pytest.approx(3.0)
    assert m[1] == pytest.approx(0.6)


def test_attack_success_rises_with_the_budget():
    r = np.random.default_rng(0)
    w = r.standard_normal(20)
    X = r.standard_normal((2000, 20))
    rates = [attempt(drills.attack_success_rate, X, w, e) for e in (0.01, 0.1, 1.0)]
    assert rates[0] < rates[1] < rates[2]
    assert rates[0] < 0.05


def test_extraction_scales_with_parameters_not_the_training_set():
    """A victim fitted on a million examples is cloned with queries of order its
    parameter count."""
    a = attempt(drills.extraction_queries_needed, 30)
    b = attempt(drills.extraction_queries_needed, 3000)
    assert b / a == pytest.approx(100.0)
    assert a < 1000


def test_hiding_the_probabilities_raises_the_cost_without_preventing_it():
    probs = attempt(drills.extraction_queries_needed, 100, "probabilities")
    labels = attempt(drills.extraction_queries_needed, 100, "labels")
    assert labels > 5 * probs
    assert np.isfinite(labels)                     # a speed bump, not a wall
