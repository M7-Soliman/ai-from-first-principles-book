"""
Tests for the Part XVII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Almost every assertion here is exact arithmetic rather than a tolerance,
because almost every claim in this part is arithmetic.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------- counting

def test_the_feedforward_holds_two_thirds_of_the_blocks():
    """Part XII's finding, as arithmetic: 8d^2 against 4d^2."""
    p = attempt(drills.transformer_params, 4096, 32, 50000, 4096)
    blocks = p["attention"] + p["ffn"]
    assert p["ffn"] / blocks == pytest.approx(2 / 3)
    assert p["attention"] / blocks == pytest.approx(1 / 3)


def test_flops_scale_as_six_n_per_token():
    assert attempt(drills.training_flops, 1e9, 1e12) == pytest.approx(6e21)
    assert attempt(drills.inference_flops, 1e9, 1e12) == pytest.approx(2e21)
    assert attempt(drills.training_flops, 1e9, 1) / \
        attempt(drills.inference_flops, 1e9, 1) == pytest.approx(3.0)


def test_serving_can_dominate_the_lifetime_cost():
    """Section 14: why models are deliberately overtrained."""
    N, D = 70e9, 1.4e12
    train_only = attempt(drills.lifetime_flops, N, D, 0)
    with_serving = attempt(drills.lifetime_flops, N, D, 1e15)
    assert with_serving > 4 * train_only


# --------------------------------------------------------------- scaling laws

def test_fit_recovers_planted_parameters():
    xs = np.logspace(3, 8, 12)
    ys = attempt(drills.power_law, xs, 40.0, 0.34, 1.85)
    A, alpha, E = attempt(drills.fit_power_law, xs, ys)
    assert alpha == pytest.approx(0.34, rel=1e-3)
    assert E == pytest.approx(1.85, rel=1e-3)
    assert A == pytest.approx(40.0, rel=1e-2)


def test_a_wrong_floor_distorts_the_exponent():
    """Section 2's pitfall: alpha and E trade off, so a fit with the floor
    pinned to the wrong value returns a different exponent."""
    xs = np.logspace(3, 7, 10)
    ys = attempt(drills.power_law, xs, 40.0, 0.34, 1.85)
    _, a_free, _ = attempt(drills.fit_power_law, xs, ys)
    _, a_wrong, _ = attempt(drills.fit_power_law, xs, ys, pin_E=1.60)
    assert a_free == pytest.approx(0.34, rel=1e-3)
    assert abs(a_wrong - 0.34) > 0.02, \
        f"pinning E wrongly should shift alpha, got {a_wrong:.4f}"


def test_equal_exponents_give_the_chinchilla_split():
    """With alpha == beta, N and D both scale as C^0.5."""
    A, B, al = 400.0, 400.0, 0.34
    Ns, Ds = [], []
    for C in (1e18, 1e20, 1e22):
        N, D = attempt(drills.compute_optimal_allocation, C, al, al, A, B)
        Ns.append(N); Ds.append(D)
        assert 6 * N * D == pytest.approx(C, rel=1e-6), "must respect the budget"
    # 10,000x the compute should raise each by 100x, since both go as C^0.5
    assert Ns[2] / Ns[0] == pytest.approx(1e4 ** 0.5, rel=1e-3)
    assert Ds[2] / Ds[0] == pytest.approx(1e4 ** 0.5, rel=1e-3)
    assert attempt(drills.tokens_per_parameter, Ns[1], Ds[1]) == pytest.approx(1.0, rel=1e-6)


def test_unequal_exponents_tilt_the_allocation():
    """Kaplan's answer in one line. N scales as C^(beta/(alpha+beta)), so a
    LARGER data exponent — data's deficit falling faster, so less of it needed
    — pushes the budget into parameters. Kaplan's N ~ C^0.73 implies beta well
    above alpha; Chinchilla found them equal."""
    C, A, B = 1e21, 400.0, 400.0
    N_eq, D_eq = attempt(drills.compute_optimal_allocation, C, 0.34, 0.34, A, B)
    N_tilt, D_tilt = attempt(drills.compute_optimal_allocation, C, 0.34, 0.60, A, B)
    assert N_tilt > N_eq and D_tilt < D_eq
    assert attempt(drills.tokens_per_parameter, N_tilt, D_tilt) < \
        attempt(drills.tokens_per_parameter, N_eq, D_eq)
    # and the other direction: a smaller data exponent demands far more data
    N_lo, D_lo = attempt(drills.compute_optimal_allocation, C, 0.34, 0.20, A, B)
    assert attempt(drills.tokens_per_parameter, N_lo, D_lo) > \
        attempt(drills.tokens_per_parameter, N_eq, D_eq)


# ------------------------------------------------------------------- memory

def test_adam_costs_six_times_the_bf16_model():
    m = attempt(drills.training_memory, 1e9)
    assert m["parameters"] == pytest.approx(2e9)
    assert m["optimizer_states"] == pytest.approx(12e9)
    assert m["optimizer_states"] / m["parameters"] == pytest.approx(6.0)
    assert sum(m.values()) == pytest.approx(16e9), "16 bytes per parameter"
    assert attempt(drills.training_memory, 1e9, optimizer="sgd")["optimizer_states"] == 0.0


def test_attention_memory_is_quadratic_in_context():
    kw = dict(batch=1, d=4096, n_layers=32, heads=32)
    a = attempt(drills.activation_memory, ctx=2048, **kw)
    b = attempt(drills.activation_memory, ctx=4096, **kw)
    assert b / a > 3.0, "doubling context more than doubles it"
    lin_a = attempt(drills.activation_memory, ctx=2048, materialise_attention=False, **kw)
    lin_b = attempt(drills.activation_memory, ctx=4096, materialise_attention=False, **kw)
    assert lin_b / lin_a == pytest.approx(2.0), "without the matrix it is linear"


def test_zero_stages_shrink_monotonically():
    N, W = 7e9, 64
    prev = None
    for stage in (0, 1, 2, 3):
        m = attempt(drills.zero_memory, N, W, stage)
        if prev is not None:
            assert m < prev
        prev = m
    full = attempt(drills.zero_memory, N, 1, 0)
    assert attempt(drills.zero_memory, N, W, 3) == pytest.approx(full / W, rel=1e-9)


def test_sharding_does_not_help_activations():
    """Section 18: the third category is what is left, and it does not shard."""
    N, W = 7e9, 64
    states = attempt(drills.zero_memory, N, W, 3)
    acts = attempt(drills.activation_memory, 1, 8192, 4096, 32, 32)
    assert acts > states, \
        f"activations {acts/2**30:.1f} GiB against sharded states {states/2**30:.1f} GiB"


# ------------------------------------------------------------ number formats

def test_float_formats_match_known_values():
    f32 = attempt(drills.float_format, 8, 23)
    assert f32["largest"] == pytest.approx(3.4028235e38, rel=1e-6)
    assert f32["smallest_normal"] == pytest.approx(1.1754944e-38, rel=1e-6)
    f16 = attempt(drills.float_format, 5, 10)
    assert f16["largest"] == pytest.approx(65504.0)
    assert f16["smallest_normal"] == pytest.approx(6.103515625e-05)


def test_bf16_trades_precision_for_range():
    """Both are 16 bits. Section 19's whole argument."""
    bf = attempt(drills.float_format, 8, 7)
    f16 = attempt(drills.float_format, 5, 10)
    assert bf["largest"] > 1e30 and f16["largest"] < 1e5
    assert bf["smallest_normal"] < f16["smallest_normal"] / 1e30
    assert bf["eps"] > f16["eps"], "bf16 is the less precise of the two"


def test_loss_scaling_recovers_small_gradients():
    r = np.random.default_rng(0)
    g = np.exp(r.normal(np.log(1e-7), 1.5, 20000))
    f16 = attempt(drills.float_format, 5, 10)
    bf16 = attempt(drills.float_format, 8, 7)
    assert attempt(drills.representable_fraction, g, f16) < 0.9
    assert attempt(drills.representable_fraction, g, bf16) > 0.999, \
        "bf16 needs no loss scaling — that is the point"
    s = attempt(drills.loss_scale_for, g, f16)
    assert s > 1
    assert attempt(drills.representable_fraction, g * s, f16) >= 0.999


# ------------------------------------------------------------- communication

def test_allreduce_does_not_depend_on_batch_size():
    """Section 22: the communication is fixed per step and the compute is not."""
    a = attempt(drills.allreduce_bytes, 7e9, 8)
    b = attempt(drills.allreduce_bytes, 7e9, 8)
    assert a == b
    assert a == pytest.approx(2 * 7e9 * 2 * 7 / 8)
    assert attempt(drills.allreduce_bytes, 7e9, 1024) > a, "more devices, more volume"


def test_efficiency_rises_with_work_per_device():
    kw = dict(N=7e9, device_flops=3e14, bandwidth_bytes_per_s=25e9 / 8)
    lo = attempt(drills.dp_efficiency, tokens_per_device=1024, **kw)
    hi = attempt(drills.dp_efficiency, tokens_per_device=65536, **kw)
    assert hi > lo
    assert lo < 0.10, f"a small per-device batch stalls: {lo:.3f}"
    fast = attempt(drills.dp_efficiency, tokens_per_device=1024, N=7e9,
                   device_flops=3e14, bandwidth_bytes_per_s=900e9 / 8)
    assert fast > lo, "a faster interconnect helps at fixed work"


def test_pipeline_bubble_is_exact():
    assert attempt(drills.pipeline_bubble, 8, 1) == pytest.approx(7 / 8)
    assert attempt(drills.pipeline_bubble, 8, 8) == pytest.approx(7 / 15)
    assert attempt(drills.pipeline_bubble, 2, 1) == pytest.approx(0.5)
    assert attempt(drills.pipeline_bubble, 1, 100) == 0.0


def test_microbatches_needed_grows_with_depth():
    for P in (2, 4, 8, 16, 32):
        M = attempt(drills.microbatches_for_bubble, P, 0.10)
        assert attempt(drills.pipeline_bubble, P, M) <= 0.10 + 1e-12
        assert attempt(drills.pipeline_bubble, P, M - 1) > 0.10
    assert attempt(drills.microbatches_for_bubble, 8, 0.10) == 63


def test_parallel_configs_multiply_out():
    cfgs = attempt(drills.parallel_configs, 64)
    assert all(t * p * d == 64 for t, p, d in cfgs)
    assert all(t <= 8 for t, _, _ in cfgs), "tensor parallelism stays in a node"
    assert (8, 1, 8) in cfgs and (1, 1, 64) in cfgs


# -------------------------------------------------------- mixture of experts

def test_moe_multiplies_parameters_and_not_compute():
    d, L, V, ctx, E = 1024, 12, 32000, 2048, 16
    tot, act = attempt(drills.moe_params, d, L, E, V, ctx)
    dense, _ = attempt(drills.moe_params, d, L, 1, V, ctx)
    assert act == pytest.approx(dense)
    assert tot > 5 * dense, "capacity grows; cost per token does not"


def test_expert_load_and_the_balancing_loss():
    n_e = 4
    uniform = np.repeat(np.arange(n_e), 100)
    collapsed = np.zeros(400, dtype=int)
    lu = attempt(drills.expert_load, uniform, n_e)
    lc = attempt(drills.expert_load, collapsed, n_e)
    assert np.allclose(lu, 0.25)
    assert lc[0] == 1.0 and lc[1:].sum() == 0.0
    p_uni = np.full((400, n_e), 1.0 / n_e)
    assert attempt(drills.load_balance_loss, uniform, p_uni, n_e) == pytest.approx(1.0)
    p_col = np.zeros((400, n_e)); p_col[:, 0] = 1.0
    assert attempt(drills.load_balance_loss, collapsed, p_col, n_e) == pytest.approx(n_e)


def test_a_collapsed_router_drops_most_tokens():
    """Section 31: dropped tokens get no expert at all, silently."""
    n_e = 8
    collapsed = np.zeros(800, dtype=int)
    uniform = np.repeat(np.arange(n_e), 100)
    assert attempt(drills.tokens_dropped, uniform, n_e, 1.0) == pytest.approx(0.0)
    d = attempt(drills.tokens_dropped, collapsed, n_e, 1.0)
    assert d == pytest.approx(1 - 1 / n_e), f"dropped {d:.3f}"
    assert attempt(drills.tokens_dropped, collapsed, n_e, 2.0) < d


# ----------------------------------------------------------------- the run

def test_checkpoint_interval_balances_the_two_costs():
    I = attempt(drills.checkpoint_interval, 60.0, 3600.0)
    assert I == pytest.approx(np.sqrt(2 * 60 * 3600))
    def overhead(i):
        return 60.0 / i + i / (2 * 3600.0)
    assert overhead(I) <= overhead(I * 0.5) and overhead(I) <= overhead(I * 2)


def test_sustained_fraction_catches_an_optimistic_plan():
    N, peak = 7e9, 1e15
    frac = attempt(drills.sustained_flops_fraction, 20000, N, peak)
    assert 0 < frac < 1
    assert frac == pytest.approx(20000 * 6 * N / peak)


def test_noise_scale_is_larger_for_a_noisier_gradient():
    r = np.random.default_rng(0)
    true = np.ones(200)
    def batch_grad(b, scale):
        return true + r.normal(0, scale / np.sqrt(b), 200)
    quiet = attempt(drills.gradient_noise_scale,
                    batch_grad(32, 0.3), batch_grad(512, 0.3), 32, 512)
    loud = attempt(drills.gradient_noise_scale,
                   batch_grad(32, 3.0), batch_grad(512, 3.0), 32, 512)
    assert loud > quiet
