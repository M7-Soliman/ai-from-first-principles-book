"""
Reference solutions for the Part XVII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------- counting FLOPs

def transformer_params(d, n_layers, vocab, ctx):
    return dict(attention=4 * d * d * n_layers,
                ffn=8 * d * d * n_layers,
                norms=4 * d * n_layers,
                embedding=vocab * d,
                positional=ctx * d)


def training_flops(N, tokens):
    return 6.0 * N * tokens


def inference_flops(N, tokens):
    return 2.0 * N * tokens


def lifetime_flops(N, train_tokens, served_tokens):
    return training_flops(N, train_tokens) + inference_flops(N, served_tokens)


# --------------------------------------------------------------- scaling laws

def power_law(x, A, alpha, E):
    return A * np.asarray(x, float) ** (-alpha) + E


def fit_power_law(xs, ys, pin_E=None):
    from scipy.optimize import curve_fit
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if pin_E is None:
        f = lambda x, lA, al, E: np.exp(lA) * x ** (-al) + E
        p0 = [np.log(max(ys.max(), 1e-6)), 0.3, float(ys.min()) * 0.9]
        (lA, al, E), _ = curve_fit(f, xs, ys, p0=p0, maxfev=400000)
        return float(np.exp(lA)), float(al), float(E)
    f = lambda x, lA, al: np.exp(lA) * x ** (-al) + pin_E
    (lA, al), _ = curve_fit(f, xs, ys, p0=[0.0, 0.3], maxfev=400000)
    return float(np.exp(lA)), float(al), float(pin_E)


def compute_optimal_allocation(C, alpha, beta, A, B):
    # minimise A N^-a + B (C/(6N))^-b  ->  d/dN = 0
    # -aA N^-(a+1) + bB (6/C)^b N^(b-1) = 0
    #  aA N^-(a+1) = bB (6/C)^b N^(b-1)
    #  N^(a+b) = aA / (bB (6/C)^b)
    N = (alpha * A / (beta * B * (6.0 / C) ** beta)) ** (1.0 / (alpha + beta))
    D = C / (6.0 * N)
    return float(N), float(D)


def tokens_per_parameter(N, D):
    return float(D) / float(N)


# ------------------------------------------------------------------- memory

def training_memory(N, bytes_param=2, optimizer="adam"):
    opt = {"adam": 12.0, "momentum": 4.0, "sgd": 0.0}[optimizer] * N
    return dict(parameters=float(N * bytes_param),
                gradients=float(N * bytes_param),
                optimizer_states=float(opt))


def activation_memory(batch, ctx, d, n_layers, heads, bytes_act=2,
                      materialise_attention=True):
    per = 10.0 * batch * ctx * d * bytes_act
    if materialise_attention:
        per += float(batch) * heads * ctx * ctx * bytes_act
    return per * n_layers


def zero_memory(N, world_size, stage, bytes_param=2):
    m = training_memory(N, bytes_param)
    if stage >= 1:
        m["optimizer_states"] /= world_size
    if stage >= 2:
        m["gradients"] /= world_size
    if stage >= 3:
        m["parameters"] /= world_size
    return float(sum(m.values()))


# ------------------------------------------------------------ number formats

def float_format(exp_bits, man_bits):
    bias = 2 ** (exp_bits - 1) - 1
    max_exp = 2 ** exp_bits - 2 - bias
    min_exp = 1 - bias
    return dict(largest=(2 - 2.0 ** -man_bits) * 2.0 ** max_exp,
                smallest_normal=2.0 ** min_exp,
                smallest_subnormal=2.0 ** (min_exp - man_bits),
                eps=2.0 ** -man_bits)


def representable_fraction(values, fmt):
    v = np.abs(np.asarray(values, float))
    ok = (v >= fmt["smallest_subnormal"]) & (v <= fmt["largest"])
    return float(ok.mean())


def loss_scale_for(values, fmt, target_frac=0.999):
    for k in range(0, 40):
        s = 2.0 ** k
        if representable_fraction(np.asarray(values, float) * s, fmt) >= target_frac:
            return s
    return 2.0 ** 40


# ------------------------------------------------------------- communication

def allreduce_bytes(N, world_size, bytes_grad=2):
    return 2.0 * N * bytes_grad * (world_size - 1) / world_size


def dp_efficiency(N, tokens_per_device, device_flops, bandwidth_bytes_per_s,
                  world_size=8, bytes_grad=2):
    comp = tokens_per_device * 6.0 * N / device_flops
    comm = allreduce_bytes(N, world_size, bytes_grad) / bandwidth_bytes_per_s
    return float(comp / (comp + comm))


def pipeline_bubble(n_stages, n_microbatches):
    return float(n_stages - 1) / float(n_microbatches + n_stages - 1)


def microbatches_for_bubble(n_stages, target):
    # (P-1)/(M+P-1) <= t  ->  M >= (P-1)(1-t)/t
    return int(np.ceil((n_stages - 1) * (1.0 - target) / target))


def parallel_configs(n_devices):
    out = []
    for t in range(1, 9):
        if n_devices % t:
            continue
        rest = n_devices // t
        for p in range(1, rest + 1):
            if rest % p == 0:
                out.append((t, p, rest // p))
    return out


# -------------------------------------------------------- mixture of experts

def moe_params(d, n_layers, n_experts, vocab, ctx):
    base = transformer_params(d, n_layers, vocab, ctx)
    dense_ffn = base["ffn"]
    total = sum(base.values()) - dense_ffn + dense_ffn * n_experts
    active = sum(base.values())
    return float(total), float(active)


def expert_load(assignments, n_experts):
    a = np.asarray(assignments).ravel()
    return np.bincount(a, minlength=n_experts)[:n_experts] / max(len(a), 1)


def load_balance_loss(assignments, probs, n_experts):
    frac = expert_load(assignments, n_experts)
    mean_p = np.asarray(probs, float).mean(0)
    return float(n_experts * (frac * mean_p).sum())


def tokens_dropped(assignments, n_experts, capacity_factor):
    a = np.asarray(assignments).ravel()
    n = len(a)
    cap = capacity_factor * n / n_experts
    counts = np.bincount(a, minlength=n_experts)[:n_experts]
    return float(np.clip(counts - cap, 0, None).sum() / max(n, 1))


# ----------------------------------------------------------------- the run

def checkpoint_interval(checkpoint_cost, mean_time_to_failure):
    return float(np.sqrt(2.0 * checkpoint_cost * mean_time_to_failure))


def gradient_noise_scale(grad_small, grad_large, b_small, b_large):
    g2s = float(np.sum(np.asarray(grad_small, float) ** 2))
    g2l = float(np.sum(np.asarray(grad_large, float) ** 2))
    # unbiased estimates of |G|^2 and tr(Sigma) from two batch sizes
    G2 = (b_large * g2l - b_small * g2s) / (b_large - b_small)
    S = (g2s - g2l) / (1.0 / b_small - 1.0 / b_large)
    return float(abs(S / max(G2, 1e-30)))


def sustained_flops_fraction(measured_tokens_per_s, N, peak_flops):
    return float(measured_tokens_per_s * 6.0 * N / peak_flops)


# --------------------------------------------------------------------- patch

for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
