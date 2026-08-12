"""
Reference solutions for the Part XXXII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------- roofline

def arithmetic_intensity(flops, bytes_moved):
    return float(flops) / float(bytes_moved)


def roofline(intensity, peak_bandwidth, peak_flops):
    return float(np.minimum(np.asarray(intensity, float) * peak_bandwidth, peak_flops))


def ridge_point(peak_bandwidth, peak_flops):
    return float(peak_flops) / float(peak_bandwidth)


def is_memory_bound(intensity, peak_bandwidth, peak_flops):
    return bool(intensity < ridge_point(peak_bandwidth, peak_flops))


def dense_layer_intensity(d_in, d_out, batch, bytes_per_element=2):
    """A dense layer reads its weights once and its activations once. At batch 1
    each weight is used once; at large batch it is reused, which changes the
    regime rather than the constant."""
    flops = 2.0 * d_in * d_out * batch
    bytes_moved = bytes_per_element * (d_in * d_out + batch * (d_in + d_out))
    return flops / bytes_moved


# -------------------------------------------------------------------- fusion

def unfused_bytes(n_elements, chain, bytes_per_element=8):
    """Each operation reads and writes the whole array."""
    return 2 * bytes_per_element * n_elements * chain


def fused_bytes(n_elements, chain, bytes_per_element=8):
    """One pass, whatever the chain length."""
    return 2 * bytes_per_element * n_elements


def fusion_speedup_bound(chain):
    return float(chain)


# --------------------------------------------------------------------- cache

def blocked_matmul(A, B, bs):
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = A.shape[0]
    C = np.zeros((n, B.shape[1]))
    for i0 in range(0, n, bs):
        for k0 in range(0, A.shape[1], bs):
            a = A[i0:i0 + bs, k0:k0 + bs]
            for j0 in range(0, B.shape[1], bs):
                C[i0:i0 + bs, j0:j0 + bs] += a @ B[k0:k0 + bs, j0:j0 + bs]
    return C


def working_set_bytes(bs, bytes_per_element=8):
    """Three blocks live at once: a tile of each input and one of the output."""
    return 3 * bs * bs * bytes_per_element


def best_block_size(cache_bytes, bytes_per_element=8, candidates=(16, 32, 64, 128, 256, 512)):
    fits = [b for b in candidates if working_set_bytes(b, bytes_per_element) <= cache_bytes]
    return max(fits) if fits else min(candidates)


# --------------------------------------------------------------- collectives

def naive_allreduce_volume(workers, buffer_bytes=1.0):
    return float(workers - 1) * buffer_bytes


def ring_allreduce_volume(workers, buffer_bytes=1.0):
    return 2.0 * (workers - 1) / workers * buffer_bytes


def ring_allreduce_steps(workers):
    return 2 * (int(workers) - 1)


def collective_time(workers, buffer_bytes, bandwidth, latency, algorithm="ring"):
    if algorithm == "ring":
        vol = ring_allreduce_volume(workers, buffer_bytes)
        steps = ring_allreduce_steps(workers)
    else:
        vol = naive_allreduce_volume(workers, buffer_bytes)
        steps = 1
    return vol / bandwidth + steps * latency


def straggler_step_time(times):
    """A synchronous step runs at the pace of the slowest worker."""
    return float(np.max(times))


def training_memory(params, bytes_per_param=4, optimiser_states=2, activations=0.0):
    """Parameters, gradients, optimiser state and activations."""
    return float(params * bytes_per_param * (2 + optimiser_states) + activations)


# -------------------------------------------------------------------- privacy

def clip_gradients(G, clip):
    G = np.asarray(G, float)
    nrm = np.linalg.norm(G, axis=1, keepdims=True)
    return G * np.minimum(1.0, clip / np.maximum(nrm, 1e-12))


def privatise(G, clip, sigma, seed=0):
    r = np.random.default_rng(seed)
    Gc = clip_gradients(G, clip)
    return Gc.sum(0) + sigma * clip * r.standard_normal(G.shape[1])


def compose_budget(eps_per_step, steps):
    """Basic composition: budgets add. Advanced accountants do better; this is
    the bound everyone should be able to compute."""
    return float(eps_per_step) * int(steps)


def group_privacy(eps, records_per_person):
    """The definition protects a record. A person contributing k records has a
    guarantee weaker by a factor of k."""
    return float(eps) * int(records_per_person)


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------- adversary

def margin_to_boundary(X, w):
    """For a linear model, the exact distance from each point to the decision
    boundary — so an attack's minimum budget is known rather than searched."""
    X = np.asarray(X, float); w = np.asarray(w, float)
    return np.abs(X @ w) / np.linalg.norm(w)


def attack_success_rate(X, w, eps):
    return float((margin_to_boundary(X, w) < eps).mean())


def extraction_queries_needed(n_parameters, outputs="probabilities"):
    """Order-of-magnitude guide: the clone must recover the parameters, not the
    victim's training set. Label-only outputs raise the constant, not the
    scaling."""
    return int(n_parameters * (1.5 if outputs == "probabilities" else 24))


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
