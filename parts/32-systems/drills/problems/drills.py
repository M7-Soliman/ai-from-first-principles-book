"""
Part XXXII drills — Systems and Security.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail. Reference solutions are in
../solutions/reference.py; read them after you have tried, not before.

Numpy only. These are arithmetic rather than timing — the timing drills in the
prose depend on your machine, and a test cannot assert a number that does.
"""
import numpy as np


def arithmetic_intensity(flops, bytes_moved):
    """Flops per byte moved. Not the flop count and not the byte count — their
    ratio is what decides which ceiling you are under."""
    raise NotImplementedError


def roofline(intensity, peak_bandwidth, peak_flops):
    """Achievable throughput: intensity times bandwidth, or peak arithmetic,
    whichever is smaller."""
    raise NotImplementedError


def ridge_point(peak_bandwidth, peak_flops):
    """The intensity at which the two ceilings meet. Below it you are waiting for
    memory and only moving less data helps."""
    raise NotImplementedError


def is_memory_bound(intensity, peak_bandwidth, peak_flops):
    """Which side of the ridge a kernel sits on. Ten minutes to compute and it
    decides what to work on."""
    raise NotImplementedError


def dense_layer_intensity(d_in, d_out, batch, bytes_per_element=2):
    """Arithmetic intensity of a dense layer at a given batch size. The same layer is
    hopelessly memory-bound at batch 1 and compute-bound at batch 512, which is
    why generation is bandwidth-limited and training is not."""
    raise NotImplementedError


def unfused_bytes(n_elements, chain, bytes_per_element=8):
    """Bytes moved by a chain of elementwise operations run as separate passes."""
    raise NotImplementedError


def fused_bytes(n_elements, chain, bytes_per_element=8):
    """Bytes moved by the same chain fused into one pass — independent of the chain
    length, which is the whole point."""
    raise NotImplementedError


def fusion_speedup_bound(chain):
    """The most fusing a chain can buy. You cannot make fewer than one pass."""
    raise NotImplementedError


def blocked_matmul(A, B, bs):
    """Matrix multiplication in blocks, so each block's working set fits in cache.
    Identical arithmetic, different memory traffic."""
    raise NotImplementedError


def working_set_bytes(bs, bytes_per_element=8):
    """Bytes live at once for a given block size: a tile of each input and one of the
    output."""
    raise NotImplementedError


def best_block_size(cache_bytes, bytes_per_element=8, candidates=(16, 32, 64, 128, 256, 512)):
    """The largest block whose working set fits the cache. Computable rather than
    swept, which is the point of knowing the hierarchy."""
    raise NotImplementedError


def naive_allreduce_volume(workers, buffer_bytes=1.0):
    """Bytes each worker sends if everyone sends their buffer to everyone."""
    raise NotImplementedError


def ring_allreduce_volume(workers, buffer_bytes=1.0):
    """Bytes each worker sends in a ring — reduce-scatter around, then all-gather
    around. Bounded by two buffers however many workers there are."""
    raise NotImplementedError


def ring_allreduce_steps(workers):
    """Sequential hops a ring takes. Optimal in volume and not in latency, which is
    why libraries switch algorithm by message size."""
    raise NotImplementedError


def collective_time(workers, buffer_bytes, bandwidth, latency, algorithm="ring"):
    """Volume over bandwidth, plus per-hop latency. The two terms dominate at
    different message sizes and the crossover is what a real library implements."""
    raise NotImplementedError


def straggler_step_time(times):
    """How long a synchronous step takes given each worker's time. The tail sets the
    pace, not the mean."""
    raise NotImplementedError


def training_memory(params, bytes_per_param=4, optimiser_states=2, activations=0.0):
    """Parameters, gradients, optimiser state and activations. Adam keeps two moments,
    so the first three are about four times the parameters before activations."""
    raise NotImplementedError


def clip_gradients(G, clip):
    """Bound each example's gradient to a fixed norm, so no single record can move
    the parameters far."""
    raise NotImplementedError


def privatise(G, clip, sigma, seed=0):
    """The private gradient step: clip, sum, then add noise calibrated to the clip.
    Both parts are necessary — clipping alone bounds influence without hiding it."""
    raise NotImplementedError


def compose_budget(eps_per_step, steps):
    """Total privacy budget over many steps under basic composition. Reporting a
    per-step epsilon without the total is how a guarantee gets overstated."""
    raise NotImplementedError


def group_privacy(eps, records_per_person):
    """The guarantee for a person contributing several records. The definition
    protects a record, so it weakens linearly in the count."""
    raise NotImplementedError


def auc(scores, labels):
    """Area under the ROC curve, for scoring a membership-inference attack. Ground
    truth is available here, which is what makes the attack exactly scorable."""
    raise NotImplementedError


def margin_to_boundary(X, w):
    """Exact distance from each point to a linear model's decision boundary, so the
    minimum successful perturbation is known rather than searched for."""
    raise NotImplementedError


def attack_success_rate(X, w, eps):
    """Fraction of points that can be flipped within a given budget."""
    raise NotImplementedError


def extraction_queries_needed(n_parameters, outputs="probabilities"):
    """Order-of-magnitude queries to clone a model. Scales with the parameters being
    recovered, not with the victim's training set — which is why a model trained
    on a proprietary corpus can be cloned by someone who has none of it."""
    raise NotImplementedError
