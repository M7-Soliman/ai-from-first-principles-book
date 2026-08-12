"""
Part XXI drills — Alignment, Interpretability and Consequence.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail, so the suite stays readable while you
work through it. Reference solutions are in ../solutions/reference.py; read them
after you have tried, not before.
"""
import numpy as np


# ------------------------------------------------------ what evidence you have

def zero_failure_bound(n, conf=0.95):
    """Upper bound on a failure rate after `n` trials with ZERO failures.

    The exact bound: the largest p for which seeing zero failures still has
    probability at least (1 - conf).  Return a float in [0, 1].
    """
    raise NotImplementedError


def rule_of_three(n):
    """The approximation everyone should carry: 3/n. Return a float."""
    raise NotImplementedError


def trials_needed(p, conf=0.95):
    """Trials to observe a failure of true rate `p` at least once, with
    probability at least `conf`. Return an integer (round up)."""
    raise NotImplementedError


def attack_success(p, n):
    """Probability that at least one of `n` independent attempts succeeds when
    each succeeds with probability `p`."""
    raise NotImplementedError


def binomial_interval(k, n, conf=0.95):
    """A (lo, hi) confidence interval for a rate given `k` failures in `n`
    trials. Any standard construction is fine as long as it contains k/n and
    stays inside [0, 1]; the tests only check those properties and the
    zero-failure case, where `hi` must agree with zero_failure_bound."""
    raise NotImplementedError


# ------------------------------------------------------------- superposition

def features_kept(W, frac=0.5):
    """`W` is (n_dims, n_features). A feature counts as represented when the
    norm of its column exceeds `frac` times the largest column norm. Return the
    count as an int."""
    raise NotImplementedError


def mean_overlap(W):
    """Mean absolute cosine between DISTINCT feature directions (the columns of
    W, normalised). Zero for an orthogonal set. Return a float."""
    raise NotImplementedError


def max_orthogonal(n_dims):
    """How many mutually orthogonal directions fit in `n_dims` dimensions.

    The point of the drill is that superposition is what happens when a model
    needs more than this, so the answer is the boundary the phenomenon crosses.
    """
    raise NotImplementedError


# ------------------------------------------------------- dictionary recovery

def best_cosine(true_dirs, found_dirs):
    """For each row of `true_dirs`, the largest absolute cosine with any row of
    `found_dirs`. Directions are unsigned, so use the absolute value. Return an
    array of length len(true_dirs)."""
    raise NotImplementedError


def recovered_count(true_dirs, found_dirs, thresh=0.9):
    """How many true directions are matched above `thresh`. Return an int."""
    raise NotImplementedError


# ---------------------------------------------------- probing against ablation

def ablate_direction(H, d):
    """Project every row of `H` (n, dim) off the direction `d`. Return the
    projected activations, same shape. `d` need not arrive normalised."""
    raise NotImplementedError


def causal_effect(acc_intact, acc_ablated, acc_random):
    """The ablation effect, expressed against the random-direction control.

    Return (raw_drop, control_adjusted_drop) where the first is
    acc_intact - acc_ablated and the second additionally subtracts the drop a
    random direction of the same norm produced. The second is the number worth
    reporting, and the drill is noticing why.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ patching

def restoration(acc_patched, acc_corrupt, acc_clean):
    """The fraction of the clean behaviour a patch restores, normalised so that
    0 means 'no better than the corrupted run' and 1 means 'as good as clean'.

    Guard the degenerate case where the clean and corrupted runs score the same:
    return 0.0 rather than dividing by zero.
    """
    raise NotImplementedError


def steering_vector(h_pos, h_neg):
    """The difference-in-means direction: mean of `h_pos` minus mean of
    `h_neg`, returned as a UNIT vector."""
    raise NotImplementedError


# ---------------------------------------------------------------- Goodhart

def best_of_n_true(true, proxy, n, rng=None):
    """Split the candidates into groups of `n`, pick the argmax of `proxy` in
    each group, and return the MEAN TRUE reward of the picks.

    This is the measurement in Figure 4, and the thing to notice while writing
    it is how easy it would have been to return the proxy score instead.
    """
    raise NotImplementedError


def tail_ratio(x, q=0.999):
    """A crude tail-heaviness statistic: the `q` quantile of |x| divided by its
    standard deviation. Bigger means heavier. Return a float."""
    raise NotImplementedError


def ensemble_proxy(proxies):
    """Average a list of proxy score arrays. The drill is measuring what this
    does to `tail_ratio`, which is the whole argument for reward-model
    ensembles."""
    raise NotImplementedError


# --------------------------------------------------------- fairness criteria

def demographic_parity(y_pred, group):
    """Positive-prediction rate per group. Return a dict {group_value: rate}."""
    raise NotImplementedError


def equalised_odds(y_pred, y_true, group):
    """Return {group_value: (tpr, fpr)}. A group with no positives (or no
    negatives) gets nan for the undefined rate rather than raising."""
    raise NotImplementedError


def calibration_by_group(p_pred, y_true, group, bins=5):
    """Mean absolute calibration gap per group: bin the predicted probabilities,
    and within each bin compare the mean prediction with the observed rate.
    Return {group_value: mean_abs_gap}."""
    raise NotImplementedError
