"""
Part XXIV drills — Uncertainty Quantification.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail. Reference solutions are in
../solutions/reference.py; read them after you have tried, not before.

An interval makes a checkable promise, so nearly every drill here is the check.
"""
import numpy as np


# ------------------------------------------------------------------ coverage

def coverage(lo, hi, y):
    """Fraction of `y` falling inside the intervals [lo, hi]."""
    raise NotImplementedError


def coverage_by_group(lo, hi, y, group):
    """Coverage within each group. Return {group: coverage}."""
    raise NotImplementedError


def interval_width(lo, hi):
    """Mean width of the intervals."""
    raise NotImplementedError


# --------------------------------------------------------- the two kinds

def decompose_variance(member_means, member_vars):
    """The law of total variance, per point.

    `member_means` and `member_vars` are (K, N): each ensemble member's mean and
    its own predicted noise variance at each point. Return (aleatoric, epistemic)
    as arrays of length N — the average of the members' variances, and the
    variance of their means.
    """
    raise NotImplementedError


def gaussian_interval(mu, sd, alpha=0.05):
    """A two-sided Gaussian interval of nominal coverage 1 - alpha.
    Return (lo, hi)."""
    raise NotImplementedError


# ------------------------------------------------------ Gaussian processes

def rbf_kernel(a, b, ell=1.0, amp=1.0):
    """The squared-exponential kernel matrix between two sets of scalar inputs."""
    raise NotImplementedError


def gp_posterior(xtr, ytr, xq, ell=1.0, amp=1.0, noise=0.1):
    """Exact GP posterior mean and variance at `xq`.

    Return (mean, var) where `var` is the variance of the LATENT function — the
    predictive variance for a new observation adds the noise on top, and keeping
    them separate is the point of drill C7.
    """
    raise NotImplementedError


def log_marginal_likelihood(xtr, ytr, ell=1.0, amp=1.0, noise=0.1):
    """The GP log marginal likelihood. Use a Cholesky factorisation rather than
    an explicit inverse."""
    raise NotImplementedError


# ---------------------------------------------------- conformal prediction

def conformal_quantile(scores, alpha):
    """The split-conformal quantile of the calibration scores.

    Note the finite-sample correction: the index is ceil((n+1)(1-alpha)), not
    ceil(n(1-alpha)). Getting this wrong under-covers slightly, and the drill is
    measuring by how much.
    """
    raise NotImplementedError


def conformal_interval(pred, q):
    """The interval around a point prediction. Return (lo, hi)."""
    raise NotImplementedError


def normalised_scores(y, pred, sigma):
    """Nonconformity scores divided by a predicted difficulty, which is what
    makes the resulting interval adapt in width."""
    raise NotImplementedError


def conformal_set(probs, q):
    """For classification: the set of labels whose score 1 - p is at or below q.

    `probs` is (N, K). Return a boolean (N, K) mask. Note that the set can be
    empty, and that this is informative rather than a bug.
    """
    raise NotImplementedError


def weighted_conformal_quantile(scores, weights, alpha):
    """The quantile of a WEIGHTED empirical distribution of scores — the repair
    for covariate shift, where the weights are the likelihood ratio."""
    raise NotImplementedError


# ---------------------------------------------------------------- decisions

def optimal_threshold(cost_fp, cost_fn):
    """The probability above which acting has lower expected cost."""
    raise NotImplementedError


def expected_cost(p, y, threshold, cost_fp, cost_fn):
    """Mean cost of thresholding the predicted probabilities `p` at
    `threshold`, given the realised outcomes `y`."""
    raise NotImplementedError


def value_of_information(p_outcome, p_signal, p_outcome_given_signal, utility):
    """Expected value of observing a binary signal before acting.

    `utility` is a (n_actions, n_outcomes) array. Return the gain in expected
    utility from seeing the signal — which is zero whenever the signal never
    changes which action is best.
    """
    raise NotImplementedError


def abstain_band(cost_fp, cost_fn, cost_abstain):
    """The (lo, hi) probability band within which abstaining has the lowest
    expected cost. If abstaining is never optimal, return (nan, nan)."""
    raise NotImplementedError
