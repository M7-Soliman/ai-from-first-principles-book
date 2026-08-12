"""
Part XXII drills — Causal Inference.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail. Reference solutions are in
../solutions/reference.py; read them after you have tried, not before.

Graphs are represented as a dict of node -> set of its DIRECT CAUSES (parents),
which is the direction that matches how a structural causal model is written.
"""
import numpy as np


# ------------------------------------------------------------ the graph tools

def parents(graph, node):
    """Direct causes of `node`. Returns a set."""
    raise NotImplementedError


def descendants(graph, node):
    """Every node reachable from `node` by following arrows forward, not
    including `node` itself."""
    raise NotImplementedError


def paths(graph, a, b):
    """All undirected paths between `a` and `b`, as lists of nodes. Arrows may
    be traversed in either direction — that is what makes a backdoor a path."""
    raise NotImplementedError


def is_collider(graph, path, i):
    """Is the node at position `i` of `path` a collider on that path? A collider
    has BOTH its path-neighbours as parents."""
    raise NotImplementedError


def path_is_blocked(graph, path, given):
    """Is this path blocked by conditioning on the set `given`?

    A path is blocked when any node on it is either
      * a non-collider that IS in `given`, or
      * a collider that is NOT in `given` and has no descendant in `given`.
    """
    raise NotImplementedError


def d_separated(graph, a, b, given=frozenset()):
    """Are `a` and `b` d-separated given `given`? True when every path between
    them is blocked."""
    raise NotImplementedError


def satisfies_backdoor(graph, x, y, z):
    """Does `z` satisfy the backdoor criterion for the effect of `x` on `y`?

    Two conditions, both from §12: no element of `z` is a descendant of `x`, and
    `z` blocks every path from `x` to `y` that starts with an arrow INTO `x`.
    """
    raise NotImplementedError


# ------------------------------------------------------------- the estimators

def naive_difference(y, t):
    """Mean outcome among treated minus mean among untreated. The quantity that
    is right about the wrong question when there is confounding."""
    raise NotImplementedError


def backdoor_adjust(y, t, z, bins=10):
    """Stratified backdoor adjustment for a single continuous covariate.

    Bin `z`, take the treated-minus-untreated difference within each bin, and
    average those differences weighted by the bin's share of the WHOLE sample —
    not by its share of the treated. Bins containing only one arm are skipped.
    """
    raise NotImplementedError


def fit_propensity(z, t, iters=25):
    """Logistic regression of `t` on `z` by Newton-Raphson. Returns fitted
    probabilities. `z` may be 1-D or 2-D."""
    raise NotImplementedError


def ipw_estimate(y, t, e, clip=0.0):
    """Inverse-probability weighted effect estimate. `clip` bounds the
    propensity into [clip, 1-clip] when non-zero."""
    raise NotImplementedError


def aipw_estimate(y, t, e, m1, m0):
    """The doubly robust estimator of §21. `m1` and `m0` are the outcome model's
    predictions under treatment and control, for every unit."""
    raise NotImplementedError


def effective_sample_size(w):
    """Kish's effective sample size for weights `w`: the sample size that would
    give the same variance if every unit were weighted equally. Return it as a
    FRACTION of len(w), so 1.0 means no loss."""
    raise NotImplementedError


def overlap_range(e, t):
    """Return (lo, hi): the range of propensity where BOTH arms have units.
    Empty overlap should return (nan, nan)."""
    raise NotImplementedError


# --------------------------------------------------------------- the designs

def two_stage_least_squares(y, x, z):
    """The instrumental-variable estimate of the effect of `x` on `y` using
    instrument `z`. Regress x on z, then y on the fitted values, and return the
    slope."""
    raise NotImplementedError


def first_stage_strength(x, z):
    """How strongly the instrument moves the treatment, as the F-statistic of
    the first-stage regression. The rule of thumb everyone quotes is F > 10, and
    drill C15 asks you to find out what that rule is actually worth."""
    raise NotImplementedError


def did_estimate(y, treated, after):
    """Difference-in-differences. `treated` and `after` are boolean arrays."""
    raise NotImplementedError


def rdd_estimate(y, running, cutoff, bandwidth):
    """Regression discontinuity: fit a line on each side of `cutoff` within
    `bandwidth`, and return the jump at the cutoff."""
    raise NotImplementedError


# ---------------------------------------------------- policies and transfer

def off_policy_value(rewards, actions, logged_prob, target_policy_prob,
                     clip=None):
    """Inverse-propensity estimate of a target policy's value from logged data.

    `logged_prob` is the probability the LOGGING policy took the action it took;
    `target_policy_prob` is the probability the TARGET policy would take that
    same action. `clip` bounds the importance ratio when given.
    """
    raise NotImplementedError


def self_normalised_value(rewards, actions, logged_prob, target_policy_prob):
    """The same estimate, dividing by the sum of the weights rather than by n.
    Biased, lower variance — drill C18 measures the trade."""
    raise NotImplementedError
