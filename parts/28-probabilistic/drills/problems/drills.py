"""
Part XXVIII drills — Probabilistic Models and Inference.

Fill in the stubs. Run the tests with

    python3 -m pytest test_drills.py -q

Unimplemented stubs SKIP rather than fail. Reference solutions are in
../solutions/reference.py; read them after you have tried, not before.

Numpy only. Almost every test here compares an approximation against the exact
answer it approximates, which is the property that makes this subject unusually
good to learn by building.
"""
import itertools

import numpy as np


def joint_from_graph(parents, tables, card):
    """The full joint from a directed model: a product over nodes of p(x_i | parents).
    `parents[i]` is a tuple of parent indices, `tables[i]` maps a parent
    configuration to a distribution over x_i. Returns an n-dimensional array."""
    raise NotImplementedError


def marginal(J, keep):
    """Sum a joint down to the axes in `keep`."""
    raise NotImplementedError


def is_collider(parents, node, a, b):
    """Is `node` a collider on the path through a and b?"""
    raise NotImplementedError


def descendants(parents, node):
    """Every node reachable by following arrows forward from `node`."""
    raise NotImplementedError


def d_separated(parents, a, b, given):
    """Are a and b d-separated by `given`? A path is blocked at a non-collider
    that is observed, and at a collider that is not observed and has no observed
    descendant. Easy to get subtly wrong and silent when you do — check it
    against sampled correlations."""
    raise NotImplementedError


def partial_corr(a, b, given, bins=12):
    """Correlation between a and b within narrow strata of `given` — a conditional
    dependence measured rather than assumed linear."""
    raise NotImplementedError


def forward_backward(pi, A, B, obs):
    """Posterior over each latent state of an HMM, and the log evidence. Scale the
    forward messages at every step or a long sequence underflows to zero."""
    raise NotImplementedError


def enumerate_posterior(pi, A, B, obs):
    """The same thing by summing over every state sequence. Exponential, and the
    ground truth that tells you whether your recursion's indices are right."""
    raise NotImplementedError


def viterbi(pi, A, B, obs):
    """The single most probable state sequence, and its log probability. Not the
    sequence of most probable states, which can have probability zero."""
    raise NotImplementedError


def kalman_scalar(y, F, H, Q, R, m0, P0):
    """A scalar Kalman filter. Returns the filtered mean and variance after each
    observation — which are the exact posterior, not an approximation to it."""
    raise NotImplementedError


def em_gmm(x, K, seed=0, iters=200):
    """EM for a one-dimensional Gaussian mixture. Returns the log-likelihood at
    every iteration and the fitted parameters. Assert the likelihood never
    decreases; a decrease means the E and M steps disagree about the model."""
    raise NotImplementedError


def meanfield_gaussian_sd(rho):
    """The mean-field marginal standard deviations for a 2-D Gaussian posterior with
    unit marginals and correlation rho. There is a closed form, and it is not 1."""
    raise NotImplementedError


def elbo_gaussian(m, log_sd, Sigma):
    """The ELBO for a factorised Gaussian q against a zero-mean Gaussian p, in
    closed form. Maximising this over m and log_sd must reproduce the answer
    above — which is what makes the error a property of the objective."""
    raise NotImplementedError


def importance_weights(logp, logq, samples):
    """Self-normalised importance weights, computed in log space."""
    raise NotImplementedError


def ess_weights(w):
    """Effective sample size of a weight vector. The number to look at before any
    estimate computed from weights."""
    raise NotImplementedError


def metropolis(logp, x0, step, n, seed=0):
    """Random-walk Metropolis. Returns the chain and the acceptance rate."""
    raise NotImplementedError


def leapfrog(gradp, x, p, step, L):
    """One leapfrog trajectory: half-kick, then L drift-kick steps, closing with a
    half-kick. Check energy conservation before trusting anything downstream."""
    raise NotImplementedError


def hmc(logp, gradp, x0, step, L, n, seed=0):
    """Hamiltonian Monte Carlo built on the leapfrog above."""
    raise NotImplementedError


def ess_chain(x):
    """Effective sample size of a correlated chain, from its autocorrelation. Quote
    the Monte Carlo error from this rather than from the number of draws."""
    raise NotImplementedError


def rhat(chains):
    """The Gelman-Rubin statistic on an (m, n) array of draws. It tests whether
    chains agree, which is necessary and not sufficient."""
    raise NotImplementedError


def euler_maruyama(f, g, x0, T, steps, seed=0):
    """Simulate dx = f(x,t) dt + g(t) dW. The noise scales as sqrt(dt), and that
    exponent is the only thing distinguishing this from an ODE solver."""
    raise NotImplementedError


def ou_marginal(x0, t, theta=1.0, sigma=np.sqrt(2.0)):
    """The exact marginal mean and variance of the Ornstein-Uhlenbeck process
    dx = -theta x dt + sigma dW started at x0, for checking the simulator above."""
    raise NotImplementedError


def wasserstein1(a, b):
    """The 1-Wasserstein distance between two samples in one dimension: sort both
    and average the absolute differences."""
    raise NotImplementedError


def kl_gaussian(m1, s1, m2, s2):
    """KL divergence between two univariate Gaussians, in closed form."""
    raise NotImplementedError


def sinkhorn(a, b, C, eps, iters=500):
    """Entropic optimal transport by alternate rescaling. Returns the coupling."""
    raise NotImplementedError
