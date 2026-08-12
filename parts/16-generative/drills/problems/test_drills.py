"""
Tests for the Part XVI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several tests assert a NEGATIVE result, because those are the part's findings:
the two KL directions give different answers, the ELBO is strictly below the
likelihood, and one number cannot report both ways of being wrong.
"""
import numpy as np
import pytest

import drills

# NumPy 2.0 renamed trapz to trapezoid and deprecated the old name; fall back
# to trapz on older NumPy where trapezoid does not exist yet.
trapezoid = getattr(np, "trapezoid", np.trapz)


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


MUS = np.array([[-1.6, -0.9], [1.5, 0.9], [-1.2, 1.7]])
SDS = np.array([0.42, 0.34, 0.30])
WS = np.array([0.40, 0.35, 0.25])


# --------------------------------------------------------- densities & scores

def test_gaussian_logpdf_integrates_to_one():
    g = np.linspace(-8, 8, 4001)
    lp = attempt(drills.gaussian_logpdf, g[:, None], np.array([0.5]), 1.3)
    assert trapezoid(np.exp(lp), g) == pytest.approx(1.0, abs=1e-6)


def test_mixture_logpdf_integrates_and_is_stable():
    g = np.linspace(-8, 8, 601)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    lp = attempt(drills.mixture_logpdf, P, MUS, SDS, WS)
    dens = np.exp(lp).reshape(GX.shape)
    mass = trapezoid(trapezoid(dens, g, axis=0), g)
    assert mass == pytest.approx(1.0, abs=2e-3)
    far = attempt(drills.mixture_logpdf, np.array([[400.0, 400.0]]), MUS, SDS, WS)
    assert np.isfinite(far).all(), "a naive exp() underflows here"


def test_mixture_score_matches_a_finite_difference():
    P = np.array([[0.2, 0.1], [-1.4, -0.7], [1.2, 1.0]])
    s = attempt(drills.mixture_score, P, MUS, SDS, WS)
    eps = 1e-5
    fd = np.zeros_like(P)
    for k in range(2):
        e = np.zeros(2); e[k] = eps
        fd[:, k] = (attempt(drills.mixture_logpdf, P + e, MUS, SDS, WS)
                    - attempt(drills.mixture_logpdf, P - e, MUS, SDS, WS)) / (2 * eps)
    assert np.abs(s - fd).max() < 1e-4


def test_smoothing_flattens_the_score():
    """Section 25's pitfall: the denoising target is the SMOOTHED score."""
    P = np.array([[0.4, 0.2], [-0.8, 0.6]])
    s0 = attempt(drills.mixture_score, P, MUS, SDS, WS, 0.0)
    sm = attempt(drills.mixture_score, P, MUS, SDS, WS, 0.6 ** 2)
    assert np.linalg.norm(sm, axis=1).mean() < np.linalg.norm(s0, axis=1).mean()


def test_energy_gives_back_a_normalised_density():
    g = np.linspace(-6, 6, 2001); dx = g[1] - g[0]
    E = 0.5 * g ** 2
    p, Z = attempt(drills.energy_to_density, E, g, dx)
    assert trapezoid(p, dx=dx) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(p, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), atol=1e-6)
    E2 = E + 137.0                                   # an energy shift is a Z shift
    p2, _ = attempt(drills.energy_to_density, E2, g, dx)
    assert np.allclose(p, p2), "adding a constant to E must not change the density"


# --------------------------------------------------------------- divergences

def test_kl_is_zero_only_for_identical_densities():
    g = np.linspace(-6, 6, 1201); dx = g[1] - g[0]
    p = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    q = np.exp(-0.5 * ((g - 0.7) / 1.2) ** 2) / (1.2 * np.sqrt(2 * np.pi))
    assert attempt(drills.kl_divergence, p, p, dx) == pytest.approx(0.0, abs=1e-8)
    assert attempt(drills.kl_divergence, p, q, dx) > 0.05
    assert attempt(drills.kl_divergence, p, q, dx) != \
        pytest.approx(attempt(drills.kl_divergence, q, p, dx), rel=1e-3), \
        "KL is not symmetric — that asymmetry is section 4"


def test_forward_kl_covers_and_reverse_kl_locks_on():
    """Section 4, as an assertion."""
    g = np.linspace(-8, 8, 1601); dx = g[1] - g[0]
    p = (0.5 * np.exp(-0.5 * ((g + 3.0) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
         + 0.5 * np.exp(-0.5 * ((g - 3.0) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi)))
    mu_f, sd_f = attempt(drills.fit_gaussian_forward_kl, g, p, dx)
    mu_r, sd_r = attempt(drills.fit_gaussian_reverse_kl, g, p, dx)
    assert abs(mu_f) < 0.2, "forward KL centres between the modes"
    assert sd_f > 2.5, f"and stretches to cover both: sd {sd_f:.2f}"
    assert abs(abs(mu_r) - 3.0) < 0.6, f"reverse KL picks a mode: mu {mu_r:.2f}"
    assert sd_r < 1.0, f"and stays narrow: sd {sd_r:.2f}"


# ------------------------------------------------------------ autoregressive

def test_chain_rule_matches_a_direct_joint():
    r = np.random.default_rng(0)
    T, V = 5, 4
    lp = np.log(r.dirichlet(np.ones(V), T))
    seq = r.integers(0, V, T)
    got = attempt(drills.chain_rule_logprob, seq, lp)
    assert got == pytest.approx(sum(lp[t, seq[t]] for t in range(T)))
    assert got <= 0.0


def test_causal_mask_shape():
    m = attempt(drills.causal_mask, 4)
    assert m.shape == (4, 4)
    assert not m[2, 0] and not m[2, 2]
    assert m[2, 3], "position 2 must not see position 3"
    assert m.sum() == 6


def test_the_sampling_asymmetry_is_exact():
    """Figure 1, as arithmetic."""
    for n in (16, 256, 4096):
        s, t = attempt(drills.sequential_dependency_depth, n)
        assert s == n and t == 1


def test_top_p_keeps_a_nucleus():
    logits = np.log(np.array([0.5, 0.25, 0.15, 0.06, 0.04]))
    out = attempt(drills.top_p_filter, logits, p=0.9)
    assert np.isfinite(out[:3]).all()
    assert np.isneginf(out[4])
    tiny = attempt(drills.top_p_filter, logits, p=0.01)
    assert np.isfinite(tiny).sum() == 1, "always keep at least the top token"


# ------------------------------------------------------------------ the ELBO

def test_kl_to_standard_normal_is_zero_at_the_prior():
    mu = np.zeros((3, 4)); logvar = np.zeros((3, 4))
    assert np.allclose(attempt(drills.gaussian_kl_to_standard_normal, mu, logvar), 0.0)
    assert (attempt(drills.gaussian_kl_to_standard_normal,
                    np.ones((3, 4)), np.zeros((3, 4))) > 0).all()


def test_reparameterise_has_the_right_moments():
    r = np.random.default_rng(0)
    mu = np.array([[1.0, -2.0]]); logvar = np.array([[np.log(4.0), np.log(0.25)]])
    eps = r.normal(size=(200000, 2))
    z = attempt(drills.reparameterise, mu, logvar, eps)
    assert np.allclose(z.mean(0), mu[0], atol=0.02)
    assert np.allclose(z.std(0), np.array([2.0, 0.5]), atol=0.02)


def test_elbo_is_below_the_iwae_bound_which_is_below_the_truth():
    """Section 17: the ELBO is the k=1 case, and the gap is real."""
    r = np.random.default_rng(1)
    n, k = 200, 400
    log_p = r.normal(-2.0, 1.0, (n, k))
    log_q = r.normal(0.0, 1.0, (n, k))
    iw_1 = attempt(drills.iwae_bound, log_p[:, :1], log_q[:, :1])
    iw_k = attempt(drills.iwae_bound, log_p, log_q)
    assert iw_k.mean() > iw_1.mean(), "more samples must tighten the bound"


def test_iwae_is_stable_with_extreme_weights():
    log_p = np.array([[-900.0, -901.0, -1200.0]])
    log_q = np.array([[0.0, 0.0, 0.0]])
    v = attempt(drills.iwae_bound, log_p, log_q)
    assert np.isfinite(v).all(), "needs log-sum-exp, not exp-then-log"


def test_elbo_terms_reward_good_reconstruction():
    x = np.array([[0.0, 0.0]])
    good = attempt(drills.elbo_terms, x, None, np.zeros((1, 2)), np.zeros((1, 2)),
                   x.copy(), 0.3)
    bad = attempt(drills.elbo_terms, x, None, np.zeros((1, 2)), np.zeros((1, 2)),
                  x + 2.0, 0.3)
    assert good[0][0] > bad[0][0]
    assert good[1][0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- diffusion machinery

def test_alpha_bar_decreases_from_near_one_to_near_zero():
    ab = attempt(drills.cosine_alpha_bar, 200)
    assert len(ab) == 200
    assert ab[0] > 0.99 and ab[-1] < 0.02
    assert np.all(np.diff(ab) < 0)


def test_betas_reproduce_their_alpha_bar():
    ab = attempt(drills.cosine_alpha_bar, 100)
    b = attempt(drills.betas_from_alpha_bar, ab)
    assert np.all((b > 0) & (b < 1))
    assert np.allclose(np.cumprod(1 - b), ab, atol=1e-6)


def test_forward_diffusion_preserves_scale():
    r = np.random.default_rng(0)
    x0 = r.normal(size=(20000, 2))
    for abar in (0.99, 0.5, 0.01):
        xt = attempt(drills.forward_diffuse, x0, abar, r.normal(size=(20000, 2)))
        assert xt.std() == pytest.approx(1.0, abs=0.03), \
            "the sqrt(abar) scaling is what keeps the variance bounded"


def test_eps_and_score_and_x0_are_the_same_information():
    r = np.random.default_rng(2)
    x0 = r.normal(size=(50, 2)); eps = r.normal(size=(50, 2)); abar = 0.4
    xt = attempt(drills.forward_diffuse, x0, abar, eps)
    assert np.allclose(attempt(drills.eps_to_x0, xt, eps, abar), x0, atol=1e-9)
    s = attempt(drills.eps_to_score, eps, abar)
    assert np.allclose(s, -eps / np.sqrt(1 - abar))


def test_ddpm_posterior_variance_vanishes_at_the_end():
    xt = np.zeros((1, 2)); eps = np.zeros((1, 2))
    _, v_mid = attempt(drills.ddpm_posterior, xt, eps, 0.02, 0.5, 0.51)
    _, v_end = attempt(drills.ddpm_posterior, xt, eps, 0.02, 0.98, 1.0)
    assert v_end < v_mid


def test_guidance_at_one_is_the_conditional_model():
    a = np.array([[1.0, 2.0]]); b = np.array([[0.5, -1.0]])
    assert np.allclose(attempt(drills.guided_eps, a, b, 1.0), a)
    assert np.allclose(attempt(drills.guided_eps, a, b, 0.0), b)
    far = attempt(drills.guided_eps, a, b, 8.0)
    assert np.linalg.norm(far - a) > np.linalg.norm(a - b), \
        "w > 1 extrapolates past the conditional prediction — section 30"


def test_langevin_climbs_toward_a_mode():
    r = np.random.default_rng(3)
    x = np.array([[3.0, 3.0]])
    for _ in range(400):
        s = attempt(drills.mixture_score, x, MUS, SDS, WS)
        x = attempt(drills.langevin_step, x, s, 0.01, r.normal(size=(1, 2)))
    assert np.linalg.norm(x - MUS, axis=1).min() < 1.0


# ------------------------------------------------------------------- the GAN

def test_optimal_discriminator_is_one_half_when_the_model_is_right():
    p = np.array([0.2, 0.5, 0.3])
    assert np.allclose(attempt(drills.optimal_discriminator, p, p), 0.5)
    d = attempt(drills.optimal_discriminator, np.array([0.9]), np.array([0.1]))
    assert d[0] == pytest.approx(0.9)


def test_jensen_shannon_is_symmetric_and_bounded():
    g = np.linspace(-6, 6, 801); dx = g[1] - g[0]
    p = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    q = np.exp(-0.5 * ((g - 2.0)) ** 2) / np.sqrt(2 * np.pi)
    a = attempt(drills.jensen_shannon, p, q, dx)
    b = attempt(drills.jensen_shannon, q, p, dx)
    assert a == pytest.approx(b, rel=1e-6), "JS is symmetric where KL is not"
    assert 0 <= a <= np.log(2) + 1e-9
    assert attempt(drills.jensen_shannon, p, p, dx) == pytest.approx(0.0, abs=1e-9)


def test_the_saturating_loss_has_a_vanishing_gradient():
    """Section 35's argument, checked numerically."""
    d = np.array([1e-6])
    h = 1e-9
    sat = lambda v: drills.generator_losses(np.array([v]))[0]
    non = lambda v: drills.generator_losses(np.array([v]))[1]
    attempt(drills.generator_losses, d)
    g_sat = abs((sat(d[0] + h) - sat(d[0] - h)) / (2 * h))
    g_non = abs((non(d[0] + h) - non(d[0] - h)) / (2 * h))
    assert g_non > 100 * g_sat, \
        f"non-saturating gradient {g_non:.3e} against saturating {g_sat:.3e}"


# ---------------------------------------------------------------- evaluation

def test_mode_weights_recover_a_known_mixture():
    r = np.random.default_rng(0)
    c = r.choice(3, 40000, p=WS)
    X = MUS[c] + r.normal(size=(40000, 2)) * SDS[c][:, None]
    w = attempt(drills.mode_weights, X, MUS)
    assert attempt(drills.total_variation, w, WS) < 0.02
    assert attempt(drills.modes_covered, X, MUS) == 3


def test_modes_covered_counts_a_dropped_mode():
    r = np.random.default_rng(1)
    c = r.choice(2, 4000, p=[0.5, 0.5])
    X = MUS[c] + r.normal(size=(4000, 2)) * SDS[c][:, None]
    assert attempt(drills.modes_covered, X, MUS) == 2, "the third mode is absent"


def test_mmd_is_near_zero_for_two_samples_of_the_same_thing():
    r = np.random.default_rng(2)
    A = r.normal(size=(400, 2)); B = r.normal(size=(400, 2))
    C = r.normal(size=(400, 2)) + 3.0
    same = attempt(drills.rbf_mmd, A, B)
    diff = attempt(drills.rbf_mmd, A, C)
    assert abs(same) < 0.01
    assert diff > 0.1


def test_precision_and_recall_separate_the_two_failures():
    """Section 43: the whole point is that one number cannot do this."""
    r = np.random.default_rng(3)
    c = r.choice(3, 900, p=WS)
    real = MUS[c] + r.normal(size=(900, 2)) * SDS[c][:, None]

    c2 = r.choice(2, 900, p=[0.5, 0.5])                 # drops the third mode
    dropper = MUS[c2] + r.normal(size=(900, 2)) * SDS[c2][:, None]
    p_drop, r_drop = attempt(drills.precision_recall_distributions, dropper, real)

    garbage = np.concatenate([real[:700], r.normal(scale=4.0, size=(200, 2))])
    p_junk, r_junk = attempt(drills.precision_recall_distributions, garbage, real)

    assert p_drop > 0.9, f"a mode-dropper makes only real things: {p_drop:.3f}"
    assert r_drop < 0.85, f"and misses part of the data: recall {r_drop:.3f}"
    assert p_junk < p_drop, f"the garbage model has worse precision: {p_junk:.3f}"
    assert r_junk > r_drop, f"and better recall: {r_junk:.3f}"


def test_memorisation_check_flags_a_copier():
    r = np.random.default_rng(4)
    train = r.normal(size=(300, 2))
    copier = train[r.integers(0, 300, 100)]
    honest = r.normal(size=(100, 2))
    d_copy = attempt(drills.nearest_training_distance, copier, train)
    d_real = attempt(drills.nearest_training_distance, honest, train)
    assert d_copy.max() < 1e-12
    assert d_real.mean() > 0.05
