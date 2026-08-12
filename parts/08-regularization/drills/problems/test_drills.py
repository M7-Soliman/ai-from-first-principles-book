"""
Tests for the Part VIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

The load-bearing tests are the equivalences. Each produces both sides
independently and asserts they agree to a stated tolerance -- and where the
book says the agreement is only approximate (early stopping), the test asserts
that too, rather than pretending otherwise.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


rng = np.random.default_rng(0)


def ill_conditioned(n=60, d=25, noise=0.4, seed=0):
    """A design whose eigenvalues span several decades -- where the whole
    story of §4 is visible."""
    r = np.random.default_rng(seed)
    scale = np.logspace(0.8, -1.6, d)
    X = r.normal(size=(n, d)) * scale
    w = r.normal(size=d)
    return X, X @ w + r.normal(0, noise, n), w


# ------------------------------------------------------------- penalties ----

def test_ridge_at_zero_is_least_squares():
    X, y, _ = ill_conditioned(n=80, d=10)
    a = attempt(drills.ridge_weights, X, y, 0.0)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(a, b, atol=1e-6)


def test_ridge_shrinks_monotonically():
    X, y, _ = ill_conditioned()
    norms = [np.linalg.norm(attempt(drills.ridge_weights, X, y, a))
             for a in (0.0, 0.1, 1.0, 10.0, 100.0)]
    assert all(x > z for x, z in zip(norms, norms[1:]))


def test_kept_fraction_matches_the_prediction():
    """§4's derivation, measured. The decisive test of Part B."""
    X, y, _ = ill_conditioned()
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    for alpha in (0.05, 1.0, 50.0):
        w_r = attempt(drills.ridge_weights, X, y, alpha)
        lam, frac = attempt(drills.kept_fraction, X, w_r, w_ols)
        pred = attempt(drills.predicted_kept_fraction, lam, alpha)
        assert np.allclose(frac, pred, atol=1e-6), \
            f"at alpha={alpha}, measured shrinkage does not match lam/(lam+alpha)"


def test_kept_fraction_is_ordered_by_curvature():
    X, y, _ = ill_conditioned()
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    lam, frac = attempt(drills.kept_fraction, X,
                        attempt(drills.ridge_weights, X, y, 1.0), w_ols)
    order = np.argsort(lam)
    assert frac[order][0] < frac[order][-1], "flat directions must lose more"
    assert frac[order][-1] > 0.5 > frac[order][0]


def test_lasso_produces_exact_zeros_and_ridge_does_not():
    r = np.random.default_rng(3)
    X = r.normal(size=(80, 8))
    w_true = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
    y = X @ w_true + r.normal(0, 0.4, 80)

    w_l1 = attempt(drills.lasso_weights, X, y, 2.0)
    w_l2 = attempt(drills.ridge_weights, X, y, 2.0 * len(y))

    n_zero_l1 = int(np.sum(np.abs(w_l1) < 1e-12))
    n_zero_l2 = int(np.sum(np.abs(w_l2) < 1e-12))
    assert n_zero_l1 >= 3, f"L1 should zero the useless features, got {n_zero_l1}"
    assert n_zero_l2 == 0, "L2 shrinks but never reaches exactly zero"


def test_lasso_keeps_the_real_features():
    r = np.random.default_rng(4)
    X = r.normal(size=(120, 6))
    w_true = np.array([4.0, -3.0, 0, 0, 0, 0])
    y = X @ w_true + r.normal(0, 0.3, 120)
    w = attempt(drills.lasso_weights, X, y, 1.0)
    assert np.all(np.abs(w[:2]) > 1.0), "signal-carrying features must survive"
    assert np.all(np.abs(w[2:]) < 0.5), "noise features must be crushed"


# ------------------------------------------------------- the equivalences ---

def test_effective_alpha_from_noise():
    assert attempt(drills.effective_alpha_from_noise, 500, 0.2) == pytest.approx(20.0)


def test_input_noise_equals_weight_decay():
    """§12's derivation, measured. Figure 4's claim."""
    X, y, _ = ill_conditioned(n=150, d=10, noise=0.3, seed=7)
    sigma = 0.35
    w_noise = attempt(drills.noise_augmented_fit, X, y, sigma, reps=120, seed=1)
    alpha = attempt(drills.effective_alpha_from_noise, len(y), sigma)
    w_ridge = attempt(drills.ridge_weights, X, y, alpha)
    rel = np.linalg.norm(w_noise - w_ridge) / np.linalg.norm(w_ridge)
    assert rel < 0.10, f"noise and ridge should nearly coincide, got {rel:.3f}"


def test_more_noise_means_more_shrinkage():
    X, y, _ = ill_conditioned(n=150, d=10, seed=8)
    norms = [np.linalg.norm(attempt(drills.noise_augmented_fit, X, y, s,
                                    reps=60, seed=2))
             for s in (0.0, 0.3, 0.8)]
    assert norms[0] > norms[1] > norms[2]


def test_early_stopping_kept_fraction_formula():
    lam = np.array([0.01, 1.0, 10.0])
    lr, steps = 0.01, 100
    got = attempt(drills.early_stopping_kept_fraction, lam, lr, steps)
    assert np.allclose(got, 1 - (1 - lr * lam) ** steps)
    assert got[0] < got[-1], "sharp directions are learned first"
    assert 0 <= got.min() and got.max() <= 1


def test_gd_converges_to_least_squares_eventually():
    """On a WELL-conditioned problem. On the ill-conditioned one the slowest
    mode needs about 1/(lr*lam_min) steps -- 74,000 for that design -- which is
    itself §14's point: flat directions are learned last."""
    r = np.random.default_rng(9)
    X = r.normal(size=(60, 8))                  # kappa of order 1
    y = X @ r.normal(size=8) + r.normal(0, 0.3, 60)
    lam = np.linalg.eigvalsh(X.T @ X)
    lr = 0.9 / lam.max()
    w_gd = attempt(drills.gd_path, X, y, lr, 20000)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w_gd, w_ols, rtol=0.02), "no stopping means no regularization"


def test_flat_directions_are_learned_last():
    """Why the test above needs a well-conditioned problem."""
    X, y, _ = ill_conditioned(n=60, d=25, seed=9)
    lam = np.linalg.eigvalsh(X.T @ X)
    lr = 0.9 / lam.max()
    frac = attempt(drills.early_stopping_kept_fraction, lam, lr, 300)
    order = np.argsort(lam)
    assert frac[order][-1] > 0.9, "the sharpest direction is nearly done"
    assert frac[order][0] < 0.1, "the flattest has barely started"


def test_early_stopping_tracks_ridge_but_not_exactly():
    """§14's honest claim: same family, same order, NOT the same point."""
    X, y, _ = ill_conditioned(n=60, d=25, seed=11)
    lam = np.linalg.eigvalsh(X.T @ X)
    lr = 0.9 / lam.max()
    steps = 300
    w_gd = attempt(drills.gd_path, X, y, lr, steps)
    alpha = attempt(drills.alpha_for_stopping, lr, steps)
    w_r = attempt(drills.ridge_weights, X, y, alpha)

    rel = np.linalg.norm(w_gd - w_r) / np.linalg.norm(w_r)
    assert rel < 0.60, f"they should be in the same neighbourhood, got {rel:.2f}"
    assert rel > 1e-3, ("they should NOT be identical -- if you got exact "
                        "agreement, check you are not computing ridge twice")


def test_stopping_earlier_is_a_stronger_penalty():
    X, y, _ = ill_conditioned(n=60, d=25, seed=12)
    lam = np.linalg.eigvalsh(X.T @ X)
    lr = 0.9 / lam.max()
    norms = [np.linalg.norm(attempt(drills.gd_path, X, y, lr, s))
             for s in (10, 100, 1000, 10000)]
    assert all(a < b for a, b in zip(norms, norms[1:])), \
        "training longer must grow the weights -- that is why stopping regularizes"
    a_early = attempt(drills.alpha_for_stopping, lr, 10)
    a_late = attempt(drills.alpha_for_stopping, lr, 10000)
    assert a_early > a_late


# -------------------------------------------------------------- ensembles ---

def test_ensemble_error_extremes():
    v = 4.0
    assert attempt(drills.ensemble_error, v, v, 10) == pytest.approx(v), \
        "perfectly correlated members: no benefit at all"
    assert attempt(drills.ensemble_error, v, 0.0, 10) == pytest.approx(v / 10), \
        "uncorrelated members: error falls like 1/k"
    mid = attempt(drills.ensemble_error, v, v / 2, 10)
    assert v / 10 < mid < v


def test_ensemble_error_floor():
    """The covariance term is a floor no ensemble size can beat."""
    v, c = 4.0, 1.0
    errs = [attempt(drills.ensemble_error, v, c, k) for k in (1, 10, 100, 10000)]
    assert all(a >= b for a, b in zip(errs, errs[1:]))
    assert errs[-1] > c * 0.99, "cannot get below the shared error"


def _bag_setup():
    grid = np.linspace(0.1, 0.9, 40).reshape(-1, 1)

    def fit_predict(Xtr, ytr, Xte):
        return np.polyval(np.polyfit(Xtr.ravel(), ytr, 6), Xte.ravel())

    def sample(seed):
        r = np.random.default_rng(seed)
        X = r.uniform(0, 1, (40, 1))
        return X, np.sin(2 * np.pi * X.ravel()) + r.normal(0, 0.35, 40)
    return grid, fit_predict, sample


def test_bagging_reduces_variance_relative_to_its_members():
    """The comparison the theory makes: an ensemble against ONE of its own
    members. Both are fitted on bootstrap resamples, so the only difference
    is the averaging."""
    grid, fit_predict, sample = _bag_setup()
    one, many = [], []
    for t in range(120):
        X, y = sample(t)
        one.append(attempt(drills.bag_predict, fit_predict, X, y, grid, k=1, seed=t))
        many.append(attempt(drills.bag_predict, fit_predict, X, y, grid, k=20, seed=t))
    v_one = np.var(np.array(one), axis=0).mean()
    v_many = np.var(np.array(many), axis=0).mean()
    assert v_many < v_one / 2, \
        f"averaging 20 members should cut variance a lot: {v_one:.4f} -> {v_many:.4f}"


def test_bootstrap_itself_costs_the_members_accuracy():
    """The caveat §18 makes. A bootstrap sample has only ~63% unique points,
    so each member is WORSE than a fit on all the data -- and for an unstable
    model that cost can exceed what the averaging recovers."""
    grid, fit_predict, sample = _bag_setup()
    full, one = [], []
    for t in range(120):
        X, y = sample(t)
        full.append(fit_predict(X, y, grid))
        one.append(attempt(drills.bag_predict, fit_predict, X, y, grid, k=1, seed=t))
    v_full = np.var(np.array(full), axis=0).mean()
    v_one = np.var(np.array(one), axis=0).mean()
    assert v_one > v_full, \
        "a single bootstrap member should be MORE variable than a full-data fit"


# ---------------------------------------------------------------- dropout ---

def test_dropout_mask_preserves_the_expectation():
    r = np.random.default_rng(0)
    m = attempt(drills.dropout_mask, (200000,), 0.5, r)
    assert abs(m.mean() - 1.0) < 0.02, \
        "inverted dropout must have expectation 1, or activations shift at test time"
    kept = (m > 0).mean()
    assert abs(kept - 0.5) < 0.02
    assert np.allclose(m[m > 0], 2.0), "survivors scale by 1/(1-p)"


def test_dropout_actually_deletes_units():
    r = np.random.default_rng(1)
    X = np.ones((8, 4))
    W = np.ones((16, 4)); b = np.zeros(16)
    a, m = attempt(drills.dropout_forward, X, W, b, 0.5, r)
    assert a.shape == (8, 16)
    assert (a == 0).any(), "some units must be dropped"
    assert (m == 0).sum() == (a == 0).sum()


def test_dropout_zero_probability_is_a_no_op():
    r = np.random.default_rng(2)
    X = np.random.default_rng(3).normal(size=(5, 4))
    W = np.random.default_rng(4).normal(size=(6, 4)); b = np.zeros(6)
    a, m = attempt(drills.dropout_forward, X, W, b, 0.0, r)
    assert np.allclose(m, 1.0)
    assert np.allclose(a, np.maximum(X @ W.T + b, 0))


# ------------------------------------------------------------ adversarial ---

def _fit_logistic(X, y, steps=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= lr * X.T @ (p - y) / len(y)
    return w


def test_adversarial_beats_random_at_the_same_budget():
    """Figure 8's claim, and the whole point of §20."""
    r = np.random.default_rng(7)
    n, d = 400, 64
    w_true = r.normal(size=d)
    X = r.normal(size=(n, d))
    y = (X @ w_true > 0).astype(float)
    w = _fit_logistic(X, y)
    clean = np.mean(((X @ w) > 0) == (y > 0))
    assert clean > 0.95

    eps = 0.15
    Xa = attempt(drills.fgsm_perturbation, w, X, y, eps)
    Xr = attempt(drills.random_perturbation, X, eps, r)
    adv = np.mean(((Xa @ w) > 0) == (y > 0))
    rnd = np.mean(((Xr @ w) > 0) == (y > 0))

    assert adv < 0.75, f"the gradient direction should do real damage, got {adv:.3f}"
    assert rnd > 0.90, f"random noise of the same size should not, got {rnd:.3f}"
    assert rnd - adv > 0.2, "the two directions are not remotely equivalent"


def test_perturbations_respect_the_budget():
    r = np.random.default_rng(8)
    X = r.normal(size=(20, 10))
    y = (X[:, 0] > 0).astype(float)
    w = r.normal(size=10)
    for f, args in ((drills.fgsm_perturbation, (w, X, y, 0.1)),
                    (drills.random_perturbation, (X, 0.1, r))):
        out = attempt(f, *args)
        assert np.allclose(np.abs(out - X), 0.1), \
            "every coordinate must move by exactly eps -- same budget for both"


def test_larger_epsilon_does_more_damage():
    r = np.random.default_rng(9)
    n, d = 300, 40
    w_true = r.normal(size=d)
    X = r.normal(size=(n, d))
    y = (X @ w_true > 0).astype(float)
    w = _fit_logistic(X, y)
    accs = []
    for e in (0.0, 0.05, 0.15, 0.4):
        Xa = attempt(drills.fgsm_perturbation, w, X, y, e)
        accs.append(np.mean(((Xa @ w) > 0) == (y > 0)))
    assert all(a >= b for a, b in zip(accs, accs[1:])), "monotone in eps"
