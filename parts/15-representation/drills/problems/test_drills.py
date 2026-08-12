"""
Tests for the Part XV drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several tests assert a NEGATIVE result, because those are the part's findings:
a linear autoencoder does not recover PCA's axes, a probe's accuracy does not
establish that a representation encodes anything, and a neighbour embedding
does not preserve cluster sizes.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------- geometry

def test_cosine_matrix_basics():
    A = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    S = attempt(drills.cosine_matrix, A)
    assert S.shape == (3, 3)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[0, 1] == pytest.approx(0.0)
    assert S[2, 2] == pytest.approx(0.0), "a zero row must not produce nan"
    assert np.isfinite(S).all()


def test_anisotropy_detects_a_cone():
    r = np.random.default_rng(0)
    spread = r.normal(size=(2000, 32))
    cone = r.normal(size=(2000, 32)) * 0.3 + np.r_[4.0, np.zeros(31)]
    a_spread = attempt(drills.anisotropy, spread)
    a_cone = attempt(drills.anisotropy, cone)
    assert abs(a_spread) < 0.05, f"isotropic data should be near 0, got {a_spread:.3f}"
    assert a_cone > 0.8, f"a cone should be near 1, got {a_cone:.3f}"


def test_centering_removes_most_of_the_cone():
    r = np.random.default_rng(1)
    cone = r.normal(size=(2000, 32)) * 0.3 + np.r_[4.0, np.zeros(31)]
    Hc, Hw = attempt(drills.center_and_whiten, cone)
    assert abs(attempt(drills.anisotropy, Hc)) < 0.05
    assert abs(attempt(drills.anisotropy, Hw)) < 0.05
    C = np.cov(Hw, rowvar=False)
    assert np.abs(C - np.eye(32)).max() < 0.05, "whitened covariance must be ~I"


def test_knn_counts_average_to_k():
    r = np.random.default_rng(2)
    H = r.normal(size=(300, 8))
    c = attempt(drills.knn_counts, H, 10)
    assert c.sum() == 300 * 10
    assert c.mean() == pytest.approx(10.0)


def test_hubness_grows_with_dimension():
    """Section 26's figure, as an assertion."""
    r = np.random.default_rng(3)
    lo = attempt(drills.hubness_skew, r.normal(size=(1200, 2)), 10)
    hi = attempt(drills.hubness_skew, r.normal(size=(1200, 200)), 10)
    assert hi > lo + 2.0, f"skew {lo:.2f} at d=2 against {hi:.2f} at d=200"
    c = attempt(drills.knn_counts, r.normal(size=(1200, 200)), 10)
    assert c.max() > 50, f"expected a large hub, biggest was {c.max()}"
    assert (c == 0).mean() > 0.1, "many points should be nobody's neighbour"


def test_mutual_knn_is_symmetric_and_sparser():
    r = np.random.default_rng(4)
    H = r.normal(size=(200, 50))
    M = attempt(drills.mutual_knn, H, 10)
    assert np.array_equal(M, M.T)
    assert not M.diagonal().any()
    assert M.sum() < 200 * 10, "mutual kNN must drop the one-directional edges"


# ------------------------------------------------------------------- subspaces

def test_principal_angles_of_identical_and_orthogonal_spaces():
    A = np.eye(3, 6)
    assert attempt(drills.principal_angles, A, A).max() < 1e-3
    B = np.eye(6)[3:6]
    assert attempt(drills.principal_angles, A, B).min() > 89.9


def test_principal_angles_are_invariant_to_rotation_of_the_basis():
    """Section 8: a rotated basis spans the same subspace, so the angles are 0
    even though the rows do not match at all."""
    r = np.random.default_rng(5)
    A = r.normal(size=(3, 10))
    R = np.linalg.qr(r.normal(size=(3, 3)))[0]
    assert attempt(drills.principal_angles, A, R @ A).max() < 1e-3
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = (R @ A) / np.linalg.norm(R @ A, axis=1, keepdims=True)
    per_axis = np.abs(np.einsum("ij,ij->i", An, Bn))
    assert per_axis.min() < 0.95, "the AXES should not agree, only the subspace"


def test_pca_subspace_beats_a_random_one():
    r = np.random.default_rng(6)
    A = r.normal(size=(3, 12))
    X = (r.normal(size=(2000, 3)) * np.array([5.0, 2.0, 1.0])) @ A
    X += r.normal(scale=0.2, size=X.shape)
    W = attempt(drills.pca_subspace, X, 3)
    assert W.shape == (3, 12)
    assert np.abs(W @ W.T - np.eye(3)).max() < 1e-8, "rows must be orthonormal"
    e_pca = attempt(drills.subspace_reconstruction_error, X, W)
    e_rand = attempt(drills.subspace_reconstruction_error, X, r.normal(size=(3, 12)))
    assert e_pca < e_rand / 5
    assert attempt(drills.principal_angles, W, A).max() < 5.0


# --------------------------------------------------------------- spread of codes

def test_effective_rank_of_isotropic_and_collapsed():
    r = np.random.default_rng(7)
    iso = r.normal(size=(4000, 16))
    assert attempt(drills.effective_rank, iso) > 15.0
    line = r.normal(size=(4000, 1)) @ r.normal(size=(1, 16))
    assert attempt(drills.effective_rank, line) < 1.01
    assert attempt(drills.participation_ratio, line) < 1.01
    assert attempt(drills.participation_ratio, iso) > 14.0


def test_collapse_report_flags_a_collapsed_code():
    r = np.random.default_rng(8)
    line = r.normal(size=(3000, 1)) @ r.normal(size=(1, 32))
    rep = attempt(drills.collapse_report, line + r.normal(scale=1e-3, size=(3000, 32)))
    assert rep["top_eigenvalue_share"] > 0.99
    assert rep["effective_rank"] < 1.1


def test_a_probe_can_succeed_on_a_nearly_collapsed_code():
    """Section 20's finding: near-total collapse with a probe still at 1.0.

    One dominant direction carries 99.9% of the variance, and a small
    perfectly-separable component survives underneath it. Effective rank says
    the representation is one-dimensional; the probe says it is perfect.
    """
    r = np.random.default_rng(9)
    y = r.integers(0, 4, 1500)
    sep = np.eye(4)[y] * 1.0                        # separable, ordinary magnitude
    big = (r.normal(size=(1500, 1)) @ r.normal(size=(1, 4))) * 12.0
    H = np.concatenate([sep + big, r.normal(scale=1e-3, size=(1500, 8))], 1)
    rep = attempt(drills.collapse_report, H)
    acc = attempt(drills.linear_probe_accuracy, H[:1000], y[:1000], H[1000:], y[1000:])
    assert rep["top_eigenvalue_share"] > 0.99
    assert rep["effective_rank"] < 1.1, f"rank {rep['effective_rank']:.3f}"
    assert acc > 0.9, f"probe still reads {acc:.3f} off a collapsed code"


def test_whether_the_probe_sees_through_collapse_depends_on_scale():
    """The same structure with the signal at a tiny ABSOLUTE magnitude: now the
    probe's L2 penalty forbids the large weights needed to amplify it, and the
    probe fails. Collapse and probe accuracy are related through the
    regularization of the probe, which is a hyperparameter nobody reports."""
    r = np.random.default_rng(9)
    y = r.integers(0, 4, 1500)
    sep = np.eye(4)[y] * 0.01                       # same signal, 100x smaller
    big = (r.normal(size=(1500, 1)) @ r.normal(size=(1, 4))) * 50.0
    H = np.concatenate([sep + big, r.normal(scale=1e-4, size=(1500, 8))], 1)
    weak = attempt(drills.linear_probe_accuracy, H[:1000], y[:1000],
                   H[1000:], y[1000:], C=1.0)
    strong = attempt(drills.linear_probe_accuracy, H[:1000], y[:1000],
                     H[1000:], y[1000:], C=1e6)
    assert weak < 0.6, f"a regularized probe misses it: {weak:.3f}"
    assert strong > 0.9, f"an unregularized one finds it: {strong:.3f}"


# ----------------------------------------------------------------- contrastive

def test_info_nce_is_minimised_by_matched_pairs():
    r = np.random.default_rng(10)
    Z = r.normal(size=(64, 16))
    perfect = attempt(drills.info_nce, Z, Z.copy())
    shuffled = attempt(drills.info_nce, Z, Z[r.permutation(64)])
    assert perfect < shuffled
    assert perfect >= 0.0


def test_info_nce_is_stable_at_small_temperature():
    r = np.random.default_rng(11)
    Z = r.normal(size=(32, 8))
    v = attempt(drills.info_nce, Z, Z.copy(), tau=1e-3)
    assert np.isfinite(v), "a naive exp() overflows here — use log-sum-exp"


def test_collapse_wins_on_alignment_and_loses_on_uniformity():
    r = np.random.default_rng(12)
    Z1 = r.normal(size=(200, 16))
    Z2 = Z1 + r.normal(scale=0.1, size=(200, 16))
    const = np.ones((200, 16)) + r.normal(scale=1e-6, size=(200, 16))
    assert attempt(drills.alignment, const, const) < attempt(drills.alignment, Z1, Z2)
    assert attempt(drills.uniformity, const) > attempt(drills.uniformity, Z1), \
        "a collapsed code is maximally NON-uniform, so its uniformity score is higher"


# --------------------------------------------------------------------- probing

def test_control_labels_are_consistent_per_identity():
    ids = np.array([3, 1, 3, 7, 1, 3])
    a = attempt(drills.control_labels, ids, 5, seed=0)
    b = attempt(drills.control_labels, ids, 5, seed=0)
    assert np.array_equal(a, b)
    assert a[0] == a[2] == a[5], "the same identity must get the same label"
    assert a[1] == a[4]


def test_selectivity_is_high_for_a_real_signal():
    r = np.random.default_rng(13)
    y = r.integers(0, 4, 2000)
    ident = r.integers(0, 40, 2000)
    H = np.eye(4)[y] * 3.0 + r.normal(scale=0.3, size=(2000, 4))
    j = attempt(drills.control_labels, ident, 4, seed=0)
    s = attempt(drills.selectivity, H[:1000], y[:1000], H[1000:], y[1000:],
                j[:1000], j[1000:])
    assert s > 0.5, f"selectivity {s:.3f}"


def test_selectivity_is_near_zero_when_the_probe_can_memorise():
    """Section 32: an over-powerful probe scores well on the control task, and
    its accuracy on the real task then establishes nothing."""
    r = np.random.default_rng(14)
    ident = r.integers(0, 30, 1600)
    y = ident % 4
    H = np.eye(30)[ident] * 5.0                     # identity is fully recoverable
    j = attempt(drills.control_labels, ident, 4, seed=1)
    real = attempt(drills.linear_probe_accuracy, H[:800], y[:800], H[800:], y[800:])
    ctrl = attempt(drills.linear_probe_accuracy, H[:800], j[:800], H[800:], j[800:])
    assert real > 0.95 and ctrl > 0.95
    assert abs(real - ctrl) < 0.1, \
        f"real {real:.3f} and control {ctrl:.3f}: accuracy proves nothing here"


# ------------------------------------------------------------ denoising & score

def test_mixture_score_matches_a_finite_difference():
    mus = np.array([[-1.0, 0.3], [1.2, -0.4]])
    sds = np.array([0.5, 0.35])
    ws = np.array([0.6, 0.4])

    def logp(P):
        tot = 0.0
        for m, sd, w in zip(mus, sds, ws):
            d2 = ((P - m) ** 2).sum(-1)
            tot = tot + w * np.exp(-d2 / (2 * sd ** 2)) / (2 * np.pi * sd ** 2)
        return np.log(tot)

    P = np.array([[0.2, 0.1], [-0.9, 0.5], [1.4, -0.2]])
    s = attempt(drills.gaussian_mixture_score, P, mus, sds, ws)
    eps = 1e-5
    fd = np.zeros_like(P)
    for k in range(2):
        e = np.zeros(2); e[k] = eps
        fd[:, k] = (logp(P + e) - logp(P - e)) / (2 * eps)
    assert np.abs(s - fd).max() < 1e-4


def test_smoothing_flattens_the_score():
    mus = np.array([[-1.0, 0.0], [1.0, 0.0]])
    sds = np.array([0.4, 0.4])
    ws = np.array([0.5, 0.5])
    P = np.array([[0.6, 0.3], [-0.7, -0.2], [1.5, 0.4]])
    s0 = attempt(drills.gaussian_mixture_score, P, mus, sds, ws, 0.0)
    sm = attempt(drills.gaussian_mixture_score, P, mus, sds, ws, 0.5 ** 2)
    assert np.linalg.norm(sm, axis=1).mean() < np.linalg.norm(s0, axis=1).mean(), \
        "a blurred density has a smaller gradient — section 12's pitfall"


def test_denoiser_recovers_the_score_of_the_smoothed_density():
    """The identity of section 12, checked against the exact answer using the
    OPTIMAL denoiser rather than a trained one."""
    mus = np.array([[-1.2, 0.0], [1.1, 0.5]])
    sds = np.array([0.45, 0.35])
    ws = np.array([0.5, 0.5])
    sigma = 0.4

    def optimal_reconstruct(X):
        # E[x_clean | x_noisy] = x + sigma^2 * score of the smoothed density
        return X + sigma ** 2 * drills.gaussian_mixture_score(
            X, mus, sds, ws, sigma ** 2)

    r = np.random.default_rng(15)
    X = r.uniform(-2, 2, size=(400, 2))
    est = attempt(drills.score_from_denoiser, optimal_reconstruct, X, sigma)
    truth = attempt(drills.gaussian_mixture_score, X, mus, sds, ws, sigma ** 2)
    cos, ratio = attempt(drills.field_agreement, est, truth)
    assert cos > 0.999 and ratio == pytest.approx(1.0, abs=1e-6)

    # against the UNSMOOTHED score the direction survives and the scale does not
    raw = attempt(drills.gaussian_mixture_score, X, mus, sds, ws, 0.0)
    cos_raw, ratio_raw = attempt(drills.field_agreement, est, raw)
    assert cos_raw > 0.9
    assert ratio_raw < 0.9, f"magnitude ratio against the raw score {ratio_raw:.3f}"


def test_field_agreement_separates_direction_from_scale():
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    cos, ratio = attempt(drills.field_agreement, A, A * 3.0)
    assert cos == pytest.approx(1.0)
    assert ratio == pytest.approx(1 / 3)


# ---------------------------------------------------------------- sparse codes

def test_soft_threshold():
    v = np.array([-2.0, -0.3, 0.0, 0.4, 5.0])
    got = attempt(drills.soft_threshold, v, 0.5)
    assert np.allclose(got, [-1.5, 0.0, 0.0, 0.0, 4.5])


def test_sparse_code_is_sparse_and_reconstructs():
    r = np.random.default_rng(16)
    k, d = 12, 20
    D = r.normal(size=(k, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    h_true = np.zeros(k); h_true[[2, 7]] = [1.5, -2.0]
    x = D.T @ h_true
    h = attempt(drills.sparse_code, x, D, lam=0.05)
    assert (np.abs(h) > 1e-6).sum() <= 5, f"{(np.abs(h) > 1e-6).sum()} nonzeros"
    assert np.linalg.norm(D.T @ h - x) < 0.2 * np.linalg.norm(x)
    assert set(np.argsort(-np.abs(h))[:2].tolist()) == {2, 7}


def test_more_penalty_means_fewer_nonzeros():
    r = np.random.default_rng(17)
    D = r.normal(size=(15, 25)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    x = D.T @ (r.normal(size=15) * (r.random(15) < 0.3))
    counts = [int((np.abs(attempt(drills.sparse_code, x, D, lam=l)) > 1e-6).sum())
              for l in (0.01, 0.2, 1.0)]
    assert counts[0] >= counts[1] >= counts[2]


def test_dictionary_recovery_is_blind_to_sign_and_order():
    r = np.random.default_rng(18)
    D = r.normal(size=(6, 10))
    perm = r.permutation(6)
    flipped = D[perm] * np.where(r.random((6, 1)) < 0.5, -1.0, 1.0)
    assert attempt(drills.dictionary_recovery, D, flipped).min() > 0.999


def test_interference_is_zero_for_orthogonal_features():
    W = np.eye(5)
    assert attempt(drills.feature_interference, W) == pytest.approx(0.0)
    W2 = np.tile(np.eye(5)[:, :1], (1, 4))          # four copies of one direction
    assert attempt(drills.feature_interference, W2) == pytest.approx(1.0)


def test_features_stored_counts_surviving_columns():
    W = np.zeros((3, 6))
    W[:, :4] = np.random.default_rng(19).normal(size=(3, 4))
    W[:, :4] /= np.linalg.norm(W[:, :4], axis=0, keepdims=True)
    assert attempt(drills.features_stored, W) == 4


# ------------------------------------ movement G: learning as compression

def test_the_gaussian_rate_distortion_function():
    assert attempt(drills.rate_distortion_gaussian, 1.0) == pytest.approx(0.0)
    assert attempt(drills.rate_distortion_gaussian, 0.25) == pytest.approx(1.0)
    assert attempt(drills.rate_distortion_gaussian, 1.0 / 64) == pytest.approx(3.0)
    assert attempt(drills.rate_distortion_gaussian, 2.0) == pytest.approx(0.0)


def test_each_bit_quarters_the_distortion():
    for r in (0.0, 1.0, 2.5, 6.0):
        d0 = attempt(drills.distortion_at_rate, r)
        d1 = attempt(drills.distortion_at_rate, r + 1)
        assert d1 == pytest.approx(d0 / 4)


def test_rate_and_distortion_invert_each_other():
    for d in (0.5, 0.1, 0.01):
        r = attempt(drills.rate_distortion_gaussian, d)
        assert attempt(drills.distortion_at_rate, r) == pytest.approx(d)


def test_entropy_bits():
    assert attempt(drills.entropy_bits, [1, 1]) == pytest.approx(1.0)
    assert attempt(drills.entropy_bits, [1, 1, 1, 1]) == pytest.approx(2.0)
    assert attempt(drills.entropy_bits, [1, 0, 0]) == pytest.approx(0.0)
    assert attempt(drills.entropy_bits, [3, 1]) == pytest.approx(0.8112781, abs=1e-6)


def test_uniform_quantisation_error_is_bounded_by_half_a_step():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    for step in (1.0, 0.3, 0.05):
        q, idx = attempt(drills.uniform_quantise, x, step)
        assert np.all(np.abs(x - q) <= step / 2 + 1e-12)
        assert np.allclose(q, idx * step)


def test_uniform_quantisation_distortion_approaches_step_squared_over_twelve():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400000)
    for step in (0.2, 0.1, 0.05):
        q, _ = attempt(drills.uniform_quantise, x, step)
        d = float(np.mean((x - q) ** 2))
        assert d == pytest.approx(step ** 2 / 12, rel=0.05)


def test_a_real_quantiser_never_beats_the_bound():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400000)
    for step in (0.5, 0.2, 0.08):
        q, idx = attempt(drills.uniform_quantise, x, step)
        d = float(np.mean((x - q) ** 2))
        _, counts = np.unique(idx, return_counts=True)
        r = attempt(drills.entropy_bits, counts)
        assert attempt(drills.excess_rate, r, d) > 0


def test_the_excess_converges_to_the_space_filling_constant():
    """0.5 log2(pi e / 6) = 0.2546 bits. The one place in this part where a
    measurement lands on a closed form to three decimals."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=600000)
    gaps = []
    for step in (0.08, 0.05, 0.03):
        q, idx = attempt(drills.uniform_quantise, x, step)
        d = float(np.mean((x - q) ** 2))
        _, counts = np.unique(idx, return_counts=True)
        gaps.append(attempt(drills.excess_rate, attempt(drills.entropy_bits, counts), d))
    assert np.mean(gaps) == pytest.approx(attempt(drills.scalar_space_filling_loss), abs=0.01)
    assert attempt(drills.scalar_space_filling_loss) == pytest.approx(0.2546, abs=1e-4)


def test_the_residual_code_shrinks_with_the_residuals():
    small = attempt(drills.gaussian_code_bits, np.full(100, 0.01))
    large = attempt(drills.gaussian_code_bits, np.full(100, 1.0))
    assert small < large


def test_the_parameter_code_grows_with_both_arguments():
    assert attempt(drills.parameter_bits, 4, 100) > attempt(drills.parameter_bits, 2, 100)
    assert attempt(drills.parameter_bits, 4, 1000) > attempt(drills.parameter_bits, 4, 100)
    assert attempt(drills.parameter_bits, 2, 64) == pytest.approx(6.0)


def test_the_description_length_has_an_interior_minimum():
    rng = np.random.default_rng(4)
    n = 60
    x = np.sort(rng.uniform(-1, 1, n))
    y = np.polyval([0.7, -1.1, 0.4, 0.2][::-1], x) + rng.normal(scale=0.3, size=n)
    lens = []
    for deg in range(0, 9):
        A = np.vander(x, deg + 1, increasing=True)
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        lens.append(attempt(drills.description_length, y - A @ w, deg + 2))
    j = int(np.argmin(lens))
    assert 0 < j < 8, f"minimum at an endpoint: {j}"


def test_residual_bits_alone_prefer_a_more_complex_model():
    """Which is the point of charging for parameters: without that term the
    criterion is a fitting criterion, not a selection criterion."""
    rng = np.random.default_rng(5)
    n = 60
    x = np.sort(rng.uniform(-1, 1, n))
    y = np.polyval([0.7, -1.1, 0.4, 0.2][::-1], x) + rng.normal(scale=0.3, size=n)
    resid_only, total = [], []
    for deg in range(0, 9):
        A = np.vander(x, deg + 1, increasing=True)
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ w
        resid_only.append(attempt(drills.gaussian_code_bits, r))
        total.append(attempt(drills.description_length, r, deg + 2))
    assert int(np.argmin(resid_only)) > int(np.argmin(total))


def test_mutual_information_of_independent_variables_is_zero():
    px = np.array([0.3, 0.7])
    py = np.array([0.2, 0.5, 0.3])
    assert attempt(drills.mutual_information_bits, np.outer(px, py)) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_of_a_perfect_pairing():
    p = np.eye(4) / 4
    assert attempt(drills.mutual_information_bits, p) == pytest.approx(2.0)


def test_the_bottleneck_objective_trades_the_two_terms():
    assert attempt(drills.bottleneck_objective, 3.0, 1.0, 0.0) == pytest.approx(3.0)
    assert attempt(drills.bottleneck_objective, 3.0, 1.0, 2.0) == pytest.approx(1.0)
    # at large beta, keeping more about X is worth it if it buys anything about Y
    lo = attempt(drills.bottleneck_objective, 1.0, 0.2, 20.0)
    hi = attempt(drills.bottleneck_objective, 3.0, 1.0, 20.0)
    assert hi < lo
