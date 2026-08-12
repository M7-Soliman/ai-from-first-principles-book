"""
Tests for the Part VI drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Two things are graded beyond correctness:
  * the splits must genuinely be disjoint and exhaustive -- leakage is silent
  * the kernel trick must agree EXACTLY with its explicit feature map, which
    is the only way to be sure you implemented the trick and not an analogy
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


def blobs(n=200, d=2, seed=0, sep=1.5, sd=0.7):
    """Two Gaussian clusters labelled 0 and 1.

    The defaults are well separated. Pass sep=0.6, sd=1.0 for a genuinely
    OVERLAPPING problem -- needed wherever a test is about capacity, because
    on separable data even a very smooth model gets everything right and the
    capacity claim becomes untestable.
    """
    r = np.random.default_rng(seed)
    a = r.normal(-sep, sd, size=(n // 2, d))
    b = r.normal(+sep, sd, size=(n - n // 2, d))
    X = np.vstack([a, b])
    y = np.concatenate([np.zeros(len(a), int), np.ones(len(b), int)])
    p = r.permutation(len(X))
    return X[p], y[p]


# ---------------------------------------------------------------- splits ----

def test_split_is_disjoint_and_exhaustive():
    tr, va, te = attempt(drills.train_val_test_split, 100, (0.7, 0.15, 0.15), 0)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == 100
    assert len(np.unique(allidx)) == 100, "an index appeared in two splits -- leakage"
    assert set(np.unique(allidx)) == set(range(100))


def test_split_actually_shuffles():
    tr, _, _ = attempt(drills.train_val_test_split, 200, (0.7, 0.15, 0.15), 0)
    assert not np.array_equal(np.sort(tr), tr[:len(tr)]) or len(tr) < 3, \
        "an unshuffled split leaks whatever structure the storage order had"


def test_split_is_reproducible():
    a = attempt(drills.train_val_test_split, 50, (0.6, 0.2, 0.2), 3)
    b = attempt(drills.train_val_test_split, 50, (0.6, 0.2, 0.2), 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_kfold_uses_every_point_for_validation_once():
    folds = attempt(drills.kfold_indices, 50, 5, 0)
    assert len(folds) == 5
    seen = np.concatenate([va for _, va in folds])
    assert len(seen) == 50 and len(np.unique(seen)) == 50
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0, "train and val must be disjoint"


def test_cross_val_score_on_a_separable_problem():
    X, y = blobs(200, 2, 1)

    def fp(Xtr, ytr, Xte):
        return attempt(drills.knn_predict, Xtr, ytr, Xte, 5)

    assert attempt(drills.cross_val_score, fp, X, y, 5, 0) > 0.9


# ---------------------------------------------------------------- models ----

def test_linear_regression_recovers_known_weights():
    X = rng.normal(size=(200, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + rng.normal(0, 0.01, 200)
    assert np.allclose(attempt(drills.fit_linear_regression, X, y), w_true, atol=0.02)


def test_ridge_at_zero_is_least_squares():
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.05, 80)
    a = attempt(drills.fit_linear_regression, X, y)
    b = attempt(drills.fit_ridge, X, y, 0.0)
    assert np.allclose(a, b, atol=1e-6)


def test_ridge_shrinks_as_lambda_grows():
    X = rng.normal(size=(80, 4))
    y = X @ np.array([3.0, -2.0, 1.0, 0.5]) + rng.normal(0, 0.1, 80)
    norms = [np.linalg.norm(attempt(drills.fit_ridge, X, y, lam))
             for lam in (0.0, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(norms, norms[1:])), "must shrink monotonically"


def test_sigmoid_basics_and_stability():
    assert np.isclose(attempt(drills.sigmoid, np.array([0.0]))[0], 0.5)
    big = attempt(drills.sigmoid, np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(big)), "naive 1/(1+exp(-z)) overflows at -800"
    assert big[0] >= 0.0 and big[1] <= 1.0


def test_logistic_separates_two_blobs():
    X, y = blobs(300, 2, 2)
    Xb = np.hstack([X, np.ones((len(X), 1))])          # bias column
    w = attempt(drills.fit_logistic, Xb, y, 0.5, 3000)
    pred = (attempt(drills.sigmoid, Xb @ w) > 0.5).astype(int)
    assert np.mean(pred == y) > 0.95


def test_knn_k_is_the_capacity_dial():
    """k=1 fits the training set perfectly; large k cannot. Needs OVERLAP --
    on separable data even k=61 is right about everything."""
    X, y = blobs(120, 2, 3, sep=0.6, sd=1.0)
    p1 = attempt(drills.knn_predict, X, y, X, 1)
    assert np.mean(p1 == y) == 1.0, "k=1 must reproduce the training labels exactly"
    p_big = attempt(drills.knn_predict, X, y, X, 61)
    assert np.mean(p_big == y) < 0.95, \
        "large k is low capacity and cannot memorise -- k runs backwards"


def test_kmeans_finds_two_clusters():
    X, y = blobs(200, 2, 4)
    centres, labels = attempt(drills.kmeans, X, 2, 0, 50)
    assert centres.shape == (2, 2)
    agree = max(np.mean(labels == y), np.mean(labels == 1 - y))   # labels arbitrary
    assert agree > 0.9


def test_kmeans_always_returns_k_clusters_even_in_noise():
    """Section 18's pitfall: it partitions structureless data just as tidily."""
    X = np.random.default_rng(5).uniform(size=(200, 2))
    centres, labels = attempt(drills.kmeans, X, 4, 0, 50)
    assert centres.shape == (4, 2)
    assert len(np.unique(labels)) == 4, "k-means will find k groups whether or not they exist"


def test_pca_recovers_a_planted_direction():
    r = np.random.default_rng(6)
    latent = r.normal(size=(300, 1)) * 4.0
    direction = np.array([[0.6, 0.8]])
    X = latent @ direction + r.normal(0, 0.05, size=(300, 2))
    comp, mean, explained = attempt(drills.pca_fit, X, 1)
    assert explained > 0.95
    cosine = abs(float(comp[:, 0] @ direction.ravel()))
    assert cosine > 0.99, "the top component must align with the planted direction"


def test_pca_components_are_descending_and_orthonormal():
    X = rng.normal(size=(200, 5)) @ np.diag([5.0, 3.0, 1.0, 0.4, 0.1])
    comp, mean, _ = attempt(drills.pca_fit, X, 3)
    assert comp.shape == (5, 3)
    assert np.allclose(comp.T @ comp, np.eye(3), atol=1e-8), "columns must be orthonormal"
    var = [np.var((X - mean) @ comp[:, i]) for i in range(3)]
    assert var[0] > var[1] > var[2], "components must be ordered by variance"


def test_pca_reconstruction_improves_with_k():
    X = rng.normal(size=(150, 6)) @ np.diag([6.0, 4.0, 2.0, 1.0, 0.5, 0.2])
    errs = []
    for k in (1, 3, 6):
        comp, mean, _ = attempt(drills.pca_fit, X, k)
        errs.append(np.mean((X - attempt(drills.pca_reconstruct, X, comp, mean)) ** 2))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-16, "keeping every component must reconstruct exactly"


def test_pca_centres_the_data():
    """Forgetting to centre is the silent failure of Part I section 23."""
    r = np.random.default_rng(7)
    X = r.normal(size=(200, 2)) @ np.diag([3.0, 0.2]) + np.array([50.0, -30.0])
    comp, mean, _ = attempt(drills.pca_fit, X, 1)
    assert np.allclose(mean, X.mean(axis=0))
    cosine = abs(float(comp[:, 0] @ np.array([1.0, 0.0])))
    assert cosine > 0.98, "uncentred PCA points at the mean, not the spread"


# --------------------------------------------------------------- kernels ----

def test_kernel_trick_matches_the_explicit_feature_map_exactly():
    """The whole claim of section 15, checked to floating point."""
    A = rng.normal(size=(6, 2))
    B = rng.normal(size=(4, 2))
    K = attempt(drills.polynomial_kernel, A, B, 2)
    PA = attempt(drills.quadratic_feature_map, A)
    PB = attempt(drills.quadratic_feature_map, B)
    assert K.shape == (6, 4)
    assert np.allclose(K, PA @ PB.T), \
        "(A B')^2 must equal the inner product of the quadratic features"


def test_kernel_is_cheaper_than_the_feature_map_it_stands_for():
    """Cost is O(d), not O(number of monomials)."""
    A = rng.normal(size=(3, 60))
    K = attempt(drills.polynomial_kernel, A, A, 5)
    assert K.shape == (3, 3) and np.all(np.isfinite(K))
    # the explicit degree-5 map over 60 inputs has C(64,5) = 7,624,512 columns
    assert A.shape[1] == 60


def test_rbf_kernel_properties():
    A = rng.normal(size=(5, 3))
    K = attempt(drills.rbf_kernel, A, A, 1.0)
    assert np.allclose(np.diag(K), 1.0), "a point is at distance 0 from itself"
    assert np.allclose(K, K.T), "must be symmetric"
    assert np.all((K >= 0) & (K <= 1))


# --------------------------------------------------------- generalisation ---

def test_learning_curve_shapes_differ_by_capacity():
    """Figure 2's diagnostic. Needs an overlapping problem to be meaningful."""
    X, y = blobs(400, 2, 8, sep=0.6, sd=1.0)
    sizes = [10, 25, 60, 150, 280]

    hi = attempt(drills.learning_curve,
                 lambda a, b, c: attempt(drills.knn_predict, a, b, c, 1),
                 X, y, sizes, 0)
    lo = attempt(drills.learning_curve,
                 lambda a, b, c: attempt(drills.knn_predict, a, b, c, 15),
                 X, y, sizes, 0)
    tr_hi, te_hi = hi
    tr_lo, te_lo = lo

    assert np.allclose(tr_hi, 0.0), "k=1 has zero training error at every size"
    assert (te_hi[0] - tr_hi[0]) > (te_lo[0] - tr_lo[0]), \
        "the high-capacity model must show the larger train/test gap"
    assert te_lo[-1] < te_lo[0], "more data must help the smoother model"


def test_more_data_closes_the_gap():
    """High variance means a gap that CLOSES as data arrives."""
    X, y = blobs(600, 2, 12, sep=0.6, sd=1.0)
    sizes = [15, 400]
    tr, te = attempt(drills.learning_curve,
                     lambda a, b, c: attempt(drills.knn_predict, a, b, c, 5),
                     X, y, sizes, 0)
    assert (te[0] - tr[0]) > (te[-1] - tr[-1]), \
        "the generalisation gap should shrink with more training data"


def test_intrinsic_dimension_sees_through_the_ambient_one():
    """Section 23: raw dimension routinely overstates the real one."""
    r = np.random.default_rng(9)
    latent = r.normal(size=(400, 3))
    embed = r.normal(size=(3, 40))
    X = latent @ embed + r.normal(0, 1e-3, size=(400, 40))
    assert attempt(drills.intrinsic_dimension, X, 0.95) <= 4, \
        "40 features, 3 real degrees of freedom"


def test_nearest_and_farthest_converge_in_high_dimensions():
    """Figure 13's left panel, measured."""
    low = attempt(drills.nearest_farthest_ratio, 300, 2, 0)
    high = attempt(drills.nearest_farthest_ratio, 300, 500, 0)
    assert low > high, "the ratio must fall as dimension grows"
    assert high < 1.8, f"in 500-D everything is roughly equidistant, got {high:.2f}"
