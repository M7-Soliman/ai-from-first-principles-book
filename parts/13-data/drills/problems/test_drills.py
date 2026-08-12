"""
Tests for the Part XIII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several tests assert that an instrument FAILS to detect something — the noise
tolerance, the variance under model mismatch, the aggregate kappa hiding a rare
class. Those are the point of the part, not oversights.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ----------------------------------------------------------------- the frame ----

def test_interval_shrinks_like_one_over_root_n():
    r = np.random.default_rng(0)
    hws = []
    for n in (100, 400, 1600):
        est, hw = attempt(drills.rate_with_interval, r.random(n) < 0.4)
        hws.append(hw)
    assert hws[1] == pytest.approx(hws[0] / 2, rel=0.25)
    assert hws[2] == pytest.approx(hws[0] / 4, rel=0.25)


def test_a_correct_frame_covers_the_truth():
    covered = 0
    for s in range(40):
        est, hw, truth = attempt(drills.frame_estimate,
                                 [0.6, 0.3], [0.6, 0.4], [0.6, 0.4], 4000, seed=s)
        covered += attempt(drills.interval_covers, est, hw, truth)
    assert covered >= 34, f"a matching frame should cover ~95% of the time, got {covered}/40"


def test_a_biased_frame_stops_covering_as_n_grows():
    """Section 2: more data makes a mis-framed estimate more confident, not more right."""
    small = sum(attempt(drills.interval_covers,
                        *attempt(drills.frame_estimate, [0.6, 0.3], [0.6, 0.4],
                                 [0.9, 0.1], 50, seed=s)[:2],
                        attempt(drills.frame_estimate, [0.6, 0.3], [0.6, 0.4],
                                [0.9, 0.1], 50, seed=s)[2])
                for s in range(30))
    big = sum(attempt(drills.interval_covers,
                      *attempt(drills.frame_estimate, [0.6, 0.3], [0.6, 0.4],
                               [0.9, 0.1], 20000, seed=s)[:2],
                      attempt(drills.frame_estimate, [0.6, 0.3], [0.6, 0.4],
                              [0.9, 0.1], 20000, seed=s)[2])
              for s in range(30))
    assert big < small, f"coverage should COLLAPSE with n, got {small}/30 -> {big}/30"
    assert big == 0, f"at n=20000 the interval should never contain the truth, got {big}/30"


# ----------------------------------------------------------------- agreement ----

def test_kappa_endpoints():
    a = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    assert attempt(drills.cohen_kappa, a, a, 2) == pytest.approx(1.0)
    r = np.random.default_rng(0)
    x, y = r.integers(0, 2, 4000), r.integers(0, 2, 4000)
    assert abs(attempt(drills.cohen_kappa, x, y, 2)) < 0.1


def test_high_raw_agreement_can_have_zero_kappa():
    """Drill C9: 90% agreement conveying nothing."""
    r = np.random.default_rng(1)
    n = 4000
    a = (r.random(n) < 0.05).astype(int)          # both annotators say 0 almost always
    b = (r.random(n) < 0.05).astype(int)
    raw = float((a == b).mean())
    k = attempt(drills.cohen_kappa, a, b, 2)
    assert raw > 0.88, f"raw agreement should be high, got {raw:.3f}"
    assert abs(k) < 0.1, f"kappa should be near zero, got {k:.3f}"


def test_aggregate_kappa_hides_a_rare_class():
    """Drill C10 — and it needs more than two classes.

    With two classes the one-vs-rest kappa for either class IS the aggregate,
    because the one-vs-rest problem is the original problem. The hiding only
    happens once a common class can dominate the average.
    """
    r = np.random.default_rng(2)
    n = 8000
    truth = r.choice([0, 1, 2, 3], n, p=[0.60, 0.30, 0.07, 0.03])
    a = truth.copy()
    b = truth.copy()
    rare = np.where(truth == 3)[0]
    # annotator b botches the rarest class, scattering it over the common ones
    b[r.choice(rare, int(0.8 * len(rare)), replace=False)] = 0
    agg = attempt(drills.cohen_kappa, a, b, 4)
    per = attempt(drills.kappa_per_class, a, b, 4)
    assert agg > 0.9, f"the aggregate should still look excellent, got {agg:.3f}"
    assert per[3] < 0.4, f"the rare class should be poor, got {per[3]:.3f}"
    assert per[0] > per[3] and per[1] > per[3]


# --------------------------------------------------------------- label noise ----

def test_flip_rate_is_what_was_asked_for():
    y = np.zeros(10000, dtype=int)
    y[5000:] = 1
    for p in (0.1, 0.3):
        yn = attempt(drills.flip_labels, y, p, 2, seed=0)
        assert float((yn != y).mean()) == pytest.approx(p, abs=0.02)


def test_asymmetric_noise_moves_the_marginal_and_symmetric_does_not():
    """Section 9: which is why one is survivable and the other is not."""
    r = np.random.default_rng(3)
    y = (r.random(20000) < 0.5).astype(int)
    sym = attempt(drills.flip_labels, y, 0.3, 2, seed=0, asymmetric=False)
    asym = attempt(drills.flip_labels, y, 0.3, 2, seed=0, asymmetric=True)
    assert abs(sym.mean() - y.mean()) < 0.02, "symmetric noise leaves the marginal alone"
    assert asym.mean() - y.mean() > 0.2, "asymmetric noise shifts it"


def test_noise_ceiling():
    assert attempt(drills.noise_ceiling, 0.3) == pytest.approx(0.7)


# ------------------------------------------------------------------ leakage ----

def test_random_split_leaks_groups_and_grouped_split_does_not():
    r = np.random.default_rng(4)
    groups = r.integers(0, 60, 1200)              # ~20 records per group
    tr, te = attempt(drills.random_split, len(groups), 0.3, seed=0)
    leak_random = attempt(drills.leakage_rate, tr, te, groups)
    tr2, te2 = attempt(drills.grouped_split, groups, 0.3, seed=0)
    leak_grouped = attempt(drills.leakage_rate, tr2, te2, groups)
    assert leak_random > 0.9, f"a random split should leak nearly everything, got {leak_random:.2f}"
    assert leak_grouped == 0.0, f"a grouped split must leak nothing, got {leak_grouped:.2f}"


def test_grouped_split_keeps_every_record():
    r = np.random.default_rng(5)
    groups = r.integers(0, 40, 900)
    tr, te = attempt(drills.grouped_split, groups, 0.25, seed=1)
    assert len(tr) + len(te) == len(groups)
    assert set(tr.tolist()) & set(te.tolist()) == set()


def test_contamination_inflation():
    assert attempt(drills.contamination_inflation, 0.805, 0.610) == pytest.approx(0.195)


# ------------------------------------------------------------- near-duplicates ----

def test_jaccard_endpoints():
    a = attempt(drills.shingles, "abcdefghij", 4)
    assert attempt(drills.jaccard, a, a) == pytest.approx(1.0)
    b = attempt(drills.shingles, "zzzzzzzzzz", 4)
    assert attempt(drills.jaccard, a, b) == pytest.approx(0.0)


def test_minhash_estimates_jaccard():
    """Section 15's identity, checked across the range."""
    rng = np.random.default_rng(6)
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    errs = []
    for edit in (0, 8, 20, 40, 70):
        base = " ".join(rng.choice(words, 40))
        other = base[:max(1, len(base) - edit)] + "q" * edit
        A = attempt(drills.shingles, base, 5)
        B = attempt(drills.shingles, other, 5)
        true = attempt(drills.jaccard, A, B)
        est = attempt(drills.minhash_similarity,
                      attempt(drills.minhash_signature, A, 512, seed=3),
                      attempt(drills.minhash_signature, B, 512, seed=3))
        errs.append(abs(est - true))
    assert max(errs) < 0.08, f"max error {max(errs):.3f} with 512 permutations"


def test_minhash_error_falls_like_one_over_root_k():
    rng = np.random.default_rng(7)
    base = " ".join(rng.choice(["a", "bb", "ccc", "dddd"], 60))
    other = base[:-30] + "z" * 30
    A = attempt(drills.shingles, base, 5)
    B = attempt(drills.shingles, other, 5)
    true = attempt(drills.jaccard, A, B)

    def err(k):
        e = [abs(attempt(drills.minhash_similarity,
                         attempt(drills.minhash_signature, A, k, seed=s),
                         attempt(drills.minhash_signature, B, k, seed=s)) - true)
             for s in range(12)]
        return float(np.mean(e))

    assert err(16) > err(256), "more permutations must mean less error"
    assert err(256) < 0.06


def test_lsh_band_structure():
    sig = np.arange(64)
    bands = attempt(drills.lsh_bands, sig, 8)
    assert len(bands) == 8 and len(bands[0]) == 8
    assert attempt(drills.lsh_candidate_probability, 1.0, 8, 8) == pytest.approx(1.0)
    assert attempt(drills.lsh_candidate_probability, 0.0, 8, 8) == pytest.approx(0.0)


def test_the_lsh_threshold_formula_is_not_the_half_height_point():
    """The quoted (1/b)**(1/r) approximates the steepest point, not P = 0.5.

    Measured, the candidate probability there is around 0.65 for every sensible
    (b, r). Worth knowing before you use the formula to choose a threshold.
    """
    ps = []
    for b, r in ((8, 8), (16, 4), (4, 16), (20, 5)):
        t = attempt(drills.lsh_threshold, b, r)
        assert 0.0 < t < 1.0
        ps.append(attempt(drills.lsh_candidate_probability, t, b, r))
    assert min(ps) > 0.55 and max(ps) < 0.75, \
        f"expected ~0.65 at the quoted threshold, got {[round(p,3) for p in ps]}"


def test_the_lsh_curve_is_monotone_and_sharpens_with_rows():
    s = np.linspace(0, 1, 21)
    p8 = attempt(drills.lsh_candidate_probability, s, 8, 8)
    assert np.all(np.diff(p8) >= -1e-12), "the S-curve must be monotone"
    p_flat = attempt(drills.lsh_candidate_probability, s, 32, 2)
    assert p_flat[5] > p8[5], "fewer rows per band means a gentler curve"


# ------------------------------------------------------------------ collapse ----

def test_naive_resampling_barely_collapses():
    """Section 28: the folk version of the result does not reproduce."""
    finals = [attempt(drills.collapse_generations, 1000, 12, keep=1.0, seed=s)[-1]
              for s in range(25)]
    assert np.mean(finals) > 0.9, \
        f"resampling alone should stay near 1.0, got {np.mean(finals):.3f}"


def test_filtering_collapses_it_completely():
    finals = [attempt(drills.collapse_generations, 1000, 12, keep=0.5, seed=s)[-1]
              for s in range(10)]
    assert np.mean(finals) < 0.01, \
        f"keeping the typical half should collapse it, got {np.mean(finals):.4f}"


def test_variance_does_not_see_the_mismatch_collapse():
    """The mode count does; the standard deviation does not."""
    r = np.random.default_rng(8)
    m = r.integers(0, 2, 8000)
    bimodal = r.normal(np.where(m, 3.0, -3.0), 0.7)
    gaussian = r.normal(bimodal.mean(), bimodal.std(ddof=1), 8000)
    assert abs(bimodal.std() - gaussian.std()) / bimodal.std() < 0.05, \
        "the standard deviations should be nearly identical"
    assert attempt(drills.mode_count, bimodal) >= 2
    assert attempt(drills.mode_count, gaussian) == 1
