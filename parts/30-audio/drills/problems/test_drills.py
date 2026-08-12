"""
Tests for the Part XXX drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert an EXACT value: the round trip with the true phase, the forward
recursion against enumeration, and the alignment counts. Several others assert a
NEGATIVE result, because this part's content includes what these representations
provably cannot do.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def signal(n=4096, fs=8000, seed=0):
    t = np.arange(n) / fs
    ph = 2 * np.pi * np.cumsum(150 + 40 * np.sin(2 * np.pi * 1.7 * t)) / fs
    x = sum(np.sin(k * ph) / k for k in range(1, 7))
    x[n // 2:n // 2 + 40] += 3.0 * np.hanning(40)
    return x / np.abs(x).max()


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


# --------------------------------------------------------------------- mel

def test_the_mel_map_round_trips():
    f = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(attempt(drills.mel_to_hz, attempt(drills.hz_to_mel, f)), f)


def test_the_mel_scale_is_nearly_linear_low_and_logarithmic_high():
    lo = attempt(drills.hz_to_mel, 200.0) - attempt(drills.hz_to_mel, 100.0)
    hi = attempt(drills.hz_to_mel, 8000.0) - attempt(drills.hz_to_mel, 7900.0)
    assert lo > 5 * hi


def test_mel_bands_pool_more_bins_at_high_frequency():
    fb, edges = attempt(drills.mel_filterbank, 64, 1024, 16000)
    width = np.array([(fb[m] > 0).sum() for m in range(64)])
    assert width[-1] > 10 * width[0]


def test_the_filterbank_covers_the_spectrum_without_rescaling_it():
    """If the bands do not sum to about one per bin, energy is being tilted with
    frequency and every downstream statistic inherits it."""
    fb, _ = attempt(drills.mel_filterbank, 40, 512, 16000)
    s = fb.sum(0)
    mid = s[len(s) // 8: -len(s) // 8]
    assert mid.min() > 0.7 and mid.max() < 1.3


def test_a_fixed_separation_falls_inside_one_band_at_high_frequency():
    """Measured as bands spanned, not by counting spectral peaks. Peak counting
    is not monotone in frequency — two wide adjacent bands can still produce two
    local maxima — so it reports a crossover that belongs to the criterion."""
    n_mels, fs = 64, 16000
    _, edges = attempt(drills.mel_filterbank, n_mels, 1024, fs)
    centres = edges[1:-1]
    per_band = (attempt(drills.hz_to_mel, fs / 2) - attempt(drills.hz_to_mel, 0.0)) / (n_mels + 1)
    span = lambda f: float(attempt(drills.hz_to_mel, f + 200.0)
                           - attempt(drills.hz_to_mel, f)) / per_band
    assert span(float(centres[0])) > 3.0        # several bands, low down
    assert span(float(centres[-1])) < 1.0       # inside one, up high
    spans = np.array([span(float(f)) for f in centres])
    assert np.all(np.diff(spans) < 0)           # and monotone, unlike peak counting


# -------------------------------------------------------------------- stft

def test_the_round_trip_is_exact_on_the_interior():
    """Exact — not 'close'. Measure across the whole signal instead and a correct
    implementation looks broken, because the end windows have fewer overlapping
    frames."""
    x = signal()
    win, hop = 512, 128
    y = attempt(drills.istft, attempt(drills.stft, x, win, hop), win, hop, len(x))
    xi = attempt(drills.interior, x, win)
    yi = attempt(drills.interior, y, win)
    assert np.abs(xi - yi).max() < 1e-12


def test_the_boundary_is_where_the_error_lives():
    x = signal()
    win, hop = 512, 128
    y = attempt(drills.istft, attempt(drills.stft, x, win, hop), win, hop, len(x))
    assert np.abs(x - y).max() > 1e-6        # over the whole signal
    assert np.abs(attempt(drills.interior, x - y, win)).max() < 1e-12


def test_random_phase_keeps_the_magnitude_and_loses_the_signal():
    x = signal()
    win, hop = 512, 128
    S = attempt(drills.stft, x, win, hop)
    mag = np.abs(S)
    r = np.random.default_rng(0)
    y = attempt(drills.istft, mag * np.exp(1j * r.uniform(-np.pi, np.pi, mag.shape)),
                win, hop, len(x))
    assert abs(corr(attempt(drills.interior, y, win),
                    attempt(drills.interior, x, win))) < 0.2


def test_phase_estimation_converges_in_spectrogram_space():
    x = signal()
    win, hop = 512, 128
    mag = np.abs(attempt(drills.stft, x, win, hop))
    _, errs = attempt(drills.griffin_lim, mag, win, hop, len(x), 60)
    assert np.all(np.diff(errs) <= 1e-9)     # monotone
    assert errs[-1] < 0.5 * errs[0]
    assert errs[-1] > 1e-3                   # and does not reach zero


# --------------------------------------------------------------------- ctc

def test_the_collapse_rule_deletes_repeats_before_blanks():
    """(a, a) becomes 'a', not 'aa'. Producing a repeated symbol needs a blank
    between, which is why the blank exists at all."""
    assert attempt(drills.ctc_collapse, [1, 1], 0) == [1]
    assert attempt(drills.ctc_collapse, [1, 0, 1], 0) == [1, 1]
    assert attempt(drills.ctc_collapse, [0, 1, 1, 0, 2], 0) == [1, 2]


def test_the_extended_target_interleaves_blanks():
    assert list(attempt(drills.ctc_expand, [1, 2], 0)) == [0, 1, 0, 2, 0]


def test_the_forward_recursion_matches_enumeration():
    r = np.random.default_rng(0)
    for T in (3, 5, 7):
        lg = r.standard_normal((T, 3))
        a = attempt(drills.ctc_logprob, lg, [1, 2])
        b = attempt(drills.ctc_enumerate_logprob, lg, [1, 2])
        assert abs(a - b) < 1e-12


def test_an_impossible_target_has_zero_probability():
    lg = np.zeros((2, 3))
    assert attempt(drills.ctc_logprob, lg, [1, 1]) == -np.inf   # needs 3 frames


def test_the_alignment_count_matches_enumeration_and_goes_further():
    import itertools
    for T in (3, 4, 5, 6):
        brute = sum(1 for p in itertools.product(range(3), repeat=T)
                    if attempt(drills.ctc_collapse, p, 0) == [1, 2])
        assert attempt(drills.ctc_alignment_count, T, [1, 2]) == brute
    assert attempt(drills.ctc_alignment_count, 80, [1, 2]) > 10 ** 6


def test_ctc_can_be_certain_about_one_sequence():
    """The contrast the next test needs. Putting nearly all the mass on a single
    output is easy — it is splitting it that the factorisation forbids."""
    V, T = 3, 4
    L = np.zeros((T, V)); L[0, 1] = 12.0; L[1, 0] = 12.0; L[2, 2] = 12.0; L[3, 0] = 12.0
    assert np.exp(attempt(drills.ctc_logprob, L, [1, 2])) > 0.99


def test_ctc_cannot_split_its_mass_evenly_between_two_sequences():
    """The independence assumption, measured. Optimised toward 0.5 on each of two
    outputs; the best it manages is a quarter each, so half the mass ends up on
    sequences the target never contains. Per-frame logits are optimised directly,
    so the result is about the family and not about the fit."""
    V, T = 3, 4
    rng = np.random.default_rng(0)
    L = rng.standard_normal((T, V)) * 0.1
    attempt(drills.ctc_logprob, L, [1, 2])          # skip cleanly if unimplemented
    for _ in range(500):
        p = lambda M: (np.exp(drills.ctc_logprob(M, [1, 2])),
                       np.exp(drills.ctc_logprob(M, [2, 1])))
        a, b = p(L)
        base = (a - 0.5) ** 2 + (b - 0.5) ** 2
        g = np.zeros_like(L)
        for i in range(T):
            for j in range(V):
                L2 = L.copy(); L2[i, j] += 1e-5
                a2, b2 = p(L2)
                g[i, j] = ((a2 - 0.5) ** 2 + (b2 - 0.5) ** 2 - base) / 1e-5
        L -= 8.0 * g
    a = np.exp(attempt(drills.ctc_logprob, L, [1, 2]))
    b = np.exp(attempt(drills.ctc_logprob, L, [2, 1]))
    assert a == pytest.approx(0.25, abs=0.05)
    assert b == pytest.approx(0.25, abs=0.05)
    assert a + b < 0.6                      # half the mass is elsewhere


# ----------------------------------------------------------- quantisation

def correlated(n, d, rho, seed=0):
    S = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    return np.random.default_rng(seed).standard_normal((n, d)) @ np.linalg.cholesky(S).T


def test_more_levels_means_less_error():
    X = correlated(1500, 3, 0.5)
    a = attempt(drills.scalar_quantise_error, X, 2)
    b = attempt(drills.scalar_quantise_error, X, 8)
    assert b < a


def test_vector_beats_scalar_on_correlated_data():
    X = correlated(1500, 3, 0.9)
    sq = attempt(drills.scalar_quantise_error, X, 4)          # 2 bits/dim = 64 cells
    _, vq = attempt(drills.kmeans, X, 64, 40, 1)
    assert vq < sq / 2


def test_vector_still_beats_scalar_when_the_dimensions_are_independent():
    """The part that surprises people. Correlation is not the only source of the
    gain — a hypercube is a poor cell whatever the data looks like."""
    X = correlated(1500, 3, 0.0)
    sq = attempt(drills.scalar_quantise_error, X, 4)
    _, vq = attempt(drills.kmeans, X, 64, 40, 1)
    assert vq < sq
    assert vq > sq / 5                                        # but by much less


def test_residual_quantisation_approaches_a_full_codebook_cheaply():
    X = correlated(1500, 4, 0.8)
    _, vq = attempt(drills.kmeans, X, 256, 40, 1)
    rq = attempt(drills.residual_quantise_error, X, 2, 16)
    assert rq < 3 * vq
    assert rq > vq                                            # close, not equal


def test_a_codebook_can_be_mostly_dead():
    """Entries far from the data are never nearest, so they receive no gradient
    and stay where they were initialised."""
    X = correlated(600, 2, 0.0)
    C = np.concatenate([X[:4], 50 + np.zeros((28, 2))])       # 28 useless entries
    assert attempt(drills.codebook_usage, X, C) <= 4


# ------------------------------------------------------------ contrastive

def test_the_contrastive_loss_is_lower_when_pairs_match():
    r = np.random.default_rng(0)
    A = r.standard_normal((32, 8))
    good = A + 0.1 * r.standard_normal((32, 8))
    bad = r.standard_normal((32, 8))
    assert attempt(drills.contrastive_loss, A, good, 0.1) < \
           attempt(drills.contrastive_loss, A, bad, 0.1)


def test_the_loss_keeps_pushing_even_when_retrieval_is_perfect():
    """Unlike a hinge, the softmax never stops — which is why these systems keep
    training long after recall@1 has reached one."""
    r = np.random.default_rng(1)
    A = r.standard_normal((16, 8))
    B = A + 0.01 * r.standard_normal((16, 8))
    assert attempt(drills.recall_at_1, A, B) == 1.0
    assert attempt(drills.contrastive_loss, A, B, 0.1) > 0.0


def test_temperature_selects_how_many_negatives_carry_the_gradient():
    r = np.random.default_rng(2)
    sims = np.sort(r.uniform(-1, 1, 512))
    hot = attempt(drills.effective_negatives, sims, 1.0)
    warm = attempt(drills.effective_negatives, sims, 0.1)
    cold = attempt(drills.effective_negatives, sims, 0.01)
    assert hot > warm > cold
    assert hot > 400 and cold < 5


def test_recall_falls_as_the_gallery_grows():
    """So a recall number without its gallery size is not a measurement."""
    out = []
    for N in (100, 1000, 10000):
        r = np.random.default_rng(N)
        A = r.standard_normal((N, 32))
        B = A + 1.5 * r.standard_normal((N, 32))
        out.append(attempt(drills.recall_at_1, A, B))
    assert out[0] > out[1] > out[2]


def test_the_three_similarities_are_different_quantities():
    r = np.random.default_rng(3)
    A = r.standard_normal((64, 16))
    B = A + 0.2 * r.standard_normal((64, 16))
    within, across, matched = attempt(drills.modality_similarities, A, B)
    assert matched > across
    assert matched > within
