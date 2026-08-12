"""
Tests for the Part XXVII drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result, because this part's subject includes what is
provably unrecoverable: that a frequency above Nyquist does not come back, that
no window resolves time and frequency at once, and that a magnitude spectrum has
thrown away the thing you wanted.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# ------------------------------------------------------------------ transform

def test_the_dft_matrix_is_unitary_up_to_scale():
    W = attempt(drills.dft_matrix, 16)
    assert np.allclose(W.conj().T @ W, 16 * np.eye(16), atol=1e-10)


def test_the_transform_finds_a_pure_tone_in_one_bin():
    n = 64
    x = np.cos(2 * np.pi * 5 * np.arange(n) / n)
    X = np.abs(attempt(drills.dft, x))
    assert X.argmax() in (5, n - 5)
    off = np.delete(X, [5, n - 5])
    assert off.max() < 1e-9 * X.max()


def test_the_fast_transform_agrees_with_the_definition():
    r = np.random.default_rng(0)
    for n in (8, 64, 512):
        x = r.standard_normal(n)
        a, b = attempt(drills.fft, x), attempt(drills.dft, x)
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-12


def test_the_fast_transform_refuses_a_length_it_cannot_handle():
    attempt(drills.fft, np.zeros(8))              # skip if unimplemented
    with pytest.raises(ValueError):
        drills.fft(np.zeros(12))


def test_the_inverse_recovers_the_input():
    r = np.random.default_rng(1)
    x = r.standard_normal(32)
    y = attempt(drills.idft, attempt(drills.dft, x))
    assert np.abs(y - x).max() < 1e-10
    assert np.abs(y.imag).max() < 1e-10        # a real signal stays real


def test_energy_is_conserved():
    r = np.random.default_rng(2)
    assert attempt(drills.parseval_ratio, r.standard_normal(128)) == pytest.approx(1.0, abs=1e-10)


def test_bin_spacing_is_set_by_the_record_length():
    f = attempt(drills.bin_frequencies, 100, 1000.0)
    assert f[1] == pytest.approx(10.0)
    assert attempt(drills.bin_frequencies, 200, 1000.0)[1] == pytest.approx(5.0)


# ------------------------------------------------------------------- sampling

def test_below_nyquist_nothing_moves():
    assert attempt(drills.alias_frequency, 300.0, 1000.0) == pytest.approx(300.0)


def test_above_nyquist_a_frequency_becomes_another_frequency():
    """Not zero, not attenuated — 700 Hz sampled at 1 kHz arrives as 300 Hz and
    is thereafter indistinguishable from a real 300 Hz component."""
    assert attempt(drills.alias_frequency, 700.0, 1000.0) == pytest.approx(300.0)
    assert attempt(drills.alias_frequency, 1300.0, 1000.0) == pytest.approx(300.0)


def test_the_aliased_tone_is_measured_where_the_theory_says():
    fs, n = 1000.0, 1024
    x = attempt(drills.sample_sinusoid, 700.0, fs, n)
    assert attempt(drills.peak_frequency, x, fs) == pytest.approx(300.0, abs=1.5)


def test_two_different_tones_sample_to_the_same_thing():
    """The exact sense in which the information is gone: the samples are equal
    up to a sign, so no algorithm can tell which tone produced them."""
    fs, n = 1000.0, 512
    a = attempt(drills.sample_sinusoid, 300.0, fs, n)
    b = attempt(drills.sample_sinusoid, 700.0, fs, n)
    assert np.abs(a + b).max() < 1e-10


# -------------------------------------------------------------------- filters

def test_a_binomial_kernel_is_a_low_pass_filter():
    h = attempt(drills.binomial_kernel, 5)
    assert h.sum() == pytest.approx(1.0)
    assert attempt(drills.is_lowpass, h)
    assert not attempt(drills.is_highpass, h)


def test_a_difference_is_a_high_pass_filter():
    for h in (np.array([-1.0, 1.0]), np.array([1.0, -2.0, 1.0])):
        assert attempt(drills.is_highpass, h)
        assert not attempt(drills.is_lowpass, h)


def test_the_centred_difference_is_band_pass_and_not_high_pass():
    """Worth knowing before trusting the label. The symmetric derivative
    kernel — the one inside every Sobel operator — has response exactly zero at
    DC AND exactly zero at Nyquist, peaking in between. It is band-pass, so it
    is blind to precisely the alternating pattern a high-pass filter is supposed
    to catch."""
    r = np.abs(attempt(drills.frequency_response, np.array([-1.0, 0.0, 1.0])))
    assert r[0] < 1e-12 and r[-1] < 1e-12
    assert r.argmax() == pytest.approx(len(r) // 2, abs=1)
    assert not attempt(drills.is_highpass, np.array([-1.0, 0.0, 1.0]))
    assert not attempt(drills.is_lowpass, np.array([-1.0, 0.0, 1.0]))


def test_a_longer_low_pass_kernel_cuts_more():
    r5 = np.abs(attempt(drills.frequency_response, attempt(drills.binomial_kernel, 5)))
    r15 = np.abs(attempt(drills.frequency_response, attempt(drills.binomial_kernel, 15)))
    mid = len(r5) // 2
    assert r15[mid] < r5[mid] / 10
    # both are exactly zero at Nyquist, which is what makes them anti-aliasers
    assert r5[-1] < 1e-12 and r15[-1] < 1e-12


def test_a_window_suppresses_leakage():
    n = 256
    x = np.cos(2 * np.pi * 10.5 * np.arange(n) / n)     # deliberately between bins
    bare = attempt(drills.leakage, x)
    win = attempt(drills.leakage, x, attempt(drills.hann, n))
    assert bare > 0.1
    assert win < bare / 5


def test_a_window_is_unnecessary_when_the_tone_lands_on_a_bin():
    n = 256
    x = np.cos(2 * np.pi * 10 * np.arange(n) / n)
    assert attempt(drills.leakage, x) < 1e-12


# ------------------------------------------------------------------ 2-D tools

def test_circular_convolution_is_exactly_shift_equivariant():
    r = np.random.default_rng(3)
    img = r.standard_normal((32, 32))
    k = r.standard_normal((3, 3))
    a = np.roll(attempt(drills.circ_conv2, img, k), 5, axis=1)
    b = attempt(drills.circ_conv2, np.roll(img, 5, axis=1), k)
    assert np.abs(a - b).max() < 1e-10


def test_downsampling_breaks_the_invariance_that_convolution_keeps():
    """The measurement of Figure 1, in six lines. Pooling an exactly
    equivariant map is exactly invariant, so everything that moves here is the
    subsampling."""
    r = np.random.default_rng(4)
    img = attempt(drills.power_law_image, 64, 2.0, seed=7)
    k = r.standard_normal((3, 3))

    def stack(x, aa):
        y = np.maximum(attempt(drills.circ_conv2, x, k), 0)
        y = attempt(drills.downsample, y, 2, aa)
        y = np.maximum(attempt(drills.circ_conv2, y, k), 0)
        return np.array([attempt(drills.downsample, y, 2, aa).mean()])

    naive = attempt(drills.shift_inconsistency, lambda x: stack(x, False), img)
    filt = attempt(drills.shift_inconsistency, lambda x: stack(x, True), img)
    assert naive > 1e-6                      # far above float64 rounding
    assert filt < naive / 2                  # the filter helps
    assert filt > 0                          # and does not fix it


def test_the_reconstruction_looks_like_its_phase_donor():
    a = attempt(drills.power_law_image, 64, 2.0, seed=1)
    b = attempt(drills.power_law_image, 64, 2.0, seed=2)
    mixed = attempt(drills.swap_phase, a, b)
    to_phase = attempt(drills.correlation, mixed, b)
    to_magnitude = attempt(drills.correlation, mixed, a)
    assert to_phase > 0.7                      # Figure 4 measures 0.736 on real images
    assert abs(to_magnitude) < 0.2
    assert to_phase > 3 * abs(to_magnitude)


def test_the_two_images_have_indistinguishable_magnitude_spectra():
    """Which is the other half of the point: a power spectrum cannot tell these
    apart, so any method that keeps only magnitude has discarded the content."""
    a = attempt(drills.power_law_image, 64, 2.0, seed=1)
    b = attempt(drills.power_law_image, 64, 2.0, seed=2)
    pa, pb = attempt(drills.radial_spectrum, a), attempt(drills.radial_spectrum, b)
    assert attempt(drills.correlation, np.log(pa[1:]), np.log(pb[1:])) > 0.99
    assert abs(attempt(drills.correlation, a, b)) < 0.2


def test_the_spectrum_estimator_recovers_an_exponent_it_was_given():
    for alpha in (1.5, 2.0, 3.0):
        img = attempt(drills.power_law_image, 128, alpha, seed=0)
        assert attempt(drills.spectrum_exponent, img) == pytest.approx(-alpha, abs=0.15)


def test_careless_downsampling_folds_power_back_into_the_band():
    img = attempt(drills.power_law_image, 256, 1.0, seed=3)
    naive = attempt(drills.radial_spectrum, attempt(drills.downsample, img, 4, False))
    filt = attempt(drills.radial_spectrum, attempt(drills.downsample, img, 4, True))
    top = slice(len(naive) // 2, len(naive) - 2)
    assert naive[top].mean() > 2 * filt[top].mean()


# -------------------------------------------------------------- graph spectra

def test_the_path_laplacian_has_rows_summing_to_zero():
    L = attempt(drills.path_laplacian, 8)
    assert np.allclose(L.sum(1), 0)
    assert np.allclose(L, L.T)


def test_its_eigenvectors_are_cosines():
    """Frequency is not a fact about waves. It is the eigenbasis of a Laplacian,
    and on a path that Laplacian is the second-difference operator."""
    n = 64
    w, V = attempt(drills.laplacian_eigenbasis, attempt(drills.path_laplacian, n))
    assert w[0] == pytest.approx(0.0, abs=1e-10)
    v = V[:, 1]
    ref = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    assert abs(attempt(drills.correlation, v, ref)) > 0.999


def test_higher_eigenvalue_means_more_disagreement_across_edges():
    n = 32
    L = attempt(drills.path_laplacian, n)
    w, V = attempt(drills.laplacian_eigenbasis, L)
    rough = [float(V[:, k] @ L @ V[:, k]) for k in (1, 5, 20)]
    assert rough[0] < rough[1] < rough[2]


def test_repeated_averaging_is_a_low_pass_filter_on_the_graph():
    """Part XXVI's over-smoothing, named in this basis: every eigenvector is
    scaled by less than one and the high ones by far less, so what survives is
    the lowest mode."""
    n = 32
    L = attempt(drills.path_laplacian, n)
    w, V = attempt(drills.laplacian_eigenbasis, L)
    A = np.eye(n) - 0.25 * L                       # one round of averaging
    survives = [float(np.abs(np.linalg.matrix_power(A, 20) @ V[:, k]).max()) for k in (1, 5, 20)]
    assert survives[0] > survives[1] > survives[2]


# ------------------------------------------------------------------- encoding

def test_the_positional_encoding_is_bounded_and_the_right_shape():
    E = attempt(drills.sinusoidal_encoding, np.arange(50), 16)
    assert E.shape == (50, 16)
    assert np.abs(E).max() <= 1.0 + 1e-12


def test_a_shift_acts_linearly_on_the_encoding():
    """The entire design rationale, checkable in five lines: relative position
    is a fixed linear map, so attention can read it off."""
    for k in (1, 7, 30):
        assert attempt(drills.shift_is_linear, 32, k) < 1e-8


def test_a_random_encoding_has_no_such_property():
    """The contrast that makes the previous test mean something: the linear
    relation is a property of the frequency construction, not of any encoding."""
    assert attempt(drills.shift_is_linear, 32, 3) < 1e-8
    r = np.random.default_rng(5)
    E0 = r.standard_normal((64, 32))
    E1 = r.standard_normal((64, 32))
    M, *_ = np.linalg.lstsq(E0, E1, rcond=None)
    assert np.abs(E0 @ M - E1).max() > 0.1


def test_the_encoding_frequencies_are_geometrically_spaced():
    E = attempt(drills.sinusoidal_encoding, np.arange(4096), 32)
    fast = np.abs(np.diff(E[:, 0])).mean()
    slow = np.abs(np.diff(E[:, -2])).mean()
    assert fast > 100 * slow


def test_fourier_features_make_a_high_frequency_linearly_available():
    """Which is what §36 claims and why coordinate networks need them: after the
    lift, a sinusoid the network could not synthesise is one of its inputs."""
    x = np.linspace(0, 1, 256, endpoint=False)
    F = attempt(drills.fourier_features, x, [1, 2, 4, 8, 16])
    y = np.sin(2 * np.pi * 16 * x)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    assert np.abs(F @ coef - y).max() < 1e-10
    # and without the lift, a linear model in x cannot do it at all
    P = np.column_stack([x ** k for k in range(6)])
    c2, *_ = np.linalg.lstsq(P, y, rcond=None)
    assert np.abs(P @ c2 - y).max() > 0.5
