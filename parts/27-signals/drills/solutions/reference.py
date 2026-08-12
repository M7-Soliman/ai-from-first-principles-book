"""
Reference solutions for the Part XXVII drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------ transform

def dft_matrix(n):
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


def dft(x):
    x = np.asarray(x, complex)
    return dft_matrix(len(x)) @ x


def idft(X):
    X = np.asarray(X, complex)
    n = len(X)
    return np.conj(dft_matrix(n)) @ X / n


def fft(x):
    x = np.asarray(x, complex)
    n = len(x)
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    bits = int(np.log2(n))
    j = np.arange(n)
    rev = np.zeros(n, np.int64)
    for b in range(bits):
        rev |= ((j >> b) & 1) << (bits - 1 - b)
    a = x[rev].copy()
    size = 2
    while size <= n:
        half = size // 2
        w = np.exp(-2j * np.pi * np.arange(half) / size)
        a = a.reshape(-1, size)
        even, odd = a[:, :half], a[:, half:] * w
        a = np.concatenate([even + odd, even - odd], axis=1).ravel()
        size *= 2
    return a


def parseval_ratio(x):
    x = np.asarray(x, complex)
    X = fft(x) if not (len(x) & (len(x) - 1)) else dft(x)
    return float((np.abs(X) ** 2).sum() / len(x) / (np.abs(x) ** 2).sum())


def bin_frequencies(n, fs):
    return np.arange(n) * fs / n


# ------------------------------------------------------------------- sampling

def alias_frequency(f, fs):
    """Where a component at f lands once sampled at fs."""
    f = np.asarray(f, float)
    r = np.mod(f, fs)
    return np.where(r > fs / 2, fs - r, r)


def sample_sinusoid(f, fs, n):
    return np.sin(2 * np.pi * f * np.arange(n) / fs)


def peak_frequency(x, fs):
    X = np.abs(np.fft.rfft(x))
    return float(np.fft.rfftfreq(len(x), 1 / fs)[X.argmax()])


# -------------------------------------------------------------------- filters

def binomial_kernel(k):
    v = np.array([1.0])
    for _ in range(k - 1):
        v = np.convolve(v, [1.0, 1.0])
    return v / v.sum()


def frequency_response(h, n=512):
    return np.fft.rfft(np.asarray(h, float), n)


def is_lowpass(h, n=512):
    r = np.abs(frequency_response(h, n))
    return bool(r[0] >= r.max() - 1e-12 and r[-1] < 0.5 * r[0])


def is_highpass(h, n=512):
    r = np.abs(frequency_response(h, n))
    return bool(r[0] < 0.5 * r.max() and r[-1] > 0.5 * r.max())


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def leakage(x, window=None):
    """Fraction of spectral energy outside the three bins around the peak."""
    x = np.asarray(x, float)
    if window is not None:
        x = x * window
    P = np.abs(np.fft.rfft(x)) ** 2
    k = P.argmax()
    near = P[max(0, k - 1):k + 2].sum()
    return float(1 - near / P.sum())


# ------------------------------------------------------------------ 2-D tools

def circ_conv2(img, ker):
    H, W = np.shape(img)
    kh, kw = np.shape(ker)
    pad = np.zeros((H, W))
    pad[:kh, :kw] = ker
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


def downsample(img, factor=2, antialias=False, k=5):
    img = np.asarray(img, float)
    if antialias:
        v = binomial_kernel(k)
        img = circ_conv2(img, np.outer(v, v))
    return img[::factor, ::factor]


def shift_inconsistency(fn, img, shifts=range(8)):
    """Max relative spread of fn's output over circular shifts of the input."""
    vs = np.array([np.ravel(fn(np.roll(img, s, axis=1))) for s in shifts])
    return float(np.abs(vs - vs.mean(0)).max() / (np.abs(vs).mean() + 1e-300))


def swap_phase(a, b):
    """Image a's magnitude with image b's phase."""
    FA, FB = np.fft.fft2(np.asarray(a, float)), np.fft.fft2(np.asarray(b, float))
    return np.real(np.fft.ifft2(np.abs(FA) * np.exp(1j * np.angle(FB))))


def correlation(a, b):
    a = np.ravel(np.asarray(a, float)); b = np.ravel(np.asarray(b, float))
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def radial_spectrum(img):
    img = np.asarray(img, float)
    P = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    H, W = img.shape
    y, x = np.indices((H, W))
    r = np.hypot(y - H / 2, x - W / 2).astype(int)
    tot = np.bincount(r.ravel(), P.ravel())
    cnt = np.bincount(r.ravel())
    m = min(H, W) // 2
    return tot[:m] / np.maximum(cnt[:m], 1)


def spectrum_exponent(img, lo=3, trim=5):
    p = radial_spectrum(img)
    f = np.arange(len(p))
    hi = len(p) - trim
    return float(np.polyfit(np.log(f[lo:hi]), np.log(p[lo:hi]), 1)[0])


def power_law_image(n, alpha, seed=0):
    """An image whose radial power spectrum is f^(-alpha), by construction."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    r = np.hypot(fy, fx)
    r[0, 0] = 1.0
    F = (r ** (-alpha / 2)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, n)))
    F[0, 0] = 0.0                      # zero mean: a huge DC term would swamp
    return np.real(np.fft.ifft2(F))    # every relative measurement taken here


# -------------------------------------------------------------- graph spectra

def path_laplacian(n):
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i + 1] = L[i + 1, i] = -1
    L[np.diag_indices(n)] = -L.sum(1)
    return L


def laplacian_eigenbasis(L):
    w, V = np.linalg.eigh(np.asarray(L, float))
    return w, V


# ------------------------------------------------------------------- encoding

def sinusoidal_encoding(positions, d, base=10000.0):
    p = np.asarray(positions, float)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = p / (base ** (2 * i / d))
    out = np.empty((len(p), d))
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


def shift_is_linear(d, k, base=10000.0, n=64):
    """Is encoding(p + k) a fixed linear function of encoding(p)?"""
    E0 = sinusoidal_encoding(np.arange(n), d, base)
    E1 = sinusoidal_encoding(np.arange(n) + k, d, base)
    M, *_ = np.linalg.lstsq(E0, E1, rcond=None)
    return float(np.abs(E0 @ M - E1).max())


def fourier_features(x, freqs):
    x = np.asarray(x, float)[:, None]
    f = np.asarray(freqs, float)[None, :]
    return np.concatenate([np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)], axis=1)


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
