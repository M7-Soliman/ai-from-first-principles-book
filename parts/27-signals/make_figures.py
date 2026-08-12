"""
Figures for Part XXVII — Signals and Spectra.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about three minutes. Numpy and matplotlib only. The two natural images
ship with matplotlib, so nothing here needs a download or an optional package.

The transforms are written out rather than imported wherever the transform is
the subject: the radix-2 FFT below is the one whose scaling Figure 2 measures,
and it is checked against an explicit DFT matrix on every run.
"""
import gzip
import os
import re
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "sg"))
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data")

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
CYAN, LIGHT, INK = "#0e7490", "#cbd5e1", "#1e293b"
MAX_PT = 900

plt.rcParams.update({
    "svg.fonttype": "path", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "axes.linewidth": 0.9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def save(fig, name):
    p = os.path.join(OUT, name + ".svg")
    fig.savefig(p, format="svg")
    plt.close(fig)
    h = float(re.search(r'height="([\d.]+)pt"', open(p).read(1200)).group(1))
    if h > MAX_PT:
        raise ValueError(f"{name}.svg is {h:.0f}pt")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------- images

def grace():
    a = mpimg.imread(os.path.join(SAMPLE, "grace_hopper.jpg")).astype(float) / 255
    g = a @ np.array([0.299, 0.587, 0.114])
    return g[44:556, :512]                     # a 512x512 centre crop


def dem_native():
    """The elevation model at its own resolution. Figure 7 must not use the
    interpolated version below: bilinear upsampling manufactures a smooth
    spectrum, which is exactly the quantity that figure measures."""
    z = np.load(os.path.join(SAMPLE, "jacksboro_fault_dem.npz"))["elevation"].astype(float)
    return (z - z.min()) / (z.max() - z.min())


def scan():
    """A medical scan, 256x256 16-bit, also shipped with matplotlib."""
    raw = gzip.open(os.path.join(SAMPLE, "s1045.ima.gz")).read()
    a = np.frombuffer(raw, dtype=">u2")[:256 * 256].reshape(256, 256).astype(float)
    return (a - a.min()) / (a.max() - a.min())


def terrain():
    z = np.load(os.path.join(SAMPLE, "jacksboro_fault_dem.npz"))["elevation"].astype(float)
    z = z[:344, :344]
    # bilinear resize to 512x512 so the two images share a grid
    yi = np.linspace(0, z.shape[0] - 1, 512)
    xi = np.linspace(0, z.shape[1] - 1, 512)
    y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
    y1, x1 = np.minimum(y0 + 1, z.shape[0] - 1), np.minimum(x0 + 1, z.shape[1] - 1)
    fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
    out = (z[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) + z[np.ix_(y1, x0)] * fy * (1 - fx)
           + z[np.ix_(y0, x1)] * (1 - fy) * fx + z[np.ix_(y1, x1)] * fy * fx)
    return (out - out.min()) / (out.max() - out.min())


# -------------------------------------------------- the transforms, written out

def dft_matrix(n):
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


def dft_naive(x):
    """The definition, as a matrix product. O(n^2), and the ground truth."""
    return dft_matrix(len(x)) @ np.asarray(x, complex)


def fft_radix2(x):
    """Iterative Cooley-Tukey. Bit-reverse, then log2(n) butterfly stages, each
    stage vectorised. Requires a power of two, which is the only case Figure 2
    times."""
    x = np.asarray(x, complex)
    n = len(x)
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    # bit reversal permutation
    j = np.arange(n)
    rev = np.zeros(n, np.int64)
    bits = int(np.log2(n))
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


# ============================================================ FIGURE 1
# Aliasing breaks the shift-invariance Part X measures at float32 rounding.

def binomial_kernel(k=5):
    v = np.array([1.0])
    for _ in range(k - 1):
        v = np.convolve(v, [1.0, 1.0])
    v /= v.sum()
    return np.outer(v, v)


def circ_conv2(img, ker):
    """Circular 2-D convolution through the frequency domain — exact, so any
    shift discrepancy measured later is aliasing and not a boundary effect."""
    H, W = img.shape
    kh, kw = ker.shape
    pad = np.zeros((H, W))
    pad[:kh, :kw] = ker
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


def stack_features(img, stages, rng, antialias, blur):
    """A fixed random convolutional stack with `stages` stride-2 downsamples,
    read out by global average pooling. Pooling a translation-equivariant map is
    exactly shift-invariant, so anything that moves is the downsampling."""
    x = [img]
    for s in range(stages):
        kers = rng.standard_normal((4, 3, 3)) / 3
        nxt = []
        for ch in x:
            for k in kers[:2]:
                y = np.maximum(circ_conv2(ch, k), 0)
                if antialias:
                    y = circ_conv2(y, blur)
                nxt.append(y[::2, ::2])
        x = nxt
    return np.array([c.mean() for c in x])


def figure1():
    img = grace()
    blur = binomial_kernel(5)
    shifts = range(8)
    res = {}
    for aa in (False, True):
        curve = []
        for stages in (1, 2, 3, 4):
            rng = np.random.default_rng(0)          # same weights for both arms
            vs = np.array([stack_features(np.roll(img, s, axis=1), stages,
                                          np.random.default_rng(0), aa, blur)
                           for s in shifts])
            scale = np.abs(vs).mean()
            dev = np.abs(vs - vs.mean(0)).max() / scale
            curve.append(dev)
        res[aa] = np.array(curve)
        print(f"  antialias={aa}: " + "  ".join(f"{v:.2e}" for v in curve))
    f32 = np.finfo(np.float32).eps
    print(f"  float32 eps = {f32:.2e}")
    print(f"  ratio at 4 stages: {res[False][-1] / res[True][-1]:.2f}x")

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    st = [1, 2, 3, 4]
    ax.plot(st, res[False], "o-", color=ORANGE, lw=1.9, ms=5, label="stride-2 downsampling")
    ax.plot(st, res[True], "o-", color=GREEN, lw=1.9, ms=5, label="low-pass, then stride-2")
    ax.axhline(f32, color=GRAY, lw=1.0, ls="--")
    ax.text(4.05, f32, " float32 rounding", va="center", fontsize=8, color=GRAY)
    ax.set_yscale("log"); ax.set_xticks(st)
    ax.set_xlabel("stride-2 stages"); ax.set_ylabel("relative change under a one-pixel shift")
    ax.set_title("Global average pooling is shift-invariant. Downsampling is not.",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax); ax.set_xlim(0.8, 4.9)
    save(fig, "01-aliasing")
    return res, f32


# ============================================================ FIGURE 2
# The FFT: exact, and n log n.

def figure2():
    errs, ns = [], [2 ** k for k in range(3, 12)]
    rng = np.random.default_rng(0)
    for n in ns:
        x = rng.standard_normal(n)
        a, b = fft_radix2(x), dft_naive(x)
        errs.append(np.abs(a - b).max() / np.abs(b).max())
    print(f"  max relative error over n=8..2048: {max(errs):.2e}")

    tn = [2 ** k for k in range(6, 14)]
    t_naive, t_mine, t_np = [], [], []
    for n in tn:
        x = rng.standard_normal(n)
        W = dft_matrix(n)
        for store, fn in ((t_naive, lambda: W @ x), (t_mine, lambda: fft_radix2(x)),
                          (t_np, lambda: np.fft.fft(x))):
            reps = max(3, int(3e6 / (n * np.log2(n))))
            fn(); t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            store.append((time.perf_counter() - t0) / reps)
    fit = lambda t, k: np.polyfit(np.log(tn[-k:]), np.log(t[-k:]), 1)[0]
    sN, sM = fit(t_naive, 4), fit(t_mine, 4)
    ratios = [a / b for a, b in zip(t_naive, t_mine)]
    print(f"  fitted exponent over the largest four sizes — naive {sN:.2f} "
          f"(theory 2.00), radix-2 {sM:.2f} (theory {1 + 1/np.log(tn[-1]):.2f})")
    print(f"  full-range fits — naive {fit(t_naive, len(tn)):.2f}, radix-2 {fit(t_mine, len(tn)):.2f}")
    print(f"  ratio naive/radix-2: " + " ".join(f"{r:.1f}" for r in ratios))
    print(f"  at n={tn[-1]}: naive {t_naive[-1]*1e3:.1f} ms, radix-2 {t_mine[-1]*1e3:.2f} ms, "
          f"numpy {t_np[-1]*1e3:.3f} ms")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a1.plot(ns, np.maximum(errs, 1e-18), "o-", color=BLUE, lw=1.8, ms=4.5)
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_ylim(1e-18, 1e-8)
    a1.axhline(np.finfo(float).eps, color=GRAY, ls="--", lw=1.0)
    a1.text(ns[0], np.finfo(float).eps * 1.4, " float64 eps", fontsize=8, color=GRAY)
    a1.set_xlabel("n"); a1.set_ylabel("relative error against the definition")
    a1.set_title("It computes the same thing", fontsize=10, pad=8)
    tidy(a1)

    a2.plot(tn, t_naive, "o-", color=ORANGE, lw=1.8, ms=4,
            label=f"the definition  (slope {sN:.2f})")
    a2.plot(tn, t_mine, "o-", color=GREEN, lw=1.8, ms=4,
            label=f"radix-2, written out  ({sM:.2f})")
    a2.plot(tn, t_np, "o-", color=BLUE, lw=1.8, ms=4, label="numpy")
    a2.set_xscale("log", base=2); a2.set_yscale("log")
    a2.set_xlabel("n"); a2.set_ylabel("seconds per transform")
    a2.set_title("At a cost that scales differently", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a2)
    fig.tight_layout()
    save(fig, "02-fft")
    return max(errs), sN, sM, ratios


# ============================================================ FIGURE 3
# Convolution through the frequency domain, and where it starts to pay.

def fft_convolve(x, k):
    n = len(x) + len(k) - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(k, m), m)[:n]


def figure3():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1 << 16)
    ks = [2 ** j for j in range(1, 13)]
    t_dir, t_fft, err = [], [], []
    for kk in ks:
        k = rng.standard_normal(kk)
        a = np.convolve(x, k)
        b = fft_convolve(x, k)
        err.append(np.abs(a - b).max() / np.abs(a).max())
        for store, fn in ((t_dir, lambda: np.convolve(x, k)),
                          (t_fft, lambda: fft_convolve(x, k))):
            fn(); reps = 3 if kk > 256 else 8
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            store.append((time.perf_counter() - t0) / reps)
    t_dir, t_fft = np.array(t_dir), np.array(t_fft)
    cross = None
    for i in range(len(ks)):
        if t_fft[i] < t_dir[i]:
            cross = ks[i]; break
    print(f"  max relative disagreement: {max(err):.2e}")
    print(f"  FFT convolution overtakes direct at kernel size {cross}")
    print(f"  at k=4096: direct {t_dir[-1]*1e3:.1f} ms, FFT {t_fft[-1]*1e3:.1f} ms "
          f"({t_dir[-1]/t_fft[-1]:.0f}x)")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ks, t_dir * 1e3, "o-", color=ORANGE, lw=1.9, ms=4.5, label="direct convolution")
    ax.plot(ks, t_fft * 1e3, "o-", color=BLUE, lw=1.9, ms=4.5, label="through the FFT")
    if cross:
        ax.axvline(cross, color=GRAY, ls="--", lw=1.0)
        ax.text(cross * 1.12, t_fft.min() * 1e3 * 1.2, f"crossover\nk = {cross}",
                fontsize=8, color=GRAY)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("kernel length"); ax.set_ylabel("milliseconds  (signal length $2^{16}$)")
    ax.set_title("The same answer, by two routes with different costs", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "03-convolution")
    return cross, max(err), t_dir[-1] / t_fft[-1]


# ============================================================ FIGURE 4
# Magnitude and phase are not equal partners.

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def figure4():
    A, B = grace(), terrain()
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    mixAB = np.real(np.fft.ifft2(np.abs(FA) * np.exp(1j * np.angle(FB))))
    mixBA = np.real(np.fft.ifft2(np.abs(FB) * np.exp(1j * np.angle(FA))))
    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, A.shape)
    ph = (ph - ph[::-1, ::-1]) / 2                 # keep it a real-valued image
    randA = np.real(np.fft.ifft2(np.abs(FA) * np.exp(1j * ph)))

    cAB_A, cAB_B = corr(mixAB, A), corr(mixAB, B)
    cBA_A, cBA_B = corr(mixBA, A), corr(mixBA, B)
    cR = corr(randA, A)
    print(f"  |A| with phase(B): corr to A {cAB_A:+.3f}, to B {cAB_B:+.3f}")
    print(f"  |B| with phase(A): corr to A {cBA_A:+.3f}, to B {cBA_B:+.3f}")
    print(f"  |A| with random phase: corr to A {cR:+.3f}")

    fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.95))
    panels = [(A, "A"), (B, "B"), (mixAB, "|A|, phase of B"),
              (mixBA, "|B|, phase of A"), (randA, "|A|, random phase")]
    for ax, (im, t) in zip(axes, panels):
        lo, hi = np.percentile(im, [1, 99])
        ax.imshow(np.clip(im, lo, hi), cmap="gray", aspect="equal")
        ax.set_title(t, fontsize=8.5, pad=4)
        bare(ax)
    fig.tight_layout(w_pad=0.6)
    save(fig, "04-phase")
    return cAB_A, cAB_B, cBA_A, cBA_B, cR


# ============================================================ FIGURE 5
# There is no window that is good at both.

def stft(x, win, hop):
    w = np.hanning(win)
    frames = 1 + (len(x) - win) // hop
    S = np.empty((frames, win // 2 + 1))
    for i in range(frames):
        seg = x[i * hop:i * hop + win] * w
        S[i] = np.abs(np.fft.rfft(seg))
    return S


def width_at_half(v, axis_scale):
    """Full width at half maximum of a unimodal profile, in axis units."""
    v = np.asarray(v, float)
    pk = v.argmax()
    half = v[pk] / 2
    lo = pk
    while lo > 0 and v[lo] > half:
        lo -= 1
    hi = pk
    while hi < len(v) - 1 and v[hi] > half:
        hi += 1
    return (hi - lo) * axis_scale


def figure5():
    fs = 4000
    n = 4096
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 400 * t)
    x[n // 2] += 12.0                                  # a click, ideally localised
    wins = [64, 256, 1024]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    rows = []
    for ax, win in zip(axes, wins):
        hop = win // 8
        S = stft(x, win, hop)
        freqs = np.fft.rfftfreq(win, 1 / fs)
        times = np.arange(S.shape[0]) * hop / fs
        ax.imshow(S.T ** 0.4, origin="lower", aspect="auto", cmap="magma",
                  extent=[times[0], times[-1], freqs[0], freqs[-1]])
        ax.set_title(f"window = {win} samples", fontsize=9, pad=5)
        ax.set_xlabel("seconds", fontsize=8.5)
        ax.tick_params(labelsize=8)
        # frequency resolution: width of the 400 Hz ridge, in a frame with no click
        col = S[2]
        df = width_at_half(col, freqs[1] - freqs[0])
        # time resolution: width of the click's energy, in a high band the tone misses
        band = S[:, freqs > 1200].sum(1)
        dt = width_at_half(band, hop / fs)
        rows.append((win, df, dt, df * dt))
        ax.text(0.03, 0.94, f"$\\Delta f$ = {df:.0f} Hz\n$\\Delta t$ = {dt*1e3:.1f} ms",
                transform=ax.transAxes, fontsize=8, color="white", va="top")
    axes[0].set_ylabel("Hz", fontsize=8.5)
    for w, df, dt, p in rows:
        print(f"  window {w:5d}:  df = {df:7.1f} Hz   dt = {dt*1e3:6.2f} ms   product = {p:.3f}")
    prods = [r[3] for r in rows]
    print(f"  product range {min(prods):.3f}-{max(prods):.3f} "
          f"while df varies {rows[0][1]/rows[-1][1]:.0f}x")
    fig.tight_layout(w_pad=0.5)
    save(fig, "05-window")
    return rows


# ============================================================ FIGURE 7
# Spectral bias: low frequencies first, and it is frequency that decides.

def train_1d(x, y, steps, hidden=256, lr=2e-3, seed=0, probe_every=25, probe=None):
    r = np.random.default_rng(seed)
    X = x[:, None]
    W1 = r.standard_normal((1, hidden)) * np.sqrt(2.0)
    b1 = np.zeros(hidden)
    W2 = r.standard_normal((hidden, hidden)) * np.sqrt(2 / hidden)
    b2 = np.zeros(hidden)
    W3 = r.standard_normal((hidden, 1)) * np.sqrt(1 / hidden)
    b3 = np.zeros(1)
    ps = [W1, b1, W2, b2, W3, b3]
    ms = [np.zeros_like(p) for p in ps]
    vs = [np.zeros_like(p) for p in ps]
    Y = y[:, None]
    out = []
    for s in range(1, steps + 1):
        h1 = np.maximum(X @ ps[0] + ps[1], 0)
        h2 = np.maximum(h1 @ ps[2] + ps[3], 0)
        p = h2 @ ps[4] + ps[5]
        d = (p - Y) * (2 / len(X))
        g5 = h2.T @ d; g6 = d.sum(0)
        d2 = (d @ ps[4].T) * (h2 > 0)
        g3 = h1.T @ d2; g4 = d2.sum(0)
        d1 = (d2 @ ps[2].T) * (h1 > 0)
        g1 = X.T @ d1; g2 = d1.sum(0)
        for i, g in enumerate([g1, g2, g3, g4, g5, g6]):
            ms[i] = 0.9 * ms[i] + 0.1 * g
            vs[i] = 0.999 * vs[i] + 0.001 * g * g
            mh = ms[i] / (1 - 0.9 ** s); vh = vs[i] / (1 - 0.999 ** s)
            ps[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if probe is not None and s % probe_every == 0:
            out.append((s, probe(p.ravel())))
    return out


def spectral_run(freqs, amps, steps=20000, seed=0):
    n = 512
    x = np.linspace(0, 1, n, endpoint=False)
    y = sum(a * np.sin(2 * np.pi * f * x) for f, a in zip(freqs, amps))
    idx = np.array(freqs)

    def probe(pred):
        c = np.fft.rfft(pred) * 2 / n
        return np.abs(c[idx])

    tgt = np.abs(np.fft.rfft(y) * 2 / n)[idx]
    hist = train_1d(x, y, steps, seed=seed, probe=probe)
    S = np.array([h[0] for h in hist])
    A = np.array([h[1] for h in hist]) / tgt
    return S, A, tgt


def figure6():
    freqs = [1, 2, 4, 8, 16]
    S1, A1, t1 = spectral_run(freqs, [1.0] * 5, seed=0)
    S2, A2, t2 = spectral_run(freqs, [1.0, 1.0, 1.0, 1.0, 4.0], seed=0)

    def t90(S, A, j):
        w = np.nonzero(A[:, j] > 0.9)[0]
        return int(S[w[0]]) if len(w) else None

    r1 = [t90(S1, A1, j) for j in range(5)]
    r2 = [t90(S2, A2, j) for j in range(5)]
    fmt = lambda r, A: ", ".join(
        f"f={f}: {v} steps" if v else f"f={f}: never ({A[-1, j]:.2f} of target)"
        for j, (f, v) in enumerate(zip(freqs, r)))
    print("  equal amplitudes         — steps to 90%: " + fmt(r1, A1))
    print("  highest frequency loudest — steps to 90%: " + fmt(r2, A2))
    print(f"  final levels, equal amplitudes: {np.round(A1[-1], 3)}")
    print(f"  final levels, loud f=16:       {np.round(A2[-1], 3)}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    cols = [BLUE, CYAN, GREEN, ORANGE, PINK]
    for ax, S, A, ttl in ((a1, S1, A1, "equal amplitudes"),
                          (a2, S2, A2, "the highest frequency given 4x the amplitude")):
        for j, f in enumerate(freqs):
            ax.plot(S, A[:, j], color=cols[j], lw=1.8, label=f"f = {f}")
        ax.axhline(1.0, color=GRAY, lw=0.9, ls="--")
        ax.set_xlabel("training step")
        ax.set_title(ttl, fontsize=10, pad=8)
        ax.set_ylim(-0.05, 1.35)
        tidy(ax)
    a1.set_ylabel("amplitude recovered / target")
    a1.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "07-spectral-bias")
    return freqs, r1, r2, A1[-1], A2[-1]


# ============================================================ FIGURE 6
# What natural images look like in frequency, and what a careless resize does.

def radial_spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    P = np.abs(F) ** 2
    H, W = img.shape
    y, x = np.indices((H, W))
    r = np.hypot(y - H / 2, x - W / 2).astype(int)
    tot = np.bincount(r.ravel(), P.ravel())
    cnt = np.bincount(r.ravel())
    prof = tot[:min(H, W) // 2] / np.maximum(cnt[:min(H, W) // 2], 1)
    return prof


def crops(img, size=128):
    """Non-overlapping tiles at the image's own resolution."""
    H, W = img.shape
    return [img[i:i + size, j:j + size]
            for i in range(0, H - size + 1, size)
            for j in range(0, W - size + 1, size)]


def verify_radial_estimator():
    """The exponent Figure 7 reports is only worth quoting if the estimator
    recovers a known one. Run on every build."""
    rng = np.random.default_rng(0)
    out = []
    for alpha in (2.0, 3.0):
        n = 128
        fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
        r = np.hypot(fy, fx); r[0, 0] = 1e-9
        img = np.real(np.fft.ifft2((r ** (-alpha / 2)) *
                                   np.exp(1j * rng.uniform(0, 2 * np.pi, (n, n)))))
        prof = radial_spectrum(img); f = np.arange(len(prof))
        got = np.polyfit(np.log(f[3:len(prof) - 5]), np.log(prof[3:len(prof) - 5]), 1)[0]
        out.append((alpha, got))
        assert abs(got + alpha) < 0.15, f"estimator off: wanted -{alpha}, got {got:.2f}"
    print("  estimator check — " + ", ".join(f"true -{a:.0f} recovered {g:.2f}" for a, g in out))
    return out


def figure7():
    verify_radial_estimator()
    tiles = crops(grace()) + crops(dem_native()) + crops(scan())
    profs = np.array([radial_spectrum(c) for c in tiles])
    prof = profs.mean(0)
    f = np.arange(len(prof))
    lo, hi = 3, len(prof) - 5
    slope, icept = np.polyfit(np.log(f[lo:hi]), np.log(prof[lo:hi]), 1)
    rng = np.random.default_rng(0)
    noise = np.array([radial_spectrum(rng.standard_normal((128, 128))) for _ in range(8)]).mean(0)
    ns = np.polyfit(np.log(f[lo:hi]), np.log(noise[lo:hi]), 1)[0]
    per = [np.polyfit(np.log(f[lo:hi]), np.log(pr[lo:hi]), 1)[0] for pr in profs]
    print(f"  natural images: power ~ f^({slope:.2f}) over {len(tiles)} tiles; "
          f"per-tile exponent {np.min(per):.2f} to {np.max(per):.2f}, "
          f"sd {np.std(per, ddof=1):.2f}")
    print(f"  white noise:    power ~ f^({ns:+.2f})")

    big = grace()
    naive = big[::4, ::4]
    filt = circ_conv2(big, binomial_kernel(9))[::4, ::4]
    pn, pf = radial_spectrum(naive), radial_spectrum(filt)
    fh = np.arange(len(pn))
    tail = slice(len(pn) // 2, len(pn) - 2)
    excess = float(np.mean(pn[tail] / pf[tail]))
    print(f"  aliased power in the top half of the band: {excess:.1f}x the filtered version")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a1.loglog(f[1:], prof[1:], color=BLUE, lw=1.9, label=f"natural images  ($f^{{{slope:.2f}}}$)")
    a1.loglog(f[lo:hi], np.exp(icept) * f[lo:hi] ** slope, color=INK, lw=1.0, ls="--")
    a1.loglog(f[1:], noise[1:], color=ORANGE, lw=1.9, label=f"white noise  ($f^{{{ns:+.2f}}}$)")
    a1.set_xlabel("spatial frequency (cycles per image)")
    a1.set_ylabel("power")
    a1.set_title("Natural images are dominated by low frequencies", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    tidy(a1)

    a2.loglog(fh[1:], pf[1:], color=GREEN, lw=1.9, label="low-pass, then subsample")
    a2.loglog(fh[1:], pn[1:], color=ORANGE, lw=1.9, label="subsample directly")
    a2.set_xlabel("spatial frequency after 4x downsampling")
    a2.set_ylabel("power")
    a2.set_title("What the discarded frequencies do instead", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    tidy(a2)
    fig.tight_layout()
    save(fig, "06-natural-spectrum")
    return slope, ns, excess


if __name__ == "__main__":
    print("Figure 1 — aliasing and shift-invariance")
    figure1()
    print("Figure 2 — the FFT")
    figure2()
    print("Figure 3 — convolution through the frequency domain")
    figure3()
    print("Figure 4 — magnitude and phase")
    figure4()
    print("Figure 5 — the time-frequency trade")
    figure5()
    print("Figure 6 — the spectrum of natural images")
    figure7()
    print("Figure 7 — spectral bias")
    figure6()
    print("done")
