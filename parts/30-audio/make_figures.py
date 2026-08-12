"""
Figures for Part XXX — Perception: Audio and Multimodality.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about three minutes. Numpy and matplotlib only.

This part spends Part XXVII. The short-time transform, the window trade and the
phase result are all used here rather than re-derived, and Figure 2 is the audio
counterpart of Part XXVII Figure 4.
"""
import os
import re
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "au"))
os.makedirs(OUT, exist_ok=True)

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


# ------------------------------------------------------------------- audio

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, fs, fmin=0.0, fmax=None):
    fmax = fmax if fmax is not None else fs / 2
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(mid - lo, 1e-12)
        dn = (hi - freqs) / max(hi - mid, 1e-12)
        fb[m] = np.clip(np.minimum(up, dn), 0, None)
    return fb, edges, freqs


def stft(x, win, hop):
    w = np.hanning(win + 1)[:-1]
    frames = 1 + (len(x) - win) // hop
    return np.array([np.fft.rfft(x[i * hop:i * hop + win] * w) for i in range(frames)])


def istft(S, win, hop, length):
    w = np.hanning(win + 1)[:-1]
    out = np.zeros(length)
    norm = np.zeros(length)
    for i, spec in enumerate(S):
        seg = np.fft.irfft(spec, win)
        out[i * hop:i * hop + win] += seg * w
        norm[i * hop:i * hop + win] += w ** 2
    return out / np.maximum(norm, 1e-8)


# ============================================================ FIGURE 1
# What the mel scale throws away.

def figure1():
    fs, n_fft, n_mels = 16000, 1024, 64
    fb, edges, freqs = mel_filterbank(n_mels, n_fft, fs)
    width = np.array([(fb[m] > 0).sum() for m in range(n_mels)], float)
    centres = edges[1:-1]
    bw = edges[2:] - edges[:-2]                     # -3 dB-ish width in Hz
    print(f"  lowest band pools {width[0]:.0f} FFT bins, highest {width[-1]:.0f}"
          f"  -> {width[-1]/max(width[0],1):.1f}x")
    print(f"  bandwidth {bw[0]:.0f} Hz at {centres[0]:.0f} Hz, "
          f"{bw[-1]:.0f} Hz at {centres[-1]:.0f} Hz")

    # How many mel bands does a fixed hertz separation span? Counting spectral
    # peaks was the first criterion and it is not monotone -- at high frequency
    # two wide adjacent bands can still form two local maxima, so the "crossover"
    # it reported was an artefact of the heuristic rather than a property of the
    # filterbank. Bands spanned is exact and monotone. Corrected before the
    # figure's claim was written.
    sep = 200.0
    bands = np.array([(hz_to_mel(f + sep) - hz_to_mel(f))
                      / ((hz_to_mel(fs / 2) - hz_to_mel(0)) / (n_mels + 1))
                      for f in centres])
    below = np.nonzero(bands < 1.0)[0]
    edge_f = float(centres[below[0]]) if len(below) else None
    print(f"  a {sep:.0f} Hz separation spans {bands[0]:.1f} mel bands at "
          f"{centres[0]:.0f} Hz and {bands[-1]:.2f} at {centres[-1]:.0f} Hz")
    print(f"  it drops below one band above about {edge_f:.0f} Hz" if edge_f
          else "  it never drops below one band")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.plot(centres, width, color=BLUE, lw=2.0)
    a1.set_xlabel("mel band centre (Hz)"); a1.set_ylabel("FFT bins pooled")
    a1.set_title("Resolution is spent at the top", fontsize=10, pad=8)
    tidy(a1)
    a2.semilogy(centres, bands, color=ORANGE, lw=2.0)
    a2.axhline(1.0, color=GREEN, lw=1.5, ls="--")
    a2.text(centres[2], 1.15, "one band", fontsize=8, color=GREEN)
    if edge_f:
        a2.axvline(edge_f, color=GRAY, lw=1.0, ls=":")
        a2.text(edge_f * 1.05, bands.max() * 0.25,
                f"a {sep:.0f} Hz gap falls\ninside one band\nabove {edge_f:.0f} Hz",
                fontsize=7.6, color=GRAY)
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel(f"mel bands spanned by {sep:.0f} Hz")
    a2.set_title("What that costs", fontsize=10, pad=8)
    tidy(a2)
    fig.tight_layout()
    save(fig, "01-mel")
    return width, bands, edge_f


# ============================================================ FIGURE 2
# Magnitude is not enough.

def griffin_lim(mag, win, hop, length, iters, seed=0):
    r = np.random.default_rng(seed)
    ang = r.uniform(-np.pi, np.pi, mag.shape)
    S = mag * np.exp(1j * ang)
    errs = []
    for k in range(iters):
        x = istft(S, win, hop, length)
        S2 = stft(x, win, hop)
        errs.append(float(np.linalg.norm(np.abs(S2) - mag) / np.linalg.norm(mag)))
        S = mag * np.exp(1j * np.angle(S2))
    return istft(S, win, hop, length), np.array(errs)


def interior(x, win):
    """Drop one window from each end. Overlap-add gives those samples fewer
    contributing frames, so their error is boundary handling rather than
    anything to do with phase."""
    return np.asarray(x)[win:len(x) - win]


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def figure2():
    fs, win, hop = 8000, 512, 128
    n = 8192
    t = np.arange(n) / fs
    r = np.random.default_rng(0)
    # a voiced-like signal: a moving fundamental with harmonics, plus a transient
    f0 = 140 + 40 * np.sin(2 * np.pi * 1.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    x = sum(np.sin(k * ph) / k for k in range(1, 9))
    x[n // 2:n // 2 + 40] += 3.0 * np.hanning(40)
    x /= np.abs(x).max()

    S = stft(x, win, hop)
    mag = np.abs(S)
    exact = istft(S, win, hop, n)
    xi, ei = interior(x, win), interior(exact, win)
    print(f"  with the true phase: correlation {corr(ei, xi):.9f}, "
          f"max abs error {np.abs(ei - xi).max():.2e} (interior)")

    r2 = np.random.default_rng(1)
    rand = istft(mag * np.exp(1j * r2.uniform(-np.pi, np.pi, mag.shape)), win, hop, n)
    print(f"  with random phase:   correlation {corr(interior(rand, win), xi):+.4f}")

    iters = [1, 2, 5, 10, 25, 50, 100, 200]
    cs, es = [], []
    y, err = griffin_lim(mag, win, hop, n, max(iters))
    for k in iters:
        yk, ek = griffin_lim(mag, win, hop, n, k)
        cs.append(abs(corr(interior(yk, win), xi))); es.append(ek[-1])
    print("  iterative phase estimation:")
    for k, c, e in zip(iters, cs, es):
        print(f"    {k:4d} iters: |correlation| {c:.4f}   spectrogram error {e:.4f}")
    print(f"  monotone in spectrogram error: {all(np.diff(err) <= 1e-12)}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.semilogx(iters, cs, "o-", color=BLUE, lw=1.9, ms=4.5, label="estimated phase")
    a1.axhline(abs(corr(ei, xi)), color=GREEN, lw=1.5, ls="--", label="true phase")
    a1.axhline(abs(corr(interior(rand, win), xi)), color=ORANGE, lw=1.5, ls=":",
               label="random phase")
    a1.set_ylim(-0.03, 1.05)
    a1.set_xlabel("iterations"); a1.set_ylabel("|correlation| with the original")
    a1.set_title("Recovering the waveform", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="center right")
    tidy(a1)
    a2.semilogy(np.arange(1, len(err) + 1), err, color=PURPLE, lw=1.9)
    a2.set_xlabel("iterations"); a2.set_ylabel("relative spectrogram error")
    a2.set_title("And what it converges to", fontsize=10, pad=8)
    tidy(a2)
    fig.tight_layout()
    save(fig, "02-phase")
    return corr(ei, xi), corr(interior(rand, win), xi), cs, es


# ============================================================ FIGURE 3
# CTC against enumeration.

def ctc_expand(target, blank=0):
    out = [blank]
    for s in target:
        out += [s, blank]
    return np.array(out)


def ctc_forward(logits, target, blank=0):
    """Returns log P(target | logits) by the standard forward recursion."""
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    ext = ctc_expand(target, blank)
    T, S = len(lp), len(ext)
    a = np.full(S, -np.inf)
    a[0] = lp[0, ext[0]]
    if S > 1:
        a[1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = a.copy()
        a = np.full(S, -np.inf)
        for s in range(S):
            best = prev[s]
            if s > 0:
                best = np.logaddexp(best, prev[s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                best = np.logaddexp(best, prev[s - 2])
            a[s] = best + lp[t, ext[s]]
    return float(np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1])


def ctc_enumerate(logits, target, blank=0):
    """Sum over every alignment, by brute force."""
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    T, V = lp.shape
    tot = -np.inf
    count = 0
    for path in itertools.product(range(V), repeat=T):
        collapsed = []
        prev = None
        for s in path:
            if s != prev and s != blank:
                collapsed.append(s)
            prev = s
        if collapsed == list(target):
            count += 1
            tot = np.logaddexp(tot, sum(lp[t, s] for t, s in enumerate(path)))
    return float(tot), count


def ctc_alignment_count(T, target, V, blank=0):
    """How many frame-label paths collapse to `target`. Counted by the same
    recursion as the forward algorithm with all probabilities set to one, so it
    reaches lengths brute force cannot."""
    ext = ctc_expand(target, blank)
    S = len(ext)
    a = np.zeros(S, dtype=object)
    a[0] = 1
    if S > 1:
        a[1] = 1
    for t in range(1, T):
        prev = a.copy()
        a = np.zeros(S, dtype=object)
        for s in range(S):
            v = prev[s]
            if s > 0:
                v += prev[s - 1]
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                v += prev[s - 2]
            a[s] = v
    return int(a[-1] + a[-2]) if S > 1 else int(a[-1])


def figure3():
    r = np.random.default_rng(0)
    V = 3                                   # blank + two symbols
    target = [1, 2]
    errs, counts, Ts = [], [], list(range(2, 10))
    for T in Ts:
        lg = r.standard_normal((T, V))
        a = ctc_forward(lg, target)
        b, n = ctc_enumerate(lg, target)
        errs.append(abs(a - b)); counts.append(n)
        if T in (4, 7, 9):
            print(f"  T={T}: forward {a:.10f}  enumeration {b:.10f}  "
                  f"|diff| {abs(a-b):.2e}   valid alignments {n}")
    print(f"  worst disagreement over T=2..9: {max(errs):.2e}")
    # the same check the other way: the DP count must match the enumerated one
    assert all(ctc_alignment_count(T, target, V) == c for T, c in zip(Ts, counts))
    Tbig = list(range(2, 81))
    big = [ctc_alignment_count(T, target, V) for T in Tbig]
    fwd_ops = [T * (2 * len(target) + 1) * 3 for T in Tbig]
    print(f"  alignments at T=9: {counts[-1]} (enumerated and counted agree)")
    print(f"  at T=80: {big[-1]:.3e} alignments against {fwd_ops[-1]} operations"
          f"  -> {big[-1]/fwd_ops[-1]:.2e}x")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.semilogy(Ts, np.maximum(errs, 1e-18), "o-", color=BLUE, lw=1.8, ms=4.5)
    a1.axhline(np.finfo(float).eps, color=GRAY, ls="--", lw=1.0)
    a1.text(Ts[0], np.finfo(float).eps * 1.6, " float64 eps", fontsize=8, color=GRAY)
    a1.set_ylim(1e-18, 1e-8)
    a1.set_xlabel("input length $T$"); a1.set_ylabel("|forward $-$ enumeration|")
    a1.set_title("The same probability", fontsize=10, pad=8)
    tidy(a1)
    a2.semilogy(Tbig, [float(b) for b in big], color=ORANGE, lw=2.0,
                label="alignments summed over")
    a2.semilogy(Tbig, fwd_ops, color=GREEN, lw=2.0, label="forward-algorithm operations")
    a2.plot(Ts, counts, "o", color=ORANGE, ms=4, label="verified by enumeration")
    a2.set_xlabel("input length $T$"); a2.set_ylabel("count")
    a2.set_title("Summed two ways", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a2)
    fig.tight_layout()
    save(fig, "03-ctc")
    return max(errs), counts, big, fwd_ops


# ============================================================ FIGURE 4
# What CTC's per-frame independence costs.

def figure4():
    """The target is an equal mixture of 'ab' and 'ba'. CTC's distribution
    factorises over frames, so any per-frame distribution that puts mass on both
    also puts mass on 'aa' and 'bb'. Measured by optimising the per-frame
    parameters directly — no network, so the result is about the family."""
    V = 3                                   # blank, a, b
    seqs = {"ab": [1, 2], "ba": [2, 1], "aa": [1, 1], "bb": [2, 2],
            "a": [1], "b": [2], "": []}
    best = None
    for T in (2, 3, 4, 5, 6):
        # optimise the T x V logits to put 0.5 on 'ab' and 0.5 on 'ba'
        rng = np.random.default_rng(0)
        lg = rng.standard_normal((T, V)) * 0.1
        for step in range(4000):
            def loss_and_grad(L):
                p = {k: np.exp(ctc_forward(L, v)) for k, v in seqs.items()}
                loss = (p["ab"] - 0.5) ** 2 + (p["ba"] - 0.5) ** 2
                g = np.zeros_like(L)
                eps = 1e-5
                for i in range(L.shape[0]):
                    for j in range(L.shape[1]):
                        L2 = L.copy(); L2[i, j] += eps
                        p2 = {k: np.exp(ctc_forward(L2, v)) for k in ("ab", "ba")
                              for v in [seqs[k]]}
                        l2 = (p2["ab"] - 0.5) ** 2 + (p2["ba"] - 0.5) ** 2
                        g[i, j] = (l2 - loss) / eps
                return loss, g
            loss, g = loss_and_grad(lg)
            lg -= 8.0 * g
            if loss < 1e-8:
                break
        p = {k: float(np.exp(ctc_forward(lg, v))) for k, v in seqs.items()}
        # The leak is everything NOT on the two targets. Note where it does NOT
        # go: (a,a) collapses to 'a' under CTC, not to 'aa', so the wrong outputs
        # that receive the mass are the single-symbol ones. Predicting "aa" and
        # "bb" was a mis-reading of the collapse rule.
        got = p["ab"] + p["ba"]
        leak = 1.0 - got
        print(f"  T={T}: P(ab)={p['ab']:.4f}  P(ba)={p['ba']:.4f}"
              f"  -> on target {got:.4f};  P(a)={p['a']:.4f} P(b)={p['b']:.4f}"
              f" P(aa)+P(bb)={p['aa']+p['bb']:.4f};  leaked {leak:.4f}")
        if best is None or leak < best[1]:
            best = (T, leak, dict(p))
    print(f"  smallest achievable leak over T=2..6: {best[1]:.4f} at T={best[0]}"
          f"  -- the target asks for 0, and half the mass cannot be moved")

    T, leak, p = best
    order = ["ab", "ba", "aa", "bb", "a", "b", ""]
    labels = ["ab", "ba", "aa", "bb", "a", "b", "(empty)"]
    want = [0.5, 0.5, 0, 0, 0, 0, 0]
    got = [p[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(order))
    ax.bar(x - 0.2, want, 0.38, color=GREEN, label="target distribution")
    ax.bar(x + 0.2, got, 0.38, color=ORANGE, label=f"best CTC model ($T$ = {T})")
    for i, v in enumerate(got):
        if v > 0.01:
            ax.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("probability")
    ax.set_title("An equal mixture of two sequences, and what CTC can represent",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "04-independence")
    return best


# ============================================================ FIGURE 5
# Quantisation.

def kmeans(X, k, iters=40, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return C, d.min(1).mean()


def scalar_quantise(X, levels, seed=0):
    tot = 0.0
    for j in range(X.shape[1]):
        C, e = kmeans(X[:, j:j + 1], levels, seed=seed)
        tot += e
    return tot


def figure5():
    n, d = 4000, 4
    bits_per_dim = [1, 2, 3]
    rows = {}
    for rho in (0.0, 0.6, 0.9):
        S = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
        L = np.linalg.cholesky(S)
        X = np.random.default_rng(0).standard_normal((n, d)) @ L.T
        sq, vq, rq = [], [], []
        for b in bits_per_dim:
            sq.append(scalar_quantise(X, 2 ** b))
            _, e = kmeans(X, 2 ** (b * d) if b * d <= 10 else 1024, seed=1)
            vq.append(e)
            # residual quantisation: b stages of a small codebook
            resid = X.copy(); tot = 0.0
            for _ in range(b * d // 4):
                C, _ = kmeans(resid, 16, seed=2)
                idx = ((resid[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
                resid = resid - C[idx]
            rq.append(float((resid ** 2).sum(1).mean()))
        rows[rho] = (np.array(sq), np.array(vq), np.array(rq))
        print(f"  rho={rho:.1f}:  scalar {sq[1]:.4f}   vector {vq[1]:.4f}   "
              f"residual {rq[1]:.4f}   (2 bits/dim)   vq advantage {sq[1]/vq[1]:.2f}x")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for rho, c in zip(rows, (BLUE, GREEN, ORANGE)):
        sq, vq, rq = rows[rho]
        ax.plot(bits_per_dim, sq, "o-", color=c, lw=1.9, ms=4.5, label=f"scalar, $\\rho$={rho}")
        ax.plot(bits_per_dim, vq, "s--", color=c, lw=1.5, ms=4, alpha=0.85,
                label=f"vector, $\\rho$={rho}")
    ax.set_yscale("log")
    ax.set_xticks(bits_per_dim)
    ax.set_xlabel("bits per dimension"); ax.set_ylabel("mean squared error")
    ax.set_title("What a vector quantiser buys, and when", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=7.4, ncol=2)
    tidy(ax)
    save(fig, "05-quantisation")
    return rows


# ============================================================ FIGURE 6
# Temperature.

def figure6():
    d, n_neg = 64, 512
    r = np.random.default_rng(0)
    q = r.standard_normal(d); q /= np.linalg.norm(q)
    pos = q + 0.35 * r.standard_normal(d); pos /= np.linalg.norm(pos)
    neg = r.standard_normal((n_neg, d)); neg /= np.linalg.norm(neg, axis=1, keepdims=True)
    sims = neg @ q
    taus = np.logspace(-2, 0, 25)
    eff = []
    for tau in taus:
        w = np.exp(sims / tau)
        w /= w.sum()
        eff.append(float(np.exp(-(w * np.log(w + 1e-300)).sum())))   # perplexity
    for tau in (0.01, 0.05, 0.2, 1.0):
        w = np.exp(sims / tau); w /= w.sum()
        e = float(np.exp(-(w * np.log(w + 1e-300)).sum()))
        print(f"  tau={tau:5.2f}: effective negatives {e:8.1f} of {n_neg}"
              f"   top-1 share {w.max():.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.loglog(taus, eff, color=BLUE, lw=2.0)
    ax.axhline(n_neg, color=GRAY, ls="--", lw=1.0)
    ax.text(taus[0], n_neg * 1.1, f" all {n_neg} negatives", fontsize=8, color=GRAY)
    ax.set_xlabel("temperature")
    ax.set_ylabel("effective number of negatives")
    ax.set_title("What the temperature actually controls", fontsize=10.5, pad=10)
    tidy(ax)
    save(fig, "06-temperature")
    return taus, eff


# ============================================================ FIGURE 7
# Retrieval against gallery size.

def figure7():
    d = 64
    sizes = [100, 300, 1000, 3000, 10000, 30000]
    res = {}
    for noise, name in ((0.55, "weaker embedding"), (0.40, "stronger embedding")):
        rec = []
        for N in sizes:
            r = np.random.default_rng(N)
            A = r.standard_normal((N, d))
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            B = A + noise * r.standard_normal((N, d))
            B /= np.linalg.norm(B, axis=1, keepdims=True)
            q = min(400, N)
            sims = B[:q] @ A.T
            rec.append(float((sims.argmax(1) == np.arange(q)).mean()))
        res[name] = np.array(rec)
        print(f"  {name}: recall@1 {rec[0]:.3f} at N={sizes[0]} -> {rec[-1]:.3f} at N={sizes[-1]}")
    sl = {k: np.polyfit(np.log10(sizes), v, 1)[0] for k, v in res.items()}
    print(f"  slope per decade of gallery: " +
          ", ".join(f"{k} {v:+.3f}" for k, v in sl.items()))
    a, b = res["weaker embedding"], res["stronger embedding"]
    print(f"  ordering constant across the range: {bool(np.all(b > a))}")

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for (k, v), c in zip(res.items(), (ORANGE, GREEN)):
        ax.semilogx(sizes, v, "o-", color=c, lw=1.9, ms=5, label=f"{k}  ({sl[k]:+.3f}/decade)")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("gallery size"); ax.set_ylabel("recall@1")
    ax.set_title("The same embedding, a different number", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    tidy(ax)
    save(fig, "07-retrieval")
    return sizes, res, sl


# ============================================================ FIGURE 8
# ILLUSTRATIVE.
#
# The measured version of this figure did not elicit the phenomenon. The modality
# gap is attributed to deep towers whose outputs occupy a narrow cone at
# initialisation, with the contrastive loss preserving that separation. Two
# linear maps from Gaussian inputs are isotropic at initialisation — measured at
# +0.001 within-modality against -0.000 across — so there was no separation for
# training to preserve and nothing to measure. By the book's figure policy the figure is drawn as a
# schematic with NO numbers on either axis.
#
# What remains measured, and is reported in the prose instead, is that matched
# pairs are far closer to each other than to anything else (+0.994 against +0.002)
# and retrieval reaches 1.000 regardless of where the two clouds sit.

def figure8():
    r = np.random.default_rng(0)
    n = 60
    # two cones on a circle, with matched pairs at matched angles within them
    base = r.uniform(0, 2 * np.pi, n)
    spreadA, spreadB = 0.55, 0.55
    cA, cB = 0.9, 3.6                       # cone centres, deliberately separated
    angA = cA + spreadA * np.sin(base)
    angB = cB + spreadB * np.sin(base)      # same ordering, so pairs correspond
    A = np.stack([np.cos(angA), np.sin(angA)], 1)
    B = np.stack([np.cos(angB), np.sin(angB)], 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=LIGHT, lw=1.4, zorder=0)
    for i in range(0, n, 4):
        ax.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], color=GRAY, lw=0.6,
                alpha=0.45, zorder=1)
    ax.scatter(A[:, 0], A[:, 1], s=22, color=BLUE, zorder=2, label="one modality")
    ax.scatter(B[:, 0], B[:, 1], s=22, color=ORANGE, zorder=2, label="the other")
    ax.annotate("", xy=(np.cos(cA) * 1.28, np.sin(cA) * 1.28), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0, ls=":"))
    ax.annotate("", xy=(np.cos(cB) * 1.28, np.sin(cB) * 1.28), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0, ls=":"))
    ax.text(0.06, 0.10, "the gap", fontsize=9, color=INK, ha="left")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.35, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Two modalities on one sphere, and the pairs between them",
                 fontsize=10.5, pad=8)
    save(fig, "08-modality-gap")
    print("  ILLUSTRATIVE — no measured quantities on either axis")
    return None


if __name__ == "__main__":
    for i, fn in enumerate([figure1, figure2, figure3, figure4,
                            figure5, figure6, figure7, figure8], 1):
        print(f"Figure {i}")
        fn()
    print("done")
