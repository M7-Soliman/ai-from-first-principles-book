"""
Figures for Part XIII — Data.

Every claim this part makes that can be measured, is measured here. The
contamination inflation, the label-noise tolerance, MinHash's accuracy and the
three mechanisms of model collapse are all measured rather than asserted — and
the collapse figure exists because the folk version of the result did not
reproduce.

All data is synthetic and generated here.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the two that train

Runtime: about a minute for --cheap, ten for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "dt"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
CYAN, LIGHT, INK = "#0e7490", "#cbd5e1", "#1e293b"
MAX_PT = 900

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
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


# ---------------------------------------------------------- 1. contamination

def _knn1(Xtr, ytr, Xq):
    """1-nearest-neighbour classification, written out.

    This used scikit-learn. It is four lines of numpy, and depending on a
    package for it meant the figure could not be regenerated the day that
    package stopped being installed — which happened.
    """
    d = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return ytr[np.argmin(d, axis=1)]


def _logreg(X, y, iters=300, lam=1e-4):
    """Binary logistic regression by Newton-Raphson. Replaces a scikit-learn
    call for the same reason _knn1 does: a figure that cannot be regenerated
    without an optional package is a figure whose claim cannot be checked."""
    X = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - lam * b)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def _predict(X, b):
    X = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    return (X @ b > 0).astype(int)


def fig_contamination():
    rng = np.random.default_rng(0)
    n, d = 2400, 20
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w + rng.normal(scale=2.0, size=n) > 0).astype(int)
    Xtr, ytr, Xte, yte = X[:1600], y[:1600], X[1600:], y[1600:]

    fracs = np.array([0.0, 0.02, 0.05, 0.10, 0.25, 0.50])
    cont, honest = [], []
    for frac in fracs:
        k = int(frac * len(Xte))
        idx = rng.choice(len(Xte), k, replace=False)
        Xtr2 = np.vstack([Xtr, Xte[idx]])
        ytr2 = np.concatenate([ytr, yte[idx]])
        cont.append(float(np.mean(_knn1(Xtr2, ytr2, Xte) == yte)))
        mask = np.ones(len(Xte), bool)
        mask[idx] = False
        honest.append(float(np.mean(_knn1(Xtr2, ytr2, Xte[mask]) == yte[mask])))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(100 * fracs, cont, "o-", color=PINK, ms=5, lw=1.9, label="reported score")
    ax.plot(100 * fracs, honest, "o-", color=BLUE, ms=5, lw=1.9,
            label="score on the uncontaminated part")
    ax.set_xlabel("% of the test set also present in training")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.bar(range(len(fracs)), np.array(cont) - np.array(honest), 0.6, color=ORANGE)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{100*f:.0f}%" for f in fracs], fontsize=8.5)
    ax.set_xlabel("% contaminated")
    ax.set_ylabel("inflation of the reported score")
    tidy(ax)

    for f, c, h in zip(fracs, cont, honest):
        print(f"    {100*f:5.0f}% leaked | reported {c:.3f} | honest {h:.3f} | "
              f"inflation {c-h:+.3f}")
    print("    note the honest score FALLS as contamination rises — the leaked copies")
    print("    displace real training signal, so you are also getting a worse model")
    fig.tight_layout()
    save(fig, "03-contamination")


# ------------------------------------------------------------- 2. label noise

def fig_label_noise():
    rng = np.random.default_rng(1)
    n, d = 4000, 20
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w > 0).astype(int)
    Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]

    ps = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48])
    tr_acc, te_acc = [], []
    for p in ps:
        accs_tr, accs_te = [], []
        for s in range(5):
            r = np.random.default_rng(100 + s)
            yn = ytr.copy()
            flip = r.random(len(yn)) < p
            yn[flip] = 1 - yn[flip]
            w = _logreg(Xtr, yn)
            accs_tr.append(float(np.mean(_predict(Xtr, w) == yn)))
            accs_te.append(float(np.mean(_predict(Xte, w) == yte)))
        tr_acc.append(np.mean(accs_tr))
        te_acc.append(np.mean(accs_te))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(100 * ps, te_acc, "o-", color=BLUE, ms=5, lw=1.9,
            label="test accuracy, clean labels")
    ax.plot(100 * ps, tr_acc, "o-", color=ORANGE, ms=5, lw=1.9,
            label="training accuracy, noisy labels")
    ax.plot(100 * ps, 1 - ps, "--", color=GRAY, lw=1.3, label="the noise ceiling, $1-p$")
    ax.set_xlabel("% of training labels flipped")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.plot(100 * ps, np.array(te_acc) - np.array(tr_acc), "o-", color=PURPLE, ms=5, lw=1.9)
    ax.axhline(0, color=GRAY, ls=":", lw=1.1)
    ax.set_xlabel("% of training labels flipped")
    ax.set_ylabel("test accuracy minus training accuracy")
    ax.set_title("training accuracy stops meaning anything", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    for p, a, b in zip(ps, tr_acc, te_acc):
        print(f"    {100*p:4.0f}% noise | training acc {a:.3f} | test acc on CLEAN "
              f"labels {b:.3f}")
    i = list(ps).index(0.4)
    print(f"    at 40% noise the model still reaches {te_acc[i]:.3f} on clean labels "
          f"while its training accuracy is {tr_acc[i]:.3f}")
    fig.tight_layout()
    save(fig, "02-label-noise")


# ----------------------------------------------------------------- 3. MinHash

def _shingles(s, k=5):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _str_hash(x):
    """A deterministic replacement for Python's `hash()`.

    `hash()` on a string is salted per process (PYTHONHASHSEED), so a figure
    built on it reproduces a different number every run. blake2b has none of
    that: the same shingle always maps to the same integer, on any machine,
    in any process.
    """
    import hashlib
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little")


def _minhash(sig, n_perm, seed=0):
    r = np.random.default_rng(seed)
    a = r.integers(1, 2 ** 31 - 1, n_perm)
    b = r.integers(0, 2 ** 31 - 1, n_perm)
    hs = np.array([_str_hash(x) % (2 ** 31 - 1) for x in sig], dtype=np.int64)
    return np.array([((a[i] * hs + b[i]) % (2 ** 31 - 1)).min() for i in range(n_perm)])


def fig_minhash():
    rng = np.random.default_rng(2)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    pairs = []
    for edit in range(0, 61, 4):
        base = " ".join(rng.choice(words, 40))
        other = base[:max(1, len(base) - edit)] + "z" * edit
        A, B = _shingles(base), _shingles(other)
        pairs.append((len(A & B) / len(A | B), A, B))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for n_perm, c in ((32, LIGHT), (64, "#93c5fd"), (256, BLUE)):
        est = [(_minhash(A, n_perm, 7) == _minhash(B, n_perm, 7)).mean()
               for _, A, B in pairs]
        ax.plot([t for t, _, _ in pairs], est, "o", color=c, ms=4,
                label=f"{n_perm} permutations")
    lim = [0, 1]
    ax.plot(lim, lim, "--", color=GRAY, lw=1.3, label="exact Jaccard")
    ax.set_xlabel("true Jaccard similarity")
    ax.set_ylabel("MinHash estimate")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    errs = {}
    for n_perm in (16, 32, 64, 128, 256, 512):
        e = [abs((_minhash(A, n_perm, 7) == _minhash(B, n_perm, 7)).mean() - t)
             for t, A, B in pairs]
        errs[n_perm] = np.mean(e)
    ks = sorted(errs)
    ax.loglog(ks, [errs[k] for k in ks], "o-", color=PINK, ms=5, lw=1.9,
              label="measured mean error")
    ax.loglog(ks, [0.5 / np.sqrt(k) for k in ks], "--", color=GRAY, lw=1.3,
              label=r"$\propto 1/\sqrt{k}$")
    ax.set_xlabel("number of permutations $k$")
    ax.set_ylabel("mean absolute error")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    mean error with 256 permutations: {errs[256]:.4f}")
    print(f"    the error falls like 1/sqrt(k) — Part V §3 again, and it is why a few")
    print(f"    hundred permutations suffice to deduplicate a corpus of any size")
    fig.tight_layout()
    save(fig, "04-minhash")


# ------------------------------------------------------------ 4. model collapse

def fig_collapse():
    def run(n, gens, mode, seed):
        r = np.random.default_rng(seed)
        mu, sd = 0.0, 1.0
        out = [1.0]
        for _ in range(gens):
            s = r.normal(mu, sd, n)
            if mode == "half":
                s = s[np.argsort(np.abs(s - s.mean()))[:len(s) // 2]]
            elif mode == "quarter":
                s = s[np.argsort(np.abs(s - s.mean()))[:max(2, len(s) // 4)]]
            mu, sd = s.mean(), s.std(ddof=1)
            out.append(sd)
        return np.array(out)

    G = 12
    curves = {}
    for mode in ("none", "half", "quarter"):
        curves[mode] = np.mean([run(1000, G, mode, s) for s in range(40)], axis=0)

    # the second mechanism: a model class narrower than the data
    r = np.random.default_rng(1)
    m = r.integers(0, 2, 6000)
    data = r.normal(np.where(m, 3.0, -3.0), 0.7)
    gens = [data]
    mu, sd = data.mean(), data.std(ddof=1)
    for _ in range(3):
        s = r.normal(mu, sd, 6000)
        gens.append(s)
        mu, sd = s.mean(), s.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for mode, c, lab in (("none", BLUE, "resample only"),
                         ("half", ORANGE, "keep the typical half"),
                         ("quarter", PINK, "keep the typical quarter")):
        ax.semilogy(range(G + 1), np.maximum(curves[mode], 1e-6), "o-", color=c,
                    ms=3.5, lw=1.8, label=lab)
    ax.set_xlabel("generation")
    ax.set_ylabel("standard deviation, as a fraction of the original")
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    ax = axes[1]
    for i, (g, c) in enumerate(zip(gens, (INK, BLUE, "#93c5fd", LIGHT))):
        ax.hist(g, bins=70, density=True, histtype="step", lw=1.8, color=c,
                label=f"generation {i}" + (" (real data)" if i == 0 else ""))
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title("a single Gaussian fitted to bimodal data", fontsize=9.5,
                 color=INK, pad=6)
    ax.legend(fontsize=7.5, frameon=False)
    tidy(ax)

    print(f"    resampling alone, n=1000: sd is {curves['none'][-1]:.4f} of the original "
          f"after {G} generations — it barely collapses")
    print(f"    keeping the typical half:    {curves['half'][-1]:.6f}")
    print(f"    keeping the typical quarter: {curves['quarter'][-1]:.6f}")
    print(f"    and the mismatch case: the two modes are gone at generation 1, while the")
    print(f"    standard deviation goes {gens[0].std():.3f} -> {gens[1].std():.3f} — "
          f"the summary statistic looks healthy and the distribution is destroyed")
    fig.tight_layout()
    save(fig, "05-collapse")


# ------------------------------------------------------------ 5. the frame gap

def fig_frame_gap():
    """Scale improves precision and does nothing for validity.

    A frame that misses part of the population produces a biased estimate. More
    data from the same frame shrinks the confidence interval around that biased
    estimate — so the number gets more precise and no less wrong, and the
    interval stops containing the truth.
    """
    rng = np.random.default_rng(4)
    # the population: two groups, different rates, one of them under-sampled
    p_a, p_b, share_b = 0.60, 0.30, 0.40
    truth = (1 - share_b) * p_a + share_b * p_b

    ns = np.array([50, 100, 200, 500, 1000, 5000, 20000, 100000])
    for frame_b in (0.40, 0.10, 0.00):
        pass

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    cols = {0.40: GREEN, 0.10: ORANGE, 0.00: PINK}
    labs = {0.40: "frame matches the population",
            0.10: "group B under-represented (10%)",
            0.00: "group B absent from the frame"}
    for frame_b, c in cols.items():
        est, lo, hi = [], [], []
        for n in ns:
            nb = int(round(frame_b * n))
            draws = np.concatenate([rng.random(n - nb) < p_a, rng.random(nb) < p_b])
            m = draws.mean()
            se = np.sqrt(m * (1 - m) / n)
            est.append(m); lo.append(m - 1.96 * se); hi.append(m + 1.96 * se)
        ax.plot(ns, est, "o-", color=c, ms=4, lw=1.8, label=labs[frame_b])
        ax.fill_between(ns, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(truth, color=INK, ls="--", lw=1.4)
    ax.text(60, truth + 0.008, "the population value", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("sample size")
    ax.set_ylabel("estimated rate, with 95% interval")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax)

    ax = axes[1]
    for frame_b, c in cols.items():
        widths, covers = [], []
        for n in ns:
            nb = int(round(frame_b * n))
            draws = np.concatenate([rng.random(n - nb) < p_a, rng.random(nb) < p_b])
            m = draws.mean()
            se = np.sqrt(max(m * (1 - m), 1e-12) / n)
            widths.append(2 * 1.96 * se)
            covers.append(abs(m - truth) <= 1.96 * se)
        ax.loglog(ns, widths, "o-", color=c, ms=4, lw=1.8)
        first_miss = next((n for n, ok in zip(ns, covers) if not ok), None)
        if first_miss is not None and frame_b > 0:
            ax.axvline(first_miss, color=c, ls=":", lw=1.2)
    ax.set_xlabel("sample size")
    ax.set_ylabel("width of the 95% interval")
    ax.set_title("the interval shrinks whatever the frame", fontsize=9.5, color=INK, pad=6)
    tidy(ax)

    print(f"    the population rate is {truth:.3f}")
    for frame_b in (0.40, 0.10, 0.00):
        nb = int(round(frame_b * 100000))
        draws = np.concatenate([rng.random(100000 - nb) < p_a, rng.random(nb) < p_b])
        m = draws.mean(); se = np.sqrt(m * (1 - m) / 100000)
        print(f"    frame with {100*frame_b:4.0f}% of group B, n = 100,000: "
              f"estimate {m:.4f} +/- {1.96*se:.4f} — "
              f"{'contains' if abs(m-truth) <= 1.96*se else 'DOES NOT CONTAIN'} the truth")
    print("    the interval narrows like 1/sqrt(n) in every case; the bias does not move,")
    print("    so more data makes a mis-framed estimate more confident and no more correct")
    fig.tight_layout()
    save(fig, "01-frame-gap")


CHEAP = [fig_frame_gap, fig_label_noise, fig_contamination, fig_minhash, fig_collapse]

if __name__ == "__main__":
    print("Part XIII — data figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
