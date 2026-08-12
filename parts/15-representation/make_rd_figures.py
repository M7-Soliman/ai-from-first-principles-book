"""
Figures 11 to 14 for Part XV — the appended movement G.

Predictions for every one of these were recorded before the
script was run. Figures 1 to 10 live in
make_figures.py and predate that discipline; nothing is invented for them here.

    python3 make_rd_figures.py

Runtime: about two minutes. Numpy only.

This is the most favourable measurement setting in the book: the rate-distortion
function of a Gaussian source has a closed form, code lengths are arithmetic, and
the information bottleneck on a discrete joint distribution is a fixed-point
iteration with a known answer. Everything below is a computed quantity compared
against a measured one.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "rp"))
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


def lloyd_max(x, k, iters=200):
    """Lloyd's algorithm: the optimal fixed-rate quantiser for this sample."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    c = np.linspace(lo, hi, k)
    for _ in range(iters):
        idx = np.abs(x[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                c[j] = x[m].mean()
    idx = np.abs(x[:, None] - c[None, :]).argmin(1)
    return c, idx


def entropy_bits(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Figure 11 — the rate-distortion curve
# ---------------------------------------------------------------------------

def figure11(n=400000, seed=0):
    """Gaussian source, sigma = 1. R(D) = 0.5 log2(1/D) for D <= 1.

    Three quantisers, all entropy-coded, so the rate is the entropy of the
    output rather than log2(k). Comparing a density-adapted quantiser against a
    uniform lattice would confound two different effects; these differ only in
    the shape of their cells, which is what the space-filling argument is about.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=(n // 2, 2))

    steps = np.array([1.4, 1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08])
    uni_r, uni_d = [], []
    for h in steps:
        q = np.round(x / h) * h
        uni_d.append(float(np.mean((x - q) ** 2)))
        _, c = np.unique(np.round(x / h).astype(np.int64), return_counts=True)
        uni_r.append(entropy_bits(c))
    uni_r, uni_d = np.array(uni_r), np.array(uni_d)

    ks = [2, 4, 8, 16, 32, 64, 128]
    ll_r, ll_d = [], []
    for k in ks:
        c, idx = lloyd_max(x[:120000], k)
        ll_d.append(float(np.mean((x[:120000] - c[idx]) ** 2)))
        ll_r.append(entropy_bits(np.bincount(idx, minlength=k)))
    ll_r, ll_d = np.array(ll_r), np.array(ll_d)

    # Nearest point in the hexagonal lattice A2. Rounding the coordinates in a
    # skewed basis finds the nearest point of a PARALLELOGRAM tiling, not the
    # hexagonal Voronoi cell, and a parallelogram has a worse second moment than
    # the hexagon — so the naive version measures the wrong lattice and reports
    # the space-filling gain with the wrong sign. A2 is the union of two
    # rectangular sublattices; quantise to each and keep the closer.
    hex_r, hex_d = [], []
    for h in steps:
        ax_, ay = h, h * np.sqrt(3) / 2
        c1 = np.stack([np.round(y[:, 0] / ax_), np.round(y[:, 1] / ay)], 1)
        p1 = np.stack([c1[:, 0] * ax_, c1[:, 1] * ay], 1)
        sh = np.array([ax_ / 2, ay / 2])
        c2 = np.stack([np.round((y[:, 0] - sh[0]) / ax_),
                       np.round((y[:, 1] - sh[1]) / ay)], 1)
        p2 = np.stack([c2[:, 0] * ax_ + sh[0], c2[:, 1] * ay + sh[1]], 1)
        d1 = np.sum((y - p1) ** 2, 1)
        d2 = np.sum((y - p2) ** 2, 1)
        use2 = d2 < d1
        hex_d.append(float(np.mean(np.where(use2, d2, d1)) / 2))
        keys = np.where(use2,
                        c2[:, 0].astype(np.int64) * 4000037 + c2[:, 1].astype(np.int64) * 2 + 1,
                        c1[:, 0].astype(np.int64) * 4000037 + c1[:, 1].astype(np.int64) * 2)
        _, c = np.unique(keys, return_counts=True)
        hex_r.append(entropy_bits(c) / 2)
    hex_r, hex_d = np.array(hex_r), np.array(hex_d)

    def gap(r, d):
        return r - 0.5 * np.log2(1.0 / d)

    gu, gl, gh = gap(uni_r, uni_d), gap(ll_r, ll_d), gap(hex_r, hex_d)
    theory_scalar = 0.5 * np.log2(np.pi * np.e / 6)
    theory_hex = 0.5 * np.log2((1.0 / 12) / (5.0 / (36 * np.sqrt(3))))

    print(f"  Gaussian source, sigma = 1")
    print(f"  entropy-coded uniform scalar:")
    for r, d, g in zip(uni_r, uni_d, gu):
        print(f"    rate {r:6.3f}  distortion {d:9.5f}  gap {g:+.4f}")
    print(f"  gap positive everywhere: {bool(np.all(gu > 0))}")
    print(f"  high-rate gap {gu[-4:].mean():.4f} against the theoretical "
          f"0.5*log2(pi*e/6) = {theory_scalar:.4f}")
    print(f"  Lloyd-Max (optimal at FIXED rate), entropy-coded: high-rate gap "
          f"{gl[-3:].mean():.4f}")
    print(f"    it is worse under entropy coding, by {gl[-3:].mean() - gu[-4:].mean():.4f} "
          f"bits — the quantiser that minimises distortion at a fixed *count* is not "
          f"the one that minimises it at a fixed *entropy*")
    print(f"  hexagonal lattice in two dimensions: high-rate gap {gh[-4:].mean():.4f}")
    print(f"    gain over uniform scalar {gu[-4:].mean() - gh[-4:].mean():+.4f} bits, "
          f"against a theoretical {theory_hex:.4f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    dd = np.linspace(0.004, 0.9, 300)
    ax.plot(0.5 * np.log2(1.0 / dd), dd, "-", color=GRAY, lw=1.6, label="$R(D)$, the bound")
    ax.plot(uni_r, uni_d, "o-", color=BLUE, lw=1.9, ms=4, label="uniform, entropy-coded")
    ax.plot(ll_r, ll_d, "^-", color=PURPLE, lw=1.6, ms=4, label="Lloyd–Max")
    ax.plot(hex_r, hex_d, "s-", color=ORANGE, lw=1.9, ms=4, label="hexagonal, 2-D")
    ax.set_yscale("log")
    ax.set_xlabel("rate, bits per sample")
    ax.set_ylabel("distortion")
    ax.set_title("What a bound is worth", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=7.6)
    tidy(ax)

    ax2.plot(uni_r, gu, "o-", color=BLUE, lw=1.9, ms=4, label="uniform")
    ax2.plot(ll_r, gl, "^-", color=PURPLE, lw=1.6, ms=4, label="Lloyd–Max")
    ax2.plot(hex_r, gh, "s-", color=ORANGE, lw=1.9, ms=4, label="hexagonal")
    ax2.axhline(theory_scalar, color=GRAY, ls="--", lw=1.2)
    ax2.text(uni_r[0], theory_scalar + 0.015, f"0.254 bits, the scalar limit",
             fontsize=7.2, color=GRAY)
    ax2.set_xlabel("rate, bits per sample")
    ax2.set_ylabel("excess rate over the bound")
    ax2.set_ylim(0.1, 0.65)
    ax2.set_title("The gap is an overhead, not a drift", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=7.6)
    tidy(ax2)
    fig.suptitle("Rate against distortion, measured against the theory",
                 fontsize=10.5, y=1.03)
    save(fig, "11-rate-distortion")
    return uni_r, uni_d, gu, gl, gh


# ---------------------------------------------------------------------------
# Figure 12 — minimum description length
# ---------------------------------------------------------------------------

def figure12(n=60, true_deg=3, noise=0.35, seed=1, trials=200):
    """Two-part code: bits to send the model, plus bits to send the residuals.

    The parameter code is the standard (k/2) log2 n — the precision worth
    sending for a parameter estimated from n samples — and the residual code is
    the Gaussian negative log-likelihood in bits. Nothing here looks at held-out
    data, which is the point.
    """
    degs = np.arange(0, 10)
    rng = np.random.default_rng(seed)
    tot = np.zeros((trials, len(degs)))
    par = np.zeros((trials, len(degs)))
    res = np.zeros((trials, len(degs)))
    test = np.zeros((trials, len(degs)))

    coef_true = rng.normal(size=true_deg + 1)
    for t in range(trials):
        r = np.random.default_rng(seed + 1000 + t)
        x = np.sort(r.uniform(-1, 1, n))
        y = np.polyval(coef_true[::-1], x) + r.normal(scale=noise, size=n)
        xt = r.uniform(-1, 1, 4000)
        yt = np.polyval(coef_true[::-1], xt) + r.normal(scale=noise, size=4000)
        for j, dg in enumerate(degs):
            A = np.vander(x, dg + 1, increasing=True)
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            resid = y - A @ w
            s2 = max(float(np.mean(resid ** 2)), 1e-12)
            # residual code: Gaussian NLL in bits, sigma fitted
            res[t, j] = 0.5 * n * np.log2(2 * np.pi * np.e * s2)
            par[t, j] = 0.5 * (dg + 2) * np.log2(n)     # +1 for sigma
            tot[t, j] = res[t, j] + par[t, j]
            At = np.vander(xt, dg + 1, increasing=True)
            test[t, j] = float(np.mean((yt - At @ w) ** 2))

    mt, mp, mr, mte = tot.mean(0), par.mean(0), res.mean(0), test.mean(0)
    j_mdl, j_test = int(np.argmin(mt)), int(np.argmin(mte))
    print(f"  data generated by a degree-{true_deg} polynomial, n = {n}, "
          f"{trials} datasets")
    print(f"  deg   model bits   residual bits   total   test MSE")
    for dg, p_, r_, t_, e_ in zip(degs, mp, mr, mt, mte):
        print(f"  {dg:3d}   {p_:10.1f}   {r_:13.1f}   {t_:6.1f}   {e_:.4f}")
    print(f"  MDL picks degree {degs[j_mdl]}; test error picks degree {degs[j_test]}; "
          f"truth is {true_deg}")
    agree = float(np.mean(tot.argmin(1) == test.argmin(1)))
    print(f"  per-dataset agreement between MDL and held-out selection: {agree:.0%}")
    hit = float(np.mean(tot.argmin(1) == true_deg))
    print(f"  MDL lands on the true degree on {hit:.0%} of datasets")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax.plot(degs, mp, "-", color=ORANGE, lw=1.8, label="model")
    ax.plot(degs, mr - mr.min() + 5, "-", color=BLUE, lw=1.8, label="residuals (shifted)")
    ax.plot(degs, mt - mt.min() + 5, "-", color=INK, lw=2.2, label="total (shifted)")
    ax.axvline(degs[j_mdl], color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("bits")
    ax.set_title("Two parts, and their sum", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)

    ax2.plot(degs, mt, "o-", color=INK, lw=2.0, ms=4.5, label="description length")
    ax2.set_xlabel("polynomial degree")
    ax2.set_ylabel("total bits")
    ax3 = ax2.twinx()
    ax3.plot(degs, mte, "s--", color=ORANGE, lw=1.8, ms=4, label="test MSE")
    ax3.set_ylabel("test MSE", color=ORANGE)
    ax3.tick_params(labelsize=8.5, colors=ORANGE)
    ax3.spines["top"].set_visible(False)
    ax2.set_title("It agrees with held-out data", fontsize=10, pad=8)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax3.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(ax2)
    fig.suptitle("Choosing a model without a validation set", fontsize=10.5, y=1.03)
    save(fig, "12-mdl")
    return degs, mt, mte, j_mdl, j_test


# ---------------------------------------------------------------------------
# Figure 13 — the information bottleneck
# ---------------------------------------------------------------------------

def ib(pxy, n_z, beta, iters=600, seed=0):
    """Blahut-Arimoto style iteration for the information bottleneck."""
    rng = np.random.default_rng(seed)
    nx, ny = pxy.shape
    px = pxy.sum(1)
    py_x = pxy / px[:, None]
    q = rng.random((nx, n_z)) + 1e-3
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        qz = px @ q                                      # p(z)
        qz = np.maximum(qz, 1e-12)
        qzy = (q * px[:, None]).T @ py_x                 # p(z,y) -> p(y|z)
        py_z = qzy / np.maximum(qzy.sum(1, keepdims=True), 1e-12)
        kl = np.array([[np.sum(py_x[i] * np.log2(np.maximum(py_x[i], 1e-12)
                                                 / np.maximum(py_z[z], 1e-12)))
                        for z in range(n_z)] for i in range(nx)])
        log_q = np.log2(qz)[None, :] - beta * kl
        log_q -= log_q.max(1, keepdims=True)
        q = 2.0 ** log_q
        q /= q.sum(1, keepdims=True)
    qz = np.maximum(px @ q, 1e-12)
    ixz = float(np.sum(px[:, None] * q * np.log2(np.maximum(q, 1e-12) / qz[None, :])))
    qzy = (q * px[:, None]).T @ py_x
    py = pxy.sum(0)
    izy = float(np.sum(qzy * np.log2(np.maximum(qzy, 1e-12)
                                     / (np.maximum(qzy.sum(1, keepdims=True), 1e-12)
                                        * py[None, :]))))
    return ixz, izy, q


def figure13(nx=24, ny=4, n_z=8, seed=2):
    """A joint distribution in which the high-variance structure and the
    predictive structure are deliberately different, so 'keep the big
    directions' and 'keep what predicts Y' can be told apart."""
    rng = np.random.default_rng(seed)
    px = rng.dirichlet(np.ones(nx) * 2.0)
    # Y depends on a coarse grouping of X; a second, finer structure carries most
    # of the variation in p(x) and none of the information about Y.
    group = np.arange(nx) % ny
    py_x = np.full((nx, ny), 0.03)
    py_x[np.arange(nx), group] = 1.0 - 0.03 * (ny - 1)
    py_x /= py_x.sum(1, keepdims=True)
    pxy = px[:, None] * py_x
    ixy = float(np.sum(pxy * np.log2(pxy / (pxy.sum(0, keepdims=True)
                                            * pxy.sum(1, keepdims=True)))))
    hx = float(-(px * np.log2(px)).sum())
    print(f"  H(X) = {hx:.3f} bits, I(X;Y) = {ixy:.3f} bits, "
          f"|Z| = {n_z}, {ny} classes")

    betas = np.concatenate([np.linspace(0.2, 3, 12), np.linspace(3.5, 40, 14)])
    pts = [ib(pxy, n_z, b, seed=seed + i) for i, b in enumerate(betas)]
    ixz = np.array([p[0] for p in pts])
    izy = np.array([p[1] for p in pts])
    o = np.argsort(ixz)
    ixz, izy = ixz[o], izy[o]

    frac = izy / ixy
    knee = next((i for i in range(len(frac)) if frac[i] >= 0.95), len(frac) - 1)
    print(f"  95% of the available I(X;Y) is reached at I(X;Z) = {ixz[knee]:.3f} bits, "
          f"which is {ixz[knee] / hx:.1%} of H(X)")
    print(f"  beyond it, {ixz[-1] - ixz[knee]:.2f} more bits of X buy "
          f"{izy[-1] - izy[knee]:.4f} bits about Y")

    # the control: keep the most probable symbols instead, a reconstruction-first rule
    order = np.argsort(-px)
    ctrl_r, ctrl_p = [], []
    for k in range(1, nx + 1):
        keep = order[:k]
        mass = px[keep].sum()
        r = float(-(px[keep] * np.log2(px[keep])).sum()) if k > 1 else 0.0
        merged = np.zeros((k + 1, ny))
        merged[:k] = pxy[keep]
        merged[k] = pxy[np.setdiff1d(np.arange(nx), keep)].sum(0)
        m = merged
        py = m.sum(0)
        pz = m.sum(1)
        val = float(np.sum(m * np.log2(np.maximum(m, 1e-12)
                                       / (np.maximum(pz[:, None], 1e-12) * py[None, :]))))
        ctrl_r.append(r)
        ctrl_p.append(val)
    ctrl_r, ctrl_p = np.array(ctrl_r), np.array(ctrl_p)
    j = int(np.argmin(np.abs(ctrl_r - ixz[knee])))
    print(f"  keeping the most probable symbols instead: at a comparable rate "
          f"({ctrl_r[j]:.2f} bits) it retains {ctrl_p[j]:.3f} bits about Y "
          f"against the bottleneck's {izy[knee]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(ixz, izy, "o-", color=BLUE, lw=2.0, ms=4, label="information bottleneck")
    ax.plot(ctrl_r, ctrl_p, "s--", color=ORANGE, lw=1.7, ms=3.5,
            label="keep the most probable symbols")
    ax.axhline(ixy, color=GRAY, ls="--", lw=1.2)
    ax.text(0.1, ixy - 0.09, "$I(X;Y)$, all there is to know", fontsize=7.6, color=GRAY)
    ax.axvline(hx, color=LIGHT, lw=1.2)
    ax.text(hx - 0.15, 0.25, "$H(X)$", fontsize=7.6, color=GRAY, ha="right")
    ax.set_xlabel("$I(X;Z)$ — bits kept about the input")
    ax.set_ylabel("$I(Z;Y)$ — bits about the label")
    ax.set_title("Compressing for prediction is not compressing for reconstruction",
                 fontsize=10.2, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax)
    save(fig, "13-bottleneck")
    return ixz, izy, ctrl_r, ctrl_p, ixy, hx


# ---------------------------------------------------------------------------
# Figure 14 — compression and generalisation
# ---------------------------------------------------------------------------

def figure14(n=80, noise=0.4, seed=3, trials=300):
    """Two model families on the same data: polynomials and Fourier features.

    Within a family, shorter description means better generalisation. Across
    families the comparison needs the cost of naming the family, and leaving it
    out is where "compression implies generalisation" goes wrong.
    """
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=4)
    fams = {"polynomial": lambda x, k: np.vander(x, k + 1, increasing=True),
            "Fourier": lambda x, k: np.hstack(
                [np.ones((len(x), 1))]
                + [f(np.pi * j * x)[:, None] for j in range(1, k + 1) for f in (np.sin, np.cos)])}
    ks = np.arange(0, 8)
    out = {}
    for name, feat in fams.items():
        code = np.zeros((trials, len(ks)))
        resid = np.zeros((trials, len(ks)))
        test = np.zeros((trials, len(ks)))
        for t in range(trials):
            r = np.random.default_rng(seed + 5000 + t)
            x = r.uniform(-1, 1, n)
            y = np.polyval(coef[::-1], x) + r.normal(scale=noise, size=n)
            xt = r.uniform(-1, 1, 4000)
            yt = np.polyval(coef[::-1], xt) + r.normal(scale=noise, size=4000)
            for j, k in enumerate(ks):
                A = feat(x, k)
                w = np.linalg.lstsq(A, y, rcond=None)[0]
                s2 = max(float(np.mean((y - A @ w) ** 2)), 1e-12)
                resid[t, j] = 0.5 * n * np.log2(2 * np.pi * np.e * s2)
                code[t, j] = resid[t, j] + 0.5 * (A.shape[1] + 1) * np.log2(n)
                test[t, j] = float(np.mean((yt - feat(xt, k) @ w) ** 2))
        out[name] = (code.mean(0), test.mean(0), resid.mean(0))
        c, e, _ = out[name]
        rho = np.corrcoef(c, e)[0, 1]
        print(f"  {name:11s}: best code {c.min():.1f} bits at k={ks[c.argmin()]}, "
              f"best test {e.min():.4f} at k={ks[e.argmin()]}, "
              f"corr(code, test) = {rho:+.3f}")

    cp, ep, rp = out["polynomial"]
    cf, ef, rf = out["Fourier"]
    print(f"  across families at each family's own optimum:")
    print(f"    polynomial: {cp.min():.1f} bits, test {ep[cp.argmin()]:.4f}")
    print(f"    Fourier   : {cf.min():.1f} bits, test {ef[cf.argmin()]:.4f}")
    shorter = "Fourier" if cf.min() < cp.min() else "polynomial"
    better = "Fourier" if ef[cf.argmin()] < ep[cp.argmin()] else "polynomial"
    print(f"    shorter code: {shorter};  better test error: {better};  "
          f"{'agree' if shorter == better else 'DISAGREE'}")
    # Now drop the model's own cost and compare residual bits only — the naive
    # version of "it compresses the data better".
    print(f"  residuals only, ignoring the cost of the parameters:")
    print(f"    polynomial best {rp.min():.1f} bits at k={ks[rp.argmin()]}, "
          f"test there {ep[rp.argmin()]:.4f}")
    print(f"    Fourier    best {rf.min():.1f} bits at k={ks[rf.argmin()]}, "
          f"test there {ef[rf.argmin()]:.4f}")
    naive = "Fourier" if rf.min() < rp.min() else "polynomial"
    print(f"    residual-only prefers: {naive};  full code prefers: {shorter};  "
          f"test error prefers: {better}")
    rho_p = np.corrcoef(rp, ep)[0, 1]
    rho_f = np.corrcoef(rf, ef)[0, 1]
    print(f"    corr(residual bits, test) within family: polynomial {rho_p:+.3f}, "
          f"Fourier {rho_f:+.3f}  — against +0.99 for the full code")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (name, (c, e, _)), col, mk in zip(out.items(), (BLUE, ORANGE), ("o", "s")):
        ax.plot(c, e, mk + "-", color=col, lw=1.8, ms=5, label=name)
        for k, cc, ee in zip(ks, c, e):
            if k in (0, 2, 4, 7):
                ax.annotate(f"k={k}", (cc, ee), fontsize=6.4, color=col,
                            xytext=(4, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("description length of the training data, bits")
    ax.set_ylabel("test MSE")
    ax.set_title("Shorter description, better generalisation — within a family",
                 fontsize=10.2, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "14-compression-generalisation")
    return ks, out


if __name__ == "__main__":
    for i, fn in enumerate([figure11, figure12, figure13, figure14], 11):
        print(f"Figure {i}")
        fn()
    print("done")
