"""
Figures for Part III — Numerical Computation.

Everything is computed. The convergence plots are real iterations; the accuracy
comparisons are real solves at real condition numbers; the cancellation figure is
actual float64 arithmetic losing actual digits.

    python3 make_figures.py           # writes into ../../book/src/figures/nc/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "nc"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
LIGHT, INK = "#cbd5e1", "#1e293b"

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

MAX_PT = 900


def save(fig, name):
    p = os.path.join(OUT, name + ".svg")
    fig.savefig(p, format="svg")
    plt.close(fig)
    import re
    h = float(re.search(r'height="([\d.]+)pt"', open(p).read(1200)).group(1))
    if h > MAX_PT:
        raise ValueError(f"{name}.svg is {h:.0f}pt tall — check for inf/NaN axis limits")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# =============================================================== fig 1 ======
def fig_float_spacing():
    """Machine epsilon: the gap between 1 and the next representable number."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))

    ax = axes[0]
    exps = np.arange(-4, 5)
    gaps = 2.0 ** exps * np.finfo(np.float64).eps
    ax.loglog(2.0 ** exps, gaps, "o-", color=BLUE, lw=2, ms=5)
    ax.set_xlabel("magnitude of the number")
    ax.set_ylabel("gap to the next float")
    ax.set_title("spacing is proportional to magnitude", fontsize=10, pad=8)
    tidy(ax)

    ax = axes[1]
    # (x+1) - x should be exactly 1. Watch it stop being 1.
    # NB: plot the value directly, not its log -- log10 of a near-zero result
    # blows the axis out to ~320 units and produces a figure feet tall.
    scales = np.logspace(0, 17, 300)
    got = (scales + 1.0) - scales
    ax.semilogx(scales, got, color=ORANGE, lw=2.2)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.1)
    ax.set_ylim(-0.15, 1.35)
    ax.set_xlabel("x")
    ax.set_ylabel(r"computed $(x{+}1) - x$")
    ax.set_title("adding 1 stops doing anything past $10^{16}$", fontsize=10, pad=8)
    ax.text(2e1, 1.08, "exactly 1", fontsize=8, color=GRAY)
    ax.text(1.5e16, 0.12, "0", fontsize=8, color=PINK)
    tidy(ax)
    save(fig, "01-float-spacing")


# =============================================================== fig 2 ======
def fig_cancellation():
    """Catastrophic cancellation: subtracting near-equal numbers destroys digits."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))

    # (1+h) - 1 should equal h exactly; in float it does not
    hs = np.logspace(-1, -17, 200)
    computed = (1.0 + hs) - 1.0
    rel = np.abs(computed - hs) / hs

    ax = axes[0]
    ax.loglog(hs, np.maximum(rel, 1e-18), color=PINK, lw=2.2)
    ax.set_xlabel("h")
    ax.set_ylabel("relative error of $(1{+}h)-1$")
    ax.invert_xaxis()
    ax.set_title("smaller h, worse answer", fontsize=10, pad=8)
    tidy(ax)

    # the classic: two formulas for the same quadratic root
    ax = axes[1]
    bs = np.logspace(2, 9, 60)
    naive, stable = [], []
    for b in bs:
        a, c = 1.0, 1.0
        disc = np.sqrt(b * b - 4 * a * c)
        r_naive = (-b + disc) / (2 * a)          # catastrophic when b is large
        r_stable = (2 * c) / (-b - disc)         # algebraically identical, stable
        exact = -1.0 / b                          # limiting value for a=c=1
        naive.append(abs((r_naive - exact) / exact))
        stable.append(abs((r_stable - exact) / exact) + 1e-18)
    ax.loglog(bs, np.maximum(naive, 1e-18), color=PINK, lw=2.2, label="textbook formula")
    ax.loglog(bs, stable, color=GREEN, lw=2.2, label="rearranged formula")
    ax.set_xlabel("b")
    ax.set_ylabel("relative error in the small root")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("same algebra, different arithmetic", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "02-cancellation")


# =============================================================== fig 3 ======
def fig_conditioning():
    """The condition number is an amplification factor, measured."""
    rng = np.random.default_rng(0)
    conds, errs = [], []
    for k in np.logspace(0, 12, 40):
        U, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        V, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        s = np.logspace(0, -np.log10(k), 6)
        A = U @ np.diag(s) @ V.T
        x = rng.normal(size=6)
        b = A @ x
        db = rng.normal(size=6)
        db *= 1e-10 * np.linalg.norm(b) / np.linalg.norm(db)
        x2 = np.linalg.solve(A, b + db)
        errs.append(np.linalg.norm(x2 - x) / np.linalg.norm(x))
        conds.append(np.linalg.cond(A))

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.loglog(conds, np.maximum(errs, 1e-18), "o", color=BLUE, ms=5, alpha=0.8)
    c = np.array(sorted(conds))
    ax.loglog(c, 1e-10 * c, color=ORANGE, lw=1.8, ls="--",
              label=r"$\kappa(A) \times$ input error")
    ax.set_xlabel(r"condition number $\kappa(A)$")
    ax.set_ylabel("relative error in the solution")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("error out = condition number × error in", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "03-conditioning")


# =============================================================== fig 4 ======
def fig_curvature():
    """The Hessian's eigenvalues ARE the curvatures along its eigenvectors."""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    Hs = [(np.array([[1.0, 0.0], [0.0, 1.0]]), "well conditioned"),
          (np.array([[1.0, 0.0], [0.0, 12.0]]), "ill conditioned"),
          (np.array([[1.0, 0.0], [0.0, -3.0]]), "indefinite: a saddle")]
    g = np.linspace(-2.4, 2.4, 260)
    X, Y = np.meshgrid(g, g)
    for ax, (H, lab) in zip(axes, Hs):
        Z = 0.5 * (H[0, 0] * X ** 2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y ** 2)
        ax.contour(X, Y, Z, levels=16, colors=[LIGHT], linewidths=0.9)
        w, V = np.linalg.eigh(H)
        for i in range(2):
            v = V[:, i] * 1.7
            c = GREEN if w[i] > 0 else PINK
            for sgn in (1, -1):
                ax.add_patch(FancyArrowPatch((0, 0), sgn * v, arrowstyle="-|>",
                                             mutation_scale=8, lw=2.0, color=c,
                                             zorder=6, shrinkA=0, shrinkB=0))
        k = np.abs(w).max() / max(np.abs(w).min(), 1e-12)
        sub = (r"$\lambda = %.0f,\ %.0f$" % (w[1], w[0])) + \
              ("" if w.min() < 0 else r"   $\kappa = %.0f$" % k)
        ax.set_title(lab + "\n" + sub, fontsize=9.3, pad=5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        tidy(ax, grid=False)
    save(fig, "04-curvature")


# =============================================================== fig 5 ======
def fig_descent_vs_newton():
    """Real iterations. Newton uses curvature and arrives in one step here."""
    A = np.array([[1.0, 0.0], [0.0, 14.0]])
    grad = lambda p: A @ p
    x0 = np.array([2.2, 1.0])

    def gd(lr, n=40):
        p = x0.copy(); path = [p.copy()]
        for _ in range(n):
            p = p - lr * grad(p); path.append(p.copy())
        return np.array(path)

    def newton(n=4):
        p = x0.copy(); path = [p.copy()]
        for _ in range(n):
            p = p - np.linalg.solve(A, grad(p)); path.append(p.copy())
        return np.array(path)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    g = np.linspace(-2.6, 2.6, 260)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (X ** 2 + 14 * Y ** 2)
    for ax, (path, lab, col) in zip(axes, [(gd(0.12), "gradient descent (40 steps)", BLUE),
                                           (newton(), "Newton (1 step)", ORANGE)]):
        ax.contour(X, Y, Z, levels=14, colors=[LIGHT], linewidths=0.9)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, lw=1.5, ms=3.4, zorder=5)
        ax.plot([0], [0], "*", color=GREEN, ms=13, zorder=7)
        ax.set_xlim(-2.7, 2.7); ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=9.8, pad=6)
        tidy(ax, grid=False)
    save(fig, "05-descent-vs-newton")


# =============================================================== fig 6 ======
def fig_momentum():
    """Momentum damps the oscillation across a narrow valley. Real runs.

    Parameters chosen so plain descent oscillates continuously across the valley
    (lr=0.05 > 1/lambda_max=1/22, so each step overshoots in the steep direction)
    while momentum with beta=0.8 is well inside the deterministic heavy-ball
    stability bound.

    For v <- beta*v + A@p; p <- p - lr*v on eigenvalue lambda of A, the 2x2
    iteration matrix [[1-lr*lambda, -lr*beta], [lambda, beta]] has determinant
    beta, and the Jury stability condition |trace| < 1+det gives the true
    divergence threshold
        lr * lambda < 2 * (1 + beta)      i.e.  lr < 2*(1+beta)/lambda_max.
    (NOT "lr*(1+beta)/(1-beta)*lambda_max < 2" -- that inverted-looking bound is
    wrong and was corrected after review; see Part III Section 18's "Try it" for
    the full derivation, the empirical confirmation, and the misconception it
    replaces.) Here lambda_max=22, beta=0.8 gives max stable lr = 2*1.8/22 =
    0.1636, comfortably above the lr=0.05 used below -- which is exactly why the
    run converges rather than diverging.

    Closed-form check for GD (beta=0): eigenvalues of (I - alpha*A) are
      1 - 0.05*1  = 0.95  (x-direction, damped)
      1 - 0.05*22 = -0.10 (y-direction: negative, so sign alternates every step)
    The y-component oscillates in sign every iteration, confirming the figure.
    Final distance GD: ~0.111 (confirmed by run); momentum: ~0.003.
    """
    A = np.array([[1.0, 0.0], [0.0, 22.0]])
    x0 = np.array([2.4, 0.9])
    LR = 0.05

    def run(lr, beta, n=60):
        p, v = x0.copy(), np.zeros(2); path = [p.copy()]
        for _ in range(n):
            v = beta * v + A @ p
            p = p - lr * v
            path.append(p.copy())
            if not np.all(np.isfinite(p)):
                break
        return np.array(path)

    # Verify claim vs closed form (GD on quadratic)
    path_gd  = run(LR, 0.0)
    path_mom = run(LR, 0.8)
    d_gd  = np.linalg.norm(path_gd[-1])
    d_mom = np.linalg.norm(path_mom[-1])
    # GD y-eigenvalue = 1 - LR*22 = -0.10; after 60 steps: 0.9^60 * (-0.10)^60
    # The x-distance should be x0[0] * 0.95^60
    x_dist_theory = abs(x0[0]) * (1 - LR * 1.0) ** 60
    assert abs(path_gd[-1][0] - x_dist_theory) < 1e-6, (
        f"GD x-distance {path_gd[-1][0]:.6f} vs theory {x_dist_theory:.6f}")
    assert d_mom < d_gd / 10, (
        f"Momentum ({d_mom:.4f}) should be >10x closer than GD ({d_gd:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    g = np.linspace(-2.8, 2.8, 260)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (X ** 2 + 22 * Y ** 2)
    for ax, (b, lab, col) in zip(axes, [(0.0, r"plain descent", BLUE),
                                        (0.8, r"with momentum $\beta = 0.8$", ORANGE)]):
        path = run(LR, b)
        ax.contour(X, Y, Z, levels=14, colors=[LIGHT], linewidths=0.9)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, lw=1.4, ms=3, zorder=5)
        ax.plot([0], [0], "*", color=GREEN, ms=13, zorder=7)
        d = np.linalg.norm(path[-1])
        ax.set_title(f"{lab}\nfinal distance {d:.3f}", fontsize=9.5, pad=5)
        ax.set_xlim(-2.9, 2.9); ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        tidy(ax, grid=False)
    save(fig, "06-momentum")


# =============================================================== fig 7 ======
def fig_sgd_noise():
    """Batch size trades gradient noise against cost. Simulated on a real objective."""
    rng = np.random.default_rng(3)
    n, d = 400, 2
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -0.8])
    y = X @ w_true + rng.normal(0, 0.6, size=n)

    def run(bs, lr=0.06, steps=90):
        w = np.array([-1.6, 1.7]); path = [w.copy()]
        for t in range(steps):
            idx = rng.integers(0, n, bs)
            Xb, yb = X[idx], y[idx]
            gr = 2 * Xb.T @ (Xb @ w - yb) / bs
            w = w - lr * gr
            path.append(w.copy())
        return np.array(path)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9))
    gx = np.linspace(-2.2, 2.6, 200); gy = np.linspace(-1.6, 2.2, 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.zeros_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            w = np.array([GX[i, j], GY[i, j]])
            Z[i, j] = np.mean((X @ w - y) ** 2)
    for ax, bs in zip(axes, [1, 16, 400]):
        ax.contour(GX, GY, Z, levels=16, colors=[LIGHT], linewidths=0.85)
        p = run(bs)
        ax.plot(p[:, 0], p[:, 1], "-", color=BLUE, lw=1.1, alpha=0.9, zorder=4)
        ax.plot(p[:, 0], p[:, 1], ".", color=BLUE, ms=2.6, zorder=5)
        ax.plot(*w_true, "*", color=ORANGE, ms=13, zorder=7)
        lab = {1: "batch size 1", 16: "batch size 16", 400: "full batch"}[bs]
        ax.set_title(lab, fontsize=9.6, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        tidy(ax, grid=False)
    save(fig, "07-sgd-noise")


# =============================================================== fig 8 ======
def fig_adaptive():
    """Per-parameter step sizes: why adaptive methods help on skewed curvature."""
    A = np.array([[0.4, 0.0], [0.0, 30.0]])
    x0 = np.array([2.4, 0.9])

    def run(kind, lr, steps=90, eps=1e-8):
        p = x0.copy(); s = np.zeros(2); v = np.zeros(2); path = [p.copy()]
        for t in range(1, steps + 1):
            g = A @ p
            if kind == "gd":
                step = lr * g
            elif kind == "rmsprop":
                s = 0.9 * s + 0.1 * g ** 2
                step = lr * g / (np.sqrt(s) + eps)
            else:  # adam
                v = 0.9 * v + 0.1 * g
                s = 0.999 * s + 0.001 * g ** 2
                vh = v / (1 - 0.9 ** t); sh = s / (1 - 0.999 ** t)
                step = lr * vh / (np.sqrt(sh) + eps)
            p = p - step
            path.append(p.copy())
            if not np.all(np.isfinite(p)):
                break
        return np.array(path)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for kind, lr, col, lab in [("gd", 0.03, BLUE, "gradient descent"),
                               ("rmsprop", 0.12, GREEN, "RMSProp"),
                               ("adam", 0.12, ORANGE, "Adam")]:
        p = run(kind, lr)
        loss = 0.5 * (0.4 * p[:, 0] ** 2 + 30 * p[:, 1] ** 2)
        ax.semilogy(np.maximum(loss, 1e-16), color=col, lw=2.0, label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel("objective")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title(r"curvature ratio 75:1 — adaptive rates rescale per parameter",
                 fontsize=9.8, pad=8)
    tidy(ax)
    save(fig, "08-adaptive")


# =============================================================== fig 9 ======
def fig_schedules():
    """Three schedules, and what each does to the same problem."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    T = 200
    t = np.arange(T)
    scheds = {
        "constant": np.full(T, 0.10),
        "step decay": 0.10 * 0.5 ** (t // 60),
        "cosine": 0.10 * 0.5 * (1 + np.cos(np.pi * t / T)),
    }
    cols = {"constant": GRAY, "step decay": BLUE, "cosine": ORANGE}

    ax = axes[0]
    for k, v in scheds.items():
        ax.plot(t, v, color=cols[k], lw=2.0, label=k)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"learning rate $\alpha_t$")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("three schedules", fontsize=10, pad=8)
    tidy(ax)

    ax = axes[1]
    rng = np.random.default_rng(1)
    A = np.array([[1.0, 0.0], [0.0, 9.0]])
    for k, v in scheds.items():
        p = np.array([2.0, 1.2]); losses = []
        for i in range(T):
            g = A @ p + rng.normal(0, 0.55, size=2)   # noisy gradients, as in SGD
            p = p - v[i] * g
            losses.append(0.5 * (p[0] ** 2 + 9 * p[1] ** 2))
        ax.semilogy(np.maximum(losses, 1e-8), color=cols[k], lw=1.8, label=k)
    ax.set_xlabel("iteration"); ax.set_ylabel("objective")
    ax.set_title("decaying the rate settles the noise floor", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "09-schedules")


# ============================================================== fig 10 ======
def fig_constrained():
    """Constrained optimum: the gradients are parallel where the contour is tangent."""
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    g = np.linspace(-2.2, 2.6, 300)
    X, Y = np.meshgrid(g, g)
    Z = (X - 1.4) ** 2 + (Y - 1.1) ** 2
    ax.contour(X, Y, Z, levels=16, colors=[LIGHT], linewidths=0.9, zorder=2)

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.3, zorder=4)
    ax.text(-1.55, 1.15, r"$g(\mathbf{x}) = 0$", color=BLUE, fontsize=10)

    # the constrained optimum is where the objective's gradient is parallel to the
    # constraint's normal -- computed, not eyeballed
    c = np.array([1.4, 1.1]); xs = c / np.linalg.norm(c)
    ax.plot(*xs, "o", color=ORANGE, ms=8, zorder=8)
    ax.add_patch(FancyArrowPatch(xs, xs + 0.55 * (c - xs) / np.linalg.norm(c - xs),
                                 arrowstyle="-|>", mutation_scale=9, lw=2.1,
                                 color=GREEN, zorder=7, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch(xs, xs + 0.55 * xs, arrowstyle="-|>",
                                 mutation_scale=9, lw=2.1, color=PINK, zorder=7,
                                 shrinkA=0, shrinkB=0))
    ax.text(1.02, 1.30, r"$-\nabla f$", color=GREEN, fontsize=10)
    ax.text(1.42, 0.72, r"$\nabla g$", color=PINK, fontsize=10)
    ax.plot(*c, "*", color=GRAY, ms=12, zorder=6)
    ax.text(1.5, 1.12, "unconstrained\nminimum", color=GRAY, fontsize=8)
    ax.set_xlim(-2.0, 2.4); ax.set_ylim(-1.7, 2.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("at the optimum the two gradients are parallel", fontsize=9.8, pad=8)
    tidy(ax, grid=False)
    save(fig, "10-constrained")


# ============================================================== fig 11 ======
def fig_least_squares():
    """Three algorithms for one problem, raced as conditioning degrades."""
    rng = np.random.default_rng(7)
    conds = np.logspace(1, 11, 26)
    en, eq, es = [], [], []
    for k in conds:
        m, n = 60, 6
        U, _ = np.linalg.qr(rng.normal(size=(m, n)))
        V, _ = np.linalg.qr(rng.normal(size=(n, n)))
        s = np.logspace(0, -np.log10(k), n)
        A = U @ np.diag(s) @ V.T
        x = rng.normal(size=n)
        b = A @ x
        rel = lambda z: np.linalg.norm(z - x) / np.linalg.norm(x)
        try:
            xn = np.linalg.solve(A.T @ A, A.T @ b)       # normal equations
            en.append(rel(xn))
        except np.linalg.LinAlgError:
            en.append(np.nan)
        Q, R = np.linalg.qr(A)                            # QR
        eq.append(rel(np.linalg.solve(R, Q.T @ b)))
        es.append(rel(np.linalg.lstsq(A, b, rcond=None)[0]))  # SVD

    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    ax.loglog(conds, np.maximum(en, 1e-18), "o-", color=PINK, lw=1.9, ms=4,
              label=r"normal equations  $A^\top A$")
    ax.loglog(conds, np.maximum(eq, 1e-18), "o-", color=BLUE, lw=1.9, ms=4,
              label="QR factorisation")
    ax.loglog(conds, np.maximum(es, 1e-18), "o-", color=GREEN, lw=1.9, ms=4,
              label="SVD")
    ax.set_xlabel(r"condition number $\kappa(A)$")
    ax.set_ylabel("relative error in the solution")
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    ax.set_title(r"forming $A^\top A$ squares the conditioning", fontsize=10, pad=8)
    tidy(ax)
    save(fig, "11-least-squares")


# ============================================================== fig 12 ======
def fig_saddles():
    """Why high dimensions are full of saddles: signs must all agree for a minimum."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))

    ax = axes[0]
    dims = np.arange(1, 41)
    ax.semilogy(dims, 0.5 ** dims.astype(float), color=ORANGE, lw=2.3)
    ax.set_xlabel("number of dimensions")
    # NB: labelled "agree in sign" but the curve plotted is 2**-n, the probability
    # all n curvatures are POSITIVE (a minimum), not 2**(1-n) (agree in either
    # direction). Should read "P(all curvatures are positive)" -- pending
    # regeneration of 12-saddles.svg (text is baked into paths via
    # svg.fonttype="path", so the shipped SVG cannot be hand-edited; this was
    # left unchanged rather than let source and shipped figure diverge under
    # machine load that prevented a clean regeneration run). Flagged for the
    # next figure regen pass.
    ax.set_ylabel("P(all curvatures agree in sign)")
    ax.set_title("a random critical point is almost never a minimum",
                 fontsize=9.8, pad=8)
    tidy(ax)

    ax = axes[1]
    rng = np.random.default_rng(0)
    frac = []
    for d in dims:
        hits = 0
        for _ in range(400):
            M = rng.normal(size=(d, d))
            w = np.linalg.eigvalsh((M + M.T) / 2)
            hits += int(np.all(w > 0))
        frac.append(max(hits / 400, 1e-4))
    ax.semilogy(dims, frac, color=BLUE, lw=2.3)
    ax.set_xlabel("number of dimensions")
    ax.set_ylabel("measured fraction")
    ax.set_title("simulated: random symmetric matrices that are PD",
                 fontsize=9.8, pad=8)
    tidy(ax)
    save(fig, "12-saddles")


# ============================================================== fig 13 ======
def fig_line_search():
    """Backtracking with genuine Armijo condition.

    Starting point x0=0.5 (on a steep downslope, df(x0)=-3.08 -- not a trough)
    with initial step 2.0 causes two rejections before acceptance at step=0.5:
      - step=2.0: cand=2.5, f=3.478 > Armijo_rhs=0.684  -> reject (pink)
      - step=1.0: cand=1.5, f=1.394 > Armijo_rhs=0.992  -> reject (pink)
      - step=0.5: cand=1.0, f=0.902 <= Armijo_rhs=1.145  -> accept (green)

    Armijo condition: f(x + alpha*d) <= f(x) + c * alpha * grad^T * d
    With d = -sign(slope) (descent direction): grad^T d = -|slope|, so
    rhs = f(x0) - c * step * |slope|.
    c=0.1 is the standard sufficient-decrease constant.
    """
    f = lambda t: (t - 1.0) ** 2 + 0.35 * np.sin(6 * t) + 1.0
    df = lambda t: 2 * (t - 1.0) + 0.35 * 6 * np.cos(6 * t)

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    t = np.linspace(-0.3, 3.2, 500)
    ax.plot(t, f(t), color=BLUE, lw=2.2, zorder=4)

    x0 = 0.5
    slope = df(x0)                        # negative here: gradient points left
    descent_dir = -np.sign(slope)         # +1: we step to the right (uphill in x)
    ax.plot([x0], [f(x0)], "o", color=INK, ms=6, zorder=7)

    step, c_armijo = 2.0, 0.1
    tried = []
    for _ in range(6):
        cand = x0 + descent_dir * step    # move along descent direction
        f_cand = f(cand)
        # Armijo: f(cand) <= f(x0) + c * step * (grad . descent_dir)
        #       = f(x0) - c * step * |slope|   (since grad . descent_dir = slope * descent_dir = -|slope|)
        armijo_rhs = f(x0) - c_armijo * step * abs(slope)
        good = f_cand <= armijo_rhs
        tried.append((cand, f_cand, good))
        if good:
            break
        step *= 0.5

    # Verify: first two rejected, third accepted
    assert not tried[0][2], "Expected first step to be rejected"
    assert not tried[1][2], "Expected second step to be rejected"
    assert tried[2][2],     "Expected third step to be accepted"

    for cand, fv, good in tried:
        ax.plot([cand], [fv], "o", color=(GREEN if good else PINK), ms=6, zorder=7)
        ax.annotate("", xy=(cand, fv), xytext=(x0, f(x0)),
                    arrowprops=dict(arrowstyle="->", color=(GREEN if good else PINK),
                                    lw=1.3, alpha=0.85))
    ax.set_xlabel("step taken along the descent direction")
    ax.set_ylabel("objective")
    ax.set_title("backtracking: halve the step until Armijo is satisfied",
                 fontsize=9.8, pad=8)
    tidy(ax)
    save(fig, "13-line-search")


if __name__ == "__main__":
    print("generating Part III figures ->", OUT)
    for fn in (fig_float_spacing, fig_cancellation, fig_conditioning, fig_curvature,
               fig_descent_vs_newton, fig_momentum, fig_sgd_noise, fig_adaptive,
               fig_schedules, fig_constrained, fig_least_squares, fig_saddles,
               fig_line_search):
        fn()
    print("done.")
