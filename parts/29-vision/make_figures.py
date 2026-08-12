"""
Figures for Part XXIX — Perception: Vision.

Predictions for every one of these were recorded before the
script was run.

    python3 make_figures.py

Runtime: about two minutes. Numpy and matplotlib only.

Every scene here is synthetic with exact ground truth — known camera matrices,
known object positions, known motion — so each error is measured against a number
that was set rather than estimated. None of these figures is a claim about what a
particular network scores on a particular benchmark; they are the geometric and
metric facts that hold whatever the network is.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "vs"))
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


# ------------------------------------------------------------ detection tools

def iou(a, b):
    """a, b are (..., 4) boxes as x1, y1, x2, y2."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    bb = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (aa + bb - inter + 1e-12)


def average_precision(dets, scores, gts, thr):
    """Standard greedy matching AP: sort by score, each ground truth may be
    matched once, then the area under the interpolated precision-recall curve."""
    order = np.argsort(-np.asarray(scores))
    dets = np.asarray(dets, float)[order]
    used = np.zeros(len(gts), bool)
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, d in enumerate(dets):
        if len(gts) == 0:
            fp[i] = 1; continue
        o = iou(d[None, :], np.asarray(gts, float))
        o = np.where(used, -1.0, o)
        j = int(np.argmax(o))
        if o[j] >= thr:
            tp[i] = 1; used[j] = True
        else:
            fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(len(gts), 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    # all-point interpolation
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def nms(boxes, scores, thr):
    order = np.argsort(-np.asarray(scores))
    boxes = np.asarray(boxes, float)
    keep = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1:
            break
        o = iou(boxes[i][None, :], boxes[order[1:]])
        order = order[1:][o < thr]
    return np.array(keep, int)


def scene(n, size=512, box=40, jitter=0.0, seed=0, spacing=None):
    """n non-overlapping ground-truth boxes, plus detections displaced by
    `jitter` pixels of localisation noise."""
    r = np.random.default_rng(seed)
    if spacing is None:
        cx = r.uniform(box, size - box, n); cy = r.uniform(box, size - box, n)
    else:
        k = int(np.ceil(np.sqrt(n)))
        g = (np.arange(k) - (k - 1) / 2) * spacing + size / 2
        X, Y = np.meshgrid(g, g)
        cx, cy = X.ravel()[:n], Y.ravel()[:n]
    gts = np.stack([cx - box / 2, cy - box / 2, cx + box / 2, cy + box / 2], 1)
    d = gts + r.normal(0, jitter, gts.shape) if jitter else gts.copy()
    return gts, d


# ============================================================ FIGURE 1
# The metric is a threshold.

def figure1():
    thrs = np.arange(0.5, 0.96, 0.05)
    curves = {}
    for name, jit in (("localisation sd 2 px", 2.0), ("localisation sd 6 px", 6.0)):
        aps = []
        for t in thrs:
            per = []
            for s in range(12):
                gts, dets = scene(30, jitter=jit, seed=s)
                sc = np.random.default_rng(100 + s).uniform(0.5, 1.0, len(dets))
                per.append(average_precision(dets, sc, gts, t))
            aps.append(np.mean(per))
        curves[name] = np.array(aps)
        print(f"  {name}: AP@0.5 {aps[0]:.3f}  AP@0.75 {aps[5]:.3f}  AP@0.9 {aps[-2]:.3f}")

    # A pair that swaps order. The first attempt — jitter 1.5 with 30% missed
    # against jitter 7.0 finding everything — did NOT cross: the gap stayed
    # positive at both ends. P1.3 is an existence claim, so the pair below is
    # chosen from the arithmetic rather than by chance, and that is worth stating
    # plainly. A detector with recall capped at r and near-perfect localisation
    # has AP close to r at every threshold; a detector with full recall and
    # localisation noise has AP falling with the threshold. They cross wherever r
    # sits between the noisy detector's loose and strict values.
    swap = {}
    for name, jit, miss in (("precise, misses half", 1.0, 0.50), ("loose, finds all", 4.0, 0.0)):
        aps = []
        for t in thrs:
            per = []
            for s in range(12):
                gts, dets = scene(30, jitter=jit, seed=s)
                r = np.random.default_rng(200 + s)
                keep = r.random(len(dets)) >= miss
                d, sc = dets[keep], r.uniform(0.5, 1.0, int(keep.sum()))
                per.append(average_precision(d, sc, gts, t))
            aps.append(np.mean(per))
        swap[name] = np.array(aps)
    lo = swap["precise, misses half"][0] - swap["loose, finds all"][0]
    hi = swap["precise, misses half"][-1] - swap["loose, finds all"][-1]
    print(f"  swap check: at IoU 0.5 the gap is {lo:+.3f}, at IoU 0.95 it is {hi:+.3f}"
          f"  -> {'ORDER REVERSES' if lo * hi < 0 else 'no reversal'}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for (n, v), c in zip(curves.items(), (BLUE, ORANGE)):
        a1.plot(thrs, v, "o-", color=c, lw=1.9, ms=4.5, label=n)
    a1.set_xlabel("IoU threshold"); a1.set_ylabel("average precision")
    a1.set_ylim(-0.03, 1.05)
    a1.set_title("The same detector, different metric", fontsize=10, pad=8)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    tidy(a1)
    for (n, v), c in zip(swap.items(), (GREEN, PURPLE)):
        a2.plot(thrs, v, "o-", color=c, lw=1.9, ms=4.5, label=n)
    a2.set_xlabel("IoU threshold"); a2.set_ylabel("average precision")
    a2.set_ylim(-0.03, 1.05)
    a2.set_title("Two detectors, and which is better", fontsize=10, pad=8)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    tidy(a2)
    fig.tight_layout()
    save(fig, "01-iou")
    return thrs, curves, swap


# ============================================================ FIGURE 2
# Suppression against crowding.

def figure2():
    spacings = np.arange(12, 61, 4)
    res = {}
    for thr in (0.3, 0.5, 0.7):
        rec, prec = [], []
        for sp in spacings:
            R, P = [], []
            for s in range(8):
                gts, _ = scene(16, size=512, box=40, seed=s, spacing=float(sp))
                r = np.random.default_rng(300 + s)
                # three proposals per object, jittered — what a dense head emits
                props = np.repeat(gts, 3, axis=0) + r.normal(0, 3.0, (len(gts) * 3, 4))
                sc = r.uniform(0.4, 1.0, len(props))
                keep = nms(props, sc, thr)
                d, ds = props[keep], sc[keep]
                used = np.zeros(len(gts), bool); tp = 0
                for i in np.argsort(-ds):
                    o = np.where(used, -1.0, iou(d[i][None, :], gts))
                    j = int(np.argmax(o))
                    if o[j] >= 0.5:
                        tp += 1; used[j] = True
                R.append(tp / len(gts)); P.append(tp / max(len(d), 1))
            rec.append(np.mean(R)); prec.append(np.mean(P))
        res[thr] = (np.array(rec), np.array(prec))
        print(f"  NMS {thr:.1f}: recall {rec[0]:.3f} (tight) -> {rec[-1]:.3f} (spread);"
              f"  precision {prec[0]:.3f} -> {prec[-1]:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
    for (thr, (rec, prec)), c in zip(res.items(), (BLUE, GREEN, ORANGE)):
        a1.plot(spacings, rec, "o-", color=c, lw=1.8, ms=4, label=f"NMS IoU {thr}")
        a2.plot(spacings, prec, "o-", color=c, lw=1.8, ms=4, label=f"NMS IoU {thr}")
    for a, t, yl in ((a1, "Recall", "recall"), (a2, "Precision", "precision")):
        a.set_xlabel("centre spacing between objects (px, boxes are 40 px)")
        a.set_ylabel(yl); a.set_ylim(-0.03, 1.05)
        a.set_title(t, fontsize=10, pad=8)
        a.legend(frameon=False, fontsize=8, loc="lower right")
        tidy(a)
    fig.tight_layout()
    save(fig, "02-nms")
    return spacings, res


# ============================================================ FIGURE 3
# What the detector is mostly looking at.

def figure3():
    sizes = [256, 512, 1024, 2048]
    n_obj = 8
    scales, ratios, strides = 3, 3, (8, 16, 32)
    rows = []
    for S in sizes:
        total = sum((S // st) ** 2 for st in strides) * scales * ratios
        pos = n_obj * scales * ratios          # roughly one matched anchor set per object
        rows.append((S, total, pos, total / pos))
        print(f"  {S:5d}px image: {total:>10,} anchors, ~{pos} positive -> "
              f"{total/pos:>8,.0f} : 1")
    rows = np.array(rows, float)

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.loglog(rows[:, 0], rows[:, 3], "o-", color=ORANGE, lw=1.9, ms=6)
    for S, tot, pos, ratio in rows:
        ax.annotate(f"{ratio:,.0f}:1", (S, ratio), textcoords="offset points",
                    xytext=(6, -11), fontsize=8, color=GRAY)
    ax.set_xlabel("image side (px)")
    ax.set_ylabel("negative anchors per positive")
    ax.set_title("Almost every location is background, and more so at higher resolution",
                 fontsize=10.5, pad=10)
    tidy(ax)
    save(fig, "03-imbalance")
    return rows


# ============================================================ FIGURE 4
# Small objects and downsampling.

def figure4():
    obj = np.array([8, 16, 32, 64, 128], float)
    stages = np.arange(0, 6)
    grid = np.zeros((len(obj), len(stages)))
    for i, o in enumerate(obj):
        for j, s in enumerate(stages):
            grid[i, j] = (o / 2 ** s) ** 2
    for i, o in enumerate(obj):
        print(f"  {int(o):3d}px object: " +
              "  ".join(f"s{j}:{grid[i, j]:8.2f}" for j in stages))
    gone = [(int(o), int(stages[np.argmax(grid[i] < 1.0)]) if (grid[i] < 1.0).any() else None)
            for i, o in enumerate(obj)]
    print("  first stage at which the object covers less than one feature pixel:", gone)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, o in enumerate(obj):
        ax.semilogy(stages, np.maximum(grid[i], 1e-3), "o-", lw=1.8, ms=4.5,
                    label=f"{int(o)} px object")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.text(5.05, 1.0, " one pixel", va="center", fontsize=8, color=INK)
    ax.set_xlabel("stride-2 stages"); ax.set_ylabel("feature-map pixels covering the object")
    ax.set_xticks(stages)
    ax.set_title("Where the small objects go", fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    tidy(ax)
    save(fig, "04-scale")
    return obj, grid


# ============================================================ FIGURE 6
# Calibration.

def figure5():
    W = H = 1024
    f = 500.0                                  # a wide field of view
    cx, cy = W / 2, H / 2
    ks = [(-0.05, 0.0), (-0.15, 0.02), (-0.30, 0.10)]
    rr = np.linspace(0, np.hypot(W / 2, H / 2), 200)
    curves = []
    for k1, k2 in ks:
        xn = rr / f
        r2 = xn ** 2
        dist = xn * (1 + k1 * r2 + k2 * r2 ** 2)
        err = np.abs(dist * f - rr)
        curves.append(err)
        print(f"  k1={k1:+.2f} k2={k2:+.2f}:  centre {err[0]:.3f} px, "
              f"half-way {err[len(rr)//2]:6.2f} px, corner {err[-1]:7.2f} px")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (k1, k2), e, c in zip(ks, curves, (BLUE, GREEN, ORANGE)):
        ax.plot(rr, e, color=c, lw=2.0, label=f"$k_1$ = {k1}, $k_2$ = {k2}")
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.text(5, 1.35, "one pixel", fontsize=8, color=GRAY)
    ax.set_yscale("log")
    ax.set_xlabel("distance from the image centre (px)")
    ax.set_ylabel("reprojection error if distortion is ignored (px)")
    ax.set_title("Negligible in the middle, tens of pixels at the corner",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "06-calibration")
    return rr, curves


# ============================================================ FIGURE 7
# Depth from stereo.

def figure6():
    f = 800.0                                  # pixels
    z = np.linspace(5, 150, 300)
    d_err = 0.1                                # a tenth of a pixel of disparity error
    rows = {}
    for b in (0.12, 0.25, 0.50):
        dz = z ** 2 * d_err / (f * b)
        rows[b] = dz
        at = lambda zz: dz[np.argmin(np.abs(z - zz))]
        print(f"  baseline {b*100:4.0f} cm: depth error {at(10):.3f} m at 10 m, "
              f"{at(50):.2f} m at 50 m, {at(100):.2f} m at 100 m")
    r = np.array([rows[0.12][-1] / rows[0.25][-1], rows[0.25][-1] / rows[0.50][-1]])
    print(f"  doubling the baseline divides the error by {r.mean():.3f} (predicted 2.0)")
    # confirm the quadratic exponent by fitting rather than asserting it
    slope = np.polyfit(np.log(z), np.log(rows[0.25]), 1)[0]
    print(f"  fitted exponent of depth error against range: {slope:.3f} (predicted 2.0)")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for b, c in zip(rows, (ORANGE, GREEN, BLUE)):
        ax.plot(z, rows[b], color=c, lw=2.0, label=f"baseline {b*100:.0f} cm")
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.text(6, 1.2, "one metre", fontsize=8, color=GRAY)
    ax.set_yscale("log")
    ax.set_xlabel("true depth (m)")
    ax.set_ylabel("depth error from 0.1 px of disparity error (m)")
    ax.set_title("Stereo depth error grows with the square of range",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax)
    save(fig, "07-stereo")
    return z, rows, slope


# ============================================================ FIGURE 5
# The aperture problem.

def structure_tensor(patch):
    gy, gx = np.gradient(patch)
    return np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                     [np.sum(gx * gy), np.sum(gy * gy)]])


def figure7():
    n = 41
    y, x = np.mgrid[0:n, 0:n].astype(float)
    edge = np.tanh((x - n / 2) / 1.5)                       # a vertical edge
    corner = np.tanh((x - n / 2) / 1.5) * np.tanh((y - n / 2) / 1.5)

    us = np.linspace(-2, 2, 121)
    vs = np.linspace(-2, 2, 121)
    out = {}
    for name, img in (("edge", edge), ("corner", corner)):
        gy, gx = np.gradient(img)
        true_u, true_v = 1.0, 0.7
        It = -(gx * true_u + gy * true_v)                   # brightness constancy
        R = np.zeros((len(vs), len(us)))
        for i, v in enumerate(vs):
            for j, u in enumerate(us):
                R[i, j] = np.sum((gx * u + gy * v + It) ** 2)
        R /= R.max() + 1e-12
        A = structure_tensor(img)
        ev = np.sort(np.linalg.eigvalsh(A))
        out[name] = (R, ev)
        flat = (R < 1e-9).sum()
        print(f"  {name:6s}: structure-tensor eigenvalues {ev[0]:.3e}, {ev[1]:.3e}"
              f"   ratio {ev[1]/max(ev[0],1e-30):.2e}   exact-zero-residual cells {flat}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (name, (R, ev)) in zip(axes, out.items()):
        im = ax.imshow(np.log10(R + 1e-12), origin="lower", cmap="magma",
                       extent=[us[0], us[-1], vs[0], vs[-1]], aspect="equal")
        ax.plot(1.0, 0.7, "o", color="white", ms=6, markeredgecolor="black")
        ax.set_xlabel("$u$"); ax.set_ylabel("$v$")
        ax.set_title(f"{name}   $\\lambda_{{\\min}}$ = {ev[0]:.1e}", fontsize=9.5, pad=7)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.82, label="$\\log_{10}$ residual")
    save(fig, "05-aperture")
    return out


# ============================================================ FIGURE 8
# Tracking through occlusion.
#
# Three things had to be fixed in this experiment before it measured anything,
# and all three were caught before an outcome was read.
#   1. Counting total switches over a sequence confounds the gap with the number
#      of visible frames left to switch on — longer gaps scored BETTER. The
#      metric is now the fraction misidentified at the first frame after the
#      gap, whose denominator does not move.
#   2. Exactly linear trajectories make a constant-velocity model exactly right,
#      so it scored zero at every gap and there was nothing to measure. The
#      trajectories now carry acceleration noise.
#   3. The occlusion has to contain the crossing, or the crossing dominates.

def reacquire_errors(traj, t0, gap, motion, gate=90.0):
    """Track up to t0, skip `gap` frames, then associate at t0 + gap. Returns the
    number of objects whose identity is wrong on that first frame back."""
    k = traj.shape[1]
    last = traj[t0 - 1].copy()
    vel = traj[t0 - 1] - traj[t0 - 2]
    ident = np.arange(k)
    pred = last + vel * (gap + 1) if motion else last
    obs = traj[t0 + gap]
    cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=-1)
    taken, wrong = set(), 0
    for tr in np.argsort(cost.min(1)):
        for o in np.argsort(cost[tr]):
            if o in taken or cost[tr, o] >= gate:
                continue
            taken.add(int(o))
            if o != ident[tr]:
                wrong += 1
            break
        else:
            wrong += 1                      # lost the object entirely
    return wrong


def figure8():
    """Objects on crossing trajectories, occluded through the crossing. The
    constant-velocity model has to bridge a gap during which the objects have
    swapped places, which is the case position alone cannot survive."""
    gaps = np.arange(0, 17, 2)
    dens = [2, 4, 8]
    res = {}
    for k in dens:
        no_motion, with_motion = [], []
        for gap in gaps:
            a, b, n = 0, 0, 0
            for s in range(120):
                r = np.random.default_rng(s + 7919 * k + 104729 * gap)
                T = 60
                ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + r.uniform(0, 1)
                pos = 250 + 160 * np.stack([np.cos(ang), np.sin(ang)], 1)
                vel = (250 - pos) / 20
                traj = np.empty((T, k, 2)); traj[0] = pos
                v = vel.copy()
                for t in range(1, T):
                    v = v + r.normal(0, 0.25, (k, 2))     # acceleration noise
                    traj[t] = traj[t - 1] + v
                t0 = 20 - gap // 2                        # the gap straddles the crossing
                a += reacquire_errors(traj, t0, gap, motion=False)
                b += reacquire_errors(traj, t0, gap, motion=True)
                n += k
            no_motion.append(a / n); with_motion.append(b / n)
        res[k] = (np.array(no_motion), np.array(with_motion))
        print(f"  {k} objects: position only {no_motion[0]*100:5.1f}% -> {no_motion[-1]*100:5.1f}%"
              f"   motion model {with_motion[0]*100:5.1f}% -> {with_motion[-1]*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for k, c in zip(dens, (BLUE, GREEN, ORANGE)):
        ax.plot(gaps, res[k][0] * 100, "o-", color=c, lw=1.9, ms=4.5,
                label=f"{k} objects, position only")
        ax.plot(gaps, res[k][1] * 100, "s--", color=c, lw=1.5, ms=3.5, alpha=0.85,
                label=f"{k} objects, motion model")
    ax.set_xlabel("frames occluded, straddling the crossing")
    ax.set_ylabel("% misidentified on the first frame back")
    ax.set_title("What a motion model buys, and where it stops buying",
                 fontsize=10.5, pad=10)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left")
    tidy(ax)
    save(fig, "08-tracking")
    return gaps, res


if __name__ == "__main__":
    print("Figure 1 — the IoU threshold")
    figure1()
    print("Figure 2 — suppression against crowding")
    figure2()
    print("Figure 3 — anchor imbalance")
    figure3()
    print("Figure 4 — small objects and downsampling")
    figure4()
    print("Figure 5 — the aperture problem")
    figure7()
    print("Figure 6 — calibration")
    figure5()
    print("Figure 7 — stereo depth")
    figure6()
    print("Figure 8 — tracking through occlusion")
    figure8()
    print("done")
