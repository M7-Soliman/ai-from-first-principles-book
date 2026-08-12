"""
Reference solutions for the Part XXIX drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------- matching

def iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    bb = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return inter / (aa + bb - inter + 1e-12)


def average_precision(dets, scores, gts, thr=0.5):
    dets = np.asarray(dets, float).reshape(-1, 4)
    gts = np.asarray(gts, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, float))
    dets = dets[order]
    used = np.zeros(len(gts), bool)
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, d in enumerate(dets):
        if len(gts) == 0:
            fp[i] = 1; continue
        o = np.where(used, -1.0, iou(d[None, :], gts))
        j = int(np.argmax(o))
        if o[j] >= thr:
            tp[i] = 1; used[j] = True
        else:
            fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(len(gts), 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def nms(boxes, scores, thr=0.5):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, float))
    keep = []
    while len(order):
        i = int(order[0]); keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou(boxes[i][None, :], boxes[order[1:]]) < thr]
    return np.array(keep, int)


def hungarian(cost):
    """Exact minimum-cost assignment for a square or rectangular matrix, by the
    classic augmenting-path method. Returns row -> column."""
    C = np.asarray(cost, float)
    n, m = C.shape
    if n > m:
        return _invert(hungarian(C.T), n, m)
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, int); way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf); usedj = np.zeros(m + 1, bool)
        while True:
            usedj[j0] = True
            i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, m + 1):
                if usedj[j]:
                    continue
                cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur; way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]; j1 = j
            for j in range(m + 1):
                if usedj[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    out = -np.ones(n, int)
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def _invert(assign, n, m):
    out = -np.ones(n, int)
    for c, r in enumerate(assign):
        if r >= 0:
            out[r] = c
    return out


# ------------------------------------------------------------------ detection

def anchor_count(size, strides=(8, 16, 32), scales=3, ratios=3):
    return int(sum((size // s) ** 2 for s in strides) * scales * ratios)


def focal_weight(p, gamma=2.0):
    """The factor the focal loss applies to a correctly-classified example of
    confidence p."""
    return float((1 - p) ** gamma)


def feature_pixels(obj_px, stages):
    return float((obj_px / 2 ** stages) ** 2)


# -------------------------------------------------------------------- dense

def mean_iou(pred, true, n_classes):
    pred = np.asarray(pred); true = np.asarray(true)
    out = []
    for c in range(n_classes):
        inter = np.sum((pred == c) & (true == c))
        union = np.sum((pred == c) | (true == c))
        if union:
            out.append(inter / union)
    return float(np.mean(out)) if out else 0.0


def structure_tensor(patch):
    gy, gx = np.gradient(np.asarray(patch, float))
    return np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                     [np.sum(gx * gy), np.sum(gy * gy)]])


def flow_is_determined(patch, tol=1e-9):
    ev = np.linalg.eigvalsh(structure_tensor(patch))
    return bool(ev.min() > tol * max(ev.max(), 1.0))


# ----------------------------------------------------------------- geometry

def project(K, R, t, X):
    X = np.asarray(X, float).reshape(-1, 3)
    Xc = X @ np.asarray(R, float).T + np.asarray(t, float)
    x = Xc @ np.asarray(K, float).T
    return x[:, :2] / x[:, 2:3]


def unproject(K, uv, depth):
    uv = np.asarray(uv, float).reshape(-1, 2)
    d = np.asarray(depth, float).reshape(-1, 1)
    Ki = np.linalg.inv(np.asarray(K, float))
    h = np.concatenate([uv, np.ones((len(uv), 1))], 1)
    return (h @ Ki.T) * d


def scale_intrinsics(K, sx, sy):
    K = np.array(K, float, copy=True)
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def radial_distort(K, uv, k1, k2=0.0):
    K = np.asarray(K, float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    xn = (uv[:, 0] - K[0, 2]) / K[0, 0]
    yn = (uv[:, 1] - K[1, 2]) / K[1, 1]
    r2 = xn ** 2 + yn ** 2
    f = 1 + k1 * r2 + k2 * r2 ** 2
    return np.stack([xn * f * K[0, 0] + K[0, 2], yn * f * K[1, 1] + K[1, 2]], 1)


def stereo_depth(f, baseline, disparity):
    return float(f) * float(baseline) / np.asarray(disparity, float)


def stereo_depth_error(f, baseline, z, disp_err):
    return np.asarray(z, float) ** 2 * float(disp_err) / (float(f) * float(baseline))


def fundamental_from_poses(K1, K2, R, t):
    t = np.asarray(t, float).ravel()
    Tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], float)
    E = Tx @ np.asarray(R, float)
    return np.linalg.inv(np.asarray(K2, float)).T @ E @ np.linalg.inv(np.asarray(K1, float))


def epipolar_residual(F, x1, x2):
    """Distance in PIXELS from each x2 to the epipolar line F x1. The raw
    algebraic form |x2^T F x1| is not usable: F is defined only up to scale, so
    that number can be made anything by rescaling F and cannot be compared with
    a tolerance."""
    F = np.asarray(F, float)
    x1 = np.asarray(x1, float).reshape(-1, 2); x2 = np.asarray(x2, float).reshape(-1, 2)
    h1 = np.concatenate([x1, np.ones((len(x1), 1))], 1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1))], 1)
    line = h1 @ F.T                                    # the epipolar line in image 2
    num = np.abs(np.einsum("ij,ij->i", h2, line))
    return num / np.maximum(np.hypot(line[:, 0], line[:, 1]), 1e-300)


# ----------------------------------------------------------------- tracking

def constant_velocity(pos, vel, steps):
    return np.asarray(pos, float) + np.asarray(vel, float) * float(steps)


def associate(tracks, dets, gate=np.inf):
    cost = np.linalg.norm(np.asarray(tracks, float)[:, None, :]
                          - np.asarray(dets, float)[None, :, :], axis=-1)
    big = cost.max() * 10 + 1.0
    assign = hungarian(np.where(cost < gate, cost, big))
    out = np.array(assign, int)
    for i, j in enumerate(out):
        if j >= 0 and cost[i, j] >= gate:
            out[i] = -1
    return out


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
