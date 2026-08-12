"""
Reference solutions for the Part X drills.

Look only after you have tried. To check the test suite against these:

    cd ../problems
    python3 -c "import sys; sys.path.insert(0,'../solutions'); \
        import reference, pytest; pytest.main(['test_drills.py','-q'])"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "problems"))


# ----------------------------------------------------------- the operation ----

def output_size(m, k, padding=0, stride=1, dilation=1):
    return (m + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def conv_matrix(K, H, W):
    K = np.asarray(K, float)
    kh, kw = K.shape
    oh, ow = H - kh + 1, W - kw + 1
    M = np.zeros((oh * ow, H * W))
    for i in range(oh):
        for j in range(ow):
            row = np.zeros((H, W))
            row[i:i + kh, j:j + kw] = K
            M[i * ow + j] = row.ravel()
    return M, (oh, ow)


def _pad(x, p):
    if p == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))


def conv2d_naive(x, k, stride=1, padding=0):
    x = _pad(np.asarray(x, float), padding)
    k = np.asarray(k, float)
    n, ci, H, W = x.shape
    co, _, kh, kw = k.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((n, co, oh, ow))
    kf = k.reshape(co, -1)
    for i in range(oh):
        for j in range(ow):
            a, b = i * stride, j * stride
            patch = x[:, :, a:a + kh, b:b + kw].reshape(n, -1)
            out[:, :, i, j] = patch @ kf.T
    return out


def im2col(x, kh, kw, stride=1, padding=0):
    x = np.ascontiguousarray(_pad(np.asarray(x, float), padding))
    n, ci, H, W = x.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    s = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, (n, ci, kh, kw, oh, ow),
        (s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride))
    return np.ascontiguousarray(view).reshape(n, ci * kh * kw, oh * ow)


def conv2d_im2col(x, k, stride=1, padding=0):
    x = np.asarray(x, float)
    k = np.asarray(k, float)
    n, ci, H, W = x.shape
    co, _, kh, kw = k.shape
    oh = (H + 2 * padding - kh) // stride + 1
    ow = (W + 2 * padding - kw) // stride + 1
    cols = im2col(x, kh, kw, stride, padding)
    return (k.reshape(co, -1) @ cols).reshape(n, co, oh, ow)


# -------------------------------------------------------------- backward ----

def conv_backward_input(g, k, in_shape, stride=1, padding=0):
    g = np.asarray(g, float)
    k = np.asarray(k, float)
    n, ci, H, W = in_shape
    co, _, kh, kw = k.shape
    _, _, oh, ow = g.shape
    dx = np.zeros((n, ci, H + 2 * padding, W + 2 * padding))
    for i in range(oh):
        for j in range(ow):
            a, b = i * stride, j * stride
            # g[:, :, i, j] is (n, co); k is (co, ci, kh, kw)
            dx[:, :, a:a + kh, b:b + kw] += np.einsum(
                "nc,cikl->nikl", g[:, :, i, j], k)
    if padding:
        dx = dx[:, :, padding:-padding, padding:-padding]
    return dx


def conv_backward_kernel(g, x, k_shape, stride=1, padding=0):
    g = np.asarray(g, float)
    x = _pad(np.asarray(x, float), padding)
    co, ci, kh, kw = k_shape
    _, _, oh, ow = g.shape
    dk = np.zeros(k_shape)
    for i in range(oh):
        for j in range(ow):
            a, b = i * stride, j * stride
            patch = x[:, :, a:a + kh, b:b + kw]          # (n, ci, kh, kw)
            dk += np.einsum("nc,nikl->cikl", g[:, :, i, j], patch)
    return dk


# -------------------------------------------------------------- pooling ----

def max_pool(x, size, stride=None):
    x = np.asarray(x, float)
    stride = size if stride is None else stride
    n, c, H, W = x.shape
    oh = (H - size) // stride + 1
    ow = (W - size) // stride + 1
    out = np.empty((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            a, b = i * stride, j * stride
            out[:, :, i, j] = x[:, :, a:a + size, b:b + size].max((2, 3))
    return out


def global_avg_pool(x):
    return np.asarray(x, float).mean(axis=(2, 3))


# ---------------------------------------------- equivariance and invariance ----

def translate(x, dy, dx):
    return np.roll(np.asarray(x), shift=(dy, dx), axis=(-2, -1))


def equivariance_error(f, x, dy, dx, margin):
    a = f(translate(x, dy, dx))
    b = translate(f(x), dy, dx)
    if margin:
        a = a[..., margin:-margin, margin:-margin]
        b = b[..., margin:-margin, margin:-margin]
    return float(np.abs(a - b).max())


def invariance_error(f, x, dy, dx):
    base = f(x)
    moved = f(translate(x, dy, dx))
    return float(np.abs(moved - base).mean() / max(np.abs(base).mean(), 1e-12))


def receptive_field(grad_fn, shape, centre, frac=0.95):
    g = np.abs(np.asarray(grad_fn(np.zeros(shape))))
    while g.ndim > 2:
        g = g.sum(axis=0)
    cy, cx = centre
    nz = np.argwhere(g > 0)
    support = 0 if len(nz) == 0 else int(max(nz.max(0) - nz.min(0)) + 1)
    tot = g.sum()
    rad = 0
    limit = max(g.shape)
    while rad < limit:
        lo_y, hi_y = max(0, cy - rad), min(g.shape[0], cy + rad + 1)
        lo_x, hi_x = max(0, cx - rad), min(g.shape[1], cx + rad + 1)
        if g[lo_y:hi_y, lo_x:hi_x].sum() >= frac * tot:
            break
        rad += 1
    return support, 2 * rad + 1


# --------------------------------------------------------------- counting ----

def conv_params(c_in, c_out, k, groups=1, bias=True):
    return c_out * (c_in // groups) * k * k + (c_out if bias else 0)


def separable_ratio(c_out, k):
    return 1.0 / c_out + 1.0 / (k * k)


def conv_macs(c_in, c_out, k, out_h, out_w, groups=1):
    return c_out * (c_in // groups) * k * k * out_h * out_w


# ------------------------------------------------------------ transposed ----

def conv_transpose2d(x, k, stride=1, padding=0):
    x = np.asarray(x, float)
    k = np.asarray(k, float)
    n, ci, H, W = x.shape
    _, co, kh, kw = k.shape
    oh = (H - 1) * stride + kh
    ow = (W - 1) * stride + kw
    out = np.zeros((n, co, oh, ow))
    for i in range(H):
        for j in range(W):
            a, b = i * stride, j * stride
            out[:, :, a:a + kh, b:b + kw] += np.einsum(
                "ni,iokl->nokl", x[:, :, i, j], k)
    if padding:
        out = out[:, :, padding:-padding, padding:-padding]
    return out


def checkerboard_values(k, s, n=12):
    x = np.ones((1, 1, n, n))
    w = np.ones((1, 1, k, k))
    y = conv_transpose2d(x, w, stride=s)[0, 0]
    inner = y[k:-k, k:-k]
    return np.unique(np.round(inner, 9))


# ------------------------------------------------------------------ Gabor ----

def gabor(K, alpha=1.0, beta_x=0.02, beta_y=0.02, f=0.8, phi=0.0,
          x0=0.0, y0=0.0, tau=0.0):
    yy, xx = np.mgrid[:K, :K] - (K - 1) / 2.0
    xr = (xx - x0) * np.cos(tau) + (yy - y0) * np.sin(tau)
    yr = -(xx - x0) * np.sin(tau) + (yy - y0) * np.cos(tau)
    return alpha * np.exp(-beta_x * xr ** 2 - beta_y * yr ** 2) * np.cos(f * xr + phi)


def complex_cell(s0, s1, image):
    a = float((np.asarray(s0) * np.asarray(image)).sum())
    b = float((np.asarray(s1) * np.asarray(image)).sum())
    return float(np.sqrt(a * a + b * b))


import drills  # noqa: E402

for _name, _fn in list(globals().items()):
    if callable(_fn) and hasattr(drills, _name):
        setattr(drills, _name, _fn)
