"""
Part X drills — implementation problems.

    python3 -m pytest test_drills.py -v

Rules:
  * NumPy only. NOT allowed: torch, scipy, sklearn.
  * np.lib.stride_tricks IS allowed, and you will want it for im2col.

Most of this part's claims are exact rather than approximate — the matrix
equivalence, the flipped-kernel backward pass, the separable ratio, the
checkerboard condition. The tests demand exactness where the book claims it.
"""
import numpy as np


# ----------------------------------------------------------- the operation ----

def output_size(m, k, padding=0, stride=1, dilation=1):
    """Spatial size of a convolution output.  (§5)

    floor((m + 2p - d(k-1) - 1) / s) + 1
    """
    raise NotImplementedError("TODO: output_size")


def conv_matrix(K, H, W):
    """The explicit matrix M with M @ x.ravel() == valid cross-correlation.  (§4)

    K is (kh, kw); the image is (H, W). Return (M, (oh, ow)) where M has shape
    (oh*ow, H*W).

    Row (i*ow + j) is a blank image with K laid in at offset (i, j), flattened.
    """
    raise NotImplementedError("TODO: conv_matrix")


def conv2d_naive(x, k, stride=1, padding=0):
    """Multi-channel cross-correlation by looping over output positions.  (§6)

    x is (N, Cin, H, W); k is (Cout, Cin, kh, kw). Zero-pad by `padding` on all
    four sides. Return (N, Cout, oh, ow).
    """
    raise NotImplementedError("TODO: conv2d_naive")


def im2col(x, kh, kw, stride=1, padding=0):
    """Gather every patch the kernel will see into a matrix.  (§17)

    x is (N, Cin, H, W). Return an array of shape (N, Cin*kh*kw, oh*ow).

    A strided view is the cheap way to build it, but you must copy before
    reshaping or the matrix multiply will not reach BLAS -- which is the whole
    point of the technique.
    """
    raise NotImplementedError("TODO: im2col")


def conv2d_im2col(x, k, stride=1, padding=0):
    """The same convolution as conv2d_naive, via im2col and one matmul.  (§17)"""
    raise NotImplementedError("TODO: conv2d_im2col")


# -------------------------------------------------------------- backward ----

def conv_backward_input(g, k, in_shape, stride=1, padding=0):
    """dJ/dx given dJ/dz.  (§16)

    g is (N, Cout, oh, ow); k is (Cout, Cin, kh, kw); in_shape is (N, Cin, H, W).
    Return dJ/dx with shape in_shape.

    With stride 1 and no padding this equals a FULL convolution of g with the
    kernel flipped in both spatial axes and with the channel axes swapped. The
    test checks that exactly.
    """
    raise NotImplementedError("TODO: conv_backward_input")


def conv_backward_kernel(g, x, k_shape, stride=1, padding=0):
    """dJ/dK given dJ/dz.  (§16)

    Return an array of shape k_shape.
    """
    raise NotImplementedError("TODO: conv_backward_kernel")


# -------------------------------------------------------------- pooling ----

def max_pool(x, size, stride=None):
    """Max pooling over non-overlapping (or strided) windows.  (§12)

    x is (N, C, H, W). stride defaults to size. Return (N, C, oh, ow).
    """
    raise NotImplementedError("TODO: max_pool")


def global_avg_pool(x):
    """Average each channel over all spatial positions.  (§15)

    x is (N, C, H, W) -> (N, C).
    """
    raise NotImplementedError("TODO: global_avg_pool")


# ---------------------------------------------- equivariance and invariance ----

def translate(x, dy, dx):
    """Circular shift along the two spatial axes.  (§9)"""
    raise NotImplementedError("TODO: translate")


def equivariance_error(f, x, dy, dx, margin):
    """max |f(shift(x)) - shift(f(x))|, ignoring `margin` pixels of border.  (§9)

    The border must be excluded because a circular shift wraps content in from
    the far side. Return the max absolute difference over the interior.
    """
    raise NotImplementedError("TODO: equivariance_error")


def invariance_error(f, x, dy, dx):
    """mean |f(shift(x)) - f(x)| / mean |f(x)|.  (§13)

    Note what is NOT here: the reference is not shifted. That single difference
    is what separates this from equivariance_error.
    """
    raise NotImplementedError("TODO: invariance_error")


def receptive_field(grad_fn, shape, centre, frac=0.95):
    """Measure a receptive field from a gradient.  (§10)

    grad_fn(x) -> the gradient of one central output unit with respect to x,
    with the same shape as x.

    Return (support, effective) where `support` is the side of the smallest
    square containing every nonzero, and `effective` is the side of the smallest
    centred square holding `frac` of the total absolute gradient.
    """
    raise NotImplementedError("TODO: receptive_field")


# --------------------------------------------------------------- counting ----

def conv_params(c_in, c_out, k, groups=1, bias=True):
    """Parameters in a convolutional layer.  (§6, §19)"""
    raise NotImplementedError("TODO: conv_params")


def separable_ratio(c_out, k):
    """Depthwise separable parameters divided by standard.  (§19)

    The closed form is 1/c_out + 1/k**2, and it is exact.
    """
    raise NotImplementedError("TODO: separable_ratio")


def conv_macs(c_in, c_out, k, out_h, out_w, groups=1):
    """Multiply-accumulates for one image.  (§6)"""
    raise NotImplementedError("TODO: conv_macs")


# ------------------------------------------------------------ transposed ----

def conv_transpose2d(x, k, stride=1, padding=0):
    """Transposed convolution.  (§27)

    x is (N, Cin, H, W); k is (Cin, Cout, kh, kw) -- note the channel order,
    which matches the transpose. Return (N, Cout, oh, ow) with
    oh = (H-1)*stride - 2*padding + kh.

    Scatter-add: each input position contributes the kernel, scaled, to a
    stride-spaced location in the output.
    """
    raise NotImplementedError("TODO: conv_transpose2d")


def checkerboard_values(k, s, n=12):
    """Distinct interior values from a constant input and a constant kernel. (§27)

    Run conv_transpose2d on an (1,1,n,n) array of ones with a (1,1,k,k) kernel
    of ones, discard a border of k, and return the sorted distinct values
    (rounded to 9 decimals) of what remains.

    One value means uniform; more means a checkerboard. The condition is
    k % s == 0.
    """
    raise NotImplementedError("TODO: checkerboard_values")


# ------------------------------------------------------------------ Gabor ----

def gabor(K, alpha=1.0, beta_x=0.02, beta_y=0.02, f=0.8, phi=0.0,
          x0=0.0, y0=0.0, tau=0.0):
    """A Gabor function on a K x K grid centred at the middle.  (§31)

    w = alpha * exp(-beta_x*x'^2 - beta_y*y'^2) * cos(f*x' + phi)

    with x' =  (x-x0)cos(tau) + (y-y0)sin(tau)
         y' = -(x-x0)sin(tau) + (y-y0)cos(tau)
    """
    raise NotImplementedError("TODO: gabor")


def complex_cell(s0, s1, image):
    """sqrt(<s0,image>^2 + <s1,image>^2) -- the L2 norm of a quadrature pair. (§31)"""
    raise NotImplementedError("TODO: complex_cell")
