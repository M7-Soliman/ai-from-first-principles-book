"""
Tests for the Part XXIX drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Every scene here is synthetic with exact ground truth. Several tests assert an
EXACT value rather than a tolerance, because in this part several answers are
geometry rather than estimates — the aperture problem's zero, the stereo error's
exponent, and the distortion's vanishing at the principal point.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def box(cx, cy, w=40.0, h=40.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ------------------------------------------------------------------- matching

def test_iou_of_a_box_with_itself_is_one():
    b = box(100, 100)
    assert attempt(drills.iou, b, b) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero_and_not_negative():
    """Without clipping the intersection at zero, the overlap goes negative and
    quietly poisons the denominator."""
    assert attempt(drills.iou, box(0, 0), box(500, 500)) == pytest.approx(0.0)
    assert attempt(drills.iou, box(0, 0), box(0, 500)) == pytest.approx(0.0)


def test_iou_is_scale_invariant():
    a = attempt(drills.iou, box(0, 0, 40, 40), box(20, 0, 40, 40))
    b = attempt(drills.iou, box(0, 0, 400, 400), box(200, 0, 400, 400))
    assert a == pytest.approx(b)


def test_a_perfect_detector_scores_one():
    gts = np.stack([box(100 * i, 100) for i in range(1, 6)])
    assert attempt(drills.average_precision, gts, np.arange(5)[::-1], gts, 0.5) == \
        pytest.approx(1.0)


def test_a_second_detection_on_one_object_is_a_false_positive():
    """Which is what makes suppression necessary rather than optional."""
    gts = np.stack([box(100, 100)])
    dets = np.stack([box(100, 100), box(102, 100)])
    ap = attempt(drills.average_precision, dets, [1.0, 0.9], gts, 0.5)
    assert ap == pytest.approx(1.0)                 # recall is still full
    assert attempt(drills.iou, dets[1], gts[0]) > 0.5   # and it did overlap


def test_average_precision_falls_as_the_threshold_rises():
    r = np.random.default_rng(0)
    gts = np.stack([box(*r.uniform(60, 450, 2)) for _ in range(25)])
    dets = gts + r.normal(0, 4.0, gts.shape)
    sc = r.uniform(0.5, 1.0, len(dets))
    aps = [attempt(drills.average_precision, dets, sc, gts, t) for t in (0.5, 0.7, 0.9)]
    assert aps[0] > aps[1] > aps[2]


def test_suppression_removes_duplicates():
    boxes = np.stack([box(100, 100), box(103, 101), box(300, 300)])
    keep = attempt(drills.nms, boxes, [1.0, 0.9, 0.8], 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_a_tight_threshold_deletes_real_objects():
    """Figure 2's finding: at a low threshold, genuinely distinct neighbours are
    suppressed as duplicates."""
    boxes = np.stack([box(100, 100), box(115, 100)])
    assert len(attempt(drills.nms, boxes, [1.0, 0.9], 0.3)) == 1
    assert len(attempt(drills.nms, boxes, [1.0, 0.9], 0.7)) == 2


def test_assignment_is_optimal_where_greedy_is_not():
    """Greedy takes the smallest entry first and is then stuck. The exact
    solution is worth the extra code precisely when objects are close."""
    C = np.array([[1.0, 2.0], [1.1, 9.0]])
    a = attempt(drills.hungarian, C)
    assert list(a) == [1, 0]                      # total 2.0 + 1.1 = 3.1
    greedy = C[0, 0] + C[1, 1]                    # 1.0 + 9.0 = 10.0
    assert sum(C[i, j] for i, j in enumerate(a)) < greedy


def test_assignment_is_a_permutation():
    r = np.random.default_rng(1)
    C = r.uniform(0, 10, (6, 6))
    a = attempt(drills.hungarian, C)
    assert sorted(a.tolist()) == list(range(6))


# ------------------------------------------------------------------ detection

def test_the_anchor_count_is_what_the_arithmetic_says():
    n = attempt(drills.anchor_count, 512, (8, 16, 32), 3, 3)
    assert n == (64 * 64 + 32 * 32 + 16 * 16) * 9
    assert n == 48384


def test_the_imbalance_grows_with_resolution():
    small = attempt(drills.anchor_count, 256)
    big = attempt(drills.anchor_count, 2048)
    assert big / small == pytest.approx(64.0)      # area ratio, exactly


def test_the_focal_term_kills_easy_examples_and_spares_hard_ones():
    assert attempt(drills.focal_weight, 0.99) < 1e-3
    assert attempt(drills.focal_weight, 0.5) == pytest.approx(0.25)
    assert attempt(drills.focal_weight, 0.1) > 0.8


def test_a_thousand_easy_negatives_outweigh_one_hard_positive():
    """The reason the focal loss exists: not that negatives are hard, but that
    there are so many easy ones."""
    easy = 1000 * -np.log(0.99)
    hard = 1 * -np.log(0.1)
    assert easy > hard
    w = attempt(drills.focal_weight, 0.99)
    assert 1000 * w * -np.log(0.99) < hard


def test_object_area_falls_as_the_square_of_the_stride():
    assert attempt(drills.feature_pixels, 16, 0) == pytest.approx(256.0)
    assert attempt(drills.feature_pixels, 16, 4) == pytest.approx(1.0)
    assert attempt(drills.feature_pixels, 16, 5) == pytest.approx(0.25)


# --------------------------------------------------------------------- dense

def test_mean_iou_averages_over_classes_not_pixels():
    """A rare class counts as much as a common one, which is the whole reason
    the metric is defined this way."""
    true = np.zeros(1000, int); true[:10] = 1
    perfect_common = np.zeros(1000, int)           # gets the rare class entirely wrong
    assert attempt(drills.mean_iou, perfect_common, true, 2) == pytest.approx(0.99 / 2, abs=0.01)


def test_the_structure_tensor_tells_flat_from_edge_from_corner():
    n = 41
    y, x = np.mgrid[0:n, 0:n].astype(float)
    flat = np.zeros((n, n))
    edge = np.tanh((x - n / 2) / 1.5)
    corner = np.tanh((x - n / 2) / 1.5) * np.tanh((y - n / 2) / 1.5)
    lo = lambda p: np.linalg.eigvalsh(attempt(drills.structure_tensor, p)).min()
    assert lo(flat) == pytest.approx(0.0, abs=1e-12)
    assert lo(edge) == pytest.approx(0.0, abs=1e-9)
    assert lo(corner) > 1.0


def test_flow_is_not_determined_on_an_edge_and_is_on_a_corner():
    """Exact, and no network changes it: the tangential motion of a featureless
    straight edge is not present in a local window."""
    n = 41
    y, x = np.mgrid[0:n, 0:n].astype(float)
    edge = np.tanh((x - n / 2) / 1.5)
    corner = edge * np.tanh((y - n / 2) / 1.5)
    assert not attempt(drills.flow_is_determined, edge)
    assert attempt(drills.flow_is_determined, corner)


# ------------------------------------------------------------------ geometry

def K_of(f=800.0, cx=512.0, cy=384.0):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def test_projection_and_back_round_trip():
    K = K_of()
    X = np.array([[0.4, -0.2, 7.0], [-1.0, 0.5, 22.0]])
    uv = attempt(drills.project, K, np.eye(3), np.zeros(3), X)
    back = attempt(drills.unproject, K, uv, X[:, 2])
    assert np.abs(back - X).max() < 1e-9


def test_a_point_on_the_axis_lands_at_the_principal_point():
    K = K_of()
    uv = attempt(drills.project, K, np.eye(3), np.zeros(3), np.array([[0.0, 0.0, 10.0]]))
    assert uv[0, 0] == pytest.approx(512.0)
    assert uv[0, 1] == pytest.approx(384.0)


def test_image_motion_from_the_same_displacement_falls_with_depth():
    """The 1/Z, stated as a measurement: the same sideways step moves a distant
    object far less."""
    K = K_of()
    d = []
    for Z in (5.0, 50.0):
        a = attempt(drills.project, K, np.eye(3), np.zeros(3), np.array([[0.0, 0, Z]]))
        b = attempt(drills.project, K, np.eye(3), np.zeros(3), np.array([[1.0, 0, Z]]))
        d.append(abs(b[0, 0] - a[0, 0]))
    assert d[0] / d[1] == pytest.approx(10.0)


def test_resizing_without_rescaling_the_intrinsics_is_wrong():
    """One of the most common bugs in multi-camera code, and it produces
    geometry that looks nearly right."""
    K = K_of()
    X = np.array([[1.0, 0.5, 12.0]])
    uv = attempt(drills.project, K, np.eye(3), np.zeros(3), X)
    K2 = attempt(drills.scale_intrinsics, K, 0.5, 0.5)
    uv2 = attempt(drills.project, K2, np.eye(3), np.zeros(3), X)
    assert np.abs(uv2 - uv / 2).max() < 1e-9       # correct
    assert np.abs(uv - uv / 2).max() > 100         # forgetting it is not subtle


def test_distortion_vanishes_at_the_principal_point():
    K = K_of()
    c = np.array([[512.0, 384.0]])
    assert np.abs(attempt(drills.radial_distort, K, c, -0.2) - c).max() < 1e-12


def test_distortion_grows_with_radius():
    K = K_of()
    pts = np.array([[512.0 + r, 384.0] for r in (50, 200, 500)])
    d = attempt(drills.radial_distort, K, pts, -0.15)
    err = np.abs(d[:, 0] - pts[:, 0])
    assert err[0] < err[1] < err[2]
    assert err[2] > 10.0                            # tens of pixels at the edge


def test_stereo_depth_inverts_disparity():
    assert attempt(drills.stereo_depth, 800.0, 0.25, 10.0) == pytest.approx(20.0)


def test_depth_error_grows_as_the_square_of_range():
    z = np.array([10.0, 20.0, 40.0])
    e = attempt(drills.stereo_depth_error, 800.0, 0.25, z, 0.1)
    assert e[1] / e[0] == pytest.approx(4.0)
    assert e[2] / e[1] == pytest.approx(4.0)


def test_doubling_the_baseline_halves_the_error():
    a = attempt(drills.stereo_depth_error, 800.0, 0.25, 100.0, 0.1)
    b = attempt(drills.stereo_depth_error, 800.0, 0.50, 100.0, 0.1)
    assert a / b == pytest.approx(2.0)


def test_the_epipolar_constraint_holds_for_true_correspondences():
    K1 = K2 = K_of()
    R = np.eye(3)
    t = np.array([-0.25, 0.0, 0.0])                # a horizontal stereo pair
    r = np.random.default_rng(2)
    X = np.column_stack([r.uniform(-3, 3, 30), r.uniform(-2, 2, 30), r.uniform(6, 40, 30)])
    x1 = attempt(drills.project, K1, np.eye(3), np.zeros(3), X)
    x2 = attempt(drills.project, K2, R, t, X)
    F = attempt(drills.fundamental_from_poses, K1, K2, R, t)
    assert attempt(drills.epipolar_residual, F, x1, x2).max() < 1e-6


def test_a_calibration_drift_shows_up_as_an_epipolar_residual():
    K1 = K2 = K_of()
    R = np.eye(3); t = np.array([-0.25, 0.0, 0.0])
    r = np.random.default_rng(3)
    X = np.column_stack([r.uniform(-3, 3, 30), r.uniform(-2, 2, 30), r.uniform(6, 40, 30)])
    x1 = attempt(drills.project, K1, np.eye(3), np.zeros(3), X)
    x2 = attempt(drills.project, K2, R, t, X)
    bad = attempt(drills.fundamental_from_poses, K1, K2, R, t + np.array([0.0, 0.01, 0.0]))
    assert attempt(drills.epipolar_residual, bad, x1, x2).max() > 1.0


# ------------------------------------------------------------------ tracking

def test_constant_velocity_predicts_where_it_says():
    p = attempt(drills.constant_velocity, np.array([10.0, 5.0]), np.array([2.0, -1.0]), 4)
    assert np.allclose(p, [18.0, 1.0])


def test_association_on_stale_positions_swaps_identities_at_a_crossing():
    """Two objects crossing: matching to the last seen position gets both wrong
    once they have passed each other."""
    last = np.array([[0.0, 0.0], [10.0, 0.0]])
    vel = np.array([[2.0, 0.0], [-2.0, 0.0]])
    obs = last + vel * 4                            # they have swapped sides
    stale = attempt(drills.associate, last, obs, 100.0)
    assert list(stale) == [1, 0]                    # matched to the wrong ones
    pred = np.stack([attempt(drills.constant_velocity, last[i], vel[i], 4) for i in range(2)])
    moving = attempt(drills.associate, pred, obs, 100.0)
    assert list(moving) == [0, 1]                   # the motion model is right


def test_the_gate_rejects_an_implausible_match():
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[500.0, 500.0]])
    assert attempt(drills.associate, tracks, dets, gate=50.0)[0] == -1
    assert attempt(drills.associate, tracks, dets, gate=5000.0)[0] == 0
