"""
Reference solutions for the Part XXXI drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import numpy as np

import drills


# ------------------------------------------------------------------- feedback

def open_loop(target, gain_true, gain_model, steps=400, dt=0.05, tau=1.0):
    x = 0.0
    u = target / gain_model
    for _ in range(steps):
        x += dt * (-x + gain_true * u) / tau
    return float(x)


def proportional(target, gain, kp, steps=400, dt=0.05, tau=1.0):
    x = 0.0
    for _ in range(steps):
        x += dt * (-x + gain * kp * (target - x)) / tau
    return float(x)


def proportional_offset(gain, kp):
    """The steady-state fraction of the target that is missed."""
    return 1.0 / (1.0 + kp * gain)


def pi_control(target, gain, kp, ki, steps=2000, dt=0.02, tau=1.0, lag=0.15, delay=0):
    x, v, acc = 0.0, 0.0, 0.0
    buf = [0.0] * (delay + 1)
    hist = []
    for _ in range(steps):
        buf.append(x)
        meas = buf.pop(0)
        e = target - meas
        acc += e * dt
        u = kp * e + ki * acc
        v += dt * (-v + u) / lag
        x += dt * (-x + gain * v) / tau
        hist.append(x)
        if not np.isfinite(x) or abs(x) > 1e6:
            return np.array(hist), False
    return np.array(hist), True


# -------------------------------------------------------------------- optimal

def dlqr(A, B, Q, R, iters=1000):
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.asarray(R, float)
    P = Q.copy()
    for _ in range(iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K, P


def closed_loop_poles(A, B, K):
    return np.linalg.eigvals(np.asarray(A, float) - np.asarray(B, float) @ np.asarray(K, float))


def rollout_cost(A, B, Q, R, K, x0, steps=300):
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); R = np.asarray(R, float); K = np.asarray(K, float)
    x = np.asarray(x0, float).copy()
    c = 0.0
    for _ in range(steps):
        u = -K @ x
        c += float(x @ Q @ x + u @ R @ u)
        x = A @ x + B @ u
        if not np.isfinite(x).all() or np.abs(x).max() > 1e8:
            return np.inf
    return c


def integrator_chain(n, dt=0.1):
    A = np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = dt
    B = np.zeros((n, 1)); B[-1, 0] = dt
    return A, B


# ----------------------------------------------------------------- transforms

def transform(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float)
    return T


def transform_inverse(T):
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ----------------------------------------------------------------- kinematics

def two_link_fk(q, l1=1.0, l2=1.0):
    q = np.asarray(q, float)
    return np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
                     l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])])


def two_link_jacobian(q, l1=1.0, l2=1.0):
    q = np.asarray(q, float)
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                     [l1 * c1 + l2 * c12, l2 * c12]])


def damped_pinv(J, lam):
    J = np.asarray(J, float)
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(J.shape[0]))


# ------------------------------------------------------------------ estimation

def dead_reckon(v_true, bias, noise_sd, steps, dt=0.1, seed=0):
    r = np.random.default_rng(seed)
    meas = v_true + bias + noise_sd * r.standard_normal(steps)
    est = np.cumsum(meas) * dt
    truth = np.arange(1, steps + 1) * v_true * dt
    return np.abs(est - truth)


# -------------------------------------------------------------------- planning

def grid_cells(dims, res=12):
    return float(res) ** int(dims)


def rollout_divergence(lam, model_err, H, trials=400, seed=0):
    """Absolute divergence between a true and a perturbed linear system after H
    steps. Relative divergence is scale-free and cannot distinguish stability."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        xt = xm = 1.0
        for _ in range(H):
            xt = lam * xt
            xm = (lam + model_err * r.choice([-1.0, 1.0])) * xm
        out.append(abs(xm - xt))
    return float(np.mean(out))


# ------------------------------------------------------------------ validation

def rule_of_three(n_trials, confidence=0.95):
    """Upper bound on a rate given zero events."""
    return float(-np.log(1 - confidence) / n_trials)


def miles_to_bound(rate, confidence=0.95):
    return float(-np.log(1 - confidence) / rate)


def miles_to_estimate(rate, rel_precision=0.2, z=1.96):
    """Enough events to pin the rate down to a relative precision."""
    return float((z / rel_precision) ** 2 / rate)


for _name, _fn in list(globals().items()):
    if callable(_fn) and not _name.startswith("_") and hasattr(drills, _name):
        setattr(drills, _name, _fn)
