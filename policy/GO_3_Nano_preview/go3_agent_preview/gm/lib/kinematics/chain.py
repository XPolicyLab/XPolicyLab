
from __future__ import annotations

import numpy as np

from ...lib.geometry.transforms import make_T, quat_to_mat, rot_err, rpy, axis_angle, wrap

CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0, 1, 0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1, 0, 0)),
)
_STATIC = [make_T(rpy(*r), np.array(p)) for p, r, _ in CHAIN]
_AXES = [np.array(a, float) for _, _, a in CHAIN]

FINGER_BASE_X = 0.08657
FINGER_TIP_X = FINGER_BASE_X + 0.071
TCP_X = 0.130
GRIP_MIN, GRIP_MAX = -0.01, 0.044

BASES = {
    "left": (np.array([-0.3, -0.45, 0.765]), np.array([0.707, 0.0, 0.0, 0.707])),
    "right": (np.array([0.3, -0.45, 0.765]), np.array([0.707, 0.0, 0.0, 0.707])),
}
JOINT_LIMITS = np.array([[-3.1, 3.1], [-0.1, 3.4], [-0.1, 3.4], [-1.9, 1.9], [-1.9, 1.9], [-3.1, 3.1]])

def base_T(arm):
    p, q = BASES[arm]
    return make_T(quat_to_mat(q), p)

_BASE = {arm: base_T(arm) for arm in BASES}
_BASE_INV = {arm: np.linalg.inv(T) for arm, T in _BASE.items()}

def fk_chain(arm, q):
    T = _BASE[arm].copy()
    frames = []
    for i in range(6):
        T = T @ _STATIC[i] @ make_T(axis_angle(_AXES[i], q[i]))
        frames.append(T)
    return frames

def fk(arm, q):
    return fk_chain(arm, q)[-1]

def tcp_from_link6(T6, offset=TCP_X):
    return T6 @ make_T(None, np.array([offset, 0.0, 0.0]))

def link6_from_tcp(Ttcp, offset=TCP_X):
    return Ttcp @ make_T(None, np.array([-offset, 0.0, 0.0]))

def jacobian(arm, q):
    frames = fk_chain(arm, q)
    p_end = frames[-1][:3, 3]
    J = np.zeros((6, 6))
    for i, T in enumerate(frames):
        z = T[:3, :3] @ _AXES[i]
        J[:3, i] = np.cross(z, p_end - T[:3, 3])
        J[3:, i] = z
    return J, frames[-1]

def ik(arm, T_target, q_seed, iters=150, tol_p=5e-4, tol_r=5e-3, w_rot=0.35, limits=JOINT_LIMITS):
    q = np.array(q_seed, float)
    lam = 0.01
    best = (q.copy(), 1e9, 1e9)
    for _ in range(iters):
        J, T = jacobian(arm, q)
        ep = T_target[:3, 3] - T[:3, 3]
        er = rot_err(T_target[:3, :3], T[:3, :3])
        np_, nr = np.linalg.norm(ep), np.linalg.norm(er)
        if np_ + w_rot * nr < best[1] + w_rot * best[2]:
            best = (q.copy(), np_, nr)
        if np_ < tol_p and nr < tol_r:
            break
        e = np.concatenate([ep, w_rot * er])
        Jw = J.copy()
        Jw[3:] *= w_rot
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * lam * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, limits[:, 0], limits[:, 1])
    return best

def ik_multi(arm, T_target, q_seed, extra_seeds=(), **kw):
    sols = []
    for s in (q_seed, *extra_seeds):
        q, ep, er = ik(arm, T_target, s, **kw)
        sols.append((ep < 2e-3 and er < 0.02, np.linalg.norm(wrap(q - q_seed)), q, ep, er))
    good = [s for s in sols if s[0]]
    pick = min(good, key=lambda s: s[1]) if good else min(sols, key=lambda s: s[3] + 0.3 * s[4])
    return pick[2], pick[3], pick[4]

def down_rotation(yaw):
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)

def gripper_to_unit(q7):
    return float(np.clip((q7 - GRIP_MIN) / (GRIP_MAX - GRIP_MIN), 0.0, 1.0))

def unit_to_width(u):
    q7 = GRIP_MIN + u * (GRIP_MAX - GRIP_MIN)
    return 2 * max(q7, 0.0) + 0.0008

def width_to_unit(w):
    q7 = (max(float(w), 0.0) - 0.0008) / 2.0
    return float(np.clip((q7 - GRIP_MIN) / (GRIP_MAX - GRIP_MIN), 0.0, 1.0))

_T6_CAM = (
    make_T(rpy(0.0, 0.0, -3.1416), np.array([0.057, 0.0, 0.0]))
    @ make_T(rpy(0.0, 0.3491, 3.141592653589793), np.array([-0.0275, 0.0, 0.05]))
)

def wrist_camera_T(arm, q):
    from ...lib.geometry.camera import euler_xyz_deg

    return fk(arm, q) @ _T6_CAM @ make_T(euler_xyz_deg(0.0, -80.0, -90.0), np.array([0.0, 0.0, 0.001]))

LINK_BOXES = {
    "base_link": ((-0.03, -0.03, 0.0), (0.03, 0.03, 0.062)),
    "link1": ((-0.022, -0.039, 0.0), (0.037, 0.039, 0.057)),
    "link2": ((-0.292, -0.03, -0.028), (0.028, 0.036, 0.028)),
    "link3": ((-0.018, -0.038, -0.085), (0.274, 0.038, 0.018)),
    "link4": ((-0.017, -0.031, -0.086), (0.099, 0.031, 0.018)),
    "link5": ((-0.031, -0.031, 0.002), (0.032, 0.031, 0.115)),
    "link6": ((0.0, -0.087, -0.03), (0.084, 0.087, 0.03)),
    "link7": ((-0.015, -0.024, -0.03), (0.071, 0.013, 0.031)),
    "link8": ((-0.015, -0.013, -0.03), (0.071, 0.024, 0.031)),
}

def link_frames(arm, q, grip=0.044):
    frames = fk_chain(arm, q)
    T6 = frames[-1]
    g = max(0.0, min(GRIP_MAX, grip))
    out = {"base_link": _BASE[arm]}
    out.update({f"link{i + 1}": T for i, T in enumerate(frames)})
    out["link7"] = T6 @ make_T(None, np.array([0.08657, 0.024896 + g, -0.0002436]))
    out["link8"] = T6 @ make_T(None, np.array([0.08657, -0.0249 - g, -0.00024366]))
    return out

def link_box_corners(arm, q, grip=0.044, pad=0.006):
    out = {}
    for name, T in link_frames(arm, q, grip).items():
        lo, hi = np.array(LINK_BOXES[name][0]) - pad, np.array(LINK_BOXES[name][1]) + pad
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        out[name] = c @ T[:3, :3].T + T[:3, 3]
    return out

def link6_for_camera(T_world_cam):
    from ...lib.geometry.camera import euler_xyz_deg

    T6cam = _T6_CAM @ make_T(euler_xyz_deg(0.0, -80.0, -90.0), np.array([0.0, 0.0, 0.001]))
    return T_world_cam @ np.linalg.inv(T6cam)
