from __future__ import annotations

import math

import numpy as np

from ...perception import cameras as C
from ...perception import robot_mask as RM
from . import vase as V
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills.runner import Arm, run_arms, tcp_pose
from .. import Program, register

CUP_R_BOT, CUP_R_TOP, CUP_H = V.UPRIGHT_CUP
LIP_R = CUP_R_TOP - 0.002
GRASP_H = 0.042
APPROACHES = ("back", 0.09), ("back", 0.05), ("up", 0.085)
CUP_TRANSIT_CLEAR = 0.005
OPEN_G = 1.0
LIP_BACK_FRAC = 0.5
LIP_BACK = 0.008, 0.02
LIP_EDGE = 0.007
LIP_EDGE_UNTRUSTED = 0.013
LIP_BACK_STEPS = 5
V_EXIT = 0.25
BALL_R = 0.005
LIP_BACK_GAIN = 0.002
ROLL_STEP = math.radians(2.0)
ROLL_GRID = np.arange(math.radians(74.0), math.radians(100.0) + 1e-09, ROLL_STEP)
ROLL_SEL = (ROLL_GRID >= math.radians(82.0) - 1e-09) & (ROLL_GRID <= math.radians(94.0) + 1e-09)
FINGER_EXTRA = 0.004
SAG = 0.004
LIP_GAP_MIN = 0.01
RIM_CLEAR = 0.005
RIM_MARGIN = 0.012
RIM_MARGIN_MODEL = 0.007
SHOULDER_DZ = 0.005
CUP_CLEAR = 0.015
PHI_START = math.radians(72.0)
PHI_SLOW_END = math.radians(100.0)
TILT_MAX = math.radians(125.0)
TILT_FAST = math.radians(3.0)
TILT_SLOW = math.radians(0.6)
TILT_MID = math.radians(2.0)
DESCEND_STEP = 0.003
POUR_DZ = 0.001
GRIP_W = 2 * (CUP_R_BOT + (CUP_R_TOP - CUP_R_BOT) * GRASP_H / CUP_H)
HOLD_STEPS = 15
CARRY_ACC = 0.5 / 625.0
CARRY_LIN = 0.012
RELEASE_DZ = 0.002
PLACE_STEP = 0.0025
SETTLE_STEPS = 3
LIFT_ABOVE = 0.1
TRAVEL_JOINT = 0.12
MIN_OTHER_DIST = 0.04
GRASP_PITCHES = 0.0, math.radians(12.0), math.radians(25.0)
PARK_Q1 = {"left": 0.7, "right": -0.7}


def rot_z(a: float) -> np.ndarray:
    return K.rot_z(a)


def cup_rot(yaw: float, s: int, phi: float) -> np.ndarray:
    return rot_z(yaw) @ K.rot_x(s * phi)


_CUP_PTS = None


def cup_points() -> np.ndarray:
    global _CUP_PTS
    if _CUP_PTS is None:
        a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        rings = []
        for t in (0.0, 0.5, 1.0):
            r = CUP_R_BOT + (CUP_R_TOP - CUP_R_BOT) * t
            rings.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, CUP_H * t)], axis=1))
        _CUP_PTS = np.vstack(rings)
    return _CUP_PTS


GRIPPER_LINKS = "link6", "link7", "link8", "camera_base", "camera"
_GRIPPER_PTS = None


def gripper_points() -> tuple[np.ndarray, np.ndarray]:
    global _GRIPPER_PTS
    if _GRIPPER_PTS is None:
        from scipy.spatial import ConvexHull

        grip = (GRIP_W / 2 - 0.0004 + 0.01) / 0.054
        poses = RM.link_poses("left", np.zeros(6), grip)
        inv6 = np.linalg.inv(poses["link6"])
        pts, fing = [], []
        for name in GRIPPER_LINKS:
            T = inv6 @ poses[name]
            for pc in RM._pieces().get(name, []):
                v = pc @ T[:3, :3].T + T[:3, 3]
                tri = v[ConvexHull(v).simplices]
                e = [v, tri.mean(axis=1)] + [(tri[:, a] + tri[:, b]) / 2 for a, b in ((0, 1), (1, 2), (0, 2))]
                e = np.vstack(e)
                pts.append(e)
                fing.append(np.full(len(e), name in ("link7", "link8")))
        p, f = np.vstack(pts), np.concatenate(fing)
        _, idx = np.unique(np.round(p / 0.005).astype(np.int64), axis=0, return_index=True)
        p, f = p[idx], f[idx]
        p[:, 0] -= K.TCP_OFFSET
        _GRIPPER_PTS = p, f
    return _GRIPPER_PTS


def exit_dir(yaw: float, s: int) -> np.ndarray:
    return -s * rot_z(yaw)[:, 1]


def lip_back(r_open: float) -> float:
    return float(np.clip(LIP_BACK_FRAC * r_open, *LIP_BACK))


def body_points(R: np.ndarray, origin, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    cup = np.asarray(origin) + cup_points() @ R.T
    Rg = R @ K.rot_y(pitch)
    tcp = np.asarray(origin) + R @ np.array([0.0, 0.0, GRASP_H])
    gp, gf = gripper_points()
    kind = np.concatenate([np.zeros(len(cup), int), np.where(gf, 1, 2)])
    return np.vstack([cup, tcp + gp @ Rg.T]), kind


def height_map(prof, below: float = 0.0, margin: float = RIM_MARGIN):
    z_e, r = prof
    ds = np.arange(0.0, 0.25, 0.001)
    H = np.full(len(ds), -np.inf)
    for i in range(len(r)):
        if below > 0 and z_e[i + 1] > z_e[-1] - below + 1e-09:
            continue
        H = np.where(ds <= r[i] + margin, np.maximum(H, z_e[i + 1]), H)

    def hmap(d):
        return H[np.clip((np.asarray(d) / 0.001).astype(int), 0, len(ds) - 1)]

    hmap.exact = below <= 0.0
    hmap.margin = margin
    return hmap


def lift_needed(
    pts: np.ndarray, kind: np.ndarray, vase_xyz, r_top: float, clear: bool, hmap=None, sag: float = 0.0
) -> float:
    z_top = float(vase_xyz[2])
    d = np.hypot(pts[:, 0] - vase_xyz[0], pts[:, 1] - vase_xyz[1])
    margin = getattr(hmap, "margin", RIM_MARGIN)
    if clear:
        req = np.where((kind < 2) | (d < r_top + margin + 0.02), z_top + CUP_CLEAR, -np.inf)
    else:
        shoulder = (kind < 2) & (d < r_top + 0.03) & (not getattr(hmap, "exact", False))
        req = np.where(d < r_top + margin, z_top + RIM_CLEAR, np.where(shoulder, z_top - SHOULDER_DZ, -np.inf))
    if hmap is not None:
        req = np.maximum(req, hmap(d) + RIM_CLEAR)
    if sag > 0:
        req = req + np.where(kind < 2, sag, 0.0)
    req = req + np.where((kind == 1) & (d < r_top + margin), FINGER_EXTRA, 0.0)
    return max(0.0, float(np.max(req - pts[:, 2])))


def pose_at(
    vase_xyz,
    r_top: float,
    yaw: float,
    s: int,
    phi: float,
    pitch: float,
    clear: bool,
    r_open: float | None = None,
    hmap=None,
    lb: float | None = None,
) -> np.ndarray:
    R = cup_rot(yaw, s, phi)
    p_lip = np.array([0.0, -s * LIP_R, CUP_H])
    r_open = 0.75 * r_top if r_open is None else r_open
    lb = lip_back(r_open) if lb is None else lb
    lip_xy = np.asarray(vase_xyz[:2], dtype=float) - lb * exit_dir(yaw, s)[:2]
    lip = np.array([lip_xy[0], lip_xy[1], vase_xyz[2] + LIP_GAP_MIN])
    origin = lip - R @ p_lip
    pts, kind = body_points(R, origin, pitch)
    origin[2] += lift_needed(pts, kind, vase_xyz, r_top, clear, hmap, SAG * abs(math.sin(phi)))
    return tcp_pose(R @ K.rot_y(pitch), origin + R @ np.array([0.0, 0.0, GRASP_H]))


def _range(a: float, b: float, step: float) -> list[float]:
    n = max(1, int(math.ceil((b - a) / step - 1e-09)))
    return [a + (b - a) * i / n for i in range(1, n + 1)]


def choose_lip_back(
    vase_xyz, r_top: float, yaw: float, s: int, pitch: float, r_open: float | None = None, hmap=None
) -> float:
    r_open = 0.75 * r_top if r_open is None else r_open
    lo = lip_back(r_open)
    hi = max(lo, r_open - (LIP_EDGE if getattr(hmap, "exact", False) else LIP_EDGE_UNTRUSTED))
    best, best_h = lo, None
    n_per = ROLL_STEP / TILT_SLOW
    p_rel = np.array([0.0, -s * LIP_R, CUP_H - GRASP_H])
    offs = np.array([(cup_rot(yaw, s, phi) @ p_rel)[2] for phi in ROLL_GRID])
    for lb in np.linspace(lo, hi, LIP_BACK_STEPS):
        z = np.array([pose_at(vase_xyz, r_top, yaw, s, phi, pitch, False, r_open, hmap, lb)[2, 3] for phi in ROLL_GRID])
        k = np.arange(len(z))
        env = np.max(z[None, :] - POUR_DZ * n_per * np.abs(k[:, None] - k[None, :]), axis=1)
        h = float(np.mean(env[ROLL_SEL] + offs[ROLL_SEL])) - float(vase_xyz[2])
        if best_h is None or h < best_h - LIP_BACK_GAIN:
            best, best_h = float(lb), h
    drop = max(0.0, best_h)
    return float(np.clip(max(best, BALL_R + V_EXIT * math.sqrt(2.0 * drop / 9.81)), lo, hi))


def pour_poses(
    vase_xyz,
    r_top: float,
    yaw: float,
    s: int,
    pitch: float,
    r_open: float | None = None,
    hmap=None,
    lb: float | None = None,
):
    if lb is None:
        lb = choose_lip_back(vase_xyz, r_top, yaw, s, pitch, r_open, hmap)
    poses = [pose_at(vase_xyz, r_top, yaw, s, 0.0, pitch, True, r_open, hmap, lb)]
    for phi in _range(0.0, PHI_START, TILT_FAST):
        poses.append(pose_at(vase_xyz, r_top, yaw, s, phi, pitch, True, r_open, hmap, lb))
    tail = [
        pose_at(vase_xyz, r_top, yaw, s, phi, pitch, False, r_open, hmap, lb)
        for phi in [PHI_START] + _range(PHI_START, PHI_SLOW_END, TILT_SLOW) + _range(PHI_SLOW_END, TILT_MAX, TILT_MID)
    ]
    smooth_heights(tail, 0)
    lo = tail[0]
    hi = poses[-1]
    for z in _range(hi[2, 3], lo[2, 3], -DESCEND_STEP) if hi[2, 3] > lo[2, 3] + 0.0001 else []:
        t = hi.copy()
        t[2, 3] = z
        poses.append(t)
    i_pour = len(poses)
    poses.extend(tail[1:])
    return poses, i_pour


def smooth_heights(poses, i0: int, dz: float = POUR_DZ) -> None:
    z = np.array([T[2, 3] for T in poses[i0:]])
    if len(z) < 2:
        return
    k = np.arange(len(z))
    env = np.max(z[None, :] - dz * np.abs(k[:, None] - k[None, :]), axis=1)
    for T, h in zip(poses[i0:], env):
        T[2, 3] = h


def chain_ik(arm: str, poses, q0) -> list[np.ndarray] | None:
    qs, q = [], np.asarray(q0, dtype=float)
    for T in poses:
        r = K.solve(arm, T, q, K.TCP_OFFSET)
        if not r.ok:
            r = K.solve_near(arm, T, q, K.TCP_OFFSET, n_random=2)
            if not r.ok or np.max(np.abs(r.q - q)) > 0.5:
                return None
        q = r.q
        qs.append(q)
    return qs


def _dist_other(arm: str, q, q_other) -> float:
    return CO.arm_distance(q, q_other) if arm == "left" else CO.arm_distance(q_other, q)


VASE_CLEAR = 0.012
VASE_CLEAR_POUR = 0.0
POUR_BODY_BELOW = 0.03
SAFE_DZ = 0.06
N_FINGER_SPHERES = 2 * len(CO._FINGER_X)
VASE_HANDLES = {4: (0.109, 0.129, 0.0528)}
CUP_PROFILE = np.array([K.TABLE_Z, K.TABLE_Z + CUP_H + 0.003]), np.array([CUP_R_TOP])


def vase_trusted(vase) -> bool:
    return vase.extra.get("model") is not None and vase.extra.get("model_iou", 0.0) >= V.MODEL_MIN_IOU


def vase_profile(vase) -> tuple[np.ndarray, np.ndarray]:
    m = vase.extra.get("model") if vase_trusted(vase) else None
    models = [m] if m is not None else sorted(V.VASE_MODELS)
    h_max = max(V.VASE_MODELS[k][0] for k in models)
    z = np.linspace(0.0, h_max, 30 + 1)
    r = np.zeros(30)
    for k in models:
        h, _, radii = V.VASE_MODELS[k]
        for i in range(30):
            lo, hi = z[i], z[i + 1]
            if lo >= h:
                continue
            j0, j1 = int(lo / h * len(radii)), min(len(radii) - 1, int(min(hi, h - 1e-09) / h * len(radii)))
            r[i] = max(r[i], max(radii[j0 : j1 + 1]))
            hd = VASE_HANDLES.get(k)
            if hd is not None and hi > hd[0] and (lo < hd[1]):
                r[i] = max(r[i], hd[2])
    return K.TABLE_Z + z, r


def vase_clearance(spheres: np.ndarray, vase_xy, prof, below: float = 0.0) -> float:
    z_e, r = prof
    d = np.hypot(spheres[:, 0] - vase_xy[0], spheres[:, 1] - vase_xy[1])
    lo, hi = spheres[:, 2] - spheres[:, 3], spheres[:, 2] + spheres[:, 3]
    best = np.inf
    for i in range(len(r)):
        if z_e[i] > z_e[-1] - below:
            break
        sel = (hi >= z_e[i]) & (lo <= z_e[i + 1])
        if sel.any():
            best = min(best, float(np.min(d[sel] - r[i] - spheres[sel, 3])))
    return best


def path_clear(
    arm: str, qs, vase_xy, prof, margin: float = VASE_CLEAR, fingers: bool = True, grip: float = 1.0, below: float = 0.0
) -> float:
    best = np.inf
    for q in qs:
        s = CO.arm_spheres(arm, q, grip)
        if not fingers:
            s = s[:-N_FINGER_SPHERES]
        best = min(best, vase_clearance(s, vase_xy, prof, below))
        if best < margin:
            break
    return best


def joint_samples(q0, q1, n: int = 10) -> list[np.ndarray]:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return [q0 + (q1 - q0) * t for t in np.linspace(0.0, 1.0, n + 1)[1:]]


def plan_for(
    arm: str,
    cup_xy,
    vase_xyz,
    r_top: float,
    q_home,
    q_other_home,
    r_open: float | None = None,
    pitches=GRASP_PITCHES,
    prof=None,
    hmap=None,
    approaches=APPROACHES[:1],
):
    base = np.array(K.BASE_POS[arm][:2])
    best = None
    vxy = np.asarray(vase_xyz[:2], dtype=float)
    yaw_c = math.atan2(cup_xy[1] - base[1], cup_xy[0] - base[0])
    yaw_v = math.atan2(vase_xyz[1] - base[1], vase_xyz[0] - base[0])
    p_grasp = np.array([cup_xy[0], cup_xy[1], K.TABLE_Z + GRASP_H])
    for pitch in pitches:
        for dg in (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9):
            yg = yaw_c + dg
            Rc = rot_z(yg)
            Rg = Rc @ K.rot_y(pitch)
            a = Rg[:, 0]
            Tg = tcp_pose(Rg, p_grasp)
            rpre, Tpre, app, grasp_ok = None, None, None, None
            for kind, dist in approaches:
                off = -dist * a if kind == "back" else np.array([0.0, 0.0, dist])
                T = tcp_pose(Rg, p_grasp + off)
                r = K.solve_near(arm, T, q_home, K.TCP_OFFSET, n_random=3)
                if not r.ok:
                    if grasp_ok is None:
                        grasp_ok = K.solve_near(arm, Tg, q_home, K.TCP_OFFSET, n_random=3).ok
                    if not grasp_ok:
                        break
                    continue
                if (kind, dist) != APPROACHES[0] and path_clear(
                    arm, joint_samples(q_home, r.q, 12), cup_xy, CUP_PROFILE, CUP_TRANSIT_CLEAR
                ) < CUP_TRANSIT_CLEAR:
                    continue
                rpre, Tpre, app = r, T, f"{kind}{round(dist * 100)}"
                break
            if rpre is None:
                continue
            qg = chain_ik(arm, [Tg], rpre.q)
            if qg is None:
                continue
            Tlift = tcp_pose(Rg, [p_grasp[0], p_grasp[1], vase_xyz[2] + LIFT_ABOVE])
            ql = chain_ik(arm, [Tlift], qg[-1])
            if ql is None:
                continue
            if min(_dist_other(arm, q, q_other_home) for q in (rpre.q, qg[-1], ql[-1])) < MIN_OTHER_DIST:
                continue
            q_high = None
            if prof is not None:
                near = joint_samples(rpre.q, qg[-1], 4) + joint_samples(qg[-1], ql[-1], 4) + [rpre.q]
                if path_clear(arm, near, vxy, prof) < VASE_CLEAR:
                    continue
                if path_clear(arm, joint_samples(q_home, rpre.q, 12), vxy, prof) < VASE_CLEAR:
                    Thigh = Tpre.copy()
                    Thigh[2, 3] = vase_xyz[2] + SAFE_DZ
                    rh = K.solve_near(arm, Thigh, rpre.q, K.TCP_OFFSET, n_random=2)
                    if (
                        not rh.ok
                        or path_clear(arm, joint_samples(q_home, rh.q, 12) + joint_samples(rh.q, rpre.q, 6), vxy, prof)
                        < VASE_CLEAR
                    ):
                        continue
                    q_high = rh.q
            for dp in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4):
                found = False
                for s in (1, -1):
                    yp = yaw_v + dp
                    lb = choose_lip_back(vase_xyz, r_top, yp, s, pitch, r_open, hmap)
                    poses, i_pour = pour_poses(vase_xyz, r_top, yp, s, pitch, r_open, hmap, lb)
                    T0 = poses[0]
                    T_above = T0.copy()
                    T_above[2, 3] = max(T0[2, 3], vase_xyz[2] + LIFT_ABOVE)
                    r0 = K.solve_near(arm, T_above, ql[-1], K.TCP_OFFSET, n_random=2)
                    if not r0.ok:
                        continue
                    if not all(
                        K.solve_near(arm, poses[i], r0.q, K.TCP_OFFSET, n_random=1).ok for i in (i_pour, len(poses) - 1)
                    ):
                        continue
                    qs = chain_ik(arm, poses, r0.q)
                    if qs is None:
                        continue
                    dmin = min(_dist_other(arm, q, q_other_home) for q in qs[::4] + [r0.q])
                    if dmin < MIN_OTHER_DIST:
                        continue
                    if prof is not None:
                        c_carry = path_clear(arm, joint_samples(ql[-1], r0.q, 8), vxy, prof)
                        if c_carry < VASE_CLEAR:
                            continue
                        c_pour = path_clear(
                            arm,
                            qs[::3] + [qs[-1]],
                            vxy,
                            prof,
                            VASE_CLEAR_POUR,
                            fingers=False,
                            grip=0.0,
                            below=POUR_BODY_BELOW,
                        )
                        if c_pour < VASE_CLEAR_POUR:
                            continue
                    found = True
                    cost = (
                        float(np.sum(np.abs(rpre.q - q_home)))
                        + float(np.sum(np.abs(r0.q - ql[-1])))
                        + 0.3 * abs(qs[-1][5] - qs[0][5])
                        + 0.5 * abs(dg)
                        + 0.2 * abs(dp)
                        + 2.0 * pitch
                        + (1.0 if q_high is not None else 0.0)
                    )
                    if best is None or cost < best["cost"]:
                        best = dict(
                            arm=arm,
                            cost=cost,
                            yg=yg,
                            yp=yp,
                            s=s,
                            pitch=pitch,
                            Tpre=Tpre,
                            Tg=Tg,
                            app=app,
                            Tlift=Tlift,
                            T_above=T_above,
                            q_pre=rpre.q,
                            q_above=r0.q,
                            q_pour=qs,
                            i_pour=i_pour,
                            T0=T0,
                            dmin=dmin,
                            q_high=q_high,
                            lb=lb,
                        )
                if found:
                    break
            if best is not None:
                break
        if best is not None:
            break
    return best


def pour_task(arm: Arm, plan: dict, cup_xy, vase_xyz, memo: dict):
    if plan.get("q_high") is not None:
        yield from arm.joints_to(plan["q_high"], grip=OPEN_G, joint_step=TRAVEL_JOINT)
        yield from arm.line_to(plan["Tpre"], lin_step=0.014)
    else:
        yield from arm.joints_to(plan["q_pre"], grip=OPEN_G, joint_step=TRAVEL_JOINT)
    yield from arm.line_to(plan["Tg"], lin_step=0.012)
    yield from arm.set_grip(0.0, 10)
    yield from arm.line_to(plan["Tlift"], lin_step=0.014, fallback_joint=False)
    yield from arm.line_to(plan["T_above"], lin_step=carry_step(arm, plan["T_above"]), ang_step=math.radians(5))
    yield from arm.line_to(plan["T0"], lin_step=0.012)
    qs = [np.asarray(q) for q in plan["q_pour"]]
    yield from arm.follow(motion.Plan(q=qs[1:]))
    yield from arm.hold(HOLD_STEPS)
    n_mid = len(_range(PHI_SLOW_END, TILT_MAX, TILT_MID))
    n_slow = len(_range(PHI_START, PHI_SLOW_END, TILT_SLOW))
    back = qs[::-1]
    back = back[:n_mid] + back[n_mid : n_mid + n_slow : 3] + back[n_mid + n_slow :] + [qs[0]]
    yield from arm.follow(motion.Plan(q=back))
    yield from arm.line_to(plan["T_above"], lin_step=0.014, fallback_joint=False)
    Rg = plan["Tg"][:3, :3]
    p_place = np.array([cup_xy[0], cup_xy[1], K.TABLE_Z + GRASP_H + RELEASE_DZ])
    T_hi = tcp_pose(Rg, [p_place[0], p_place[1], vase_xyz[2] + LIFT_ABOVE])
    yield from arm.line_to(T_hi, lin_step=carry_step(arm, T_hi), ang_step=math.radians(5))
    yield from arm.line_to(tcp_pose(Rg, p_place + [0.0, 0.0, 0.03]), lin_step=0.014)
    yield from arm.line_to(tcp_pose(Rg, p_place), lin_step=PLACE_STEP)
    yield from arm.hold(SETTLE_STEPS)
    yield from arm.set_grip(OPEN_G, 6)
    yield from arm.hold(SETTLE_STEPS)
    yield from arm.line_to(tcp_pose(Rg, p_place + (plan["Tpre"][:3, 3] - plan["Tg"][:3, 3])), lin_step=0.01)
    memo["pour_done"] = True
    prof, vxy = plan.get("prof"), np.asarray(vase_xyz[:2], dtype=float)
    if prof is not None and path_clear(arm.name, joint_samples(arm.q, arm.q_home, 12), vxy, prof) < VASE_CLEAR:
        T_up = arm.T.copy()
        p = K.point_of(T_up, K.TCP_OFFSET)
        yield from arm.line_to(tcp_pose(T_up[:3, :3], [p[0], p[1], vase_xyz[2] + SAFE_DZ]), lin_step=0.014)
    yield from arm.joints_to(arm.q_home, grip=OPEN_G, joint_step=TRAVEL_JOINT)
    memo["done"] = True


def carry_step(arm: Arm, T_goal: np.ndarray) -> float:
    d = float(np.linalg.norm(T_goal[:3, 3] - arm.tcp()))
    return float(np.clip(math.sqrt(CARRY_ACC * max(d, 0.001) / 2.0), 0.003, CARRY_LIN))


def park_q(arm: str, q_home) -> np.ndarray:
    q = np.asarray(q_home, dtype=float).copy()
    q[0] += PARK_Q1[arm]
    return q


def idle_task(arm: Arm, memo: dict):
    yield from arm.joints_to(park_q(arm.name, arm.q_home), joint_step=TRAVEL_JOINT)
    yield from arm.wait_until(lambda: memo.get("pour_done", False), max_steps=2000)
    yield from arm.joints_to(arm.q_home, joint_step=TRAVEL_JOINT)


@register
class PourBalls(Program):
    name = "pour_balls"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "pour" in s and "ball" in s and ("vase" in s):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.first
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        img = o.images.get("cam_head")
        cam = C.head()
        robot = RM.mask(cam, arms["left"].q, arms["right"].q)
        cup, vase = V.detect(img, cam, robot)
        if cup is None or vase is None:
            return
        vase_xyz = np.array([vase.x, vase.y, vase.z_top])
        prof = vase_profile(vase)
        hmap = (
            height_map(prof, margin=RIM_MARGIN_MODEL) if vase_trusted(vase) else height_map(prof, below=POUR_BODY_BELOW)
        )
        plans = []
        lv, pt, a0, a1 = GRASP_PITCHES[:1], GRASP_PITCHES[1:], APPROACHES[:1], APPROACHES[1:]
        for pitches, apps, pr, hm in (
            (lv, a0, prof, hmap),
            (lv, a1, prof, hmap),
            (pt, a0, prof, hmap),
            (pt, a1, prof, hmap),
            (GRASP_PITCHES, APPROACHES, None, None),
        ):
            for a in K.ARMS:
                other = "right" if a == "left" else "left"
                p = plan_for(
                    a,
                    cup.xy,
                    vase_xyz,
                    vase.r_top,
                    arms[a].q_home,
                    park_q(other, arms[other].q_home),
                    r_open=vase.r_open,
                    pitches=pitches,
                    prof=pr,
                    hmap=hm,
                    approaches=apps,
                )
                if p is not None:
                    plans.append(p)
            if plans:
                break
        if not plans:
            return
        plan = min(plans, key=lambda p: p["cost"])
        plan["prof"] = prof
        memo = {}
        other = "right" if plan["arm"] == "left" else "left"
        yield from run_arms(
            arms,
            {
                plan["arm"]: pour_task(arms[plan["arm"]], plan, cup.xy, vase_xyz, memo),
                other: idle_task(arms[other], memo),
            },
        )
