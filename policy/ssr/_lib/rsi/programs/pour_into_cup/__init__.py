from __future__ import annotations

import math

import numpy as np

from ...perception import cameras as C
from . import scene as P
from ...perception import robot_mask as RM
from ...planning import kinematics as K
from ...planning import motion
from ...skills.runner import Arm, run_arms
from .. import Program, register

TABLE_Z = K.TABLE_Z
GRASP_H = 0.08
AX_SHIFT = 0.004
PRE_BACK = 0.07
OPEN_G = 1.0
LIFT_Z = GRASP_H + 0.17
PAD_HALF = 0.0305
PHI_1 = math.radians(85.0)
PHI_MAX = math.radians(145.0)
TILT1_STEP = math.radians(2.5)
TILT2_STEP = math.radians(2.0)
PHI_ONSET = math.radians(120.0)
TILT_ONSET_STEP = math.radians(1.0)
HOLD = 40
RETRY_HOLD = 30
BACK_STEP = 2
PHI_SETTLE = math.radians(55.0)
SETTLE = 12
POUR_CUP_MARGIN = 0.01
EARLY_CLEAR = 0.02
EARLY_MARGIN = 0.015
EARLY_PHI0 = math.radians(45.0)
EARLY_PHI1 = math.radians(75.0)
POUR_RIM_CLEAR = 0.012
JET_V = 0.575
JET_REF_DEPTH = 0.03
JET_PHI0 = math.radians(95.0)
LIP_MARGIN = 0.008
G_ACC = 9.81
POUR_MOUTH_MIN = 0.05
TRAVEL_JOINT = 0.1
LIFT_LIN = 0.013
CARRY_JOINT = 0.04
CARRY_LIN = 0.015
CARRY_ANG = math.radians(4.0)
START_JOINT = 0.03
GRASP_YAWS = 0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.2, -1.2
GRASP_PITCHES = 0.0, math.radians(10.0)
NEAR_PITCHES = 0.0, math.radians(10.0), math.radians(20.0)
NEAR_STEEP_PITCHES = (math.radians(30.0),)
NEAR_LIMIT_MARGIN = 0.15
VIA_ACTIONS = 32.0
CARRY_ACTIONS_PER_DEG = 0.6


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def choose_grasp(bxy, q: dict, cup):
    near, far = ("left", "right") if bxy[0] < 0 else ("right", "left")
    obst = cup_cylinders(cup)

    def cost(g, p):
        turn = abs(_wrap_deg(math.degrees(g["yaw"] - p["yaw"])))
        return (VIA_ACTIONS if g["via"] is not None else 0.0) + CARRY_ACTIONS_PER_DEG * turn

    cands = []
    for pitch in NEAR_PITCHES + NEAR_STEEP_PITCHES:
        steep = pitch in NEAR_STEEP_PITCHES
        if cands and steep:
            break
        g = plan_grasp(near, bxy, q[near], obst, pitches=(pitch,), via_mode="direct" if steep else "any")
        p = plan_pour(near, g, cup, g["q_lift"]) if g is not None else None
        if p is None:
            continue
        if not cands and g["via"] is None:
            return near, g, p
        cands.append((cost(g, p), pitch, g, p))
        if g["via"] is not None:
            g2 = plan_grasp(near, bxy, q[near], obst, pitches=(pitch,), via_mode="direct")
            p2 = plan_pour(near, g2, cup, g2["q_lift"]) if g2 is not None else None
            if p2 is not None:
                cands.append((cost(g2, p2), pitch, g2, p2))
    if cands:
        c = min(cands, key=lambda c: (c[0], c[1]))
        return near, c[2], c[3]
    for pitch in NEAR_PITCHES:
        g = plan_grasp(near, bxy, q[near], obst, pitches=(pitch,), via_mode="up", up_first=True)
        p = plan_pour(near, g, cup, g["q_lift"]) if g is not None else None
        if p is not None:
            cands.append((cost(g, p), pitch, g, p))
    if cands:
        c = min(cands, key=lambda c: (c[0], c[1]))
        return near, c[2], c[3]
    for pitch in GRASP_PITCHES:
        g = plan_grasp(far, bxy, q[far], obst, pitches=(pitch,))
        if g is None:
            continue
        p = plan_pour(far, g, cup, g["q_lift"])
        if p is not None:
            return far, g, p


POUR_DIRS = tuple(math.radians(a) for a in (90.0, 60.0, 120.0, 30.0, 150.0, 0.0, 180.0))
CUP_CLEAR = 0.008
TRAVEL_CLEAR = 0.0
VIA_CLEAR = 0.005
VIA_ZS = 0.26, 0.3
VIA_LIN = 0.03
BOTTLE_BODY_TOP = 0.127
BOTTLE_BODY_R = 0.0425
BOTTLE_NECK_R = 0.0165
UPRIGHT_WAIT = 12
MEASURE_PHI = math.radians(88.0)
MEASURE_HOLD = 4
MEASURE_MIN_GAIN = 0.01
FIX_JOINT = 0.02
PARTIAL_TURN = 0.67, 0.33, 0.0
GRASP_TOL = 0.004
GRASP_WAIT = 15
GRASP_REPLAN = 0.01


def bottle_rot(yaw: float, s: int, phi: float) -> np.ndarray:
    return K.rot_z(yaw) @ K.rot_x(s * phi)


def grip_in_bottle(pitch: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = K.rot_y(pitch)
    T[:3, 3] = [AX_SHIFT, 0.0, GRASP_H]
    return T


def contact_in_tcp(grasp_h: float = GRASP_H) -> np.ndarray:
    zs = np.linspace(grasp_h - PAD_HALF, grasp_h + PAD_HALF, 61)
    prof = np.asarray(P.BOTTLE_PROFILE, float)
    z = float(zs[int(np.argmax(np.interp(zs, prof[:, 0], prof[:, 1])))])
    return np.array([-AX_SHIFT, 0.0, z - grasp_h])


def pose(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


_GRIP_PTS = None


def gripper_points(g: float = OPEN_G, dense: bool = False) -> np.ndarray:
    global _GRIP_PTS
    if dense:
        if g not in _GRIP_PTS_DENSE:
            _GRIP_PTS_DENSE[g] = _gripper_points(g, 0.004)
        return _GRIP_PTS_DENSE[g]
    if g != OPEN_G:
        if g not in _GRIP_PTS_G:
            _GRIP_PTS_G[g] = _gripper_points(g)
        return _GRIP_PTS_G[g]
    if _GRIP_PTS is None:
        _GRIP_PTS = _gripper_points(OPEN_G)
    return _GRIP_PTS


_GRIP_PTS_G: dict = {}
_GRIP_PTS_DENSE: dict = {}


def _gripper_points(g: float, spacing: float | None = None) -> np.ndarray:
    from scipy.spatial import ConvexHull

    poses = RM.link_poses("left", np.zeros(6), g)
    inv6 = np.linalg.inv(poses["link6"])
    pts = []
    for name in ("link5", "link6", "link7", "link8", "camera_base", "camera"):
        T = inv6 @ poses[name]
        for pc in RM._pieces().get(name, []):
            v = pc @ T[:3, :3].T + T[:3, 3]
            tri = v[ConvexHull(v).simplices]
            pts += [v, tri.mean(axis=1)]
            if spacing is not None:
                for a, b, c in tri:
                    n = int(
                        math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing)
                    )
                    if n < 2:
                        continue
                    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
                    m = i + j <= n
                    wi, wj = i[m][:, None] / n, j[m][:, None] / n
                    pts.append(a + wi * (b - a) + wj * (c - a))
    p = np.vstack(pts)
    p[:, 0] -= K.TCP_OFFSET
    return p


def clearance(T_tcp: np.ndarray, cylinders, g: float = OPEN_G, dense: bool = False) -> float:
    p = T_tcp[:3, 3] + gripper_points(g, dense) @ T_tcp[:3, :3].T
    best = 1.0
    for xy, r, ztop in cylinders:
        d_xy = np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) - r
        d_z = p[:, 2] - ztop
        d = np.where(d_xy < 0, d_z, np.where(d_z > 0, np.hypot(d_xy, d_z), d_xy))
        best = min(best, float(d.min()))
    return best


def chain_ik(arm: str, poses, q0, partial: bool = False) -> list[np.ndarray] | None:
    qs, q = [], np.asarray(q0, dtype=float)
    for T in poses:
        r = K.solve(arm, T, q, K.TCP_OFFSET)
        if not r.ok:
            r = K.solve_near(arm, T, q, K.TCP_OFFSET, n_random=2)
            if not r.ok or np.max(np.abs(r.q - q)) > 0.5:
                return qs if partial else None
        q = r.q
        qs.append(q)
    return qs


def _range(a0: float, a1: float, step: float) -> list[float]:
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    return [a0 + (a1 - a0) * (i + 1) / n for i in range(n)]


def cup_cylinders(cup, margin: float = 0.0, z_extra: float = 0.0):
    m = cup.model
    top = TABLE_Z + m.height + z_extra
    cyl = [(cup.xy, m.r_out + margin, top)]
    if m.handle is not None:
        hx0, hx1, hy, _, hz1 = m.handle
        c = cup.xy + np.array([(hx0 + hx1) / 2, 0.0])
        cyl.append((c, math.hypot((hx1 - hx0) / 2, hy) + margin, TABLE_Z + hz1 + z_extra))
    return cyl


def bottle_cylinders(bxy):
    c = np.asarray(bxy, float)
    return [(c, BOTTLE_BODY_R, TABLE_Z + BOTTLE_BODY_TOP), (c, BOTTLE_NECK_R, TABLE_Z + P.BOTTLE_H)]


def path_clearance(arm: str, qs, cylinders) -> float:
    best = 1.0
    for q in qs:
        T = K.fk(arm, q)
        T[:3, 3] = K.point_of(T, K.TCP_OFFSET)
        best = min(best, clearance(T, cylinders))
    return best


def plan_travel(arm: str, q_ref, Tpre: np.ndarray, q_pre, cylinders, rej: dict, up_first: bool = False):
    if path_clearance(arm, motion.joint_move(q_ref, q_pre, joint_step=TRAVEL_JOINT).q, cylinders) >= TRAVEL_CLEAR:
        return None
    for z in VIA_ZS:
        T_via = Tpre.copy()
        T_via[2, 3] = max(Tpre[2, 3], TABLE_Z + z)
        r = K.solve_near(arm, T_via, q_ref, K.TCP_OFFSET, extra_seeds=[q_pre], n_random=2)
        if not r.ok:
            rej["via_ik"] = rej.get("via_ik", 0) + 1
            continue
        down = chain_ik(arm, [T_via + (Tpre - T_via) * t for t in np.linspace(0.1, 1.0, 10)], r.q)
        if down is None or np.max(np.abs(down[-1] - q_pre)) > 0.3:
            rej["via_chain"] = rej.get("via_chain", 0) + 1
            continue
        path = list(motion.joint_move(q_ref, r.q, joint_step=TRAVEL_JOINT).q) + down
        if path_clearance(arm, path, cylinders) < VIA_CLEAR:
            rej["travel"] = rej.get("travel", 0) + 1
            continue
        return T_via, r.q
    if up_first:
        up = travel_up_first(arm, q_ref, Tpre, q_pre, cylinders, rej)
        if up is not None:
            return up
    return False


def travel_up_first(arm: str, q_ref, Tpre: np.ndarray, q_pre, cylinders, rej: dict):
    T0 = K.fk(arm, q_ref)
    T0[:3, 3] = K.point_of(T0, K.TCP_OFFSET)
    for z in VIA_ZS:
        T_up = T0.copy()
        T_up[2, 3] = max(T0[2, 3], TABLE_Z + z)
        up = chain_ik(arm, [T0 + (T_up - T0) * t for t in np.linspace(0.1, 1.0, 8)], q_ref)
        if up is None or path_clearance(arm, up, cylinders) < TRAVEL_CLEAR:
            rej["up_first"] = rej.get("up_first", 0) + 1
            continue
        T_via = Tpre.copy()
        T_via[2, 3] = max(Tpre[2, 3], TABLE_Z + z)
        r = K.solve_near(arm, T_via, up[-1], K.TCP_OFFSET, extra_seeds=[q_pre], n_random=2)
        if not r.ok:
            rej["up_via_ik"] = rej.get("up_via_ik", 0) + 1
            continue
        down = chain_ik(arm, [T_via + (Tpre - T_via) * t for t in np.linspace(0.1, 1.0, 10)], r.q)
        if down is None or np.max(np.abs(down[-1] - q_pre)) > 0.3:
            rej["up_via_chain"] = rej.get("up_via_chain", 0) + 1
            continue
        path = list(motion.joint_move(up[-1], r.q, joint_step=TRAVEL_JOINT).q) + down
        if path_clearance(arm, path, cylinders) < VIA_CLEAR:
            rej["up_travel"] = rej.get("up_travel", 0) + 1
            continue
        return T_via, r.q, T_up


def plan_grasp(arm: str, bxy, q_ref, obstacles, pitches=GRASP_PITCHES, via_mode: str = "any", up_first: bool = False):
    base = np.array(K.BASE_POS[arm][:2])
    yaw0 = math.atan2(bxy[1] - base[1], bxy[0] - base[0])
    rej = {}
    for pitch in pitches:
        for dy in GRASP_YAWS:
            yaw = yaw0 + (dy if arm == "right" else -dy)
            Tb = pose(K.rot_z(yaw), [bxy[0], bxy[1], TABLE_Z])
            Tg = Tb @ grip_in_bottle(pitch)
            a = Tg[:3, 0]
            Tpre = Tg.copy()
            Tpre[:3, 3] -= PRE_BACK * a
            Tl = Tg.copy()
            Tl[2, 3] = TABLE_Z + LIFT_Z
            cl = min(clearance(Tpre + (Tg - Tpre) * t, obstacles) for t in (0.0, 0.33, 0.66, 1.0))
            if cl < CUP_CLEAR:
                rej["clear"] = rej.get("clear", 0) + 1
                continue
            r = K.solve_near(arm, Tpre, q_ref, K.TCP_OFFSET, n_random=3)
            if not r.ok:
                rej["pre_ik"] = rej.get("pre_ik", 0) + 1
                continue
            qs = chain_ik(arm, [Tpre + (Tg - Tpre) * t for t in (0.5, 1.0)] + [Tl], r.q)
            if qs is None:
                rej["chain"] = rej.get("chain", 0) + 1
                continue
            if pitch > 0.0:
                lim = K.JOINT_LIMITS
                margin = min(float(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q))) for q in [r.q] + qs)
                if margin < NEAR_LIMIT_MARGIN:
                    rej["margin"] = rej.get("margin", 0) + 1
                    continue
            via = plan_travel(arm, q_ref, Tpre, r.q, bottle_cylinders(bxy) + list(obstacles), rej, up_first=up_first)
            if via is False:
                continue
            if (
                via_mode == "direct"
                and via is not None
                or (via_mode == "via" and via is None)
                or (via_mode == "up" and (via is None or len(via) != 3))
            ):
                rej["mode"] = rej.get("mode", 0) + 1
                continue
            plan = dict(
                arm=arm,
                yaw=yaw,
                pitch=pitch,
                Tpre=Tpre,
                Tg=Tg,
                Tl=Tl,
                q_pre=r.q,
                q_grasp=qs[1],
                q_lift=qs[-1],
                xy=np.asarray(bxy, float),
                clear=cl,
                via=via,
            )
            return plan


_BOTTLE_PTS = None


def bottle_points() -> np.ndarray:
    global _BOTTLE_PTS
    if _BOTTLE_PTS is None:
        a = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
        _BOTTLE_PTS = np.vstack(
            [np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], axis=1) for z, r in P.BOTTLE_PROFILE]
        )
    return _BOTTLE_PTS


def held_points(pitch: float, T: np.ndarray | None = None) -> np.ndarray:
    if T is None:
        T = grip_in_bottle(pitch)
    gp = T[:3, 3] + gripper_points() @ T[:3, :3].T
    return np.vstack([bottle_points(), gp])


def mouth_lift(pts_rel: np.ndarray, mxy, obstacles, floor: float) -> float:
    need = floor - float(pts_rel[:, 2].min())
    x = pts_rel[:, 0] + mxy[0]
    y = pts_rel[:, 1] + mxy[1]
    for c, r, ztop in obstacles:
        inside = np.hypot(x - c[0], y - c[1]) < r
        if np.any(inside):
            need = max(need, ztop - float(pts_rel[inside, 2].min()))
    return need


def pour_schedule() -> list[float]:
    return (
        [0.0]
        + _range(0.0, PHI_1, TILT1_STEP)
        + _range(PHI_1, PHI_ONSET, TILT_ONSET_STEP)
        + _range(PHI_ONSET, PHI_MAX, TILT2_STEP)
    )


def jet_advance(phi: float, drop: float) -> float:
    a = max(phi, JET_PHI0) - math.pi / 2
    v = JET_V * math.sqrt(math.sin(a))
    vh, vz = v * math.cos(a), v * math.sin(a)
    t = (-vz + math.sqrt(vz * vz + 2.0 * G_ACC * max(drop, 0.0))) / G_ACC
    return vh * t


def jet_back(phi: float, mouth_z: float, rim_z: float, r_in: float = 1.0) -> float:
    d0 = mouth_z - rim_z
    mid = 0.5 * (jet_advance(phi, d0) + jet_advance(phi, d0 + JET_REF_DEPTH))
    cap = r_in - LIP_MARGIN - P.NECK_R_IN * max(0.0, -math.cos(phi))
    return max(0.0, min(mid, cap))


def early_clear(phi: float) -> float:
    if phi <= EARLY_PHI0:
        return EARLY_CLEAR
    if phi >= EARLY_PHI1:
        return 0.0
    return EARLY_CLEAR * (EARLY_PHI1 - phi) / (EARLY_PHI1 - EARLY_PHI0)


def pour_bottle_poses(
    cup_xy,
    d,
    rim_z: float,
    yaw: float,
    s: int,
    pitch: float,
    obstacles,
    mouth_min: float,
    r_in: float = 1.0,
    T_bg: np.ndarray | None = None,
):
    pts = held_points(pitch, T_bg)
    top = np.array([0.0, 0.0, P.BOTTLE_H])
    phis = pour_schedule()
    c = np.asarray(cup_xy, float)
    d2 = np.asarray(d, float)[:2]
    h = None
    backs = [0.0] * len(phis)
    for _ in range(2):
        if h is not None:
            backs = [jet_back(phi, z, rim_z, r_in) for phi, z in zip(phis, h)]
            i0 = next((i for i, phi in enumerate(phis) if phi >= JET_PHI0), len(phis) - 1)
            backs[:i0] = [backs[i0]] * i0
        req = []
        for phi, b in zip(phis, backs):
            R = bottle_rot(yaw, s, phi)
            rel = (pts - top) @ R.T
            ex = early_clear(phi)
            obs = (
                [(oc, r + ex * EARLY_MARGIN / EARLY_CLEAR, z + ex) for oc, r, z in obstacles] if ex > 0.0 else obstacles
            )
            req.append(mouth_lift(rel, c - b * d2, obs, TABLE_Z + 0.012))
        req = np.asarray(req)
        h = np.maximum(mouth_min, np.maximum.accumulate(req[::-1])[::-1])
    poses = []
    for phi, z, b in zip(phis, h, backs):
        R = bottle_rot(yaw, s, phi)
        m = c - b * d2
        poses.append(pose(R, np.array([m[0], m[1], z]) - R @ top))
    return poses, phis, h, backs


def plan_pour(arm: str, grasp: dict, cup, q_ref):
    T_bg = grasp.get("T_bg")
    if T_bg is None:
        T_bg = grip_in_bottle(grasp["pitch"])
    obstacles = cup_cylinders(cup, POUR_CUP_MARGIN, POUR_RIM_CLEAR)
    cands = []
    for ang in POUR_DIRS:
        d = np.array([math.cos(ang), math.sin(ang), 0.0])
        for side in (1, -1):
            xb = side * np.array([d[1], -d[0], 0.0])
            yaw = math.atan2(xb[1], xb[0])
            y_b = K.rot_z(yaw)[:, 1]
            s = -1 if float(np.dot(y_b, d)) > 0 else 1
            bposes, phis, h, backs = pour_bottle_poses(
                cup.xy,
                d,
                TABLE_Z + cup.model.height,
                yaw,
                s,
                grasp["pitch"],
                obstacles,
                TABLE_Z + POUR_MOUTH_MIN,
                cup.model.r_in,
                T_bg=T_bg,
            )
            flow = np.asarray(phis) >= math.radians(95.0)
            cost = float(np.mean(h[flow])) - TABLE_Z
            cands.append((cost, ang, yaw, s, bposes, h, backs))
    cands.sort(key=lambda c: c[0])
    best = None
    for cost, ang, yaw, s, bposes, h, backs in cands:
        tposes = [Tb @ T_bg for Tb in bposes]
        T_start = tposes[0]
        T_hi = T_start.copy()
        T_hi[2, 3] = max(T_start[2, 3], TABLE_Z + LIFT_Z)
        r = K.solve_near(arm, T_hi, q_ref, K.TCP_OFFSET, n_random=2)
        if not r.ok:
            continue
        ok = True
        for i in (len(tposes) // 2, len(tposes) - 1):
            if not K.solve_near(arm, tposes[i], r.q, K.TCP_OFFSET, n_random=2).ok:
                ok = False
                break
        if not ok:
            continue
        qs = chain_ik(arm, tposes, r.q)
        if qs is None:
            continue
        best = dict(
            yaw=yaw,
            s=s,
            ang=ang,
            T_hi=T_hi,
            q_hi=r.q,
            q_pour=qs,
            cost=cost,
            h=h,
            phis=pour_schedule(),
            bposes=bposes,
            T_bg=T_bg,
            backs=backs,
        )
        break
    return best


def mouth_matched(bposes, T_tb_cmd: np.ndarray, T_tb_true: np.ndarray) -> list[np.ndarray]:
    top = np.array([0.0, 0.0, P.BOTTLE_H, 1.0])
    T_cmd = np.linalg.inv(T_tb_cmd)
    out = []
    for Tb in bposes:
        Tt = Tb @ T_cmd
        Tt[:3, 3] += (Tb @ top)[:3] - (Tt @ T_tb_true @ top)[:3]
        out.append(Tt)
    return out


def measure_in_hand(arm: Arm, p: dict, i_m: int):
    ctx = arm.ctx
    img = ctx.obs.images.get("cam_head")
    img0 = ctx.first.images.get("cam_head")
    if img is None or img0 is None:
        return None
    cam = C.head()
    st = ctx.obs.state
    q = {a: np.asarray(st[f"{a}_arm_joint_state"], float)[:6] for a in K.ARMS}
    g = {a: float(np.ravel(st[f"{a}_ee_joint_state"])[0]) for a in K.ARMS}
    bad = RM.mask(cam, q["left"], q["right"], g["left"], g["right"], dilate=3)
    T = K.fk(arm.name, q[arm.name])
    T[:3, 3] = K.point_of(T, K.TCP_OFFSET)
    T_tb0 = np.linalg.inv(p["T_bg"])
    fit = P.fit_held_bottle(cam, P.changed_mask(img, img0), bad, T, T_tb0, contact_in_tcp(GRASP_H))
    if fit is None:
        return None
    if fit["iou"] < fit["iou0"] + MEASURE_MIN_GAIN:
        return None
    T_bg = np.linalg.inv(fit["T_tb"])
    tposes = [Tb @ T_bg for Tb in p["bposes"]]
    fwd = chain_ik(arm.name, tposes[i_m:], q[arm.name])
    if fwd is None:
        for alpha in PARTIAL_TURN:
            T_tb_a = P.held_pose(T_tb0, contact_in_tcp(GRASP_H), alpha * fit["theta"], fit["psi"], fit["dx"], fit["dz"])
            tposes = mouth_matched(p["bposes"], T_tb_a, fit["T_tb"])
            fwd = chain_ik(arm.name, tposes[i_m:], q[arm.name])
            if fwd is not None:
                break
    if fwd is None:
        return None
    bwd = chain_ik(arm.name, tposes[i_m::-1], q[arm.name], partial=True)
    n_orig = i_m + 1 - len(bwd)
    path = list(p["q_pour"][:n_orig]) + bwd[::-1] + fwd[1:]
    return path


def tcp_observed(arm: Arm) -> np.ndarray:
    return K.point_of(K.pose_to_mat(np.ravel(arm.ctx.obs.state[f"{arm.name}_ee_pose"])), K.TCP_OFFSET)


def measured_grasp(g: dict, tcp_obs) -> np.ndarray:
    Tb = pose(K.rot_z(g["yaw"]), [g["xy"][0], g["xy"][1], TABLE_Z])
    t = (np.linalg.inv(Tb) @ np.append(np.asarray(tcp_obs, float), 1.0))[:3]
    T = grip_in_bottle(g["pitch"]).copy()
    T[0, 3], T[2, 3] = t[0], t[2]
    return T


def pour_task(arm: Arm, g: dict, p: dict, memo: dict, cup=None):
    if g.get("via") is not None:
        if len(g["via"]) == 3:
            yield from arm.line_to(g["via"][2], lin_step=VIA_LIN, grip=OPEN_G)
        _, q_via = g["via"][:2]
        yield from arm.joints_to(q_via, grip=OPEN_G, joint_step=TRAVEL_JOINT)
        yield from arm.line_to(g["Tpre"], lin_step=VIA_LIN, grip=OPEN_G)
    else:
        yield from arm.joints_to(g["q_pre"], grip=OPEN_G, joint_step=TRAVEL_JOINT)
    yield from arm.set_grip(OPEN_G, 2)
    yield from arm.line_to(g["Tg"], lin_step=0.013)
    goal = g["Tg"][:3, 3]
    yield from arm.wait_until(lambda: float(np.linalg.norm(tcp_observed(arm) - goal)) < GRASP_TOL, max_steps=GRASP_WAIT)
    err = float(np.linalg.norm(tcp_observed(arm) - goal))
    if err > GRASP_REPLAN and cup is not None:
        T_bg = measured_grasp(g, tcp_observed(arm))
        p2 = plan_pour(arm.name, dict(g, T_bg=T_bg), cup, g["q_lift"])
        if p2 is not None:
            p = p2
    yield from arm.set_grip(0.0, 10)
    yield from arm.line_to(g["Tl"], lin_step=LIFT_LIN, fallback_joint=False)
    qs = p["q_pour"]
    yield from arm.line_to(p["T_hi"], lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=CARRY_JOINT)
    yield from arm.joints_to(qs[0], joint_step=START_JOINT)
    hold = HOLD
    phis = np.asarray(p["phis"])
    i_m = int(np.argmin(np.abs(phis - MEASURE_PHI)))
    for attempt in range(2):
        if attempt == 0:
            yield from arm.follow(motion.Plan(q=qs[1 : i_m + 1]))
            yield from arm.hold(MEASURE_HOLD)
            fixed = measure_in_hand(arm, p, i_m)
            if fixed is not None:
                yield from arm.joints_to(fixed[i_m], joint_step=FIX_JOINT)
                qs = fixed
            yield from arm.follow(motion.Plan(q=qs[i_m + 1 :]))
        else:
            yield from arm.follow(motion.Plan(q=qs[1:-1:3] + [qs[-1]]))
        i_set = int(np.argmin(np.abs(phis - PHI_SETTLE)))
        back = [qs[i] for i in range(len(qs) - 2, i_set, -BACK_STEP)] + [qs[i_set]]
        if attempt == 0:
            h = deadline_hold(arm.ctx.obs.step, len(back), phis, i_set)
            if h < hold:
                hold = h
        yield from arm.hold(hold)
        yield from arm.follow(motion.Plan(q=back))
        yield from arm.hold(SETTLE)
        yield from arm.follow(motion.Plan(q=[qs[i] for i in range(i_set - 2, 0, -2)] + [qs[0]]))
        yield from arm.hold(UPRIGHT_WAIT)
        hold = RETRY_HOLD
    memo["done"] = True
    yield from arm.joints_to(p["q_hi"], joint_step=0.05)


EPISODE_STEPS = 400
CHECK_PHI = math.radians(30.0)
DEADLINE_MARGIN = 10
HOLD_MIN = 15


def deadline_hold(step: int, n_back: int, phis, i_set: int) -> int:
    phis = np.asarray(phis)
    i_chk = int(np.argmin(np.abs(phis[: i_set + 1] - CHECK_PHI)))
    n_ret = n_back + SETTLE + len(range(i_set - 2, i_chk, -2)) + 1
    return int(max(HOLD_MIN, min(HOLD, EPISODE_STEPS - DEADLINE_MARGIN - int(step) - n_ret)))


def idle_task(arm: Arm, memo: dict, brace: dict | None = None):
    if brace is not None:
        yield from arm.joints_to(brace["q_pre"], grip=brace["g"], joint_step=TRAVEL_JOINT)
        yield from arm.line_to(brace["T"], lin_step=BRACE_LIN, grip=brace["g"])
    yield from arm.wait_until(lambda: memo.get("done", False), max_steps=2000)


LIQUID_VOL = 0.000164
CUP_FILL_MAX = 0.85
BRACE_GS = 0.0, 1.0
BRACE_Z = 0.035
BRACE_GAP = 0.006
BRACE_SLIDE_MAX = 0.015
BRACE_WS = (
    0.01,
    0.0,
    -0.005,
)
BRACE_LIFT = 0.1
BRACE_LIN = 0.01
BRACE_CLEAR = 0.025
BRACE_CLEAR_CAP = 0.1


def cup_capacity(model) -> float:
    prof = np.asarray(model.profile, float)
    wall = float(prof[-1, 1]) - model.r_in
    zs = np.linspace(model.floor, model.height, 60)
    r = np.maximum(np.interp(zs, prof[:, 0], prof[:, 1]) - wall, 0.0)
    return float(np.trapz(math.pi * r * r, zs))


def needs_brace(model) -> bool:
    return cup_capacity(model) * CUP_FILL_MAX < LIQUID_VOL


def plan_brace(arm: str, cup, p: dict, q_ref, bxy):
    Rb = p["bposes"][-1][:3, :3]
    u = np.asarray(Rb[:2, 2], float)
    if np.linalg.norm(u) < 1e-06:
        return None
    u = u / np.linalg.norm(u)
    rej = {}
    cands = []
    for g in BRACE_GS:
        for wo in BRACE_WS:
            c = _brace_pose(cup, u, g, wo, arm, rej)
            if c is not None:
                cands.append(c)
    cands.sort(key=lambda c: (round(c["slide"], 3), BRACE_GS.index(c["g"])))
    out = None
    for c in cands:
        out = _brace_check(arm, c, p, q_ref, bxy, cup, rej)
        if out is not None:
            break
    return out


def slide_gap(T: np.ndarray, cup, g: float, u) -> float:
    pts = T[:3, 3] + gripper_points(g, dense=True) @ T[:3, :3].T
    rel = pts[:, :2] - np.asarray(cup.xy, float)
    a = rel @ np.asarray(u, float)
    lat = rel @ np.array([-u[1], u[0]])
    r = cup.model.r_out
    m = (np.abs(lat) < r) & (pts[:, 2] <= TABLE_Z + cup.model.height) & (a > 0.0)
    if not np.any(m):
        return 1.0
    return float(np.min(a[m] - np.sqrt(r * r - lat[m] ** 2)))


def _brace_pose(cup, u, g: float, wo: float, arm: str, rej: dict):
    c = np.asarray(cup.xy, float)
    base = np.asarray(K.BASE_POS[arm][:2], float)
    w = np.array([-u[1], u[0]])
    if float(w @ (c - base)) < 0.0:
        w = -w
    R = K.rot_z(math.atan2(w[1], w[0]))
    cyl = cup_cylinders(cup)
    key = "g%d_w%d_" % (round(g), round(wo * 1000))
    for s_ in np.arange(cup.model.r_out, cup.model.r_out + 0.08, 0.002):
        xy = c + s_ * u + wo * w
        T = pose(R, [xy[0], xy[1], TABLE_Z + BRACE_Z])
        if clearance(T, cyl, g) < BRACE_GAP:
            continue
        gap = clearance(T, cyl, g, dense=True)
        if gap >= BRACE_GAP:
            slide = slide_gap(T, cup, g, u)
            if slide > BRACE_SLIDE_MAX:
                rej[key + "slide"] = round(slide, 4)
                return None
            return dict(T=T, gap=gap, slide=slide, g=g, w=wo, key=key)
    rej[key + "gap"] = 1


def _brace_check(arm: str, c: dict, p: dict, q_ref, bxy, cup, rej: dict):
    from scipy.spatial import cKDTree

    T, g, key = c["T"], c["g"], c["key"]
    gp = T[:3, 3] + gripper_points(g) @ T[:3, :3].T
    tree = cKDTree(gp)
    lo, hi = gp.min(axis=0) - BRACE_CLEAR_CAP, gp.max(axis=0) + BRACE_CLEAR_CAP
    held = held_points(0.0, p.get("T_bg"))
    clear = BRACE_CLEAR_CAP
    for Tb in p["bposes"]:
        hw = Tb[:3, 3] + held @ Tb[:3, :3].T
        hw = hw[np.all((hw >= lo) & (hw <= hi), axis=1)]
        if len(hw):
            clear = min(clear, float(tree.query(hw, distance_upper_bound=BRACE_CLEAR_CAP)[0].min()))
    if clear < BRACE_CLEAR:
        rej[key + "held"] = round(clear, 4)
        return None
    Tpre = T.copy()
    Tpre[2, 3] += BRACE_LIFT
    r = K.solve_near(arm, Tpre, q_ref, K.TCP_OFFSET, n_random=3)
    down = chain_ik(arm, [Tpre + (T - Tpre) * t for t in np.linspace(0.2, 1.0, 5)], r.q) if r.ok else None
    if down is None:
        rej[key + "ik"] = 1
        return None
    lim = K.JOINT_LIMITS
    margin = min(float(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q))) for q in [r.q] + down)
    if margin < 0.05:
        rej[key + "margin"] = round(margin, 3)
        return None
    travel = path_clearance(
        arm, motion.joint_move(q_ref, r.q, joint_step=TRAVEL_JOINT).q, bottle_cylinders(bxy) + cup_cylinders(cup)
    )
    if travel < VIA_CLEAR:
        rej[key + "travel"] = round(travel, 4)
        return None
    return dict(c, q_pre=r.q, q=down[-1], clear=clear)


def perceive(ctx, arms, obs=None, tag: str = "head_0000"):
    o = ctx.first if obs is None else obs
    img = o.images.get("cam_head")
    if img is None:
        return None, None
    cam = C.head()
    if obs is None:
        q = {a: arms[a].q for a in K.ARMS}
    else:
        q = {a: np.asarray(o.state[f"{a}_arm_joint_state"], dtype=float) for a in K.ARMS}
    robot = RM.mask(cam, q["left"], q["right"])
    bottle, cup = P.detect(img, cam, robot)
    return bottle, cup


CLEAR_LIFT = 0.1
CLEAR_LIN = 0.025
CLEAR_SETTLE = 10
RELOOK_WAIT = 15
RELOOKS = 3


def clear_reason(bottle, cup, plan) -> str | None:
    if bottle is None or cup is None:
        return "perception"
    if plan is None:
        return "no plan"
    near = "left" if bottle.x < 0 else "right"
    if plan[0] != near:
        return "far arm"
    via = plan[1].get("via")
    if via is not None and len(via) == 3:
        return "up first"


def clear_task(arm: Arm):
    T = arm.T.copy()
    T[:3, 3] = K.point_of(T, K.TCP_OFFSET) + np.array([0.0, 0.0, CLEAR_LIFT])
    yield from arm.line_to(T, lin_step=CLEAR_LIN, joint_step=TRAVEL_JOINT, grip=arm.g)


@register
class PourIntoCup(Program):
    name = "pour_into_cup"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "pour" in s and "bottle" in s and ("cup" in s) and ("bowl" not in s) and ("ball" not in s):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.first
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        bottle, cup = perceive(ctx, arms)
        plan = None
        if bottle is not None and cup is not None:
            plan = choose_grasp(bottle.xy, {a: arms[a].q for a in K.ARMS}, cup)
        why = clear_reason(bottle, cup, plan)
        if why is not None:
            gripper = why in ("far arm", "up first")
            lift = ["left" if bottle.x < 0 else "right"] if gripper else list(K.ARMS)
            yield from run_arms(arms, {a: clear_task(arms[a]) for a in lift})
            for i in range(1 + (0 if gripper else RELOOKS)):
                if not gripper:
                    yield from run_arms(arms, {"left": arms["left"].hold(CLEAR_SETTLE if i == 0 else RELOOK_WAIT)})
                b2, c2 = perceive(ctx, arms, obs=ctx.obs, tag=f"head_clear{i}")
                if b2 is not None and (not (b2.weak and bottle is not None and (not bottle.weak))):
                    bottle = b2
                cup = c2 if c2 is not None else cup
                if bottle is None or cup is None:
                    continue
                plan2 = choose_grasp(bottle.xy, {a: arms[a].q for a in K.ARMS}, cup)
                if plan2 is not None:
                    plan = plan2
                    break
            if bottle is None or cup is None:
                return
        if plan is None:
            return
        a, g, p = plan
        other = "right" if a == "left" else "left"
        memo = {}
        brace = plan_brace(other, cup, p, arms[other].q, bottle.xy) if needs_brace(cup.model) else None
        yield from run_arms(arms, {a: pour_task(arms[a], g, p, memo, cup), other: idle_task(arms[other], memo, brace)})
