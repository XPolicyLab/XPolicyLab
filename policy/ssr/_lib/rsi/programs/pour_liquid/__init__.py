from __future__ import annotations

import math
import re

import numpy as np

from . import bottles as B
from ...perception import cameras as C
from ...perception import robot_mask as RM
from ...planning import collision as CO
from ...planning import kinematics as K
from ...planning import motion
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

TABLE_Z = K.TABLE_Z
ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
GRASP_H = 0.08
AX_SHIFT = 0.004
PRE_BACK = 0.07
OPEN_G = 1.0
LIFT_Z = 0.25
VIA_Y = -0.15
POUR_MOUTH_Z = 0.075
PHI_1 = math.radians(90.0)
PHI_MAX = math.radians(145.0)
TILT1_STEP = math.radians(6.0)
TILT2_STEP = math.radians(2.5)
HOLD = 30
POUR_BOWL_MARGIN = 0.008
POUR_RIM_CLEAR = 0.012
POUR_BOTTLE_MARGIN = 0.015
MOUTH_BACK = 0.0
TRAVEL_JOINT = 0.12
CARRY_JOINT = 0.085
CORR_BACK = 0.16
CORR_Z = 0.1
HIGH_Z = 0.3
TRANSFER_CLEAR = 0.012
PARK = {"left": (-0.4, -0.2), "right": (0.4, -0.2)}
PARK_Z = 0.25
PARK_R = 0.1
DROP = {"left": (-0.47, -0.1), "right": (0.47, -0.1)}
PREFETCH_CLEAR = 0.03
CARRY_LIN = 0.022
CARRY_ANG = math.radians(5.0)
HIGH_CARRY_Z = 0.36
EMPTY_JOINT = 0.12
GRASP_YAWS = 0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6
GRASP_PITCHES = 0.0, math.radians(10.0)
POUR_DIRS = tuple(math.radians(a) for a in (90.0, 60.0, 120.0, 30.0, 150.0, 0.0, 180.0))
BOWL_CLEAR = 0.006
BOTTLE_CLEAR = 0.008


def parse_instruction(text: str) -> list[tuple[str, str]]:
    s = text.lower()
    pat = re.compile(
        r"(?:(first|second|third|fourth|fifth)\s+)?([a-z]+)\s+bottle\s+into\s+the\s+(?:(?:first|second|third|fourth|fifth)\s+)?([a-z]+)\s+bowl"
    )
    out = []
    for i, m in enumerate(pat.finditer(s)):
        k = ORDINALS.get(m.group(1), i) if m.group(1) else i
        out.append((k, m.group(2), m.group(3)))
    out.sort(key=lambda t: t[0])
    return [(b, w) for _, b, w in out]


def bottle_rot(yaw: float, s: int, phi: float) -> np.ndarray:
    return K.rot_z(yaw) @ K.rot_x(s * phi)


def grip_in_bottle(pitch: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = K.rot_y(pitch)
    T[:3, 3] = [AX_SHIFT, 0.0, GRASP_H]
    return T


def bottle_pose(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


_GRIP_PTS = None


def gripper_points() -> np.ndarray:
    global _GRIP_PTS
    if _GRIP_PTS is None:
        from scipy.spatial import ConvexHull

        poses = RM.link_poses("left", np.zeros(6), OPEN_G)
        inv6 = np.linalg.inv(poses["link6"])
        pts = []
        for name in ("link5", "link6", "link7", "link8", "camera_base", "camera"):
            T = inv6 @ poses[name]
            for pc in RM._pieces().get(name, []):
                v = pc @ T[:3, :3].T + T[:3, 3]
                tri = v[ConvexHull(v).simplices]
                pts += [v, tri.mean(axis=1)]
        p = np.vstack(pts)
        p[:, 0] -= K.TCP_OFFSET
        _GRIP_PTS = p
    return _GRIP_PTS


def clearance(T_tcp: np.ndarray, bowls, bottles) -> float:
    p = T_tcp[:3, 3] + gripper_points() @ T_tcp[:3, :3].T
    best = 1.0
    for xy, r, ztop in list(bowls) + list(bottles):
        d_xy = np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) - r
        d_z = p[:, 2] - ztop
        d = np.where(d_xy < 0, d_z, np.where(d_z > 0, np.hypot(d_xy, d_z), d_xy))
        best = min(best, float(d.min()))
    return best


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


def _range(a0: float, a1: float, step: float) -> list[float]:
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    return [a0 + (a1 - a0) * (i + 1) / n for i in range(n)]


def plan_grasp(arm: str, bxy, q_ref, bowls, bottles):
    base = np.array(K.BASE_POS[arm][:2])
    yaw0 = math.atan2(bxy[1] - base[1], bxy[0] - base[0])
    for pitch in GRASP_PITCHES:
        for dy in GRASP_YAWS:
            yaw = yaw0 + dy
            Tb = bottle_pose(K.rot_z(yaw), [bxy[0], bxy[1], TABLE_Z])
            Tg = Tb @ grip_in_bottle(pitch)
            a = Tg[:3, 0]
            Tpre = Tg.copy()
            Tpre[:3, 3] -= PRE_BACK * a
            Tl = Tg.copy()
            Tl[2, 3] = TABLE_Z + LIFT_Z
            cl = min(clearance(Tpre + (Tg - Tpre) * t, bowls, bottles) for t in (0.0, 0.33, 0.66, 1.0))
            if cl < min(BOWL_CLEAR, BOTTLE_CLEAR):
                continue
            r = K.solve_near(arm, Tpre, q_ref, K.TCP_OFFSET, n_random=3)
            if not r.ok:
                continue
            qs = chain_ik(arm, [Tpre + (Tg - Tpre) * t for t in (0.5, 1.0)] + [Tl], r.q)
            if qs is None:
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
            )
            return plan


_BOTTLE_PTS = None


def bottle_points() -> np.ndarray:
    global _BOTTLE_PTS
    if _BOTTLE_PTS is None:
        a = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
        _BOTTLE_PTS = np.vstack(
            [np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], axis=1) for z, r in B.BOTTLE_PROFILE]
        )
    return _BOTTLE_PTS


def held_points(pitch: float) -> np.ndarray:
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
    return [0.0] + _range(0.0, PHI_1, TILT1_STEP) + _range(PHI_1, PHI_MAX, TILT2_STEP)


def pour_bottle_poses(M_xy, yaw: float, s: int, pitch: float, obstacles, mouth_min: float):
    pts = held_points(pitch)
    top = np.array([0.0, 0.0, B.BOTTLE_H])
    phis = pour_schedule()
    req = []
    for phi in phis:
        R = bottle_rot(yaw, s, phi)
        rel = (pts - top) @ R.T
        req.append(mouth_lift(rel, M_xy, obstacles, TABLE_Z + 0.012))
    req = np.asarray(req)
    h = np.maximum(mouth_min, np.maximum.accumulate(req[::-1])[::-1])
    poses = []
    for phi, z in zip(phis, h):
        R = bottle_rot(yaw, s, phi)
        M = np.array([M_xy[0], M_xy[1], z])
        poses.append(bottle_pose(R, M - R @ top))
    return poses, phis, h


def plan_pour(arm: str, grasp: dict, bowl_xy, q_ref, obstacles):
    T_bg = grip_in_bottle(grasp["pitch"])
    cands = []
    for ang in POUR_DIRS:
        d = np.array([math.cos(ang), math.sin(ang), 0.0])
        for side in (1, -1):
            xb = side * np.array([d[1], -d[0], 0.0])
            yaw = math.atan2(xb[1], xb[0])
            y_b = K.rot_z(yaw)[:, 1]
            s = -1 if float(np.dot(y_b, d)) > 0 else 1
            M_xy = np.asarray(bowl_xy, float) - MOUTH_BACK * d[:2]
            bposes, phis, h = pour_bottle_poses(M_xy, yaw, s, grasp["pitch"], obstacles, TABLE_Z + POUR_MOUTH_Z)
            flow = np.asarray(phis) >= math.radians(95.0)
            cost = float(np.mean(h[flow])) - TABLE_Z
            cands.append((cost, ang, yaw, s, bposes, h))
    cands.sort(key=lambda c: c[0])
    best = None
    for cost, ang, yaw, s, bposes, h in cands:
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
        best = dict(yaw=yaw, s=s, ang=ang, T_hi=T_hi, q_hi=r.q, q_pour=qs, cost=cost, h=h)
        break
    return best


class Scene:
    def __init__(self, bottles: dict, bowls: dict, pours: list[tuple[str, str]]):
        self.bottles, self.bowls, self.pours = bottles, bowls, pours

    def obstacles(self, exclude_bottle: str | None = None, standing=None):
        bowls = [(b.xy, B.BOWL_R + BOWL_CLEAR, TABLE_Z + B.BOWL_H + BOWL_CLEAR) for b in self.bowls.values()]
        bots = [
            (b.xy, 0.0415 + BOTTLE_CLEAR, TABLE_Z + B.BOTTLE_H + BOTTLE_CLEAR)
            for c, b in self.bottles.items()
            if c != exclude_bottle and (standing is None or c in standing)
        ]
        return bowls, bots


def assign_arms(scene: Scene) -> list[str]:
    xs = [scene.bottles[b].x for b, _ in scene.pours]
    arms: list[str | None] = [None] * len(xs)
    for i, x in enumerate(xs):
        if x < -0.08:
            arms[i] = "left"
        elif x > 0.08:
            arms[i] = "right"
    for i in range(len(xs)):
        if arms[i] is None:
            prev = arms[i - 1] if i > 0 else None
            nxt = arms[i + 1] if i + 1 < len(xs) else None
            if prev is not None:
                arms[i] = "right" if prev == "left" else "left"
            elif nxt is not None:
                arms[i] = "right" if nxt == "left" else "left"
            else:
                arms[i] = "left" if xs[i] <= 0 else "right"
    return arms


def back_pose(Tg: np.ndarray, back: float, z: float) -> np.ndarray:
    a = Tg[:3, 0].copy()
    a[2] = 0.0
    a /= max(1e-09, float(np.linalg.norm(a)))
    out = Tg.copy()
    out[:3, 3] = Tg[:3, 3] - back * a
    out[2, 3] = z
    return out


def path_clear(arm: str, q0, q1, bowls, bottles, n: int = 10) -> float:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    best = 1.0
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        T = K.fk(arm, q0 + (q1 - q0) * t)
        T[:3, 3] = K.point_of(T, K.TCP_OFFSET)
        best = min(best, clearance(T, bowls, bottles))
    return best


def transfer(arm: Arm, g: dict, bowls, bottles, T_release: np.ndarray | None = None):
    C2 = back_pose(g["Tg"], CORR_BACK, TABLE_Z + CORR_Z)
    if T_release is not None:
        C1 = back_pose(T_release, CORR_BACK - AX_SHIFT, TABLE_Z + CORR_Z)
        yield from arm.line_to(C1, lin_step=0.016)
    p0 = arm.tcp()
    if p0[2] < TABLE_Z + CORR_Z - 0.005:
        T_up = arm.T.copy()
        T_up[:3, 3] = [p0[0], p0[1], TABLE_Z + CORR_Z]
        yield from arm.line_to(T_up, lin_step=0.012)
    r2 = K.solve_near(arm.name, C2, arm.q, K.TCP_OFFSET, n_random=2)
    if arm.status == "ok" and r2.ok:
        if path_clear(arm.name, arm.q, r2.q, bowls, bottles) >= TRANSFER_CLEAR:
            yield from arm.joints_to(r2.q, grip=OPEN_G, joint_step=TRAVEL_JOINT)
        else:
            yield from arm.line_to(C2, lin_step=0.012, ang_step=math.radians(4))
        if arm.status == "ok":
            yield from arm.line_to(g["Tpre"], lin_step=0.018)
            return
    T_up = arm.T.copy()
    T_up[:3, 3] = K.point_of(T_up, K.TCP_OFFSET)
    T_up[2, 3] = TABLE_Z + HIGH_Z
    yield from arm.line_to(T_up, lin_step=0.014)
    H2 = g["Tpre"].copy()
    H2[2, 3] = TABLE_Z + HIGH_Z
    yield from arm.move_to(H2, min_z=TABLE_Z + HIGH_Z - 0.02, joint_step=TRAVEL_JOINT)
    yield from arm.line_to(g["Tpre"], lin_step=0.014)


def drop_poses(arm: str, pitch: float):
    x, y = DROP[arm]
    base = np.array(K.BASE_POS[arm][:2])
    yaw = math.atan2(y - base[1], x - base[0])
    Tb = bottle_pose(K.rot_z(yaw), [x, y, TABLE_Z])
    Tset = Tb @ grip_in_bottle(pitch)
    Tset[2, 3] += 0.003
    Thi = Tset.copy()
    Thi[2, 3] = TABLE_Z + LIFT_Z
    Tback = Tset.copy()
    Tback[:3, 3] -= PRE_BACK * Tset[:3, 0]
    return Thi, Tset, Tback


def carry(arm: Arm, T_goal: np.ndarray, joint_step: float = CARRY_JOINT):
    p = arm.tcp()
    z = max(float(p[2]), TABLE_Z + LIFT_Z)
    T0 = arm.T.copy()
    T0[:3, 3] = p
    g = T_goal[:3, 3]
    way = []
    if abs(p[1] - VIA_Y) > 0.01 or p[2] < z - 0.005:
        W = T0.copy()
        W[:3, 3] = [p[0], VIA_Y, z]
        way.append(W)
    if abs(g[0] - p[0]) > 0.01:
        W = T_goal.copy()
        W[:3, 3] = [g[0], VIA_Y, max(z, float(g[2]))]
        way.append(W)
    way.append(T_goal)
    for W in way:
        yield from arm.line_to(W, lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=joint_step, fallback_joint=False)
        if arm.status == "blocked":
            yield from arm.hold(10)
            yield from arm.line_to(
                W, lin_step=CARRY_LIN, ang_step=CARRY_ANG, joint_step=joint_step, fallback_joint=False
            )
        if arm.status != "ok":
            yield from arm.transit(T_goal, TABLE_Z + HIGH_CARRY_Z, joint_step=joint_step)
            return


def park_pose(arm: str, pitch: float) -> np.ndarray:
    x, y = PARK[arm]
    base = np.array(K.BASE_POS[arm][:2])
    yaw = math.atan2(y - base[1], x - base[0])
    return tcp_pose(K.rot_z(yaw) @ K.rot_y(pitch), [x, y, TABLE_Z + PARK_Z])


def held_bottle_points(arm: str, q, pitch: float) -> np.ndarray:
    T = K.fk(arm, q)
    T[:3, 3] = K.point_of(T, K.TCP_OFFSET)
    Tb = T @ np.linalg.inv(grip_in_bottle(pitch))
    return Tb[:3, 3] + bottle_points() @ Tb[:3, :3].T


def _pts_spheres(pts: np.ndarray, sph: np.ndarray) -> float:
    d = np.linalg.norm(pts[:, None, :] - sph[None, :, :3], axis=2) - sph[None, :, 3]
    return float(d.min())


def sweep_clearance(arm_a: str, qs_a, pitch_a: float, arm_b: str, q_b, pitch_b: float, b_xy) -> float:
    sph_b = CO.arm_spheres(arm_b, q_b, 0.7)
    bot_b = held_bottle_points(arm_b, q_b, pitch_b)
    best = 1.0
    for q in qs_a:
        sph_a = CO.arm_spheres(arm_a, q, 0.7)
        pts_a = held_bottle_points(arm_a, q, pitch_a)
        ql, qr = (q, q_b) if arm_a == "left" else (q_b, q)
        best = min(best, CO.arm_distance(ql, qr, 0.7, 0.7), _pts_spheres(pts_a, sph_b), _pts_spheres(bot_b, sph_a))
        dxy = np.hypot(pts_a[:, 0] - b_xy[0], pts_a[:, 1] - b_xy[1]) - 0.0415
        best = min(best, float(np.where(pts_a[:, 2] < TABLE_Z + B.BOTTLE_H, dxy, 1.0).min()))
    return best


def corridor_points(T_from: np.ndarray, T_to: np.ndarray, z: float) -> list[np.ndarray]:
    p0, p1 = T_from[:3, 3], T_to[:3, 3]
    knots = [p0, np.array([p0[0], VIA_Y, z]), np.array([p1[0], VIA_Y, max(z, p1[2])]), p1]
    out = []
    for i, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / 0.04)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            W = np.eye(4)
            f = (i + t) / 3.0
            W[:3, :3] = K.slerp_mat(T_from[:3, :3], T_to[:3, :3], f)
            W[:3, 3] = a + (b - a) * t
            out.append(W)
    return out


def job_sweep(arm: str, g: dict, p: dict | None, q_home) -> list[np.ndarray]:
    qh = np.asarray(q_home, float)
    out = [qh + (g["q_pre"] - qh) * t for t in (0.25, 0.5, 0.75, 1.0)]
    out += [g["q_grasp"], g["q_lift"]]
    if p is not None:
        q = np.asarray(g["q_lift"], float)
        for W in corridor_points(g["Tl"], p["T_hi"], TABLE_Z + LIFT_Z):
            r = K.solve(arm, W, q, K.TCP_OFFSET, iters=40)
            if r.pos_err < 0.01:
                q = r.q
                out.append(q)
        out += list(p["q_pour"][::3]) + [p["q_pour"][-1]]
    return out


def decide_prefetch(scene: Scene, who: list[str], q_home: dict) -> dict[int, bool]:
    out = {}
    plans = {}
    for k, (bc, wc) in enumerate(scene.pours):
        a = who[k]
        first = who.index(a) == k
        if not first:
            continue
        bowls_obs, bots_obs = scene.obstacles(exclude_bottle=bc)
        g = plan_grasp(a, scene.bottles[bc].xy, q_home[a], bowls_obs, bots_obs)
        p = None
        if g is not None:
            pobs = [
                (b.xy, B.BOWL_R + POUR_BOWL_MARGIN, TABLE_Z + B.BOWL_H + POUR_RIM_CLEAR) for b in scene.bowls.values()
            ]
            pobs += [(b.xy, 0.0415 + POUR_BOTTLE_MARGIN, TABLE_Z + 0.31) for c, b in scene.bottles.items() if c != bc]
            p = plan_pour(a, g, scene.bowls[wc].xy, g["q_lift"], pobs)
        plans[k] = a, g, p
    for k, (a, g, p) in plans.items():
        if k == 0 or g is None:
            continue
        before = [j for j in range(k) if who[j] != a]
        if any(j not in plans for j in before):
            continue
        d = 1.0
        for j in before:
            aj, gj, pj = plans[j]
            if gj is None:
                continue
            d = min(
                d,
                sweep_clearance(
                    aj,
                    job_sweep(aj, gj, pj, q_home[aj]),
                    gj["pitch"],
                    a,
                    g["q_grasp"],
                    g["pitch"],
                    scene.bottles[scene.pours[k][0]].xy,
                ),
            )
        out[k] = d >= PREFETCH_CLEAR
    return out


def arm_task(arm: Arm, jobs: list[int], scene: Scene, memo: dict):
    prev_g = None
    prev_release = None
    for idx, k in enumerate(jobs):
        last = idx == len(jobs) - 1
        bcol, wcol = scene.pours[k]
        bot = scene.bottles[bcol]
        bowl = scene.bowls[wcol]
        if k > 0 and (not (idx == 0 and memo["prefetch"].get(k))):
            yield from arm.wait_until(lambda: k - 1 in memo["upright"], max_steps=900)
        standing = [c for c in scene.bottles if c not in memo["moved"]]
        bowls_obs, bots_obs = scene.obstacles(exclude_bottle=bcol, standing=standing)
        g = plan_grasp(arm.name, bot.xy, arm.q, bowls_obs, bots_obs)
        if g is None:
            for f in ("upright", "clear"):
                memo[f].add(k)
            continue
        if prev_g is None:
            yield from arm.joints_to(g["q_pre"], grip=OPEN_G, joint_step=TRAVEL_JOINT)
        else:
            _, bots_all = scene.obstacles(exclude_bottle=None, standing=standing)
            yield from transfer(arm, g, bowls_obs, bots_all, T_release=prev_release)
        yield from arm.set_grip(OPEN_G, 2)
        yield from arm.line_to(g["Tg"], lin_step=0.013)
        yield from arm.set_grip(0.0, 10)
        memo["moved"].add(bcol)
        if k > 0:
            yield from arm.wait_until(lambda: k - 1 in memo["clear"], max_steps=900)
        yield from arm.line_to(g["Tl"], lin_step=0.02, fallback_joint=False)
        pour_obs = [
            (b.xy, B.BOWL_R + POUR_BOWL_MARGIN, TABLE_Z + B.BOWL_H + POUR_RIM_CLEAR) for b in scene.bowls.values()
        ]
        pour_obs += [(b.xy, 0.0415 + POUR_BOTTLE_MARGIN, TABLE_Z + 0.31) for c, b in scene.bottles.items() if c != bcol]
        pour_obs += [
            (np.asarray(xy, float), PARK_R, TABLE_Z + 0.45) for a, xy in memo["parked"].items() if a != arm.name
        ]
        p = plan_pour(arm.name, g, bowl.xy, arm.q, pour_obs)
        Tl = g["Tl"]
        poured = False
        if p is not None:
            yield from arm.wait_until(lambda: all(j in memo["upright"] for j in range(k)), max_steps=900)
            qs = p["q_pour"]
            T_start = K.fk(arm.name, qs[0])
            T_start[:3, 3] = K.point_of(T_start, K.TCP_OFFSET)
            yield from carry(arm, T_start, CARRY_JOINT)
            yield from arm.joints_to(qs[0], joint_step=0.05)
            yield from arm.follow(motion.Plan(q=qs[1:]))
            yield from arm.hold(HOLD)
            back = qs[::-1]
            yield from arm.follow(motion.Plan(q=back[1::2] + [qs[0]]))
            yield from arm.joints_to(p["q_hi"], joint_step=0.06)
            poured = True
        memo["upright"].add(k)
        if last:
            yield from carry(arm, park_pose(arm.name, g["pitch"]), EMPTY_JOINT)
            memo["parked"][arm.name] = PARK[arm.name]
            memo["clear"].add(k)
            continue
        nxt = jobs[idx + 1]
        nx = scene.bottles[scene.pours[nxt][0]].x
        x_from = bowl.x if poured else bot.x
        Thi, Tset, Tback = drop_poses(arm.name, g["pitch"])
        use_spot = nxt == k + 1 and abs(x_from - bot.x) + abs(bot.x - nx) <= abs(x_from - DROP[arm.name][0]) + abs(
            DROP[arm.name][0] - nx
        )
        drop_q = None
        if not use_spot:
            r_hi = K.solve_near(arm.name, Thi, arm.q, K.TCP_OFFSET, n_random=2)
            drop_q = (
                chain_ik(arm.name, [Thi.copy() + (Tset - Thi) * t for t in np.linspace(0.2, 1.0, 5)] + [Tback], r_hi.q)
                if r_hi.ok
                else None
            )
        if drop_q is not None:
            yield from carry(arm, Thi, EMPTY_JOINT)
            memo["clear"].add(k)
            new_xy = DROP[arm.name]
        else:
            yield from carry(arm, Tl, EMPTY_JOINT)
            Tset = g["Tg"].copy()
            Tset[2, 3] += 0.003
            Tback = g["Tpre"]
            new_xy = bot.x, bot.y
        Tmid = Tset.copy()
        Tmid[2, 3] += 0.03
        yield from arm.line_to(Tmid, lin_step=0.025)
        yield from arm.line_to(Tset, lin_step=0.008)
        yield from arm.set_grip(OPEN_G, 6)
        yield from arm.line_to(Tback, lin_step=0.015)
        scene.bottles[bcol] = B.Fit("bottle", bcol, float(new_xy[0]), float(new_xy[1]), 0.0, 0)
        memo["moved"].discard(bcol)
        memo["clear"].add(k)
        prev_release = Tset
        prev_g = g


@register
class PourLiquid(Program):
    name = "pour_liquid"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower()
        if "pour" in s and "bottle" in s and ("bowl" in s) and parse_instruction(s):
            return 0.9
        return 0.0

    def run(self, ctx):
        o = ctx.first
        pours = parse_instruction(o.instruction)
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        img = o.images.get("cam_head")
        cam = C.head()
        bad = RM.mask(cam, arms["left"].q, arms["right"].q)
        bottles = B.detect_bottles(img, cam, bad)
        bowls = B.detect_bowls(img, cam, bad)
        missing = [c for p in pours for c in p if c not in bottles and c not in bowls]
        if not pours or missing:
            return
        scene = Scene(bottles, bowls, pours)
        who = assign_arms(scene)
        memo = {"upright": set(), "clear": set(), "moved": set(), "prefetch": {}, "parked": {}}
        memo["prefetch"] = decide_prefetch(scene, who, {a: arms[a].q_home for a in K.ARMS})
        link_arms(arms)
        tasks = {}
        for a in K.ARMS:
            jobs = [k for k, w in enumerate(who) if w == a]
            if jobs:
                tasks[a] = arm_task(arms[a], jobs, scene, memo)
        yield from run_arms(arms, tasks)
