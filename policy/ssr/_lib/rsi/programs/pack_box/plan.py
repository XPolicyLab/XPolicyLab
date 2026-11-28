from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import fit as PF
from ...planning import collision as CO
from ...planning import kinematics as K
from ...skills import pinch as P
from ...skills.grasp import base_seed

FLOOR = 0.003
WALL_H = 0.1
TIP_OVER_WALL = 0.015
DROP = 0.012
INNER_U = 0.095
INNER_V = 0.08
FINGER_MARGIN = 0.006
FINGER_OUT = 0.012
RELEASE_OPEN = 0.02
OBJ_CLEAR = 0.01
CARRY_CLEAR = 0.025
CARRY_CLEAR_CAT = {"shoe": 0.05}
DROP_CAT = {"shoe": 0.035}
HOVER_DZ = 0.07
TILTS = 0.0, math.radians(15), math.radians(25), math.radians(35)
MAX_TILT = {"car": math.radians(25), "electric_toothbrush": math.radians(25)}
ROLLS = tuple(math.radians(d) for d in (0, -20, 20, -40, 40, -60, 60))
PER_TILT = 20
PLAN_BUDGET = 15.0
SLOT_V = {"car": 0.0, "electric_toothbrush": 0.05, "hammer": -0.045, "shoe": 0.0}
ORDER = "car", "electric_toothbrush", "hammer", "shoe"
SUPPORT_OVERLAP = -0.012
ARM_MARGIN = 0.012
OBJ_MARGIN = 0.01
NEAR_MARGIN = 0.005
FRONT_GAP = 0.09
SAFE_ZS = 0.2, 0.22, 0.24, 0.27
RELEASE_RAISE = 0.0, 0.015, 0.03, 0.045, 0.06, 0.08
ROUTE_STEP = 0.08
HOUSING_C = np.array([0.042, 0.0, 0.0])
HOUSING_HALF = np.array([0.042, 0.087, 0.03])
IK_STALL = 8


def _ik(arm, target, current, offset=0.0, **kw):
    kw.setdefault("stall", IK_STALL)
    return K.solve_near(arm, target, current, offset, **kw)


def _spheres():
    from ...perception import robot_mask as RM

    return RM.spheres()


_SPH: dict = {}
TABLE_LINKS = "link3", "link4", "link5", "camera", "camera_base"


def _arm_sph() -> dict:
    if not _SPH:
        _SPH.update({k: v for k, v in _spheres().items() if k != "link6"})
    return _SPH


STATS: dict = {}


def _count(key: str) -> None:
    STATS[key] = STATS.get(key, 0) + 1


class BoxObstacle:
    def __init__(self, fit: PF.ObjFit):
        from scipy.spatial import cKDTree

        a = PF.assets()[fit.key]
        self.pts = fit.points()[a.points[:, 2] > 0.012]
        self.tree = cKDTree(self.pts)

    def dist(self, pts) -> np.ndarray:
        return self.tree.query(np.asarray(pts, dtype=float).reshape(-1, 3), k=1)[0]

    def arm_clear(self, arm: str, q, grip: float, fingers: bool = True) -> float:
        return float(self.arm_clear_many(arm, [q], grip, fingers)[0])

    def arm_clear_many(self, arm: str, qs, grip: float, fingers: bool = True, table: bool = False) -> np.ndarray:
        from ...perception import robot_mask as RM

        sph = _arm_sph()
        qs = np.asarray(qs, dtype=float).reshape(-1, 6)
        cs, rs, own, low_cs, low_rs, low_own, t6s = [], [], [], [], [], [], []
        qf = max(0.0, RM.finger_q(grip))
        fx = np.asarray(CO._FINGER_X, dtype=float)
        for i, q in enumerate(qs):
            poses = RM.link_poses(arm, q, grip)
            for name, s in sph.items():
                t = poses.get(name)
                if t is None:
                    continue
                c = s[:, :3] @ t[:3, :3].T + t[:3, 3]
                cs.append(c)
                rs.append(s[:, 3])
                own.append(np.full(len(s), i))
                if table and name in TABLE_LINKS:
                    low_cs.append(c)
                    low_rs.append(s[:, 3])
                    low_own.append(np.full(len(s), i))
            t6 = poses["link6"]
            t6s.append(t6)
            if fingers or table:
                for sign in (1.0, -1.0):
                    c = t6[:3, 3] + np.outer(fx, t6[:3, 0]) + t6[:3, 1] * sign * (0.0004 + qf + 0.006)
                    r = np.full(len(fx), CO._FINGER_R)
                    if fingers:
                        cs.append(c)
                        rs.append(r)
                        own.append(np.full(len(fx), i))
                    if table:
                        low_cs.append(c)
                        low_rs.append(r)
                        low_own.append(np.full(len(fx), i))
        out = np.full(len(qs), 9.0)
        d = self.dist(np.vstack(cs)) - np.concatenate(rs)
        np.minimum.at(out, np.concatenate(own), d)
        c6 = np.array([t[:3, :3] @ HOUSING_C + t[:3, 3] for t in t6s])
        for i, idx in enumerate(self.tree.query_ball_point(c6, float(np.linalg.norm(HOUSING_HALF)) + 0.05)):
            if idx:
                t6 = t6s[i]
                L = (self.pts[idx] - t6[:3, 3]) @ t6[:3, :3] - HOUSING_C
                out[i] = min(out[i], float(np.linalg.norm(np.maximum(np.abs(L) - HOUSING_HALF, 0.0), axis=1).min()))
        if table:
            z = np.vstack(low_cs)[:, 2] - np.concatenate(low_rs) - K.TABLE_Z
            np.minimum.at(out, np.concatenate(low_own), z)
        return out

    def held_clear(self, arm: str, q, held_l6) -> float:
        if held_l6 is None:
            return 9.0
        T = K.fk(arm, q)
        return float(self.dist(held_l6 @ T[:3, :3].T + T[:3, 3]).min())


def housing_clear(obs: BoxObstacle, R: np.ndarray, tcp) -> float:
    o6 = np.asarray(tcp, dtype=float) - R[:, 0] * K.TCP_OFFSET
    c = R @ HOUSING_C + o6
    idx = obs.tree.query_ball_point(c, float(np.linalg.norm(HOUSING_HALF)) + 0.05)
    if not idx:
        return 9.0
    L = (obs.pts[idx] - o6) @ R - HOUSING_C
    return float(np.linalg.norm(np.maximum(np.abs(L) - HOUSING_HALF, 0.0), axis=1).min())


def held_in_link6(arm: str, q_grasp, pts_world: np.ndarray, n: int = 300) -> np.ndarray:
    idx = np.linspace(0, len(pts_world) - 1, min(n, len(pts_world))).astype(int)
    T = K.fk(arm, q_grasp)
    return (pts_world[idx] - T[:3, 3]) @ T[:3, :3]


def seg_clear(
    obs: BoxObstacle,
    arm: str,
    q0,
    q1,
    grip: float,
    held_l6=None,
    *,
    margin_arm: float = ARM_MARGIN,
    margin_obj: float = OBJ_MARGIN,
    fingers: bool = True,
    t0: float = 0.0,
    table_clear: float = 0.01,
    table_arm: bool = False,
) -> float:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    n = max(2, int(math.ceil(float(np.max(np.abs(q1 - q0))) / ROUTE_STEP)) + 1)
    qs = q0 + (q1 - q0) * np.linspace(t0, 1.0, n)[:, None]
    c = obs.arm_clear_many(arm, qs, grip, fingers, table=table_arm) - margin_arm
    if held_l6 is not None:
        P_ = np.stack([held_l6 @ T[:3, :3].T + T[:3, 3] for T in (K.fk(arm, q) for q in qs)])
        d = obs.dist(P_.reshape(-1, 3)).reshape(len(qs), -1).min(axis=1)
        c = np.minimum(c, d - margin_obj)
        c = np.minimum(c, P_[:, :, 2].min(axis=1) - (K.TABLE_Z + table_clear))
    return float(c.min())


def route_clear(obs, arm, q0, qs, grip, held_l6=None, **kw) -> float:
    worst, prev = 9.0, q0
    for q in qs:
        worst = min(worst, seg_clear(obs, arm, prev, q, grip, held_l6, **kw))
        prev = q
    return worst


def plan_route(obs: BoxObstacle, arm: str, q0, goal, grip: float, held_l6=None, *, rrt: bool = True, **kw):
    q0 = np.asarray(q0, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if goal.shape == (4, 4):
        r = _ik(arm, goal, q0, K.TCP_OFFSET, n_random=2)
        if not r.ok:
            return None, -9.0, "unreachable"
        q_goal = r.q
        T_goal = goal
    else:
        q_goal = goal
        T_goal = None
    best = [q_goal], route_clear(obs, arm, q0, [q_goal], grip, held_l6, **kw), "direct"
    if best[1] >= 0.0:
        return best
    T0 = K.fk(arm, q0)
    p0 = K.point_of(T0, K.TCP_OFFSET)
    Tg = T_goal if T_goal is not None else K.fk(arm, q_goal)
    pg = K.point_of(Tg, K.TCP_OFFSET) if T_goal is None else T_goal[:3, 3]
    Rg = Tg[:3, :3]
    y_front = float(obs.pts[:, 1].min()) - FRONT_GAP
    z_f = max(float(p0[2]), float(pg[2]))
    if p0[1] > y_front or pg[1] > y_front:
        qs, prev, ok = [], q0, True
        for xy_w, Rw in (((p0[0], min(p0[1], y_front)), T0[:3, :3]), ((pg[0], min(pg[1], y_front)), Rg)):
            Tw = np.eye(4)
            Tw[:3, :3] = Rw
            Tw[:3, 3] = [xy_w[0], xy_w[1], z_f]
            rw = _ik(arm, Tw, prev, K.TCP_OFFSET, n_random=1)
            if not rw.ok:
                ok = False
                break
            qs.append(rw.q)
            prev = rw.q
        if ok:
            if T_goal is not None:
                rg = _ik(arm, T_goal, prev, K.TCP_OFFSET, n_random=1)
                ok = rg.ok
                if ok:
                    qs.append(rg.q)
            else:
                qs.append(q_goal)
        if ok:
            c = route_clear(obs, arm, q0, qs, grip, held_l6, **kw)
            if c > best[1]:
                best = qs, c, "front"
            if c >= 0.0:
                return best
    for dz in SAFE_ZS:
        z = K.TABLE_Z + dz
        qs, prev = [], q0
        if p0[2] < z - 0.01:
            Tu = T0.copy()
            Tu[:3, 3] = [p0[0], p0[1], z]
            ru = _ik(arm, Tu, prev, K.TCP_OFFSET, n_random=1)
            if not ru.ok:
                continue
            qs.append(ru.q)
            prev = ru.q
        if pg[2] < z - 0.01:
            Ta = np.eye(4)
            Ta[:3, :3] = Rg
            Ta[:3, 3] = [pg[0], pg[1], z]
            ra = _ik(arm, Ta, prev, K.TCP_OFFSET, n_random=1)
            if not ra.ok:
                continue
            qs.append(ra.q)
            prev = ra.q
        if T_goal is not None:
            rg = _ik(arm, T_goal, prev, K.TCP_OFFSET, n_random=1)
            if not rg.ok:
                continue
            qs.append(rg.q)
        else:
            qs.append(q_goal)
        c = route_clear(obs, arm, q0, qs, grip, held_l6, **kw)
        if c > best[1]:
            best = qs, c, f"over{int(dz * 100)}"
        if c >= 0.0:
            return best
    if not rrt:
        return None, best[1], "blocked"
    qs = rrt_connect(obs, arm, q0, q_goal, grip, held_l6, **kw)
    if qs is not None:
        return qs, route_clear(obs, arm, q0, qs, grip, held_l6, **kw), "rrt"
    return None, best[1], "blocked"


RRT_STEP = 0.3
RRT_ITERS = 250
RRT_BUDGET = 4.0
RRT_SPAN = 0.8
RRT_GOAL_BIAS = 0.15


def rrt_connect(
    obs: BoxObstacle, arm: str, q0, q1, grip: float, held_l6=None, *, seed: int = 0, budget: float | None = None, **kw
):
    import time

    kw = dict(kw)
    kw["table_arm"] = True
    t_end = time.monotonic() + (RRT_BUDGET if budget is None else budget)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    lo = np.maximum(K.JOINT_LIMITS[:, 0], np.minimum(q0, q1) - RRT_SPAN)
    hi = np.minimum(K.JOINT_LIMITS[:, 1], np.maximum(q0, q1) + RRT_SPAN)
    rng = np.random.default_rng(seed)

    def ok(a, b) -> bool:
        return seg_clear(obs, arm, a, b, grip, held_l6, **kw) >= 0.0

    def steer(a, b):
        d = b - a
        m = float(np.max(np.abs(d)))
        return b.copy() if m <= RRT_STEP else a + d * (RRT_STEP / m)

    def nearest(nodes, q) -> int:
        return int(np.argmin(np.max(np.abs(np.asarray(nodes) - q), axis=1)))

    def branch(tree, i):
        nodes, par = tree
        out = []
        while i is not None:
            out.append(nodes[i])
            i = par[i]
        return out

    ta, tb = ([q0], [None]), ([q1], [None])
    path = None
    for _ in range(RRT_ITERS):
        if time.monotonic() > t_end:
            _count("rrt_budget")
            break
        qr = tb[0][0] if rng.random() < RRT_GOAL_BIAS else rng.uniform(lo, hi)
        i = nearest(ta[0], qr)
        qn = steer(ta[0][i], qr)
        if ok(ta[0][i], qn):
            ta[0].append(qn)
            ta[1].append(i)
            while True:
                j = nearest(tb[0], qn)
                qm = steer(tb[0][j], qn)
                if not ok(tb[0][j], qm):
                    break
                tb[0].append(qm)
                tb[1].append(j)
                if float(np.max(np.abs(qm - qn))) < 1e-09:
                    a = branch(ta, len(ta[0]) - 1)[::-1]
                    b = branch(tb, len(tb[0]) - 1)
                    path = a + b[1:]
                    if float(np.max(np.abs(path[0] - q0))) > 1e-09:
                        path = path[::-1]
                    break
            if path is not None:
                break
        ta, tb = tb, ta
    if path is None:
        _count("rrt_none")
        return None
    out, i = [], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and (not ok(path[i], path[j])):
            j -= 1
        out.append(path[j])
        i = j
    return out


@dataclass
class BoxFrame:
    xy: np.ndarray
    u: np.ndarray
    v: np.ndarray
    floor_z: float

    def point(self, du: float, dv: float) -> np.ndarray:
        return self.xy + du * self.u + dv * self.v


def box_frame(fit: PF.ObjFit) -> BoxFrame:
    ay = fit.R[:, 1].copy()
    ay[2] = 0.0
    ay /= np.linalg.norm(ay)
    u = ay if ay[0] < 0 else -ay
    v = np.array([-u[1], u[0], 0.0])
    if v[1] < 0:
        v = -v
    return BoxFrame(np.array([fit.x, fit.y]), u[:2], v[:2], K.TABLE_Z + FLOOR)


def rodrigues(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


LEVEL_STEP = 0.06
LEVEL_JUMP = 0.3
LEVEL_LAST = {"why": ""}
LEVEL_LAMS = 1.0, 0.5


def _rodrigues(w) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * Kx @ Kx


def _rotvec(M) -> np.ndarray:
    th = math.acos(max(-1.0, min(1.0, (float(np.trace(M)) - 1.0) / 2.0)))
    if th < 1e-09:
        return np.zeros(3)
    if th > math.pi - 1e-06:
        A = (np.asarray(M) + np.eye(3)) / 2.0
        k = A[:, int(np.argmax(np.diag(A)))]
        return k / np.linalg.norm(k) * th
    w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def front_in_link6(arm: str, q_grasp, fit: PF.ObjFit) -> np.ndarray:
    td = np.asarray(fit.target_dir, dtype=float).ravel()
    v = np.array([td[0], td[1], td[2] if td.size > 2 else 0.0])
    return K.fk(arm, q_grasp)[:3, :3].T @ (v / max(1e-09, float(np.linalg.norm(v))))


def level_route(obs, arm: str, q0, route, held_l6, f6, grip: float = 0.0, lam: float = 1.0):
    q0 = np.asarray(q0, dtype=float)
    f6 = np.asarray(f6, dtype=float)
    f6 = f6 / max(1e-09, float(np.linalg.norm(f6)))
    segs, prev = [], q0
    for q in route:
        q = np.asarray(q, dtype=float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q - prev))) / LEVEL_STEP)))
        segs.append([prev + (q - prev) * (k / n) for k in range(1, n + 1)])
        prev = q
    flat = [q0] + [q for sg in segs for q in sg]
    d = np.concatenate([[0.0], np.cumsum([float(np.max(np.abs(b - a))) for a, b in zip(flat[:-1], flat[1:])])])
    if d[-1] < 1e-06:
        return None
    s_par = d / d[-1]
    Ts = [K.fk(arm, q) for q in flat]
    Ra, Rb = Ts[0][:3, :3], Ts[-1][:3, :3]
    fa, fb = Ra @ f6, Rb @ f6
    if math.hypot(fa[0], fa[1]) < 0.9 or math.hypot(fb[0], fb[1]) < 0.9:
        _count("level_axis")
        LEVEL_LAST["why"] = "axis"
        return None
    hs = [math.atan2(f[1], f[0]) for f in (T[:3, :3] @ f6 for T in Ts) if math.hypot(f[0], f[1]) > 0.3]
    hu = np.unwrap(np.array(hs))
    dpsi = float(hu[-1] - hu[0])
    w = _rotvec(rot_z(-dpsi) @ Rb @ Ra.T)
    out, q_prev, i = [], q0, 0
    for sg in segs:
        dense = []
        for q in sg:
            i += 1
            if i == len(flat) - 1:
                q_new = flat[-1]
            else:
                R = rot_z(s_par[i] * dpsi) @ _rodrigues(s_par[i] * w) @ Ra
                if lam < 1.0:
                    Ro = Ts[i][:3, :3]
                    R = Ro @ _rodrigues(lam * _rotvec(Ro.T @ R))
                r = _ik(
                    arm, pose(R, K.point_of(Ts[i], K.TCP_OFFSET)), q_prev, K.TCP_OFFSET, extra_seeds=(q,), n_random=0
                )
                jump = float(np.max(np.abs(np.asarray(r.q) - q_prev)))
                if not r.ok or jump > LEVEL_JUMP:
                    _count("level_ik")
                    LEVEL_LAST["why"] = (
                        f"ik@{s_par[i]:.2f}:{int(r.ok)}:{jump:.2f}:{dpsi:.2f}:{float(np.linalg.norm(w)):.2f}"
                    )
                    return None
                q_new = np.asarray(r.q, dtype=float)
            dense.append(q_new)
            q_prev = q_new
        out.append(np.array(dense))
    clear = route_clear(obs, arm, q0, [q for sg in out for q in sg], grip, held_l6)
    if clear < 0.0:
        _count("level_col")
        LEVEL_LAST["why"] = f"col{clear:.3f}"
        return None
    LEVEL_LAST["why"] = "ok"
    return out, clear


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


@dataclass
class Place:
    T_rel: np.ndarray
    T_above: np.ndarray
    T_front: np.ndarray
    q_rel: np.ndarray
    roll: float
    obj_low: float
    support: float
    q_above: np.ndarray = None


@dataclass
class Plan:
    arm: str
    category: str
    T_grasp: np.ndarray
    T_hover: np.ndarray
    q_grasp: np.ndarray
    pinch: P.Pinch
    place: Place
    hang: float
    carry_z: float
    q_hover: np.ndarray = None
    T_up: np.ndarray = None
    q_up: np.ndarray = None
    held_l6: np.ndarray = None
    carry: list = None
    carry_clear: float = 0.0
    route_name: str = ""
    footprint: tuple = None
    secs: float = 0.0
    hover_route: list = None
    q_start: np.ndarray = None


def facing_rotation(fit: PF.ObjFit, box: "BoxFrame") -> np.ndarray:
    phi0 = PF.heading(fit.target_dir)
    phi1 = math.atan2(box.u[1], box.u[0])
    return rot_z(math.remainder(phi1 - phi0, 2 * math.pi))


def footprint_v(fit: PF.ObjFit, box: "BoxFrame") -> tuple[float, float]:
    P_ = PF.assets()[fit.key].points @ (facing_rotation(fit, box) @ fit.R).T
    d = P_[:, :2] @ box.v
    return float(d.min()), float(d.max())


def slot_target(box: BoxFrame, fit: PF.ObjFit, placed=()) -> tuple[np.ndarray, float, tuple[float, float]]:
    v_s = SLOT_V.get(fit.category, 0.0)
    lo, hi = footprint_v(fit, box)
    support = box.floor_z
    for a, b, top in placed:
        if min(b, v_s + hi) - max(a, v_s + lo) > SUPPORT_OVERLAP:
            support = max(support, top)
    return box.point(0.0, v_s), support, (v_s + lo, v_s + hi)


def finger_floor(box: BoxFrame, R: np.ndarray, tip: np.ndarray, width: float, placed=()) -> float:
    half = 0.5 * (width + RELEASE_OPEN) + FINGER_OUT
    z = box.floor_z
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            c = tip + sy * half * R[:, 1] + sz * P.FINGER_HALF_W * R[:, 2]
            d = c[:2] - box.xy
            du, dv = float(d @ box.u), float(d @ box.v)
            if abs(du) > INNER_U - FINGER_MARGIN or abs(dv) > INNER_V - FINGER_MARGIN:
                return K.TABLE_Z + WALL_H + TIP_OVER_WALL
            for a, b, top in placed:
                if a - FINGER_MARGIN < dv < b + FINGER_MARGIN:
                    z = max(z, top + OBJ_CLEAR)
    return z


def _key(T: np.ndarray, pos_res: float = 0.01, rot_res: float = 0.1) -> tuple:
    return tuple(np.round(T[:3, :3].ravel() / rot_res).astype(int)) + tuple(np.round(T[:3, 3] / pos_res).astype(int))


def plan_place(
    arm: str,
    fit: PF.ObjFit,
    T_grasp: np.ndarray,
    box: BoxFrame,
    xy: np.ndarray,
    support: float,
    q_seed,
    width: float = 0.06,
    placed=(),
    obs: BoxObstacle | None = None,
    held_l6=None,
    g_open: float = 1.0,
    memo: dict | None = None,
) -> Place | None:
    Rz = facing_rotation(fit, box)
    T_obj = pose(fit.R, [fit.x, fit.y, fit.z])
    rel = np.linalg.inv(T_obj) @ T_grasp
    pts_obj = PF.assets()[fit.key].points
    u3 = np.array([box.u[0], box.u[1], 0.0])
    for high, roll in ((h, r) for h in (True, False) for r in ROLLS):
        R_obj = rodrigues(u3, roll) @ Rz @ fit.R
        Tg = pose(R_obj, [0.0, 0.0, 0.0]) @ rel
        R = Tg[:3, :3]
        tcp_rel = Tg[:3, 3]
        low = float((pts_obj @ R_obj.T)[:, 2].min())
        tip = R[:, 0] * P.TIP_BELOW + tcp_rel
        z_origin = support + DROP_CAT.get(fit.category, DROP) - low
        for _ in range(2):
            origin = np.array([xy[0], xy[1], z_origin])
            z_origin = max(z_origin, finger_floor(box, R, origin + tip, width, placed) - float(tip[2]))
        hang = float(tcp_rel[2] - low)
        carry = max(
            K.TABLE_Z + WALL_H + CARRY_CLEAR_CAT.get(fit.category, CARRY_CLEAR) + hang, z_origin + float(tcp_rel[2])
        )
        if high:
            carry_hi = K.TABLE_Z + BOX_TOP + HIGH_CLEAR + hang
            if carry_hi < carry + 0.01:
                continue
            carry = carry_hi
        T_above = pose(R, [xy[0] + tcp_rel[0], xy[1] + tcp_rel[1], carry])
        kab = "above", _key(T_above)
        if memo is not None and kab in memo:
            _count("ik_above_memo")
            continue
        ra = _ik(
            arm,
            T_above,
            q_seed,
            K.TCP_OFFSET,
            n_random=1,
            extra_seeds=(base_seed(arm, T_above[:2, 3]), base_seed(arm, T_above[:2, 3])),
        )
        if not ra.ok:
            _count("ik_above")
            if memo is not None:
                memo[kab] = False
            continue
        if obs is not None and obs.arm_clear(arm, ra.q, 0.0) < ARM_MARGIN:
            _count("col_above")
            continue
        found = None
        for dz in RELEASE_RAISE:
            zo = z_origin + dz
            p_tcp = np.array([xy[0], xy[1], zo]) + tcp_rel
            if p_tcp[2] > carry + 1e-06:
                break
            T_rel = pose(R, p_tcp)
            r = _ik(arm, T_rel, ra.q, K.TCP_OFFSET, n_random=1)
            if not r.ok:
                continue
            if obs is not None:
                if obs.arm_clear(arm, r.q, g_open, fingers=False) < NEAR_MARGIN:
                    continue
                if obs.held_clear(arm, r.q, held_l6) < 0.003:
                    continue
                if (
                    seg_clear(
                        obs, arm, ra.q, r.q, g_open, held_l6, margin_arm=NEAR_MARGIN, margin_obj=0.003, fingers=False
                    )
                    < 0.0
                ):
                    continue
            found = T_rel, r, zo
            break
        if found is None:
            _count("no_release")
            continue
        T_rel, r, z_origin = found
        pl = Place(T_rel, T_above, T_above, r.q, roll, z_origin + low, support)
        pl.q_above = ra.q
        pl.high = high
        return pl


FLIP = np.diag([1.0, -1.0, -1.0])
BOX_TOP = 0.182
HIGH_CLEAR = 0.02


def box_reachable(arm: str, box: BoxFrame, q_seed) -> bool:
    u3 = np.array([box.u[0], box.u[1], 0.0])
    heading = math.atan2(box.v[1], box.v[0])
    for roll in ROLLS:
        R = rodrigues(u3, roll) @ K.top_down(heading)
        for Rf in (R, R @ FLIP):
            T = pose(Rf, [box.xy[0], box.xy[1], K.TABLE_Z + 0.17])
            if _ik(arm, T, q_seed, K.TCP_OFFSET, n_random=1).ok:
                return True
    return False


CLOSE_ACROSS = math.radians(20.0)


def _across(p: P.Pinch, target_dir) -> bool:
    c = np.asarray(p.closing[:2], dtype=float)
    n = float(np.linalg.norm(c))
    if n < 1e-06:
        return False
    return abs(float(c @ np.asarray(target_dir, dtype=float))) / n <= math.sin(CLOSE_ACROSS)


RPERP_GOOD = 0.02
CENTRE_TIGHT = 0.015
CENTRE_BIN = 0.003
YIELD_S = 1.5
CARRY_RRT_TRIES = 2
CARRY_RRT_S = 2.0


def plan_object_iter(
    arm: str,
    fit: PF.ObjFit,
    box: BoxFrame,
    q_cur,
    placed=(),
    budget: float = PLAN_BUDGET,
    obs: BoxObstacle | None = None,
    stage: PF.ObjFit | None = None,
    avoid=(),
    centre: str | None = "both",
    tilts=None,
    max_tilt: float | None = None,
):
    import time

    t0 = time.monotonic()
    t_end = t0 + budget
    t_y = t0
    memo: dict = {}
    pts = fit.points()
    nrm = fit.normals()
    c_xy = pts[:, :2].mean(axis=0)
    base = np.asarray(K.BASE_POS[arm][:2], dtype=float)
    lean = c_xy - base
    xy, support, _fp = slot_target(box, fit, placed)
    rrt_left = [CARRY_RRT_TRIES]
    com = fit.com()
    c_ref = com[:2]
    through = [com]
    passes = {"both": ("tight", "loose"), "tight": ("tight",), "loose": ("loose",)}.get(centre, (None,))
    cand_memo: dict = {}
    lim = MAX_TILT.get(fit.category, TILTS[-1]) if max_tilt is None else max_tilt
    for ps, tilt in ((ps, t) for ps in passes for t in TILTS):
        if tilt > lim + 1e-09 or (tilts is not None and (not any(abs(tilt - t) < 1e-06 for t in tilts))):
            continue
        if tilt not in cand_memo:
            cand_memo[tilt] = [
                c
                for c in P.candidates(pts, tilt=tilt, lean=lean, normals=nrm, through=through)
                if c.cost < P.CONE_PENALTY / 2 and _across(c, fit.target_dir)
            ]
        cands = list(cand_memo[tilt])
        if ps is None:
            cands.sort(key=lambda c: (_avoided(c, fit, avoid), _r_perp(c, c_xy) > RPERP_GOOD, c.cost))
        else:
            cands = [c for c in cands if (_r_perp(c, c_ref) <= CENTRE_TIGHT) == (ps == "tight")]
            cands.sort(key=lambda c: (_avoided(c, fit, avoid), int(_r_perp(c, c_ref) / CENTRE_BIN), c.cost))
        if obs is not None:
            ok = []
            for c in cands:
                R0 = P.rotation(c.heading, arm, c.approach, c.closing)
                if (
                    min(housing_clear(obs, R0, c.tcp), housing_clear(obs, R0, c.tcp - c.approach * HOVER_DZ))
                    < NEAR_MARGIN
                ):
                    _count("pre_housing")
                    continue
                ok.append(c)
                if len(ok) >= PER_TILT:
                    break
            cands = ok
        for p in cands[:PER_TILT]:
            now = time.monotonic()
            if now > t_end:
                _count("budget")
                return None
            if now - t_y > YIELD_S:
                yield None
                t_y = time.monotonic()
            R0 = P.rotation(p.heading, arm, p.approach, p.closing)
            g_open = P.grip_for_gap(p.gap)
            for R in (R0, R0 @ FLIP):
                T = pose(R, p.tcp)
                kg = "grasp", _key(T)
                if kg in memo:
                    _count("ik_grasp_memo")
                    continue
                r = _ik(arm, T, base_seed(arm, p.tcp[:2]), K.TCP_OFFSET, n_random=2)
                if not r.ok:
                    _count("ik_grasp")
                    memo[kg] = False
                    continue
                Th = pose(R, p.tcp - p.approach * HOVER_DZ)
                rh = _ik(arm, Th, r.q, K.TCP_OFFSET, n_random=1)
                if not rh.ok:
                    _count("ik_hover")
                    continue
                hang = float(p.tcp[2] - pts[:, 2].min())
                carry = K.TABLE_Z + WALL_H + CARRY_CLEAR_CAT.get(fit.category, CARRY_CLEAR) + hang
                T_up = pose(R, [p.tcp[0], p.tcp[1], max(carry, p.tcp[2])])
                ru = _ik(arm, T_up, r.q, K.TCP_OFFSET, n_random=1)
                if not ru.ok:
                    _count("ik_up")
                    continue
                held = held_in_link6(arm, r.q, pts)
                if obs is not None:
                    if seg_clear(obs, arm, rh.q, r.q, g_open, margin_arm=NEAR_MARGIN) < 0.0:
                        _count("col_descent")
                        continue
                    if seg_clear(obs, arm, r.q, ru.q, 0.0, margin_arm=NEAR_MARGIN) < 0.0:
                        _count("col_lift")
                        continue
                if stage is None:
                    pl = plan_place(
                        arm,
                        fit,
                        T,
                        box,
                        xy,
                        support,
                        r.q,
                        width=p.width,
                        placed=placed,
                        obs=obs,
                        held_l6=held,
                        g_open=P.grip_for_gap(p.width + 0.012),
                        memo=memo,
                    )
                else:
                    pl = plan_place_table(
                        arm, fit, T, stage, r.q, obs=obs, held_l6=held, g_open=P.grip_for_gap(p.width + 0.012)
                    )
                if pl is None:
                    _count("no_place")
                    continue
                if getattr(pl, "high", False) and T_up[2, 3] < pl.T_above[2, 3] - 0.01:
                    T_hi = pose(R, [p.tcp[0], p.tcp[1], float(pl.T_above[2, 3])])
                    rh2 = _ik(arm, T_hi, ru.q, K.TCP_OFFSET, n_random=1)
                    if rh2.ok and (obs is None or seg_clear(obs, arm, r.q, rh2.q, 0.0, margin_arm=NEAR_MARGIN) >= 0.0):
                        ru, T_up = rh2, T_hi
                    else:
                        _count("hi_lift")
                route, clear, name = [pl.q_above], 9.0, "none"
                hover_route = None
                if obs is not None:
                    route, clear, name = plan_route(obs, arm, ru.q, pl.q_above, 0.0, held, rrt=False)
                    if route is None and rrt_left[0] > 0:
                        rrt_left[0] -= 1
                        route = rrt_connect(obs, arm, ru.q, pl.q_above, 0.0, held, budget=CARRY_RRT_S)
                        if route is not None:
                            clear, name = route_clear(obs, arm, ru.q, route, 0.0, held), "rrt"
                    if route is None or clear < 0.0:
                        _count("col_carry")
                        continue
                    hover_route, _, _ = plan_route(obs, arm, q_cur, rh.q, g_open)
                    if hover_route is None:
                        _count("col_hover")
                        continue
                if obs is not None:
                    f6 = front_in_link6(arm, r.q, fit)
                    why = []
                    for lam in LEVEL_LAMS:
                        lv = level_route(obs, arm, ru.q, route, held, f6, lam=lam)
                        if lv is not None:
                            route, clear = lv
                            name += f"+lv{lam:g}"
                            break
                        why.append(LEVEL_LAST["why"])
                    else:
                        name += "-" + "/".join(why)
                if getattr(pl, "high", False):
                    name = "hi" + ("U" if T_up[2, 3] >= pl.T_above[2, 3] - 0.01 else "") + "-" + name
                plan = Plan(arm, fit.category, T, Th, r.q, p, pl, hang, max(carry, pl.T_above[2, 3]), rh.q, T_up)
                plan.q_up, plan.held_l6, plan.carry, plan.carry_clear, plan.route_name = ru.q, held, route, clear, name
                plan.hover_route, plan.q_start = hover_route, np.asarray(q_cur, dtype=float).copy()
                plan.footprint = _fp
                plan.secs = time.monotonic() - t0
                return plan


STAGE_XS = tuple(
    np.round(np.arange(-0.2, 0.201, 0.04), 3)
)
STAGE_YS = -0.26, -0.22, -0.3, -0.18
STAGE_TURNS = (
    False,
    True,
)
STAGE_CLEAR = 0.025
STAGE_BOX_CLEAR = 0.04
STAGE_DROP = 0.008
STAGE_PLAN_BUDGET = 10.0
STAGE_MAX_SPOTS = 4


def moved(fit: PF.ObjFit, R_new: np.ndarray, x: float, y: float) -> PF.ObjFit:
    a = PF.assets()[fit.key]
    t = R_new @ a.target
    n = max(1e-09, float(np.linalg.norm(t[:2])))
    return PF.ObjFit(fit.category, fit.key, float(x), float(y), fit.z, fit.yaw, fit.iou, fit.alt_iou, R_new, t[:2] / n)


def plan_place_table(
    arm: str,
    fit: PF.ObjFit,
    T_grasp: np.ndarray,
    stage: PF.ObjFit,
    q_seed,
    *,
    obs: BoxObstacle | None = None,
    held_l6=None,
    g_open: float = 1.0,
) -> Place | None:
    rel = np.linalg.inv(pose(fit.R, [fit.x, fit.y, fit.z])) @ T_grasp
    T_rel = pose(stage.R, [stage.x, stage.y, stage.z + STAGE_DROP]) @ rel
    R = T_rel[:3, :3]
    low = float(stage.points()[:, 2].min()) + STAGE_DROP
    hang = float(T_rel[2, 3] - low)
    carry = max(K.TABLE_Z + WALL_H + CARRY_CLEAR + hang, T_rel[2, 3])
    T_above = pose(R, [T_rel[0, 3], T_rel[1, 3], carry])
    ra = _ik(
        arm,
        T_above,
        q_seed,
        K.TCP_OFFSET,
        n_random=1,
        extra_seeds=(base_seed(arm, T_above[:2, 3]), base_seed(arm, T_above[:2, 3])),
    )
    if not ra.ok:
        _count("st_ik_above")
        return None
    r = _ik(arm, T_rel, ra.q, K.TCP_OFFSET, n_random=1)
    if not r.ok:
        _count("st_ik_rel")
        return None
    if (
        obs is not None
        and seg_clear(obs, arm, ra.q, r.q, g_open, held_l6, margin_arm=NEAR_MARGIN, table_clear=0.0) < 0.0
    ):
        _count("st_col")
        return None
    pl = Place(T_rel, T_above, T_above, r.q, 0.0, low, K.TABLE_Z)
    pl.q_above = ra.q
    return pl


def plan_stage_iter(
    arm_a: str,
    arm_b: str,
    fit: PF.ObjFit,
    box: BoxFrame,
    q_a,
    q_b,
    placed=(),
    *,
    obs: BoxObstacle | None = None,
    others=(),
    budget: float = 40.0,
    tilts=None,
    max_tilt: float | None = None,
):
    import time
    from scipy.spatial import cKDTree

    t_end = time.monotonic() + budget
    tree = cKDTree(np.vstack([o[:, :2] for o in others])) if len(others) else None
    Rz = facing_rotation(fit, box)
    spots = []
    for turn in STAGE_TURNS:
        R_s = Rz @ fit.R if turn else fit.R
        for y in STAGE_YS:
            for x in STAGE_XS:
                sf = moved(fit, R_s, x, y)
                pts = sf.points()[::6]
                if tree is not None and float(tree.query(pts[:, :2], k=1)[0].min()) < STAGE_CLEAR:
                    _count("st_occupied")
                    continue
                if obs is not None and float(obs.dist(pts).min()) < STAGE_BOX_CLEAR:
                    _count("st_near_box")
                    continue
                spots.append((abs(x) + 0.5 * abs(y + 0.24) + 0.02 * turn, sf))
    spots.sort(key=lambda t: t[0])
    for _, sf in spots[:STAGE_MAX_SPOTS]:
        if time.monotonic() > t_end:
            _count("st_budget")
            return None
        yield None
        left = t_end - time.monotonic()
        plan_b = yield from plan_object_iter(arm_b, sf, box, q_b, placed, obs=obs, budget=min(STAGE_PLAN_BUDGET, left))
        if plan_b is None:
            _count("st_no_b")
            continue
        left = t_end - time.monotonic()
        plan_a = yield from plan_object_iter(
            arm_a,
            fit,
            box,
            q_a,
            placed,
            obs=obs,
            stage=sf,
            budget=min(STAGE_PLAN_BUDGET, left),
            tilts=tilts,
            max_tilt=max_tilt,
        )
        if plan_a is None:
            _count("st_no_a")
            continue
        return plan_a, sf, plan_b


AVOID_R = 0.02


def pinch_offset(fit: PF.ObjFit, tcp) -> np.ndarray:
    return np.asarray(fit.R, dtype=float).T @ (np.asarray(tcp, dtype=float) - np.array([fit.x, fit.y, fit.z]))


def _avoided(p: P.Pinch, fit: PF.ObjFit, avoid) -> bool:
    if not len(avoid):
        return False
    off = pinch_offset(fit, p.tcp)
    return any(float(np.linalg.norm(off - np.asarray(a, dtype=float))) < AVOID_R for a in avoid)


def _r_perp(p: P.Pinch, c_xy) -> float:
    a2 = np.asarray(p.closing[:2], dtype=float)
    n = float(np.linalg.norm(a2))
    if n < 1e-06:
        return 0.0
    r = np.asarray(c_xy, dtype=float) - p.tcp[:2]
    return abs(float(r[0] * a2[1] - r[1] * a2[0])) / n
