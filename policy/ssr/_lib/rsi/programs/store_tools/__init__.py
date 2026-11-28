from __future__ import annotations

import math

import numpy as np

from ...perception import cameras as C
from . import toolbox as TB
from ...planning import kinematics as K
from ...planning import motion
from ...skills import dual
from ...skills import grasp as G
from ...skills.runner import Arm, Zone, link_arms, run_arms, tcp_pose
from .. import Program, register

LOOK_Q = np.array([0.0, 0.0, 0.3, 0.6, 0.0, 0.0])
ORDER = {"left": ("hammer", "wrench"), "right": ("pliers", "tape_measure")}
Z_HOVER = 0.84
Z_CARRY = 0.86
Z_CARRY_TOOL = {"wrench": 0.845}
TIP_CLEAR = 0.002
GRASP_DY = {"hammer": 0.0043, "wrench": 0.0136, "pliers": 0.003, "tape_measure": 0.0014}
GRASP_DY_TILT = {"wrench": 0.00125 / math.radians(25.0)}
PLACE_DY = {"hammer": -0.0003, "tape_measure": -0.0007}
PLACE_DY_TILTED = {"hammer": -0.0006}
TILTED_MIN = math.radians(2.5)
PLACE_DX: dict[str, float] = {}
RELEASE_GAP = 0.01
OPEN_MARGIN = {"hammer": 0.024, "wrench": 0.024, "pliers": 0.016, "tape_measure": 0.016}
RELEASE_DZ = 0.004
RELEASE_DZ_TOOL = {"tape_measure": (0.0175, 0.00012 / math.radians(1.0))}
ZONE_R = 0.15
LIFT_STEP = 0.005
RELAY_TOOLS = ("wrench",)
MAX_THIN_TILT = math.radians(10.0)
TOOLBOX_HALF = 0.139
RELAY_GAP = 0.03
CARRY_JOINT_STEP = 0.04
CARRY_ACC = 0.6
CONTROL_HZ = 25.0
CARRY_ACC_TOOL = {"hammer": 0.3}
CARRY_SLOW_FACTOR = 0.6
CARRY_LIN_STEP = 0.012
TILTS = tuple(math.radians(t) for t in (0, 5, 10, 15, 20, 25, 30, 35, 40, 50))
CHECK_ORDER = 3, 4, 0, 1, 2


def tilted(yaw: float, tilt: float) -> np.ndarray:
    r = G.grasp_rotation(yaw, tilt)
    if r[1, 0] < -1e-09:
        r = G.grasp_rotation(yaw, -tilt)
    return r


def plan_rot(arm: str, pts_fn, q_cur):
    for tilt in TILTS:
        pts = pts_fn(tilt)
        best = None
        for yaw in (0.0, math.pi):
            rot = tilted(yaw, tilt)
            q = np.asarray(q_cur, dtype=float)
            qs, cost = [], 0.0
            for i in CHECK_ORDER:
                p = pts[i]
                seeds = [G.base_seed(arm, p[:2])]
                r = K.solve_near(arm, tcp_pose(rot, p), q, K.TCP_OFFSET, extra_seeds=seeds, n_random=2 if qs else 3)
                if not r.ok:
                    qs = None
                    break
                cost += float(np.max(np.abs(r.q - q)))
                q = r.q
                qs.append(r.q)
            if qs is not None and (best is None or cost < best[0]):
                best = cost, rot, qs
        if best is not None:
            return best[1], tilt, pts, best[2]


def minjerk(n: int) -> np.ndarray:
    t = np.arange(1, n + 1) / n
    return t**3 * (10.0 - 15.0 * t + 6.0 * t * t)


def carry(
    arm: Arm,
    T_goal: np.ndarray,
    *,
    min_z: float,
    grip=None,
    peak_step: float = CARRY_JOINT_STEP,
    acc: float = CARRY_ACC,
):
    r = K.solve_near(arm.name, T_goal, arm.q, K.TCP_OFFSET, n_random=3)
    if r.ok:
        q0 = np.asarray(arm.q, dtype=float)
        dq = np.asarray(r.q, dtype=float) - q0
        d = float(np.linalg.norm(T_goal[:3, 3] - arm.tcp()))
        n = max(
            1,
            math.ceil(1.875 * float(np.max(np.abs(dq))) / peak_step),
            math.ceil(CONTROL_HZ * math.sqrt(5.77 * d / acc)),
        )
        plan = motion.Plan([q0 + dq * float(s) for s in minjerk(n)])
        zs = [K.point_of(K.fk(arm.name, q), K.TCP_OFFSET)[2] for q in plan.q[:: max(1, n // 8)]]
        if min(zs) >= min_z:
            ok = yield from arm.follow(plan, grip)
            arm.status = "ok" if ok else "blocked"
            return
    yield from arm.move_to(T_goal, min_z=min_z, grip=grip, joint_step=peak_step)


def carry_line(arm: Arm, T_goal: np.ndarray, *, grip=None, peak_step: float = CARRY_LIN_STEP):
    p0, p1 = arm.tcp(), T_goal[:3, 3]
    d = float(np.linalg.norm(p1 - p0))
    n = max(1, math.ceil(1.875 * d / peak_step), math.ceil(CONTROL_HZ * math.sqrt(5.77 * d / CARRY_ACC)))
    q = np.asarray(arm.q, dtype=float)
    qs = []
    for k, sk in enumerate(minjerk(n)):
        way = T_goal.copy()
        way[:3, 3] = p0 + (p1 - p0) * float(sk)
        r = K.solve(arm.name, way, q, K.TCP_OFFSET, iters=60)
        if not r.ok and k == n - 1 or r.pos_err > 0.004 or float(np.max(np.abs(r.q - q))) > 0.3:
            yield from arm.move_to(T_goal, min_z=p1[2] - 0.03, grip=grip, joint_step=CARRY_JOINT_STEP)
            return
        q = r.q
        qs.append(q.copy())
    ok = yield from arm.follow(motion.Plan(qs), grip)
    arm.status = "ok" if ok else "blocked"


def settle(arm: Arm, tol: float = 0.01, max_steps: int = 10):
    for _ in range(max_steps):
        q_obs = np.asarray(arm.ctx.obs.state[f"{arm.name}_arm_joint_state"], dtype=float)
        if float(np.max(np.abs(q_obs - arm.q))) < tol:
            return
        yield


def plan_move(arm_name: str, q, name: str, tool: TB.Tool, target, in_slot: bool = True):
    dx = TB.TOOL_PRIOR[name].grasp[0]
    pdy = PLACE_DY.get(name, 0.0) if in_slot else 0.0
    pdx = PLACE_DX.get(name, 0.0) if in_slot else 0.0

    def rows(tilt):
        dy = GRASP_DY[name] + GRASP_DY_TILT.get(name, 0.0) * tilt
        pdy_t = PLACE_DY_TILTED.get(name, 0.0) if in_slot and tilt >= TILTED_MIN else 0.0
        return tool.x + dx, tool.y + dy, target[0] + dx + pdx, target[1] + dy + pdy + pdy_t

    def pts_fn(tilt):
        gx, gy, px, py = rows(tilt)
        gz = K.TABLE_Z + G.FINGER_DEPTH * math.cos(tilt) + G.FINGER_HALF_WIDTH * math.sin(tilt) + TIP_CLEAR
        rdz = RELEASE_DZ
        if in_slot and name in RELEASE_DZ_TOOL:
            r0, r1 = RELEASE_DZ_TOOL[name]
            rdz = max(RELEASE_DZ, r0 + r1 * tilt)
        pz = target[2] + (gz - tool.z) + rdz
        zc = Z_CARRY_TOOL.get(name, Z_CARRY)
        return [
            np.array(p, dtype=float)
            for p in ((gx, gy, Z_HOVER), (gx, gy, gz), (gx, gy, zc), (px, py, zc), (px, py, pz))
        ]

    return plan_rot(arm_name, pts_fn, q), rows


def relay_spot(box: TB.Toolbox, name: str, tool: TB.Tool) -> np.ndarray:
    slot = box.slot(name)
    return np.array([slot[0], box.y - TOOLBOX_HALF - TB.TOOL_PRIOR[name].size[1] / 2 - RELAY_GAP, tool.z])


def choose_relay(arms, name: str, tool: TB.Tool, box: TB.Toolbox):
    a = next(x for x in K.ARMS if name in ORDER[x])
    b = next(x for x in K.ARMS if x != a)
    direct, _ = plan_move(a, arms[a].q, name, tool, box.slot(name))
    t_direct = direct[1] if direct is not None else math.inf
    info = {"tilt_direct": math.degrees(t_direct) if direct is not None else None}
    if t_direct <= MAX_THIN_TILT + 1e-06:
        return None, info
    spot = relay_spot(box, name, tool)
    there, _ = plan_move(a, arms[a].q, name, tool, spot, in_slot=False)
    rtool = TB.Tool(name, float(spot[0]), float(spot[1]), tool.z, tool.yaw, tool.score, tool.n_px)
    on, _ = plan_move(b, arms[b].q, name, rtool, box.slot(name))
    info.update(
        tilt_there=None if there is None else math.degrees(there[1]),
        tilt_on=None if on is None else math.degrees(on[1]),
        spot=spot[:2].tolist(),
    )
    if there is None or on is None or there[1] > MAX_THIN_TILT + 1e-06 or (on[1] > t_direct + 1e-06):
        return None, info
    return (spot, rtool), info


def store(ctx, arm: Arm, name: str, tool: TB.Tool, box: TB.Toolbox, zone: Zone, relay_to=None):
    pr = TB.TOOL_PRIOR[name]
    slot = box.slot(name) if relay_to is None else np.asarray(relay_to, dtype=float)
    plan, rows = plan_move(arm.name, arm.q, name, tool, slot, in_slot=relay_to is None)
    if plan is None:
        _, _, px, py = rows(0.0)
        return False
    rot, tilt, pts, _ = plan
    _, _, px, py = rows(tilt)
    g_open = dual.grip_for_gap(pr.width + OPEN_MARGIN[name])
    T = lambda p: tcp_pose(rot, p)
    yield from arm.transit(T(pts[0]), Z_HOVER, grip=g_open)
    yield from arm.line_to(T(pts[1]), lin_step=0.008, grip=g_open)
    yield from settle(arm)
    yield from arm.set_grip(0.0, n=10)
    yield from arm.hold(4)
    if relay_to is None:
        yield from arm.wait_until(lambda: zone.take(arm.name), max_steps=400)
    yield from arm.line_to(T(pts[2]), lin_step=LIFT_STEP, grip=0.0)
    if relay_to is None:
        yield from carry(
            arm,
            T(pts[3]),
            min_z=pts[3][2] - 0.03,
            grip=0.0,
            acc=CARRY_ACC_TOOL.get(name, CARRY_ACC),
            peak_step=CARRY_JOINT_STEP * (CARRY_SLOW_FACTOR if name in CARRY_ACC_TOOL else 1.0),
        )
    else:
        yield from carry_line(arm, T(pts[3]), grip=0.0)
    yield from arm.line_to(T(pts[4]), lin_step=0.006, grip=0.0)
    yield from settle(arm)
    w3 = dual.held_width(float(ctx.obs.state[f"{arm.name}_ee_joint_state"][0]), 0.0)
    g_rel = dual.grip_for_gap(w3 + RELEASE_GAP) if w3 > 0 else g_open
    yield from arm.set_grip(g_rel, n=8)
    yield from arm.hold(2)
    yield from arm.line_to(T(np.array([px, py, Z_HOVER])), lin_step=0.01, grip=g_rel)
    return True


@register
class StoreTools(Program):
    name = "store_tools"

    def match(self, instruction, first_obs) -> float:
        s = instruction.lower()
        if "tool" in s and "toolbox" in s:
            return 0.9
        return 0.0

    def look(self, ctx, arms):
        yield from run_arms(arms, {a: arms[a].joints_to(LOOK_Q, force=True) for a in K.ARMS})
        yield from run_arms(arms, {a: settle(arms[a], 0.005, 20) for a in K.ARMS})
        img = ctx.obs.images.get("cam_head")
        return TB.perceive(img, C.head(), fit=True, debias=True) if img is not None else None

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        sc = yield from self.look(ctx, arms)
        for attempt in range(2):
            ok = sc is not None and sc.toolbox is not None and (len(sc.tools) == len(TB.TOOLS))
            if ok or attempt:
                break
            yield from run_arms(arms, {a: arms[a].hold(5) for a in K.ARMS})
            sc = yield from self.look(ctx, arms)
        zone = Zone((sc.toolbox.x, sc.toolbox.y) if sc is not None and sc.toolbox is not None else (0.0, 0.0), ZONE_R)
        link_arms(arms, zone)
        jobs = {a: [] for a in K.ARMS}
        relay = {"state": "none"}
        if sc is not None and sc.toolbox is not None:
            jobs = {a: [(n, sc.tools[n], None, False) for n in ORDER[a] if n in sc.tools] for a in K.ARMS}
            for name in RELAY_TOOLS:
                if name not in sc.tools:
                    continue
                r, _ = choose_relay(arms, name, sc.tools[name], sc.toolbox)
                if r is None:
                    continue
                spot, rtool = r
                a = next(x for x in K.ARMS if name in ORDER[x])
                b = next(x for x in K.ARMS if x != a)
                jobs[a] = [(name, sc.tools[name], spot, False)] + [j for j in jobs[a] if j[0] != name]
                jobs[b].insert(min(1, len(jobs[b])), (name, rtool, None, True))
                relay["state"] = "pending"

        def arm_task(a):
            arm = arms[a]
            todo = jobs[a]
            for i, (name, tool, relay_to, after_relay) in enumerate(todo):
                if after_relay:
                    yield from arm.wait_until(lambda: relay["state"] != "pending", max_steps=500)
                    if relay["state"] != "ready":
                        continue
                ok = yield from store(ctx, arm, name, tool, sc.toolbox, zone, relay_to=relay_to)
                zone.free(a)
                if i + 1 < len(todo) and (not todo[i + 1][3]):
                    nn, nt = todo[i + 1][0], todo[i + 1][1]
                    p = np.array([nt.x + TB.TOOL_PRIOR[nn].grasp[0], nt.y + GRASP_DY[nn], Z_HOVER])
                    yield from arm.move_to(tcp_pose(arm.T[:3, :3], p), min_z=Z_HOVER - 0.03)
                else:
                    yield from arm.joints_to(arm.q_home, grip=1.0, force=True)
                zone.free(a)
                if relay_to is not None:
                    relay["state"] = "ready" if ok else "failed"
            yield from arm.joints_to(arm.q_home, grip=1.0, force=True)

        yield from run_arms(arms, {a: arm_task(a) for a in K.ARMS})
        yield from run_arms(arms, {a: arms[a].joints_to(arms[a].q_home, grip=1.0, force=True) for a in K.ARMS})
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
