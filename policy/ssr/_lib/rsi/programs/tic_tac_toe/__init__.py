from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...perception import cameras as C
from ...perception import robot_mask as RM
from . import board as TT
from ...planning import kinematics as K
from ...skills import grasp as G
from ...skills.dual import grip_for_gap, held_width
from ...skills.runner import Arm, link_arms, run_arms, tcp_pose
from .. import Program, register

RING_BELOW_TCP = 0.011
GRASP_TCP_Z = G.tcp_min_z(K.TABLE_Z, 0.0)
PICK_OPEN = grip_for_gap(TT.PIECE_R * 2 + 0.024)
RELEASE_OPEN = grip_for_gap(TT.PIECE_R * 2 + 0.012)
HOVER_DZ = 0.03
LIFT_DZ = 0.04
MIN_Z_FREE = 0.8
MIN_Z_HOLD = 0.82
PLACE_CLEAR = 0.003
MIN_HELD = 0.035
PLACE_TILTS = 0.0, math.radians(15.0), math.radians(30.0)
GRASP_YAWS = 0.0, math.pi
MAX_PICK = 3
FREE_JOINT_STEP = 0.12
HOLD_JOINT_STEP = 0.1
CONFLICT_DIST = 0.18
ZONE_MOVING = 0.005
ZONE_STATIC = 0.001
STATIC_FRAMES = 18
REST_MATCH = 0.003
START_TIMEOUT = 45
OPP_TIMEOUT = 260


def g_obs(arm: Arm) -> float:
    return float(arm.ctx.obs.state[f"{arm.name}_ee_joint_state"][0])


def now(ctx) -> int:
    return int(ctx.obs.step)


@dataclass
class Ring:
    xy: np.ndarray
    state: str = "table"
    arm: str | None = None
    fails: int = 0


@dataclass
class Game:
    board: TT.Board
    rings: list
    zone: np.ndarray
    ref: np.ndarray
    placed: int = 0
    cells_taken: list = field(default_factory=lambda: [False] * 9)
    holding: dict = field(default_factory=dict)
    rest: np.ndarray | None = (
        None
    )


def head_view(ctx, arms, dilate: int = 3):
    img = ctx.obs.images.get("cam_head")
    if img is None:
        return None, None, None
    cam = C.head()
    robot = RM.mask(cam, arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g, dilate=dilate)
    return img, cam, robot


def perceive(ctx, arms, board=None, tag: str = "perceive", ref=None):
    img, cam, robot = head_view(ctx, arms)
    if img is None:
        return None, [], None
    if board is None:
        board = TT.detect_board(img, cam, robot)
    pieces = TT.detect_pieces(img, cam, board, robot)
    fill = TT.cell_fill(img, cam, board, robot, ref=ref) if board is not None else None
    return board, pieces, fill


def reachable_xy(arm: str, xy) -> bool:
    bx, by = K.BASE_POS[arm][:2]
    return math.hypot(xy[0] - bx, xy[1] - by) <= G.MAX_REACH_XY


def pick_grasp(arm: str, ring_xy, q_cur):
    best = None
    for yaw in GRASP_YAWS:
        c = G.grasp_candidates(
            arm,
            np.array([ring_xy[0], ring_xy[1], GRASP_TCP_Z]),
            TT.PIECE_R * 2,
            yaw,
            q_cur,
            z_support=K.TABLE_Z,
            hover_dz=HOVER_DZ,
            tilts=(0.0,),
        )
        c = [g for g in c if abs(math.remainder(g.yaw - yaw, 2 * math.pi)) < 1e-06]
        if c and (best is None or c[0].cost < best.cost):
            best = c[0]
    return best


def place_pose(arm: str, cell_xy, q_cur):
    bx, by = K.BASE_POS[arm][:2]
    to_base = np.array([bx - cell_xy[0], by - cell_xy[1]])
    to_base /= max(1e-09, np.linalg.norm(to_base))
    seed0 = G.base_seed(arm, cell_xy)
    q_cur = np.asarray(q_cur, dtype=float)
    for tilt in PLACE_TILTS:
        best = None
        for k in range(4):
            yaw = k * math.pi / 2
            for sgn in (1.0,) if tilt == 0 else (1.0, -1.0):
                rot = G.grasp_rotation(yaw, sgn * tilt)
                if tilt and float(np.dot(-rot[:2, 0], to_base)) < 0.2:
                    continue
                cz = TT.BOARD_TOP + PLACE_CLEAR + TT.PIECE_R * math.sin(tilt) + TT.PIECE_H / 2 * math.cos(tilt)
                centre = np.array([cell_xy[0], cell_xy[1], cz])
                tcp = centre - RING_BELOW_TCP * rot[:, 0]
                r = None
                for s in (q_cur, seed0, seed0 + np.array([0, 0, 0, 0, 0, yaw])):
                    rr = K.solve(arm, G.pose(rot, tcp), s, K.TCP_OFFSET, iters=80)
                    if rr.ok:
                        r = rr
                        break
                if r is None:
                    continue
                rh = K.solve(arm, G.pose(rot, tcp + [0, 0, HOVER_DZ]), r.q, K.TCP_OFFSET, iters=80)
                if not rh.ok:
                    continue
                cost = float(np.sum(K.JOINT_COST_W * np.abs(rh.q - q_cur)))
                if best is None or cost < best[0]:
                    best = cost, rot, tcp, r.q, rh.q, sgn * tilt
        if best is not None:
            return best[1:]


def cell_cost(arm: str, cell_xy, q_home):
    pl = place_pose(arm, cell_xy, q_home)
    if pl is None:
        return None
    cost = 3.0 * abs(pl[4]) + 0.3 * float(np.sum(K.JOINT_COST_W * np.abs(pl[3] - q_home)))
    return cost, pl


def ring_cost(arm: str, ring: Ring, q_home):
    if not reachable_xy(arm, ring.xy):
        return None
    tcp = K.point_of(K.fk(arm, q_home), K.TCP_OFFSET)
    return float(np.linalg.norm(tcp[:2] - ring.xy))


def pick(arm: Arm, game: Game, ring: Ring, arms):
    for attempt in range(MAX_PICK):
        g = pick_grasp(arm.name, ring.xy, arm.q)
        if g is None:
            break
        p = ring.xy
        hz = min(g.hover_z, GRASP_TCP_Z + HOVER_DZ)
        yield from arm.move_to(
            tcp_pose(g.rot, [p[0], p[1], hz]),
            min_z=MIN_Z_FREE,
            grip=PICK_OPEN,
            joint_step=FREE_JOINT_STEP,
            seeds=(g.q_hover,),
        )
        if arm.status != "ok":
            yield from arm.hold(5)
            continue
        yield from arm.set_grip(PICK_OPEN, 2)
        yield from arm.line_to(tcp_pose(g.rot, g.tcp), lin_step=0.01)
        yield from arm.hold(1)
        yield from arm.set_grip(0.0, 8)
        yield from arm.hold(2)
        w = held_width(g_obs(arm))
        if w >= MIN_HELD:
            yield from arm.line_to(tcp_pose(g.rot, [p[0], p[1], GRASP_TCP_Z + LIFT_DZ]), lin_step=0.012)
            ring.state, ring.arm = "held", arm.name
            game.holding[arm.name] = ring
            return True
        ring.fails += 1
        yield from arm.set_grip(PICK_OPEN, 4)
        yield from arm.line_to(tcp_pose(g.rot, [p[0], p[1], GRASP_TCP_Z + LIFT_DZ]), lin_step=0.012)
        yield from arm.joints_to(arm.q_home, grip=PICK_OPEN, force=True)
        yield from arm.hold(2)
        _, pieces, _ = perceive(arm.ctx, arms, game.board, tag="perceive_retry", ref=game.ref)
        near = [q for q in pieces if np.linalg.norm(q.xy - ring.xy) < 0.04]
        if not near:
            ring.state = "lost"
            return False
        ring.xy = min(near, key=lambda q: float(np.linalg.norm(q.xy - ring.xy))).xy.copy()
    ring.state = "lost" if ring.fails >= MAX_PICK else "table"
    return False


def place(arm: Arm, game: Game, ci: int, pl, result: dict):
    rot_p, tcp_p, _, q_ph, _ = pl
    hover = tcp_p + np.array([0.0, 0.0, HOVER_DZ])
    for _attempt in range(4):
        yield from arm.move_to(
            tcp_pose(rot_p, hover), min_z=min(MIN_Z_HOLD, hover[2] - 0.002), seeds=(q_ph,), joint_step=HOLD_JOINT_STEP
        )
        if arm.status == "ok":
            break
        yield from arm.hold(6)
    if arm.status != "ok":
        result["ok"] = False
        result["why"] = f"move {arm.status}"
        return
    yield from arm.line_to(tcp_pose(rot_p, tcp_p), lin_step=0.008)
    yield from arm.hold(1)
    w = held_width(g_obs(arm))
    yield from arm.set_grip(RELEASE_OPEN, 5)
    ring = game.holding.pop(arm.name, None)
    if ring is not None:
        ring.state = "placed" if w >= MIN_HELD else "lost"
    result["ok"] = w >= MIN_HELD
    result["width"] = round(w, 4)
    yield from arm.line_to(tcp_pose(rot_p, hover), lin_step=0.012)


def go_home(arm: Arm, holding: bool):
    yield from arm.joints_to(
        arm.q_home,
        grip=None if holding else 1.0,
        force=True,
        joint_step=HOLD_JOINT_STEP if holding else FREE_JOINT_STEP,
    )
    if not holding:
        yield from arm.set_grip(1.0, 2)


def placer_task(arm: Arm, game: Game, arms, ring_for_pick, ci: int, result: dict):
    if game.holding.get(arm.name) is None:
        if ring_for_pick is None:
            result["ok"] = False
            result["why"] = "no ring"
            return
        ok = yield from pick(arm, game, ring_for_pick, arms)
        if not ok:
            result["ok"] = False
            result["why"] = "pick"
            yield from go_home(arm, False)
            return
    c = cell_cost(arm.name, game.board.cells()[ci], arm.q)
    if c is None:
        result["ok"] = False
        result["why"] = "place unreachable"
        yield from go_home(arm, True)
        return
    yield from place(arm, game, ci, c[1], result)
    yield from go_home(arm, game.holding.get(arm.name) is not None)


def helper_task(arm: Arm, game: Game, arms, ring: Ring, result: dict):
    ok = yield from pick(arm, game, ring, arms)
    result["ok"] = ok
    yield from go_home(arm, ok)


def merge_taken(old, new):
    return [o if n is None else bool(n) for o, n in zip(old, new)]


def wait_opponent(ctx, arms, game: Game, n_expected: int):
    t0 = now(ctx)
    moved = False
    static = 0
    prev = None
    last_log = -100
    n = None
    peak = 0
    while now(ctx) - t0 < OPP_TIMEOUT:
        img, cam, robot = head_view(ctx, arms)
        if img is not None:
            g = TT.grey(img)
            first = prev is None
            ch = 0.0 if first else TT.zone_change(g, prev, game.zone)
            prev = g
            if not first and ch > ZONE_MOVING:
                moved = True
            if moved and static and (ch >= ZONE_STATIC):
                peak = max(peak, static)
            static = 0 if first or ch >= ZONE_STATIC else static + 1
            at_rest = static >= STATIC_FRAMES
            if not at_rest and game.rest is not None and (static >= 2):
                at_rest = TT.zone_change(g, game.rest, game.zone) < REST_MATCH
            if at_rest and moved:
                fill = TT.cell_fill(img, cam, game.board, robot, ref=game.ref)
                taken = TT.taken_cells(fill)
                n = sum(1 for t in taken if t)
                if n >= n_expected:
                    game.cells_taken = merge_taken(game.cells_taken, taken)
                    if static >= STATIC_FRAMES:
                        game.rest = g
                    return True
            if now(ctx) - last_log >= 20:
                last_log = now(ctx)
            if not moved and now(ctx) - t0 >= START_TIMEOUT:
                return False
        yield from run_arms(arms, {a: arms[a].hold(1) for a in K.ARMS})
    return False


@register
class TicTacToe(Program):
    name = "tic_tac_toe"

    def match(self, instruction: str, first_obs) -> float:
        s = instruction.lower().replace("-", " ")
        return 0.95 if "tic tac toe" in s else 0.0

    def run(self, ctx):
        o = ctx.obs
        arms = {
            a: Arm(a, ctx, o.state[f"{a}_arm_joint_state"], float(o.state[f"{a}_ee_joint_state"][0])) for a in K.ARMS
        }
        link_arms(arms)
        game = None
        for _ in range(4):
            img = ctx.obs.images.get("cam_head")
            board, pieces, fill = perceive(ctx, arms, ref=img)
            if board is not None and pieces and (img is not None):
                rings = [
                    Ring(p.xy.copy())
                    for p in pieces
                    if not board.contains(p.xy, 0.01) and any(reachable_xy(a, p.xy) for a in K.ARMS)
                ]
                if rings:
                    game = Game(board, rings, TT.opponent_zone(C.head(), img.shape), img.copy())
                    game.cells_taken = [False] * 9
                    break
            yield from run_arms(arms, {a: arms[a].hold(3) for a in K.ARMS})
        if game is None:
            while True:
                yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})
        n_opp = 0
        for k in range(12):
            if game.placed >= 5:
                break
            plan = self.plan_round(ctx, arms, game)
            if plan is None:
                break
            placer, ring_p, ci, helper, ring_h = plan
            res_p, res_h = {}, {}
            tasks = {placer: placer_task(arms[placer], game, arms, ring_p, ci, res_p)}
            if helper is not None:
                tasks[helper] = helper_task(arms[helper], game, arms, ring_h, res_h)
            yield from run_arms(arms, tasks)
            if not res_p.get("ok"):
                continue
            game.placed += 1
            game.cells_taken[ci] = True
            if game.placed >= 5 or n_opp >= 4:
                continue
            ok = yield from wait_opponent(ctx, arms, game, sum(1 for t in game.cells_taken if t) + 1)
            if ok:
                n_opp += 1
            else:
                _, _, fill = perceive(ctx, arms, game.board, tag="perceive_after_wait", ref=game.ref)
                if fill is not None:
                    game.cells_taken = merge_taken(game.cells_taken, TT.taken_cells(fill))
        perceive(ctx, arms, game.board, tag="perceive_end", ref=game.ref)
        while True:
            yield from run_arms(arms, {a: arms[a].hold(10) for a in K.ARMS})

    def plan_round(self, ctx, arms, game: Game):
        empty = [i for i, t in enumerate(game.cells_taken) if t is False]
        if not empty:
            return None
        cells = game.board.cells()
        free = [r for r in game.rings if r.state == "table"]
        best = None
        cache = {}

        def cc_of(a, ci):
            if (a, ci) not in cache:
                cache[a, ci] = cell_cost(a, cells[ci], arms[a].q_home)
            return cache[a, ci]

        for a in K.ARMS:
            held = game.holding.get(a)
            if held is None and (not free):
                continue
            q_home = arms[a].q_home
            rc = 0.0
            ring = None
            if held is None:
                opts = [(ring_cost(a, r, q_home), i) for i, r in enumerate(free)]
                opts = [(c, i) for c, i in opts if c is not None]
                if not opts:
                    continue
                rc, i = min(opts)
                ring = free[i]
            other = "right" if a == "left" else "left"
            want_helper = game.placed + 1 < 5 and game.holding.get(other) is None
            rest = [r for r in free if r is not ring]
            h_opts = [(ring_cost(other, r, arms[other].q_home), r) for r in rest] if want_helper else []
            h_opts = [(c, r) for c, r in h_opts if c is not None]
            for ci in empty:
                cc = cc_of(a, ci)
                if cc is None:
                    continue
                cost = cc[0] + rc + (0.0 if held is not None else 5.0)
                ring_h, hc = None, 0.0
                if h_opts:
                    def h_cost(t):
                        d = float(np.linalg.norm(t[1].xy - cells[ci]))
                        return t[0] + 10.0 * max(0.0, CONFLICT_DIST - d)

                    hc, ring_h = min(((h_cost(t), t[1]) for t in h_opts), key=lambda u: u[0])
                if best is None or cost + hc < best[0]:
                    best = cost + hc, a, ring, ci, ring_h
        if best is None:
            return None
        _, placer, ring_p, ci, ring_h = best
        helper = "right" if placer == "left" else "left"
        if ring_h is None:
            helper = None
        return placer, ring_p, ci, helper, ring_h
