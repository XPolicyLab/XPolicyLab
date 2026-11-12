from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from ..core import actions
from ..planning import collision as CO
from ..planning import kinematics as K
from ..planning import motion

MIN_ARM_DIST = 0.03
FAST_JOINT_STEP = 0.08
MAX_WAIT = 150
IDLE_WAIT = 8
BACK_OFF_MARGIN = 0.02
BACK_OFF_MAX = 3
BACK_OFF_WAIT = 80


class Zone:
    def __init__(self, xy, radius: float):
        self.xy = np.asarray(xy, dtype=float)[:2]
        self.radius = float(radius)
        self.owner: str | None = None

    def inside(self, p) -> bool:
        return float(np.linalg.norm(np.asarray(p)[:2] - self.xy)) < self.radius

    def take(self, name: str) -> bool:
        if self.owner in (None, name):
            self.owner = name
            return True
        return False

    def free(self, name: str) -> None:
        if self.owner == name:
            self.owner = None


class Arm:
    yield_empty_first = False
    back_off_steps = 0

    def __init__(self, name: str, ctx, q0, g0: float):
        self.name = name
        self.ctx = ctx
        self.q = np.asarray(q0, dtype=float).copy()
        self.g = float(g0)
        self.q_home = self.q.copy()
        self.status = "idle"
        self.trail: list = []
        self.blocked = False
        self.held = None
        self.other: "Arm | None" = None
        self.zone: Zone | None = None
        self.waited = 0
        self.active = False
        self.waiting = False
        self.waiting_zone = False
        self.idle_guard: float | None = None
        self.ik_limits = None

    @property
    def T(self) -> np.ndarray:
        return K.fk(self.name, self.q)

    def tcp(self) -> np.ndarray:
        return K.point_of(self.T, K.TCP_OFFSET)

    def owns_zone(self) -> bool:
        return self.zone is not None and self.zone.owner == self.name

    def _dist(self, q_self) -> float:
        o = self.other
        if self.name == "left":
            return CO.arm_distance(q_self, o.q, self.g, o.g)
        return CO.arm_distance(o.q, q_self, o.g, self.g)

    def _may_step(self, q_next, guard: float = MIN_ARM_DIST) -> bool:
        z = self.zone
        if z is not None and z.owner not in (None, self.name):
            p_next = K.point_of(K.fk(self.name, q_next), K.TCP_OFFSET)
            d_next = np.linalg.norm(p_next[:2] - z.xy)
            if z.inside(p_next) and d_next < np.linalg.norm(self.tcp()[:2] - z.xy) - 0.0001:
                return False
        if self.other is None:
            return True
        if self.idle_guard is not None and (not self.other.active):
            guard = min(guard, self.idle_guard)
        d_next = self._dist(q_next)
        return d_next >= guard or d_next >= self._dist(self.q)

    def _after_step(self) -> None:
        pass

    def follow(self, plan: motion.Plan, grip=None, force: bool = False, guard: float = MIN_ARM_DIST) -> Iterator[None]:
        n = len(plan.q)
        g0 = self.g
        backs = 0
        for i, q in enumerate(plan.q):
            q = np.asarray(q, dtype=float)
            w = 0
            while not self._may_step(q, guard):
                w += 1
                self.waited += 1
                self.waiting = True
                self.blocked = True
                o = self.other
                idle = o is None or not o.active
                limit = MAX_WAIT
                if idle:
                    limit = IDLE_WAIT
                elif o.waiting:
                    if self.yield_empty_first and (self.held is None) != (o.held is None):
                        yields = self.held is None
                    else:
                        yields = not self.owns_zone() and (o.owns_zone() or self.name == "right")
                    limit = IDLE_WAIT if yields else 3 * IDLE_WAIT
                if (
                    w >= limit
                    and self.back_off_steps
                    and (not idle)
                    and o.blocked
                    and self.trail
                    and (backs < BACK_OFF_MAX)
                ):
                    backs += 1
                    self.waiting = self.blocked = False
                    ok = yield from self._back_off(q, guard)
                    if ok:
                        w = 0
                        continue
                    self.waiting = self.blocked = True
                if w >= limit:
                    self.waiting = self.blocked = False
                    if force:
                        break
                    self.status = "blocked"
                    return False
                yield
            self.waiting = self.blocked = False
            if self.back_off_steps:
                self.trail.append(self.q.copy())
                del self.trail[: -self.back_off_steps]
            self.q = q
            if grip is not None:
                self.g = g0 + (grip - g0) * min(1.0, (i + 1) / max(1, min(n, 6)))
            self._after_step()
            yield
        return True

    def _back_off(self, q_goal, guard: float) -> Iterator[None]:
        ahead: list = []
        stuck = 0
        ok = False
        for t in range(BACK_OFF_WAIT):
            o = self.other
            if o is not None and o.active and o.blocked:
                if self.trail and self._may_step(self.trail[-1], guard):
                    ahead.append(self.q.copy())
                    self.q = self.trail.pop()
                    stuck = 0
                    self._after_step()
                    yield
                    continue
                stuck += 1
                if stuck > IDLE_WAIT:
                    break
            elif self._dist(q_goal) >= guard + BACK_OFF_MARGIN:
                if not ahead:
                    ok = True
                    break
                if self._may_step(ahead[-1], guard):
                    self.trail.append(self.q.copy())
                    self.q = ahead.pop()
                    self._after_step()
                    yield
                    continue
            elif o is None or not o.active:
                break
            yield
        return ok

    def hold(self, n: int) -> Iterator[None]:
        for _ in range(max(0, n)):
            yield

    def set_grip(self, value: float, n: int = 8) -> Iterator[None]:
        g0 = self.g
        ramp = max(1, n // 2)
        if self.back_off_steps:
            self.trail = []
        for i in range(n):
            self.g = g0 + (value - g0) * min(1.0, (i + 1) / ramp)
            yield

    def line_to(
        self,
        T_goal: np.ndarray,
        offset: float = K.TCP_OFFSET,
        *,
        lin_step: float = motion.MAX_LIN_STEP,
        ang_step: float = motion.MAX_ANG_STEP,
        joint_step: float = motion.MAX_JOINT_STEP,
        grip=None,
        fallback_joint: bool = True,
    ) -> Iterator[None]:
        plan = motion.line(
            self.name,
            self.q,
            T_goal,
            offset,
            lin_step=lin_step,
            ang_step=ang_step,
            joint_step=joint_step,
            limits=self.ik_limits,
        )
        if not plan.ok:
            if not fallback_joint:
                self.status = "fail"
                return
            r = K.solve_near(self.name, T_goal, self.q, offset, limits=self.ik_limits)
            if not r.ok:
                self.status = "fail"
                return
            plan = motion.joint_move(self.q, r.q, joint_step=joint_step)
        ok = yield from self.follow(plan, grip)
        self.status = "ok" if ok else "blocked"

    def joints_to(
        self, q_goal, grip=None, joint_step: float = FAST_JOINT_STEP, force: bool = False, guard: float = MIN_ARM_DIST
    ) -> Iterator[None]:
        ok = yield from self.follow(
            motion.joint_move(self.q, q_goal, joint_step=joint_step), grip, force=force, guard=guard
        )
        self.status = "ok" if ok else "blocked"

    def move_to(
        self,
        T_goal: np.ndarray,
        *,
        min_z: float,
        offset: float = K.TCP_OFFSET,
        grip=None,
        joint_step: float = FAST_JOINT_STEP,
        lin_step: float = 0.016,
        ang_step: float = math.radians(6),
        seeds=(),
    ) -> Iterator[None]:
        r = K.solve_near(self.name, T_goal, self.q, offset, extra_seeds=seeds, n_random=3, limits=self.ik_limits)
        if r.ok:
            plan = motion.joint_move(self.q, r.q, joint_step=joint_step)
            zs = [K.point_of(K.fk(self.name, q), offset)[2] for q in plan.q[:: max(1, len(plan.q) // 8)]]
            if min(zs + [10.0]) >= min_z:
                ok = yield from self.follow(plan, grip)
                self.status = "ok" if ok else "blocked"
                return
        yield from self.line_to(T_goal, offset, lin_step=lin_step, ang_step=ang_step, joint_step=joint_step, grip=grip)

    def wait_until(self, cond, max_steps: int = 10**6) -> Iterator[None]:
        n = 0
        while not cond() and n < max_steps:
            n += 1
            self.waiting = True
            yield
        self.waiting = False

    def transit(
        self,
        T_goal: np.ndarray,
        z_safe: float,
        *,
        offset: float = K.TCP_OFFSET,
        grip=None,
        seeds=(),
        joint_step: float = FAST_JOINT_STEP,
        up_step: float = 0.014,
    ) -> Iterator[None]:
        p0 = K.point_of(self.T, offset)
        pg = T_goal[:3, 3]
        z_cross = max(z_safe, pg[2])
        if p0[2] < z_safe - 0.003:
            up = self.T.copy()
            up[:3, 3] = [p0[0], p0[1], z_safe]
            yield from self.line_to(up, offset, lin_step=up_step, grip=grip)
            if self.status != "ok":
                return
        T_cross = T_goal.copy()
        T_cross[:3, 3] = [pg[0], pg[1], z_cross]
        yield from self.move_to(
            T_cross, min_z=z_safe - 0.005, offset=offset, grip=grip, seeds=seeds, joint_step=joint_step
        )
        if self.status != "ok":
            return
        if z_cross > pg[2] + 0.002:
            yield from self.line_to(T_goal, offset, lin_step=0.012)


def run_arms(arms: dict[str, Arm], tasks: dict[str, Iterator[None]]) -> Iterator[list[dict]]:
    live = dict(tasks)
    for a in arms.values():
        a.active = a.name in live
        a.waiting = a.waiting_zone = False
    while live:
        for name in list(live):
            try:
                next(live[name])
            except StopIteration:
                del live[name]
                arms[name].active = False
                arms[name].waiting = arms[name].waiting_zone = False
        yield [actions.joint_action(arms["left"].q, arms["right"].q, arms["left"].g, arms["right"].g)]


def link_arms(arms: dict[str, Arm], zone: Zone | None = None) -> None:
    arms["left"].other = arms["right"]
    arms["right"].other = arms["left"]
    for a in arms.values():
        a.zone = zone


def tcp_pose(rot: np.ndarray, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = p
    return t


def wrap(a: float) -> float:
    return math.remainder(a, 2 * math.pi)
