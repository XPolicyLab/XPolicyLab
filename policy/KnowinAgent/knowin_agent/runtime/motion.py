"""Motion programs: planned flange moves, gripper commands and holds.

A move is planned before the arm moves. With a kinematics planner the arm is
driven by joint targets (joint space for longer free moves that stay near the
straight line, straight lines for precise and contact moves; each waypoint is
solved from the previous one so the arm never flips IK branch). Without one the
flange steps toward the goal in bounded Cartesian increments.

A joint plan advances to its next waypoint only when the arm has arrived near the
current one, so a lagging arm is waited for instead of being commanded further ahead.
With the gripper closed on something, a move stops as soon as the fingers close
on nothing (the part slipped out).

Each function is a generator used inside a track (or composed with `yield from`
in skills): it yields {arm: ArmCommand} once per tick and returns a result dict
whose `status` is one of reached, blocked (contact stopped it, when allowed),
stalled, dropped (the held part slipped out), path_blocked (nothing moved), timeout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from knowin_agent.core.geometry import Pose, slerp
from knowin_agent.runtime.backend import ArmCommand, ArmState, RobotProfile
from knowin_agent.runtime.executor import ProgramContext

FAST_STEP_M = 0.04     # empty gripper, free space
CARRY_STEP_M = 0.03    # gripper closed: a held part should not swing out
NOT_FOLLOWING_RAD = 0.3
ARRIVED_RAD = 0.2      # a joint plan moves on to its next waypoint once every joint is this close to the current one
LAG_TICKS = 3          # ticks the arm may stay farther than that from a waypoint before the move is given up
EE_NOT_FOLLOWING_M = 0.04
GRIP_MAX_TICKS = 8
GRIP_OPEN_MAX_TICKS = 3  # a release needs the fingers off the part, not fully open
GRIP_STILL = 0.004     # opening change per tick below which the fingers have stopped
GRIP_EMPTY_CLOSED = 0.0
GRIP_HOLD_MARGIN = 0.02
GRIP_SLIP_TICKS = 2    # ticks with the closed fingers at the empty opening before a carried part counts as dropped
GRIP_OPEN = 0.98


@dataclass(frozen=True)
class MotionLimits:
    translation_step_m: Optional[float] = None  # fixed step; None: transit_open_m open, transit_carry_m closed
    transit_open_m: float = FAST_STEP_M
    transit_carry_m: float = CARRY_STEP_M
    rotation_step_rad: float = 0.15
    position_tol_m: float = 0.003
    rotation_tol_rad: float = 0.03
    precision_radius_m: float = 0.02
    precision_step_m: float = 0.003
    precision_tol_m: float = 0.001
    precision_tol_rad: float = 0.015
    settle_ticks: int = 2
    stall_window: int = 5
    stall_progress_m: float = 0.001
    stall_progress_rad: float = 0.005
    max_ticks: int = 150


DEFAULT_LIMITS = MotionLimits()


def limits_with(**overrides) -> MotionLimits:
    return replace(DEFAULT_LIMITS, **{k: v for k, v in overrides.items() if v is not None})


def step_for(limits: MotionLimits, grip: float) -> float:
    if limits.translation_step_m is not None:
        return float(limits.translation_step_m)
    return limits.transit_open_m if grip >= 0.5 else min(limits.transit_carry_m, limits.transit_open_m)


@dataclass
class Plan:
    arm: str
    target: Pose
    how: str                      # joint | line | joint+precise_line | line+precise_line | ee | blocked
    ways: Optional[list] = None   # joint waypoints (None: Cartesian stepping or blocked)
    why: Optional[str] = None
    planned_actions: int = 0
    reach: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.how == "blocked"


def plan_move(profile: RobotProfile, state: ArmState, arm: str, target: Pose, *, grip: float,
              limits: MotionLimits = DEFAULT_LIMITS, precision: bool = False, straight: bool = False,
              path: str = "auto", reach=None) -> Plan:
    """Plan before moving. path: auto | line | joint | ee (ee = simulator IK through flange poses; with a
    kinematics planner it is tried as a straight line first, since simulator-IK moves stalled in about half of
    the recorded cases)."""
    step_m = step_for(limits, grip)
    kin = profile.kinematics
    if kin is not None and hasattr(kin, "plan"):
        reach = reach or kin.reach(arm, target, state.joints)
        if reach.reachable is True:
            ways, how = kin.plan(arm, state.joints, target, reach.joint_solution, step_m=step_m,
                                 rot_step=limits.rotation_step_rad, precision=precision,
                                 path="line" if path == "ee" else path, straight=straight)
            if ways is not None:
                return Plan(arm, target, how, ways=ways, planned_actions=len(ways), reach=reach.as_dict())
            if path != "ee":
                return Plan(arm, target, "blocked", why=how, reach=reach.as_dict())
    dist, angle = state.ee_pose.errors_to(target)
    n = math.ceil(max(dist / step_m, angle / limits.rotation_step_rad, 1e-9))
    if precision:
        n += math.ceil(min(dist, limits.precision_radius_m) / limits.precision_step_m)
    return Plan(arm, target, "ee", planned_actions=int(n), reach=reach.as_dict() if reach is not None else {})


def _result(status: str, arm: str, target: Pose, current: Pose, ticks: int, plan: Optional[Plan] = None, **extra) -> dict:
    dist, angle = current.errors_to(target)
    out = {"status": status, "arm": arm, "ticks": ticks, "position_error_mm": round(dist * 1000, 2),
           "rotation_error_deg": round(math.degrees(angle), 2), "reached_xyz": [round(v, 4) for v in current.xyz]}
    if plan is not None:
        out.update(path=plan.how, planned_actions=plan.planned_actions)
    out.update(extra)
    return out


def follow(ctx: ProgramContext, plan: Plan, *, gripper: Optional[float] = None,
           limits: MotionLimits = DEFAULT_LIMITS, precision: bool = False, stop_on_contact: bool = False,
           max_ticks: Optional[int] = None):
    """Execute a plan. Generator; returns a result dict."""
    arm, target = plan.arm, plan.target
    current = ctx.obs.arms[arm].ee_pose
    if plan.blocked:
        return _result("path_blocked", arm, target, current, 0, plan, reason=plan.why)
    grip = ctx.hold(arm).gripper if gripper is None else float(gripper)
    pos_tol = limits.precision_tol_m if precision else limits.position_tol_m
    rot_tol = limits.precision_tol_rad if precision else limits.rotation_tol_rad
    budget = max_ticks or max(limits.max_ticks, plan.planned_actions + 3 * limits.stall_window)
    history: list[tuple[float, float]] = []
    settled_at: Optional[int] = None
    off_track = 0
    step_m = step_for(limits, grip)
    way, lagging = 0, 0
    last_way = len(plan.ways) - 1 if plan.ways is not None else 0
    opening = ctx.obs.arms[arm].gripper_opening
    # Carrying: the gripper was already commanded closed before this move and its fingers stopped on something
    # (a move that itself closes an open gripper is not carrying anything yet).
    carrying = (grip < 0.5 and ctx.hold(arm).gripper < 0.5 and opening is not None
                and GRIP_EMPTY_CLOSED + GRIP_HOLD_MARGIN < opening < GRIP_OPEN - 0.1)
    slipped = 0
    for index in range(budget):
        if plan.ways is not None:
            q = plan.ways[way]
            command = ArmCommand(None, grip, joints=q)
        else:
            delta = target.position - current.position
            gap = float(np.linalg.norm(delta))
            step = limits.precision_step_m if precision and gap <= limits.precision_radius_m else min(step_m, 0.02)
            xyz = current.position + delta * min(1.0, step / max(gap, 1e-12))
            _, angle = current.errors_to(target)
            quat = slerp(current.quat, target.quat, min(1.0, limits.rotation_step_rad / max(angle, 1e-12)))
            if settled_at is not None:
                xyz, quat = target.position, np.array(target.quat)
            command = ArmCommand(Pose.make(xyz, quat), grip)
        yield {arm: command}
        state = ctx.obs.arms[arm]
        current = state.ee_pose
        dist, angle = current.errors_to(target)
        history.append((dist, angle))
        if carrying and state.gripper_opening is not None:
            slipped = slipped + 1 if state.gripper_opening <= GRIP_EMPTY_CLOSED + GRIP_HOLD_MARGIN / 2 else 0
            if slipped >= GRIP_SLIP_TICKS:
                return _result("dropped", arm, target, current, index + 1, plan,
                               reason="the closed fingers reached the empty opening: the held part slipped out",
                               opening=round(float(state.gripper_opening), 4))
        if settled_at is not None:
            if index - settled_at >= limits.settle_ticks:
                return _result("reached", arm, target, current, index + 1, plan)
            continue
        if dist <= pos_tol and angle <= rot_tol and (plan.ways is None or way >= last_way):
            if precision or stop_on_contact:
                settled_at = index
                continue
            return _result("reached", arm, target, current, index + 1, plan)
        if plan.ways is not None:
            lag = float(np.max(np.abs(np.asarray(state.joints, dtype=float) - command.joints)))
            if lag <= ARRIVED_RAD:
                way, lagging = min(way + 1, last_way), 0
            else:
                lagging += 1
            lost = lagging >= LAG_TICKS
        else:
            lost = float(np.linalg.norm(current.position - command.ee_pose.position)) > EE_NOT_FOLLOWING_M
        off_track = off_track + 1 if lost else 0
        if off_track >= (1 if plan.ways is not None else 2):
            status = "blocked" if stop_on_contact and dist <= 0.05 else "stalled"
            return _result(status, arm, target, current, index + 1, plan, reason="arm_not_following")
        window = limits.stall_window
        past_end = plan.ways is None or way >= last_way
        if len(history) > window and past_end:
            past_dist, past_angle = history[-window - 1]
            if past_dist - dist < limits.stall_progress_m and past_angle - angle < limits.stall_progress_rad:
                status = "blocked" if stop_on_contact else "stalled"
                return _result(status, arm, target, current, index + 1, plan)
    return _result("timeout", arm, target, current, budget, plan)


def move_flange(ctx: ProgramContext, arm: str, target: Pose, *, gripper: Optional[float] = None,
                limits: MotionLimits = DEFAULT_LIMITS, precision: bool = False, stop_on_contact: bool = False,
                max_ticks: Optional[int] = None, path: str = "auto"):
    """Plan from the current state, then follow. Generator; returns a result dict."""
    grip = ctx.hold(arm).gripper if gripper is None else float(gripper)
    plan = plan_move(ctx.profile, ctx.obs.arms[arm], arm, target, grip=grip, limits=limits, precision=precision,
                     straight=stop_on_contact, path=path)
    return (yield from follow(ctx, plan, gripper=grip, limits=limits, precision=precision,
                              stop_on_contact=stop_on_contact, max_ticks=max_ticks))


def set_gripper(ctx: ProgramContext, arm: str, value: float, *, settle_ticks: Optional[int] = None,
                until_still: bool = True):
    """Command the gripper while holding the arm, until the fingers stop (measured) or settle_ticks.

    After a close, `holding` is 'yes' when the fingers stopped on something, 'no' when they closed on nothing,
    'unknown' when the robot does not report the opening."""
    hold = ctx.hold(arm)
    start = ctx.obs.arms[arm].gripper_opening
    measured = start is not None
    if until_still and measured and value >= GRIP_OPEN and start >= GRIP_OPEN and hold.gripper >= GRIP_OPEN:
        return {"status": "done", "arm": arm, "commanded": float(value), "opening": round(start, 4), "holding": "no",
                "ticks": 0}
    limit = settle_ticks or (GRIP_OPEN_MAX_TICKS if value >= 0.5 else GRIP_MAX_TICKS if measured else 4)
    openings = [start]
    ticks = 0
    for index in range(limit):
        yield {arm: ArmCommand(hold.ee_pose, float(value), joints=hold.joints)}
        ticks += 1
        openings.append(ctx.obs.arms[arm].gripper_opening)
        if until_still and measured and index >= 1 and abs(openings[-1] - openings[-2]) < GRIP_STILL:
            break
    final = openings[-1]
    if value < 0.5:
        holding = "unknown" if final is None else ("yes" if final > GRIP_EMPTY_CLOSED + GRIP_HOLD_MARGIN else "no")
    else:
        holding = "no"
    return {"status": "done", "arm": arm, "commanded": float(value),
            "opening": "unknown" if final is None else round(final, 4), "holding": holding, "ticks": ticks}


def hold(ctx: ProgramContext, arms: tuple, ticks: int):
    """Keep `arms` at their hold commands for a number of ticks."""
    for _ in range(max(0, ticks)):
        yield {arm: ctx.hold(arm) for arm in arms}
    return {"status": "done", "ticks": ticks}


def move_both(ctx: ProgramContext, targets: dict, *, limits: MotionLimits = DEFAULT_LIMITS,
              precision: bool = False, max_ticks: Optional[int] = None):
    """Synchronised two-arm move: both arms advance every tick; ends when both finish."""
    programs = {arm: move_flange(ctx, arm, pose, limits=limits, precision=precision, max_ticks=max_ticks)
                for arm, pose in targets.items()}
    pending, results = {}, {}
    for arm, gen in list(programs.items()):
        try:
            pending[arm] = next(gen)
        except StopIteration as stop:
            results[arm] = stop.value
            programs.pop(arm)
    while programs:
        merged = {}
        for commands in pending.values():
            merged.update(commands)
        yield merged
        for arm in list(programs):
            try:
                pending[arm] = programs[arm].send(None)
            except StopIteration as stop:
                results[arm] = stop.value
                programs.pop(arm)
                pending.pop(arm)
    ok = all(r.get("status") == "reached" for r in results.values())
    return {"status": "reached" if ok else "partial", "arms": results}
