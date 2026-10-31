"""Inverse kinematics and paths for one arm.

Solver: Levenberg-Marquardt on the 6-D pose error of the grasp point (position in metres,
orientation as a rotation vector scaled by ``ROT_LENGTH``), with Marquardt's diagonal scaling and
an adaptive damping factor.  Joint ranges are enforced by clipping every step.  Several starting
configurations are tried; among the solutions that reach the target without putting the arm
into the table, the one needing the least (weighted) joint travel wins.

Paths: a straight line of the grasp point (orientation interpolated along the shortest turn),
solved waypoint by waypoint from the previous solution; or a joint-space swing, checked for table
clearance along the way (``swing_curve`` says how far its grasp point leaves the straight line).

Speed: a :class:`Speed` sets how much one action may travel.  Motions start and end at the "end"
limits and may go faster in between (a trapezoid-like profile), so the delicate last centimetres
of a descent run as slowly as ever while long moves across the table take fewer actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..geometry import angle_between, log_so3, slerp
from .chain import ArmChain, slide
from .rig import ROBODOJO, Rig

#: Metres of position error that weigh as much as one radian of orientation error.
ROT_LENGTH = 0.12
#: A pose counts as reached within these.
POS_TOL = 0.002
ROT_TOL = math.radians(1.0)
#: Relative weight of each joint's travel when choosing among solutions (shoulder moves cost more).
TRAVEL_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
#: Table clearance (metres above the table top) for the arm's joints and for the fingertips.
JOINT_CLEARANCE = 0.015
#: (pointing straight down, the prompt's lowest grasp point, 1.5 cm up, leaves the fingertips 2 mm
#: above the table: 1.5 mm keeps that height clear of rounding)
TIP_CLEARANCE = 0.0015
#: Straight-line paths: one waypoint per this much travel / turn.
LINE_STEP_M = 0.02
LINE_STEP_RAD = math.radians(8.0)
#: Consecutive line waypoints may not differ by more than this in any joint (a singularity).
LINE_JUMP_RAD = 0.6


@dataclass(frozen=True)
class Speed:
    """Per-action limits of a motion: at its ends, and in its middle ("cruise").

    ``ramp`` actions at each end of a motion run at the end limits, the ones between at the cruise
    limits - when that takes fewer actions than running all of them at the end limits; a straight
    line by an arm that holds nothing ramps over ``ramp_empty`` actions instead, when that is set.
    Straight lines are solved at waypoints every ``fine_m`` / ``fine_rad``; actions fall between
    them.
    """
    line_m: float = LINE_STEP_M
    line_rad: float = LINE_STEP_RAD
    cruise_m: float = LINE_STEP_M
    cruise_rad: float = LINE_STEP_RAD
    joint: float = 0.15
    joint_cruise: float = 0.15
    ramp: int = 2
    fine_m: float = LINE_STEP_M
    fine_rad: float = LINE_STEP_RAD
    ramp_empty: int | None = None

    @property
    def uniform(self) -> bool:
        """No faster middle, and line waypoints one action apart (every action as in R0)."""
        return (self.cruise_m <= self.line_m and self.cruise_rad <= self.line_rad
                and self.joint_cruise <= self.joint
                and self.fine_m == self.line_m and self.fine_rad == self.line_rad)


#: "v1": every action at the end limits (R0); "v2": faster in the middle of long motions; "v3": v2,
#: with one end action instead of two on straight lines by an empty hand (replayed on R1's line
#: moves: 19 actions an episode fewer than v2, 7 %; untried on the arms).
SPEEDS = {
    "v1": Speed(),
    "v2": Speed(cruise_m=0.04, cruise_rad=math.radians(12.0), joint_cruise=0.22,
                fine_m=0.01, fine_rad=math.radians(4.0)),
}
SPEEDS["v3"] = replace(SPEEDS["v2"], ramp_empty=1)


def speed_named(name: str) -> Speed:
    try:
        return SPEEDS[name]
    except KeyError:
        raise ValueError(f"unknown motion speed {name!r}; known: {', '.join(SPEEDS)}") from None


@dataclass
class Solution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    problems: list[str] = field(default_factory=list)

    @property
    def reached(self) -> bool:
        return self.pos_err <= POS_TOL and self.rot_err <= ROT_TOL

    @property
    def clear(self) -> bool:
        return not self.problems


def pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """[dp (m); rotation vector taking current to target, in the world frame (rad)]."""
    dp = target[:3, 3] - current[:3, 3]
    w = log_so3(target[:3, :3] @ current[:3, :3].T)
    return np.concatenate([dp, w])


def refine(
    chain: ArmChain,
    target: np.ndarray,
    q0,
    *,
    reach: float | None = None,
    iterations: int = 160,
) -> Solution:
    """Levenberg-Marquardt from ``q0`` toward the grasp-point pose ``target`` (``reach``: another
    point along link6's x axis)."""
    scale = np.array([1.0, 1.0, 1.0, ROT_LENGTH, ROT_LENGTH, ROT_LENGTH])
    q = chain.clip(q0)
    jac, pose = chain.jacobian(q, reach)
    err = pose_error(target, pose)
    cost = float(np.sum((scale * err) ** 2))
    damping = 1e-3
    for _ in range(iterations):
        if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < math.radians(0.05):
            break
        jw = jac * scale[:, None]
        ew = err * scale
        h = jw.T @ jw
        g = jw.T @ ew
        step = np.linalg.solve(h + damping * (np.diag(np.diag(h)) + 1e-9 * np.eye(6)), g)
        big = float(np.max(np.abs(step)))
        if big > 0.4:
            step *= 0.4 / big
        trial = chain.clip(q + step)
        trial_jac, trial_pose = chain.jacobian(trial, reach)
        trial_err = pose_error(target, trial_pose)
        trial_cost = float(np.sum((scale * trial_err) ** 2))
        if trial_cost < cost:
            q, jac, err, cost = trial, trial_jac, trial_err, trial_cost
            damping = max(damping / 4.0, 1e-7)
        else:
            damping = min(damping * 6.0, 1e4)
            if damping >= 1e4:
                break
    pos_err = float(np.linalg.norm(err[:3]))
    rot_err = float(np.linalg.norm(err[3:]))
    return Solution(q=q, pos_err=pos_err, rot_err=rot_err, problems=clearance_problems(chain, q))


def clearance_problems(chain: ArmChain, q) -> list[str]:
    """Ways joint angles ``q`` would put the arm into the table (empty when clear)."""
    rig = chain.rig
    frames = chain.frames(q)
    problems = []
    names = ("shoulder", "upper arm", "elbow", "forearm", "wrist", "flange")
    for name, frame in list(zip(names, frames))[1:]:
        if _over_table(frame[:3, 3], rig) and frame[2, 3] < rig.table_z + JOINT_CLEARANCE:
            problems.append(f"{name} below the table top")
    tip = slide(frames[-1], rig.tip_reach)[:3, 3]
    if _over_table(tip, rig) and tip[2] < rig.table_z + TIP_CLEARANCE:
        problems.append("fingertips into the table")
    return problems


def lowest_z(rotation: np.ndarray, rig: Rig = ROBODOJO) -> float:
    """The lowest grasp-point height (to the mm above) at which the fingertips and the flange clear
    the table, with the hand turned by ``rotation`` (its x axis is the pointing direction)."""
    down = -float(rotation[2, 0])  # 1 pointing straight down, 0 level
    tips = rig.table_z + TIP_CLEARANCE + (rig.tip_reach - rig.grasp_reach) * max(down, 0.0)
    flange = rig.table_z + JOINT_CLEARANCE - rig.grasp_reach * down
    return math.ceil(max(tips, flange) * 1000.0 - 1e-6) / 1000.0


def _over_table(p, rig: Rig = ROBODOJO) -> bool:
    return rig.table_x[0] <= p[0] <= rig.table_x[1] and rig.table_y[0] <= p[1] <= rig.table_y[1]


def travel(q_from, q_to) -> float:
    return float(np.sum(TRAVEL_WEIGHTS * np.abs(np.asarray(q_to) - np.asarray(q_from))))


def seeds_for(chain: ArmChain, current, extra=(), count: int = 6) -> list[np.ndarray]:
    """Starting configurations: the current one, its wrist half-turned, extras, perturbations."""
    current = np.asarray(current, float)
    out = [current.copy()]
    for turn in (math.pi, -math.pi):
        q = current.copy()
        q[5] += turn
        if chain.within_range(q):
            out.append(q)
    out.extend(np.asarray(q, float) for q in extra)
    rng = np.random.default_rng(7)
    for _ in range(count):
        out.append(chain.clip(current + rng.normal(0.0, 0.7, 6)))
    return out


def solve(chain: ArmChain, target: np.ndarray, current, extra_seeds=()) -> tuple[Solution, list[Solution]]:
    """The best solution for ``target`` from several seeds, and every solution found.

    Best = reached and clear with the least travel; failing that the closest miss (clear misses
    before ones into the table).
    """
    found = []
    for seed in seeds_for(chain, current, extra_seeds):
        sol = refine(chain, target, seed)
        found.append(sol)
        if sol.reached and sol.clear and travel(current, sol.q) < 1.0:
            break  # an easy nearby solution: no need to search further
    good = [s for s in found if s.reached and s.clear]
    if good:
        return min(good, key=lambda s: travel(current, s.q)), found
    return min(found, key=lambda s: (not s.clear, s.pos_err + ROT_LENGTH * s.rot_err)), found


# -- paths -----------------------------------------------------------------------------------------
@dataclass
class Path:
    waypoints: list[np.ndarray]  # joint configurations, the first is the start
    kind: str  # "line" or "swing"
    note: str = ""
    dist: float = 0.0  # a line's travel (m) and turn (rad), spread evenly over its waypoints
    turn: float = 0.0
    #: a line's travel (m) and turn (rad) from each waypoint to the next, as solved
    legs: list[tuple[float, float]] = field(default_factory=list)


def line_path(chain: ArmChain, q_start, goal_q, goal_pose: np.ndarray,
              speed: Speed = SPEEDS["v1"]) -> tuple[Path | None, str]:
    """A straight line of the grasp point from where ``q_start`` puts it to ``goal_pose``."""
    start_pose = chain.grasp_pose(q_start)
    dist = float(np.linalg.norm(goal_pose[:3, 3] - start_pose[:3, 3]))
    turn = angle_between(start_pose[:3, :3], goal_pose[:3, :3])
    n = max(1, math.ceil(dist / speed.fine_m), math.ceil(turn / speed.fine_rad))
    q = np.asarray(q_start, float)
    out = [q.copy()]
    legs, here = [], start_pose
    for i in range(1, n + 1):
        t = i / n
        pose = np.eye(4)
        pose[:3, :3] = slerp(start_pose[:3, :3], goal_pose[:3, :3], t)
        pose[:3, 3] = (1 - t) * start_pose[:3, 3] + t * goal_pose[:3, 3]
        sol = refine(chain, pose, q, iterations=60)
        if i == n and float(np.max(np.abs(sol.q - goal_q))) < 0.05:
            # same branch: end exactly on the goal the solver chose and checked. Solved again, the
            # last waypoint can come out a hair lower and fail the table check at the height limit
            # (R0: 22 straight descents to exactly 1.5 cm refused "100% of the way").
            goal_q = np.asarray(goal_q, float)
            sol = Solution(q=goal_q.copy(), pos_err=0.0, rot_err=0.0, problems=clearance_problems(chain, goal_q))
        if not (sol.pos_err <= 0.004 and sol.rot_err <= math.radians(3.0)):
            return None, f"the straight line leaves the arm's reach {100 * t:.0f}% of the way"
        if sol.problems:
            return None, f"the straight line runs into the table ({sol.problems[0]}) {100 * t:.0f}% of the way"
        if float(np.max(np.abs(sol.q - q))) > LINE_JUMP_RAD:
            return None, f"the straight line passes a wrist singularity {100 * t:.0f}% of the way"
        q = sol.q
        out.append(q.copy())
        there = chain.grasp_pose(q)
        legs.append((float(np.linalg.norm(there[:3, 3] - here[:3, 3])), angle_between(here[:3, :3], there[:3, :3])))
        here = there
    return Path(out, "line", dist=dist, turn=turn, legs=legs), ""


def swing_path(chain: ArmChain, q_start, goal_q, samples: int = 12) -> tuple[Path | None, str]:
    """Joints interpolated directly from start to goal (the grasp point moves on a curve)."""
    q0 = np.asarray(q_start, float)
    q1 = np.asarray(goal_q, float)
    for i in range(1, samples):
        q = q0 + (q1 - q0) * (i / samples)
        problems = clearance_problems(chain, q)
        if problems:
            return None, f"a direct joint swing would put the {problems[0]} on the way"
    return Path([q0.copy(), q1.copy()], "swing"), ""


def swing_curve(chain: ArmChain, q_start, goal_q, samples: int = 24) -> tuple[float, float]:
    """The grasp point on a direct joint swing: (the farthest it gets from the straight line between
    the ends, how far it dips below the lower end), in metres."""
    q0 = np.asarray(q_start, float)
    q1 = np.asarray(goal_q, float)
    a = chain.grasp_pose(q0)[:3, 3]
    b = chain.grasp_pose(q1)[:3, 3]
    ab = b - a
    length2 = float(ab @ ab)
    floor = min(float(a[2]), float(b[2]))
    off = dip = 0.0
    for i in range(1, samples):
        p = chain.grasp_pose(q0 + (q1 - q0) * (i / samples))[:3, 3]
        t = 0.0 if length2 < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / length2))
        off = max(off, float(np.linalg.norm(p - (a + t * ab))))
        dip = max(dip, floor - float(p[2]))
    return off, dip


def to_actions(path: Path, max_step: float) -> list[np.ndarray]:
    """Joint targets, one per action, no joint moving more than ``max_step`` rad per action."""
    out = []
    for a, b in zip(path.waypoints[:-1], path.waypoints[1:]):
        n = max(1, math.ceil(float(np.max(np.abs(b - a))) / max_step))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return out


def actions_for(path: Path, speed: Speed, *, empty: bool = False) -> list[np.ndarray]:
    """Joint targets, one per action, for ``path`` at ``speed`` (see :class:`Speed`); ``empty``: the
    arm holds nothing, and its gripper does not close on the way."""
    if speed.uniform or (path.kind == "line" and path.dist == 0.0 and path.turn == 0.0):
        return to_actions(path, speed.joint)
    if path.kind == "line":
        if empty and speed.ramp_empty is not None:
            speed = replace(speed, ramp=speed.ramp_empty)
        return _line_actions(path, speed)
    out = []
    for a, b in zip(path.waypoints[:-1], path.waypoints[1:]):
        big = float(np.max(np.abs(b - a)))
        out.extend(a + (b - a) * t for t, _ in _profile(big / speed.joint, big / speed.joint_cruise, speed.ramp))
    return out


def _profile(ends: float, fast: float, ramp: int) -> list[tuple[float, bool]]:
    """Where a motion's actions end (fractions of it) and which of them cruise.

    ``ends`` and ``fast`` are the motion's length in end-limit and in cruise-limit actions.  Without
    a saving, every action is an end-limit one, evenly spread (as :func:`to_actions` spreads them).
    """
    plain = max(1, math.ceil(ends))
    if ends > 2 * ramp:
        a = ramp / ends  # the share of the motion in each end's ramp
        n_mid = max(1, math.ceil((1.0 - 2.0 * a) * fast))
        if 2 * ramp + n_mid < plain:
            marks = [a * i / ramp for i in range(1, ramp + 1)]
            marks += [a + (1.0 - 2.0 * a) * i / n_mid for i in range(1, n_mid + 1)]
            marks += [1.0 - a + a * i / ramp for i in range(1, ramp + 1)]
            marks[-1] = 1.0
            return list(zip(marks, [False] * ramp + [True] * n_mid + [False] * ramp))
    return [(i / plain, False) for i in range(1, plain + 1)]


def _line_actions(path: Path, speed: Speed) -> list[np.ndarray]:
    """A line's actions, placed by cost along its waypoints.

    Each leg between waypoints costs the largest of its travel, turn and joint move over one action's
    limits (at the end pace, and at the cruise pace); an action ends where the cost since the last one
    reaches 1, so none exceeds a limit, and what one leg leaves over is carried into the next instead
    of rounding up to a whole action.  The profile is :func:`_profile`'s: ``ramp`` actions at each end
    at the end limits, the ones between at the cruise limits, when that takes fewer actions.
    """
    # R1 cut each profile interval to the joint limit and rounded it up, so an interval that needed 1.1
    # actions took 2: replayed this way, R1's line moves take 10 % fewer actions (31 an episode), 21 of
    # 13529 one more
    n = len(path.waypoints) - 1
    legs = np.asarray(path.legs, float) if path.legs else np.tile([path.dist / n, path.turn / n], (n, 1))
    # the legs as solved, less up to 0.1 % of solver noise (a waypoint may be 0.2 mm, 0.05 degrees off
    # the line), which would put an exact 8 cm descent a hair over 4 actions
    sums = legs.sum(axis=0)
    legs = legs * np.clip(np.divide([path.dist, path.turn], sums, out=np.ones(2), where=sums > 0), 0.999, 1.0)
    moves = np.max(np.abs(np.diff(np.asarray(path.waypoints), axis=0)), axis=1)

    def cost(m: float, rad: float, joint: float) -> np.ndarray:  # in actions, at each waypoint
        return np.concatenate([[0.0], np.cumsum(np.maximum.reduce([legs[:, 0] / m, legs[:, 1] / rad, moves / joint]))])

    ends = cost(speed.line_m, speed.line_rad, speed.joint)
    fast = cost(speed.cruise_m, speed.cruise_rad, speed.joint_cruise)
    index = np.arange(n + 1.0)
    total, ramp = float(ends[-1]), speed.ramp
    plain = max(1, math.ceil(total - 1e-9))
    marks = np.interp(total * np.arange(1, plain + 1) / plain, ends, index)
    if total > 2 * ramp:
        a, b = np.interp([ramp, total - ramp], ends, index)  # where the ramps end
        lo, hi = np.interp([a, b], index, fast)
        n_mid = max(1, math.ceil(hi - lo - 1e-9))
        if 2 * ramp + n_mid < plain:
            marks = np.concatenate([np.interp(np.arange(1.0, ramp + 1), ends, index),
                                    np.interp(lo + (hi - lo) * np.arange(1, n_mid + 1) / n_mid, fast, index),
                                    np.interp(total - ramp + np.arange(1.0, ramp + 1), ends, index)])
    return [_along(path.waypoints, s / n) for s in marks[:-1]] + [path.waypoints[-1].copy()]


def _along(waypoints: list[np.ndarray], t: float) -> np.ndarray:
    """Joints a fraction ``t`` of the way along evenly spaced waypoints."""
    n = len(waypoints) - 1
    if t >= 1.0:
        return waypoints[-1].copy()
    x = max(t, 0.0) * n
    i = min(int(x), n - 1)
    return waypoints[i] + (waypoints[i + 1] - waypoints[i]) * (x - i)
