"""`Kinematics` for RoboDojo's two ARX-X5 arms over ``knowin_agent.robot_model``.

Poses are of the flange (``link6``) in the RoboDojo env frame, the frame of the
native ``<arm>_ee_pose`` observation. Verdicts are kinematic evidence only:
the chain-length bound proves a pose unreachable, a converged full-pose IK
proves it reachable (collisions not checked), anything else is unknown.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from knowin_agent.core.geometry import Pose, as_vector
from knowin_agent.runtime.backend import Reach
from knowin_agent.robot_model import kinematics as robot_model

ARMS = robot_model.ARMS
CHAIN_LENGTH_M = robot_model.CHAIN_REACH_M
BASE_XYZ = {arm: np.asarray(robot_model.BASES[arm][0], dtype=float) for arm in ARMS}


def _joints(joints) -> np.ndarray:
    return as_vector(joints, 6, "X5 joint vector")


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")


class ArxKinematics:
    """Full-pose FK/IK of the flange, with an optional check against the observed flange."""

    def __init__(
        self,
        position_tolerance_m: float = 0.003,
        angle_tolerance_rad: float = 0.08,
        model_position_tolerance_m: float = 0.002,
        model_angle_tolerance_rad: float = math.radians(1.0),
    ):
        self.position_tolerance_m = float(position_tolerance_m)
        self.angle_tolerance_rad = float(angle_tolerance_rad)
        self.model_position_tolerance_m = float(model_position_tolerance_m)
        self.model_angle_tolerance_rad = float(model_angle_tolerance_rad)

    def fk(self, arm: str, joints: np.ndarray) -> Pose:
        _check_arm(arm)
        return Pose.from_matrix(robot_model.fk(arm, _joints(joints)))

    def model_residual(self, arm: str, joints: np.ndarray, observed: Pose) -> tuple[float, float]:
        """(metres, radians) between FK of the joints and an observed flange pose."""
        return self.fk(arm, joints).errors_to(observed)

    def reach(self, arm: str, flange: Pose, joints: np.ndarray, observed: Optional[Pose] = None) -> Reach:
        """Reachability of a flange pose from the current joints.

        When ``observed`` (the measured flange pose) is given, the model must
        reproduce it first; a mismatch makes every verdict unknown.
        """
        _check_arm(arm)
        q = _joints(joints)
        if observed is not None:
            dp, da = self.model_residual(arm, q, observed)
            if dp > self.model_position_tolerance_m or da > self.model_angle_tolerance_rad:
                return Reach(None, f"kinematic model disagrees with the observed flange ({dp * 1000:.1f} mm, "
                                   f"{math.degrees(da):.1f} deg)")
        distance = float(np.linalg.norm(flange.position - BASE_XYZ[arm]))
        if distance > CHAIN_LENGTH_M + self.position_tolerance_m:
            return Reach(False, f"{distance:.3f} m from the {arm} base exceeds the chain length "
                                f"{CHAIN_LENGTH_M:.3f} m")
        solution = robot_model.best_solution(
            arm,
            flange.matrix(),
            q,
            offset=0.0,
            pos_tol=self.position_tolerance_m,
            rot_tol_deg=math.degrees(self.angle_tolerance_rad),
            use_table_filter=False,
        )
        if solution.ok:
            travel = float(np.degrees(np.max(np.abs(np.asarray(solution.q) - q))))
            return Reach(True, "full-pose IK solution (collisions not checked)", list(solution.q), travel)
        return Reach(None, f"IK did not converge (closest miss {solution.pos_err_m * 1000:.1f} mm, "
                           f"{solution.rot_err_deg:.1f} deg)")

    def plan(self, arm: str, q0, flange: Pose, q_goal, *, step_m: float, rot_step: float, precision: bool,
             path: str = "auto", straight: bool = False, max_joint_step: float = None):
        """Joint waypoints from q0 to the flange goal: (list of joint vectors, how) or (None, why).

        auto: joint space for moves of at least LINE_BELOW_M when the grasp point stays within CORRIDOR_M of
        the straight line, else a straight line, else unconstrained joint space; precise and contact moves
        (straight) only go in a straight line. precision: the last PRECISION_RADIUS_M is a straight line in
        PRECISION_STEP_M steps. Each waypoint is solved from the previous one, so the arm never flips IK branch,
        and no waypoint moves a joint more than max_joint_step (default LINE_JOINT_STEP_RAD) from the one before.
        """
        _check_arm(arm)
        if path not in ("auto", "line", "joint"):
            raise ValueError("path must be auto, line or joint")
        q0 = _joints(q0)
        goal = flange.matrix()
        start = robot_model.fk(arm, q0)
        dist = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        main_goal, q_main = goal, _joints(q_goal)
        if precision and dist > PRECISION_RADIUS_M:
            main_goal = goal.copy()
            main_goal[:3, 3] = goal[:3, 3] - (goal[:3, 3] - start[:3, 3]) / dist * PRECISION_RADIUS_M
            res = robot_model.solve(arm, main_goal, q_main, offset=0.0)
            if not res.ok:
                return None, "no joint solution short of the goal"
            q_main = np.asarray(res.q, dtype=float)
        if precision and dist <= PRECISION_RADIUS_M:
            attempts = [("line", None)]
        elif path == "auto":
            attempts = [("line", None)] if precision or straight else (
                [("joint", CORRIDOR_M)] * (dist >= LINE_BELOW_M) + [("line", None), ("joint", None)])
        else:
            attempts = [(path, None)]
        ways, how, why = None, None, "no path"
        for kind, corridor in attempts:
            step = PRECISION_STEP_M if precision and dist <= PRECISION_RADIUS_M else step_m
            if kind == "joint":
                ways, why = _joint_segment(arm, q0, q_main, step, rot_step, corridor)
            else:
                ways, why = _line_segment(arm, q0, main_goal, step, rot_step)
            if ways is not None:
                how = kind
                break
        if ways is None:
            return None, why
        if main_goal is not goal:
            tail, why = _line_segment(arm, ways[-1], goal, PRECISION_STEP_M, rot_step)
            if tail is None:
                return None, why
            ways += tail
            how += "+precise_line"
        final = robot_model.solve(arm, goal, ways[-1], offset=0.0, pos_tol=0.0005, rot_tol_deg=0.3)
        if final.ok and float(np.max(np.abs(np.asarray(final.q) - ways[-1]))) < JOINT_STEP_RAD:
            ways[-1] = np.asarray(final.q, dtype=float)
        return rate_limited(q0, ways, max_joint_step or LINE_JOINT_STEP_RAD), how


PRECISION_RADIUS_M = 0.02
PRECISION_STEP_M = 0.003
JOINT_STEP_RAD = 0.15
# Joint steps the controller tracks in one action are below about 0.3 rad (larger ones stopped as arm_not_following);
# 0.2 leaves a margin for tracking lag and adds 0.8 waypoints to an average 10.6-waypoint line plan.
LINE_JOINT_STEP_RAD = 0.2
CORRIDOR_M = 0.05
LINE_BELOW_M = 0.05


def rate_limited(q0, ways, max_step: float) -> list:
    """Joint-space points inserted so that no waypoint moves a joint more than max_step from the previous
    one. A straight-line plan solves each Cartesian step by IK, and near a singular pose consecutive
    solutions can differ by up to 0.6 rad; the controller does not track a jump above about 0.3 rad in one
    action (the arm_not_following stops). Consecutive line waypoints are a few centimetres apart, so the
    inserted points stay on the line to within millimetres."""
    out, prev = [], np.asarray(q0, dtype=float)
    for q in ways:
        q = np.asarray(q, dtype=float)
        n = max(1, math.ceil(float(np.max(np.abs(q - prev))) / max_step - 1e-9))
        out.extend(prev + (q - prev) * k / n for k in range(1, n + 1))
        prev = q
    return out


def _grasp_point(t: np.ndarray) -> np.ndarray:
    return t[:3, 3] + t[:3, 0] * robot_model.TCP_OFFSET


def _fingertips(t: np.ndarray) -> np.ndarray:
    return t[:3, 3] + t[:3, 0] * robot_model.TIP_OFFSET


def _joint_segment(arm, q0, q1, step_m, rot_step, corridor):
    """Straight interpolation in joint space, or (None, why) when it touches the table, takes the fingertips
    below both ends (a turning wrist sweeps them low), or (with a corridor) bows away from the straight line."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    a, b = robot_model.fk(arm, q0), robot_model.fk(arm, q1)
    p0, p1 = _grasp_point(a), _grasp_point(b)
    turn = math.radians(robot_model.rotation_error(a[:3, :3], b[:3, :3]))
    n = max(1, math.ceil(float(np.max(np.abs(q1 - q0))) / JOINT_STEP_RAD),
            math.ceil(float(np.linalg.norm(p1 - p0)) / step_m), math.ceil(turn / rot_step))
    line, floor = p1 - p0, min(_fingertips(a)[2], _fingertips(b)[2]) - 0.002
    ways = []
    for k in range(1, n + 1):
        q = q0 + (q1 - q0) * k / n
        if robot_model.clearance_violations(arm, q):
            return None, "a joint-space move would touch the table"
        t = robot_model.fk(arm, q)
        p = _grasp_point(t)
        if _fingertips(t)[2] < floor:
            return None, "a joint-space move would take the fingertips below both ends"
        if corridor is not None:
            s = float(np.clip(np.dot(p - p0, line) / max(float(np.dot(line, line)), 1e-12), 0, 1))
            if float(np.linalg.norm(p - (p0 + s * line))) > corridor:
                return None, "a joint-space move would bow away from the straight line"
        ways.append(q)
    return ways, None


def _line_segment(arm, q0, goal, step_m, rot_step):
    ways, why = robot_model.line_path(arm, q0, goal, offset=0.0, step_m=step_m, step_deg=math.degrees(rot_step))
    return (None, why) if why else ([np.asarray(q, dtype=float) for q in ways], None)
