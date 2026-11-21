"""Turn one TCP target into a 25 Hz joint sequence.

The arm moves on a straight line in Cartesian space: inverse kinematics is
solved at waypoints about 2 cm apart, each seeded by the previous solution,
and joints are interpolated linearly in between. If any waypoint has no
solution, or the arm would have to change its configuration, the command fails.

Timing uses only the limits of the official robot: the joint velocity limit of the x5 arm actuators in the simulator
(velocity_limit_sim = 5 rad/s) and the joint acceleration limit of the official cuRobo configuration
(cspace.max_acceleration = 10 rad/s^2). There is no Cartesian speed cap of our own.
"""

import os

import numpy as np

from roboshell.server import geometry as geo

CONTROL_HZ = 25.0
WAYPOINT_SPACING_M = 0.02
WAYPOINT_SPACING_DEG = 10.0
MAX_JOINT_JUMP = 1.5  # rad between neighbouring waypoints: beyond this the arm changed its configuration
MAX_JOINT_SPEED = float(os.environ.get("ROBOSHELL_JOINT_SPEED", 5.0))   # rad/s, official: x5 actuator velocity_limit_sim
MAX_JOINT_ACCEL = float(os.environ.get("ROBOSHELL_JOINT_ACCEL", 10.0))  # rad/s^2, official: cuRobo cspace max_acceleration
MIN_STEPS = 1
IK_ATTEMPTS = 4

# Kept for task tools written before 2026-10-02 that read these names. They no longer limit anything the server does:
# a tool that paces its own joint sequences with them keeps the pace it was tuned with.
MAX_LINEAR_SPEED = 0.20   # m/s (historical)
MAX_ANGULAR_SPEED = 90.0  # deg/s (historical)


class PlanFailure(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def time_path(waypoints):
    """Sample a joint path (rows = waypoints, the first one is the current state) at the control rate.

    The path is traversed with a trapezoidal speed profile: the fastest joint accelerates at MAX_JOINT_ACCEL, cruises at
    MAX_JOINT_SPEED and decelerates to rest at the last waypoint. Returns [steps, dof] targets, without the start row.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    length = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(waypoints, axis=0)).max(axis=1))])  # rad travelled by the fastest joint
    total = float(length[-1])
    if total < 1e-6:
        return waypoints[-1:].copy()
    speed, accel = MAX_JOINT_SPEED, MAX_JOINT_ACCEL
    if total < speed * speed / accel:        # too short to reach the speed limit: accelerate, then decelerate
        ramp = float(np.sqrt(total / accel)); peak = accel * ramp; duration = 2 * ramp
    else:
        ramp = speed / accel; peak = speed; duration = total / speed + ramp
    steps = max(MIN_STEPS, int(np.ceil(duration * CONTROL_HZ - 1e-9)))
    time = np.arange(1, steps + 1) / steps * duration
    travelled = np.where(time < ramp, 0.5 * accel * time**2,
                         np.where(time < duration - ramp, 0.5 * accel * ramp**2 + peak * (time - ramp),
                                  total - 0.5 * accel * (duration - time) ** 2))
    travelled = np.clip(travelled, 0.0, total)
    out = np.stack([np.interp(travelled, length, waypoints[:, j]) for j in range(waypoints.shape[1])], axis=1)
    out[-1] = waypoints[-1]
    return out


def _smooth(fractions):
    """Historical ease-in/ease-out curve, kept for task tools that build their own sequences."""
    fractions = np.asarray(fractions, dtype=float)
    return 3 * fractions**2 - 2 * fractions**3


def retime(sequence, start_joints):
    """Historical name: re-sample a joint sequence so it respects the joint speed and acceleration limits."""
    return time_path(np.vstack([np.asarray(start_joints, dtype=float)[None], np.asarray(sequence, dtype=float)]))


def solve_ik(planner, robot, pose, seed, start_joints):
    """cuRobo's numerical IK occasionally fails on a reachable pose. Retry from other seeds before giving up:
    the previous waypoint's solution, the joints at the start of the motion, then small perturbations."""
    seeds = [np.asarray(seed, dtype=float), np.asarray(start_joints, dtype=float)]
    rng = np.random.default_rng(0)
    for attempt in range(IK_ATTEMPTS):
        candidate = seeds[attempt] if attempt < len(seeds) else seeds[0] + rng.normal(0.0, 0.15, size=len(seeds[0]))
        result = planner.solve_ik_to_joint(candidate, pose, real_robot_pose=list(robot.entity_origin_pose))
        if result["status"] == "Success":
            return result
    return result


def wrap_near(joints, reference):
    """cuRobo allows +-10 rad on the arm joints, so a solution can be a full turn away from the arm."""
    joints = np.asarray(joints, dtype=float)
    return joints + 2 * np.pi * np.round((np.asarray(reference, dtype=float) - joints) / (2 * np.pi))


def check_limits(joints, limits):
    if limits is not None and (np.any(joints < limits[0] - 1e-3) or np.any(joints > limits[1] + 1e-3)):
        raise PlanFailure("ik_unreachable")


def plan_line(planner, robot, start_joints, start_ee, target_ee, limits=None):
    """Return an array [steps, dof] of joint targets from start to target (end link poses as 4x4)."""
    distance = float(np.linalg.norm(target_ee[:3, 3] - start_ee[:3, 3]))
    angle = geo.angle_between_deg(start_ee[:3, :3], target_ee[:3, :3])
    segments = max(1, int(np.ceil(max(distance / WAYPOINT_SPACING_M, angle / WAYPOINT_SPACING_DEG))))

    waypoints = [np.asarray(start_joints, dtype=float)]
    for index in range(1, segments + 1):
        fraction = index / segments
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * start_ee[:3, 3] + fraction * target_ee[:3, 3]
        pose[:3, :3] = geo.slerp(start_ee[:3, :3], target_ee[:3, :3], fraction)
        result = solve_ik(planner, robot, geo.matrix_to_pose(pose), waypoints[-1], start_joints)
        if result["status"] != "Success":
            raise PlanFailure("ik_unreachable", f"no solution at waypoint {index}/{segments}, {distance * index / segments:.3f} m along the line")
        joints = wrap_near(result["joint_value"], waypoints[-1])
        check_limits(joints, limits)
        if np.abs(joints - waypoints[-1]).max() > MAX_JOINT_JUMP:
            raise PlanFailure("ik_jump", f"configuration change at waypoint {index}/{segments}, joint jump {np.abs(joints - waypoints[-1]).max():.2f} rad")
        waypoints.append(joints)

    return time_path(np.stack(waypoints))


def plan(planner, robot, start_joints, start_ee, target_ee, sim_dt, limits=None):
    """Returns (joint sequence, method). Only the straight line with cuRobo inverse kinematics is used,
    the same solver the official ee action channel uses; the cuRobo motion planner is not."""
    try:
        return plan_line(planner, robot, start_joints, start_ee, target_ee, limits), "line"
    except PlanFailure as failure:
        if failure.reason == "ik_jump":
            raise PlanFailure("ik_unreachable", failure.detail)
        raise
