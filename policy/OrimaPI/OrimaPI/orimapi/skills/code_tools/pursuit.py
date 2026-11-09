"""Finite fresh-measurement point pursuit; no prediction, re-identification or lift.

The session must supply a newly checked RGB-triangulated material point at every
observation. Original IK, joint-step, robot/table/collision and TCP arrival checks
are reused. Target updates never renew the original operation allowance.
"""

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import vector, verify_pose
from .motion import down_rotation

MAX_TCP_TRAVEL = 0.15
MAX_TARGET_PATH = 0.15
MAX_TARGET_STEP = 0.015
MAX_CARTESIAN_STEP = 0.01
MIN_CLOSE_STEPS = 8


class PointPursuit:
    def __init__(self, geometry, motion, observation, measurement, *, side, offset,
                 max_steps, available_actions, close=False):
        if side not in ("left", "right") or type(close) is not bool:
            raise ValueError("Pursuit needs a valid arm and explicit boolean close mode")
        if type(max_steps) is not int or not 8 <= max_steps <= 120:
            raise ValueError("Pursuit max_steps must be in8..120")
        if type(available_actions) is not int or available_actions < 1:
            raise ValueError("Pursuit needs a positive remaining native allowance")
        if measurement.get("step") != observation.step or measurement.get("method") != "rgb_triangulation":
            raise ValueError("Pursuit starts only from fresh current RGB triangulation")
        self.geometry, self.motion = geometry, motion
        self.side, self.close = side, close
        self.offset = vector(offset)
        if np.linalg.norm(self.offset) > 0.15 + 1e-9:
            raise ValueError("Pursuit target-to-TCP offset is bounded to0.15m")
        self.start_step = observation.step
        self.start_tcp = geometry.fk(side, observation.robot_state[f"{side}_arm_joint_state"])
        self.rotation = self.start_tcp[:3, :3].copy()
        self.initial_target = vector(measurement["position"])
        self.last_target = self.initial_target.copy()
        self.target_path = 0.0
        self.last_step = None
        self.commands = 0
        self.hold_joints = {
            arm: np.asarray(observation.robot_state[f"{arm}_arm_joint_state"], float).copy()
            for arm in ("left", "right")
        }
        self.grippers = deepcopy(motion.gripper_targets)
        for value in self.grippers.values():
            if len(value) != 1 or not np.isfinite(value[0]) or not 0 <= value[0] <= 1:
                raise ValueError("Pursuit cannot use an invalid prior gripper command")
        key = f"{side}_ee_joint_state"
        self.grip_command = float(self.grippers[key][0])
        self.grip_command_start = self.grip_command
        self.grip_initial = float(observation.robot_state[key][0])
        if not np.isfinite(self.grip_initial) or not 0 <= self.grip_initial <= 1:
            raise ValueError("Pursuit requires a finite observed gripper state")
        self.grip_previous = self.grip_initial
        self.grip_stable = 0
        goal = self.goal(self.initial_target)
        if np.linalg.norm(goal[:3, 3] - self.start_tcp[:3, 3]) > MAX_TCP_TRAVEL + 1e-9:
            raise ValueError("Pursuit starts only within0.15m of its current goal; stage safely first")
        if close and verify_pose(self.start_tcp, goal)["status"] != "pass":
            raise ValueError("Explicit pursuit closure requires current relative TCP arrival before review")
        tilt_x, tilt_y, yaw = Rotation.from_matrix(self.rotation @ down_rotation(0).T).as_euler("XYZ")
        operation = {
            "op": "move", "label": "initial current-point pursuit preflight", "arm": side,
            "position": goal[:3, 3].tolist(), "tilt_x": float(tilt_x),
            "tilt_y": float(tilt_y), "yaw": float(yaw),
            "gripper": self.grip_command, "steps": 1,
        }
        self.preflight = motion.preflight([operation], observation, available_actions)
        minimum = max(self.preflight["kinematic_minimum_steps"], 1)
        # A moving target does not turn into an unbounded sequence of fresh stages.
        self.allowance = min(max_steps, minimum + 40, available_actions)
        if self.allowance < (MIN_CLOSE_STEPS if close else 1):
            raise ValueError("Insufficient pursuit allowance after native/cleanup reserves")
        self.latest = {
            "start_step": self.start_step, "allowed_actions": self.allowance,
            "commands_dispatched": 0, "target_path_m": 0.0,
        }

    def goal(self, target):
        result = np.eye(4)
        result[:3, :3] = self.rotation
        result[:3, 3] = vector(target) + self.offset
        return result

    def _closing_settled(self, observed):
        # Same normalized aperture/ramp/settling constants as the existing grip
        # primitive. This is command/settling evidence, never retention proof.
        if not np.isfinite(observed) or not 0 <= observed <= 1:
            raise ValueError("Pursuit closure lost valid observed gripper state")
        self.grip_stable = self.grip_stable + 1 if abs(observed - self.grip_previous) < 0.006 else 0
        self.grip_previous = observed
        contact_settled = self.grip_stable >= 3 and (
            observed < self.grip_initial - 0.03
            or (
                self.grip_initial < 0.95
                and observed <= self.grip_initial + 0.006
                and self.grip_command_start < self.grip_initial - 0.03
            )
        )
        return (
            self.commands >= MIN_CLOSE_STEPS
            and abs(self.grip_command) < 1e-6
            and (abs(observed) < 0.03 or contact_settled)
        )

    def _candidate_joints(self, current, actual, goal):
        final = self.motion._solve(self.side, goal[:3, 3], self.rotation, current)
        delta = float(np.max(np.abs(final - current)))
        distance = float(np.linalg.norm(goal[:3, 3] - actual[:3, 3]))
        fraction = min(
            1.0, self.motion.max_joint_step / max(delta, 1e-12),
            MAX_CARTESIAN_STEP / max(distance, 1e-12),
        )
        rotations = Slerp([0, 1], Rotation.from_matrix([actual[:3, :3], self.rotation]))
        for _ in range(8):
            xyz = actual[:3, 3] + fraction * (goal[:3, 3] - actual[:3, 3])
            rotation = rotations(fraction).as_matrix()
            joints = self.motion._solve(self.side, xyz, rotation, current)
            proposed = self.geometry.fk(self.side, joints)
            if (
                float(np.max(np.abs(joints - current))) <= self.motion.max_joint_step + 1e-9
                and np.linalg.norm(proposed[:3, 3] - actual[:3, 3]) <= MAX_CARTESIAN_STEP + 1e-9
                and np.linalg.norm(proposed[:3, 3] - self.start_tcp[:3, 3]) <= MAX_TCP_TRAVEL + 1e-9
            ):
                return joints
            fraction *= 0.5
        raise ValueError("No accurate continuity-bounded pursuit waypoint; no branch jump permitted")

    def _guard_proposal(self, action):
        clearance = self.motion.clearance
        saved = clearance.static if clearance is not None else None
        try:
            if clearance is not None:
                clearance.static = {
                    side: clearance.objects(side, action[f"{side}_arm_joint_state"])
                    for side in ("left", "right")
                }
            for side in ("left", "right"):
                self.motion._guard(side, action[f"{side}_arm_joint_state"])
        finally:
            if clearance is not None:
                clearance.static = saved

    def step(self, observation, measurement):
        if measurement.get("step") != observation.step or measurement.get("method") != "rgb_triangulation":
            raise ValueError("Pursuit requires fresh current-observation RGB triangulation")
        if self.last_step is not None and observation.step != self.last_step + 1:
            raise ValueError("Pursuit observations must be consecutive; no stale-frame continuation")
        target = vector(measurement["position"])
        travel = float(np.linalg.norm(target - self.last_target))
        if travel > MAX_TARGET_STEP + 1e-9 or self.target_path + travel > MAX_TARGET_PATH + 1e-9:
            raise ValueError("Pursuit target jump/path exceeds its fixed evidence bounds")
        current = np.asarray(observation.robot_state[f"{self.side}_arm_joint_state"], float)
        actual = self.geometry.fk(self.side, current)
        goal = self.goal(target)
        if (
            np.linalg.norm(actual[:3, 3] - self.start_tcp[:3, 3]) > MAX_TCP_TRAVEL + 1e-9
            or np.linalg.norm(goal[:3, 3] - self.start_tcp[:3, 3]) > MAX_TCP_TRAVEL + 1e-9
        ):
            raise ValueError("Pursuit left its fixed local TCP workspace")
        verification = verify_pose(actual, goal)
        self.target_path += travel
        self.last_target, self.last_step = target.copy(), observation.step
        self.latest = {
            "start_step": self.start_step, "step": observation.step,
            "elapsed_actions": observation.step - self.start_step,
            "commands_dispatched": self.commands, "allowed_actions": self.allowance,
            "target_path_m": self.target_path, "target_step_m": travel,
            "current_measurement": deepcopy(measurement),
            "relative_tcp_verification": verification,
            "gripper_command": self.grip_command,
            "scope": "Current relative robot/point tracking only; no object identity, capture-window or retention certificate",
        }
        close_done = self._closing_settled(
            float(observation.robot_state[f"{self.side}_ee_joint_state"][0])
        ) if self.close else True
        if verification["status"] == "pass" and close_done:
            return None, {
                "status": "completed", "verification": verification,
                "closure_settled_not_retention": self.close,
                "official_success": None, "diagnostics": deepcopy(self.latest),
            }
        if (
            self.commands >= self.allowance
            or observation.step - self.start_step >= self.allowance
        ):
            raise ValueError("Pursuit exhausted its original bounded allowance; target updates do not reset it")
        if self.close and self.commands >= MIN_CLOSE_STEPS + 30 - 1 and not close_done:
            raise ValueError("Gripper failed to settle within original closing allowance")
        joints = self._candidate_joints(current, actual, goal)
        action = {}
        for side in ("left", "right"):
            key = f"{side}_arm_joint_state"
            observed = np.asarray(observation.robot_state[key], float)
            command = joints if side == self.side else observed + np.clip(
                self.hold_joints[side] - observed, -self.motion.max_joint_step, self.motion.max_joint_step
            )
            if float(np.max(np.abs(command - observed))) > self.motion.max_joint_step + 1e-9:
                raise ValueError("Pursuit proposal violates the original joint-step bound")
            action[key] = command.tolist()
            action[f"{side}_ee_joint_state"] = list(self.grippers[f"{side}_ee_joint_state"])
        proposed_grip = max(0.0, self.grip_command - 0.15) if self.close else self.grip_command
        action[f"{self.side}_ee_joint_state"] = [proposed_grip]
        self._guard_proposal(action)
        # Transactional: rejected proposals cannot silently change grip commands.
        self.grip_command = proposed_grip
        self.grippers[f"{self.side}_ee_joint_state"] = [proposed_grip]
        self.commands += 1
        self.latest["commands_dispatched"] = self.commands
        self.latest["gripper_command"] = proposed_grip
        return action, None
