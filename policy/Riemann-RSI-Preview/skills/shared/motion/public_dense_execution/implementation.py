"""Mature dense-command execution using only explicit public observations."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np


STATE_WIDTHS = MappingProxyType({
    "left_arm_joint_state": 6,
    "right_arm_joint_state": 6,
    "left_ee_pose": 7,
    "right_ee_pose": 7,
    "left_ee_joint_state": 1,
    "right_ee_joint_state": 1,
})
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class NativeEpisodeEnded(RuntimeError):
    """The public execution protocol ended the episode, without a reward read."""


class PublicObservationSeries:
    """Keep the last two public frames while preserving the frame ordinal."""

    def __init__(self):
        self._frames = deque(maxlen=2)
        self._count = 0

    def append(self, frame):
        self._frames.append(np.asarray(frame).copy())
        self._count += 1

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if index not in (-1, -2):
            raise IndexError("Only the two latest public frames are retained")
        return self._frames[index].copy()


class PublicObservationMemory:
    """Project six robot fields and three RGB cameras; never retain metadata."""

    def __init__(self, observation, output_root):
        self.output_root = Path(output_root)
        self.images = {name: PublicObservationSeries() for name in CAMERAS}
        self.sim_steps = PublicObservationSeries()
        self.state_left_arm = PublicObservationSeries()
        self.state_right_arm = PublicObservationSeries()
        self.state_left_ee = PublicObservationSeries()
        self.state_right_ee = PublicObservationSeries()
        self.state_left_gripper = PublicObservationSeries()
        self.state_right_gripper = PublicObservationSeries()
        self.denied_reads = []
        self._state = {}
        self._sim_step = 0
        self.update(observation, sim_step=0)

    def update(self, observation, *, sim_step):
        state = observation["state"]
        if set(state) != set(STATE_WIDTHS):
            raise ValueError("Expected only the six declared public robot fields")
        projected = {}
        for key, width in STATE_WIDTHS.items():
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != (width,) or not np.all(np.isfinite(value)):
                raise ValueError("Invalid public robot observation: " + key)
            if key.endswith("ee_joint_state") and np.any((value < 0.) | (value > 1.)):
                raise ValueError("Public gripper feedback must be normalized")
            projected[key] = value.copy()
        frames = {}
        for camera in CAMERAS:
            image = np.asarray(observation["vision"][camera]["colors"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError("Expected decoded public RGB: " + camera)
            frames[camera] = image[..., :3]
        self._state = projected
        for arm in ("left", "right"):
            getattr(self, "state_" + arm + "_arm").append(projected[arm + "_arm_joint_state"])
            getattr(self, "state_" + arm + "_ee").append(projected[arm + "_ee_pose"])
            getattr(self, "state_" + arm + "_gripper").append(projected[arm + "_ee_joint_state"])
        for camera, image in frames.items():
            self.images[camera].append(image)
        self._sim_step = int(sim_step)
        self.sim_steps.append(self._sim_step)

    def state(self, key):
        return self._state[key].copy()


def project_observation(observation):
    """Build a detached actor input from the official observation payload."""
    return {
        "state": {key: np.asarray(observation["state"][key], dtype=np.float32).copy()
                  for key in STATE_WIDTHS},
        "vision": {camera: {"colors": np.asarray(observation["vision"][camera][
            "color" if "color" in observation["vision"][camera] else "colors"])[..., :3].copy()}
                   for camera in CAMERAS},
    }


@dataclass(frozen=True)
class PublicRobotDescription:
    arm_name: str
    robot_name: str
    gripper_scale: tuple[float, float]
    gripper_sign: int
    gripper_mimic: tuple[float, float]
    gripper_name: str = ""
    type: str = "target"
    ee_type: str = "gripper"

    def __post_init__(self):
        if self.arm_name not in ("left_arm", "right_arm"):
            raise ValueError("Expected one of the two public X5 arms")
        lower, upper = self.gripper_scale
        if not np.isfinite([lower, upper, *self.gripper_mimic]).all() or upper <= lower:
            raise ValueError("Invalid declared gripper command geometry")
        if self.gripper_sign not in (-1, 1):
            raise ValueError("Invalid declared gripper direction")
        expected = self.arm_name.split("_")[0] + "_ee"
        if self.gripper_name and self.gripper_name != expected:
            raise ValueError("Gripper name must match the public observation key")
        object.__setattr__(self, "gripper_name", expected)

    def command_pair(self, normalized_value):
        lower, upper = self.gripper_scale
        normalized_value = float(normalized_value)
        primary = (normalized_value * (upper - lower) + lower
                   if self.gripper_sign == 1 else upper - normalized_value * (upper - lower))
        multiplier, offset = self.gripper_mimic
        return np.asarray([primary, primary * multiplier + offset], dtype=float)


class PublicRobotManager:
    """Own planners/configuration; joint observations come exclusively from memory."""

    def __init__(self, memory, robots, planners, joint_limits):
        self.memory = memory
        self.robot_list = tuple(robots)
        if {robot.arm_name for robot in self.robot_list} != {"left_arm", "right_arm"} or len(self.robot_list) != 2:
            raise ValueError("Exactly two declared X5 arms are required")
        self.planner = MappingProxyType(dict(planners))
        limits = {}
        for robot in self.robot_list:
            if robot.robot_name not in self.planner:
                raise ValueError("Missing owned planner: " + robot.robot_name)
            limit = np.asarray(joint_limits[robot.arm_name], dtype=float).copy()
            if limit.shape != (6, 2) or not np.isfinite(limit).all() or np.any(limit[:, 0] > limit[:, 1]):
                raise ValueError("Expected public URDF joint limits")
            limit.setflags(write=False)
            limits[robot.arm_name] = limit
        self.joint_limits = MappingProxyType(limits)
        self.gripper_seed_audit = []

    @staticmethod
    def process_name(name):
        return name if name.endswith("_state") else name + "_joint_state"

    def get_joint(self, robot, env_idx_list=None):
        if env_idx_list not in (None, [0], (0,)):
            raise ValueError("Only the current public episode is available")
        return {0: self.memory.state(self.process_name(robot.arm_name))}

    def _robot(self, arm_tag):
        return next(robot for robot in self.robot_list if robot.arm_name == arm_tag)

    def plan_ee(self, env_idx, arm_tag, result):
        if env_idx != 0:
            raise ValueError("Only the current public episode is available")
        if result is None or result.get("status") != "Success":
            return None
        robot = self._robot(arm_tag)
        return [{self.process_name(robot.arm_name): {
            "position": result["position"][index], "velocity": result["velocity"][index]}}
            for index in range(result["position"].shape[0])]

    def plan_endeffector_joint(self, env_idx, arm_tag, target_val, need_plan=True, control_step_num=100):
        if env_idx != 0 or need_plan is not True:
            raise ValueError("Only declared normalized gripper plans are supported")
        robot = self._robot(arm_tag)
        requested = np.asarray(target_val, dtype=float).reshape(-1)
        if requested.shape != (1,) or not np.isfinite(requested).all() or not 0. <= requested[0] <= 1.:
            raise ValueError("Expected one normalized gripper command")
        steps = max(1, int(control_step_num))
        key = self.process_name(robot.gripper_name)
        feedback = float(self.memory.state(key)[0])
        anchor = robot.command_pair(feedback)
        target = robot.command_pair(requested[0])
        self.gripper_seed_audit.append({
            "source": "normalized_public_limited_target",
            "feedback": feedback,
            "requested": float(requested[0]),
            "actual_width": None,
            "contact_evidence": "unknown",
        })
        interpolation_steps = int(steps * .9)
        controls = []
        for step in range(steps):
            if interpolation_steps <= 1:
                position = target
            elif step < interpolation_steps:
                position = anchor + step / (interpolation_steps - 1) * (target - anchor)
            else:
                position = target
            controls.append({key: {"position": position.tolist(), "velocity": [0., 0.]}})
        return controls


@dataclass(frozen=True)
class PublicStepReply:
    observation: Mapping
    episode_ended: bool = False


class PublicExecutionClock:
    """An actor-owned command count, not a view of the simulator environment."""

    def __init__(self, robot_manager, step_limit):
        if type(step_limit) is not int or not 1 <= step_limit <= 1400:
            raise ValueError("Invalid public action budget")
        self.dt = .004
        self.step_lim = step_limit
        self.take_action_cnt = [0]
        self.robot_manager = robot_manager
        self.denied_reads = []


class PublicCommandBridge:
    """Preserve original sparse fill, fractional sampling, preload and ACK order."""

    def __init__(self, memory, robot_manager, exchange: Callable, *, time_scale=1.4, step_limit=1400):
        if not callable(exchange) or not np.isfinite(time_scale) or time_scale <= 0.:
            raise ValueError("An explicit public exchange and finite positive speed are required")
        self.memory = memory
        self.robot_manager = robot_manager
        self.exchange = exchange
        self.time_scale = float(time_scale)
        self.frequency = 25.
        self.interval = 10
        self.clock = PublicExecutionClock(robot_manager, step_limit)
        self.normalized_actions = []
        self.action_sim_steps = []
        self.action_phases = []
        self.episode_ended = False
        self._targets = {}
        self._grippers = {}
        for robot in robot_manager.robot_list:
            arm_key = robot_manager.process_name(robot.arm_name)
            grip_key = robot_manager.process_name(robot.gripper_name)
            self._targets[arm_key] = np.asarray(memory.state(arm_key), dtype=float).reshape(-1)
            self._targets[grip_key] = robot.command_pair(memory.state(grip_key)[0])
            self._grippers[grip_key] = robot

    def _raise_if_ended(self):
        if self.episode_ended or self.clock.take_action_cnt[0] >= self.clock.step_lim:
            raise NativeEpisodeEnded("Public episode ended or native action budget reached")

    def _normalized_action(self, targets):
        action = {}
        for key, target in targets.items():
            if key not in self._grippers:
                action[key] = target.tolist()
                continue
            robot = self._grippers[key]
            lower, upper = robot.gripper_scale
            value = float(np.clip((target[0] - lower) / (upper - lower), 0., 1.))
            if robot.gripper_sign != 1:
                value = 1. - value
            action[key] = [value]
        return action

    def execute(self, controls, phase):
        self._raise_if_ended()
        if not controls:
            return
        dense = []
        current = {key: value.copy() for key, value in self._targets.items()}
        for control in controls:
            for key, value in control.items():
                if key not in current:
                    raise ValueError(f"Unknown joint control key {key!r}")
                value = value.get("position") if isinstance(value, dict) else value
                if value is None:
                    continue
                target = np.asarray(value, dtype=float).reshape(-1)
                if target.shape != current[key].shape or not np.isfinite(target).all():
                    raise ValueError(f"Invalid joint command for {key!r}")
                current[key] = target
            dense.append({key: value.copy() for key, value in current.items()})
        arm_keys = set(self._targets) - set(self._grippers)
        has_arm_motion = any(key in arm_keys for control in controls for key in control)
        stride = self.interval * (self.time_scale if has_arm_motion else 1.)
        count = max(1, int(math.ceil(len(dense) / stride)))
        for index in range(count):
            self._raise_if_ended()
            source = min((index + 1) * stride, len(dense)) - 1.
            lower = max(0, int(math.floor(source)))
            upper = min(lower + 1, len(dense) - 1)
            fraction = max(0., source - lower)
            targets = {key: (1. - fraction) * dense[lower][key] + fraction * dense[upper][key]
                       for key in current}
            action = self._normalized_action(targets)
            reply = self.exchange(deepcopy(action))
            if not isinstance(reply, PublicStepReply) or type(reply.episode_ended) is not bool:
                raise TypeError("Exchange must return the restricted public reply contract")
            self.clock.take_action_cnt[0] += 1
            self._targets = {key: value.copy() for key, value in targets.items()}
            self.normalized_actions.append(deepcopy(action))
            self.action_sim_steps.append(self.clock.take_action_cnt[0] * self.interval)
            self.action_phases.append(str(phase))
            self.memory.update(reply.observation, sim_step=self.action_sim_steps[-1])
            self.episode_ended = reply.episode_ended
            self._raise_if_ended()


__all__ = ('NativeEpisodeEnded', 'PublicObservationSeries', 'PublicObservationMemory', 'project_observation', 'PublicRobotDescription', 'PublicRobotManager', 'PublicStepReply', 'PublicExecutionClock', 'PublicCommandBridge', 'STATE_WIDTHS', 'CAMERAS')
