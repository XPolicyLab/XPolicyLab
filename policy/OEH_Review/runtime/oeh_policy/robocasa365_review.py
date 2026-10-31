"""Review mode on RoboCasa365: the review method over PandaOmron's native action12 (one Panda arm on a mobile base).

The single-arm port of ``oeh_policy.review.ReviewMethod`` (docs/benchmark-adapters/README.md, section 3). The decision
contract, the limits and every correction (``edit``, ``eef``, ``home``, ``hold``) are RoboDojo's, computed in the
review layout for the one arm, ``left`` (OpenWAM-α's single-arm block, ``oeh_policy.action_space``); only what
differs on RoboCasa365 is overridden here:

- ``space``: ``oeh_policy.action_space.RoboCasa365Action12``. Student rows leave as OpenWAM's own native rows, base
  motion and control mode included; a correction's rows leave as a ``FollowingRows`` whose every row is an arm-mode
  OSC delta from the newest achieved pose that holds the base and torso still. A correction never moves the base.
- The arm rides a base that OpenWAM's accepted rows may drive across the kitchen, so the review frame moves with it:
  the episode-start pose (``home``, the packet's ``episode_start_eef``) is the arm's start posture relative to the
  base, placed in the world at the base's current pose, and the workspace box is centred on the base's current centre
  (``review_workspace``). ``home`` returns the arm to its start posture where the base now stands.
- ``seed``: RoboCasa365's state carries no gripper command, so the ledger starts at the measured opening read as a
  command (the Panda gripper starts half open, 0.52, which reads open); every acknowledged row updates it.
- ``check_row``, ``student_rows``, ``executed_rows``: one arm and native action12 rows instead of RoboDojo's dict rows.
- ``packet_state``: the measured finger opening beside the commanded one, the frame the proposal table is in, the
  base's pose (``mobile_base``) and the base motion OpenWAM's proposal commands (``student_proposal.base_motion``),
  which its table, the arm relative to the base, leaves out.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .action_space import RoboCasa365Action12, Toward, check_review_row, validate_robocasa365_action
from .profiles import PLACEHOLDER, check_box
from .review import ReviewMethod

WORKSPACE_KEYS = ("bounds_frame", "workspace_bounds_m", "bounds_source")
WORKSPACE_FRAME = "base_centred_world_axes"
# One native step: RoboCasa365's control rate is 20 Hz (robocasa a07e365 environments/kitchen/kitchen.py
# control_freq=20; oeh_policy.robocasa365 checks it in the pinned file).
STEP_SECONDS = 0.05
# The base's command per unit of its native dims (oeh_policy.robocasa365 checks each in the pinned robosuite files):
# forward and side velocity in m/s and yaw rate in rad/s (omron_mobile_base.xml actuator ctrlrange, onto which the
# legacy base controller maps [-1, 1]) and the torso's height delta in m per step (JointPositionController's default
# output_max). Each dim is clipped to [-1, 1] on the way (the actuator's ctrlrange, the controller's input range).
BASE_SCALE = (1.0, 1.0, 1.5, 0.05)
# How the packet's proposal table reads on RoboCasa365 (oeh_policy.review.proposal_view's `frame`).
PROPOSAL_FRAME = ("RoboCasa365 world frame; position m; roll/pitch/yaw deg (extrinsic xyz); opening 1=open 0=closed; "
                  "the arm's OSC goals relative to the mobile base, chained from the measured pose and placed at the "
                  "base's current pose (any base motion the rows command is in base_motion, not in this table)")


def check_review_workspace(value) -> dict:
    """The frozen review workspace: a box in metres on the world's axes around the mobile base's current centre (the
    ``center`` site, where the arm is mounted), which OpenWAM's rows may move. It must contain that centre."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"review_workspace needs exactly {list(WORKSPACE_KEYS)}")
    if value["bounds_frame"] != WORKSPACE_FRAME:
        raise ValueError(f"review_workspace.bounds_frame must be {WORKSPACE_FRAME} (around the base's current centre)")
    box = check_box(value["workspace_bounds_m"], "review_workspace.workspace_bounds_m")
    if any(low > 0 or high < 0 for low, high in box):
        raise ValueError("review_workspace.workspace_bounds_m must contain the base's centre")
    source = value["bounds_source"]
    if not isinstance(source, str) or not source.strip() or PLACEHOLDER in source:
        raise ValueError("review_workspace.bounds_source must say where the box comes from")
    return {"bounds_frame": WORKSPACE_FRAME, "workspace_bounds_m": box, "bounds_source": source}


def heading_deg(rotation) -> float:
    """The base's heading: the direction of its forward (+x) axis in the world's x-y plane, degrees from +x toward
    +y."""
    rotation = np.asarray(rotation, dtype=float)
    return round(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])), 1)


def base_motion(candidate: list) -> dict:
    """What OpenWAM's rows command the base to do, summed over the rows as the controllers read them (each dim clipped
    to [-1, 1], times ``BASE_SCALE`` and, for the velocities, one step's 0.05 s): the forward and side travel in the
    base's own frame, the turn and the torso's height change, if every step ran at its command; and how many rows
    are in base mode."""
    rows = np.asarray([validate_robocasa365_action(row) for row in candidate], dtype=float)
    commands = np.clip(rows[:, 7:11], -1.0, 1.0) * np.asarray(BASE_SCALE)
    forward, side, yaw = (commands[:, :3] * STEP_SECONDS).sum(axis=0)
    return {"forward_m": round(float(forward), 3), "side_m": round(float(side), 3),
            "turn_deg": round(math.degrees(float(yaw)), 1), "torso_m": round(float(commands[:, 3].sum()), 3),
            "rows_in_base_mode": int((rows[:, 11] > 0).sum())}


class RoboCasa365ReviewMethod(ReviewMethod):
    """``ReviewMethod`` on RoboCasa365's PandaOmron: action12 in and out, the episode-start posture and the workspace
    box carried with the mobile base (module docstring)."""

    space = RoboCasa365Action12()

    def __init__(self, config: dict):
        super().__init__(config)
        self.workspace = np.asarray(check_review_workspace(config.get("review_workspace"))["workspace_bounds_m"],
                                    dtype=float)
        self.start = None  # the arm's start posture in the base frame: (position, rotation, opening)
        self.start_base = None  # the base centre's episode-start world position and rotation
        self.anchor = None  # the base centre's current world position

    def _at(self, state) -> None:
        """Move the review frame to ``state``'s base pose: the box's centre and the start posture placed there."""
        base_position, _ = self.space.base(state)
        position, rotation = self.space.world_pose(state, self.start[0], self.start[1])
        self.anchor = np.asarray(base_position, dtype=float)
        self.home = self.space._pose_row(position, rotation, self.start[2])

    def seed(self, state):
        if self.start is None:
            position, rotation = self.space.arm_in_base(state)
            opening = 1.0 if self.space.measured_opening(state) >= 0.5 else 0.0
            self.start = (np.asarray(position, dtype=float), np.asarray(rotation, dtype=float), opening)
            self.start_base = self.space.base(state)
            self.gripper_commands = {self.space.arm: opening}
        self._at(state)

    def bounds(self, arm: str):
        if self.anchor is None:
            raise ValueError("the episode-start pose has not been observed")
        return self.anchor[:, None] + self.workspace

    def check_row(self, row: Mapping) -> dict:
        arm = self.space.arm
        result = check_review_row(row, self.arms)
        bounds = np.asarray(self.bounds(arm))
        pose = result[arm + "_ee_pose"]
        if np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1]):
            raise ValueError("review row outside the frozen workspace around the mobile base")
        if abs(float(np.linalg.norm(pose[3:])) - 1) > 1e-3:
            raise ValueError("review row needs a unit wxyz quaternion")
        if not 0 <= float(result[arm + "_ee_joint_state"][0]) <= 1:
            raise ValueError("review row opening outside [0, 1]")
        return result

    def student_rows(self, candidate: list, steps: int) -> list:
        return [validate_robocasa365_action(row) for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        """A correction's review targets (all of its rows, also those computed only after later steps), or the
        student rows chained from ``state`` (``RoboCasa365Action12.review_rows``)."""
        follow = getattr(rows, "follow", None)
        if isinstance(follow, Toward):
            return list(follow.targets)
        return self.space.review_rows(rows, state)

    def rows(self, params: Mapping, candidate: list | None, state: Mapping) -> list:
        if self.start is not None:
            self._at(state)
        return super().rows(params, candidate, state)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        if self.start is not None:
            self._at(state)
        fields = super().packet_state(candidate, state, truncated)
        arm = self.space.arm
        fields["current_eef"][arm]["gripper_measured_opening"] = round(self.space.measured_opening(state), 3)
        proposal = fields["student_proposal"]
        if "frame" in proposal:
            proposal["frame"] = PROPOSAL_FRAME
        if candidate:
            proposal["base_motion"] = base_motion(candidate)
        base_position, base_rotation = self.space.base(state)
        start_position, start_rotation = self.start_base if self.start_base is not None else (base_position,
                                                                                             base_rotation)
        turned = (heading_deg(base_rotation) - heading_deg(start_rotation) + 180.0) % 360.0 - 180.0
        view = {"position": [round(float(v), 4) for v in base_position], "heading_deg": heading_deg(base_rotation),
                "moved_since_start_m": round(float(np.linalg.norm((base_position - start_position)[:2])), 4),
                "turned_since_start_deg": round(turned, 1)}
        # Packet order: the base right after the arm's measured pose.
        return {"current_eef": fields["current_eef"], "mobile_base": view,
                **{key: value for key, value in fields.items() if key != "current_eef"}}
