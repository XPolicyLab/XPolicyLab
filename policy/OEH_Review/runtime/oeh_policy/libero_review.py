"""Review mode on the LIBERO family (LIBERO, LIBERO-Plus, LIBERO-Pro): the review method over the 7D OSC delta.

The single-arm port of ``oeh_policy.review.ReviewMethod`` (docs/benchmark-adapters/README.md, section 3). The decision
contract, the limits and every correction (``edit``, ``eef``, ``home``, ``hold``) are RoboDojo's, computed in the
review layout for the one arm, ``left`` (OpenWAM-α's single-arm block, ``oeh_policy.action_space``); only what
differs on LIBERO is overridden here:

- ``space``: ``oeh_policy.action_space.LiberoEEF10`` with the frozen OSC_POSE scale and the official conversion.
  Student rows leave as OpenWAM's own native rows; a correction's rows leave as a ``FollowingRows`` whose every row
  is computed from the newest achieved pose.
- ``seed``: LIBERO's state has no gripper command, so the ledger starts at the frozen settle action's opening (the
  released loop commands it for 30 steps before the first observation): 1.0, open.
- ``bounds``: the hybrid primitives' gross-error box, anchored at the episode-start position (the policy observation
  has no robot base, and its arena offset differs by scene; ``review_workspace``).
- ``check_row``, ``student_rows``, ``executed_rows``: one arm and native 7D OSC rows instead of RoboDojo's dict rows.
- ``packet_state``: the measured finger opening beside the commanded one, and the frame the proposal table is in.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .action_space import LiberoEEF10, Toward, check_review_row
from .libero_boundary import validate_native_action
from .osc_controller import OscScale
from .profiles import check_box
from .review import ReviewMethod

WORKSPACE_KEYS = ("bounds_frame", "workspace_bounds_m", "bounds_source")
WORKSPACE_FRAME = "home_anchored_base_axes"
# How the packet's proposal table reads on LIBERO (oeh_policy.review.proposal_view's `frame`).
PROPOSAL_FRAME = ("LIBERO world frame; position m; roll/pitch/yaw deg (extrinsic xyz); opening 1=open 0=closed; "
                  "OSC goals chained from the measured pose (the arm lags them)")


def check_review_workspace(value) -> dict:
    """The frozen review workspace: a box in metres around the episode-start position, on the robot base's axes
    (LIBERO translates the base per scene and never rotates it). It must contain that position."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"review_workspace needs exactly {list(WORKSPACE_KEYS)}")
    if value["bounds_frame"] != WORKSPACE_FRAME:
        raise ValueError(f"review_workspace.bounds_frame must be {WORKSPACE_FRAME} (the episode-start position)")
    box = check_box(value["workspace_bounds_m"], "review_workspace.workspace_bounds_m")
    if any(low > 0 or high < 0 for low, high in box):
        raise ValueError("review_workspace.workspace_bounds_m must contain the episode-start position")
    source = value["bounds_source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("review_workspace.bounds_source must say where the box comes from")
    return {"bounds_frame": WORKSPACE_FRAME, "workspace_bounds_m": box, "bounds_source": source}


class LiberoReviewMethod(ReviewMethod):
    """``ReviewMethod`` on one Franka Panda with LIBERO's OSC delta controller. ``codec`` is the official conversion
    (``LiberoBackend.encode``), through which every harness-made row is admitted exactly as the released client will
    convert it."""

    space = LiberoEEF10()

    def __init__(self, config: dict, codec):
        super().__init__(config)
        if codec is None or not callable(getattr(codec, "encode", None)):
            raise ValueError("LIBERO review rows need the official OSC conversion")
        self.space = LiberoEEF10(OscScale(config["osc_controller"]), codec)
        self.workspace = np.asarray(check_review_workspace(config["review_workspace"])["workspace_bounds_m"],
                                    dtype=float)
        self.settle_opening = self.space.openings(config["native"]["settle_action"])[self.space.arm]

    def seed(self, state):
        if self.home is None:
            self.home = self.space.review_state(state)
            self.gripper_commands = {self.space.arm: self.settle_opening}

    def bounds(self, arm: str):
        if self.home is None:
            raise ValueError("the episode-start pose has not been observed")
        return np.asarray(self.home[arm + "_ee_pose"][:3], dtype=float)[:, None] + self.workspace

    def check_row(self, row: Mapping) -> dict:
        arm = self.space.arm
        result = check_review_row(row, self.arms)
        bounds = np.asarray(self.bounds(arm))
        pose = result[arm + "_ee_pose"]
        if np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1]):
            raise ValueError("review row outside the frozen home-anchored workspace")
        if abs(float(np.linalg.norm(pose[3:])) - 1) > 1e-3:
            raise ValueError("review row needs a unit wxyz quaternion")
        if not 0 <= float(result[arm + "_ee_joint_state"][0]) <= 1:
            raise ValueError("review row opening outside [0, 1]")
        return result

    def student_rows(self, candidate: list, steps: int) -> list:
        return [validate_native_action(row) for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        """A correction's review targets (all of its rows, also those computed only after later steps), or the
        student rows chained from ``state`` (``LiberoEEF10.review_rows``)."""
        follow = getattr(rows, "follow", None)
        if isinstance(follow, Toward):
            return list(follow.targets)
        return self.space.review_rows(rows, state)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        packet = super().packet_state(candidate, state, truncated)
        measured = self.space.review_state(state)
        arm = self.space.arm
        packet["current_eef"][arm]["gripper_measured_opening"] = round(float(measured[arm + "_ee_joint_state"][0]), 3)
        if "frame" in packet["student_proposal"]:
            packet["student_proposal"]["frame"] = PROPOSAL_FRAME
        return packet
