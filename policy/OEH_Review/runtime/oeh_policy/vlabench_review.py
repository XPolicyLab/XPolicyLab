"""VLABench's review port: OEH review mode (``oeh_policy.review``) over the released OpenWAM VLABench chunk.

The single-arm port of ``review.ReviewMethod`` (docs/benchmark-adapters/README.md, section 3). The decision contract,
the limits and every correction (``edit``, ``eef``, ``home``, ``hold``) are RoboDojo's, computed in the review layout
for the one arm, ``left`` (OpenWAM-α's single-arm block); only what differs on VLABench is overridden here:

- ``space``: ``oeh_policy.action_space.VLABenchEE``. VLABench takes an absolute end-effector target per native step,
  so a correction's rows leave as absolute targets, never chained deltas.
- ``seed``: VLABench's state carries no gripper command (``ee_state``'s flag is the inverted upstream reading, never
  read here), so the ledger starts at the opening the robot starts with: both fingers at 0.04, fully open (configs/
  robot_config.json franka ``default_qpos``, written by ``Franka.initialize_episode``), 1.0. The first observation
  must show the robot base where the frozen workspace says.
- ``bounds``: the frozen ``review_workspace`` box, which ``oeh_policy.vlabench_bounds`` derives from the pinned files
  (world frame, the Panda's reach about its shoulder).
- ``check_row``, ``student_rows``, ``executed_rows``: one arm and native 8-value rows instead of RoboDojo's dict rows;
  OpenWAM's own rows leave unchanged and unbounded (the direct condition's rows).
- ``packet_state``: the measured finger opening beside the commanded one, and the frame the proposal table is in.

The candidate is one fresh full OpenWAM chunk on the current observation (``vlabench.OpenWAMBackend.review_chunk``),
the official loop gets one row per ``predict`` call (``Runtime.next_action``) and the teacher's views are the three
cameras the released client sends OpenWAM. The teacher's prompt set is ``teacher/vlabench/``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .action_space import VLABenchEE, check_review_row, vlabench_row
from .profiles import PLACEHOLDER, check_box
from .review import ReviewMethod

WORKSPACE_KEYS = ("frame", "quaternion", "base_position", "left_xyz_bounds", "bounds_source")
FRAME = "vlabench_world"
# The robot base the first observation must report (``robot_frame``) against the frozen one, in metres.
BASE_TOLERANCE_M = 1e-6
# The opening the ledger starts at: VLABench's Franka default_qpos sets both fingers to 0.04, fully open.
START_OPENING = 1.0
# How the packet's proposal table names its frame (oeh_policy.review.proposal_view's `frame`).
PROPOSAL_FRAME = "VLABench world; position m; roll/pitch/yaw deg (extrinsic xyz); opening 1=open 0=closed"


def check_review_workspace(value) -> dict:
    """The frozen review workspace: VLABench's world frame, wxyz quaternions, the robot base it was derived for, the
    arm's finite 3x2 box and the pinned files it comes from. An unfilled ``READ_FROM_PINNED`` placeholder fails: the
    box is never typed by hand."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"VLABench review mode requires review_workspace with exactly {sorted(WORKSPACE_KEYS)}")
    workspace = dict(value)
    if workspace["frame"] != FRAME or workspace["quaternion"] != "wxyz":
        raise ValueError("the VLABench review workspace is in VLABench's world frame with wxyz quaternions")
    source = workspace["bounds_source"]
    if isinstance(source, str) and PLACEHOLDER in source:
        raise ValueError("review_workspace.bounds_source still needs its value from the pinned sources")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("review_workspace.bounds_source must cite the pinned files the bounds come from")
    base = workspace["base_position"]
    if isinstance(base, str) and PLACEHOLDER in base:
        raise ValueError("review_workspace.base_position still needs its value from the pinned sources")
    try:
        base = np.asarray(base, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("review_workspace.base_position must be three finite numbers") from None
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("review_workspace.base_position must be three finite numbers")
    workspace["base_position"] = base.tolist()
    workspace["left_xyz_bounds"] = check_box(workspace["left_xyz_bounds"], "review_workspace.left_xyz_bounds")
    return workspace


class VLABenchReviewMethod(ReviewMethod):
    """``review.ReviewMethod`` on VLABench's Franka: native 8-value absolute rows in and out, the review workspace
    from ``review_workspace``."""

    space = VLABenchEE()

    def __init__(self, config: dict):
        super().__init__(config)
        self.workspace = check_review_workspace(config.get("review_workspace"))

    def seed(self, state):
        if self.home is None:
            base = np.asarray(state["robot_frame"], dtype=float).reshape(-1)
            frozen = np.asarray(self.workspace["base_position"], dtype=float)
            if base.shape != (3,) or not np.all(np.abs(base - frozen) <= BASE_TOLERANCE_M):
                raise ValueError(f"the robot base {base.tolist()} is not the frozen workspace's {frozen.tolist()}")
            self.home = self.space.review_state(state)
            self.gripper_commands = {self.space.arm: START_OPENING}

    def bounds(self, arm: str):
        return self.workspace[arm + "_xyz_bounds"]

    def check_row(self, row: Mapping) -> dict:
        """A correction or hold row (review layout) inside the frozen workspace, unit quaternion, opening in [0, 1]."""
        arm = self.space.arm
        result = check_review_row(row, self.arms)
        bounds = np.asarray(self.bounds(arm), dtype=float)
        pose = result[arm + "_ee_pose"]
        if np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1]):
            raise ValueError("review row outside the frozen VLABench workspace")
        if abs(float(np.linalg.norm(pose[3:])) - 1) > 1e-3:
            raise ValueError("review row needs a unit wxyz quaternion")
        if not 0 <= float(result[arm + "_ee_joint_state"][0]) <= 1:
            raise ValueError("review row opening outside [0, 1]")
        return result

    def student_rows(self, candidate: list, steps: int) -> list:
        """The first ``steps`` rows of OpenWAM's chunk exactly as the released client converted them (8 finite
        float64 values each, VLABench's native contract): copies, never clipped or renormalized."""
        return [vlabench_row(row) for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        return self.space.review_rows(rows, state)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        packet = super().packet_state(candidate, state, truncated)
        measured = self.space.review_state(state)
        arm = self.space.arm
        packet["current_eef"][arm]["gripper_measured_opening"] = round(float(measured[arm + "_ee_joint_state"][0]), 3)
        if "frame" in packet["student_proposal"]:
            packet["student_proposal"]["frame"] = PROPOSAL_FRAME
        return packet
