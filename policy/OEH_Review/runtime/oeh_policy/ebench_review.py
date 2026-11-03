"""EBench's review port: OEH review mode (``oeh_policy.review``) over OpenWAM's released EBench chunk.

The method is RoboDojo's ``review.ReviewMethod`` with EBench's action space (``action_space.EBenchEE``): the teacher
reads and corrects the review layout (per arm EE pose xyz + wxyz and the opening) and every row leaves as GenManip's
native ``ee_pose`` action. The decisions mean the same motion as on RoboDojo; what differs is where EBench's native
contract differs:

- Frames: GenManip's poses are in each arm's own base frame (the cuRobo ``base_link`` of that arm's planner, which
  moves with the mobile base), so the workspace box and every target are per arm in that arm's frame
  (``review_workspace`` frame ``ebench_arm_base``; ``oeh_policy.ebench_bounds`` derives the boxes at freeze).
- ``student`` rows are OpenWAM's native rows, unchanged and unbounded, base motion included: the rows the direct
  condition executes (``action_limits`` is GenManip's native action for every condition).
- A correction's rows (``edit``, ``eef``, ``home``) and the harness's holds keep the base still (``base_motion``
  ``[0, 0, 0]``, ``EBenchEE.native_rows``), so an absolute arm target means one place while they run.
- The observation has no gripper command: the ledger starts at the measured opening (mean finger / 0.044 m) of the
  first observation (``ReviewMethod.seed`` reads the review state, which is the measured opening here) and then holds
  the last commanded opening; the packet shows the measured opening beside it (``gripper_measured_opening``).
- The packet adds the mobile base: its pose and its episode-start pose (``base``, [x m, y m, yaw deg] in the axes the
  robot was placed in), each candidate row's base displacement as GenManip applies it (clipped to 0.015 m and 1 deg
  per step, ``genmanip/core/robot/dualarm_manip.py`` ``delta_move_to``, lines 113-116) and its net displacement.

The teacher's prompt set is ``teacher/ebench/``.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .action_space import EBenchEE
from .actions import validate_action
from .ebench_boundary import native_action
from .profiles import PLACEHOLDER, check_box
from .review import ReviewMethod

WORKSPACE_KEYS = ("frame", "quaternion", "left_xyz_bounds", "right_xyz_bounds", "bounds_source")
WORKSPACE_FRAME = "ebench_arm_base"
# How the packet's proposal table names its frames.
PROPOSAL_FRAME = ("each arm's own base frame (moves with the mobile base); position m; roll/pitch/yaw deg (extrinsic "
                  "xyz); opening 1=open 0=closed; base dx/dy m and dyaw deg per step as GenManip applies them (clipped "
                  "to 0.015 m and 1 deg)")
BASE_CLIP_M = 0.015  # dualarm_manip.py delta_move_to: np.clip(delta_x, -0.015, 0.015), the same for delta_y
BASE_CLIP_DEG = 1.0  # np.clip(delta_yaw, -1, 1), degrees, converted with np.deg2rad
BASE_COLUMNS = ["base_dx", "base_dy", "base_dyaw"]


def check_review_workspace(value) -> dict:
    """The frozen review workspace: one finite 3x2 box per arm in that arm's own base frame, wxyz quaternions and the
    pinned files the boxes were derived from. An unfilled ``READ_FROM_PINNED`` placeholder fails."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"EBench review mode requires review_workspace with exactly {sorted(WORKSPACE_KEYS)}")
    workspace = dict(value)
    if workspace["frame"] != WORKSPACE_FRAME or workspace["quaternion"] != "wxyz":
        raise ValueError(f"the EBench review workspace is per arm in that arm's base frame ({WORKSPACE_FRAME}) with "
                         "wxyz quaternions")
    source = workspace["bounds_source"]
    if isinstance(source, str) and PLACEHOLDER in source:
        raise ValueError("review_workspace.bounds_source still needs its value from the pinned sources")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("review_workspace.bounds_source must cite the pinned files the bounds come from")
    for arm in ("left", "right"):
        workspace[arm + "_xyz_bounds"] = check_box(workspace[arm + "_xyz_bounds"], f"review_workspace.{arm}_xyz_bounds")
    return workspace


def base_view(base) -> list[float]:
    """GenManip's base joints [x m, y m, yaw rad] as the packet shows them: [x m, y m, yaw deg]."""
    x, y, yaw = (float(v) for v in np.asarray(base, dtype=float).reshape(3))
    return [round(x, 4), round(y, 4), round(math.degrees(yaw), 2)]


def applied_base(row) -> list[float]:
    """A native row's base displacement as GenManip applies it: each component clipped (dx, dy m; dyaw deg)."""
    dx, dy, dyaw = native_action(row)["base_motion"]
    return [float(np.clip(dx, -BASE_CLIP_M, BASE_CLIP_M)), float(np.clip(dy, -BASE_CLIP_M, BASE_CLIP_M)),
            float(np.clip(dyaw, -BASE_CLIP_DEG, BASE_CLIP_DEG))]


class EBenchReviewMethod(ReviewMethod):
    """``review.ReviewMethod`` on EBench: GenManip's ee_pose action in and out, per-arm frames and workspace boxes."""

    space = EBenchEE()

    def __init__(self, config: dict):
        super().__init__(config)
        self.workspace = check_review_workspace(config.get("review_workspace"))
        self.base_start = None

    def seed(self, state):
        if self.home is None:
            self.base_start = base_view(state["base"])
        super().seed(state)

    def bounds(self, arm: str):
        return self.workspace[arm + "_xyz_bounds"]

    def check_row(self, row: Mapping) -> dict:
        """A correction or hold row (review layout) inside its arm's frozen box, unit quaternion, opening in [0, 1]."""
        return validate_action(row, self.workspace)

    def student_rows(self, candidate: list, steps: int) -> list:
        """The first ``steps`` native rows of OpenWAM's chunk exactly as the released bridge built them (the runtime
        already checked each against GenManip's native contract): copies, never clipped or renormalized."""
        return [native_action(row) for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        return self.space.review_rows(rows, state)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        fields = super().packet_state(candidate, state, truncated)
        measured = self.space.review_state(state)
        for arm in self.arms:
            fields["current_eef"][arm]["gripper_measured_opening"] = round(
                float(measured[arm + "_ee_joint_state"][0]), 3)
        fields["base"] = {"pose": base_view(state["base"]), "start": self.base_start}
        view = fields["student_proposal"]
        if candidate:
            moves = [applied_base(row) for row in candidate]
            view["columns"] = view["columns"] + BASE_COLUMNS
            for line, move in zip(view["table"], moves):
                line += [round(move[0], 4), round(move[1], 4), round(move[2], 2)]
            net = np.sum(np.asarray(moves, dtype=float), axis=0)
            view["diagnostics"]["base"] = {"net_displacement": [round(float(net[0]), 4), round(float(net[1]), 4),
                                                                round(float(net[2]), 2)]}
            view["frame"] = PROPOSAL_FRAME
        return fields
