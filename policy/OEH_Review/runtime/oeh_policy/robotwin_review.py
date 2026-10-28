"""RoboTwin's review port: OEH review mode (``oeh_policy.review``) over the released OpenWAM RoboTwin-Full chunk.

The method is RoboDojo's ``review.ReviewMethod`` with RoboTwin's action space (``action_space.RoboTwinEE16``): the
teacher reads and corrects the review layout (per arm EE pose xyz + wxyz and the commanded opening), and every row
leaves as RoboTwin's native EE16. The decisions mean the same motion as on RoboDojo; what differs is where RoboTwin's
native contract differs:

- ``student`` rows are OpenWAM's native EE16 rows, unchanged and unbounded: the rows the direct condition executes
  (RoboTwin's ``action_limits`` is ``{"native_action": "ee16", "finite": true}`` for every condition, and the paired
  comparisons require it equal, so the review condition cannot add a bound to OpenWAM's own rows).
- The workspace box the harness's and the teacher's own rows (hold, edit, eef, home) start in and are checked
  against is the frozen ``review_workspace``: the box ``oeh_policy.robotwin_bounds`` derives at freeze from the
  pinned RoboTwin files, the one the hybrid's slow primitives use. RoboDojo reads it from ``action_limits``.
- The previous-result record reads the executed native rows in the review layout (``executed_rows``).

The candidate is one fresh full OpenWAM chunk on the current observation (``robotwin.OpenWAMBackend.review_chunk``),
the official loop gets one row per call (``Runtime.next_action``) and the teacher's views are the three released RGB
cameras. The teacher's prompt set is ``teacher/robotwin/``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .action_space import RoboTwinEE16
from .actions import validate_action
from .profiles import PLACEHOLDER, check_box
from .review import ReviewMethod

WORKSPACE_KEYS = ("frame", "quaternion", "left_xyz_bounds", "right_xyz_bounds", "bounds_source")
# How the packet's proposal table names its frame: RoboTwin has one world frame, not RoboDojo's env-relative one.
PROPOSAL_FRAME = "RoboTwin world; position m; roll/pitch/yaw deg (extrinsic xyz); opening 1=open 0=closed"


def check_review_workspace(value) -> dict:
    """The frozen review workspace: RoboTwin's world frame, wxyz quaternions, each arm's finite 3x2 box and the pinned
    files it was derived from. An unfilled ``READ_FROM_PINNED`` placeholder fails: the box is never typed by hand."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"RoboTwin review mode requires review_workspace with exactly {sorted(WORKSPACE_KEYS)}")
    workspace = dict(value)
    if workspace["frame"] != "robotwin_world" or workspace["quaternion"] != "wxyz":
        raise ValueError("the RoboTwin review workspace is in RoboTwin's world frame with wxyz quaternions")
    source = workspace["bounds_source"]
    if isinstance(source, str) and PLACEHOLDER in source:
        raise ValueError("review_workspace.bounds_source still needs its value from the pinned sources")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("review_workspace.bounds_source must cite the pinned files the bounds come from")
    for arm in ("left", "right"):
        workspace[arm + "_xyz_bounds"] = check_box(workspace[arm + "_xyz_bounds"], f"review_workspace.{arm}_xyz_bounds")
    return workspace


class RoboTwinReviewMethod(ReviewMethod):
    """``review.ReviewMethod`` on RoboTwin: EE16 in and out, the review workspace from ``review_workspace``."""

    space = RoboTwinEE16()

    def __init__(self, config: dict):
        super().__init__(config)
        self.workspace = check_review_workspace(config.get("review_workspace"))

    def bounds(self, arm: str):
        return self.workspace[arm + "_xyz_bounds"]

    def check_row(self, row: Mapping) -> dict:
        """A correction or hold row (review layout) inside the frozen workspace, unit quaternion, opening in [0, 1]."""
        return validate_action(row, self.workspace)

    def student_rows(self, candidate: list, steps: int) -> list:
        """The first ``steps`` EE16 rows of OpenWAM's chunk exactly as it returned them (the runtime already checked
        each is 16 finite float32 values, RoboTwin's native contract): copies, never clipped or renormalized."""
        return [np.asarray(row, dtype=np.float32).reshape(-1).copy() for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        return self.space.review_rows(rows, state)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        fields = super().packet_state(candidate, state, truncated)
        if "frame" in fields["student_proposal"]:
            fields["student_proposal"]["frame"] = PROPOSAL_FRAME
        return fields
