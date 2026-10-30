"""Robot-interface notes for the Cartesian ARX X5 surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.embodiment_docs import (
    MOUNTING_HEADER,
    format_mounting_notes,
)

ARX_X5_EEF_DOCS = """Two ARX X5 arms with parallel-jaw grippers.
Positions are world-frame grasp points in metres. Orientations follow the
current Tool interface. Raw robot end-effector poses refer to the flange;
use the robot geometry conversion once when a grasp point is required.
Gripper commands range from 0 (closed) to 1 (open). Their reported values are
command targets, not measured aperture, contact or object possession.
Use the live instruction, current RGB, robot state and execution feedback to
choose targets. Robot reachability checks do not establish object clearance.
The environment determines completion from executed actions."""


def build_arx_x5_eef_docs(mount_poses: Mapping[str, Sequence[float]] | None = None) -> str:
    """Cartesian operating notes, with the same live mounting appendix as joint docs."""
    mounting = format_mounting_notes(mount_poses or {})
    if not mounting:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + mounting


def eef_docs_from_joint_docs(joint_docs: str) -> str:
    """Keep the live mounting appendix, replace the joint cheat-sheet."""
    _, found, rest = joint_docs.partition(MOUNTING_HEADER)
    if not found:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + MOUNTING_HEADER + rest
