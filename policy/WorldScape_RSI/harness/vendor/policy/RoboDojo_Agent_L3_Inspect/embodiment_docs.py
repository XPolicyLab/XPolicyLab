"""Verified model-facing operating notes for the RoboDojo ARX X5 embodiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ARX_X5_JOINT_DOCS = """Two identical 6-DoF ARX X5 arms, prefixed left_ and right_, each with a
parallel-jaw gripper. Joint targets are absolute positions in radians; omitted
dimensions hold their observed value. Each arm has its own base frame: +x
points forward, which is the direction the folded gripper points at all-zero
joints, +y left, and +z up.
Joint guide (positive direction, identical for both arms):
- left_joint1 / right_joint1: base yaw about +z; positive turns
  counterclockwise when viewed from above and rotates the entire downstream arm.
- left_joint2 / right_joint2: shoulder pitch; positive raises the upper-arm
  endpoint from the all-zero folded configuration.
- left_joint3 / right_joint3: elbow pitch; positive unfolds and raises the
  forearm from the all-zero folded configuration.
- left_joint4 / right_joint4: wrist pitch; positive pitches the distal wrist
  upward from the all-zero configuration.
- left_joint5 / right_joint5: wrist yaw; positive turns the distal wrist
  clockwise when viewed from above in the all-zero configuration.
- left_joint6 / right_joint6: wrist roll about the outward tool axis; positive
  follows the right-hand rule and primarily changes jaw orientation, not reach.
- left_gripper / right_gripper: 0 is fully closed, 1 is fully open.
Proportions: upper arm 0.264 m, forearm 0.251 m, and link6 to fingertip 0.158 m.
Do not assume the outstretched layout of a standard 6-axis arm. At all-zero
joints this arm is folded back on itself: the upper arm lies horizontally
backward from the shoulder, the forearm doubles back forward over it, and the
wrist and gripper emerge forward slightly above shoulder height. The 0.515 m
of upper arm and forearm therefore leave the fingertips only about 0.26 m in
front of the base and about 0.16 m above it. Effects on fingertip position
depend on the complete joint configuration, so move deliberately and re-check
the fresh camera and joint observation after each motion."""


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return a rotation matrix for one normalized wxyz quaternion."""
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("mount quaternion must be finite and nonzero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _cardinal_axis(vector: np.ndarray, *, tolerance: float = 1e-3) -> str | None:
    """Name a world cardinal axis when a vector is sufficiently close to one."""
    axes = (
        ("+x", np.asarray([1.0, 0.0, 0.0])),
        ("-x", np.asarray([-1.0, 0.0, 0.0])),
        ("+y", np.asarray([0.0, 1.0, 0.0])),
        ("-y", np.asarray([0.0, -1.0, 0.0])),
        ("+z", np.asarray([0.0, 0.0, 1.0])),
        ("-z", np.asarray([0.0, 0.0, -1.0])),
    )
    for name, expected in axes:
        if np.allclose(vector, expected, atol=tolerance, rtol=0.0):
            return name
    return None


#: Opening words of the mounting appendix, so a surface that replaces the joint
#: cheat-sheet can still find where the live configuration starts.
MOUNTING_HEADER = "RoboDojo mounting"


def format_mounting_notes(mount_poses: Mapping[str, Sequence[float]]) -> str:
    """Render non-privileged robot-root poses as world-frame mounting notes."""
    lines = [f"{MOUNTING_HEADER} (base poses are fixed embodiment configuration):"]
    for side in ("left", "right"):
        raw_pose = mount_poses.get(side)
        if raw_pose is None:
            continue
        pose = np.asarray(raw_pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            continue
        try:
            rotation = _quaternion_matrix_wxyz(pose[3:])
        except ValueError:
            continue
        axis_names = tuple(_cardinal_axis(rotation[:, index]) for index in range(3))
        origin = ", ".join(f"{value:.3f}" for value in pose[:3])
        if all(name is not None for name in axis_names):
            axes = (
                f"base +x maps to world {axis_names[0]}, "
                f"base +y to world {axis_names[1]}, "
                f"base +z to world {axis_names[2]}"
            )
        else:
            columns = ", ".join(
                "(" + ", ".join(f"{value:.3f}" for value in rotation[:, index]) + ")"
                for index in range(3)
            )
            axes = f"base +x/+y/+z axes in world are {columns}"
        lines.append(f"- {side} base origin in world: ({origin}) m; {axes}.")
    return "\n".join(lines) if len(lines) > 1 else ""


def build_arx_x5_docs(mount_poses: Mapping[str, Sequence[float]] | None = None) -> str:
    """Combine invariant X5 facts with optional live mounting configuration."""
    mounting = format_mounting_notes(mount_poses or {})
    if not mounting:
        return ARX_X5_JOINT_DOCS
    return ARX_X5_JOINT_DOCS + "\n\n" + mounting
