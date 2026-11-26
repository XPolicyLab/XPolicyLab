"""World-frame Cartesian dimensions the Inspect EEF agent names one at a time.

Orientation is three angles in degrees measured from ``TOP_DOWN_QUAT_WXYZ``,
the modal grasp orientation on this rig, rather than an absolute roll/pitch/yaw
about the world axes. That reference matters: expressed absolutely in the
conventional extrinsic-XYZ convention, a straight-down flange sits at
``pitch = +90``, exactly the convention's gimbal lock, so the single most
common orientation on this embodiment would lose a degree of freedom. Measured
from the reference it is all zeros instead, and the angles stay well away from
the singularity across every orientation seen in practice.

The ``_deg`` suffix is part of each angle's dimension name so the unit cannot
be misread as radians.
"""

from __future__ import annotations

import math

import numpy as np

#: Flange orientation of a straight-down grasp: the zero of the angle triple.
#: Its outward tool axis (flange body +x) points along world -z.
TOP_DOWN_QUAT_WXYZ: tuple[float, float, float, float] = (0.5, -0.5, 0.5, 0.5)

#: Fixed base origins in world coordinates, as RoboDojo mounts the two arms.
ARM_BASE_ORIGIN: dict[str, tuple[float, float, float]] = {
    "left": (-0.300, -0.450, 0.765),
    "right": (0.300, -0.450, 0.765),
}

#: Distance from the flange (link6) to the centre of the gripping face, along
#: the flange's own +x. Every position on this surface is that point, not the
#: flange, so the model never applies a tool offset itself.
#:
#: Measured from the RoboDojo asset ``Assets/Robots/x5/X5A.urdf``: the finger
#: joints sit at x = 0.08657 and the gripping face of ``link7`` spans x =
#: 0.0560 to 0.0710 in its own frame, so the face runs 0.1426 to 0.1576 from
#: the flange and its centre is 0.15007. The fingers are prismatic along the
#: jaw axis, so this distance does not change as the gripper opens.
GRASP_POINT_OFFSET_M = 0.1501

#: Depth of the gripping face along the tool axis, the span ``link7`` covers in
#: the measurement above. The commanded point is its centre, so half of this is
#: how far the jaws reach past that point towards whatever they close on.
JAW_DEPTH_M = 0.015

#: Largest flange-to-base distance observed across accepted planner requests
#: (0.624 m left, 0.642 m right), plus the grasp-point offset the flange
#: carries out in front of it, rounded up. Position bounds are this box, so
#: they never reject a pose the Cartesian planner would have accepted: all
#: 3543 accepted targets in the trace archive land inside it.
GRASP_REACH_M = 0.80

#: Table surface height in world z. The table cube is centred at z = 0.74
#: with thickness 0.05, so the top face is 0.765.
TABLE_SURFACE_Z = 0.765

#: Floor for a commanded grasp point, below the table surface so pressing down
#: on it is never blocked by the bounds check. The lowest grasp point ever
#: accepted by the planner is 0.740.
_MIN_GRASP_Z = 0.70

POSITION_AXES: tuple[str, ...] = ("x", "y", "z")
ANGLE_AXES: tuple[str, ...] = ("pitch_deg", "roll_deg", "yaw_deg")

#: The canonical Euler domain: holding the middle angle (roll) in [-90, 90]
#: covers every orientation exactly once, so no reachable orientation is
#: outside these bounds and none has two spellings inside them.
ANGLE_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch_deg": (-180.0, 180.0),
    "roll_deg": (-90.0, 90.0),
    "yaw_deg": (-180.0, 180.0),
}

GRIPPER_BOUNDS = (0.0, 1.0)

ARMS: tuple[str, ...] = ("left", "right")

_GIMBAL_EPS = 1e-7


def arm_labels(arm: str) -> tuple[str, ...]:
    """Name every dimension of one arm in the order the tool advertises them."""
    return (
        *(f"{arm}_{axis}" for axis in POSITION_AXES),
        *(f"{arm}_{axis}" for axis in ANGLE_AXES),
        f"{arm}_gripper",
    )


def labels() -> tuple[str, ...]:
    """Name all 14 Cartesian dimensions, left arm first."""
    return tuple(label for arm in ARMS for label in arm_labels(arm))


def state_reference_keys() -> tuple[str, ...]:
    """Name the observation key each Cartesian dimension is measured from.

    An absolute-target space needs a proprioceptive reference aligned with the
    action space, and the usual way to have one is to declare a single
    observation field shaped like the action. This surface has no such field:
    the 14-vector is synthesized per step from the flange pose, shifted to the
    grasp point, and the gripper channel. Recording where each entry came from
    keeps that synthesis auditable from a trace alone, instead of only from the
    code.
    """
    return tuple(
        f"{arm}_ee_joint_state" if axis == "gripper" else f"{arm}_ee_pose"
        for arm, _, axis in (label.partition("_") for label in labels())
    )


def bounds(label: str) -> tuple[float, float]:
    """Return the advertised ``[low, high]`` for one dimension name."""
    arm, _, axis = label.partition("_")
    if axis == "gripper":
        return GRIPPER_BOUNDS
    if axis in ANGLE_BOUNDS:
        return ANGLE_BOUNDS[axis]
    origin = ARM_BASE_ORIGIN[arm]
    index = POSITION_AXES.index(axis)
    low = origin[index] - GRASP_REACH_M
    high = origin[index] + GRASP_REACH_M
    if axis == "z":
        low = _MIN_GRASP_Z
    return (round(low, 3), round(high, 3))


def bounds_text() -> str:
    """Render every dimension's bounds for the tool description."""
    return ", ".join(
        f"{label}: [{low:.4g}, {high:.4g}]"
        for label, (low, high) in ((label, bounds(label)) for label in labels())
    )


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two wxyz quaternions (``left`` applied after ``right``)."""
    w0, x0, y0, z0 = (float(value) for value in left)
    w1, x1, y1, z1 = (float(value) for value in right)
    return np.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Invert a unit wxyz quaternion."""
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    """Return the unit wxyz quaternion, raising on a degenerate input."""
    array = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        raise ValueError("a quaternion needs four finite numbers")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        raise ValueError("a quaternion must have non-zero norm")
    return array / norm


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion, columns being the body axes."""
    w, x, y, z = quat_normalize(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


_REFERENCE_QUAT = np.asarray(TOP_DOWN_QUAT_WXYZ, dtype=np.float64)


def _axis_quat(axis: int, radians: float) -> np.ndarray:
    half = radians / 2.0
    quaternion = np.zeros(4, dtype=np.float64)
    quaternion[0] = math.cos(half)
    quaternion[axis + 1] = math.sin(half)
    return quaternion


def angles_to_quat(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Build the wxyz flange quaternion for one angle triple.

    The angles are extrinsic rotations about the world axes applied to the
    top-down reference: pitch about world +x, then roll about world +y, then
    yaw about world +z.
    """
    rotation = _axis_quat(0, math.radians(float(pitch_deg)))
    rotation = quat_multiply(_axis_quat(1, math.radians(float(roll_deg))), rotation)
    rotation = quat_multiply(_axis_quat(2, math.radians(float(yaw_deg))), rotation)
    return quat_multiply(rotation, _REFERENCE_QUAT)


def quat_to_angles(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Recover ``(pitch_deg, roll_deg, yaw_deg)`` from a wxyz flange quaternion."""
    relative = quat_multiply(
        quat_normalize(quaternion), quat_conjugate(_REFERENCE_QUAT)
    )
    matrix = quat_to_matrix(relative)
    roll = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(roll)) > _GIMBAL_EPS:
        pitch = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        # Roll at +/-90 couples pitch and yaw; pin yaw and put the whole
        # remaining rotation on pitch so the triple still round-trips.
        yaw = 0.0
        sign = 1.0 if matrix[2, 0] < 0 else -1.0
        pitch = math.atan2(sign * float(matrix[0, 1]), float(matrix[1, 1]))
    # Adding zero folds -0.0 into 0.0 so the reference orientation renders as
    # plain zeros in the observation the model reads.
    return (
        math.degrees(pitch) + 0.0,
        math.degrees(roll) + 0.0,
        math.degrees(yaw) + 0.0,
    )


def tool_axis(quaternion: np.ndarray) -> np.ndarray:
    """World direction the jaw points along: the flange's own +x axis."""
    return quat_to_matrix(quaternion)[:, 0]


def pose_to_values(pose: np.ndarray) -> dict[str, float]:
    """Split a ``[x, y, z, qw, qx, qy, qz]`` flange pose into named axis values.

    The position comes out at the grasp point rather than the flange. This
    function and :func:`values_to_pose` are the only two places the offset is
    applied, so everything above them -- observation, targets, arrival check --
    speaks one frame and the model is never asked to add a tool offset itself.
    """
    array = np.asarray(pose, dtype=np.float64).reshape(-1)
    if array.size < 7 or not np.isfinite(array[:7]).all():
        raise ValueError("a flange pose needs seven finite numbers")
    pitch, roll, yaw = quat_to_angles(array[3:7])
    position = array[:3] + GRASP_POINT_OFFSET_M * tool_axis(array[3:7])
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "pitch_deg": pitch,
        "roll_deg": roll,
        "yaw_deg": yaw,
    }


def values_to_pose(values: dict[str, float]) -> np.ndarray:
    """Rebuild the ``[x, y, z, qw, qx, qy, qz]`` flange pose the planner takes.

    Named positions are grasp points, so the offset comes back off here. The
    direction it comes off along is the *target* orientation's tool axis, which
    is what makes a rotation about the grasp point stay put: the flange swings
    around it rather than the grasp point swinging around the flange.
    """
    quaternion = angles_to_quat(
        values["pitch_deg"], values["roll_deg"], values["yaw_deg"]
    )
    grasp = np.asarray(
        [values["x"], values["y"], values["z"]],
        dtype=np.float64,
    )
    flange = grasp - GRASP_POINT_OFFSET_M * tool_axis(quaternion)
    return np.concatenate([flange, quaternion])


def orientation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    """Angle in degrees between two wxyz orientations."""
    dot = float(abs(np.dot(quat_normalize(left), quat_normalize(right))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
