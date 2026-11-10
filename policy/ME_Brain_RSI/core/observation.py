"""Select the public RoboDojo inputs used by the RGB policies."""

from collections.abc import Mapping


CAMERAS = frozenset(("cam_head", "cam_left_wrist", "cam_right_wrist"))
CAMERA_FIELDS = frozenset(("color", "shape", "intrinsic_matrix", "extrinsic_matrix", "extrinsics_matrix"))
STATE_FIELDS = frozenset(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in ("arm_joint_state", "ee_joint_state", "ee_pose", "tcp_pose", "delta_ee_pose")
)


def restrict_public_observation(observation):
    """Copy the allowed mappings without changing RGB or robot state values."""
    vision = {
        name: {key: value for key, value in frame.items() if key in CAMERA_FIELDS}
        for name, frame in observation.get("vision", {}).items()
        if name in CAMERAS and isinstance(frame, Mapping) and "color" in frame
    }
    state = {key: value for key, value in observation.get("state", {}).items() if key in STATE_FIELDS}
    result = {"vision": vision, "state": state}
    for key in ("instruction", "instructions"):
        text = observation.get(key)
        if isinstance(text, str) or (
            isinstance(text, (list, tuple)) and all(isinstance(item, str) for item in text)
        ):
            result[key] = text
    return result
