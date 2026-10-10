"""Mobile state/action layout: 69 raw values, 75 with rotation-6D poses."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

MODEL_DIM = 75
RAW_DIM = 69
FPS = 30.0
ROBOT_TYPE = "m92uw"
EMBODIMENT_TAG = "mobile_m92uw"

# HDF5 key, width, and whether the value is a [x,y,z,qw,qx,qy,qz] pose.
KEYS = (
    ("left_arm_joint_states", 7, False),
    ("right_arm_joint_states", 7, False),
    ("left_ee_joint_states", 12, False),
    ("right_ee_joint_states", 12, False),
    ("left_base_ee_poses", 7, True),
    ("right_base_ee_poses", 7, True),
    ("waist_joint_states", 2, False),
    ("leg_joint_states", 2, False),
    ("head_base_poses", 7, True),
    ("root_linear_velocity", 3, False),
    ("root_angular_velocity", 3, False),
)

# The extended packet holds three unpredicted world-frame poses at observed values.
EXTENDED_ACTION_KEYS = (
    "left_arm_joint_states", "right_arm_joint_states",
    "left_ee_joint_states", "right_ee_joint_states",
    "base_head_poses", "head_poses",
    "left_base_ee_poses", "left_ee_poses",
    "leg_joint_states", "right_base_ee_poses", "right_ee_poses",
    "root_angular_velocity", "root_linear_velocity", "waist_joint_states",
)
EXTENDED_ACTION_DIMS = {
    "left_arm_joint_states": 7, "right_arm_joint_states": 7,
    "left_ee_joint_states": 12, "right_ee_joint_states": 12,
    "base_head_poses": 7, "head_poses": 7,
    "left_base_ee_poses": 7, "left_ee_poses": 7,
    "leg_joint_states": 2, "right_base_ee_poses": 7,
    "right_ee_poses": 7, "root_angular_velocity": 3,
    "root_linear_velocity": 3, "waist_joint_states": 2,
}
EXTENDED_EXTRA_POSES = ("head_poses", "left_ee_poses", "right_ee_poses")


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_ALIASES = {
    "cam_head": ("primary", "head_infra1", "head_infra", "head_camera"),
    "cam_left_wrist": ("wrist_left", "left_camera"),
    "cam_right_wrist": ("wrist_right", "right_camera"),
}


def matches_layout(metadata: Mapping[str, object]) -> bool:
    """Recognize the explicit mobile dimensions, including older layout labels."""
    if metadata.get("layout") == EMBODIMENT_TAG:
        return True
    dims = metadata.get("robot_action_dim_info", {})
    return (
        isinstance(dims, Mapping)
        and dims.get("model_dim") == MODEL_DIM
        and dims.get("raw_dim") == RAW_DIM
        and dims.get("arm_dim") == [7, 7]
        and dims.get("ee_dim") == [12, 12]
        and metadata.get("state_dim") == MODEL_DIM
        and metadata.get("action_dim") == MODEL_DIM
    )


def uses_hdf5(metadata: Mapping[str, object]) -> bool:
    return matches_layout(metadata) and str(metadata.get("storage_format", "")).endswith("_hdf5")


def decode_camera_frame(image_bits) -> np.ndarray:
    """Decode stored image buffers through the shared RGB decoder."""
    from XPolicyLab.utils.process_data import decode_image_bit

    return decode_image_bit(image_bits)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, z = [q[..., i] for i in range(4)]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                      2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                      2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                      1 - 2 * (x * x + y * y)], axis=-1),
        ], axis=-2,
    )


def pose_to_model(pose: np.ndarray) -> np.ndarray:
    """Convert wxyz poses to position + rotation-6D."""
    pose = np.asarray(pose, dtype=np.float64)
    matrix = _quat_to_matrix(pose[..., 3:7])
    rot6d = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    return np.concatenate([pose[..., :3], rot6d], axis=-1)


def pack(values: Mapping[str, object]) -> np.ndarray:
    """Pack HDF5/runtime fields into ``(..., 75)`` float32 vectors."""
    nested = values.get("state") if isinstance(values, Mapping) else None
    if isinstance(nested, Mapping):
        values = {**nested, **values}
    parts = []
    lead = None
    for key, width, is_pose in KEYS:
        # HDF5 fields are plural; runtime fields may be singular.
        singular = key[:-1] if key.endswith("s") else key
        aliases = ()
        if key == "head_base_poses":
            # Older Mobile clients called this field ``base_head_pose``.
            aliases = ("base_head_pose", "base_head_poses")
        candidates = (key, singular, *aliases, f"state/{key}", f"state/{singular}",
                      *(f"state/{alias}" for alias in aliases),
                      f"state.{key}", f"state.{singular}",
                      *(f"state.{alias}" for alias in aliases))
        value = next((values[name] for name in candidates
                      if name in values and values[name] is not None), None)
        if value is None:
            raise KeyError(f"Missing Mobile field {key!r}")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != width:
            raise ValueError(f"{key}: expected last dimension {width}, got {array.shape}")
        if lead is None:
            lead = array.shape[:-1]
        elif array.shape[:-1] != lead:
            raise ValueError(f"Mobile fields have inconsistent leading shapes: {key} {array.shape[:-1]} != {lead}")
        parts.append(pose_to_model(array) if is_pose else array)
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != MODEL_DIM or not np.isfinite(result).all():
        raise ValueError("Mobile packed values are not finite 75-dimensional vectors")
    return result


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Vectorized Shepperd conversion with scalar-first (wxyz) output."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = np.stack([trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], axis=-1)
    pivot = np.argmax(candidates, axis=-1)
    q = np.zeros(m.shape[:-2] + (4,), dtype=np.float64)

    def put(mask, w, x, y, z):
        q[..., 0][mask], q[..., 1][mask] = w[mask], x[mask]
        q[..., 2][mask], q[..., 3][mask] = y[mask], z[mask]

    s = np.sqrt(np.clip(1 + trace, 1e-8, None)) * 2
    put(pivot == 0, .25 * s, (m[..., 2, 1] - m[..., 1, 2]) / s,
        (m[..., 0, 2] - m[..., 2, 0]) / s, (m[..., 1, 0] - m[..., 0, 1]) / s)
    s = np.sqrt(np.clip(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-8, None)) * 2
    put(pivot == 1, (m[..., 2, 1] - m[..., 1, 2]) / s, .25 * s,
        (m[..., 0, 1] + m[..., 1, 0]) / s, (m[..., 0, 2] + m[..., 2, 0]) / s)
    s = np.sqrt(np.clip(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1e-8, None)) * 2
    put(pivot == 2, (m[..., 0, 2] - m[..., 2, 0]) / s,
        (m[..., 0, 1] + m[..., 1, 0]) / s, .25 * s,
        (m[..., 1, 2] + m[..., 2, 1]) / s)
    s = np.sqrt(np.clip(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], 1e-8, None)) * 2
    put(pivot == 3, (m[..., 1, 0] - m[..., 0, 1]) / s,
        (m[..., 0, 2] + m[..., 2, 0]) / s,
        (m[..., 1, 2] + m[..., 2, 1]) / s, .25 * s)
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    return np.where(q[..., :1] < 0, -q, q)


def unpack(vector: np.ndarray, *, key_style: str = "singular") -> dict[str, np.ndarray]:
    """Unpack canonical vectors into singular runtime action/state fields."""
    if key_style not in {"singular", "plural"}:
        raise ValueError("key_style must be 'singular' or 'plural'")
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[-1] != MODEL_DIM:
        raise ValueError(f"Expected last dimension {MODEL_DIM}, got {vector.shape}")
    output = {}
    cursor = 0
    for key, width, is_pose in KEYS:
        model_width = 9 if is_pose else width
        part = vector[..., cursor:cursor + model_width]
        cursor += model_width
        if is_pose:
            matrix = np.stack([part[..., 3:6], part[..., 6:9]], axis=-1)
            # Re-orthogonalize predicted rotation columns.
            first = matrix[..., :, 0]
            first = first / np.clip(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8, None)
            second = matrix[..., :, 1] - np.sum(first * matrix[..., :, 1], axis=-1, keepdims=True) * first
            second = second / np.clip(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8, None)
            third = np.cross(first, second)
            pose = np.concatenate([part[..., :3], _matrix_to_quat(np.stack([first, second, third], axis=-1))], axis=-1)
            name = key[:-1] if key.endswith("s") else key
            output[name if key_style == "singular" else key] = pose.astype(np.float32)
        else:
            name = key[:-1] if key.endswith("s") else key
            output[name if key_style == "singular" else key] = part.astype(np.float32)
    return output


def _singular_key(key: str) -> str:
    if key.endswith("states") or key.endswith("poses"):
        return key[:-1]
    return key


def _field(values: Mapping[str, object], key: str):
    """Find a field under plural, singular or legacy aliases."""
    nested = values.get("state") if isinstance(values, Mapping) else None
    if isinstance(nested, Mapping):
        values = {**nested, **values}
    singular = _singular_key(key)
    aliases = ()
    if key == "base_head_poses":
        aliases = ("head_base_poses", "head_base_pose")
    candidates = (key, singular, *aliases,
                  *(f"state/{name}" for name in (key, singular, *aliases)),
                  *(f"state.{name}" for name in (key, singular, *aliases)))
    for name in candidates:
        if name in values and values[name] is not None:
            return np.asarray(values[name], dtype=np.float32).reshape(-1)
    return None


def to_extended_action(
    action: Mapping[str, object],
    observed_state: Mapping[str, object],
    *,
    key_style: str = "singular",
) -> dict[str, np.ndarray]:
    """Expand the action to 14 fields, holding three observed world-frame poses."""
    if key_style not in {"singular", "plural"}:
        raise ValueError("key_style must be 'singular' or 'plural'")
    canonical = {}
    for name, value in action.items():
        key = str(name)
        if key.endswith("state") or key.endswith("pose"):
            key = key + "s"
        if key == "head_base_poses":
            key = "base_head_poses"
        canonical[key] = np.asarray(value, dtype=np.float32).reshape(-1)
    for key in EXTENDED_ACTION_KEYS:
        if key not in canonical:
            if key not in EXTENDED_EXTRA_POSES:
                raise KeyError(f"MoPA action has no extended mobile field {key!r}")
            value = _field(observed_state, key)
            if value is None:
                raise KeyError(
                    f"extended mobile compatibility requires observed field {key!r} "
                    "to hold its world-frame pose"
                )
            canonical[key] = value
        expected = EXTENDED_ACTION_DIMS[key]
        if canonical[key].shape != (expected,) or not np.isfinite(canonical[key]).all():
            raise ValueError(f"extended mobile action {key!r} expected finite shape ({expected},), got {canonical[key].shape}")
    return {
        (_singular_key(key) if key_style == "singular" else key): canonical[key].astype(np.float32)
        for key in EXTENDED_ACTION_KEYS
    }


def metadata(cameras=CAMERAS, image_size=(224, 224), *, env_cfg_type="m92uw") -> dict:
    return {
        "layout": EMBODIMENT_TAG,
        "env_cfg_type": str(env_cfg_type),
        "action_type": "joint",
        "robot_action_dim_info": {"model_dim": MODEL_DIM, "raw_dim": RAW_DIM,
                                   "arm_dim": [7, 7], "ee_dim": [12, 12]},
        "state_dim": MODEL_DIM,
        "action_dim": MODEL_DIM,
        "manipulation_slice": [0, 56],
        "mobility_slice": [56, 75],
        "mobility_fields": ["waist_joint_states", "leg_joint_states",
                             "head_base_poses", "root_linear_velocity",
                             "root_angular_velocity"],
        "cameras": list(cameras),
        "image_size": list(image_size),
        "fps": FPS,
        "pose_format": "[x, y, z, qw, qx, qy, qz] in base_link",
    }
