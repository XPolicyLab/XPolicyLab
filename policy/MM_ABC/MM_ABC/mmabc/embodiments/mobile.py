"""Mobile m92uw keys mapped to 75-d and 80-d state/action vectors.

Storage and runtime poses use [x, y, z, qw, qx, qy, qz]. The 75-d model
uses position plus rotation-6D; the 80-d variant preserves pretraining slots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mmabc.canonical import rotation as rot

EMBODIMENT_TAG = "mobile_m92uw"
ROBOT_TYPE = "m92uw"
FPS = 30.0

# (raw key, raw dim, kind). Order here is the order in the model vector.
KEYS: tuple[tuple[str, int, str], ...] = (
    ("left_arm_joint_states", 7, "vec"),
    ("right_arm_joint_states", 7, "vec"),
    ("left_ee_joint_states", 12, "vec"),
    ("right_ee_joint_states", 12, "vec"),
    ("left_base_ee_poses", 7, "pose"),
    ("right_base_ee_poses", 7, "pose"),
    ("waist_joint_states", 2, "vec"),
    ("leg_joint_states", 2, "vec"),
    ("head_base_poses", 7, "pose"),
    ("root_linear_velocity", 3, "vec"),
    ("root_angular_velocity", 3, "vec"),
)

# Canonical view slot -> Mobile camera. cam_chest_fisheye is not used.
CAMERAS: dict[str, str] = {
    "primary": "cam_head",
    "wrist_left": "cam_left_wrist",
    "wrist_right": "cam_right_wrist",
}

# Other spellings a runtime observation may carry for the same quantity.
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "head_base_poses": ("base_head_poses",),
}
CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "cam_head": ("head_infra1", "head_infra", "head_camera"),
    "cam_left_wrist": ("left_camera",),
    "cam_right_wrist": ("right_camera",),
}


def _model_width(dim: int, kind: str) -> int:
    return 9 if kind == "pose" else dim


@dataclass(frozen=True)
class KeySlot:
    key: str
    raw_dim: int
    kind: str
    start: int
    end: int


def _build_slots() -> tuple[KeySlot, ...]:
    slots, cursor = [], 0
    for key, dim, kind in KEYS:
        width = _model_width(dim, kind)
        slots.append(KeySlot(key, dim, kind, cursor, cursor + width))
        cursor += width
    return tuple(slots)


SLOTS = _build_slots()
MODEL_DIM = SLOTS[-1].end  # 75
RAW_DIM = sum(dim for _, dim, _ in KEYS)  # 69


def singular(key: str) -> str:
    """``left_arm_joint_states`` -> ``left_arm_joint_state``; velocities unchanged."""
    if key.endswith("_states"):
        return key[: -len("_states")] + "_state"
    if key.endswith("_poses"):
        return key[: -len("_poses")] + "_pose"
    return key


def pose_to_model(pose: np.ndarray) -> np.ndarray:
    """(..., 7) [x,y,z,qw,qx,qy,qz] -> (..., 9) [x,y,z, rot6d]."""
    pose = np.asarray(pose, dtype=np.float64)
    r6 = rot.matrix_to_rot6d(rot.quat_wxyz_to_matrix(pose[..., 3:7]))
    return np.concatenate([pose[..., :3], r6], axis=-1)


def model_to_pose(vec: np.ndarray) -> np.ndarray:
    """(..., 9) -> (..., 7), quaternion wxyz with w >= 0."""
    vec = np.asarray(vec, dtype=np.float64)
    q = rot.matrix_to_quat_wxyz(rot.rot6d_to_matrix(vec[..., 3:9]))
    return np.concatenate([vec[..., :3], q], axis=-1)


def _lookup(values: Mapping[str, object], key: str):
    for cand in (key, singular(key), *KEY_ALIASES.get(key, ()),
                 *(singular(a) for a in KEY_ALIASES.get(key, ()))):
        for name in (cand, f"state.{cand}", f"state/{cand}"):
            if name in values:
                return values[name]
    return None


def pack(values: Mapping[str, object], *, allow_missing: bool = False) -> np.ndarray:
    """Raw per-key arrays -> model vector.

    Accepts plural (HDF5) or singular (runtime) keys, each shaped ``(D,)`` or
    ``(T, D)``. Returns ``(75,)`` or ``(T, 75)`` float64. A missing key raises,
    unless ``allow_missing`` fills it with a neutral value (zeros, identity
    rotation) -- only for clients that genuinely cannot report it.
    """
    parts, lead = [], None
    for slot in SLOTS:
        raw = _lookup(values, slot.key)
        if raw is None:
            if not allow_missing:
                raise KeyError(f"missing Mobile key {slot.key!r} (or {singular(slot.key)!r})")
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape[-1] != slot.raw_dim:
            raise ValueError(f"{slot.key}: expected last dim {slot.raw_dim}, got {arr.shape}")
        lead = arr.shape[:-1]
        break
    lead = () if lead is None else lead

    for slot in SLOTS:
        raw = _lookup(values, slot.key)
        if raw is None:
            if not allow_missing:
                raise KeyError(f"missing Mobile key {slot.key!r} (or {singular(slot.key)!r})")
            neutral = np.zeros(lead + (slot.raw_dim,), dtype=np.float64)
            if slot.kind == "pose":
                neutral[..., 3] = 1.0
            raw = neutral
        arr = np.asarray(raw, dtype=np.float64).reshape(lead + (slot.raw_dim,))
        parts.append(pose_to_model(arr) if slot.kind == "pose" else arr)
    return np.concatenate(parts, axis=-1)


def unpack(vec: np.ndarray, *, key_style: str = "singular") -> dict[str, np.ndarray]:
    """Model vector ``(..., 75)`` -> raw per-key arrays (poses back to quaternions)."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape[-1] != MODEL_DIM:
        raise ValueError(f"expected last dim {MODEL_DIM}, got {vec.shape}")
    out: dict[str, np.ndarray] = {}
    for slot in SLOTS:
        part = vec[..., slot.start : slot.end]
        value = model_to_pose(part) if slot.kind == "pose" else part
        name = singular(slot.key) if key_style == "singular" else slot.key
        out[name] = value.astype(np.float32)
    return out


# Pretraining slot order; base twist stores vx, vy and wz only.
# Head pose occupies the reserved block as xyz plus a wxyz quaternion.
C80_TAG = "mobile_m92uw_c80"
C80_DIM = 80

# (c80 slice, model-75 slice) pairs copied verbatim.
_C80_COPY = (
    ((0, 7), (0, 7)),      # left arm joints
    ((7, 16), (38, 47)),   # left eef pos + rot6d (base frame)
    ((17, 29), (14, 26)),  # left hand
    ((29, 36), (7, 14)),   # right arm joints
    ((36, 45), (47, 56)),  # right eef pos + rot6d
    ((46, 58), (26, 38)),  # right hand
    ((58, 60), (69, 71)),  # base twist vx, vy
    ((60, 61), (74, 75)),  # base twist wz
    ((64, 66), (56, 58)),  # torso: waist
    ((66, 68), (58, 60)),  # torso: leg
    ((73, 76), (60, 63)),  # head position (reserved block)
)
_C80_HEAD_QUAT = (76, 80)
_M75_HEAD_ROT6D = (63, 69)
C80_VALID = sorted(
    {i for (lo, hi), _ in _C80_COPY for i in range(lo, hi)} | set(range(*_C80_HEAD_QUAT))
)


def to_c80(vec: np.ndarray) -> np.ndarray:
    """(..., 75) model vector -> (..., 80) pretrain-canonical vector."""
    vec = np.asarray(vec, dtype=np.float64)
    out = np.zeros(vec.shape[:-1] + (C80_DIM,), dtype=np.float64)
    for (clo, chi), (mlo, mhi) in _C80_COPY:
        out[..., clo:chi] = vec[..., mlo:mhi]
    r6 = vec[..., _M75_HEAD_ROT6D[0] : _M75_HEAD_ROT6D[1]]
    out[..., _C80_HEAD_QUAT[0] : _C80_HEAD_QUAT[1]] = rot.matrix_to_quat_wxyz(rot.rot6d_to_matrix(r6))
    return out


def from_c80(vec: np.ndarray) -> np.ndarray:
    """(..., 80) -> (..., 75); vz, wx, wy (absent from the 80-d layout) become 0."""
    vec = np.asarray(vec, dtype=np.float64)
    out = np.zeros(vec.shape[:-1] + (MODEL_DIM,), dtype=np.float64)
    for (clo, chi), (mlo, mhi) in _C80_COPY:
        out[..., mlo:mhi] = vec[..., clo:chi]
    q = vec[..., _C80_HEAD_QUAT[0] : _C80_HEAD_QUAT[1]]
    out[..., _M75_HEAD_ROT6D[0] : _M75_HEAD_ROT6D[1]] = rot.matrix_to_rot6d(rot.quat_wxyz_to_matrix(q))
    return out


def contract(variant: str = "m75") -> dict:
    """Machine-readable description written next to every checkpoint.

    ``variant`` is the model-side layout: ``m75`` (from-scratch Mobile layout)
    or ``c80`` (pretrain-compatible canonical layout, see :func:`to_c80`).
    """
    return {
        "variant": variant,
        "model_width": C80_DIM if variant == "c80" else MODEL_DIM,
        "embodiment_tag": C80_TAG if variant == "c80" else EMBODIMENT_TAG,
        "robot_type": ROBOT_TYPE,
        "fps": FPS,
        "model_dim": MODEL_DIM,
        "raw_dim": RAW_DIM,
        "cameras": dict(CAMERAS),
        "keys": [
            {
                "key": s.key,
                "runtime_key": singular(s.key),
                "raw_dim": s.raw_dim,
                "kind": s.kind,
                "model_slice": [s.start, s.end],
            }
            for s in SLOTS
        ],
        "pose_format": "[x, y, z, qw, qx, qy, qz] in base_link",
    }
