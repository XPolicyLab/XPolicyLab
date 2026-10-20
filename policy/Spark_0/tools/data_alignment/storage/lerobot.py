"""HDF5 → LeRobot for Tianji dual-arm + Wuji, EE-pose layout.

54-D state/action: left/right ``ee_poses`` (7) + ``ee_joint_states`` (20).
``--input_dir`` + ``--repo_id`` walks a recursive HDF5 tree. EgoPro files
already truncated by ``metadata/episode_truncation`` are copied as-is;
legacy files drop the last frame so each row has a next-state action.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # optional dependency; keeps --help available
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from lerobot.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        HF_LEROBOT_HOME,
        LeRobotDataset,
    )
except ImportError:  # optional dependency; conversion reports a clear error in main
    CODEBASE_VERSION = None
    HF_LEROBOT_HOME = os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
    LeRobotDataset = None


from XPolicyLab.utils.process_data import decode_image_bit

from .hdf5_reader import load, load_metadata


DEFAULT_DATASET_NAME = "xspark"
DEFAULT_ENV_CFG_TYPE = "tianji_marvin_wuji"
DEFAULT_TASKS = ("apply_drill", "close_laptop", "stack_bowls")

TARGET_IMAGE_WIDTH = 640
TARGET_IMAGE_HEIGHT = 480

# Ordered mapping key -> dim. Flat-vector ranges are derived cumulatively.
STATE_KEYS: dict[str, int] = {
    "left_ee_poses": 7,
    "left_ee_joint_states": 20,
    "right_ee_poses": 7,
    "right_ee_joint_states": 20,
}
ACTION_KEYS: dict[str, int] = {
    "left_ee_poses": 7,
    "left_ee_joint_states": 20,
    "right_ee_poses": 7,
    "right_ee_joint_states": 20,
}

CAMERA_CANDIDATES = {
    "cam_high": [
        ("vision", "cam_head", "colors"),
    ],
    "cam_left_wrist": [
        ("vision", "cam_left_wrist", "colors"),
    ],
    "cam_right_wrist": [
        ("vision", "cam_right_wrist", "colors"),
    ],
}
OMIT_EMPTY_CAMERAS = True
CAMERA_EXTRINSICS_CANDIDATES = {
    "cam_high": [
        ("vision", "cam_head", "extrinsics"),
        ("extrinsics",),
    ],
}
CAMERA_EXTRINSICS_VALIDITY = {
    "cam_high": ("validity", "cam_head_pose"),
}
REQUIRE_CAMERA_EXTRINSICS = True
ENABLE_TACTILE = True
# ``--finger-only`` trims only tactile 3-D/patch features; stored MANO is
# still exported.
EXPORT_STORED_MANO_WITH_FINGER_ONLY = True
TACTILE_TAXEL_LEFT_KEY = "observation.tactile.taxel_pressure_3D.left_ee"
TACTILE_TAXEL_RIGHT_KEY = "observation.tactile.taxel_pressure_3D.right_ee"
TACTILE_REGION_LEFT_KEY = (
    "observation.tactile.region_pressure_vector.left_ee"
)
TACTILE_REGION_RIGHT_KEY = (
    "observation.tactile.region_pressure_vector.right_ee"
)
TACTILE_2D_KEY = "observation.tactile.2d"
@dataclasses.dataclass(frozen=True)
class ManoField:
    """One stored MANO field and its source/export names."""

    group: str
    name: str
    dim: int
    attribute: str
    source_validity: str
    output_key: str

    @property
    def output_validity(self) -> str:
        prefix, _, name = self.output_key.partition(".mano.")
        return f"{prefix}.mano.validity.{name}"


MANO_FIELDS = tuple(
    ManoField(
        group=group,
        name=f"{side}_{name}",
        dim=dim,
        attribute=f"{group}_{side}_{attribute}",
        source_validity=f"{'action_' if group == 'action' else ''}{side}_{validity}",
        output_key=f"{prefix}.mano.{side}_{name}",
    )
    for group, prefix in (("state", "observation"), ("action", "action"))
    for side in ("left", "right")
    for name, attribute, validity, dim in (
        ("ee_poses", "poses", "ee_pose", 7),
        ("ee_joint_states", "joints", "ee_joint_states", 45),
        ("hand_betas", "betas", "hand_betas", 10),
    )
)
MANO_HDF5_KEYS = tuple(f"mano/{field.group}/{field.name}" for field in MANO_FIELDS)
TACTILE_TAXELS = 320
TACTILE_POINT_FIELDS = 4
TACTILE_REGION_HISTORY = 50
TACTILE_REGION_DIMS = 6
TACTILE_REGION_FIELDS = ("palm", "thumb", "index", "middle", "ring", "pinky")
TACTILE_REGION_SLICES = {
    "palm": [0, 240],
    "thumb": [240, 256],
    "index": [256, 272],
    "middle": [272, 288],
    "ring": [288, 304],
    "pinky": [304, 320],
}
TACTILE_2D_HW = (480, 640)
TACTILE_2D_VALUE_RANGE = (0.0, 1.0)
TACTILE_2D_SEMANTICS = (
    "continuous contact-conditioned geodesic proximity rasterized on the "
    "projected hand mesh in the observation.images.cam_high image plane; "
    "0.0 is background or no detected active contact, and larger values in "
    "(0,1] indicate stronger/closer contact proximity"
)
TACTILE_FINGER_TAXEL_START = TACTILE_REGION_SLICES["thumb"][0]
TACTILE_FINGER_TAXELS = TACTILE_TAXELS - TACTILE_FINGER_TAXEL_START
TACTILE_FINGER_REGION_FIELDS = TACTILE_REGION_FIELDS[1:]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None
    video_codec: str = "libsvtav1"
    streaming_encoding: bool = False
    video_crf: int | None = None
    # LeRobot StreamingVideoEncoder GOP / keyint. Default upstream is 2
    # (near all-intra, very slow). None keeps the library default.
    video_gop: int | None = None
    video_queue_maxsize: int = 30


DEFAULT_DATASET_CONFIG = DatasetConfig(
    video_codec="h264",
    streaming_encoding=True,
    video_crf=23,
    video_gop=50,
    video_queue_maxsize=120,
)


@dataclasses.dataclass(frozen=True)
class TactileEpisode:
    """Validated camera-frame tactile arrays aligned to retained frames."""

    left_3d: np.ndarray
    right_3d: np.ndarray
    region_left: np.ndarray
    region_right: np.ndarray
    maps_2d: np.ndarray


@dataclasses.dataclass(frozen=True)
class ManoEpisode:
    """Stored MANO values and scalar validity aligned to output frames."""

    state_left_poses: np.ndarray
    state_left_poses_validity: np.ndarray
    state_left_joints: np.ndarray
    state_left_joints_validity: np.ndarray
    state_left_betas: np.ndarray
    state_left_betas_validity: np.ndarray
    state_right_poses: np.ndarray
    state_right_poses_validity: np.ndarray
    state_right_joints: np.ndarray
    state_right_joints_validity: np.ndarray
    state_right_betas: np.ndarray
    state_right_betas_validity: np.ndarray
    action_left_poses: np.ndarray
    action_left_poses_validity: np.ndarray
    action_left_joints: np.ndarray
    action_left_joints_validity: np.ndarray
    action_left_betas: np.ndarray
    action_left_betas_validity: np.ndarray
    action_right_poses: np.ndarray
    action_right_poses_validity: np.ndarray
    action_right_joints: np.ndarray
    action_right_joints_validity: np.ndarray
    action_right_betas: np.ndarray
    action_right_betas_validity: np.ndarray


def _key_ranges(key_dims: dict[str, int]) -> dict[str, tuple[int, int]]:
    ranges = {}
    offset = 0
    for key, dim in key_dims.items():
        ranges[key] = (offset, offset + dim)
        offset += dim
    return ranges


def _build_motor_names(key_dims: dict[str, int]) -> list[str]:
    motors = []
    for key, dim in key_dims.items():
        motors.extend(f"{key}_{index}" for index in range(dim))
    return motors


def _total_dim(key_dims: dict[str, int]) -> int:
    return sum(key_dims.values())


def _get_nested(data, *keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_2d_float32(array, name):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def _ensure_utf8_strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _ensure_utf8_strings(value.item())
        return [
            str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x)
            for x in value.tolist()
        ]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_ensure_utf8_strings(item))
        return result
    return [str(value)]


def _find_instructions(data):
    candidates = [
        _get_nested(data, "instructions"),
        _get_nested(data, "instruction"),
    ]
    for candidate in candidates:
        strings = [s for s in _ensure_utf8_strings(candidate) if s]
        if strings:
            return strings
    return []


def _choose_instruction(data):
    instructions = _find_instructions(data)
    if not instructions:
        return ""
    return random.choice(instructions)


def _resize_image(image):
    if image.shape[:2] == (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
        return image
    return cv2.resize(
        image,
        (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


def _decode_images_if_needed(images):
    """Encoded or already-decoded camera frames -> [T, H, W, 3] uint8 RGB.

    ``decode_image_bit`` is the only decoder allowed on XPolicyLab data: it
    handles every image-bit layout RoboDojo/Spark HDF5 files use and returns
    RGB, so no channel conversion belongs here or downstream.
    """
    image_array = np.asarray(images)
    if image_array.ndim == 4 and image_array.shape[-1] == 3:
        frames = image_array
    elif image_array.ndim == 3 and image_array.shape[-1] == 3:
        frames = image_array[None, ...]
    else:
        frames = np.asarray(decode_image_bit(images))
        if frames.ndim == 3:
            frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected decoded frames with shape [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if frames.shape[1:3] != (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH):
        frames = np.stack([_resize_image(frame) for frame in frames], axis=0)
    return frames


def _find_camera_array(data, camera_name):
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            return _decode_images_if_needed(value)
    return None


def _find_camera_extrinsics(
    data, camera_name, source_frames, *, already_truncated: bool = False
):
    """Return per-frame world_from_camera poses without filling invalid gaps."""
    value = None
    for keys in CAMERA_EXTRINSICS_CANDIDATES.get(camera_name, ()):
        value = _get_nested(data, *keys)
        if value is not None:
            break
    if value is None:
        return None

    poses = np.asarray(value, dtype=np.float32)
    if poses.shape == (7,):
        poses = np.repeat(poses[None], source_frames, axis=0)
    if poses.shape != (source_frames, 7):
        raise ValueError(
            f"{camera_name} extrinsics must be [7] or [{source_frames},7], "
            f"got {poses.shape}"
        )
    retained = poses if already_truncated else poses[:-1]
    validity_path = CAMERA_EXTRINSICS_VALIDITY.get(camera_name)
    if validity_path is not None:
        validity = _get_nested(data, *validity_path)
        if validity is not None:
            validity = np.asarray(validity, dtype=bool)
            if validity.shape != (source_frames,):
                raise ValueError(
                    f"{camera_name} extrinsics validity must be [{source_frames}], "
                    f"got {validity.shape}"
                )
            retained_validity = validity if already_truncated else validity[:-1]
            if not np.all(retained_validity):
                invalid = np.flatnonzero(~retained_validity)
                raise ValueError(
                    f"{camera_name} extrinsics contain invalid retained frames: "
                    f"{invalid[:10].tolist()}"
                )
    if not np.all(np.isfinite(retained)):
        raise ValueError(f"{camera_name} extrinsics contain NaN/Inf in retained frames")
    quaternion_norm = np.linalg.norm(retained[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
        raise ValueError(f"{camera_name} extrinsics contain non-unit quaternions")
    return retained


def _validate_source_timestamps(data, source_frames, fps):
    """Reject missing, repeated, or non-uniform source frame timestamps."""
    value = _get_nested(data, "timestamps")
    if value is None:
        return
    timestamps = np.asarray(value, dtype=np.float64).reshape(-1)
    if timestamps.shape != (source_frames,):
        raise ValueError(
            f"timestamps must have {source_frames} entries, got {timestamps.shape}"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contain NaN/Inf")
    delta = np.diff(timestamps)
    expected = 1.0 / float(fps)
    if np.any(delta <= 0) or not np.allclose(delta, expected, atol=1e-6, rtol=1e-5):
        bad = np.flatnonzero(
            (delta <= 0) | ~np.isclose(delta, expected, atol=1e-6, rtol=1e-5)
        )
        raise ValueError(
            f"timestamps contain temporal gaps at intervals {bad[:10].tolist()}"
        )


@dataclasses.dataclass(frozen=True)
class SourceSchema:
    camera_names: tuple[str, ...]
    tactile: bool
    mano: bool


def _complete_modality(file: h5py.File, keys: tuple[str, ...], name: str) -> bool:
    present = [key in file for key in keys]
    if any(present) and not all(present):
        missing = [key for key, exists in zip(keys, present) if not exists]
        raise ValueError(f"Incomplete {name} data in {file.filename}: missing {missing}")
    return all(present)


def _inspect_input_schema(input_files: list[Path]) -> SourceSchema:
    """Inspect camera/tactile/MANO metadata with one open per source episode."""
    tactile_keys = (
        "tactile/taxel_pressure_3D/left_ee",
        "tactile/taxel_pressure_3D/right_ee",
        "tactile/region_pressure_vector/left_ee",
        "tactile/region_pressure_vector/right_ee",
    )
    cameras: set[str] = set() if OMIT_EMPTY_CAMERAS else set(CAMERA_CANDIDATES)
    tactile_count = mano_count = 0
    for input_path in input_files:
        with h5py.File(input_path, "r") as file:
            for camera_name, candidates in CAMERA_CANDIDATES.items():
                if camera_name in cameras:
                    continue
                for keys in candidates:
                    dataset = file.get("/".join(keys))
                    if dataset is not None and dataset.shape and dataset.shape[0] > 0:
                        cameras.add(camera_name)
                        break
            if ENABLE_TACTILE:
                tactile_count += _complete_modality(file, tactile_keys, "tactile")
            mano_count += _complete_modality(file, MANO_HDF5_KEYS, "MANO")

    for name, count in (("tactile", tactile_count), ("MANO", mano_count)):
        if 0 < count < len(input_files):
            print(
                f"Selected inputs mix {name} and non-{name} episodes; "
                f"omitting {name} from this LeRobot dataset"
            )
    return SourceSchema(
        camera_names=tuple(name for name in CAMERA_CANDIDATES if name in cameras),
        tactile=bool(tactile_count) and tactile_count == len(input_files),
        mano=bool(mano_count) and mano_count == len(input_files),
    )


def _should_export_stored_mano(
    *, stored_mano_available: bool, finger_only: bool
) -> bool:
    """Resolve MANO export independently from tactile finger selection."""
    return stored_mano_available and (
        not finger_only or EXPORT_STORED_MANO_WITH_FINGER_ONLY
    )


def _require_stored_mano_for_tactile(
    *, include_tactile: bool, stored_mano_available: bool
) -> None:
    if (
        include_tactile
        and not stored_mano_available
        and not _external_tactile_2d_enabled()
    ):
        raise ValueError(
            "Tactile 2D reuses stored mano/state|action. Re-run EgoPro "
            "retarget-camera-aligned with --mano-assets so HDF5 contains "
            "mano/ before converting to LeRobot."
        )


def _external_tactile_2d_enabled() -> bool:
    """Whether the shard runner supplies an independent 2D implementation."""
    return bool(globals().get("_EXTERNAL_TACTILE_2D_ACTIVE", False))


def _episode_proximity_maps(input_path: str | Path, *, mano_assets=None):
    from ..tactile.contact_map import episode_proximity_maps
    return episode_proximity_maps(
        input_path,
        workers=1,
        mano_assets=mano_assets,
    )


def _extract_tactile(
    data,
    input_path: str | Path,
    source_frames: int,
    *,
    already_truncated: bool,
    maps_2d=None,
    mano_assets=None,
    finger_only: bool = False,
) -> TactileEpisode:
    """Validate and align camera-frame tactile fields to retained frames."""

    def _array(*keys, dtype=None):
        value = _get_nested(data, *keys)
        if value is None:
            raise KeyError(f"Missing '{'/'.join(keys)}'")
        return np.asarray(value, dtype=dtype)

    left_3d = _array("tactile", "taxel_pressure_3D", "left_ee", dtype=np.float32)
    right_3d = _array("tactile", "taxel_pressure_3D", "right_ee", dtype=np.float32)
    expected_3d_shape = (source_frames, TACTILE_TAXELS, TACTILE_POINT_FIELDS)
    for side, points in (("left", left_3d), ("right", right_3d)):
        if points.shape != expected_3d_shape:
            raise ValueError(
                f"tactile/taxel_pressure_3D/{side}_ee must have shape "
                f"{expected_3d_shape}, "
                f"got {points.shape}"
            )
    # Each 3-D point is [x, y, z, pressure] in the upright OpenCV head camera
    # frame (x right, y down, z forward), followed by the tactile pressure.
    region_left = _array(
        "tactile", "region_pressure_vector", "left_ee", dtype=np.float32
    )
    region_right = _array(
        "tactile", "region_pressure_vector", "right_ee", dtype=np.float32
    )
    expected_region_shape = (
        source_frames,
        TACTILE_REGION_HISTORY,
        TACTILE_REGION_DIMS,
    )
    for side, history in (("left", region_left), ("right", region_right)):
        if history.shape != expected_region_shape:
            raise ValueError(
                f"tactile/region_pressure_vector/{side}_ee must have shape "
                f"{expected_region_shape}, got {history.shape}"
            )

    if not already_truncated:
        left_3d = left_3d[:-1]
        right_3d = right_3d[:-1]
        region_left = region_left[:-1]
        region_right = region_right[:-1]
    for side, points in (
        ("left", left_3d),
        ("right", right_3d),
        ("left", region_left),
        ("right", region_right),
    ):
        if not np.all(np.isfinite(points)):
            raise ValueError(
                f"tactile {side} contains NaN/Inf in retained frames"
            )
    if maps_2d is None:
        maps_2d = _episode_proximity_maps(input_path, mano_assets=mano_assets)
    if maps_2d.shape[0] != source_frames:
        raise ValueError(
            f"proximity 2D has {maps_2d.shape[0]} frames, expected {source_frames}"
        )
    if maps_2d.shape[1:] != TACTILE_2D_HW:
        raise ValueError(
            f"proximity 2D must have shape [T,{TACTILE_2D_HW[0]},{TACTILE_2D_HW[1]}], "
            f"got {maps_2d.shape}"
        )
    if not already_truncated:
        maps_2d = maps_2d[:-1]
    if not np.all(np.isfinite(maps_2d)):
        raise ValueError("proximity 2D contains NaN/Inf in retained frames")
    map_min = float(np.min(maps_2d)) if maps_2d.size else 0.0
    map_max = float(np.max(maps_2d)) if maps_2d.size else 0.0
    if map_min < TACTILE_2D_VALUE_RANGE[0] or map_max > TACTILE_2D_VALUE_RANGE[1]:
        raise ValueError(
            "proximity 2D values must be in "
            f"{TACTILE_2D_VALUE_RANGE}, got [{map_min}, {map_max}]"
        )
    # Compute 2D from the complete source arrays first: palm taxels remain part
    # of the proximity image even when vector/point outputs are finger-only.
    if finger_only:
        left_3d = left_3d[:, TACTILE_FINGER_TAXEL_START:, :]
        right_3d = right_3d[:, TACTILE_FINGER_TAXEL_START:, :]
        region_left = region_left[:, :, 1:]
        region_right = region_right[:, :, 1:]
    return TactileEpisode(
        left_3d=left_3d,
        right_3d=right_3d,
        region_left=region_left,
        region_right=region_right,
        maps_2d=maps_2d.astype(np.float32, copy=False),
    )


def _extract_mano(
    data,
    source_frames: int,
    *,
    already_truncated: bool,
) -> ManoEpisode:
    """Copy MANO values and reduce per-component masks to scalar validity."""
    retained = source_frames if already_truncated else source_frames - 1

    def _field(
        group: str,
        key: str,
        validity_key: str,
        dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        value = _get_nested(data, "mano", group, key)
        if value is None:
            raise KeyError(f"Missing 'mano/{group}/{key}'")
        arr = _ensure_2d_float32(value, f"mano.{group}.{key}")
        if arr.shape[1] != dim:
            raise ValueError(
                f"mano/{group}/{key} dim mismatch: expected {dim}, got {arr.shape[1]}"
            )

        validity_value = _get_nested(data, "mano", "validity", validity_key)
        if validity_value is None:
            component_validity = np.ones_like(arr, dtype=bool)
        else:
            component_validity = np.asarray(validity_value, dtype=bool)
            if component_validity.ndim == 1:
                component_validity = component_validity[:, None]
            if (
                component_validity.ndim != 2
                or component_validity.shape[0] != arr.shape[0]
                or component_validity.shape[1] not in (1, dim)
            ):
                raise ValueError(
                    f"mano/validity/{validity_key} must have shape "
                    f"({arr.shape[0]}, 1) or {arr.shape}, got "
                    f"{component_validity.shape}"
                )
        scalar_validity = np.all(component_validity, axis=1, keepdims=True)
        scalar_validity &= np.all(np.isfinite(arr), axis=1, keepdims=True)
        scalar_validity = scalar_validity.astype(np.float32, copy=False)

        if already_truncated:
            if arr.shape[0] != retained:
                raise ValueError(
                    f"mano/{group}/{key} has {arr.shape[0]} frames, expected {retained}"
                )
            return arr, scalar_validity
        if arr.shape[0] < source_frames:
            raise ValueError(
                f"mano/{group}/{key} has {arr.shape[0]} frames, "
                f"need at least {source_frames}"
            )
        return arr[:retained], scalar_validity[:retained]

    fields = {}
    for spec in MANO_FIELDS:
        value, validity = _field(spec.group, spec.name, spec.source_validity, spec.dim)
        fields[spec.attribute] = value
        fields[f"{spec.attribute}_validity"] = validity
    return ManoEpisode(**fields)



def _empty_image_sequence(num_frames):
    return np.zeros(
        (num_frames, TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )


def _resolve_vector_part(data, group_name, key, dim, *, fallback_group=None):
    value = _get_nested(data, group_name, key)
    used_fallback = False
    if value is None and fallback_group is not None:
        value = _get_nested(data, fallback_group, key)
        used_fallback = value is not None
    if value is None:
        raise KeyError(
            f"Missing '{group_name}/{key}'"
            + (f" (and fallback '{fallback_group}/{key}')" if fallback_group else "")
        )
    arr = _ensure_2d_float32(value, f"{group_name}.{key}")
    if arr.shape[1] != dim:
        raise ValueError(
            f"{group_name}.{key} dim mismatch: expected {dim}, got {arr.shape[1]}"
        )
    return arr, used_fallback


def _extract_ordered_vector(data, group_name, key_dims, *, fallback_group=None):
    parts = []
    fallback_keys = []
    horizon = None
    for key, dim in key_dims.items():
        arr, used_fallback = _resolve_vector_part(
            data,
            group_name,
            key,
            dim,
            fallback_group=fallback_group,
        )
        if used_fallback:
            fallback_keys.append(key)
        if horizon is None:
            horizon = arr.shape[0]
        elif arr.shape[0] != horizon:
            raise ValueError(
                f"{group_name}.{key} horizon mismatch: expected {horizon}, got {arr.shape[0]}"
            )
        parts.append(arr)
    return np.concatenate(parts, axis=1), fallback_keys


def _extract_state(data):
    state, _ = _extract_ordered_vector(data, "state", STATE_KEYS)
    return state


def _read_episode_truncation(data) -> int:
    """Return HDF5-declared truncation; ``0`` means the converter must truncate."""
    value = _get_nested(data, "metadata", "episode_truncation")
    if value is None:
        return 0
    truncation = int(np.asarray(value).reshape(()))
    if truncation < 0:
        raise ValueError(f"metadata/episode_truncation must be >= 0, got {truncation}")
    return truncation


def _extract_action(data, *, already_truncated: bool):
    """Assemble 54-D next-state action.

    When ``already_truncated`` is true, action arrays are used as-is (EgoPro
    HDF5 already dropped the final source frame). Otherwise this function
    drops the last frame / synthesizes EE action from ``state[1:]``.
    """
    parts = []
    fallback_keys = []
    horizon = None
    for key, dim in ACTION_KEYS.items():
        action_value = _get_nested(data, "action", key)
        if action_value is not None:
            arr = _ensure_2d_float32(action_value, f"action.{key}")
            if arr.shape[1] != dim:
                raise ValueError(
                    f"action.{key} dim mismatch: expected {dim}, got {arr.shape[1]}"
                )
            if already_truncated:
                if arr.shape[0] < 1:
                    raise ValueError(f"action.{key} is empty")
            else:
                if arr.shape[0] < 2:
                    raise ValueError(
                        f"action.{key} needs at least 2 frames to drop the last, "
                        f"got {arr.shape[0]}"
                    )
                arr = arr[:-1]
        elif key.endswith("_ee_poses"):
            if already_truncated:
                raise KeyError(
                    f"Missing 'action/{key}' on already-truncated HDF5; "
                    "next-state EE action must be stored explicitly"
                )
            state_value = _get_nested(data, "state", key)
            if state_value is None:
                raise KeyError(f"Missing 'action/{key}' and fallback 'state/{key}'")
            state_arr = _ensure_2d_float32(state_value, f"state.{key}")
            if state_arr.shape[1] != dim:
                raise ValueError(
                    f"state.{key} dim mismatch: expected {dim}, got {state_arr.shape[1]}"
                )
            if state_arr.shape[0] < 2:
                raise ValueError(
                    f"state.{key} needs at least 2 frames for next-pose action, "
                    f"got {state_arr.shape[0]}"
                )
            arr = state_arr[1:]
            fallback_keys.append(key)
        else:
            arr, used_fallback = _resolve_vector_part(
                data, "action", key, dim, fallback_group="state"
            )
            if already_truncated:
                if arr.shape[0] < 1:
                    raise ValueError(f"action.{key} is empty")
            else:
                if arr.shape[0] < 2:
                    raise ValueError(
                        f"action.{key} needs at least 2 frames to drop the last, "
                        f"got {arr.shape[0]}"
                    )
                arr = arr[:-1]
            if used_fallback:
                fallback_keys.append(key)

        if horizon is None:
            horizon = arr.shape[0]
        elif arr.shape[0] != horizon:
            raise ValueError(
                f"action.{key} horizon mismatch: expected {horizon}, got {arr.shape[0]}"
            )
        parts.append(arr)
    return np.concatenate(parts, axis=1), fallback_keys


def _read_fps(data, default_fps=25):
    freq = _get_nested(data, "additional_info", "frequency")
    if freq is None:
        return default_fps
    if isinstance(freq, np.ndarray):
        freq = freq.item()
    return int(freq)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    motors: list[str],
    fps: int,
    mode: Literal["video", "image"] = "video",
    *,
    root: Path | None = None,
    dataset_config: DatasetConfig | None = None,
    camera_names: tuple[str, ...] | None = None,
    include_tactile: bool = False,
    include_mano: bool = False,
    finger_only: bool = False,
) -> LeRobotDataset:
    if dataset_config is None:
        dataset_config = DEFAULT_DATASET_CONFIG
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    camera_names = tuple(CAMERA_CANDIDATES) if camera_names is None else camera_names
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }
        if REQUIRE_CAMERA_EXTRINSICS and cam in CAMERA_EXTRINSICS_CANDIDATES:
            features[f"observation.camera_extrinsics.{cam}"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            }
    if include_tactile:
        output_taxels = TACTILE_FINGER_TAXELS if finger_only else TACTILE_TAXELS
        output_regions = (
            len(TACTILE_FINGER_REGION_FIELDS) if finger_only else TACTILE_REGION_DIMS
        )
        # The four fields for every taxel are x, y, z, pressure in the upright
        # OpenCV head camera frame.
        taxel_feature = {
            "dtype": "float32",
            "shape": (output_taxels, TACTILE_POINT_FIELDS),
            "names": ["taxel", "x_y_z_pressure"],
        }
        features[TACTILE_TAXEL_LEFT_KEY] = taxel_feature.copy()
        features[TACTILE_TAXEL_RIGHT_KEY] = taxel_feature.copy()
        region_feature = {
            "dtype": "float32",
            "shape": (TACTILE_REGION_HISTORY, output_regions),
            "names": ["history", "region"],
        }
        features[TACTILE_REGION_LEFT_KEY] = region_feature.copy()
        features[TACTILE_REGION_RIGHT_KEY] = region_feature.copy()
        features[TACTILE_2D_KEY] = {
            "dtype": "float32",
            "shape": TACTILE_2D_HW,
            "names": ["height", "width"],
        }
    if include_mano:
        for spec in MANO_FIELDS:
            features[spec.output_key] = {
                "dtype": "float32",
                "shape": (spec.dim,),
                "names": [f"{spec.name}_{index}" for index in range(spec.dim)],
            }
            features[spec.output_validity] = {
                "dtype": "float32",
                "shape": (1,),
                "names": [f"{spec.name}_valid"],
            }

    dataset_root = Path(root) if root is not None else Path(HF_LEROBOT_HOME) / repo_id
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    create_kwargs = dict(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )
    if CODEBASE_VERSION == "v3.0":
        create_kwargs["vcodec"] = dataset_config.video_codec
        create_kwargs["streaming_encoding"] = dataset_config.streaming_encoding
        create_kwargs["encoder_queue_maxsize"] = dataset_config.video_queue_maxsize
    # Newer LeRobot accepts explicit root; older ones only use HF_LEROBOT_HOME.
    try:
        dataset = LeRobotDataset.create(root=dataset_root, **create_kwargs)
    except TypeError:
        if root is not None:
            raise
        dataset = LeRobotDataset.create(**create_kwargs)
    if dataset_config.video_crf is not None or dataset_config.video_gop is not None:
        streaming_encoder = getattr(dataset, "_streaming_encoder", None)
        if streaming_encoder is None:
            raise RuntimeError(
                "video_crf/video_gop require LeRobot streaming_encoding=True"
            )
        if dataset_config.video_crf is not None:
            streaming_encoder.crf = dataset_config.video_crf
        if dataset_config.video_gop is not None:
            # Applied when each episode opens its encode stream.
            streaming_encoder.g = int(dataset_config.video_gop)
    return dataset


def add_frame_compat(dataset: LeRobotDataset, frame: dict, task: str) -> None:
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame = {**frame, "task": task}
        dataset.add_frame(frame)


def save_episode_compat(dataset: LeRobotDataset, task: str) -> None:
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def release_dataset_memory(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "create_hf_dataset") and hasattr(dataset, "hf_dataset"):
        try:
            dataset.hf_dataset = dataset.create_hf_dataset()
        except Exception:
            pass


def write_modality_json(
    dataset_root: Path,
    camera_names: tuple[str, ...] | None = None,
    *,
    include_tactile: bool = False,
    include_mano: bool = False,
    finger_only: bool = False,
) -> None:
    """Write Spark_0-compatible modality.json for the EE-pose layout."""
    state_ranges = _key_ranges(STATE_KEYS)
    action_ranges = _key_ranges(ACTION_KEYS)

    def _slice_meta(ranges, original_key):
        return {
            key: {
                "start": start,
                "end": end,
                "absolute": True,
                "dtype": "float32",
                "original_key": original_key,
            }
            for key, (start, end) in ranges.items()
        }

    camera_names = tuple(CAMERA_CANDIDATES) if camera_names is None else camera_names
    modality = {
        "state": _slice_meta(state_ranges, "observation.state"),
        "action": _slice_meta(action_ranges, "action"),
        "video": {
            cam: {"original_key": f"observation.images.{cam}"}
            for cam in camera_names
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    if REQUIRE_CAMERA_EXTRINSICS:
        modality["camera_extrinsics"] = {
            cam: {
                "original_key": f"observation.camera_extrinsics.{cam}",
                "semantics": "world_from_camera",
                "camera_axes": "x_right_y_up_z_back",
                "pose_order": "x_y_z_qw_qx_qy_qz",
            }
            for cam in camera_names
            if cam in CAMERA_EXTRINSICS_CANDIDATES
        }
    if include_tactile:
        def _ee(original_key, shape, **meta):
            return {"original_key": original_key, "dtype": "float32", "shape": shape, **meta}

        output_taxels = TACTILE_FINGER_TAXELS if finger_only else TACTILE_TAXELS
        output_region_fields = (
            TACTILE_FINGER_REGION_FIELDS if finger_only else TACTILE_REGION_FIELDS
        )
        taxel_meta = dict(
            fields=["x", "y", "z", "pressure"],
            coordinate_frame="camera_opencv_upright_head",
        )
        if not finger_only:
            taxel_meta.update(
                layout="palm_then_thumb_index_middle_ring_pinky",
                region_slices=TACTILE_REGION_SLICES,
            )
        region_meta = dict(
            fields=list(output_region_fields),
            unit="newton",
            history_order="most_recent_first",
        )
        modality["tactile"] = {
            "taxel_pressure_3D": {
                "left_ee": _ee(
                    TACTILE_TAXEL_LEFT_KEY,
                    [output_taxels, TACTILE_POINT_FIELDS],
                    **taxel_meta,
                ),
                "right_ee": _ee(
                    TACTILE_TAXEL_RIGHT_KEY,
                    [output_taxels, TACTILE_POINT_FIELDS],
                    **taxel_meta,
                ),
            },
            "region_pressure_vector": {
                "left_ee": _ee(
                    TACTILE_REGION_LEFT_KEY,
                    [TACTILE_REGION_HISTORY, len(output_region_fields)],
                    **region_meta,
                ),
                "right_ee": _ee(
                    TACTILE_REGION_RIGHT_KEY,
                    [TACTILE_REGION_HISTORY, len(output_region_fields)],
                    **region_meta,
                ),
            },
            "2d": _ee(
                TACTILE_2D_KEY,
                list(TACTILE_2D_HW),
                semantics=TACTILE_2D_SEMANTICS,
                alignment="observation.images.cam_high",
                value_range=list(TACTILE_2D_VALUE_RANGE),
                background_value=0.0,
            ),
        }
    if include_mano:
        modality["mano"] = {
            name: {"absolute": True, "dtype": "float32", "original_key": key}
            for spec in MANO_FIELDS
            for name, key in (
                (f"{spec.group}.{spec.name}", spec.output_key),
                (f"{spec.group}.validity.{spec.name}", spec.output_validity),
            )
        }
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_one(
    input_path,
    dataset,
    data_type,
    data_version,
    camera_names: tuple[str, ...] | None = None,
    *,
    include_tactile: bool = False,
    include_mano: bool = False,
    mano_assets=None,
    finger_only: bool = False,
):
    data = load(str(input_path), data_type=data_type, data_version=data_version)

    episode_truncation = _read_episode_truncation(data)
    already_truncated = episode_truncation > 0
    state = _extract_state(data)
    action, action_fallback_keys = _extract_action(
        data, already_truncated=already_truncated
    )
    if already_truncated:
        if state.shape[0] < 1:
            raise ValueError("Already-truncated HDF5 contains no frames")
        source_frames = state.shape[0]
    else:
        if state.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 frames to drop the last after action shift, "
                f"got {state.shape[0]}"
            )
        # Pair state[i] with next-frame targets; drop the final source frame.
        state = state[:-1]
        source_frames = state.shape[0] + 1
    if state.shape != action.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} vs {action.shape}")
    expected_dim = _total_dim(STATE_KEYS)
    if state.shape[1] != expected_dim:
        raise ValueError(f"state dim mismatch: expected {expected_dim}, got {state.shape[1]}")
    _validate_source_timestamps(data, source_frames, _read_fps(data))

    instruction = _choose_instruction(data)
    if not instruction:
        raise ValueError("No instruction found in data")

    camera_names = tuple(CAMERA_CANDIDATES) if camera_names is None else camera_names

    def _load_cameras():
        loaded_images = {}
        loaded_extrinsics = {}
        for camera_name in camera_names:
            image_array = _find_camera_array(data, camera_name)
            if image_array is not None:
                if already_truncated:
                    if len(image_array) != state.shape[0]:
                        raise ValueError(
                            f"{camera_name} has {len(image_array)} frames, "
                            f"expected {state.shape[0]} on already-truncated HDF5"
                        )
                    loaded_images[camera_name] = image_array
                else:
                    if len(image_array) < state.shape[0] + 1:
                        raise ValueError(
                            f"{camera_name} has {len(image_array)} frames, "
                            f"need at least {state.shape[0] + 1} to drop the last"
                        )
                    loaded_images[camera_name] = image_array[:-1]
            if REQUIRE_CAMERA_EXTRINSICS and camera_name in CAMERA_EXTRINSICS_CANDIDATES:
                extrinsics = _find_camera_extrinsics(
                    data,
                    camera_name,
                    source_frames,
                    already_truncated=already_truncated,
                )
                if extrinsics is None:
                    raise ValueError(
                        f"{camera_name} camera extrinsics are required but missing"
                    )
                loaded_extrinsics[camera_name] = extrinsics
        return loaded_images, loaded_extrinsics

    tactile = (
        _extract_tactile(
            data,
            input_path,
            source_frames,
            already_truncated=already_truncated,
            mano_assets=mano_assets,
            finger_only=finger_only,
        )
        if include_tactile
        else None
    )
    mano = (
        _extract_mano(
            data,
            source_frames,
            already_truncated=already_truncated,
        )
        if include_mano
        else None
    )
    images, camera_extrinsics = _load_cameras()

    num_frames = state.shape[0]
    for camera_name in camera_names:
        if camera_name not in images:
            images[camera_name] = _empty_image_sequence(num_frames)

    for i in range(num_frames):
        frame = {
            "observation.state": np.asarray(state[i]),
            "action": np.asarray(action[i]),
        }
        for image_name, image_array in images.items():
            frame[f"observation.images.{image_name}"] = image_array[i]
        for camera_name, extrinsics in camera_extrinsics.items():
            frame[f"observation.camera_extrinsics.{camera_name}"] = extrinsics[i]
        if tactile is not None:
            frame[TACTILE_TAXEL_LEFT_KEY] = tactile.left_3d[i]
            frame[TACTILE_TAXEL_RIGHT_KEY] = tactile.right_3d[i]
            frame[TACTILE_REGION_LEFT_KEY] = tactile.region_left[i]
            frame[TACTILE_REGION_RIGHT_KEY] = tactile.region_right[i]
            frame[TACTILE_2D_KEY] = tactile.maps_2d[i]
        if mano is not None:
            for spec in MANO_FIELDS:
                frame[spec.output_key] = getattr(mano, spec.attribute)[i]
                frame[spec.output_validity] = getattr(mano, f"{spec.attribute}_validity")[i]
        add_frame_compat(dataset, frame, instruction)

    save_episode_compat(dataset, instruction)
    release_dataset_memory(dataset)
    return action_fallback_keys


def find_input_files(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("*.hdf5"))
    files.extend(sorted(input_dir.rglob("*.h5")))
    unique_files = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def select_input_shard(input_files, num_shards=None, shard_index=None):
    """Return ``(global_index, path)`` pairs using deterministic modulo sharding."""
    if (num_shards is None) != (shard_index is None):
        raise ValueError("--num_shards and --shard_index must be provided together")
    indexed = list(enumerate(input_files))
    if num_shards is None:
        return indexed
    if num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard_index must satisfy 0 <= index < num_shards")
    selected = indexed[shard_index::num_shards]
    if not selected:
        raise ValueError(
            f"Shard {shard_index} of {num_shards} is empty for {len(input_files)} files"
        )
    return selected


def write_shard_manifest(
    dataset_root,
    *,
    input_root,
    global_source_count,
    num_shards,
    shard_index,
    converted,
):
    """Record source ordering required for gap-free canonical shard merging."""
    meta_dir = Path(dataset_root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    root = Path(input_root).resolve()
    payload = {
        "format": "xpolicylab_lerobot_v30_shard_v1",
        "global_source_count": global_source_count,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selection": "global_source_index_mod_num_shards",
        "episodes": [
            {
                "local_episode_index": local_index,
                "global_source_index": global_index,
                "source_relative": str(Path(path).resolve().relative_to(root)),
            }
            for local_index, (global_index, path) in enumerate(converted)
        ],
    }
    (meta_dir / "shard_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def discover_targets(data_root: Path, tasks: list[str], env_cfg_type: str):
    targets = []
    for task_name in tasks:
        input_dir = data_root / task_name / env_cfg_type / "data"
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Missing input dir for task {task_name}: {input_dir}")
        targets.append((task_name, env_cfg_type, input_dir))
    return targets


def _probe_fps(input_files, data_type, data_version, default_fps=25):
    if not input_files:
        return default_fps
    data = load_metadata(str(input_files[0]), data_type=data_type, data_version=data_version)
    return _read_fps(data, default_fps=default_fps)


def _convert_dataset(
    indexed_files: list[tuple[int, Path]],
    *,
    repo_id: str,
    output_root: Path | None,
    data_type: str,
    data_version: str,
    max_episode: int,
    robot_type: str,
    mano_assets=None,
    finger_only: bool = False,
):
    """Shared conversion path for per-task and unified/sharded datasets."""
    input_files = [path for _index, path in indexed_files]
    fps = _probe_fps(input_files, data_type, data_version)
    schema = _inspect_input_schema(input_files)
    include_mano = _should_export_stored_mano(
        stored_mano_available=schema.mano, finger_only=finger_only,
    )
    _require_stored_mano_for_tactile(
        include_tactile=schema.tactile, stored_mano_available=schema.mano,
    )
    if schema.tactile and not mano_assets and not _external_tactile_2d_enabled():
        raise ValueError("Tactile 2D FK requires --mano-assets")
    print(f"[{repo_id}] cameras={list(schema.camera_names)} tactile={schema.tactile} mano={include_mano}")
    dataset_root = None if output_root is None else Path(output_root) / repo_id
    modalities = dict(
        camera_names=schema.camera_names,
        include_tactile=schema.tactile,
        include_mano=include_mano,
        finger_only=finger_only,
    )
    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        motors=_build_motor_names(STATE_KEYS),
        fps=fps,
        root=dataset_root,
        **modalities,
    )
    converted = []
    failures = []
    fallback_keys_seen = set()
    for global_index, input_path in tqdm(indexed_files, desc=f"Converting {repo_id}"):
        if len(converted) >= max_episode:
            print(f"Reached max_episode={max_episode}")
            break
        try:
            fallback_keys = convert_one(
                input_path, dataset, data_type, data_version,
                mano_assets=mano_assets, **modalities,
            )
            fallback_keys_seen.update(fallback_keys)
            converted.append((global_index, input_path))
        except Exception as exc:
            print(exc)
            failures.append((str(input_path), str(exc)))
    final_root = Path(dataset.root) if hasattr(dataset, "root") else (
        dataset_root if dataset_root is not None else Path(HF_LEROBOT_HOME) / repo_id
    )
    write_modality_json(final_root, **modalities)
    print(f"[{repo_id}] state/action dim={_total_dim(STATE_KEYS)} fps={fps}")
    if fallback_keys_seen:
        print(
            f"[{repo_id}] action keys filled from state "
            "(ee_poses use state[i+1], episode length T-1; "
            f"other keys use same-frame state[:-1]): {sorted(fallback_keys_seen)}"
        )
    print(f"[{repo_id}] converted {len(converted)}/{len(input_files)} -> {final_root}")
    for path, reason in failures:
        print(f"  - failed {path}: {reason}")
    return final_root, converted, failures


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _apply_thread_limits(threads: int) -> None:
    threads = max(1, int(threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
        "TACTILE_2D_TORCH_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("TACTILE_2D_WORKERS", "1")
    global DEFAULT_DATASET_CONFIG
    DEFAULT_DATASET_CONFIG = dataclasses.replace(
        DEFAULT_DATASET_CONFIG,
        image_writer_threads=threads,
        image_writer_processes=0,
    )
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transform Tianji WujiHand spark_0 HDF5 data to LeRobot v3.0 "
            "with left/right ee_poses + ee_joint_states layout."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data_root",
        type=str,
        help="Root containing <task>/<env_cfg_type>/data/*.hdf5",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Direct recursive HDF5 root for one unified dataset.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help=f"Task directory names under data_root (default: {list(DEFAULT_TASKS)})",
    )
    parser.add_argument(
        "--env_cfg_type",
        type=str,
        default=DEFAULT_ENV_CFG_TYPE,
        help=f"Env/robot directory name (default: {DEFAULT_ENV_CFG_TYPE})",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Single unified repo_id. Ignored when --per_task is set.",
    )
    parser.add_argument(
        "--per_task",
        action="store_true",
        help="Write one LeRobot dataset per task (repo_id=<task>_<env_cfg_type>).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional parent directory for datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument("--data_type", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--data_version", type=str, default="v1.0")
    parser.add_argument("--max_episode", type=int, default=10**9)
    parser.add_argument(
        "--robot_type",
        type=str,
        default=DEFAULT_ENV_CFG_TYPE,
        help="LeRobot info.json robot_type label (metadata only).",
    )
    parser.add_argument("--num_shards", type=int)
    parser.add_argument("--shard_index", type=int)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="LeRobot image-writer threads (default: $LEROBOT_THREADS or 4).",
    )
    parser.add_argument(
        "--mano-assets",
        type=str,
        default=None,
        help=(
            "Official mano_v1_2 directory (models/MANO_LEFT.pkl, MANO_RIGHT.pkl). "
            "Required by the default MANO proximity backend; independent external "
            "real-data tactile projectors do not use it."
        ),
    )
    parser.add_argument(
        "--finger-only",
        action="store_true",
        help=(
            "Export only thumb/index/middle/ring/pinky for tactile 3D and patch; "
            "tactile.2d still uses the complete palm+finger source. "
            "Stored MANO remains exported."
        ),
    )
    args = parser.parse_args()
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(
            "This converter requires a LeRobot v3.0 writer "
            f"(got CODEBASE_VERSION={CODEBASE_VERSION!r}). Install lerobot>=0.4."
        )
    if args.threads is None:
        args.threads = _env_int("LEROBOT_THREADS", 4)
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    _apply_thread_limits(args.threads)

    output_root = Path(args.output_root) if args.output_root else None
    if args.input_dir is not None:
        if args.repo_id is None or args.per_task:
            parser.error("--input_dir requires --repo_id and does not support --per_task")
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            parser.error(f"--input_dir is not a directory: {input_dir}")
        targets = [("recursive_input", args.env_cfg_type, input_dir)]
    else:
        if args.num_shards is not None or args.shard_index is not None:
            parser.error("--num_shards/--shard_index require --input_dir")
        targets = discover_targets(Path(args.data_root), args.tasks, args.env_cfg_type)

    print("Matched targets:")
    for task_name, env_cfg_type, input_dir in targets:
        print(f"  - {task_name}/{env_cfg_type}: {input_dir}")

    total_success = 0
    total_files = 0
    all_failures = []

    if args.per_task or args.repo_id is None:
        for task_name, env_cfg_type, input_dir in targets:
            repo_id = f"{task_name}_{env_cfg_type}".lower()
            input_files = find_input_files(input_dir)
            if not input_files:
                raise FileNotFoundError(f"No .hdf5/.h5 files under {input_dir}")
            _root, converted, failures = _convert_dataset(
                list(enumerate(input_files)),
                repo_id=repo_id,
                output_root=output_root,
                data_type=args.data_type,
                data_version=args.data_version,
                max_episode=args.max_episode,
                robot_type=args.robot_type,
                mano_assets=args.mano_assets,
                finger_only=args.finger_only,
            )
            total_success += len(converted)
            total_files += len(input_files)
            all_failures.extend(failures)
    else:
        all_input_files = []
        for task_name, env_cfg_type, input_dir in targets:
            files = find_input_files(input_dir)
            all_input_files.extend(files)
            print(f"  - {task_name}/{env_cfg_type}: {len(files)} files")
        if not all_input_files:
            raise FileNotFoundError("No input files found for unified conversion")
        selected = select_input_shard(all_input_files, args.num_shards, args.shard_index)
        final_root, converted, all_failures = _convert_dataset(
            selected,
            repo_id=args.repo_id,
            output_root=output_root,
            data_type=args.data_type,
            data_version=args.data_version,
            max_episode=args.max_episode,
            robot_type=args.robot_type,
            mano_assets=args.mano_assets,
            finger_only=args.finger_only,
        )
        if all_failures:
            raise RuntimeError(
                f"Conversion failed for {len(all_failures)} files; "
                "refusing to publish a shard with source gaps"
            )
        write_shard_manifest(
            final_root,
            input_root=args.input_dir or args.data_root,
            global_source_count=len(all_input_files),
            num_shards=args.num_shards or 1,
            shard_index=args.shard_index or 0,
            converted=converted,
        )
        total_success = len(converted)
        total_files = len(selected)

    print(f"Done. Converted {total_success}/{total_files} episodes.")
    if all_failures:
        print(f"Total failures: {len(all_failures)}")


if __name__ == "__main__":
    main()
