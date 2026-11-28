"""Strict conversion between RoboDojo dictionaries and VAP tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import os

import numpy as np
from scipy.spatial.transform import Rotation


VIEW_ORDER = (
    ("cam_head", 1),
    ("cam_left_wrist", 0),
    ("cam_right_wrist", 2),
)
CAMERA_NAMES = tuple(name for name, _ in VIEW_ORDER)
ACTION_STEPS = 40
VIDEO_FRAMES = 20
ACTION_DIM = 20
RFU_FRAME = "rfu"
LFU_FRAME = "lfu"
# EE-v3 was exported in LFU while RoboDojo observes and controls RFU.
# p_lfu = RFU_TO_LFU @ p_rfu is a -90 degree rotation around world Z.
RFU_TO_LFU = np.asarray(
    ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
ACTION_DIMS = {
    "left_ee_pose": 7,
    "left_ee_joint_state": 1,
    "right_ee_pose": 7,
    "right_ee_joint_state": 1,
}

# The RoboDojo SFT manifests point at ``sim_x2`` JSON trajectories.  Those
# materialized ``follow_*_position/quaternion`` values are raw X5 link6 poses:
# they match FK from the X5 URDF directly.  Do not re-apply RoboTwin's upstream
# ``_trans_endpose`` gripper-control transform here.  Doing so adds a spurious
# 2.5 cm lever arm and flips the wrist frame by 180 degrees relative to the
# exact labels consumed during training.


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def ensure_uint8_rgb(frame: Any, *, name: str) -> np.ndarray:
    """Return a copied uint8 RGB image without guessing channel layouts."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be [H,W,3] RGB, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite pixels")
    if array.dtype == np.uint8:
        return np.array(array, copy=True, order="C")
    if np.issubdtype(array.dtype, np.floating):
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"floating {name} must be in [0,1], got [{minimum},{maximum}]"
            )
        return np.rint(array * 255.0).astype(np.uint8)
    raise TypeError(f"{name} must use uint8 or floating [0,1], got {array.dtype}")


def _apply_rgb_gain(
    frame: np.ndarray, rgb_gain: tuple[float, float, float] | None
) -> np.ndarray:
    """Apply an optional serving-only RGB domain calibration in uint8 space."""

    if rgb_gain is None:
        return frame
    gain = np.asarray(rgb_gain, dtype=np.float32)
    if gain.shape != (3,) or not np.isfinite(gain).all() or np.any(gain <= 0.0):
        raise ValueError(
            f"rgb_gain must contain three finite positive values, got {rgb_gain!r}"
        )
    calibrated = np.rint(frame.astype(np.float32) * gain.reshape(1, 1, 3))
    return np.clip(calibrated, 0.0, 255.0).astype(np.uint8)


def resize_rgb(
    frame: Any,
    *,
    height: int,
    width: int,
    name: str,
    rgb_gain: tuple[float, float, float] | None = None,
) -> np.ndarray:
    rgb = ensure_uint8_rgb(frame, name=name)
    resize_mode = os.environ.get("VAP_IMAGE_RESIZE_MODE", "tensor_bilinear")
    if resize_mode not in ("tensor_bilinear", "pil_bicubic"):
        raise ValueError(f"unsupported VAP_IMAGE_RESIZE_MODE: {resize_mode!r}")
    if rgb.shape[:2] == (height, width):
        return _apply_rgb_gain(rgb, rgb_gain)
    # Match x2robot_dataset_v2 VideoVisionProcessor's fixed-resolution path:
    # torchvision uint8 bilinear.  OpenCV INTER_AREA is visibly close but not
    # pixel-identical, which shifts the VAE input distribution at serving.
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tv_functional

    if resize_mode == "pil_bicubic":
        # Match the training augmented path's PIL Bicubic kernel, but retain
        # the complete image. No random crop/rotation/photometric transform.
        from PIL import Image
        resized = np.array(
            tv_functional.resize(
                Image.fromarray(rgb), [height, width],
                interpolation=InterpolationMode.BICUBIC,
            ),
            copy=True,
        )
        if resized.shape != (height, width, 3):
            raise RuntimeError(f"resized {name} has unexpected shape {resized.shape}")
        return np.ascontiguousarray(_apply_rgb_gain(resized, rgb_gain))

    chw = torch.from_numpy(np.array(rgb, copy=True, order="C")).permute(2, 0, 1)
    resized = (
        tv_functional.resize(
            chw,
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
        )
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    if resized.shape != (height, width, 3):
        raise RuntimeError(f"resized {name} has unexpected shape {resized.shape}")
    return np.ascontiguousarray(_apply_rgb_gain(resized, rgb_gain))


def _camera_color(observation: Mapping[str, Any], camera: str) -> Any:
    vision = _require_mapping(observation.get("vision"), "observation['vision']")
    camera_data = _require_mapping(vision.get(camera), f"vision[{camera!r}]")
    if "color" not in camera_data:
        raise KeyError(f"vision[{camera!r}] is missing 'color'")
    return camera_data["color"]


def extract_resized_views(
    observation: Mapping[str, Any],
    *,
    height: int,
    width: int,
    rgb_gain: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Extract [head, left wrist, right wrist] as ``[3,H,W,3]``."""

    observation = _require_mapping(observation, "observation")
    return np.stack(
        [
            resize_rgb(
                _camera_color(observation, camera),
                height=height,
                width=width,
                name=f"vision[{camera!r}]['color']",
                rgb_gain=rgb_gain,
            )
            for camera in CAMERA_NAMES
        ],
        axis=0,
    )


def extract_executed_video(
    payload: Mapping[str, Any],
    *,
    height: int,
    width: int,
    video_frames: int = VIDEO_FRAMES,
    rgb_gain: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Convert client history to ``[3,T,H,W,3]`` in VAP view order."""

    raw = _require_mapping(payload.get("executed_video"), "_vap.executed_video")
    views = []
    for camera in CAMERA_NAMES:
        frames = np.asarray(raw.get(camera))
        if frames.ndim != 4 or frames.shape[0] != video_frames:
            raise ValueError(
                f"executed_video[{camera!r}] must be "
                f"[{video_frames},H,W,3], got {frames.shape}"
            )
        views.append(
            np.stack(
                [
                    resize_rgb(
                        frame,
                        height=height,
                        width=width,
                        name=f"executed_video[{camera!r}][{index}]",
                        rgb_gain=rgb_gain,
                    )
                    for index, frame in enumerate(frames)
                ],
                axis=0,
            )
        )
    return np.stack(views, axis=0)


def build_anchor_and_history(
    anchor: np.ndarray,
    history: np.ndarray,
    *,
    video_frames: int = VIDEO_FRAMES,
) -> np.ndarray:
    if anchor.ndim != 4 or anchor.shape[0] != len(VIEW_ORDER):
        raise ValueError(f"anchor must be [3,H,W,3], got {anchor.shape}")
    expected = (len(VIEW_ORDER), video_frames, *anchor.shape[1:])
    if history.shape != expected:
        raise ValueError(f"history shape {history.shape} != {expected}")
    return np.concatenate((anchor[:, None], history), axis=1)


def sixd_to_rotmat(value: Any) -> np.ndarray:
    six = np.asarray(value, dtype=np.float64)
    if six.shape != (6,) or not np.isfinite(six).all():
        raise ValueError(f"rotation 6D must be finite shape (6,), got {six.shape}")

    # Match x2robot_dataset_v2 RotationConverter.rotation_6d_to_matrix.
    # Model predictions are only approximately valid rotation rows, so they
    # must be projected back to SO(3) before composing with the reference pose.
    # Directly appending cross(r1, r2) leaves scale/skew in the matrix and can
    # turn small 6D prediction errors into large wrist-orientation errors.
    eps = 1e-12
    row1 = six[:3].copy()
    row2 = six[3:].copy()
    row1_norm = float(np.linalg.norm(row1))
    if row1_norm <= eps:
        raise ValueError("rotation 6D first row is degenerate")
    row1 /= row1_norm + eps
    row2 -= float(np.dot(row1, row2)) * row1
    row2_norm = float(np.linalg.norm(row2))
    if row2_norm <= eps:
        raise ValueError("rotation 6D second row is degenerate")
    row2 /= row2_norm + eps
    matrix = np.stack((row1, row2, np.cross(row1, row2)), axis=0)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(f"rotation 6D produced invalid determinant {determinant}")
    return matrix


def quaternion_wxyz_to_rot6d(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(
            f"quaternion must be finite shape (4,), got {quaternion.shape}"
        )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError("quaternion norm is zero")
    xyzw = np.roll(quaternion / norm, -1)
    return Rotation.from_quat(xyzw).as_matrix()[:2].reshape(6)


def _pose_wxyz_to_position_rotation(
    value: Any, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be finite shape (7,), got {pose.shape}")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError(f"{name} quaternion norm is zero")
    rotation = Rotation.from_quat(np.roll(quaternion / norm, -1)).as_matrix()
    return pose[:3], rotation


def _rotmat_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.roll(xyzw, 1).astype(np.float64)


def _coordinate_rotation(source: str, target: str) -> np.ndarray:
    source_name = str(source).strip().lower()
    target_name = str(target).strip().lower()
    supported = {RFU_FRAME, LFU_FRAME}
    if source_name not in supported or target_name not in supported:
        raise ValueError(
            f"coordinate frames must be one of {sorted(supported)}, got "
            f"source={source!r} target={target!r}"
        )
    if source_name == target_name:
        return np.eye(3, dtype=np.float64)
    return RFU_TO_LFU if source_name == RFU_FRAME else RFU_TO_LFU.T


def transform_absolute20_coordinate_frame(
    values: Any, *, source: str, target: str
) -> np.ndarray:
    """Transform absolute dual-arm position/rotation-6D values between frames.

    The leading dimensions are preserved. Gripper values remain absolute and
    unchanged. Rotation-6D is decoded/projected and then re-encoded row-major,
    exactly matching the training action processor.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"absolute values must end in ({ACTION_DIM},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("absolute values contain non-finite entries")
    transform = _coordinate_rotation(source, target)
    if np.array_equal(transform, np.eye(3, dtype=np.float64)):
        return np.asarray(array, dtype=np.float32).copy()
    flat = array.reshape(-1, ACTION_DIM)
    converted = np.array(flat, copy=True)
    for offset in (0, 10):
        converted[:, offset : offset + 3] = flat[:, offset : offset + 3] @ transform.T
        for index in range(flat.shape[0]):
            rotation = sixd_to_rotmat(flat[index, offset + 3 : offset + 9])
            converted[index, offset + 3 : offset + 9] = (transform @ rotation)[
                :2
            ].reshape(6)
    return converted.reshape(array.shape).astype(np.float32)


def transform_action_dicts_coordinate_frame(
    actions: Sequence[Mapping[str, Any]], *, source: str, target: str
) -> list[dict[str, np.ndarray]]:
    """Transform absolute RoboDojo-style action dictionaries between frames."""

    transform = _coordinate_rotation(source, target)
    converted: list[dict[str, np.ndarray]] = []
    for action in actions:
        validate_action_dict(action)
        result: dict[str, np.ndarray] = {}
        for arm in ("left", "right"):
            pose_key = f"{arm}_ee_pose"
            gripper_key = f"{arm}_ee_joint_state"
            position, rotation = _pose_wxyz_to_position_rotation(
                action[pose_key], name=pose_key
            )
            target_pose = np.concatenate(
                (
                    transform @ position,
                    _rotmat_to_quaternion_wxyz(transform @ rotation),
                )
            ).astype(np.float32)
            result[pose_key] = target_pose
            result[gripper_key] = np.asarray(
                action[gripper_key], dtype=np.float32
            ).copy()
        validate_action_dict(result)
        converted.append(result)
    return converted


def build_agent_pos(observation: Mapping[str, Any]) -> np.ndarray:
    state = _require_mapping(observation.get("state"), "observation['state']")
    return build_agent_pos_from_state(state)


def build_agent_pos_from_state(state: Mapping[str, Any]) -> np.ndarray:
    state = _require_mapping(state, "state")
    result: list[np.ndarray] = []
    for arm in ("left", "right"):
        pose_key = f"{arm}_ee_pose"
        gripper_key = f"{arm}_ee_joint_state"
        pose = np.asarray(state.get(pose_key), dtype=np.float64)
        gripper = np.asarray(state.get(gripper_key), dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"state[{pose_key!r}] must be finite shape (7,)")
        if gripper.shape != (1,) or not np.isfinite(gripper).all():
            raise ValueError(f"state[{gripper_key!r}] must be finite shape (1,)")
        # The materialized sim_x2 training labels and RoboDojo observations
        # share the same raw link6 world-pose contract.
        result.extend(
            (
                pose[:3],
                quaternion_wxyz_to_rot6d(pose[3:]),
                gripper,
            )
        )
    agent_pos = np.concatenate(result).astype(np.float32)
    if agent_pos.shape != (ACTION_DIM,):
        raise RuntimeError(f"agent_pos shape {agent_pos.shape} != ({ACTION_DIM},)")
    return agent_pos


def extract_executed_agent_pos(
    payload: Mapping[str, Any], *, action_steps: int = ACTION_STEPS
) -> np.ndarray:
    """Decode post-action RoboDojo states to absolute 20-D poses."""

    raw = _require_mapping(payload.get("executed_state"), "_vap.executed_state")
    arrays: dict[str, np.ndarray] = {}
    for key, dimension in ACTION_DIMS.items():
        value = np.asarray(raw.get(key), dtype=np.float64)
        expected = (action_steps, dimension)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(
                f"executed_state[{key!r}] must be finite shape {expected}, "
                f"got {value.shape}"
            )
        arrays[key] = value
    return np.stack(
        [
            build_agent_pos_from_state({key: arrays[key][index] for key in ACTION_DIMS})
            for index in range(action_steps)
        ],
        axis=0,
    )


def absolute20_to_relative20(
    reference: np.ndarray, absolute_actions: np.ndarray
) -> np.ndarray:
    """Encode executed absolute states with the training relative-action rule."""

    reference = np.asarray(reference, dtype=np.float64)
    absolute = np.asarray(absolute_actions, dtype=np.float64)
    if reference.shape != (ACTION_DIM,):
        raise ValueError(f"reference shape {reference.shape} != ({ACTION_DIM},)")
    if absolute.ndim != 2 or absolute.shape[1] != ACTION_DIM or absolute.shape[0] <= 0:
        raise ValueError(
            f"absolute action shape {absolute.shape} must be [T,{ACTION_DIM}]"
        )
    if not np.isfinite(reference).all() or not np.isfinite(absolute).all():
        raise ValueError("reference/absolute action contains non-finite values")

    relative = np.array(absolute, copy=True)
    for offset in (0, 10):
        relative[:, offset : offset + 3] -= reference[offset : offset + 3]
        reference_rotation = sixd_to_rotmat(reference[offset + 3 : offset + 9])
        for index in range(absolute.shape[0]):
            absolute_rotation = sixd_to_rotmat(absolute[index, offset + 3 : offset + 9])
            local_rotation = absolute_rotation @ reference_rotation.T
            relative[index, offset + 3 : offset + 9] = local_rotation[:2].reshape(6)
        # Gripper remains an absolute target, exactly as in training.
        relative[:, offset + 9] = absolute[:, offset + 9]
    return relative.astype(np.float32)


def relative20_to_action_dicts(
    reference: np.ndarray, relative_actions: np.ndarray
) -> list[dict[str, np.ndarray]]:
    """Decode physical relative-20 actions to RoboDojo EE dictionaries."""

    reference = np.asarray(reference, dtype=np.float64)
    actions = np.asarray(relative_actions, dtype=np.float64)
    if reference.shape != (ACTION_DIM,):
        raise ValueError(f"reference shape {reference.shape} != ({ACTION_DIM},)")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] <= 0:
        raise ValueError(
            f"relative action shape {actions.shape} must be [T,{ACTION_DIM}]"
        )
    if not np.isfinite(reference).all() or not np.isfinite(actions).all():
        raise ValueError("reference/action contains non-finite values")

    ref_rot = (sixd_to_rotmat(reference[3:9]), sixd_to_rotmat(reference[13:19]))
    decoded = []
    for row in actions:
        arm_results = []
        for arm_index, offset in enumerate((0, 10)):
            position = reference[offset : offset + 3] + row[offset : offset + 3]
            rotation = sixd_to_rotmat(row[offset + 3 : offset + 9]) @ ref_rot[arm_index]
            pose = np.concatenate((position, _rotmat_to_quaternion_wxyz(rotation)))
            gripper = np.array([np.clip(row[offset + 9], 0.0, 1.0)])
            arm_results.append((pose.astype(np.float32), gripper.astype(np.float32)))
        action = {
            "left_ee_pose": arm_results[0][0],
            "left_ee_joint_state": arm_results[0][1],
            "right_ee_pose": arm_results[1][0],
            "right_ee_joint_state": arm_results[1][1],
        }
        validate_action_dict(action)
        decoded.append(action)
    return decoded


def validate_action_dict(action: Mapping[str, Any]) -> None:
    unexpected = sorted(set(action) - set(ACTION_DIMS))
    missing = sorted(set(ACTION_DIMS) - set(action))
    if unexpected or missing:
        raise ValueError(
            f"invalid action keys; missing={missing}, unexpected={unexpected}"
        )
    for key, dimension in ACTION_DIMS.items():
        value = np.asarray(action[key])
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(
                f"action[{key!r}] must be finite shape ({dimension},), "
                f"got {value.shape}"
            )


def copy_raw_camera_frames(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        camera: ensure_uint8_rgb(
            _camera_color(observation, camera),
            name=f"vision[{camera!r}]['color']",
        )
        for camera in CAMERA_NAMES
    }


def stack_frame_history(
    frames: Mapping[str, Sequence[np.ndarray]],
    *,
    video_frames: int = VIDEO_FRAMES,
) -> dict[str, np.ndarray]:
    result = {}
    for camera in CAMERA_NAMES:
        camera_frames = list(frames.get(camera, ()))
        if len(camera_frames) != video_frames:
            raise ValueError(
                f"camera {camera!r} has {len(camera_frames)} history frames; "
                f"expected {video_frames}"
            )
        result[camera] = np.stack(camera_frames, axis=0)
    return result


__all__ = [
    "ACTION_DIM",
    "ACTION_DIMS",
    "ACTION_STEPS",
    "CAMERA_NAMES",
    "VIDEO_FRAMES",
    "VIEW_ORDER",
    "LFU_FRAME",
    "RFU_FRAME",
    "RFU_TO_LFU",
    "absolute20_to_relative20",
    "build_agent_pos",
    "build_agent_pos_from_state",
    "build_anchor_and_history",
    "copy_raw_camera_frames",
    "extract_executed_video",
    "extract_executed_agent_pos",
    "extract_resized_views",
    "relative20_to_action_dicts",
    "stack_frame_history",
    "transform_absolute20_coordinate_frame",
    "transform_action_dicts_coordinate_frame",
]
