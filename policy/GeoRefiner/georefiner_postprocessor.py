"""GeoRefiner inference integration for the standalone XPolicyLab adapter.

This module deliberately keeps the benchmark/action-space bridge separate from
the X-VLA model lifecycle.  GeoRefiner is imported lazily so ``disabled`` mode
has exactly the same dependency and model-loading behavior as the base policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation


LOGGER = logging.getLogger(__name__)

CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "left_wrist": ("cam_left_wrist", "left_wrist", "left_wrist_camera"),
    "right_wrist": ("cam_right_wrist", "right_wrist", "right_wrist_camera"),
}
_IMAGE_VALUE_KEYS = ("color", "rgb", "image", "colors")


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Return an RGB image as contiguous HWC uint8 data."""

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 RGB image, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("RGB image contains NaN or Inf values.")
        image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] in (1, 3, 4):
        image = image[..., :3]
    elif image.shape[0] in (1, 3, 4):
        image = np.transpose(image[:3], (1, 2, 0))
    else:
        raise ValueError(f"Unsupported RGB image layout: {image.shape}.")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image)


def extract_camera_image(observation: Mapping[str, Any], camera: str) -> Any:
    """Read a logical camera from either supported RoboDojo observation layout."""

    if camera not in CAMERA_ALIASES:
        raise KeyError(f"Unknown logical camera {camera!r}.")
    aliases = CAMERA_ALIASES[camera]
    available: set[str] = set()
    for container_name in ("images", "vision"):
        container = observation.get(container_name, {})
        if not isinstance(container, Mapping):
            continue
        available.update(str(key) for key in container)
        for alias in aliases:
            if alias not in container:
                continue
            value = container[alias]
            if isinstance(value, Mapping):
                for image_key in _IMAGE_VALUE_KEYS:
                    if image_key in value:
                        return value[image_key]
            else:
                return value
    raise KeyError(
        f"Could not find {camera!r} RGB camera. Tried aliases {aliases}; "
        f"available cameras: {sorted(available)}."
    )


def _state_vector(state: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Observation state is missing {key!r}.")
    value = np.asarray(state[key], dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"State {key!r} must have shape [{size}], got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"State {key!r} contains NaN or Inf.")
    return value


@dataclass(slots=True)
class GeoRefinerObservation:
    """The observation fields consumed after X-VLA nominal inference."""

    env_idx: int
    instruction: str
    head_rgb: np.ndarray
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    current_poses: np.ndarray
    current_grippers: np.ndarray

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, Any],
        *,
        instruction: str,
        env_idx: int,
        head_rgb: np.ndarray | None = None,
    ) -> "GeoRefinerObservation":
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise KeyError("Observation must contain a state mapping.")
        left_pose = _state_vector(state, "left_ee_pose", 7)
        right_pose = _state_vector(state, "right_ee_pose", 7)
        left_gripper = _state_vector(state, "left_ee_joint_state", 1)[0]
        right_gripper = _state_vector(state, "right_ee_joint_state", 1)[0]
        return cls(
            env_idx=int(env_idx),
            instruction=instruction,
            head_rgb=(
                ensure_hwc_uint8(extract_camera_image(observation, "head"))
                if head_rgb is None
                else ensure_hwc_uint8(head_rgb)
            ),
            left_wrist_rgb=ensure_hwc_uint8(
                extract_camera_image(observation, "left_wrist")
            ),
            right_wrist_rgb=ensure_hwc_uint8(
                extract_camera_image(observation, "right_wrist")
            ),
            current_poses=np.stack((left_pose, right_pose), axis=0),
            current_grippers=np.asarray(
                [left_gripper, right_gripper], dtype=np.float64
            ),
        )


class XVLAActionBridge:
    """Convert X-VLA absolute 20D chunks to/from GeoRefiner canonical actions."""

    NATIVE_ACTION_DIM = 20
    ARM_ACTION_DIM = 10
    CANONICAL_ACTION_DIM = 7

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray, *, name: str) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if np.any(norms < 1e-10):
            raise ValueError(f"{name} contains zero-length vectors.")
        return vectors / norms

    @classmethod
    def rotation6d_to_matrix(cls, rotation6d: np.ndarray) -> np.ndarray:
        """Decode the interleaved first-two-column representation into SO(3)."""

        value = np.asarray(rotation6d, dtype=np.float64)
        if value.shape[-1] != 6 or not np.isfinite(value).all():
            raise ValueError(
                f"rotation6d must be finite with last dimension 6, got {value.shape}."
            )
        first = value[..., 0:5:2]
        second = value[..., 1:6:2]

        first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
        fallback_first = np.zeros_like(first)
        fallback_first[..., 0] = 1.0
        first = np.where(first_norm < 1e-10, fallback_first, first)
        first = first / np.linalg.norm(first, axis=-1, keepdims=True)

        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
        basis_index = np.argmin(np.abs(first), axis=-1)
        fallback_second = np.zeros_like(second)
        np.put_along_axis(
            fallback_second, basis_index[..., None], 1.0, axis=-1
        )
        fallback_second -= (
            np.sum(first * fallback_second, axis=-1, keepdims=True) * first
        )
        fallback_second /= np.linalg.norm(
            fallback_second, axis=-1, keepdims=True
        )
        second = np.where(second_norm < 1e-10, fallback_second, second)
        second = second / np.linalg.norm(second, axis=-1, keepdims=True)
        third = np.cross(first, second)
        matrix = np.stack((first, second, third), axis=-1)
        cls.validate_rotation_matrices(matrix)
        return matrix

    @classmethod
    def matrix_to_rotation6d(cls, matrix: np.ndarray) -> np.ndarray:
        matrix = cls.project_to_rotation_matrix(matrix)
        return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))

    @classmethod
    def project_to_rotation_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
            raise ValueError(
                f"Rotation matrices must be finite [...,3,3], got {value.shape}."
            )
        u, _, vh = np.linalg.svd(value)
        projected = u @ vh
        signs = np.ones(value.shape[:-2] + (3,), dtype=np.float64)
        signs[..., -1] = np.where(np.linalg.det(projected) < 0.0, -1.0, 1.0)
        projected = (u * signs[..., None, :]) @ vh
        cls.validate_rotation_matrices(projected)
        return projected

    @staticmethod
    def validate_rotation_matrices(matrix: np.ndarray, atol: float = 1e-5) -> None:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid rotation matrix shape/content: {value.shape}.")
        identity = np.eye(3, dtype=np.float64)
        orthogonal_error = np.max(
            np.abs(value @ np.swapaxes(value, -1, -2) - identity)
        )
        determinants = np.linalg.det(value)
        if orthogonal_error > atol or np.max(np.abs(determinants - 1.0)) > atol:
            raise ValueError(
                "Rotation matrix is not in SO(3): "
                f"orthogonal_error={orthogonal_error:.3e}, "
                f"determinant_range=[{determinants.min():.6f}, "
                f"{determinants.max():.6f}]."
            )

    @classmethod
    def wxyz_to_matrix(cls, quaternion: np.ndarray) -> np.ndarray:
        value = np.asarray(quaternion, dtype=np.float64)
        if value.shape[-1] != 4 or not np.isfinite(value).all():
            raise ValueError(
                f"Quaternion must be finite with last dimension 4, got {value.shape}."
            )
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        if np.any(norm < 1e-10):
            raise ValueError("Quaternion contains a zero-length value.")
        value = value / norm
        xyzw = np.concatenate((value[..., 1:], value[..., :1]), axis=-1)
        matrix = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix()
        matrix = matrix.reshape(value.shape[:-1] + (3, 3))
        cls.validate_rotation_matrices(matrix)
        return matrix

    @classmethod
    def matrix_to_wxyz(cls, matrix: np.ndarray) -> np.ndarray:
        matrix = cls.project_to_rotation_matrix(matrix)
        xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
        xyzw = xyzw.reshape(matrix.shape[:-2] + (4,))
        wxyz = np.concatenate((xyzw[..., 3:], xyzw[..., :3]), axis=-1)
        return cls._normalize_vectors(wxyz, name="Quaternion")

    @classmethod
    def matrix_to_rotvec(cls, matrix: np.ndarray) -> np.ndarray:
        matrix = cls.project_to_rotation_matrix(matrix)
        result = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec()
        return result.reshape(matrix.shape[:-2] + (3,))

    @classmethod
    def rotvec_to_matrix(cls, rotvec: np.ndarray) -> np.ndarray:
        value = np.asarray(rotvec, dtype=np.float64)
        if value.shape[-1] != 3 or not np.isfinite(value).all():
            raise ValueError(
                f"Rotation vectors must be finite [...,3], got {value.shape}."
            )
        matrix = Rotation.from_rotvec(value.reshape(-1, 3)).as_matrix()
        matrix = matrix.reshape(value.shape[:-1] + (3, 3))
        cls.validate_rotation_matrices(matrix)
        return matrix

    @classmethod
    def validate_action_chunk(cls, action_chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(action_chunk, dtype=np.float64)
        if value.ndim != 3 or value.shape[-1] != cls.NATIVE_ACTION_DIM:
            raise ValueError(
                "X-VLA action must have shape [B,T,20], "
                f"got {value.shape}."
            )
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(f"X-VLA action batch/horizon must be non-empty: {value.shape}.")
        if not np.isfinite(value).all():
            raise FloatingPointError("X-VLA action contains NaN or Inf.")
        cls.rotation6d_to_matrix(value[..., 3:9])
        cls.rotation6d_to_matrix(value[..., 13:19])
        return value

    @classmethod
    def validate_robot_state(
        cls, current_poses: np.ndarray, current_grippers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(current_poses, dtype=np.float64)
        grippers = np.asarray(current_grippers, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (2, 7):
            raise ValueError(f"current_poses must have shape [B,2,7], got {poses.shape}.")
        if grippers.shape != poses.shape[:2]:
            raise ValueError(
                f"current_grippers must have shape {poses.shape[:2]}, got {grippers.shape}."
            )
        if not np.isfinite(poses).all() or not np.isfinite(grippers).all():
            raise FloatingPointError("Current robot state contains NaN or Inf.")
        cls.wxyz_to_matrix(poses[..., 3:7])
        return poses, np.clip(grippers, 0.0, 1.0)

    @classmethod
    def absolute_to_canonical(
        cls, action_chunk: np.ndarray, current_poses: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return left/right ``[B,T,7]`` world-frame incremental streams."""

        action = cls.validate_action_chunk(action_chunk)
        poses = np.asarray(current_poses, dtype=np.float64)
        if poses.shape != (action.shape[0], 2, 7):
            raise ValueError(
                f"current_poses must have shape {(action.shape[0], 2, 7)}, "
                f"got {poses.shape}."
            )
        current_rotations = cls.wxyz_to_matrix(poses[..., 3:7])
        results: list[np.ndarray] = []
        for arm_index, offset in enumerate((0, cls.ARM_ACTION_DIM)):
            branch = action[..., offset : offset + cls.ARM_ACTION_DIM]
            absolute_positions = branch[..., :3]
            absolute_rotations = cls.rotation6d_to_matrix(branch[..., 3:9])
            previous_positions = np.concatenate(
                (poses[:, arm_index, None, :3], absolute_positions[:, :-1]), axis=1
            )
            previous_rotations = np.concatenate(
                (
                    current_rotations[:, arm_index, None],
                    absolute_rotations[:, :-1],
                ),
                axis=1,
            )
            delta_positions = absolute_positions - previous_positions
            delta_rotations = absolute_rotations @ np.swapaxes(
                previous_rotations, -1, -2
            )
            delta_rotvecs = cls.matrix_to_rotvec(delta_rotations)
            canonical = np.concatenate(
                (delta_positions, delta_rotvecs, branch[..., 9:10]), axis=-1
            )
            if not np.isfinite(canonical).all():
                raise FloatingPointError("Canonical action contains NaN or Inf.")
            results.append(canonical.astype(np.float32))
        return results[0], results[1]

    @classmethod
    def build_canonical_state(
        cls, current_poses: np.ndarray, current_grippers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build left/right ``[B,16]`` self/other/identity state vectors."""

        poses, grippers = cls.validate_robot_state(current_poses, current_grippers)
        rotvecs = cls.matrix_to_rotvec(cls.wxyz_to_matrix(poses[..., 3:7]))
        arm_states = np.concatenate(
            (poses[..., :3], rotvecs, grippers[..., None]), axis=-1
        )
        batch_size = poses.shape[0]
        left_identity = np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float64), (batch_size, 1)
        )
        right_identity = np.tile(
            np.asarray([[0.0, 1.0]], dtype=np.float64), (batch_size, 1)
        )
        left = np.concatenate(
            (arm_states[:, 0], arm_states[:, 1], left_identity), axis=-1
        )
        right = np.concatenate(
            (arm_states[:, 1], arm_states[:, 0], right_identity), axis=-1
        )
        if left.shape != (batch_size, 16) or right.shape != (batch_size, 16):
            raise AssertionError("Internal canonical state construction failed.")
        return left.astype(np.float32), right.astype(np.float32)

    @classmethod
    def _integrate_branch(
        cls,
        canonical: np.ndarray,
        current_pose: np.ndarray,
        nominal_branch: np.ndarray,
        *,
        refined_gripper: bool,
        max_position_deviation_m: float,
        max_rotation_deviation_rad: float,
    ) -> np.ndarray:
        canonical = np.asarray(canonical, dtype=np.float64)
        nominal = np.asarray(nominal_branch, dtype=np.float64)
        pose = np.asarray(current_pose, dtype=np.float64)
        if canonical.ndim != 3 or canonical.shape[-1] != cls.CANONICAL_ACTION_DIM:
            raise ValueError(
                f"Canonical action must have shape [B,T,7], got {canonical.shape}."
            )
        if nominal.shape != canonical.shape[:2] + (cls.ARM_ACTION_DIM,):
            raise ValueError(
                f"Nominal branch shape {nominal.shape} does not match {canonical.shape}."
            )
        if pose.shape != (canonical.shape[0], 7):
            raise ValueError(
                f"Current branch pose must have shape {(canonical.shape[0], 7)}, "
                f"got {pose.shape}."
            )
        if not np.isfinite(canonical).all():
            raise FloatingPointError("Refined canonical action contains NaN or Inf.")

        nominal_rotations = cls.rotation6d_to_matrix(nominal[..., 3:9])
        position = pose[:, :3].copy()
        rotation = cls.wxyz_to_matrix(pose[:, 3:7])
        result = np.empty_like(nominal)
        for step in range(canonical.shape[1]):
            candidate_position = position + canonical[:, step, :3]
            candidate_rotation = (
                cls.rotvec_to_matrix(canonical[:, step, 3:6]) @ rotation
            )

            position_correction = candidate_position - nominal[:, step, :3]
            position_norm = np.linalg.norm(
                position_correction, axis=-1, keepdims=True
            )
            position_scale = np.minimum(
                1.0,
                max_position_deviation_m / np.maximum(position_norm, 1e-12),
            )
            position = nominal[:, step, :3] + position_correction * position_scale

            rotation_correction = candidate_rotation @ np.swapaxes(
                nominal_rotations[:, step], -1, -2
            )
            correction_rotvec = cls.matrix_to_rotvec(rotation_correction)
            correction_angle = np.linalg.norm(
                correction_rotvec, axis=-1, keepdims=True
            )
            rotation_scale = np.minimum(
                1.0,
                max_rotation_deviation_rad / np.maximum(correction_angle, 1e-12),
            )
            rotation = (
                cls.rotvec_to_matrix(correction_rotvec * rotation_scale)
                @ nominal_rotations[:, step]
            )

            result[:, step, :3] = position
            result[:, step, 3:9] = cls.matrix_to_rotation6d(rotation)
            gripper = (
                canonical[:, step, 6]
                if refined_gripper
                else nominal[:, step, 9]
            )
            result[:, step, 9] = np.clip(gripper, 0.0, 1.0)
        return result

    @classmethod
    def canonical_to_absolute(
        cls,
        left_canonical: np.ndarray,
        right_canonical: np.ndarray,
        current_poses: np.ndarray,
        nominal_action: np.ndarray,
        *,
        refined_gripper: bool = False,
        max_position_deviation_m: float = np.inf,
        max_rotation_deviation_rad: float = np.inf,
    ) -> np.ndarray:
        """Integrate canonical streams and merge them into an X-VLA 20D chunk."""

        nominal = cls.validate_action_chunk(nominal_action)
        poses = np.asarray(current_poses, dtype=np.float64)
        left = cls._integrate_branch(
            left_canonical,
            poses[:, 0],
            nominal[..., :10],
            refined_gripper=refined_gripper,
            max_position_deviation_m=max_position_deviation_m,
            max_rotation_deviation_rad=max_rotation_deviation_rad,
        )
        right = cls._integrate_branch(
            right_canonical,
            poses[:, 1],
            nominal[..., 10:20],
            refined_gripper=refined_gripper,
            max_position_deviation_m=max_position_deviation_m,
            max_rotation_deviation_rad=max_rotation_deviation_rad,
        )
        merged = np.concatenate((left, right), axis=-1).astype(np.float32)
        cls.validate_action_chunk(merged)
        return merged

    @classmethod
    def maximum_pose_deviation(
        cls, refined: np.ndarray, nominal: np.ndarray
    ) -> tuple[float, float]:
        refined = cls.validate_action_chunk(refined)
        nominal = cls.validate_action_chunk(nominal)
        position_max = 0.0
        rotation_max = 0.0
        for offset in (0, cls.ARM_ACTION_DIM):
            position_max = max(
                position_max,
                float(
                    np.linalg.norm(
                        refined[..., offset : offset + 3]
                        - nominal[..., offset : offset + 3],
                        axis=-1,
                    ).max()
                ),
            )
            refined_rotation = cls.rotation6d_to_matrix(
                refined[..., offset + 3 : offset + 9]
            )
            nominal_rotation = cls.rotation6d_to_matrix(
                nominal[..., offset + 3 : offset + 9]
            )
            difference = refined_rotation @ np.swapaxes(
                nominal_rotation, -1, -2
            )
            rotation_max = max(
                rotation_max,
                float(np.linalg.norm(cls.matrix_to_rotvec(difference), axis=-1).max()),
            )
        return position_max, rotation_max


@dataclass(frozen=True, slots=True)
class GeoRefinerPostProcessorConfig:
    mode: str = "refine"
    root: str | None = None
    checkpoint_path: str | None = None
    artifact_dir: str | None = None
    device: str = "cuda"
    dtype: str = "float32"
    micro_batch_size: int = 2
    refine_gripper: bool = False
    fallback_to_nominal: bool = True
    max_position_deviation_m: float = 0.02
    max_rotation_deviation_rad: float = 0.15
    log_interval: int = 50

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "GeoRefinerPostProcessorConfig":
        values = dict(values or {})
        # Keep the checked-in deploy.yml useful for GeoRefiner experiments while
        # allowing a process-local, unambiguous X-VLA baseline.  The launcher
        # inherits this variable into the policy server process.
        environment_mode = os.environ.get("GEOREFINER_MODE")
        if environment_mode is not None:
            values["mode"] = environment_mode.strip()
        known = cls.__dataclass_fields__.keys()
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"Unknown georefiner configuration keys: {unknown}.")
        config = cls(**values)
        if config.mode not in {"disabled", "shadow", "refine"}:
            raise ValueError(
                "georefiner.mode must be disabled, shadow, or refine; "
                f"got {config.mode!r}."
            )
        if config.dtype not in {"float32", "bfloat16", "bf16"}:
            raise ValueError(
                "georefiner.dtype must be float32, bfloat16, or bf16; "
                f"got {config.dtype!r}."
            )
        if config.micro_batch_size <= 0 or config.log_interval <= 0:
            raise ValueError("micro_batch_size and log_interval must be positive.")
        if config.max_position_deviation_m <= 0:
            raise ValueError("max_position_deviation_m must be positive.")
        if config.max_rotation_deviation_rad <= 0:
            raise ValueError("max_rotation_deviation_rad must be positive.")
        return config


class GeoRefinerPostProcessor:
    """Run shared-weight, batched dual-arm GeoRefiner postprocessing."""

    def __init__(
        self,
        config: Mapping[str, Any] | GeoRefinerPostProcessorConfig | None,
        *,
        action_horizon: int,
        policy_dir: Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, GeoRefinerPostProcessorConfig)
            else GeoRefinerPostProcessorConfig.from_mapping(config)
        )
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        self.policy_dir = (policy_dir or Path(__file__).resolve().parent).resolve()
        self.bridge = XVLAActionBridge()
        self.model: torch.nn.Module | None = None
        self.input_type: type[Any] | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.resolved_root: Path | None = None
        self.resolved_checkpoint: Path | None = None
        self.resolved_artifact_dir: Path | None = None
        self.checkpoint_load_info: dict[str, Any] | None = None
        self._call_count = 0

        if self.enabled:
            self._load_model()

    @property
    def enabled(self) -> bool:
        return self.config.mode != "disabled"

    @staticmethod
    def _resolve_path(value: str | None, base: Path) -> Path | None:
        if value is None or not str(value).strip():
            return None
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()

    def _resolve_model_paths(self) -> tuple[Path, Path, Path]:
        root_value = self.config.root or os.environ.get("GEOREFINER_ROOT")
        # The PR-ready default is the inference package vendored in this policy.
        # GEOREFINER_ROOT remains available for development against an external
        # checkout without requiring any machine-specific path in configuration.
        root = self._resolve_path(root_value, self.policy_dir) or self.policy_dir

        checkpoint_value = self.config.checkpoint_path or os.environ.get(
            "GEOREFINER_CHECKPOINT"
        )
        checkpoint = self._resolve_path(checkpoint_value, root)
        if checkpoint is None:
            checkpoint = (
                root
                / "checkpoints"
                / "georefiner"
                / "georefiner_xvla_trained_fp32.pt"
            ).resolve()

        artifact_value = self.config.artifact_dir or os.environ.get(
            "GEOREFINER_ARTIFACT_DIR"
        )
        artifact = self._resolve_path(artifact_value, root) or checkpoint.parent
        return root, checkpoint, artifact.resolve()

    def _load_model(self) -> None:
        root, checkpoint, artifact = self._resolve_model_paths()
        self.resolved_root = root
        self.resolved_checkpoint = checkpoint
        self.resolved_artifact_dir = artifact
        missing = [
            f"GeoRefiner root: {root}" if not root.is_dir() else None,
            f"GeoRefiner checkpoint: {checkpoint}" if not checkpoint.is_file() else None,
            f"GeoRefiner artifact dir: {artifact}" if not artifact.is_dir() else None,
        ]
        missing = [item for item in missing if item is not None]
        if missing:
            raise FileNotFoundError(
                "GeoRefiner integration paths are missing:\n  - "
                + "\n  - ".join(missing)
                + "\nResolution priority is deploy.yml, GEOREFINER_* environment "
                "variables, then policy/GeoRefiner and its checkpoints/georefiner "
                "asset directory."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"GeoRefiner resolved device {self.device}, but CUDA is unavailable."
            )
        self.dtype = (
            torch.float32
            if self.config.dtype == "float32"
            else torch.bfloat16
        )

        try:
            package = importlib.import_module("georefiner")
            types_module = importlib.import_module("georefiner.types")
            model_class = getattr(package, "GeoRefiner")
            self.input_type = getattr(types_module, "GeoRefinerInput")
            model = model_class.from_checkpoint(
                checkpoint,
                artifact_dir=artifact,
                map_location="cpu",
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to strictly load GeoRefiner without a mock fallback. "
                f"resolved root={root}, checkpoint={checkpoint}, "
                f"artifact_dir={artifact}."
            ) from exc

        info = dict(getattr(model, "_checkpoint_load_info", {}))
        missing_keys = list(info.get("missing_keys", []))
        unexpected_keys = list(info.get("unexpected_keys", []))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "GeoRefiner strict checkpoint restore reported incompatible keys: "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}."
            )
        if model.config.action_dim != 7 or model.config.state_dim != 16:
            raise ValueError(
                "GeoRefiner checkpoint is incompatible with the X-VLA bridge: "
                f"action_dim={model.config.action_dim}, state_dim={model.config.state_dim}."
            )
        model.config.max_action_horizon = max(
            int(model.config.max_action_horizon), self.action_horizon
        )
        model.config.return_intermediates = False
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.checkpoint_load_info = {
            **info,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
        LOGGER.warning(
            "Loaded GeoRefiner checkpoint %s (missing_keys=[], unexpected_keys=[], "
            "horizon=%d, dtype=%s). DINOv2 and CLIP are pretrained, but the "
            "GeoRefiner core checkpoint is an untrained initialization; successful "
            "execution does not imply benchmark improvement.",
            checkpoint,
            model.config.max_action_horizon,
            self.config.dtype,
        )

    @staticmethod
    def _stack_rgb(images: Sequence[np.ndarray]) -> torch.Tensor:
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(
                "GeoRefiner cameras in one visual batch must share HWC shape; "
                f"got {sorted(shapes)}."
            )
        array = np.stack(images, axis=0)
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def _encode_visuals(
        self, observations: Sequence[GeoRefinerObservation]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.model is not None and self.device is not None
        backbone = self.model.encoder.geometry_encoder.visual_backbone
        head_parts: list[torch.Tensor] = []
        left_parts: list[torch.Tensor] = []
        right_parts: list[torch.Tensor] = []
        micro = self.config.micro_batch_size
        for start in range(0, len(observations), micro):
            group = observations[start : start + micro]
            group_size = len(group)
            images = [item.head_rgb for item in group]
            images.extend(item.left_wrist_rgb for item in group)
            images.extend(item.right_wrist_rgb for item in group)
            image_tensor = self._stack_rgb(images).to(self.device)
            tokens = backbone(image_tensor)
            if tokens.ndim != 3 or tokens.shape[0] != 3 * group_size:
                raise ValueError(
                    "DINOv2 returned an unexpected token shape for the combined "
                    f"3B image batch: {tuple(tokens.shape)}."
                )
            head, left, right = tokens.split(group_size, dim=0)
            head_parts.append(head)
            left_parts.append(left)
            right_parts.append(right)
        return (
            torch.cat(head_parts, dim=0),
            torch.cat(left_parts, dim=0),
            torch.cat(right_parts, dim=0),
        )

    def _encode_language(
        self, observations: Sequence[GeoRefinerObservation]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.model is not None
        backbone = self.model.encoder.language_encoder.text_backbone
        output = backbone(instruction=[item.instruction for item in observations])
        if output.tokens.ndim != 3 or output.tokens.shape[0] != len(observations):
            raise ValueError(
                f"CLIP returned unexpected language token shape {tuple(output.tokens.shape)}."
            )
        return output.tokens, output.padding_mask

    def _forward_group(
        self,
        indices: Sequence[int],
        left_nominal: np.ndarray,
        right_nominal: np.ndarray,
        left_state: np.ndarray,
        right_state: np.ndarray,
        head_tokens: torch.Tensor,
        left_wrist_tokens: torch.Tensor,
        right_wrist_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        language_padding_mask: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert (
            self.model is not None
            and self.input_type is not None
            and self.device is not None
            and self.dtype is not None
        )
        selection = torch.as_tensor(indices, device=head_tokens.device, dtype=torch.long)

        def numeric(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                array[np.asarray(indices)], device=self.device, dtype=self.dtype
            )

        nominal = torch.cat((numeric(left_nominal), numeric(right_nominal)), dim=0)
        state = torch.cat((numeric(left_state), numeric(right_state)), dim=0)
        head = head_tokens.index_select(0, selection)
        head = torch.cat((head, head), dim=0)
        wrists = torch.cat(
            (
                left_wrist_tokens.index_select(0, selection),
                right_wrist_tokens.index_select(0, selection),
            ),
            dim=0,
        )
        language = language_tokens.index_select(0, selection)
        language = torch.cat((language, language), dim=0)
        language_mask = None
        if language_padding_mask is not None:
            language_mask = language_padding_mask.index_select(0, selection)
            language_mask = torch.cat((language_mask, language_mask), dim=0)

        inputs = self.input_type(
            nominal_action=nominal,
            state=state,
            language_tokens=language,
            language_padding_mask=language_mask,
            global_visual_tokens=head,
            wrist_visual_tokens=wrists,
        )
        output = self.model(inputs)
        refined = output.refined_action
        expected = (2 * len(indices), self.action_horizon, 7)
        if tuple(refined.shape) != expected:
            raise ValueError(
                f"GeoRefiner returned shape {tuple(refined.shape)}, expected {expected}."
            )
        if not torch.isfinite(refined).all():
            raise FloatingPointError("GeoRefiner returned NaN or Inf actions.")
        correction = output.applied_correction
        if correction is None or tuple(correction.shape) != expected:
            raise ValueError(
                "GeoRefiner did not return an applied_correction matching its "
                f"refined action; got {None if correction is None else tuple(correction.shape)}."
            )
        if not torch.isfinite(correction).all():
            raise FloatingPointError("GeoRefiner returned NaN or Inf corrections.")
        refined = refined.float().cpu().numpy()
        correction = correction.float().cpu().numpy()
        split = len(indices)
        exactly_zero = np.logical_and(
            np.all(correction[:split] == 0.0, axis=(1, 2)),
            np.all(correction[split:] == 0.0, axis=(1, 2)),
        )
        return refined[:split], refined[split:], exactly_zero

    def _record_failure(self, observation: GeoRefinerObservation, exc: Exception) -> None:
        message = (
            f"GeoRefiner failed for env_idx={observation.env_idx}; "
            f"using nominal X-VLA action: {type(exc).__name__}: {exc}"
        )
        if not self.config.fallback_to_nominal:
            raise RuntimeError(message) from exc
        LOGGER.error(message)

    def process(
        self,
        nominal_chunks: Sequence[np.ndarray],
        observations: Sequence[GeoRefinerObservation] | None = None,
    ) -> list[np.ndarray]:
        """Postprocess one X-VLA chunk per environment, preserving list protocol."""

        source_chunks = [np.asarray(chunk) for chunk in nominal_chunks]
        if not source_chunks:
            return []
        if not self.enabled:
            return [chunk.copy() for chunk in source_chunks]
        nominal_list = [np.asarray(chunk, dtype=np.float32) for chunk in source_chunks]
        if observations is None or len(observations) != len(nominal_list):
            raise ValueError(
                "Enabled GeoRefiner requires one cached observation per nominal chunk."
            )
        if any(chunk.shape != (self.action_horizon, 20) for chunk in nominal_list):
            shapes = [tuple(chunk.shape) for chunk in nominal_list]
            raise ValueError(
                f"Expected X-VLA chunks [{self.action_horizon},20], got {shapes}."
            )

        self._call_count += 1
        nominal_batch = np.stack(nominal_list, axis=0)
        returned = [chunk.copy() for chunk in nominal_list]
        valid_observations: list[GeoRefinerObservation] = []
        valid_original_indices: list[int] = []
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        left_state_parts: list[np.ndarray] = []
        right_state_parts: list[np.ndarray] = []

        for index, observation in enumerate(observations):
            try:
                action = nominal_batch[index : index + 1]
                poses = observation.current_poses[None]
                grippers = observation.current_grippers[None]
                left, right = self.bridge.absolute_to_canonical(action, poses)
                left_state, right_state = self.bridge.build_canonical_state(
                    poses, grippers
                )
            except Exception as exc:
                self._record_failure(observation, exc)
                continue
            valid_observations.append(observation)
            valid_original_indices.append(index)
            left_parts.append(left[0])
            right_parts.append(right[0])
            left_state_parts.append(left_state[0])
            right_state_parts.append(right_state[0])

        if not valid_observations:
            return returned
        left_nominal = np.stack(left_parts)
        right_nominal = np.stack(right_parts)
        left_state = np.stack(left_state_parts)
        right_state = np.stack(right_state_parts)

        try:
            with torch.inference_mode():
                head_tokens, left_wrist_tokens, right_wrist_tokens = (
                    self._encode_visuals(valid_observations)
                )
                language_tokens, language_padding_mask = self._encode_language(
                    valid_observations
                )
        except Exception as exc:
            for observation in valid_observations:
                self._record_failure(observation, exc)
            return returned

        refined_left = left_nominal.copy()
        refined_right = right_nominal.copy()
        forward_success = np.zeros(len(valid_observations), dtype=bool)
        exactly_zero_correction = np.zeros(len(valid_observations), dtype=bool)
        micro = self.config.micro_batch_size
        with torch.inference_mode():
            for start in range(0, len(valid_observations), micro):
                group = list(
                    range(start, min(start + micro, len(valid_observations)))
                )
                try:
                    left, right, zero_correction = self._forward_group(
                        group,
                        left_nominal,
                        right_nominal,
                        left_state,
                        right_state,
                        head_tokens,
                        left_wrist_tokens,
                        right_wrist_tokens,
                        language_tokens,
                        language_padding_mask,
                    )
                    refined_left[group] = left
                    refined_right[group] = right
                    forward_success[group] = True
                    exactly_zero_correction[group] = zero_correction
                    continue
                except Exception as group_exc:
                    if len(group) == 1:
                        self._record_failure(valid_observations[group[0]], group_exc)
                        continue
                for local_index in group:
                    try:
                        left, right, zero_correction = self._forward_group(
                            [local_index],
                            left_nominal,
                            right_nominal,
                            left_state,
                            right_state,
                            head_tokens,
                            left_wrist_tokens,
                            right_wrist_tokens,
                            language_tokens,
                            language_padding_mask,
                        )
                        refined_left[local_index] = left[0]
                        refined_right[local_index] = right[0]
                        forward_success[local_index] = True
                        exactly_zero_correction[local_index] = zero_correction[0]
                    except Exception as exc:
                        self._record_failure(valid_observations[local_index], exc)

        if not self.config.refine_gripper:
            refined_left[..., 6] = left_nominal[..., 6]
            refined_right[..., 6] = right_nominal[..., 6]

        successful_refined: list[np.ndarray] = []
        successful_nominal: list[np.ndarray] = []
        for local_index, succeeded in enumerate(forward_success):
            if not succeeded:
                continue
            original_index = valid_original_indices[local_index]
            observation = valid_observations[local_index]
            try:
                if exactly_zero_correction[local_index]:
                    # The complete GeoRefiner forward has run, but returning the
                    # original representation avoids introducing even the tiny
                    # floating-point error of absolute -> delta -> absolute.
                    refined = nominal_batch[original_index].copy()
                else:
                    refined = self.bridge.canonical_to_absolute(
                        refined_left[local_index : local_index + 1],
                        refined_right[local_index : local_index + 1],
                        observation.current_poses[None],
                        nominal_batch[original_index : original_index + 1],
                        refined_gripper=self.config.refine_gripper,
                        max_position_deviation_m=self.config.max_position_deviation_m,
                        max_rotation_deviation_rad=self.config.max_rotation_deviation_rad,
                    )[0]
                self.bridge.validate_action_chunk(refined[None])
            except Exception as exc:
                self._record_failure(observation, exc)
                continue
            successful_refined.append(refined)
            successful_nominal.append(nominal_batch[original_index])
            if self.config.mode == "refine":
                returned[original_index] = refined

        if successful_refined and self._call_count % self.config.log_interval == 0:
            refined_batch = np.stack(successful_refined)
            nominal_stats = np.stack(successful_nominal)
            position, rotation = self.bridge.maximum_pose_deviation(
                refined_batch, nominal_stats
            )
            LOGGER.info(
                "GeoRefiner mode=%s calls=%d successful_envs=%d/%d "
                "max_position_deviation_m=%.6f max_rotation_deviation_rad=%.6f",
                self.config.mode,
                self._call_count,
                len(successful_refined),
                len(nominal_list),
                position,
                rotation,
            )
        return returned

    def reset_stats(self) -> None:
        self._call_count = 0
