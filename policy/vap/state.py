"""Transactional episode state for closed-loop VAP inference."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .bridge import (
    ACTION_DIM,
    absolute20_to_relative20,
    build_agent_pos,
    build_agent_pos_from_state,
    build_anchor_and_history,
    extract_executed_agent_pos,
    extract_executed_video,
    extract_resized_views,
    relative20_to_action_dicts,
    sixd_to_rotmat,
    transform_absolute20_coordinate_frame,
    transform_action_dicts_coordinate_frame,
)
from .profile import LEGACY_PROFILE, resolve_evaluation_profile


INFERENCE_MODE_AR = "ar"
INFERENCE_MODE_SINGLE_FRAME = "single_frame"
SUPPORTED_INFERENCE_MODES = frozenset({INFERENCE_MODE_AR, INFERENCE_MODE_SINGLE_FRAME})
AR_PREFIX_ACTION_SOURCE_ISSUED = "issued"
AR_PREFIX_ACTION_SOURCE_PRE_FIR = "pre_fir"
AR_PREFIX_ACTION_SOURCE_MEASURED_STATE = "measured_state"
AR_PREFIX_ACTION_SOURCE_POSEGRIP = "measured_pose_issued_gripper"
AR_PREFIX_ACTION_SOURCE_RAW_MODEL = "raw_model"
SUPPORTED_AR_PREFIX_ACTION_SOURCES = frozenset(
    {
        AR_PREFIX_ACTION_SOURCE_ISSUED,
        AR_PREFIX_ACTION_SOURCE_PRE_FIR,
        AR_PREFIX_ACTION_SOURCE_MEASURED_STATE,
        AR_PREFIX_ACTION_SOURCE_POSEGRIP,
        AR_PREFIX_ACTION_SOURCE_RAW_MODEL,
    }
)
SE3_SMOOTHING_MODE_FIR = "fir"
SE3_SMOOTHING_MODE_BOUNDED_FIR = "bounded_fir"
SE3_SMOOTHING_MODE_MIN_JERK = "min_jerk"
SUPPORTED_SE3_SMOOTHING_MODES = frozenset(
    {
        SE3_SMOOTHING_MODE_FIR,
        SE3_SMOOTHING_MODE_BOUNDED_FIR,
        SE3_SMOOTHING_MODE_MIN_JERK,
    }
)


@dataclass(frozen=True)
class GripperHysteresisConfig:
    """Inference-only binary gripper latch with delayed release."""

    close_threshold: float = 0.25
    open_threshold: float = 0.75
    open_confirm_steps: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.close_threshold < self.open_threshold <= 1.0:
            raise ValueError(
                "gripper thresholds must satisfy "
                "0 <= close_threshold < open_threshold <= 1"
            )
        if self.open_confirm_steps <= 0:
            raise ValueError("gripper open_confirm_steps must be positive")


@dataclass(frozen=True)
class GripperHysteresisState:
    """Committed latch state for the left and right grippers."""

    is_open: tuple[bool, bool]
    open_streak: tuple[int, int]


@dataclass(frozen=True)
class SE3SmoothingConfig:
    """Fixed-horizon, inference-only SE(3) trajectory smoothing."""

    mode: str = SE3_SMOOTHING_MODE_FIR
    neighbor_weight: float = 0.25
    acceleration_weight: float = 0.25
    jerk_weight: float = 1.0
    position_max_deviation_m: float = 0.002
    rotation_max_deviation_deg: float = 0.5

    def __post_init__(self) -> None:
        normalized_mode = normalize_se3_smoothing_mode(self.mode)
        object.__setattr__(self, "mode", normalized_mode)
        if not 0.0 <= self.neighbor_weight <= 0.5:
            raise ValueError("SE(3) smoothing neighbor_weight must be in [0, 0.5]")
        for name, value in (
            ("acceleration_weight", self.acceleration_weight),
            ("jerk_weight", self.jerk_weight),
            ("position_max_deviation_m", self.position_max_deviation_m),
            ("rotation_max_deviation_deg", self.rotation_max_deviation_deg),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"SE(3) smoothing {name} must be finite and nonnegative"
                )

    @property
    def center_weight(self) -> float:
        return 1.0 - 2.0 * self.neighbor_weight


def _rotation_matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def normalize_se3_smoothing_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "three_point": SE3_SMOOTHING_MODE_FIR,
        "threepoint": SE3_SMOOTHING_MODE_FIR,
        "bounded_three_point": SE3_SMOOTHING_MODE_BOUNDED_FIR,
        "bounded_threepoint": SE3_SMOOTHING_MODE_BOUNDED_FIR,
        "minimum_jerk": SE3_SMOOTHING_MODE_MIN_JERK,
        "minimumjerk": SE3_SMOOTHING_MODE_MIN_JERK,
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_SE3_SMOOTHING_MODES:
        supported = ", ".join(sorted(SUPPORTED_SE3_SMOOTHING_MODES))
        raise ValueError(f"unsupported SE(3) smoothing mode {value!r}; use {supported}")
    return mode


def normalize_ar_prefix_action_source(value: Any) -> str:
    source = str(value).strip().lower().replace("-", "_")
    aliases = {
        "post_fir": AR_PREFIX_ACTION_SOURCE_ISSUED,
        "executed": AR_PREFIX_ACTION_SOURCE_ISSUED,
        "raw": AR_PREFIX_ACTION_SOURCE_PRE_FIR,
        "pre_smoothing": AR_PREFIX_ACTION_SOURCE_PRE_FIR,
        "model": AR_PREFIX_ACTION_SOURCE_RAW_MODEL,
        "raw_master": AR_PREFIX_ACTION_SOURCE_RAW_MODEL,
        "normalized_master": AR_PREFIX_ACTION_SOURCE_RAW_MODEL,
        "measured": AR_PREFIX_ACTION_SOURCE_MEASURED_STATE,
        "follower": AR_PREFIX_ACTION_SOURCE_MEASURED_STATE,
        "achieved": AR_PREFIX_ACTION_SOURCE_MEASURED_STATE,
    }
    source = aliases.get(source, source)
    if source not in SUPPORTED_AR_PREFIX_ACTION_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_AR_PREFIX_ACTION_SOURCES))
        raise ValueError(
            f"unsupported AR prefix action source {value!r}; use {supported}"
        )
    return source


def _minimum_jerk_series(
    values: np.ndarray,
    *,
    reference: np.ndarray,
    acceleration_weight: float,
    jerk_weight: float,
) -> np.ndarray:
    """Solve one endpoint-constrained quadratic smoother for vector samples."""

    raw = np.asarray(values, dtype=np.float64)
    boundary = np.asarray(reference, dtype=np.float64)
    if raw.ndim != 2 or boundary.shape != (raw.shape[1],):
        raise ValueError("minimum-jerk values/reference shapes are inconsistent")
    if not np.isfinite(raw).all() or not np.isfinite(boundary).all():
        raise ValueError("minimum-jerk values/reference must be finite")
    horizon = raw.shape[0]
    if horizon <= 1 or (acceleration_weight == 0.0 and jerk_weight == 0.0):
        return np.array(raw, copy=True)

    # Difference penalties include the measured current pose as a fixed left
    # boundary. This suppresses a new chunk's initial acceleration jump without
    # inventing an unobserved pre-boundary velocity.
    system = np.eye(horizon, dtype=np.float64)
    rhs = np.array(raw, copy=True)
    for order, weight in ((2, acceleration_weight), (3, jerk_weight)):
        if weight == 0.0 or horizon + 1 <= order:
            continue
        full_difference = np.diff(np.eye(horizon + 1), n=order, axis=0)
        variable = full_difference[:, 1:]
        fixed = full_difference[:, :1] * boundary[None, :]
        system += weight * (variable.T @ variable)
        rhs -= weight * (variable.T @ fixed)

    # The final action is an equality constraint. Eliminate it from the linear
    # system instead of correcting it after smoothing, so the interior optimum
    # is solved with the exact endpoint already present.
    endpoint = raw[-1]
    free_system = system[:-1, :-1]
    free_rhs = rhs[:-1] - system[:-1, -1, None] * endpoint[None, :]
    result = np.empty_like(raw)
    result[:-1] = np.linalg.solve(free_system, free_rhs)
    result[-1] = endpoint
    return result


def _limit_vector_deviation(
    raw: np.ndarray, smoothed: np.ndarray, *, max_deviation: float
) -> np.ndarray:
    """Limit every sample by globally scaling the smoothing correction."""

    if max_deviation == 0.0:
        return np.array(raw, copy=True)
    correction = np.asarray(smoothed, dtype=np.float64) - np.asarray(
        raw, dtype=np.float64
    )
    peak = float(np.linalg.norm(correction, axis=1).max(initial=0.0))
    if peak > max_deviation:
        correction *= max_deviation / peak
    result = np.asarray(raw, dtype=np.float64) + correction
    result[-1] = raw[-1]
    return result


def minimum_jerk_se3_action_chunk(
    actions: list[dict[str, Any]],
    *,
    reference: np.ndarray,
    acceleration_weight: float = 0.25,
    jerk_weight: float = 1.0,
    position_max_deviation_m: float = 0.002,
    rotation_max_deviation_deg: float = 0.5,
) -> list[dict[str, Any]]:
    """Smooth a complete SE(3) chunk while preserving its final pose exactly.

    Translation and rotation-tangent trajectories minimize acceleration and
    jerk subject to a soft fidelity term. A globally scaled correction enforces
    the configured maximum deviation without adding pointwise clipping kinks.
    Gripper commands and the caller-owned input dictionaries are unchanged.
    """

    config = SE3SmoothingConfig(
        mode=SE3_SMOOTHING_MODE_MIN_JERK,
        acceleration_weight=float(acceleration_weight),
        jerk_weight=float(jerk_weight),
        position_max_deviation_m=float(position_max_deviation_m),
        rotation_max_deviation_deg=float(rotation_max_deviation_deg),
    )
    if not actions:
        raise ValueError("cannot smooth an empty action chunk")
    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.shape != (ACTION_DIM,) or not np.isfinite(reference_array).all():
        raise ValueError(f"reference must be finite shape ({ACTION_DIM},)")
    if len(actions) == 1:
        return deepcopy(actions)

    filtered = deepcopy(actions)
    horizon = len(actions)
    for arm, offset in (("left", 0), ("right", 10)):
        pose_key = f"{arm}_ee_pose"
        poses = np.stack(
            [np.asarray(action[pose_key], dtype=np.float64) for action in actions],
            axis=0,
        )
        if poses.shape != (horizon, 7) or not np.isfinite(poses).all():
            raise ValueError(
                f"{pose_key} trajectory must be finite shape ({horizon}, 7)"
            )

        positions = poses[:, :3]
        smoothed_positions = _minimum_jerk_series(
            positions,
            reference=reference_array[offset : offset + 3],
            acceleration_weight=config.acceleration_weight,
            jerk_weight=config.jerk_weight,
        )
        smoothed_positions = _limit_vector_deviation(
            positions,
            smoothed_positions,
            max_deviation=config.position_max_deviation_m,
        )

        quaternion_norms = np.linalg.norm(poses[:, 3:], axis=1)
        if np.any(quaternion_norms <= 1e-8):
            raise ValueError(f"{pose_key} trajectory contains a zero quaternion")
        rotations = np.stack(
            [
                Rotation.from_quat(np.roll(pose[3:] / quaternion_norm, -1)).as_matrix()
                for pose, quaternion_norm in zip(poses, quaternion_norms, strict=True)
            ],
            axis=0,
        )
        reference_rotation = sixd_to_rotmat(reference_array[offset + 3 : offset + 9])
        tangent = np.stack(
            [
                Rotation.from_matrix(rotation @ reference_rotation.T).as_rotvec()
                for rotation in rotations
            ],
            axis=0,
        )
        smoothed_tangent = _minimum_jerk_series(
            tangent,
            reference=np.zeros(3, dtype=np.float64),
            acceleration_weight=config.acceleration_weight,
            jerk_weight=config.jerk_weight,
        )
        candidate_rotations = np.stack(
            [
                Rotation.from_rotvec(rotvec).as_matrix() @ reference_rotation
                for rotvec in smoothed_tangent
            ],
            axis=0,
        )
        rotation_corrections = np.stack(
            [
                Rotation.from_matrix(candidate @ raw.T).as_rotvec()
                for candidate, raw in zip(candidate_rotations, rotations, strict=True)
            ],
            axis=0,
        )
        peak_rotation_deviation = float(
            np.linalg.norm(rotation_corrections, axis=1).max(initial=0.0)
        )
        max_rotation_deviation = np.deg2rad(config.rotation_max_deviation_deg)
        if max_rotation_deviation == 0.0:
            rotation_scale = 0.0
        elif peak_rotation_deviation > max_rotation_deviation:
            rotation_scale = max_rotation_deviation / peak_rotation_deviation
        else:
            rotation_scale = 1.0
        smoothed_rotations = np.stack(
            [
                Rotation.from_rotvec(rotation_scale * correction).as_matrix() @ raw
                for correction, raw in zip(rotation_corrections, rotations, strict=True)
            ],
            axis=0,
        )
        # Copy the model's terminal pose bit-for-bit. This is stronger than a
        # numerical equality check and makes the controller endpoint invariant.
        smoothed_positions[-1] = positions[-1]
        smoothed_rotations[-1] = rotations[-1]
        for index in range(horizon - 1):
            filtered[index][pose_key] = np.concatenate(
                (
                    smoothed_positions[index],
                    _rotation_matrix_to_wxyz(smoothed_rotations[index]),
                )
            ).astype(np.float32)
        filtered[-1][pose_key] = np.array(actions[-1][pose_key], copy=True)

    return filtered


def smooth_se3_action_chunk(
    actions: list[dict[str, Any]],
    *,
    reference: np.ndarray,
    neighbor_weight: float = 0.25,
    position_max_deviation_m: float | None = None,
    rotation_max_deviation_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Suppress one-step EE jitter without changing the action horizon.

    Position uses the centered FIR kernel ``[w, 1-2w, w]``. Rotation applies
    the same neighbor correction in the current rotation's tangent space. The
    measured current EE pose is the left boundary. A slowly varying endpoint
    correction preserves the model's final pose without restoring a last-step
    discontinuity. Gripper commands are copied bit-for-bit and never filtered.
    """

    bounded = (
        position_max_deviation_m is not None or rotation_max_deviation_deg is not None
    )
    if bounded and (
        position_max_deviation_m is None or rotation_max_deviation_deg is None
    ):
        raise ValueError(
            "bounded FIR requires both position and rotation maximum deviations"
        )
    config = SE3SmoothingConfig(
        mode=(SE3_SMOOTHING_MODE_BOUNDED_FIR if bounded else SE3_SMOOTHING_MODE_FIR),
        neighbor_weight=float(neighbor_weight),
        position_max_deviation_m=(
            float(position_max_deviation_m)
            if position_max_deviation_m is not None
            else 0.002
        ),
        rotation_max_deviation_deg=(
            float(rotation_max_deviation_deg)
            if rotation_max_deviation_deg is not None
            else 0.5
        ),
    )
    if not actions:
        raise ValueError("cannot smooth an empty action chunk")
    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.shape != (ACTION_DIM,) or not np.isfinite(reference_array).all():
        raise ValueError(f"reference must be finite shape ({ACTION_DIM},)")
    if config.neighbor_weight == 0.0 or len(actions) == 1:
        return deepcopy(actions)

    filtered = deepcopy(actions)
    horizon = len(actions)
    progress = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    weight = config.neighbor_weight
    center_weight = config.center_weight

    for arm, offset in (("left", 0), ("right", 10)):
        pose_key = f"{arm}_ee_pose"
        poses = np.stack(
            [np.asarray(action[pose_key], dtype=np.float64) for action in actions],
            axis=0,
        )
        if poses.shape != (horizon, 7) or not np.isfinite(poses).all():
            raise ValueError(
                f"{pose_key} trajectory must be finite shape ({horizon}, 7)"
            )

        positions = poses[:, :3]
        previous_positions = np.concatenate(
            (reference_array[offset : offset + 3][None], positions[:-1]), axis=0
        )
        next_positions = np.concatenate((positions[1:], positions[-1:]), axis=0)
        smoothed_positions = (
            weight * previous_positions
            + center_weight * positions
            + weight * next_positions
        )
        # Distribute the endpoint correction over the complete horizon. A
        # linear ramp is low-frequency, unlike overwriting only the last point.
        position_endpoint_error = positions[-1] - smoothed_positions[-1]
        smoothed_positions += progress[:, None] * position_endpoint_error[None]
        if bounded:
            correction = smoothed_positions - positions
            correction_norm = np.linalg.norm(correction, axis=1)
            correction_scale = np.minimum(
                1.0,
                config.position_max_deviation_m / np.maximum(correction_norm, 1e-12),
            )
            smoothed_positions = positions + correction_scale[:, None] * correction
            # Preserve both master boundary targets exactly. Only interior
            # controller targets receive the bounded high-frequency correction.
            smoothed_positions[0] = positions[0]
        # Preserve the terminal target bit-for-bit for both FIR variants.
        smoothed_positions[-1] = positions[-1]

        quaternion_norms = np.linalg.norm(poses[:, 3:], axis=1)
        if np.any(quaternion_norms <= 1e-8):
            raise ValueError(f"{pose_key} trajectory contains a zero quaternion")
        rotations = [
            Rotation.from_quat(np.roll(pose[3:] / quaternion_norm, -1)).as_matrix()
            for pose, quaternion_norm in zip(poses, quaternion_norms, strict=True)
        ]
        reference_rotation = sixd_to_rotmat(reference_array[offset + 3 : offset + 9])
        smoothed_rotations: list[np.ndarray] = []
        for index, current_rotation in enumerate(rotations):
            previous_rotation = (
                reference_rotation if index == 0 else rotations[index - 1]
            )
            next_rotation = (
                rotations[index + 1] if index + 1 < horizon else rotations[-1]
            )
            previous_delta = Rotation.from_matrix(
                previous_rotation @ current_rotation.T
            ).as_rotvec()
            next_delta = Rotation.from_matrix(
                next_rotation @ current_rotation.T
            ).as_rotvec()
            correction = Rotation.from_rotvec(
                weight * (previous_delta + next_delta)
            ).as_matrix()
            smoothed_rotations.append(correction @ current_rotation)

        rotation_endpoint_error = Rotation.from_matrix(
            rotations[-1] @ smoothed_rotations[-1].T
        ).as_rotvec()
        for index in range(horizon):
            endpoint_correction = Rotation.from_rotvec(
                progress[index] * rotation_endpoint_error
            ).as_matrix()
            smoothed_rotation = endpoint_correction @ smoothed_rotations[index]
            if bounded:
                rotation_correction = Rotation.from_matrix(
                    smoothed_rotation @ rotations[index].T
                ).as_rotvec()
                correction_norm = float(np.linalg.norm(rotation_correction))
                max_rotation_deviation = np.deg2rad(config.rotation_max_deviation_deg)
                if correction_norm > max_rotation_deviation:
                    rotation_correction *= max_rotation_deviation / correction_norm
                    smoothed_rotation = (
                        Rotation.from_rotvec(rotation_correction).as_matrix()
                        @ rotations[index]
                    )
                if index == 0:
                    smoothed_rotation = rotations[index]
            if index + 1 == horizon:
                filtered[index][pose_key] = np.array(
                    actions[index][pose_key], copy=True
                )
                continue
            filtered[index][pose_key] = np.concatenate(
                (
                    smoothed_positions[index],
                    _rotation_matrix_to_wxyz(smoothed_rotation),
                )
            ).astype(np.float32)

    return filtered


def normalize_inference_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "autoregressive": INFERENCE_MODE_AR,
        "single": INFERENCE_MODE_SINGLE_FRAME,
        "singleframe": INFERENCE_MODE_SINGLE_FRAME,
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"unsupported VAP inference mode {value!r}; "
            f"expected one of {sorted(SUPPORTED_INFERENCE_MODES)}"
        )
    return mode


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def action_chunk_seed(base_seed: int, episode_id: str, window_index: int) -> int:
    """Return a stable flow-noise seed for one model action chunk.

    Python's built-in ``hash`` is intentionally process-randomized, while
    ``base_seed + window_index`` repeats the same noise schedule after every
    episode reset. SHA-256 gives each ``(noise identity, window)`` an
    independent, reproducible 63-bit seed without touching the environment or
    global RNG state. ``episode_id`` is retained as the compatibility name for
    that identity.
    """

    episode = str(episode_id).strip()
    if not episode:
        raise ValueError("episode_id must be non-empty when deriving a chunk seed")
    if isinstance(window_index, bool) or not isinstance(window_index, int):
        raise TypeError("window_index must be an integer when deriving a chunk seed")
    if window_index < 0:
        raise ValueError("window_index must be non-negative when deriving a chunk seed")
    key = f"vap-action|{int(base_seed)}|{episode}|{window_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _log_action_diagnostics(
    *,
    episode_id: str,
    window_index: int,
    inference_mode: str,
    instruction: str,
    current_reference: np.ndarray,
    action_reference: np.ndarray,
    physical: np.ndarray,
    actions: list[dict[str, Any]],
) -> None:
    """Emit one compact train/inference-contract line per action chunk."""

    if os.environ.get("VAP_LOG_ACTION_DIAGNOSTICS", "0") != "1":
        return
    fields = [
        f"episode={episode_id}",
        f"window={window_index}",
        f"mode={inference_mode}",
        f"instruction={instruction!r}",
    ]
    for arm, offset in (("left", 0), ("right", 10)):
        position_norm = np.linalg.norm(physical[:, offset : offset + 3], axis=1)
        rotation = physical[:, offset + 3 : offset + 9]
        row1 = rotation[:, :3]
        row2 = rotation[:, 3:]
        row1_norm = np.linalg.norm(row1, axis=1)
        row2_norm = np.linalg.norm(row2, axis=1)
        row_dot = np.sum(row1 * row2, axis=1)
        grip = physical[:, offset + 9]
        first_position = np.asarray(actions[0][f"{arm}_ee_pose"][:3])
        first_jump = float(
            np.linalg.norm(first_position - current_reference[offset : offset + 3])
        )
        target_anchor_gap = float(
            np.linalg.norm(
                action_reference[offset : offset + 3]
                - current_reference[offset : offset + 3]
            )
        )
        fields.extend(
            (
                f"{arm}_first_jump={first_jump:.6f}",
                f"{arm}_target_anchor_gap={target_anchor_gap:.6f}",
                f"{arm}_delta_norm=[{position_norm.min():.6f},"
                f"{position_norm.max():.6f}]",
                f"{arm}_rot_row_norm=[{row1_norm.min():.6f},"
                f"{row1_norm.max():.6f};{row2_norm.min():.6f},"
                f"{row2_norm.max():.6f}]",
                f"{arm}_rot_abs_dot_max={np.abs(row_dot).max():.6f}",
                f"{arm}_grip=[{grip.min():.6f},{grip.max():.6f}]",
            )
        )
    print("[VAP_ACTION_DIAGNOSTICS] " + " ".join(fields), flush=True)


def _trace_action_chunk(
    *,
    episode_id: str,
    window_index: int,
    inference_mode: str,
    evaluation_profile: str,
    seed: int,
    instruction: str,
    current_reference: np.ndarray,
    action_reference: np.ndarray,
    normalized_raw: np.ndarray,
    normalized_issued: np.ndarray,
    physical_relative: np.ndarray,
    raw_actions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> None:
    """Persist one behavior-neutral action trace when explicitly requested."""

    trace_root = os.environ.get("VAP_ACTION_TRACE_DIR", "").strip()
    if not trace_root:
        return
    decoded_absolute_raw = np.stack(
        [build_agent_pos_from_state(action) for action in raw_actions], axis=0
    ).astype(np.float32, copy=False)
    decoded_absolute = np.stack(
        [build_agent_pos_from_state(action) for action in actions], axis=0
    ).astype(np.float32, copy=False)
    trace_dir = Path(trace_root).expanduser()
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{episode_id}_window_{window_index:04d}.npz"
    np.savez_compressed(
        trace_path,
        episode_id=np.asarray(episode_id),
        window_index=np.asarray(window_index, dtype=np.int64),
        inference_mode=np.asarray(inference_mode),
        evaluation_profile=np.asarray(evaluation_profile),
        seed=np.asarray(seed, dtype=np.int64),
        instruction=np.asarray(instruction),
        current_reference=np.asarray(current_reference, dtype=np.float32),
        action_reference=np.asarray(action_reference, dtype=np.float32),
        normalized_raw=np.asarray(normalized_raw, dtype=np.float32),
        normalized_issued=np.asarray(normalized_issued, dtype=np.float32),
        physical_relative=np.asarray(physical_relative, dtype=np.float32),
        decoded_absolute_raw=decoded_absolute_raw,
        decoded_absolute=decoded_absolute,
    )
    print(f"[VAP_ACTION_TRACE] path={trace_path}", flush=True)


class VAPPolicyState:
    """Own one websocket client's deterministic VAP rollout state.

    State is committed only after a complete model call and action conversion.
    A repeated ``(episode_id, window_index)`` request returns the cached result,
    which makes websocket retries idempotent.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        height: int = 192,
        width: int = 224,
        base_seed: int = 0,
        fixed_action_chunk_seed: int | None = None,
        inference_mode: str = INFERENCE_MODE_AR,
        ar_prefix_action_source: str = AR_PREFIX_ACTION_SOURCE_ISSUED,
        image_rgb_gain: tuple[float, float, float] | None = None,
        evaluation_profile: str = LEGACY_PROFILE,
        gripper_hysteresis: bool = False,
        gripper_close_threshold: float = 0.25,
        gripper_open_threshold: float = 0.75,
        gripper_open_confirm_steps: int = 4,
        se3_smoothing: bool = False,
        se3_smoothing_mode: str = SE3_SMOOTHING_MODE_FIR,
        se3_smoothing_neighbor_weight: float = 0.25,
        se3_smoothing_acceleration_weight: float = 0.25,
        se3_smoothing_jerk_weight: float = 1.0,
        se3_smoothing_position_max_deviation_m: float = 0.002,
        se3_smoothing_rotation_max_deviation_deg: float = 0.5,
        allow_instruction_refresh: bool = False,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.runtime = runtime
        self.evaluation_profile = resolve_evaluation_profile(evaluation_profile)
        self.action_steps = int(
            getattr(
                runtime, "action_steps_per_chunk", self.evaluation_profile.action_steps
            )
        )
        self.video_frames = int(
            getattr(
                runtime, "video_frames_per_chunk", self.evaluation_profile.video_frames
            )
        )
        expected_horizon = (
            self.evaluation_profile.action_steps,
            self.evaluation_profile.video_frames,
        )
        if (self.action_steps, self.video_frames) != expected_horizon:
            raise ValueError(
                f"profile {self.evaluation_profile.name!r} requires "
                f"action/video horizon {expected_horizon}, runtime has "
                f"{(self.action_steps, self.video_frames)}"
            )
        self.height = int(height)
        self.width = int(width)
        self.base_seed = int(base_seed)
        if fixed_action_chunk_seed is not None:
            fixed_action_chunk_seed = int(fixed_action_chunk_seed)
            if not 0 <= fixed_action_chunk_seed <= 0x7FFFFFFFFFFFFFFF:
                raise ValueError("fixed_action_chunk_seed must be in [0, 2**63 - 1]")
        self.fixed_action_chunk_seed = fixed_action_chunk_seed
        self.inference_mode = normalize_inference_mode(inference_mode)
        self.ar_prefix_action_source = normalize_ar_prefix_action_source(
            ar_prefix_action_source
        )
        self.image_rgb_gain = image_rgb_gain
        self.gripper_hysteresis = (
            GripperHysteresisConfig(
                close_threshold=float(gripper_close_threshold),
                open_threshold=float(gripper_open_threshold),
                open_confirm_steps=int(gripper_open_confirm_steps),
            )
            if gripper_hysteresis
            else None
        )
        self.se3_smoothing = (
            SE3SmoothingConfig(
                mode=se3_smoothing_mode,
                neighbor_weight=float(se3_smoothing_neighbor_weight),
                acceleration_weight=float(se3_smoothing_acceleration_weight),
                jerk_weight=float(se3_smoothing_jerk_weight),
                position_max_deviation_m=float(se3_smoothing_position_max_deviation_m),
                rotation_max_deviation_deg=float(
                    se3_smoothing_rotation_max_deviation_deg
                ),
            )
            if se3_smoothing
            else None
        )
        self.allow_instruction_refresh = bool(allow_instruction_refresh)
        if (
            self.ar_prefix_action_source != AR_PREFIX_ACTION_SOURCE_ISSUED
            and self.inference_mode != INFERENCE_MODE_AR
        ):
            raise ValueError(
                "non-issued AR prefix action sources require AR inference mode"
            )
        if self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_PRE_FIR:
            if (
                self.se3_smoothing is None
                or self.se3_smoothing.mode != SE3_SMOOTHING_MODE_FIR
            ):
                raise ValueError(
                    "AR prefix action source 'pre_fir' requires FIR smoothing"
                )
            if self.gripper_hysteresis is not None:
                raise ValueError(
                    "AR prefix action source 'pre_fir' requires gripper "
                    "hysteresis to be disabled"
                )
        if (
            self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_RAW_MODEL
            and self.evaluation_profile.prefix_source != "issued_master"
        ):
            raise ValueError(
                "AR prefix action source 'raw_model' requires an issued-master "
                "evaluation profile"
            )
        self.reset()

    def reset(self) -> None:
        self.episode_id: str | None = None
        self.expected_window = 0
        self.anchor_views: np.ndarray | None = None
        # Episode-local control-step coordinate of ``anchor_views``.  An AR
        # continuation predicts global A{k} from the preceding local training
        # window ``I{k-1}, A{k-1}, V{k-1} -> A{k}``, so its immutable episode
        # Memory must use the age of I{k-1}, not the start of A{k}.
        self.anchor_age_steps: int | None = None
        # Encoded multi-view observation from resolved episode-local index 0.
        # Unlike ``anchor_views`` (the current AR window's I), this Anchor Bank
        # is immutable for the complete episode and is released on reset.
        self.first_frame_memory: Any | None = None
        from wall_x.model.vap.history import HistoryObservationBuffer

        self.memory_history = HistoryObservationBuffer(
            getattr(self.runtime, "history_offsets", ())
        )
        self.previous_action_reference: np.ndarray | None = None
        self.previous_target_boundary: np.ndarray | None = None
        self.previous_prefix_actions: list[dict[str, np.ndarray]] | None = None
        # Exact normalized LFU master proposal used by the model, after the
        # serving range clamp but before coordinate conversion, SO(3)
        # projection, smoothing, IK, or measured-state reconstruction.
        self.previous_model_prefix: np.ndarray | None = None
        self.text_context: Any = None
        self.instruction: str | None = None
        self._last_request: tuple[str, int] | None = None
        self._last_actions: list[dict[str, Any]] | None = None
        self.gripper_hysteresis_state: GripperHysteresisState | None = None

    def _postprocess_grippers(
        self,
        actions: list[dict[str, Any]],
        *,
        reference: np.ndarray,
    ) -> tuple[list[dict[str, Any]], GripperHysteresisState | None]:
        """Apply inference-only hysteresis without mutating committed state."""

        config = self.gripper_hysteresis
        if config is None:
            return actions, None

        state = self.gripper_hysteresis_state
        if state is None:
            is_open = [bool(reference[offset + 9] >= 0.5) for offset in (0, 10)]
            open_streak = [0, 0]
        else:
            is_open = list(state.is_open)
            open_streak = list(state.open_streak)

        filtered = deepcopy(actions)
        for action in filtered:
            for arm_index, arm in enumerate(("left", "right")):
                key = f"{arm}_ee_joint_state"
                command = float(np.asarray(action[key]).reshape(-1)[0])
                if is_open[arm_index]:
                    if command <= config.close_threshold:
                        is_open[arm_index] = False
                    open_streak[arm_index] = 0
                elif command >= config.open_threshold:
                    open_streak[arm_index] += 1
                    if open_streak[arm_index] >= config.open_confirm_steps:
                        is_open[arm_index] = True
                        open_streak[arm_index] = 0
                else:
                    open_streak[arm_index] = 0
                action[key] = np.asarray(
                    [1.0 if is_open[arm_index] else 0.0], dtype=np.float32
                )

        return filtered, GripperHysteresisState(
            is_open=(is_open[0], is_open[1]),
            open_streak=(open_streak[0], open_streak[1]),
        )

    def _metadata(
        self,
        observation: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str, str, int, str]:
        metadata = observation.get("_vap")
        if not isinstance(metadata, Mapping):
            raise TypeError("observation['_vap'] must be a mapping")
        protocol_version = metadata.get("protocol_version")
        expected_protocol = self.evaluation_profile.protocol_version
        if protocol_version != expected_protocol:
            raise ValueError(
                "VAP evaluation protocol mismatch: "
                f"client={protocol_version!r}, profile "
                f"{self.evaluation_profile.name!r} requires {expected_protocol}"
            )
        if expected_protocol >= 3:
            expected_fields = {
                "evaluation_profile": self.evaluation_profile.name,
                "action_steps": self.action_steps,
                "video_frames": self.video_frames,
                "image_height": self.height,
                "image_width": self.width,
            }
            for key, expected_value in expected_fields.items():
                actual_value = metadata.get(key)
                if actual_value != expected_value:
                    raise ValueError(
                        f"_vap.{key}={actual_value!r} does not match server "
                        f"profile value {expected_value!r}"
                    )
        episode_id = str(metadata.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("_vap.episode_id must be non-empty")
        model_noise_id = str(metadata.get("model_noise_id", episode_id)).strip()
        if not model_noise_id:
            raise ValueError("_vap.model_noise_id must be non-empty")
        window_index = metadata.get("window_index")
        if isinstance(window_index, bool) or not isinstance(window_index, int):
            raise TypeError("_vap.window_index must be an integer")
        if window_index < 0:
            raise ValueError("_vap.window_index must be non-negative")
        inference_mode = normalize_inference_mode(
            metadata.get("inference_mode", INFERENCE_MODE_AR)
        )
        return metadata, episode_id, model_noise_id, window_index, inference_mode

    @staticmethod
    def _instruction(observation: Mapping[str, Any]) -> str:
        instruction = str(observation.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("observation instruction must be non-empty")
        return instruction

    def infer(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        metadata, episode_id, model_noise_id, window_index, request_mode = (
            self._metadata(observation)
        )
        if request_mode != self.inference_mode:
            raise ValueError(
                f"VAP inference mode mismatch: client={request_mode!r} "
                f"server={self.inference_mode!r}"
            )
        request_key = (episode_id, window_index)
        if request_key == self._last_request:
            if self._last_actions is None:
                raise RuntimeError("cached VAP request has no cached actions")
            return deepcopy(self._last_actions)

        if self.episode_id is None:
            if window_index != 0:
                raise ValueError("a fresh VAP episode must start at window_index=0")
        elif episode_id != self.episode_id:
            raise ValueError("episode_id changed without calling reset")
        if window_index != self.expected_window:
            raise ValueError(
                f"received window_index={window_index}; expected {self.expected_window}"
            )

        current_views = extract_resized_views(
            observation,
            height=self.height,
            width=self.width,
            rgb_gain=self.image_rgb_gain,
        )
        memory_enabled = bool(
            getattr(self.runtime, "first_frame_memory_enabled", False)
        )
        if memory_enabled:
            first_frame_memory = self.first_frame_memory
            current_history_memory = None
            pending_memory_history = None
            if getattr(self.runtime, "history_offsets", ()):
                # Protocol window_index counts fully executed fixed-size action
                # chunks.  Encode each observed I once and stage it in a
                # shallow-cloned buffer so a failed model call cannot mutate
                # the committed episode state or duplicate an observation on
                # retry.  Latent tensors themselves remain shared/read-only.
                step = window_index * self.action_steps
                encoded_current = self.runtime.encode_first_frame_memory(
                    current_views[:, None]
                )
                pending_memory_history = copy(self.memory_history)
                pending_memory_history.frames = dict(self.memory_history.frames)
                pending_memory_history.append(step, encoded_current)
                anchors, valid, ages = pending_memory_history.snapshot(step)
                current_history_memory = self.runtime.assemble_history_memory(
                    anchors, valid, ages
                )
                if (
                    window_index == 0
                    or self.inference_mode == INFERENCE_MODE_SINGLE_FRAME
                ):
                    first_frame_memory = current_history_memory
            if first_frame_memory is None:
                if window_index != 0:
                    raise RuntimeError(
                        "first-frame Memory is missing after episode start"
                    )
                first_frame_memory = self.runtime.encode_first_frame_memory(
                    current_views[:, None]
                )
        else:
            if self.first_frame_memory is not None:
                raise RuntimeError(
                    "disabled first-frame Memory retained stale episode state"
                )
            first_frame_memory = None
            current_history_memory = None
            pending_memory_history = None
        # RoboDojo observes/controls RFU. Legacy policies were trained in that
        # frame, while EE-v3 was exported in LFU. Keep both representations so
        # model conditioning and controller-bound postprocessing cannot mix.
        reference_rfu = build_agent_pos(observation)
        reference = transform_absolute20_coordinate_frame(
            reference_rfu,
            source="rfu",
            target=self.evaluation_profile.coordinate_frame,
        )
        use_state_conditioning = bool(
            getattr(self.runtime, "state_conditioning_enabled", False)
        )
        # This seed controls only model flow-noise. RoboDojo's eval seed and
        # layout are untouched. The standard path derives independent,
        # reproducible noise for every task/layout/window. The fixed path is an
        # explicit ablation that intentionally reuses one noise tensor for all
        # action chunks; it must never become the benchmark default.
        seed = (
            self.fixed_action_chunk_seed
            if self.fixed_action_chunk_seed is not None
            else action_chunk_seed(self.base_seed, model_noise_id, window_index)
        )

        if window_index == 0 or self.inference_mode == INFERENCE_MODE_SINGLE_FRAME:
            if "executed_video" in metadata or "executed_state" in metadata:
                raise ValueError(
                    f"{self.inference_mode} inference must not include executed history"
                )
            instruction = self._instruction(observation)
            if self.instruction is None:
                text_context = self.runtime.encode_text(instruction)
            else:
                if instruction != self.instruction:
                    raise ValueError("instruction changed within one VAP episode")
                text_context = self.text_context
            sample_kwargs = {"text_context": text_context, "seed": seed}
            if memory_enabled:
                sample_kwargs["first_frame_memory"] = first_frame_memory
                if (
                    not getattr(self.runtime, "history_offsets", ())
                    and getattr(
                        self.runtime,
                        "first_frame_memory_position_mode",
                        "window_origin",
                    )
                    == "relative_age"
                ):
                    sample_kwargs["first_frame_memory_age_steps"] = (
                        window_index * self.action_steps
                    )
            if use_state_conditioning:
                # In single-frame/A0 inference the current measured absolute
                # link6 state is exactly the training window's S0 anchor.
                sample_kwargs["physical_state"] = reference
            normalized = self.runtime.sample_first_action(
                current_views[:, None], **sample_kwargs
            )
            action_reference = reference
        else:
            prefix_state_missing = (
                self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_RAW_MODEL
                and self.previous_model_prefix is None
            ) or (
                self.ar_prefix_action_source
                in {
                    AR_PREFIX_ACTION_SOURCE_ISSUED,
                    AR_PREFIX_ACTION_SOURCE_PRE_FIR,
                    AR_PREFIX_ACTION_SOURCE_POSEGRIP,
                }
                and self.previous_prefix_actions is None
            )
            if (
                self.anchor_views is None
                or self.anchor_age_steps is None
                or self.previous_action_reference is None
                or self.previous_target_boundary is None
                or (
                    (
                        self.evaluation_profile.prefix_source == "issued_master"
                        or self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_POSEGRIP
                    )
                    and self.ar_prefix_action_source
                    != AR_PREFIX_ACTION_SOURCE_MEASURED_STATE
                    and prefix_state_missing
                )
            ):
                raise RuntimeError("VAP continuation state is incomplete")
            # The video and S1 always come from the measured rollout. The clean
            # action prefix is profile-specific: legacy data recorded follower
            # states, whereas EE-v3 records the issued master command.
            instruction = self._instruction(observation)
            history = extract_executed_video(
                metadata,
                height=self.height,
                width=self.width,
                video_frames=self.video_frames,
                rgb_gain=self.image_rgb_gain,
            )
            if not np.array_equal(history[:, -1], current_views):
                raise ValueError(
                    "the final executed_video frame must equal the current observation"
                )
            continuous_video = build_anchor_and_history(
                self.anchor_views, history, video_frames=self.video_frames
            )
            executed_absolute_rfu = extract_executed_agent_pos(
                metadata, action_steps=self.action_steps
            )
            if not np.allclose(
                executed_absolute_rfu[-1], reference_rfu, atol=1e-6, rtol=0.0
            ):
                raise ValueError(
                    "the final executed_state must equal the current observation state"
                )
            use_measured_prefix = (
                self.ar_prefix_action_source in {
                    AR_PREFIX_ACTION_SOURCE_MEASURED_STATE,
                    AR_PREFIX_ACTION_SOURCE_POSEGRIP,
                }
                or self.evaluation_profile.prefix_source != "issued_master"
            )
            if not use_measured_prefix:
                if self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_RAW_MODEL:
                    # Strict training-domain prefix: reuse the exact normalized
                    # LFU master proposal cached before every controller-only
                    # transform. This deliberately bypasses RFU round-trips,
                    # SO(3) projection, smoothing, IK, and follower feedback.
                    executed_prefix = np.array(
                        self.previous_model_prefix, dtype=np.float32, copy=True
                    )
                else:
                    # EE-v3 clean action is the recorded master command, not the
                    # follower state reached after IK/contact. The maintained
                    # path prefixes final issued targets. ``pre_fir`` is the
                    # legacy control-domain ablation that prefixes decoded RFU
                    # targets before FIR, including the SO(3) projection and
                    # coordinate round-trip.
                    issued_absolute_rfu = np.stack(
                        [
                            build_agent_pos_from_state(action)
                            for action in self.previous_prefix_actions
                        ],
                        axis=0,
                    )
                    issued_absolute_model = transform_absolute20_coordinate_frame(
                        issued_absolute_rfu,
                        source="rfu",
                        target=self.evaluation_profile.coordinate_frame,
                    )
                    executed_relative = absolute20_to_relative20(
                        self.previous_action_reference, issued_absolute_model
                    )
                    executed_prefix = _to_numpy(
                        self.runtime.normalize_action_prefix(executed_relative)
                    ).astype(np.float32, copy=True)
            else:
                if self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_POSEGRIP:
                    # PoseGrip: row k pairs the achieved post-action pose with
                    # the command actually issued at that same control step.
                    # Physical gripper opening under contact is NOT closing
                    # intent. Merge in physical RFU space BEFORE conversion,
                    # rebasing and normalization. Never mutate observation data.
                    if len(self.previous_prefix_actions) != self.action_steps:
                        raise RuntimeError("PoseGrip issued-action history length mismatch")
                    issued = np.stack(
                        [build_agent_pos_from_state(action)
                         for action in self.previous_prefix_actions]
                    )
                    if (
                        issued.shape != (self.action_steps, ACTION_DIM)
                        or not np.isfinite(issued).all()
                    ):
                        raise RuntimeError("PoseGrip issued-action history is invalid")
                    executed_absolute_rfu = np.array(executed_absolute_rfu, copy=True)
                    executed_absolute_rfu[:, [9, 19]] = issued[:, [9, 19]]
                # ``measured_state`` reconstructs all 32 clean A0 rows from
                # the post-action link6/gripper states actually reached in the
                # simulator. It therefore pairs the exact achieved trajectory
                # with the real V0 frames, including FIR, IK and tracking
                # effects, before applying the training normalizer.
                executed_absolute = transform_absolute20_coordinate_frame(
                    executed_absolute_rfu,
                    source="rfu",
                    target=self.evaluation_profile.coordinate_frame,
                )
                executed_relative = absolute20_to_relative20(
                    self.previous_action_reference, executed_absolute
                )
                executed_prefix = _to_numpy(
                    self.runtime.normalize_action_prefix(executed_relative)
                ).astype(np.float32, copy=True)
            expected = (self.action_steps, ACTION_DIM)
            if executed_prefix.shape != expected:
                raise RuntimeError(
                    "runtime returned normalized executed-action prefix "
                    f"{executed_prefix.shape}; expected {expected}"
                )
            if not np.isfinite(executed_prefix).all():
                raise RuntimeError(
                    "normalized executed-action prefix contains non-finite values"
                )
            # Agent->VAP may refresh its high-level caption while the physical
            # AR trajectory remains continuous. Re-encode text but preserve
            # video/action prefix, episode identity, and window index.
            if instruction != self.instruction and not self.allow_instruction_refresh:
                raise ValueError("instruction changed within one VAP episode")
            text_context = (
                self.runtime.encode_text(instruction)
                if instruction != self.instruction
                else self.text_context
            )
            sample_kwargs = {"text_context": text_context, "seed": seed}
            if memory_enabled:
                sample_kwargs["first_frame_memory"] = first_frame_memory
                if (
                    not getattr(self.runtime, "history_offsets", ())
                    and getattr(
                        self.runtime,
                        "first_frame_memory_position_mode",
                        "window_origin",
                    )
                    == "relative_age"
                ):
                    # ``continuous_video`` starts at the stored I-anchor from
                    # the preceding request.  Match training's
                    # ``episode_first_memory_age_steps=query_local_idx`` for
                    # that two-chunk I/A0/V0/A1 sample.  In particular, the
                    # first continuation (global A1) still has age 0.
                    sample_kwargs["first_frame_memory_age_steps"] = (
                        self.anchor_age_steps
                    )
            if use_state_conditioning:
                # Match training's per-chunk contract for the shifted local
                # I_anchor+A0+V0->A1 window: clean A0 keeps its original S0,
                # while active A1 receives the measured terminal S1.
                sample_kwargs["physical_state"] = np.stack(
                    (self.previous_action_reference, reference), axis=0
                )
            normalized = self.runtime.sample_next_action(
                continuous_video, executed_prefix, **sample_kwargs
            )
            # The generated chunk is relative to the measured terminal follower
            # state, matching the A0/A1 rebase boundary used during training.
            action_reference = reference

        normalized_array = _to_numpy(normalized).astype(np.float32, copy=True)
        expected = (self.action_steps, ACTION_DIM)
        if normalized_array.shape != expected:
            raise RuntimeError(
                f"runtime returned normalized action {normalized_array.shape}; "
                f"expected {expected}"
            )
        # Apply the checkpoint's exact normalizer contract before inversion.
        # The training clip policy comes from the materialized checkpoint
        # config (the current A32/V16 profile requires clip=true). Independent
        # controller-side SO(3)/gripper projection happens after inversion and
        # is reflected in the measured clean AR prefix.
        issued_normalized, physical = self.runtime.prepare_action_for_execution(
            normalized_array
        )
        issued_normalized = _to_numpy(issued_normalized).astype(np.float32, copy=True)
        if issued_normalized.shape != expected:
            raise RuntimeError(
                "runtime returned issued normalized action "
                f"{issued_normalized.shape}; expected {expected}"
            )
        if not np.isfinite(issued_normalized).all():
            raise RuntimeError(
                "issued normalized VAP action contains non-finite values"
            )
        raw_model_actions = relative20_to_action_dicts(action_reference, physical)
        raw_actions = transform_action_dicts_coordinate_frame(
            raw_model_actions,
            source=self.evaluation_profile.coordinate_frame,
            target="rfu",
        )
        if len(raw_actions) != self.action_steps:
            raise RuntimeError(
                f"decoded {len(raw_actions)} actions; expected {self.action_steps}"
            )
        if self.se3_smoothing is None:
            se3_actions = raw_actions
        elif self.se3_smoothing.mode == SE3_SMOOTHING_MODE_FIR:
            se3_actions = smooth_se3_action_chunk(
                raw_actions,
                reference=reference_rfu,
                neighbor_weight=self.se3_smoothing.neighbor_weight,
            )
        elif self.se3_smoothing.mode == SE3_SMOOTHING_MODE_BOUNDED_FIR:
            se3_actions = smooth_se3_action_chunk(
                raw_actions,
                reference=reference_rfu,
                neighbor_weight=self.se3_smoothing.neighbor_weight,
                position_max_deviation_m=(self.se3_smoothing.position_max_deviation_m),
                rotation_max_deviation_deg=(
                    self.se3_smoothing.rotation_max_deviation_deg
                ),
            )
        elif self.se3_smoothing.mode == SE3_SMOOTHING_MODE_MIN_JERK:
            se3_actions = minimum_jerk_se3_action_chunk(
                raw_actions,
                reference=reference_rfu,
                acceleration_weight=self.se3_smoothing.acceleration_weight,
                jerk_weight=self.se3_smoothing.jerk_weight,
                position_max_deviation_m=(self.se3_smoothing.position_max_deviation_m),
                rotation_max_deviation_deg=(
                    self.se3_smoothing.rotation_max_deviation_deg
                ),
            )
        else:  # pragma: no cover - validated by SE3SmoothingConfig
            raise AssertionError(f"unhandled smoothing mode {self.se3_smoothing.mode}")
        actions, next_gripper_hysteresis_state = self._postprocess_grippers(
            se3_actions,
            reference=reference_rfu,
        )

        next_target_boundary = None
        if self.inference_mode == INFERENCE_MODE_AR:
            # Keep the command boundary only for diagnostics. The next AR call
            # will build its clean prefix and decode origin from measured state.
            next_target_boundary = build_agent_pos_from_state(actions[-1])
            if (
                next_target_boundary.shape != (ACTION_DIM,)
                or not np.isfinite(next_target_boundary).all()
            ):
                raise RuntimeError("issued VAP target boundary is invalid")
        _log_action_diagnostics(
            episode_id=episode_id,
            window_index=window_index,
            inference_mode=self.inference_mode,
            instruction=instruction,
            current_reference=reference_rfu,
            action_reference=reference_rfu,
            physical=physical,
            actions=actions,
        )
        _trace_action_chunk(
            episode_id=episode_id,
            window_index=window_index,
            inference_mode=self.inference_mode,
            evaluation_profile=self.evaluation_profile.name,
            seed=seed,
            instruction=instruction,
            current_reference=reference_rfu,
            action_reference=reference_rfu,
            normalized_raw=normalized_array,
            normalized_issued=issued_normalized,
            physical_relative=_to_numpy(physical),
            raw_actions=raw_actions,
            actions=actions,
        )

        # Commit only after every validation and conversion above has succeeded.
        self.episode_id = episode_id
        self.expected_window = window_index + 1
        self.anchor_views = (
            current_views if self.inference_mode == INFERENCE_MODE_AR else None
        )
        self.anchor_age_steps = (
            window_index * self.action_steps
            if self.inference_mode == INFERENCE_MODE_AR
            else None
        )
        if pending_memory_history is not None:
            self.memory_history = pending_memory_history
            self.first_frame_memory = current_history_memory
        else:
            self.first_frame_memory = first_frame_memory
        self.previous_action_reference = (
            action_reference if self.inference_mode == INFERENCE_MODE_AR else None
        )
        self.previous_target_boundary = next_target_boundary
        if self.inference_mode == INFERENCE_MODE_AR:
            if self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_MEASURED_STATE:
                self.previous_prefix_actions = None
                self.previous_model_prefix = None
            elif self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_RAW_MODEL:
                self.previous_prefix_actions = None
                self.previous_model_prefix = np.array(
                    issued_normalized, dtype=np.float32, copy=True
                )
            else:
                prefix_actions = (
                    raw_actions
                    if self.ar_prefix_action_source == AR_PREFIX_ACTION_SOURCE_PRE_FIR
                    else actions
                )
                self.previous_prefix_actions = deepcopy(prefix_actions)
                self.previous_model_prefix = None
        else:
            self.previous_prefix_actions = None
            self.previous_model_prefix = None
        self.text_context = text_context
        self.instruction = instruction
        self.gripper_hysteresis_state = next_gripper_hysteresis_state
        self._last_request = request_key
        self._last_actions = deepcopy(actions)
        return actions


__all__ = [
    "INFERENCE_MODE_AR",
    "INFERENCE_MODE_SINGLE_FRAME",
    "AR_PREFIX_ACTION_SOURCE_ISSUED",
    "AR_PREFIX_ACTION_SOURCE_MEASURED_STATE",
    "AR_PREFIX_ACTION_SOURCE_POSEGRIP",
    "AR_PREFIX_ACTION_SOURCE_PRE_FIR",
    "AR_PREFIX_ACTION_SOURCE_RAW_MODEL",
    "GripperHysteresisConfig",
    "GripperHysteresisState",
    "SE3_SMOOTHING_MODE_BOUNDED_FIR",
    "SE3_SMOOTHING_MODE_FIR",
    "SE3_SMOOTHING_MODE_MIN_JERK",
    "SE3SmoothingConfig",
    "SUPPORTED_INFERENCE_MODES",
    "SUPPORTED_SE3_SMOOTHING_MODES",
    "VAPPolicyState",
    "action_chunk_seed",
    "minimum_jerk_se3_action_chunk",
    "normalize_inference_mode",
    "normalize_se3_smoothing_mode",
    "normalize_ar_prefix_action_source",
    "smooth_se3_action_chunk",
]
