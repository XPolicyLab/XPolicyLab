"""LeRobot dataset that re-encodes EE-pose keys at load time.

Raw data: 7-d ``xyz_wxyz``. Embodiment ``pose_representation`` picks the model
encoding (HyVLA ``relative_chunk_ee_RT``):

- ``raw`` (7): unchanged
- ``abs_xyz_rot6d`` (9): absolute xyz + rot6d (first two rows of R)
- ``cam_abs_xyz_rot6d`` (9): same encoding as ``abs_xyz_rot6d``, but requires
  ``ee_in_camera_frame=true`` (chunk-t0 head-cam absolute pose). Prefer this
  over ``abs_xyz_rot6d`` + camera flag when the model targets are meant to be
  camera-frame spatial poses; train.sh / deploy enforce the camera frame.
- ``delta_xyz_rot6d`` (9, action only): chunk vs current ``state[0]`` —
  ``T_rel = T_state^{-1} @ T_action``; row 0 is usually not identity. Deploy
  composes the chunk onto the observed state. Same-named state key required.

Convert in ``_apply_action_mode`` (after pad, before norm); q99 uses converted
stats via ``_get_metadata``. Non-pose keys pass through.

``ee_in_camera_frame`` (train-config, default True): before convert, express
chunk state+action EE in head-cam at t0 (``T_cam^{-1} @ T_ee``). Extrinsics
from ``modality.json`` ``camera_extrinsics`` (see ``_resolve_camera_extrinsics``);
``camera_from_world`` inverted on load; wrong declarations raise. Fixed camera
⇒ action delta unchanged (only abs state/stats move). Math mirrors
``tools/ee_delta_action.py`` / ``tools/relative_pose.py``.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LE_ROBOT_DATA_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
    _check_force_stats_recompute_allowed,
    _dataset_meta_fingerprint,
    _episode_index_from_parquet_name,
    _force_stats_recompute,
    _is_stats_main_process,
    _load_precomputed_stats_or_raise,
    _load_stats_cache_for_config,
    _precomputed_stats_required,
    _save_stats_cache,
    _stats_cache_path,
    _wait_for_stats_cache,
)
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
from starVLA.dataloader.gr00t_lerobot.streaming_stats import (  # noqa: F401
    _DEFAULT_STATS_STREAMING_THRESHOLD,
    _STATS_MODE_ENV,
    _STATS_STREAMING_THRESHOLD_ENV,
    _STREAMING_COLLECT_BUDGET,
    _STREAMING_COLLECT_CAP,
    _STREAMING_COLLECT_FLOOR,
    _CollectBudget,
    _StreamingStatsBuilder,
    _f32_from_ordered_key,
    _f32_ordered_key,
    _lerp_float32_pair,
    _resolve_stats_mode,
)

# Raw layout stored in the LeRobot parquet for every pose key.
RAW_POSE_FORMAT = "xyz_wxyz"
RAW_POSE_DIM = 7  # xyz(3) + quat wxyz(4)

# format -> dim the model sees for that key.
POSE_FORMAT_DIMS = {
    "raw": RAW_POSE_DIM,
    "abs_xyz_rot6d": 9,  # xyz(3) + rot6d(6)
    "cam_abs_xyz_rot6d": 9,  # abs_xyz_rot6d in chunk-t0 camera frame
    "delta_xyz_rot6d": 9,  # delta_xyz(3) + rel_rot6d(6)
}
STATE_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d")
ACTION_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d", "delta_xyz_rot6d")
# Formats that encode absolute xyz+rot6d (world or camera; camera via flag).
ABS_XYZ_ROT6D_FORMATS = ("abs_xyz_rot6d", "cam_abs_xyz_rot6d")
# Formats that require ee_in_camera_frame=true.
CAMERA_FRAME_POSE_FORMATS = ("cam_abs_xyz_rot6d",)

# Converted-space stats (separate from raw meta/stats_xpolicy.json). The file
# actually written is a config-addressed sibling (see _stats_cache_path), so
# runs with different pose formats / chunk lengths / camera frames coexist in
# the same dataset directory instead of evicting each other.
EE_DELTA_STATS_FILENAME = "meta/stats_xpolicy_ee_delta.json"
# v3: per-episode chunks (not per parquet file). v4: delta vs state[0].
# v5: optional ee_in_camera_frame. v6: abs actions also chunked in cam[t0]
# (v5 used camera[t] per row). v7: action stats also carry per-horizon-row
# quantiles. Bump invalidates caches.
_STATS_VERSION = "7"

# Converted-stats implementation selector. "vectorized" (default) computes
# whole-episode tensor ops; "loop" is the original per-base-index Python
# loop, kept for A/B verification. Per-block values agree to within float64
# round-off (~1e-13, only at near-zero entries — BLAS accumulates batched
# and looped tiny matmuls in different orders), and the pooled statistics
# over a full dataset come out identical, so the cache hash deliberately
# does NOT include the implementation and the two share caches.
_STATS_IMPL_ENV = "SPARK_0_STATS_IMPL"


def _stats_impl() -> str:
    impl = os.environ.get(_STATS_IMPL_ENV, "vectorized").strip().lower()
    if impl not in ("vectorized", "loop"):
        raise ValueError(
            f"{_STATS_IMPL_ENV}={impl!r}; expected 'vectorized' or 'loop'."
        )
    return impl

# Fallback when modality.json has no camera_extrinsics block. Layouts: see
# _load_camera_extrinsics_array. Pose = T_world_from_camera [xyz, qwxyz].
DEFAULT_CAMERA_EXTRINSICS_KEY = "observation.camera_extrinsics.cam_high"
CAMERA_EXTRINSICS_NAMES = ("cam_high", "cam_head", "head_camera", "top_camera")

WORLD_FROM_CAMERA = "world_from_camera"
CAMERA_FROM_WORLD = "camera_from_world"
EXTRINSICS_SEMANTICS = (WORLD_FROM_CAMERA, CAMERA_FROM_WORLD)
# Pipeline writes world_from_camera; other modality.json values raise unless
# overridden via datasets.vla_data.camera_extrinsics_semantics.
EXPECTED_EXTRINSICS_SEMANTICS = WORLD_FROM_CAMERA
EXPECTED_EXTRINSICS_POSE_ORDER = "x_y_z_qw_qx_qy_qz"
# Train/deploy must agree (OpenCV = diag(1,-1,-1) vs OpenGL). In stats-cache key.
DEFAULT_CAMERA_AXES = "x_right_y_up_z_back"


def _check_extrinsics_pose_order(camera: str, entry: dict) -> None:
    """Reject pose_order the loader cannot read (e.g. xyzw would silently corrupt)."""
    pose_order = str(entry.get("pose_order", EXPECTED_EXTRINSICS_POSE_ORDER))
    if pose_order != EXPECTED_EXTRINSICS_POSE_ORDER:
        raise ValueError(
            f"modality.json camera_extrinsics.{camera} declares "
            f"pose_order={pose_order!r}; ee_in_camera_frame only reads "
            f"{EXPECTED_EXTRINSICS_POSE_ORDER!r} (xyz + quat wxyz)."
        )


def _resolve_extrinsics_semantics(camera: str, entry: dict, override: str | None) -> str:
    """Resolve extrinsics direction; refuse silent flips of world_from_camera.

    Non-default modality.json semantics raise unless
    ``datasets.vla_data.camera_extrinsics_semantics`` explicitly overrides
    (known-wrong label or a truly inverted column).
    """
    if override is not None:
        semantics = str(override)
        source = "datasets.vla_data.camera_extrinsics_semantics"
    else:
        semantics = str(entry.get("semantics", EXPECTED_EXTRINSICS_SEMANTICS))
        source = f"modality.json camera_extrinsics.{camera}"
    if semantics not in EXTRINSICS_SEMANTICS:
        raise ValueError(
            f"{source} declares semantics={semantics!r}; expected one of "
            f"{EXTRINSICS_SEMANTICS}."
        )
    if override is None and semantics != EXPECTED_EXTRINSICS_SEMANTICS:
        raise ValueError(
            f"{source} declares semantics={semantics!r}, but this pipeline "
            f"writes {EXPECTED_EXTRINSICS_SEMANTICS!r} extrinsics. Reading the "
            "column with the wrong direction corrupts every EE pose without "
            "raising, so this is refused. Fix the dataset annotation, or — if "
            "the column really is inverted — acknowledge it explicitly with "
            f"datasets.vla_data.camera_extrinsics_semantics: {semantics}"
        )
    return semantics


# ---------------------------------------------------------------------------
# Pose math (numpy only)
# ---------------------------------------------------------------------------
def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Batched unit-quaternion (w, x, y, z) -> rotation matrix (..., 3, 3)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two ROWS, row-major flatten."""
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def _matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """Single rotation matrix -> unit quaternion [w, x, y, z] (w >= 0).

    Shepperd branch selection; mirror of ``tools/ee_delta_action.py`` for the
    single-matrix case (this module stays dependency-free).
    """
    m = np.asarray(m, dtype=np.float64)
    q_abs_sq = np.array(
        [
            1.0 + m[0, 0] + m[1, 1] + m[2, 2],
            1.0 + m[0, 0] - m[1, 1] - m[2, 2],
            1.0 - m[0, 0] + m[1, 1] - m[2, 2],
            1.0 - m[0, 0] - m[1, 1] + m[2, 2],
        ]
    )
    q_abs = np.sqrt(np.maximum(q_abs_sq, 0.0))
    pivot = int(np.argmax(q_abs))
    candidates = (
        [q_abs[0] ** 2, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
        [m[2, 1] - m[1, 2], q_abs[1] ** 2, m[1, 0] + m[0, 1], m[0, 2] + m[2, 0]],
        [m[0, 2] - m[2, 0], m[1, 0] + m[0, 1], q_abs[2] ** 2, m[1, 2] + m[2, 1]],
        [m[1, 0] - m[0, 1], m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], q_abs[3] ** 2],
    )
    quat = np.asarray(candidates[pivot], dtype=np.float64) / (2.0 * q_abs[pivot])
    quat = quat / np.linalg.norm(quat)
    return -quat if quat[0] < 0.0 else quat


def _perturb_pose(
    pose_xyz_wxyz: np.ndarray,
    rot_noise_deg: float,
    trans_noise_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a small local SE(3) perturbation to a 7-d [xyz, quat_wxyz] pose.

    Rotation: random axis, angle ~ N(0, rot_noise_deg) clipped to 3 sigma,
    composed in the WRIST frame (``R' = R @ R_noise``) — the same way
    closed-loop tracking drift appears relative to the plan. Translation:
    ~ N(0, trans_noise_m) per axis, clipped to 3 sigma, in the base frame.
    """
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64).reshape(RAW_POSE_DIM).copy()
    if trans_noise_m > 0.0:
        pose[:3] += np.clip(
            rng.normal(0.0, trans_noise_m, size=3), -3.0 * trans_noise_m, 3.0 * trans_noise_m
        )
    if rot_noise_deg > 0.0:
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            axis /= norm
            angle = np.radians(
                float(np.clip(rng.normal(0.0, rot_noise_deg), -3.0 * rot_noise_deg, 3.0 * rot_noise_deg))
            )
            k = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            rot_noise = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
            pose[3:7] = _matrix_to_quat_wxyz(_quat_wxyz_to_matrix(pose[3:7]) @ rot_noise)
    return pose


def _check_raw_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    assert pose.ndim == 2 and pose.shape[1] == RAW_POSE_DIM, (
        f"expected raw pose (T, {RAW_POSE_DIM}) [{RAW_POSE_FORMAT}], got {pose.shape}"
    )
    return pose


def pose_to_abs_xyz_rot6d(pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    """``abs_xyz_rot6d``: (T, 7) [xyz, quat_wxyz] -> (T, 9) [xyz, rot6d]."""
    pose = _check_raw_pose(pose_xyz_wxyz)
    rot6d = _matrix_to_rot6d(_quat_wxyz_to_matrix(pose[:, 3:7]))
    return np.concatenate([pose[:, :3], rot6d], axis=1).astype(np.float32)


def pose_chunk_to_delta_xyz_rot6d(
    pose_chunk_xyz_wxyz: np.ndarray,
    reference_pose_xyz_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """``delta_xyz_rot6d``: (H, 7) -> (H, 9), RT-relative to a reference frame.

    The reference frame is the CURRENT robot ``state`` pose (``state[0]``),
    passed via ``reference_pose_xyz_wxyz`` (7-d ``[xyz, quat_wxyz]``). This
    matches the deployment codec, which composes the predicted delta chunk
    onto the current observed state. When no reference is given it falls back
    to the chunk's own frame 0 (the legacy RT-relative-to-action[0] behaviour,
    kept only for the standalone tools mirror / tests).

    So ``delta_xyz_i = R_ref^T (p_action_i - p_ref)`` and
    ``rel_rot6d_i = rot6d(R_ref^T R_action_i)``. Unlike the frame-0 variant,
    row 0 is generally NOT the identity (``action[0] != state[0]``).
    """
    chunk = _check_raw_pose(pose_chunk_xyz_wxyz)
    positions = chunk[:, :3]
    rotations = _quat_wxyz_to_matrix(chunk[:, 3:7])  # (H, 3, 3)

    if reference_pose_xyz_wxyz is None:
        ref_pos = positions[0]
        ref_rot = rotations[0]
    else:
        ref = np.asarray(reference_pose_xyz_wxyz, dtype=np.float64).reshape(RAW_POSE_DIM)
        ref_pos = ref[:3]
        ref_rot = _quat_wxyz_to_matrix(ref[3:7])

    r0_t = ref_rot.T  # R_ref^{-1}
    delta_xyz = (positions - ref_pos) @ r0_t.T  # rows: R_ref^T (p_i - p_ref)
    rel_rot = r0_t @ rotations  # broadcast over H
    return np.concatenate([delta_xyz, _matrix_to_rot6d(rel_rot)], axis=1).astype(np.float32)


def convert_pose(pose_xyz_wxyz: np.ndarray, pose_format: str) -> np.ndarray:
    """Dispatch raw (T, 7) poses to the requested format (see module docstring)."""
    if pose_format == "raw":
        return np.asarray(pose_xyz_wxyz)
    if pose_format in ABS_XYZ_ROT6D_FORMATS:
        return pose_to_abs_xyz_rot6d(pose_xyz_wxyz)
    if pose_format == "delta_xyz_rot6d":
        return pose_chunk_to_delta_xyz_rot6d(pose_xyz_wxyz)
    raise ValueError(f"Unknown pose format: {pose_format!r} (expected one of {list(POSE_FORMAT_DIMS)})")


def poses_world_to_frame(
    poses_xyz_wxyz: np.ndarray,
    frame_pose_xyz_wxyz: np.ndarray,
) -> np.ndarray:
    """Express world-frame poses in ``frame_pose``'s coordinate frame.

    Same SE(3) relative as ``tools/relative_pose.relative_pose_b_in_a``:
    ``T_frame_from_pose = T_world_from_frame^{-1} @ T_world_from_pose``.

    Args:
        poses_xyz_wxyz: (N, 7) poses in the shared world frame.
        frame_pose_xyz_wxyz: (7,) reference frame (e.g. camera at chunk t0).

    Returns:
        (N, 7) poses in the reference frame (quat sign ``w >= 0``).
    """
    poses = _check_raw_pose(poses_xyz_wxyz)
    frame = np.asarray(frame_pose_xyz_wxyz, dtype=np.float64).reshape(RAW_POSE_DIM)
    r_frame = _quat_wxyz_to_matrix(frame[3:7])
    r_frame_t = r_frame.T
    p_rel = (poses[:, :3] - frame[:3]) @ r_frame_t.T
    r_rel = r_frame_t @ _quat_wxyz_to_matrix(poses[:, 3:7])
    quats = np.stack([_matrix_to_quat_wxyz(r_rel[i]) for i in range(len(poses))], axis=0)
    return np.concatenate([p_rel, quats], axis=1)


def _matrix_to_quat_wxyz_batch(matrices: np.ndarray) -> np.ndarray:
    """Batched Shepperd: (N, 3, 3) -> (N, 4) unit quats [w, x, y, z], w >= 0.

    Elementwise mirror of ``_matrix_to_quat_wxyz``: identical float64
    expressions per row (same pivot rule — ``argmax`` keeps the first max —
    and the same candidate formulas), so batching does not change any value.
    """
    m = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    q_abs_sq = np.stack(
        [
            1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
            1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
            1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
        ],
        axis=1,
    )
    q_abs = np.sqrt(np.maximum(q_abs_sq, 0.0))
    pivot = np.argmax(q_abs, axis=1)
    candidates = np.stack(
        [
            np.stack(
                [q_abs[:, 0] ** 2, m[:, 2, 1] - m[:, 1, 2],
                 m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]],
                axis=1,
            ),
            np.stack(
                [m[:, 2, 1] - m[:, 1, 2], q_abs[:, 1] ** 2,
                 m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]],
                axis=1,
            ),
            np.stack(
                [m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1],
                 q_abs[:, 2] ** 2, m[:, 1, 2] + m[:, 2, 1]],
                axis=1,
            ),
            np.stack(
                [m[:, 1, 0] - m[:, 0, 1], m[:, 2, 0] + m[:, 0, 2],
                 m[:, 2, 1] + m[:, 1, 2], q_abs[:, 3] ** 2],
                axis=1,
            ),
        ],
        axis=1,
    )  # (N, 4 pivot candidates, 4 quat components)
    rows = np.arange(len(m))
    quat = candidates[rows, pivot] / (2.0 * q_abs[rows, pivot])[:, None]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return np.where(quat[:, :1] < 0.0, -quat, quat)


def _poses_world_to_frame_rows(
    poses_xyz_wxyz: np.ndarray,
    frame_poses_xyz_wxyz: np.ndarray,
) -> np.ndarray:
    """Per-row ``poses_world_to_frame``: (N, 7) poses, each expressed in its
    own reference frame from ``frame_poses_xyz_wxyz`` (N, 7).

    Same expressions as the single-frame version, with the frame varying per
    row: ``p_rel = (p - f) @ R_frame`` (row-vector form of
    ``R_frame^T (p - f)``) and ``R_rel = R_frame^T R_pose``.
    """
    poses = _check_raw_pose(poses_xyz_wxyz)
    frames = _check_raw_pose(frame_poses_xyz_wxyz)
    if len(frames) != len(poses):
        raise ValueError(
            f"per-row frame transform needs one frame per pose, got "
            f"{len(frames)} frames for {len(poses)} poses"
        )
    r_frame = _quat_wxyz_to_matrix(frames[:, 3:7])  # (N, 3, 3)
    r_frame_t = np.swapaxes(r_frame, -1, -2)
    p_rel = ((poses[:, :3] - frames[:, :3])[:, None, :] @ r_frame)[:, 0, :]
    r_rel = r_frame_t @ _quat_wxyz_to_matrix(poses[:, 3:7])
    quats = _matrix_to_quat_wxyz_batch(r_rel)
    return np.concatenate([p_rel, quats], axis=1)


def _pose_chunks_to_delta_xyz_rot6d(
    pose_chunks_xyz_wxyz: np.ndarray,
    reference_poses_xyz_wxyz: np.ndarray,
) -> np.ndarray:
    """Batched ``pose_chunk_to_delta_xyz_rot6d``: (L, H, 7) chunks, each row
    relative to its chunk's own reference from ``reference_poses_xyz_wxyz``
    (L, 7) -> (L, H, 9) float32.

    Per (l, h): ``delta_xyz = (p - p_ref) @ R_ref`` (row-vector form of
    ``R_ref^T (p - p_ref)``) and ``R_rel = R_ref^T R``, exactly the
    single-chunk formulas.
    """
    chunks = np.asarray(pose_chunks_xyz_wxyz, dtype=np.float64)
    assert chunks.ndim == 3 and chunks.shape[-1] == RAW_POSE_DIM, (
        f"expected chunks (L, H, {RAW_POSE_DIM}), got {chunks.shape}"
    )
    refs = np.asarray(reference_poses_xyz_wxyz, dtype=np.float64).reshape(
        -1, RAW_POSE_DIM
    )
    if len(refs) != len(chunks):
        raise ValueError(
            f"need one reference pose per chunk, got {len(refs)} for "
            f"{len(chunks)} chunks"
        )
    positions = chunks[..., :3]  # (L, H, 3)
    rotations = _quat_wxyz_to_matrix(chunks[..., 3:7])  # (L, H, 3, 3)
    ref_pos = refs[:, :3]  # (L, 3)
    ref_rot = _quat_wxyz_to_matrix(refs[:, 3:7])  # (L, 3, 3)
    r0_t = np.swapaxes(ref_rot, -1, -2)  # R_ref^{-1}, (L, 3, 3)
    diff = positions - ref_pos[:, None, :]  # (L, H, 3)
    delta_xyz = (diff[..., None, :] @ ref_rot[:, None])[:, :, 0, :]
    rel_rot = r0_t[:, None] @ rotations  # (L, H, 3, 3)
    return np.concatenate(
        [delta_xyz, _matrix_to_rot6d(rel_rot)], axis=-1
    ).astype(np.float32)


def invert_poses(poses_xyz_wxyz: np.ndarray) -> np.ndarray:
    """Invert each rigid transform: ``T -> T^{-1}`` (``R.T`` / ``-R.T @ t``).

    Used to turn a declared ``camera_from_world`` extrinsics column into the
    ``world_from_camera`` poses the rest of the module expects.
    """
    poses = _check_raw_pose(poses_xyz_wxyz)
    rotations = _quat_wxyz_to_matrix(poses[:, 3:7])
    rotations_t = np.swapaxes(rotations, -1, -2)
    positions = -np.einsum("nij,nj->ni", rotations_t, poses[:, :3])
    quats = np.stack(
        [_matrix_to_quat_wxyz(rotations_t[i]) for i in range(len(poses))], axis=0
    )
    return np.concatenate([positions, quats], axis=1)


# ---------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------
# The two-pass machinery is shared with the raw parquet statistics; see
# starVLA/dataloader/gr00t_lerobot/streaming_stats.py for the exactness
# contract. Re-exported here because this module's own helpers (and the
# regression tests) address them as module attributes.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EEPoseReprLeRobotDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset that re-encodes selected EE-pose keys at load time.

    Args:
        pose_keys: short key names (without the ``state.``/``action.`` prefix)
            whose RAW layout is 7-d ``xyz_wxyz``.
        state_pose_format: encoding for ``state.<key>``; one of
            ``raw`` (7-d), ``abs_xyz_rot6d`` (9-d), or ``cam_abs_xyz_rot6d`` (9-d).
        action_pose_format: encoding for ``action.<key>``; one of
            ``raw`` (7-d), ``abs_xyz_rot6d`` / ``cam_abs_xyz_rot6d`` (9-d), or
            ``delta_xyz_rot6d`` (9-d).

    All other constructor arguments are forwarded to LeRobotSingleDataset.
    """

    def __init__(
        self,
        *args,
        pose_keys: list[str],
        state_pose_format: str = "abs_xyz_rot6d",
        action_pose_format: str = "delta_xyz_rot6d",
        ref_noise: dict | None = None,
        ee_in_camera_frame: bool = True,
        camera_extrinsics_key: str | None = None,
        camera_extrinsics_semantics: str | None = None,
        state_history_frames: int = 1,
        **kwargs,
    ):
        # Must be set before super().__init__: the base constructor calls
        # _get_metadata(), which our override below depends on.
        self._pose_keys = [str(k).split(".")[-1] for k in pose_keys]
        if not self._pose_keys:
            raise ValueError("pose_keys must be non-empty; use LeRobotSingleDataset instead.")
        if state_pose_format not in STATE_POSE_FORMATS:
            raise ValueError(
                f"state_pose_format={state_pose_format!r}; expected one of {STATE_POSE_FORMATS} "
                "(state is a single frame, chunk-wise delta does not apply)"
            )
        if action_pose_format not in ACTION_POSE_FORMATS:
            raise ValueError(
                f"action_pose_format={action_pose_format!r}; expected one of {ACTION_POSE_FORMATS}"
            )
        ee_in_camera_frame = bool(ee_in_camera_frame)
        for fmt, label in (
            (state_pose_format, "state_pose_format"),
            (action_pose_format, "action_pose_format"),
        ):
            if fmt in CAMERA_FRAME_POSE_FORMATS and not ee_in_camera_frame:
                raise ValueError(
                    f"{label}={fmt!r} requires ee_in_camera_frame=true "
                    "(camera-frame absolute spatial pose)."
                )
        self._state_pose_format = state_pose_format
        self._action_pose_format = action_pose_format
        self._ee_in_camera_frame = ee_in_camera_frame
        # Must match datasets.vla_data.state_history_frames: the converted
        # STATE statistics rebuild the same history block the sampler emits
        # (all rows in the chunk-t0 camera), so the quantiles calibrate the
        # distribution the model actually sees.
        self._state_history_frames = max(1, int(state_history_frames))
        # Resolved lazily from meta/modality.json on first use (dataset_path is
        # only available after super().__init__); an explicit train-config
        # override short-circuits the lookup.
        self._camera_extrinsics_key_override = (
            str(camera_extrinsics_key) if camera_extrinsics_key else None
        )
        self._camera_extrinsics_semantics_override = (
            str(camera_extrinsics_semantics) if camera_extrinsics_semantics else None
        )
        self._camera_extrinsics_key_resolved: str | None = None
        self._camera_extrinsics_semantics_resolved: str | None = None
        self._camera_axes_resolved: str | None = None
        # Lazily resolved memory|streaming decision for the converted stats
        # (depends on the parquet sizes via the data fingerprint).
        self._stats_mode_resolved: str | None = None
        # Filled by get_step_data so _apply_action_mode can read chunk-t0 camera.
        self._pose_ctx_trajectory_id: int | None = None
        self._pose_ctx_base_index: int | None = None
        # episode_id -> (T, 7) cache for sidecar .npy loads
        self._camera_extrinsics_cache: dict[int, np.ndarray] = {}

        # ref_noise: perturb delta reference (= model state). Stats stay clean.
        ref_noise = dict(ref_noise or {})
        self._ref_noise_prob = float(ref_noise.get("prob", 0.0))
        self._ref_noise_rot_deg = float(ref_noise.get("rot_deg", 0.0))
        self._ref_noise_trans_m = float(ref_noise.get("trans_m", 0.0))
        if not 0.0 <= self._ref_noise_prob <= 1.0:
            raise ValueError(f"ref_noise.prob must be in [0, 1], got {self._ref_noise_prob}")
        if self._ref_noise_rot_deg < 0.0 or self._ref_noise_trans_m < 0.0:
            raise ValueError("ref_noise.rot_deg / trans_m must be >= 0")
        if self._ref_noise_prob > 0.0 and action_pose_format != "delta_xyz_rot6d":
            raise ValueError(
                "ref_noise only applies to action_pose_format='delta_xyz_rot6d' "
                f"(got {action_pose_format!r})"
            )
        self._ref_noise_rng: np.random.Generator | None = None
        self._ref_noise_pid: int | None = None
        super().__init__(*args, **kwargs)

    def _ref_noise_generator(self) -> np.random.Generator:
        """RNG from ``torch.initial_seed()`` + dataset_name; re-seed after fork."""
        import hashlib
        import os

        import torch

        pid = os.getpid()
        if self._ref_noise_rng is None or self._ref_noise_pid != pid:
            material = f"{torch.initial_seed()}|{self.dataset_name}".encode()
            seed = int.from_bytes(hashlib.md5(material).digest()[:8], "little")
            self._ref_noise_rng = np.random.default_rng(seed)
            self._ref_noise_pid = pid
        return self._ref_noise_rng

    # ------------------------------------------------------------------
    # Data path: convert pose keys after raw loading, before transforms.
    # ------------------------------------------------------------------
    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Stash chunk context so camera-frame conversion can use t0 extrinsics."""
        self._pose_ctx_trajectory_id = int(trajectory_id)
        self._pose_ctx_base_index = int(base_index)
        try:
            return super().get_step_data(trajectory_id, base_index)
        finally:
            self._pose_ctx_trajectory_id = None
            self._pose_ctx_base_index = None

    @property
    def _camera_extrinsics_key(self) -> str:
        """train-config override > modality.json original_key > default column."""
        if self._camera_extrinsics_key_resolved is None:
            self._resolve_camera_extrinsics()
        return self._camera_extrinsics_key_resolved

    @property
    def _camera_extrinsics_semantics(self) -> str:
        """Transform direction of the extrinsics column (see the module docs)."""
        if self._camera_extrinsics_semantics_resolved is None:
            self._resolve_camera_extrinsics()
        return self._camera_extrinsics_semantics_resolved

    @property
    def _camera_axes(self) -> str:
        """Axis convention of the camera frame; deployment must match it."""
        if self._camera_axes_resolved is None:
            self._resolve_camera_extrinsics()
        return self._camera_axes_resolved

    def _resolve_camera_extrinsics(self) -> None:
        """Resolve the extrinsics column name and its transform direction."""
        modality_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        declared = {}
        if modality_path.is_file():
            with open(modality_path, "r") as f:
                declared = (json.load(f) or {}).get("camera_extrinsics") or {}
        if not isinstance(declared, dict):
            declared = {}

        names = [n for n in CAMERA_EXTRINSICS_NAMES if n in declared]
        names += [n for n in declared if n not in names]
        camera = names[0] if names else CAMERA_EXTRINSICS_NAMES[0]
        entry = declared.get(camera) or {}
        if not isinstance(entry, dict):
            entry = {}

        key = self._camera_extrinsics_key_override
        if key is None:
            key = str(entry.get("original_key") or DEFAULT_CAMERA_EXTRINSICS_KEY)
        _check_extrinsics_pose_order(camera, entry)
        self._camera_extrinsics_semantics_resolved = _resolve_extrinsics_semantics(
            camera, entry, self._camera_extrinsics_semantics_override
        )
        self._camera_axes_resolved = str(entry.get("camera_axes") or DEFAULT_CAMERA_AXES)
        self._camera_extrinsics_key_resolved = key

    def _stack_pose_column(self, column) -> np.ndarray:
        """Stack a parquet pose/extrinsics column into (T, 7)."""
        arr = np.stack(column)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != RAW_POSE_DIM:
            raise ValueError(
                f"Expected extrinsics/pose column shape (T, {RAW_POSE_DIM}), got {arr.shape}"
            )
        return arr

    def _as_world_from_camera(self, extrinsics: np.ndarray) -> np.ndarray:
        """Normalize loaded extrinsics to ``T_world_from_camera``."""
        if self._camera_extrinsics_semantics == CAMERA_FROM_WORLD:
            return invert_poses(extrinsics)
        return extrinsics

    def _load_camera_extrinsics_array(
        self,
        *,
        trajectory_id: int | None = None,
        episode_frame=None,
        episode_index: int | None = None,
    ) -> np.ndarray:
        """Load ``(T, 7)`` world-from-camera poses (invert if camera_from_world).

        Lookup order: parquet column → ``…/episode_{id:06d}.npy`` →
        video-aligned extrinsics sidecar.
        """
        key = self._camera_extrinsics_key
        # 1) parquet column
        frame = episode_frame
        if frame is None and self.curr_traj_data is not None:
            frame = self.curr_traj_data
        if frame is not None and key in frame.columns:
            return self._as_world_from_camera(self._stack_pose_column(frame[key]))

        ep_id = episode_index if episode_index is not None else trajectory_id
        if ep_id is None:
            raise ValueError(
                f"ee_in_camera_frame requires trajectory/episode id to load {key!r}"
            )
        ep_id = int(ep_id)
        if ep_id in self._camera_extrinsics_cache:
            return self._camera_extrinsics_cache[ep_id]

        # 2) per-episode npy under the extrinsics key directory
        episode_path = self.dataset_path / key / f"episode_{ep_id:06d}.npy"
        candidates = [episode_path]

        # 3) video-aligned sidecar (LeRobot v3 chunk/file indices)
        try:
            video_path = self.get_video_path(ep_id, "cam_high")
            candidates.append(
                video_path.parent.parent
                / "extrinsics"
                / video_path.parent.name
                / f"{video_path.stem}.npy"
            )
            # Also: videos/.../cam_high/extrinsics/chunk-XXX/file-YYY.npy
            # when video is videos/.../cam_high/chunk-XXX/file-YYY.mp4
            candidates.append(
                self.dataset_path
                / key
                / video_path.parent.name
                / f"{video_path.stem}.npy"
            )
        except Exception:
            pass

        for path in candidates:
            if path.is_file():
                arr = np.asarray(np.load(path), dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != RAW_POSE_DIM:
                    raise ValueError(
                        f"Camera extrinsics at {path} must have shape (T, {RAW_POSE_DIM}), "
                        f"got {arr.shape}"
                    )
                if frame is not None and len(arr) != len(frame):
                    raise ValueError(
                        f"Camera extrinsics length {len(arr)} at {path} != "
                        f"episode length {len(frame)}"
                    )
                arr = self._as_world_from_camera(arr)
                self._camera_extrinsics_cache[ep_id] = arr
                return arr

        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"ee_in_camera_frame=True but no camera extrinsics found for episode "
            f"{ep_id} under key {key!r}. Tried parquet column {key!r} and: {searched}. "
            f"Declare the column in {LE_ROBOT_MODALITY_FILENAME} under "
            "camera_extrinsics.<camera>.original_key, set "
            "datasets.vla_data.camera_extrinsics_key, or turn off "
            "datasets.vla_data.ee_in_camera_frame."
        )

    def _chunk_camera_pose(self, extrinsics: np.ndarray, base_index: int) -> np.ndarray:
        """Camera pose at chunk t0 with first/last clipping (matches pose padding)."""
        if extrinsics.ndim != 2 or extrinsics.shape[1] != RAW_POSE_DIM:
            raise ValueError(
                f"Expected extrinsics (T, {RAW_POSE_DIM}), got {extrinsics.shape}"
            )
        index = int(np.clip(base_index, 0, len(extrinsics) - 1))
        return np.asarray(extrinsics[index], dtype=np.float64).reshape(RAW_POSE_DIM)

    def get_state_or_action(self, trajectory_id: int, modality: str, key: str, base_index: int) -> np.ndarray:
        """First/last-pad pose keys (raw poses are absolute; zero-pad → NaN rot6d)."""
        short_key = key.split(".")[-1]
        if short_key in self._pose_keys:
            key_meta = getattr(self.metadata.modalities, modality)[short_key]
            if not key_meta.absolute:
                key_meta.absolute = True
                try:
                    return super().get_state_or_action(trajectory_id, modality, key, base_index)
                finally:
                    key_meta.absolute = False
        return super().get_state_or_action(trajectory_id, modality, key, base_index)

    def _apply_action_mode(self, data: dict) -> dict:
        camera_pose = None
        if self._ee_in_camera_frame:
            if self._pose_ctx_trajectory_id is None or self._pose_ctx_base_index is None:
                raise RuntimeError(
                    "ee_in_camera_frame requires get_step_data to set trajectory/base context"
                )
            extrinsics = self._load_camera_extrinsics_array(
                trajectory_id=self._pose_ctx_trajectory_id
            )
            camera_pose = self._chunk_camera_pose(extrinsics, self._pose_ctx_base_index)

        for short_key in self._pose_keys:
            state_key = f"state.{short_key}"
            action_key = f"action.{short_key}"
            # The CURRENT raw state pose is the delta reference (before state
            # convert). With state history frames the state block is (T, 7)
            # with delta indices [1-T .. 0], so the current frame is the LAST
            # row; T == 1 keeps the legacy behaviour bit-identical.
            reference_pose = None
            if state_key in data:
                raw_state = np.asarray(data[state_key], dtype=np.float64)
                if camera_pose is not None:
                    # Every history row is expressed in the SAME chunk-t0
                    # camera frame (mirrors the action chunk convention).
                    raw_state = poses_world_to_frame(raw_state, camera_pose)
                reference_pose = raw_state[-1].copy()
                # ref_noise after world→cam; same pose for model state + delta ref.
                if self._ref_noise_prob > 0.0:
                    rng = self._ref_noise_generator()
                    if rng.random() < self._ref_noise_prob:
                        reference_pose = _perturb_pose(
                            reference_pose,
                            self._ref_noise_rot_deg,
                            self._ref_noise_trans_m,
                            rng,
                        )
                        raw_state = raw_state.copy()
                        raw_state[-1] = reference_pose
                data[state_key] = convert_pose(raw_state, self._state_pose_format)
            if action_key in data:
                action_raw = np.asarray(data[action_key], dtype=np.float64)
                if camera_pose is not None:
                    action_raw = poses_world_to_frame(action_raw, camera_pose)
                if self._action_pose_format == "delta_xyz_rot6d":
                    if reference_pose is None:
                        raise KeyError(
                            f"action key {action_key!r} uses delta_xyz_rot6d but its "
                            f"state key {state_key!r} (the delta reference pose) is "
                            "missing from the sample"
                        )
                    data[action_key] = pose_chunk_to_delta_xyz_rot6d(action_raw, reference_pose)
                else:
                    data[action_key] = convert_pose(action_raw, self._action_pose_format)
        # Base behaviour (linear delta/rel modes) still applies to other keys.
        return super()._apply_action_mode(data)

    # ------------------------------------------------------------------
    # Metadata: replace raw 7-d stats with stats over converted values.
    # ------------------------------------------------------------------
    def _get_metadata(self, embodiment_tag) -> DatasetMetadata:
        metadata = super()._get_metadata(embodiment_tag)

        modality_formats = {
            "state": self._state_pose_format,
            "action": self._action_pose_format,
        }
        if all(fmt == "raw" for fmt in modality_formats.values()):
            return metadata

        converted_stats = self._load_or_compute_converted_stats()

        meta_dict = metadata.model_dump()
        for short_key in self._pose_keys:
            for modality, pose_format in modality_formats.items():
                if pose_format == "raw":
                    continue
                full_key = f"{modality}.{short_key}"
                if short_key not in meta_dict["statistics"].get(modality, {}):
                    continue
                meta_dict["statistics"][modality][short_key] = converted_stats[full_key]
                key_meta = meta_dict["modalities"][modality][short_key]
                key_meta["shape"] = [POSE_FORMAT_DIMS[pose_format]]
                if pose_format == "delta_xyz_rot6d":
                    # Chunk-wise deltas are relative, not absolute targets.
                    key_meta["absolute"] = False
        return DatasetMetadata.model_validate(meta_dict)

    def _converted_stats_cache_config(self, action_indices: list[int]) -> dict:
        config = {
            "version": _STATS_VERSION,
            "keys": sorted(self._pose_keys),
            "raw_format": RAW_POSE_FORMAT,
            "state_format": self._state_pose_format,
            "action_format": self._action_pose_format,
            "ee_in_camera_frame": bool(self._ee_in_camera_frame),
            "action_indices": [int(i) for i in action_indices],
            **(
                {"state_history_frames": int(self._state_history_frames)}
                # >1 changes the state-stat construction (chunk-t0 history
                # blocks); == 1 stays bit-identical to pre-history caches,
                # so those remain interchangeable.
                if self._state_history_frames > 1
                else {}
            ),
            **(
                {"stats_mode": "streaming"}
                # Streaming statistics have float64 mean/std (memory-mode
                # caches must not be adopted), while min/max/quantiles are
                # bit-identical. Recorded only when streaming so every
                # memory-mode cache keeps its existing hash.
                if self._stats_mode() == "streaming"
                else {}
            ),
            "data_fingerprint": self._all_parquet_fingerprint(),
        }
        # Resolve/validate extrinsics only when the feature is on.
        extrinsics_globs: tuple[str, ...] = ()
        if self._ee_in_camera_frame:
            config["camera_extrinsics_key"] = self._camera_extrinsics_key
            config["camera_extrinsics_semantics"] = self._camera_extrinsics_semantics
            config["camera_axes"] = self._camera_axes
            # Sidecar fallbacks of _load_camera_extrinsics_array, for datasets
            # that store the poses outside the parquet (all shipped ones use a
            # column, so these match nothing and cost one empty glob).
            extrinsics_globs = (
                f"{self._camera_extrinsics_key}/*.npy",
                "videos/*/extrinsics/*/*.npy",
            )
        # modality.json is read for every key's original_key / start / end, and
        # the parquet-only data fingerprint does not cover it.
        config["meta_fingerprint"] = _dataset_meta_fingerprint(
            self.dataset_path, extrinsics_globs
        )
        if self._excluded_episodes:
            config["exclude_episodes"] = sorted(int(v) for v in self._excluded_episodes)
        # Recorded only when the embodiment's layout actually moves a pose key
        # off what modality.json declares: a no-op override produces identical
        # statistics, so those caches stay interchangeable.
        key_range_overrides = self._pose_key_range_overrides()
        if key_range_overrides:
            config["key_ranges"] = key_range_overrides
        return config

    def _pose_key_range_overrides(self) -> dict:
        """Effective ``[start, end]`` of every pose key that
        ``modality_key_ranges`` moves off its ``modality.json`` declaration.

        The converted statistics slice the parquet columns through the same
        overridden metadata the runtime data path uses, so two runs whose
        embodiments lay the flat vector out differently must not share one
        cache — the layout is not otherwise visible in this config.
        """
        # Resolve through the accessor FIRST: it carries the actionable
        # "provide a modality.json" assertion for a dataset that has none.
        effective = self._get_lerobot_modality_meta()
        modality_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            declared = json.load(f) or {}
        overrides: dict[str, list[int]] = {}
        for modality in ("state", "action"):
            declared_keys = declared.get(modality) or {}
            effective_keys = getattr(effective, modality, None) or {}
            for short_key in self._pose_keys:
                if short_key not in effective_keys:
                    continue
                key_meta = effective_keys[short_key]
                declared_key = declared_keys.get(short_key) or {}
                if (
                    declared_key.get("start") == key_meta.start
                    and declared_key.get("end") == key_meta.end
                ):
                    continue
                overrides[f"{modality}.{short_key}"] = [
                    int(key_meta.start),
                    int(key_meta.end),
                ]
        return overrides

    def _stats_mode(self) -> str:
        """``memory`` | ``streaming`` for the converted-statistics compute.

        Resolved once from the parquet sizes (deterministic for a given
        dataset state), so the cache config and the compute dispatch always
        agree — including between the offline precompute and training.
        """
        # getattr: tests drive this on partially-initialized fakes.
        resolved = getattr(self, "_stats_mode_resolved", None)
        if resolved is None:
            fingerprint = self._all_parquet_fingerprint()
            total_bytes = sum(int(entry[1]) for entry in fingerprint)
            resolved = _resolve_stats_mode(self.data_cfg, total_bytes)
            self._stats_mode_resolved = resolved
        return resolved

    def _load_or_compute_converted_stats(self) -> dict:
        """Rank-0 computes and caches; other ranks wait and load (same
        pattern as the base class's raw-statistics step).

        Non-zero ranks poll the cache file instead of blocking on an NCCL
        ``barrier`` during the (possibly hour-long) compute: DeepSpeed/Accelerate
        process-group timeouts are often still 600s and abort the wait.
        """
        base_path = self.dataset_path / EE_DELTA_STATS_FILENAME
        action_cfg = self.modality_configs["action"]
        action_indices = list(action_cfg.delta_indices)
        cache_config = self._converted_stats_cache_config(action_indices)
        stats_path = _stats_cache_path(base_path, cache_config)

        if _force_stats_recompute():
            # Refuse silently-misbehaving combinations up front: strict mode
            # (never computes) and multi-rank (stale reads on other ranks).
            _check_force_stats_recompute_allowed(self.data_cfg)

        if _precomputed_stats_required(self.data_cfg):
            # Strict mode: every rank reads the content-addressed cache
            # directly; a miss fails the launch instead of scanning parquet.
            return _load_precomputed_stats_or_raise(
                base_path,
                cache_config,
                description="converted-pose statistics cache",
            )

        is_main = _is_stats_main_process

        stats = None
        if is_main():
            if _force_stats_recompute():
                stats, cached = None, False
            else:
                stats, cached = _load_stats_cache_for_config(base_path, cache_config)
            if stats is None:
                stats = self._compute_converted_stats(action_indices)
            if not cached:
                # Freshly computed, or adopted from the legacy shared path:
                # give this config its own file for later runs.
                _save_stats_cache(stats_path, cache_config, stats)
        if stats is None:
            stats = _wait_for_stats_cache(
                stats_path,
                cache_config,
                description="converted-pose statistics cache",
            )
        return stats

    def _converted_episode_blocks(
        self,
        *,
        state_poses: np.ndarray,
        action_poses: np.ndarray,
        extrinsics: np.ndarray | None,
        offsets: np.ndarray,
        impl: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Converted state/action blocks for ONE episode.

        Returns ``(state_block, action_block)``. State: ``(L*K, Ds)`` when
        extrinsics are on (K history rows per chunk base, all in the chunk-t0
        camera), else ``(L, Ds)``. Action: ``(L, H, Da)`` when the format is
        chunked (delta, or abs in the camera frame), else ``(L, Da)``. Either
        is ``None`` when that modality's pose format is ``raw``.

        ``impl="loop"`` is the original per-base-index Python loop, kept for
        A/B verification (``SPARK_0_STATS_IMPL``); ``"vectorized"`` computes
        the same values with whole-episode tensor ops. Values agree to within
        float64 round-off (batched vs looped BLAS matmuls accumulate in
        different orders; observed <= 1e-13, only at near-zero entries) —
        immaterial for the pooled quantile/mean statistics, and the stats
        cache hash does not include the implementation.
        """
        length = int(state_poses.shape[0])
        state_block = None
        action_block = None

        if self._state_pose_format != "raw":
            if extrinsics is None:
                state_block = convert_pose(state_poses, self._state_pose_format)
            elif impl == "loop":
                # Mirror the training-time state block exactly: for every
                # chunk base t0, the history rows are the first/last-padded
                # frames clip(t0 - k), ALL in the chunk-t0 camera. state[t] in
                # camera[t] per frame only matches the current row and would
                # normalize history rows against a distribution the model
                # never sees. state_history_frames == 1 reduces to the
                # per-frame values bit-identically.
                history_offsets = np.arange(1 - self._state_history_frames, 1)
                ep_rows: list[np.ndarray] = []
                for base_index in range(length):
                    indices = np.clip(history_offsets + base_index, 0, length - 1)
                    block = poses_world_to_frame(
                        state_poses[indices], extrinsics[base_index]
                    )
                    ep_rows.append(convert_pose(block, self._state_pose_format))
                state_block = np.concatenate(ep_rows, axis=0)
            else:
                # Same construction as the loop, batched: (L, K) clip-padded
                # gather indices, per-base cameras repeated per history row.
                history_offsets = np.arange(1 - self._state_history_frames, 1)
                history_idx = np.clip(
                    history_offsets[None, :] + np.arange(length)[:, None],
                    0,
                    length - 1,
                )
                blocks = state_poses[history_idx].reshape(-1, RAW_POSE_DIM)
                cams = np.repeat(extrinsics, self._state_history_frames, axis=0)
                state_block = convert_pose(
                    _poses_world_to_frame_rows(blocks, cams),
                    self._state_pose_format,
                )

        if self._action_pose_format in ABS_XYZ_ROT6D_FORMATS:
            if extrinsics is None:
                action_block = pose_to_abs_xyz_rot6d(action_poses)
            elif impl == "loop":
                # Whole chunk in cam[t0] (not camera[t] per row).
                ep_rows = []
                for base_index in range(length):
                    indices = np.clip(offsets + base_index, 0, length - 1)
                    action_chunk = poses_world_to_frame(
                        action_poses[indices], extrinsics[base_index]
                    )
                    ep_rows.append(pose_to_abs_xyz_rot6d(action_chunk))
                action_block = np.stack(ep_rows, axis=0)
            else:
                chunk_idx = np.clip(
                    offsets[None, :] + np.arange(length)[:, None], 0, length - 1
                )
                chunks = action_poses[chunk_idx].reshape(-1, RAW_POSE_DIM)
                cams = np.repeat(extrinsics, len(offsets), axis=0)
                action_block = pose_to_abs_xyz_rot6d(
                    _poses_world_to_frame_rows(chunks, cams)
                ).reshape(length, len(offsets), -1)
        elif self._action_pose_format == "delta_xyz_rot6d":
            if impl == "loop":
                ep_rows = []
                for base_index in range(length):
                    indices = np.clip(offsets + base_index, 0, length - 1)
                    action_chunk = action_poses[indices]
                    state_ref = state_poses[base_index]
                    if extrinsics is not None:
                        cam = extrinsics[base_index]
                        action_chunk = poses_world_to_frame(action_chunk, cam)
                        state_ref = poses_world_to_frame(
                            state_ref.reshape(1, RAW_POSE_DIM), cam
                        )[0]
                    ep_rows.append(
                        pose_chunk_to_delta_xyz_rot6d(action_chunk, state_ref)
                    )
                action_block = np.stack(ep_rows, axis=0)
            else:
                chunk_idx = np.clip(
                    offsets[None, :] + np.arange(length)[:, None], 0, length - 1
                )
                chunks = action_poses[chunk_idx]  # (L, H, 7)
                refs = state_poses  # (L, 7)
                if extrinsics is not None:
                    cams = np.repeat(extrinsics, len(offsets), axis=0)
                    chunks = _poses_world_to_frame_rows(
                        chunks.reshape(-1, RAW_POSE_DIM), cams
                    ).reshape(length, len(offsets), RAW_POSE_DIM)
                    refs = _poses_world_to_frame_rows(state_poses, extrinsics)
                action_block = _pose_chunks_to_delta_xyz_rot6d(chunks, refs)

        return state_block, action_block

    def _iter_pose_key_episode_blocks(
        self,
        parquet_paths: list,
        offsets: np.ndarray,
        impl: str,
        *,
        desc: str,
    ):
        """Yield ``(file_index, short_key, state_block, action_block)`` per episode.

        Iteration order is the legacy accumulation order — parquet file
        (sorted), then pose key, then episode — so the in-memory statistics
        accumulate byte-identical arrays through this generator.
        """
        # Same accessor the runtime data path uses, so datasets.vla_data.
        # modality_key_ranges (the embodiment's cumulative layout) applies to
        # the statistics too. Reading modality.json directly here sliced the
        # stats by the on-disk declaration while training sliced the same
        # columns by the override.
        modality_meta = self._get_lerobot_modality_meta()

        import pandas as pd  # heavy import, keep local like the base module usage

        for file_index, parquet_path in enumerate(tqdm(parquet_paths, desc=desc)):
            # Same column projection the runtime path uses: pose statistics
            # never touch the tactile columns, and materialising them as pandas
            # objects dominates the read (17.7 s vs 0.12 s per file on the
            # egopro export).
            frame = pd.read_parquet(
                parquet_path, columns=self._pandas_columns(parquet_path)
            )
            # LeRobot v3 packs many episodes into one parquet file; group so
            # chunks / camera extrinsics never span episode boundaries.
            # (v2 layout has one episode per file and no such column.)
            if "episode_index" in frame.columns:
                episode_groups = list(frame.groupby("episode_index", sort=True))
            else:
                fallback_ep = _episode_index_from_parquet_name(parquet_path.name)
                episode_groups = [(fallback_ep if fallback_ep is not None else 0, frame)]

            for short_key in self._pose_keys:
                # The state column is read even for a raw state format: the
                # delta action encoding references state[base_index].
                state_meta = modality_meta.get_key_meta(f"state.{short_key}")
                action_meta = (
                    modality_meta.get_key_meta(f"action.{short_key}")
                    if self._action_pose_format != "raw"
                    else None
                )

                for episode_index, episode_frame in episode_groups:
                    state_col = np.stack(episode_frame[state_meta.original_key])
                    state_poses = state_col[:, state_meta.start : state_meta.end]
                    if action_meta is None:
                        action_poses = np.empty((len(state_poses), RAW_POSE_DIM))
                    else:
                        action_col = np.stack(episode_frame[action_meta.original_key])
                        action_poses = action_col[:, action_meta.start : action_meta.end]
                    length = action_poses.shape[0]

                    extrinsics = None
                    if self._ee_in_camera_frame:
                        extrinsics = self._load_camera_extrinsics_array(
                            episode_frame=episode_frame,
                            episode_index=int(episode_index),
                        )
                        if len(extrinsics) != length:
                            raise ValueError(
                                f"Camera extrinsics length {len(extrinsics)} != "
                                f"episode {episode_index} length {length}"
                            )

                    state_block, action_block = self._converted_episode_blocks(
                        state_poses=state_poses,
                        action_poses=action_poses,
                        extrinsics=extrinsics,
                        offsets=offsets,
                        impl=impl,
                    )
                    yield file_index, short_key, state_block, action_block

    def _compute_converted_stats(self, action_indices: list[int]) -> dict:
        """Statistics over the CONVERTED representation.

        Mirrors ``calculate_delta_action_statistics``: for actions, every
        base index t of every trajectory yields one converted chunk (with
        first/last padding via index clipping, matching the runtime padding
        of absolute keys); statistics run over all chunk rows. State stats
        use the same per-base-index construction: every chunk base yields
        its state-history block (clip-padded, all rows in the chunk-t0
        camera) and statistics run over all history rows.

        Action keys additionally get ``q01_per_row`` / ``q99_per_row`` of shape
        ``(H, D)``: the same quantiles taken within each horizon row instead of
        over the pooled rows. The pooled fields are unchanged, so a run that
        does not ask for per-row normalization is unaffected.

        Two compute modes (see :func:`_resolve_stats_mode`): ``memory`` pools
        all converted values in RAM (bit-identical to every cache since
        ``_STATS_VERSION`` 7); ``streaming`` makes two passes with per-column
        order histograms (~1 GB RAM at any dataset size; min/max/quantiles
        bit-identical, mean/std float64-accurate).
        """
        # Same episode exclusion as sampling (datasets.vla_data.exclude_episodes),
        # so the converted stats and the training data cover the same episodes.
        parquet_paths = sorted(
            p
            for p in self.dataset_path.glob(LE_ROBOT_DATA_FILENAME)
            if _episode_index_from_parquet_name(p.name) not in self._excluded_episodes
        )
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files under {self.dataset_path}")

        offsets = np.asarray(action_indices, dtype=np.int64)
        impl = _stats_impl()
        mode = self._stats_mode()
        started = time.monotonic()
        if mode == "streaming":
            stats = self._compute_converted_stats_streaming(parquet_paths, offsets, impl)
        else:
            stats = self._compute_converted_stats_memory(parquet_paths, offsets, impl)
        # getattr: tests drive this method on partially-initialized fakes.
        name = getattr(self, "_dataset_name", None) or "dataset"
        print(
            f"[{name}] converted-pose statistics computed with "
            f"impl={impl} mode={mode} in {time.monotonic() - started:.1f}s"
        )
        return stats

    def _compute_converted_stats_memory(
        self, parquet_paths: list, offsets: np.ndarray, impl: str
    ) -> dict:
        """Pool every converted value in RAM and reduce with numpy.

        This is the historical implementation: the per-file accumulation
        boundaries and the final concatenation order are unchanged, so the
        resulting statistics are bit-identical to pre-refactor versions.
        """
        accum: dict[str, list[np.ndarray]] = {}
        for short_key in self._pose_keys:
            if self._state_pose_format != "raw":
                accum[f"state.{short_key}"] = []
            if self._action_pose_format != "raw":
                accum[f"action.{short_key}"] = []

        file_rows: dict[str, list[np.ndarray]] = {}
        current_file: int | None = None

        def flush_file() -> None:
            for full_key, rows in file_rows.items():
                accum[full_key].append(np.concatenate(rows, axis=0).astype(np.float32))
            file_rows.clear()

        for file_index, short_key, state_block, action_block in self._iter_pose_key_episode_blocks(
            parquet_paths, offsets, impl, desc="Pose-repr stats (converted)"
        ):
            if current_file is None:
                current_file = file_index
            elif file_index != current_file:
                flush_file()
                current_file = file_index
            if state_block is not None:
                file_rows.setdefault(f"state.{short_key}", []).append(state_block)
            if action_block is not None:
                file_rows.setdefault(f"action.{short_key}", []).append(action_block)
        flush_file()

        stats: dict[str, dict] = {}
        for full_key, arrays in accum.items():
            values = np.concatenate(arrays, axis=0).astype(np.float32)
            per_row = None
            if values.ndim == 3:
                per_row = values  # (num_chunks, H, D)
                values = values.reshape(-1, values.shape[-1])
            key_stats = {
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist(),
                "min": np.min(values, axis=0).tolist(),
                "max": np.max(values, axis=0).tolist(),
                "q01": np.quantile(values, 0.01, axis=0).tolist(),
                "q99": np.quantile(values, 0.99, axis=0).tolist(),
            }
            if per_row is not None:
                key_stats["q01_per_row"] = np.quantile(per_row, 0.01, axis=0).tolist()
                key_stats["q99_per_row"] = np.quantile(per_row, 0.99, axis=0).tolist()
            stats[full_key] = key_stats
        return stats

    def _compute_converted_stats_streaming(
        self, parquet_paths: list, offsets: np.ndarray, impl: str
    ) -> dict:
        """Two-pass exact statistics with ~1 GB RAM at any dataset size.

        Pass 1 builds the per-column order histograms and Welford
        accumulators; pass 2 collects only the values inside each target
        rank's histogram bucket and selects exactly. See
        :class:`_StreamingStatsBuilder` for the exactness contract.
        """
        # One allowance for the whole run: every key gets its own builder and
        # every builder collects for two scopes, so a per-builder budget would
        # scale the pass-2 peak with the number of pose keys.
        budget = _CollectBudget()
        builders: dict[str, _StreamingStatsBuilder] = {}
        for short_key in self._pose_keys:
            if self._state_pose_format != "raw":
                builders[f"state.{short_key}"] = _StreamingStatsBuilder(budget=budget)
            if self._action_pose_format != "raw":
                builders[f"action.{short_key}"] = _StreamingStatsBuilder(budget=budget)

        for _, short_key, state_block, action_block in self._iter_pose_key_episode_blocks(
            parquet_paths, offsets, impl, desc="Pose-repr stats (pass 1/2: histograms)"
        ):
            if state_block is not None:
                builders[f"state.{short_key}"].update(state_block)
            if action_block is not None:
                builders[f"action.{short_key}"].update(action_block)

        for builder in builders.values():
            builder.finalize_pass1()

        for _, short_key, state_block, action_block in self._iter_pose_key_episode_blocks(
            parquet_paths, offsets, impl, desc="Pose-repr stats (pass 2/2: quantiles)"
        ):
            if state_block is not None:
                builders[f"state.{short_key}"].collect_update(state_block)
            if action_block is not None:
                builders[f"action.{short_key}"].collect_update(action_block)

        return {full_key: builder.result() for full_key, builder in builders.items()}
