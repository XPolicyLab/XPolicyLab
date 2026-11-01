"""RoboDojo ``action_mode: wuji58`` — the 58-D WUJI hybrid wrist/hand reader.

Torch-free companion of :mod:`openwam.dataloader.robodojo` for the
``tianji_wuji`` embodiment (two 7-DOF arms carrying 20-DOF Wuji hands). The
representation is NOT EEF20: it is the WUJI SimBench ``wrist_rel_te2`` layout
owned by ``XPolicyLab.utils.wuji58``::

    [ 0:20]  left hand qpos          absolute joints
    [20:29]  left wrist xyz+rot6d    state: absolute in the wrist frame
                                     action: frame-space delta vs. the SAME
                                     frame's measured wrist
    [29:49]  right hand qpos
    [49:58]  right wrist xyz+rot6d

Differences from the EEF20 path, all deliberate:

* no dual-X5 base calibration — poses stay in the env-origin world frame and
  are re-expressed in the fixed wrist frame (TE2 head camera by default);
* state and action rows differ (absolute vs. delta wrists), so the stats file
  carries TWO blocks: ``wuji58`` (action rows, used to unnormalize model
  output) and ``wuji58_state`` (state rows, used to normalize proprio);
* ``unify_action`` is not supported: the raw width (58) is the model width.

Everything here is importable without torch so the reader contract can be
unit-tested in a plain numpy/h5py environment.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# XPolicyLab owns the 58-D codec. The checkout root (made importable by its
# XPolicyLab.py shim) sits five levels above this file.
_XPOLICYLAB_ROOT = Path(__file__).resolve().parents[5]
if str(_XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPOLICYLAB_ROOT))

from XPolicyLab.utils import wuji58 as _wuji58

from openwam.dataloader.robodojo_contract import (
    ROBODOJO_SIM_VARIANT,
    ROBODOJO_WUJI58_ACTION_MODE,
    ROBODOJO_WUJI58_CONTRACT_ID,
    ROBODOJO_WUJI58_EMBODIMENT,
    ROBODOJO_WUJI58_SOURCE_FRAME,
    WUJI58_DIM,
    validate_embodiment,
    wuji58_frame_contract,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_WUJI58, STAT_KEYS

WUJI58_STATS_KEY = ROBODOJO_WUJI58_ACTION_MODE  # action-row block
WUJI58_STATE_STATS_KEY = f"{ROBODOJO_WUJI58_ACTION_MODE}_state"  # state-row block (deploy proprio)
DEFAULT_WRIST_FRAME = "te2"

_STATE_POSE_KEYS = ("state/left_ee_poses", "state/right_ee_poses")
_STATE_HAND_KEYS = ("state/left_ee_joint_states", "state/right_ee_joint_states")
_ACTION_POSE_KEYS = ("action/left_ee_poses", "action/right_ee_poses")
_ACTION_HAND_KEYS = ("action/left_ee_joint_states", "action/right_ee_joint_states")
WUJI58_REQUIRED_STATE_DATASETS = (*_STATE_POSE_KEYS, *_STATE_HAND_KEYS)
WUJI58_OPTIONAL_ACTION_DATASETS = (*_ACTION_POSE_KEYS, *_ACTION_HAND_KEYS)
_QUATERNION_ATOL = 1e-6

_ROT6D_IDENTITY = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}


def resolve_wrist_frame(wrist_frame: Any) -> str:
    """Canonical ``wrist_frame`` name (``te2`` default), validated by the codec."""
    name = str(wrist_frame or DEFAULT_WRIST_FRAME).strip().lower()
    _wuji58.WristFrame.from_name(name)  # raises on an unknown frame
    return name


def wuji58_frame_contract_fingerprint(wrist_frame: Any = DEFAULT_WRIST_FRAME) -> str:
    """SHA-256 of the canonical wuji58 frame contract (stored in the stats metadata)."""
    payload = wuji58_frame_contract(resolve_wrist_frame(wrist_frame))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _check_poses(poses: np.ndarray, *, source: str) -> None:
    if poses.dtype.kind not in "fiu" or not np.all(np.isfinite(poses)):
        raise ValueError(f"{source} must contain only finite poses")
    norms = np.linalg.norm(poses[:, 3:7].astype(np.float64), axis=-1)
    bad = ~np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_ATOL)
    if np.any(bad):
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{source} must contain unit wxyz quaternions; row {index} has norm {norms[index]:.8g}")


def _check_hands(hands: np.ndarray, *, source: str) -> None:
    if hands.dtype.kind not in "fiu" or not np.all(np.isfinite(hands)):
        raise ValueError(f"{source} must contain only finite hand joint values")


def episode_has_recorded_actions(handle: h5py.File | h5py.Group) -> bool:
    """True when the episode carries the full ``action/`` group for this mode."""
    return all(key in handle for key in WUJI58_OPTIONAL_ACTION_DATASETS)


def validate_wuji58_state_action_datasets(
    handle: h5py.File | h5py.Group,
    *,
    source: str,
) -> int:
    """Validate the wuji58 state (and optional action) datasets; return T.

    Camera / instruction validation is shared with the EEF path and lives in
    :func:`openwam.dataloader.robodojo.validate_robodojo_episode`.
    """
    for key in WUJI58_REQUIRED_STATE_DATASETS:
        if key not in handle:
            raise KeyError(f"{source}: missing required dataset {key!r}")
    has_actions = episode_has_recorded_actions(handle)
    partial = [key for key in WUJI58_OPTIONAL_ACTION_DATASETS if key in handle]
    if partial and not has_actions:
        raise KeyError(
            f"{source}: action/ group is incomplete for wuji58; present={partial}, "
            f"required={list(WUJI58_OPTIONAL_ACTION_DATASETS)}"
        )
    pose_keys = _STATE_POSE_KEYS + (_ACTION_POSE_KEYS if has_actions else ())
    hand_keys = _STATE_HAND_KEYS + (_ACTION_HAND_KEYS if has_actions else ())
    for key in pose_keys:
        shape = handle[key].shape
        if len(shape) != 2 or shape[1] != _wuji58.POSE_DIM:
            raise ValueError(f"{source}:{key} must have exact shape (T, 7), got {shape}")
    for key in hand_keys:
        shape = handle[key].shape
        if len(shape) != 2 or shape[1] != _wuji58.HAND_DIM:
            raise ValueError(
                f"{source}:{key} must have exact shape (T, {_wuji58.HAND_DIM}) for the Wuji hand, got {shape}"
            )
    lengths = {key: int(handle[key].shape[0]) for key in (*pose_keys, *hand_keys)}
    unique = set(lengths.values())
    if len(unique) != 1 or 0 in unique:
        raise ValueError(f"{source}: all wuji58 state/action datasets must have equal non-zero T; got {lengths}")
    length = next(iter(unique))
    if length < 2:
        raise ValueError(f"{source}: RoboDojo episodes need at least two frames, got T={length}")
    for key in pose_keys:
        _check_poses(np.asarray(handle[key][()]), source=f"{source}:{key}")
    for key in hand_keys:
        _check_hands(np.asarray(handle[key][()]), source=f"{source}:{key}")
    return length


def read_wuji58(
    episode: h5py.File | h5py.Group | str | Path,
    start: int | None = None,
    end: int | None = None,
    *,
    wrist_frame: Any = DEFAULT_WRIST_FRAME,
    embodiment: str = ROBODOJO_WUJI58_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(state58, action58)`` rows ``[start, end)`` of one episode.

    ``state58[t]`` is the measured frame ``t`` (absolute wrists in the wrist
    frame); ``action58[t]`` is the target of frame ``t`` encoded RELATIVE to
    ``state58[t]``'s measured wrist, exactly as ``XPolicyLab.utils.wuji58.trajectory58``
    defines it. Targets are the recorded ``action/`` group when the episode has
    one, otherwise the next measured state (``state[t+1]``, the EEF20 path's
    convention) — in which case the LAST row has no real target and is dropped,
    so the returned arrays have ``T-1`` rows for a full-episode read.

    The stats computation imports and calls this function directly, guaranteeing
    reader/stats conversion parity.
    """
    validate_embodiment(embodiment, variant=variant, action_mode=ROBODOJO_WUJI58_ACTION_MODE)
    if isinstance(episode, (str, Path)):
        with h5py.File(episode, "r") as handle:
            return read_wuji58(handle, start, end, wrist_frame=wrist_frame, embodiment=embodiment, variant=variant)

    frame = _wuji58.WristFrame.from_name(resolve_wrist_frame(wrist_frame))
    length = int(episode[_STATE_POSE_KEYS[0]].shape[0])
    lo, hi, _ = slice(start, end).indices(length)
    if hi <= lo:
        raise IndexError(f"empty wuji58 window [{start}, {end}) for episode length {length}")

    if episode_has_recorded_actions(episode):
        rows = slice(lo, hi)
        state = {
            "left_ee_poses": np.asarray(episode[_STATE_POSE_KEYS[0]][rows]),
            "right_ee_poses": np.asarray(episode[_STATE_POSE_KEYS[1]][rows]),
            "left_ee_joint_states": np.asarray(episode[_STATE_HAND_KEYS[0]][rows]),
            "right_ee_joint_states": np.asarray(episode[_STATE_HAND_KEYS[1]][rows]),
        }
        action = {
            "left_ee_poses": np.asarray(episode[_ACTION_POSE_KEYS[0]][rows]),
            "right_ee_poses": np.asarray(episode[_ACTION_POSE_KEYS[1]][rows]),
            "left_ee_joint_states": np.asarray(episode[_ACTION_HAND_KEYS[0]][rows]),
            "right_ee_joint_states": np.asarray(episode[_ACTION_HAND_KEYS[1]][rows]),
        }
    else:
        # Next-state targets: read one extra row when available so every kept
        # row has a real target.
        hi_target = min(hi + 1, length)
        rows = slice(lo, hi_target)
        full = {
            "left_ee_poses": np.asarray(episode[_STATE_POSE_KEYS[0]][rows]),
            "right_ee_poses": np.asarray(episode[_STATE_POSE_KEYS[1]][rows]),
            "left_ee_joint_states": np.asarray(episode[_STATE_HAND_KEYS[0]][rows]),
            "right_ee_joint_states": np.asarray(episode[_STATE_HAND_KEYS[1]][rows]),
        }
        if full["left_ee_poses"].shape[0] < 2:
            raise IndexError(f"wuji58 window [{start}, {end}) has no real next-state target (episode length {length})")
        state = {key: value[:-1] for key, value in full.items()}
        action = {key: value[1:] for key, value in full.items()}

    state58, action58 = _wuji58.trajectory58(state, action, frame)
    return state58.astype(np.float32, copy=False), action58.astype(np.float32, copy=False)


def iter_episode_wuji58(
    episode_paths: Iterable[str | Path],
    *,
    wrist_frame: Any = DEFAULT_WRIST_FRAME,
    embodiment: str = ROBODOJO_WUJI58_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
):
    """Yield each episode's full ``(state58, action58)`` pair (stats pooling)."""
    for episode_path in episode_paths:
        path = Path(episode_path)
        with h5py.File(path, "r") as handle:
            validate_wuji58_state_action_datasets(handle, source=str(path))
            yield read_wuji58(handle, wrist_frame=wrist_frame, embodiment=embodiment, variant=variant)


def pin_wuji58_rot6d_identity(stats: dict) -> None:
    """In-place: pin the two rot6d blocks of a 58-D stats dict to identity."""
    for key, value in _ROT6D_IDENTITY.items():
        for i in ROT6D_DIMS_WUJI58:
            stats[key][i] = value


def _validated_block(block: Any, *, label: str) -> dict[str, np.ndarray]:
    if not isinstance(block, Mapping):
        raise KeyError(f"{label}: normalization stats must contain a mapping block")
    out: dict[str, np.ndarray] = {}
    for key in STAT_KEYS:
        if key not in block:
            raise KeyError(f"{label} is missing stats vector {key!r}")
        value = np.asarray(block[key])
        if value.shape != (WUJI58_DIM,):
            raise ValueError(f"{label}.{key} must have exact shape ({WUJI58_DIM},), got {value.shape}")
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise ValueError(f"{label}.{key} must contain only finite numeric values")
        out[key] = value.astype(np.float32, copy=False)
    dims = np.asarray(ROT6D_DIMS_WUJI58, dtype=np.int64)
    stale = [
        key
        for key, expected in _ROT6D_IDENTITY.items()
        if not np.array_equal(out[key][dims], np.full(len(dims), expected, dtype=np.float32))
    ]
    if stale:
        raise ValueError(
            f"{label} has non-identity rot6d statistics for {', '.join(stale)} on dimensions "
            "23..28 and 52..57; regenerate the stats file with robodojo_stats_computation"
        )
    if np.any(out["std"] <= 0):
        raise ValueError(f"{label}.std values must all be positive")
    if np.any(out["max"] < out["min"]):
        raise ValueError(f"{label}.max must be >= min")
    if np.any(out["q99"] < out["q01"]):
        raise ValueError(f"{label}.q99 must be >= q01")
    return out


def validate_wuji58_stats_payload(
    payload: Mapping,
    *,
    source: str,
    wrist_frame: Any = DEFAULT_WRIST_FRAME,
    expected_tasks: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate a wuji58 stats payload; return ``(action_stats, state_stats)``."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: normalization stats payload must be a mapping")
    action_stats = _validated_block(payload.get(WUJI58_STATS_KEY), label=f"{source}:{WUJI58_STATS_KEY}")
    state_stats = _validated_block(payload.get(WUJI58_STATE_STATS_KEY), label=f"{source}:{WUJI58_STATE_STATS_KEY}")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}:metadata must be a mapping")
    expected_fingerprint = wuji58_frame_contract_fingerprint(wrist_frame)
    saved = metadata.get("frame_contract_fingerprint")
    if not isinstance(saved, str):
        raise ValueError(f"{source}: metadata.frame_contract_fingerprint is required")
    if saved != expected_fingerprint:
        raise ValueError(
            "RoboDojo wuji58 normalization stats frame contract fingerprint mismatch "
            f"(wrist_frame={resolve_wrist_frame(wrist_frame)!r}): stats={saved}, expected={expected_fingerprint}"
        )
    checks = {
        "variant": ROBODOJO_SIM_VARIANT,
        "embodiment": ROBODOJO_WUJI58_EMBODIMENT,
        "action_mode": ROBODOJO_WUJI58_ACTION_MODE,
        "repr_mode": _wuji58.REPR_MODE,
        "source_frame": ROBODOJO_WUJI58_SOURCE_FRAME,
        "contract_id": ROBODOJO_WUJI58_CONTRACT_ID,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{source}: metadata.{key} must be {expected!r}, got {metadata.get(key)!r}")
    if expected_tasks is not None:
        recorded = metadata.get("tasks")
        wanted = sorted(set(str(task) for task in expected_tasks))
        if not isinstance(recorded, (list, tuple)):
            raise ValueError(f"{source}: metadata.tasks is required")
        if sorted(set(str(task) for task in recorded)) != wanted:
            raise ValueError(f"{source}: metadata.tasks does not match the selected corpus; stats={list(recorded)}, selected={wanted}")
    return action_stats, state_stats


__all__ = [
    "DEFAULT_WRIST_FRAME",
    "WUJI58_OPTIONAL_ACTION_DATASETS",
    "WUJI58_REQUIRED_STATE_DATASETS",
    "WUJI58_STATE_STATS_KEY",
    "WUJI58_STATS_KEY",
    "episode_has_recorded_actions",
    "iter_episode_wuji58",
    "pin_wuji58_rot6d_identity",
    "read_wuji58",
    "resolve_wrist_frame",
    "validate_wuji58_state_action_datasets",
    "validate_wuji58_stats_payload",
    "wuji58_frame_contract_fingerprint",
]
