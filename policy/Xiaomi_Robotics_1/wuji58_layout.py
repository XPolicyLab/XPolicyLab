"""WUJI 58-D hybrid layout for the Xiaomi_Robotics_1 (XR-1) adapter.

XR-1 works on a fixed 60-slot state and action space (``STATE_DIM`` /
``ACTION_DIM`` in ``mibot/utils/io.py``). The arx_x5 path fills those slots
with the ``compose_state`` / ``ACTION_PARTS`` layouts; this module defines the
alternative layout used for the ``tianji_wuji`` robot (two 20-DOF Wuji hands
driven in end-effector space):

    slots [0:58]   the WUJI 58-D hybrid vector (``XPolicyLab.utils.wuji58``)
                   [L hand 20 | L wrist 9 | R hand 20 | R wrist 9]
    slots [58:60]  zero, masked out

State and action share the same slot layout here (unlike arx_x5):

* state   — hands: absolute joints; wrists: absolute ``xyz + rot6d`` pose in the
            wrist frame (``wrist_frame`` in deploy.yml, ``te2`` by default).
* action  — hands: absolute joint targets; wrists: frame-space delta of the
            target vs. the wrist observed at the START of the chunk, which is
            the XR-1 convention (every ``ACTION_PARTS`` delta is w.r.t. the
            observed pose) and exactly what ``decode_action_chunk58`` inverts.

The arx_x5 path is untouched: the vendored ``mibot`` package only grows a
small branch in ``JsonDataset`` that delegates to :func:`training_sample`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from XPolicyLab.utils import wuji58 as W
except ImportError:  # launched from the vendored xr1 tree without the parent on sys.path
    _IMPORT_ROOT = str(Path(__file__).resolve().parents[3])
    if _IMPORT_ROOT not in sys.path:
        sys.path.insert(0, _IMPORT_ROOT)
    from XPolicyLab.utils import wuji58 as W

SLOT_DIM = 60
WUJI_SLOTS = slice(0, W.WUJI58_DIM)
LAYOUT_NAME = "wuji58"

# Packed slots holding wrist deltas (gaussian over moving frames) and hand
# targets (gaussian over every frame) in the action statistics.
WRIST_SLOTS = (W.LEFT_WRIST, W.RIGHT_WRIST)
HAND_SLOTS = (W.LEFT_HAND, W.RIGHT_HAND)

# Statistics rules, mirroring scripts/process_data.py for the arx_x5 layout.
MOTION_EPS = 1e-4
MIN_MOTION_FRAMES = 10
MIN_STD = 1e-6

STATE_KEYS = W.STATE_KEYS


# --- robot resolution --------------------------------------------------------


def resolve_robot_action_dim_info(env_cfg_type: str) -> dict:
    """``get_robot_action_dim_info`` with the shared shell registry as fallback.

    The Python helper reads ``<parent>/env_cfg``, which only exists inside a
    benchmark workspace; a standalone checkout falls back to the registry
    ``train.sh`` uses (``utils/robot/_robot_info.json``, keyed by env_cfg_type).
    """
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    try:
        return get_robot_action_dim_info(env_cfg_type)
    except (FileNotFoundError, KeyError, OSError, TypeError):
        registry = Path(W.__file__).resolve().parent / "robot" / "_robot_info.json"
        with open(registry, "r", encoding="utf-8") as handle:
            table = json.load(handle)
        if env_cfg_type not in table:
            raise KeyError(
                f"[Xiaomi_Robotics_1] env_cfg_type {env_cfg_type!r} is registered "
                f"neither in <parent>/env_cfg nor in {registry}"
            )
        return table[env_cfg_type]


def is_wuji58_run(env_cfg_type: str, action_type: str) -> bool:
    return W.is_wuji58(resolve_robot_action_dim_info(env_cfg_type), action_type)


# --- inference side ----------------------------------------------------------


def pad_to_slots(vec58) -> np.ndarray:
    """``(..., 58)`` -> ``(..., 60)`` with zero padding slots."""
    arr = np.asarray(vec58, dtype=np.float32)
    if arr.shape[-1] != W.WUJI58_DIM:
        raise ValueError(f"expected a trailing dim of {W.WUJI58_DIM}, got {arr.shape}")
    out = np.zeros(arr.shape[:-1] + (SLOT_DIM,), dtype=np.float32)
    out[..., WUJI_SLOTS] = arr
    return out


def compose_state58(obs_state: Mapping[str, Any], frame: W.WristFrame) -> np.ndarray:
    """Observation ``obs["state"]`` -> packed ``(1, 60)`` state."""
    return pad_to_slots(W.state58_from_obs(obs_state, frame))[None]


def action_mask58(action_length: int, temporal_mask=None) -> np.ndarray:
    """``(action_length, 60)`` mask: ones on the 58 WUJI slots, zero on padding."""
    temporal = (
        np.ones(action_length, dtype=np.int32)
        if temporal_mask is None
        else np.asarray(temporal_mask, dtype=np.int32)
    )
    mask = np.zeros((action_length, SLOT_DIM), dtype=np.int32)
    mask[:, WUJI_SLOTS] = temporal[:, None]
    return mask


def decode_actions58(actions, obs_state: Mapping[str, Any], frame: W.WristFrame):
    """``(T, 60)`` unnormalized chunk -> list of standard absolute action dicts."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != SLOT_DIM:
        raise ValueError(f"expected a (T, {SLOT_DIM}) chunk, got {arr.shape}")
    # XR-1 forms every delta of a chunk against the wrist observed at the chunk start
    # (``chunk_actions``), so decoding must use the same reference, not the default
    # per-frame/iterative one the official converters' data uses.
    return W.decode_action_chunk58(arr[:, WUJI_SLOTS], obs_state, frame, reference="chunk_start")


def snapshot_obs_state(obs_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy of the four standard keys the chunk is decoded against."""
    return {key: np.array(obs_state[key], dtype=np.float64).reshape(-1) for key in STATE_KEYS}


# --- dataset side: JSON encoding ---------------------------------------------


def encode_episode(state: Mapping[str, Any], action: Mapping[str, Any], frame: W.WristFrame):
    """Trajectory ``state/`` and ``action/`` groups -> ``(state58, target58)``.

    Both are ``(T, 58)`` with ABSOLUTE wrists in ``frame`` (``encode_wrist_state``):
    ``state58`` is what the policy observes, ``target58`` the absolute target per
    frame. Relative wrist deltas are formed per chunk at load time
    (:func:`chunk_actions`), so the stored JSON stays frame-independent.
    """
    s_lp = _rows(state, "left_ee_poses", W.POSE_DIM)
    s_lh = _rows(state, "left_ee_joint_states", W.HAND_DIM)
    s_rp = _rows(state, "right_ee_poses", W.POSE_DIM)
    s_rh = _rows(state, "right_ee_joint_states", W.HAND_DIM)
    a_lp = _rows(action, "left_ee_poses", W.POSE_DIM)
    a_lh = _rows(action, "left_ee_joint_states", W.HAND_DIM)
    a_rp = _rows(action, "right_ee_poses", W.POSE_DIM)
    a_rh = _rows(action, "right_ee_joint_states", W.HAND_DIM)
    lengths = {arr.shape[0] for arr in (s_lp, s_lh, s_rp, s_rh, a_lp, a_lh, a_rp, a_rh)}
    if len(lengths) != 1:
        raise ValueError(f"state/action lengths disagree: {sorted(lengths)}")
    T = lengths.pop()
    state58 = np.empty((T, W.WUJI58_DIM), dtype=np.float64)
    target58 = np.empty((T, W.WUJI58_DIM), dtype=np.float64)
    for t in range(T):
        state58[t] = W.pack58(
            s_lh[t], W.encode_wrist_state(s_lp[t], frame), s_rh[t], W.encode_wrist_state(s_rp[t], frame)
        )
        target58[t] = W.pack58(
            a_lh[t], W.encode_wrist_state(a_lp[t], frame), a_rh[t], W.encode_wrist_state(a_rp[t], frame)
        )
    return state58, target58


def _rows(group: Mapping[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in group:
        raise KeyError(f"wuji58 encoding needs '{key}'")
    arr = np.asarray(group[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"'{key}' must be (T, {dim}), got {arr.shape}")
    return arr


def _wrist_delta(target9: np.ndarray, current9: np.ndarray) -> np.ndarray:
    """Frame-space delta between two absolute ``xyz+rot6d`` wrists (same math as
    ``encode_wrist_relative``, which ``decode_wrist_relative`` inverts)."""
    R_t = W.rot6d_to_matrix(target9[3:])
    R_c = W.rot6d_to_matrix(current9[3:])
    return np.concatenate([target9[:3] - current9[:3], W.matrix_to_rot6d(R_t @ R_c.T)])


def chunk_actions(state58_t, target58_chunk) -> np.ndarray:
    """``(58,)`` state at the chunk start + ``(k, 58)`` absolute targets -> ``(k, 58)`` actions.

    Hands pass through as absolute targets; wrists become deltas w.r.t. the
    wrist observed at the chunk start, which is what ``decode_action_chunk58``
    inverts against the observation the chunk was predicted from.
    """
    cur = np.asarray(state58_t, dtype=np.float64).reshape(W.WUJI58_DIM)
    tgt = np.asarray(target58_chunk, dtype=np.float64)
    if tgt.ndim != 2 or tgt.shape[1] != W.WUJI58_DIM:
        raise ValueError(f"expected (k, {W.WUJI58_DIM}) targets, got {tgt.shape}")
    out = tgt.copy()
    for k in range(tgt.shape[0]):
        out[k, W.LEFT_WRIST] = _wrist_delta(tgt[k, W.LEFT_WRIST], cur[W.LEFT_WRIST])
        out[k, W.RIGHT_WRIST] = _wrist_delta(tgt[k, W.RIGHT_WRIST], cur[W.RIGHT_WRIST])
    return out


# --- dataset side: training sample (called from JsonDataset) ----------------


def training_sample(traj: Mapping[str, Any], frame: int, steps: int, action_length: int):
    """Raw (unnormalized) packed tensors for one training sample.

    Returns ``(state (1, 60), action (action_length, 60), action_mask (action_length, 60))``.
    The tail beyond ``steps`` repeats the last row and is masked, matching the
    arx_x5 loader's ``_pad`` / ``_mask`` behaviour.
    """
    state58 = np.asarray(traj["proprios"]["wuji58_state"], dtype=np.float64)
    target58 = np.asarray(traj["actions"]["wuji58_target"], dtype=np.float64)
    if steps <= 0 or frame + steps > target58.shape[0]:
        raise IndexError(f"invalid wuji58 sample: frame={frame} steps={steps} T={target58.shape[0]}")
    action = chunk_actions(state58[frame], target58[frame : frame + steps])
    if steps < action_length:
        action = np.concatenate([action, np.repeat(action[-1:], action_length - steps, axis=0)], axis=0)
    temporal = np.zeros(action_length, dtype=np.int32)
    temporal[:steps] = 1
    return (
        pad_to_slots(state58[frame])[None],
        pad_to_slots(action),
        action_mask58(action_length, temporal),
    )


# --- dataset side: statistics ------------------------------------------------


def episode_action_samples(state58, target58, action_length: int) -> np.ndarray:
    """``(n, action_length, 60)`` packed actions over the whole-horizon frames."""
    state58 = np.asarray(state58, dtype=np.float64)
    target58 = np.asarray(target58, dtype=np.float64)
    num_frames = state58.shape[0]
    frames = range(0, max(0, num_frames - action_length + 1))
    out = np.zeros((len(frames), action_length, SLOT_DIM), dtype=np.float64)
    for row, frame in enumerate(frames):
        out[row, :, WUJI_SLOTS] = chunk_actions(state58[frame], target58[frame : frame + action_length])
    return out


def episode_state_samples(state58, action_length: int) -> np.ndarray:
    """``(n, 60)`` packed states over the same whole-horizon window."""
    state58 = np.asarray(state58, dtype=np.float64)
    keep = max(0, state58.shape[0] - action_length + 1)
    return pad_to_slots(state58[:keep]).astype(np.float64)


def compute_stats58(action_samples, state_samples):
    """Normalization statistics for the wuji58 slot layout.

    * wrist action slots: gaussian per (step, dim) over moving frames only
      (``|x| >= MOTION_EPS``), identity fallback below ``MIN_MOTION_FRAMES``;
    * hand action slots: gaussian per (step, dim) over every frame (absolute
      targets are not centred on zero, so no motion filter applies);
    * state: 1st/99th percentiles over the 58 slots; constant slots and the
      two padding slots are written as 0/0, which ``validate_quantiles`` treats
      as padding and ``normalize_quantile`` passes through as zero.

    Returns ``(mean (L, 60), std (L, 60), q01 (1, 60), q99 (1, 60))``.
    """
    actions = np.asarray(action_samples, dtype=np.float64)
    states = np.asarray(state_samples, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[2] != SLOT_DIM:
        raise ValueError(f"action samples must be (N, L, {SLOT_DIM}), got {actions.shape}")
    if states.ndim != 2 or states.shape[1] != SLOT_DIM:
        raise ValueError(f"state samples must be (N, {SLOT_DIM}), got {states.shape}")
    if actions.shape[0] == 0:
        raise ValueError("no action samples")

    steps = actions.shape[1]
    mean = np.zeros((steps, SLOT_DIM), dtype=np.float64)
    std = np.ones((steps, SLOT_DIM), dtype=np.float64)

    for slc in WRIST_SLOTS:
        for dim in range(slc.start, slc.stop):
            for step in range(steps):
                values = actions[:, step, dim]
                moving = values[np.abs(values) >= MOTION_EPS]
                if moving.size < MIN_MOTION_FRAMES:
                    continue
                mean[step, dim] = moving.mean()
                std[step, dim] = max(moving.std(), MIN_STD)
    for slc in HAND_SLOTS:
        mean[:, slc] = actions[:, :, slc].mean(axis=0)
        std[:, slc] = np.maximum(actions[:, :, slc].std(axis=0), MIN_STD)

    q01 = np.zeros((1, SLOT_DIM), dtype=np.float64)
    q99 = np.zeros((1, SLOT_DIM), dtype=np.float64)
    q01[0, WUJI_SLOTS] = np.percentile(states[:, WUJI_SLOTS], 1, axis=0)
    q99[0, WUJI_SLOTS] = np.percentile(states[:, WUJI_SLOTS], 99, axis=0)
    degenerate = q99 <= q01
    q01[degenerate] = 0.0
    q99[degenerate] = 0.0
    return mean, std, q01, q99


__all__ = [
    "SLOT_DIM",
    "WUJI_SLOTS",
    "LAYOUT_NAME",
    "STATE_KEYS",
    "resolve_robot_action_dim_info",
    "is_wuji58_run",
    "pad_to_slots",
    "compose_state58",
    "action_mask58",
    "decode_actions58",
    "snapshot_obs_state",
    "encode_episode",
    "chunk_actions",
    "training_sample",
    "episode_action_samples",
    "episode_state_samples",
    "compute_stats58",
]
