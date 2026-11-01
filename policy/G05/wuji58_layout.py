"""WUJI 58-D hybrid layout helpers for the G05 adapter (numpy only, importable without torch).

G05 consumes per-key state dicts and emits per-key action dicts. For ``tianji_wuji`` in
``action_type=ee`` the keys are the four contiguous blocks of the WUJI SimBench ``wrist58``
vector (``XPolicyLab/utils/wuji58.py``), and the processor configs under
``G05/configs/{data/robodojo_wuji58,task/robodojo_tianji_wuji_ee}.yaml`` slice them from the
flat 58-D LeRobot columns in the same order.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from XPolicyLab.utils import wuji58 as W

#: G05 state/action keys in flat-tensor order, with their codec slices.
WUJI58_KEYS = ("left_hand", "left_wrist", "right_hand", "right_wrist")
WUJI58_SLICES = {
    "left_hand": W.LEFT_HAND,
    "left_wrist": W.LEFT_WRIST,
    "right_hand": W.RIGHT_HAND,
    "right_wrist": W.RIGHT_WRIST,
}
WUJI58_EMBODIMENT = "robodojo_wuji58"


def build_raw_state(obs_state: Mapping[str, Any], frame: W.WristFrame) -> dict[str, np.ndarray]:
    """Standard ``obs["state"]`` → G05 raw state dict of the four 58-D blocks (float32)."""
    vec = W.state58_from_obs(obs_state, frame).astype(np.float32)
    return {key: np.ascontiguousarray(vec[sl]) for key, sl in WUJI58_SLICES.items()}


def chunk_from_parts(parts: Mapping[str, np.ndarray], max_steps: int) -> np.ndarray:
    """G05 per-key action output (each ``[H, D]``) → ``(min(H, max_steps), 58)`` relative chunk."""
    missing = [key for key in WUJI58_KEYS if key not in parts]
    if missing:
        raise KeyError(f"G05 wuji58 action output missing keys {missing}; available={sorted(parts)}")
    horizon = min(int(max_steps), min(int(np.asarray(parts[key]).shape[0]) for key in WUJI58_KEYS))
    chunk = np.zeros((horizon, W.WUJI58_DIM), dtype=np.float32)
    for key, sl in WUJI58_SLICES.items():
        block = np.asarray(parts[key], dtype=np.float32)
        width = sl.stop - sl.start
        if block.ndim != 2 or block.shape[1] != width:
            raise ValueError(f"G05 wuji58 part '{key}' must be [H, {width}], got {block.shape}")
        chunk[:, sl] = block[:horizon]
    return chunk


def decode_chunk(
    parts: Mapping[str, np.ndarray], obs_state: Mapping[str, Any], frame: W.WristFrame, max_steps: int
) -> list[dict[str, np.ndarray]]:
    """G05 per-key output → ``list[dict]`` of absolute standard action dicts, decoded against ``obs_state``."""
    return W.decode_action_chunk58(chunk_from_parts(parts, max_steps), obs_state, frame)


__all__ = [
    "WUJI58_KEYS",
    "WUJI58_SLICES",
    "WUJI58_EMBODIMENT",
    "build_raw_state",
    "chunk_from_parts",
    "decode_chunk",
]
