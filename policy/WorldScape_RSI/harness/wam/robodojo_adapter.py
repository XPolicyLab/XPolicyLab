"""Map decoded RoboDojo ARX X5 observations/actions to the online protocol.

This module only uses NumPy. It does not initialize Isaac, decode JPEG buffers,
connect to a server, or execute actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

CAMERA_MAP = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}
STATE_FIELDS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


@dataclass(frozen=True)
class ObservationFrame:
    """Owned copies of one control observation, with RGB camera arrays."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    instruction: str
    env_idx: int | None


def capture_frame(obs: Mapping[str, Any]) -> ObservationFrame:
    """Call immediately after TASK_ENV.get_obs() to snapshot camera buffers."""
    if not isinstance(obs, Mapping):
        raise TypeError("Pass one observation mapping, not a get_obs_batch() list")
    vision, state = obs.get("vision"), obs.get("state")
    if not isinstance(vision, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Observation must contain vision and state mappings")
    images = {}
    for name, camera in CAMERA_MAP.items():
        value = vision[camera]["color"]
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
                or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1):
            raise ValueError(
                f"vision.{camera}.color must be decoded uint8 RGB [H,W,3]. "
                "For encoded XPolicyLab observations use its decode_obs_images helper first."
            )
        images[name] = np.ascontiguousarray(value).copy()
    values = []
    for key, width in STATE_FIELDS:
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"state.{key} must contain {width} finite values")
        values.append(value)
    instruction = obs.get("instruction")
    if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
        instruction = instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Observation instruction must be a nonempty string")
    env_idx = obs.get("env_idx")
    if env_idx is not None:
        if isinstance(env_idx, bool) or not isinstance(env_idx, (int, np.integer)) or env_idx < 0:
            raise ValueError("env_idx must be a nonnegative integer")
        env_idx = int(env_idx)
    return ObservationFrame(images, np.concatenate(values), instruction, env_idx)


def build_inputs(frames: Sequence[ObservationFrame]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pack one initial frame or 24 consecutive post-action frames into a request.

    The caller owns episode boundaries: never mix frames across resets, even
    when the same environment ID and instruction are reused.
    """
    if len(frames) not in (1, 24):
        raise ValueError("Provide 1 initial frame or exactly 24 consecutive post-action frames")
    if any(frame.env_idx != frames[0].env_idx for frame in frames):
        raise ValueError("Cannot combine observations from different environments")
    if any(frame.instruction != frames[0].instruction for frame in frames):
        raise ValueError("Instruction changed within a chunk; reset for the new task")
    images = {name: np.stack([frame.images[name] for frame in frames]) for name in CAMERA_MAP}
    return images, frames[-1].state.copy()


def action_to_robodojo(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split one absolute [14] target for TASK_ENV.take_action; do not execute it."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("One action must contain 14 finite values")
    result, offset = {}, 0
    for key, width in STATE_FIELDS:
        result[key] = value[offset:offset + width].copy()
        offset += width
    return result
