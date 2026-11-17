"""Pack-and-pour fruit trajectory replay and head-camera blue-side detection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAJECTORY_FILE = Path(__file__).with_name("pack_and_pour_fruit_trajectories.npz")
TRAJECTORY_LENGTH = 90
TRAJECTORY_CHUNK_SIZE = 50


@dataclass
class FruitReplayState:
    prompt_key: str
    blue_side: str | None
    left_score: int
    right_score: int
    trajectory_index: int = 0
    first_inference_done: bool = False
    observation_generation: int = 1
    last_chunk_generation: int = 0
    last_chunk: np.ndarray | None = None

    @property
    def active(self):
        return self.blue_side in {"left", "right"} and self.trajectory_index < TRAJECTORY_LENGTH


def load_trajectories(path=TRAJECTORY_FILE):
    """Load the two embedded 14-D action trajectories and validate their shape."""
    with np.load(path, allow_pickle=False) as data:
        trajectories = {
            "left": np.asarray(data["trajectory_1"], dtype=np.float32),
            "right": np.asarray(data["trajectory_2"], dtype=np.float32),
        }
    for side, trajectory in trajectories.items():
        if trajectory.shape != (TRAJECTORY_LENGTH, 14) or not np.isfinite(trajectory).all():
            raise ValueError(
                f"pack_and_pour_fruit trajectory {side} must be finite with shape "
                f"({TRAJECTORY_LENGTH}, 14), got {trajectory.shape}"
            )
    return trajectories


def _as_rgb_hwc(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError(f"head image must be uint8 RGB with 3 dimensions, got {image.shape} {image.dtype}")
    if image.shape[-1] == 3:
        return np.array(image, copy=False)
    if image.shape[0] == 3:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"head image must have three RGB channels, got {image.shape}")


def detect_blue_side(image):
    """Return ``left``/``right`` from a head image, or None when ambiguous.

    The supplied reference videos vertically concatenate head and wrist views.
    Their aspect ratio is taller than wide, so only the upper half is used for
    calibration. Normal head-camera frames are wider than tall and use the
    complete frame.
    """
    rgb = _as_rgb_hwc(image)
    height, width = rgb.shape[:2]
    if height > width * 0.9:
        rgb = rgb[: height // 2]
    red, green, blue = [rgb[..., index].astype(np.float32) for index in range(3)]
    blue_mask = (
        (blue > 60.0)
        & (blue > red * 1.15 + 8.0)
        & (blue > green * 1.05 + 5.0)
        & ((blue - np.minimum(red, green)) > 20.0)
    )
    midpoint = blue_mask.shape[1] // 2
    left_score = int(blue_mask[:, :midpoint].sum())
    right_score = int(blue_mask[:, midpoint:].sum())
    minimum_pixels = max(20, int(blue_mask.shape[0] * blue_mask.shape[1] * 0.005))
    if max(left_score, right_score) < minimum_pixels:
        return None, left_score, right_score
    if left_score >= right_score * 1.5:
        return "left", left_score, right_score
    if right_score >= left_score * 1.5:
        return "right", left_score, right_score
    return None, left_score, right_score


def next_trajectory_chunk(trajectories, state):
    """Return the next maximum-50 action prefix and advance replay state."""
    if not state.active:
        return None
    trajectory = trajectories[state.blue_side]
    start = state.trajectory_index
    stop = min(start + TRAJECTORY_CHUNK_SIZE, TRAJECTORY_LENGTH)
    state.trajectory_index = stop
    return trajectory[start:stop].copy()
