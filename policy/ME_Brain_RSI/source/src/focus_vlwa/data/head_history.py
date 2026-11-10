"""Episode-grid sampling for full-frame head history."""

import math
from collections import deque

import numpy as np

HISTORY_FRAMES = 20
HISTORY_INTERVAL = 25
RUNTIME_FPS = 25
HISTORY_FPS = 1.0
IMAGE_SIZE = 224
POOL_SIZE = 4
TOKENS_PER_FRAME = POOL_SIZE**2
HISTORY_TOKENS = HISTORY_FRAMES * TOKENS_PER_FRAME


def history_indices(frame_index):
    """Return completed episode-grid slots, oldest first, excluding the current slot."""
    if int(frame_index) != frame_index or frame_index < 0:
        raise ValueError("Frame index must be a nonnegative integer")
    anchor = int(frame_index) // HISTORY_INTERVAL * HISTORY_INTERVAL
    indices = anchor - np.arange(HISTORY_FRAMES, 0, -1, dtype=np.int64) * HISTORY_INTERVAL
    return indices, indices >= 0


def history_offsets() -> list[float]:
    """Return nominal seconds relative to a grid anchor, not an arbitrary frame."""
    return [-slot / HISTORY_FPS for slot in range(HISTORY_FRAMES, 0, -1)]


def frame_offsets(source_fps: float = RUNTIME_FPS) -> list[int]:
    """Validate the online control-rate contract and return grid-anchor offsets."""
    if not math.isfinite(source_fps) or source_fps != RUNTIME_FPS:
        raise ValueError("Episode-grid online history requires source_fps=25")
    return [-slot * HISTORY_INTERVAL for slot in range(HISTORY_FRAMES, 0, -1)]


def full_frame(image) -> np.ndarray:
    """Convert RGB pixels and resize with aspect-preserving padding."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("History must contain main-view RGB frames, not multi-camera crops")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError("History frames must have three RGB channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating history pixels must be finite and in [0, 1]")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("History pixels must be uint8 or [0, 1] floats")
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        from focus_vlwa.data.image_tools import resize_with_pad
        image = np.asarray(resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE))
    return image


def pack_history(images=(), mask=None) -> dict[str, np.ndarray]:
    """Left-pad oldest-first images; callers must supply the episode-grid slots."""
    images = list(images)
    if len(images) > HISTORY_FRAMES:
        raise ValueError(f"At most {HISTORY_FRAMES} main-view history frames are supported")
    valid = np.ones(len(images), dtype=bool) if mask is None else np.asarray(mask)
    if valid.shape != (len(images),) or not np.isin(valid, [0, 1]).all():
        raise ValueError("History mask must have one binary value per frame")
    frames = np.zeros((HISTORY_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    padded_mask = np.zeros(HISTORY_FRAMES, dtype=np.float32)
    start = HISTORY_FRAMES - len(images)
    for i, image in enumerate(images):
        if valid[i]:
            frames[start + i] = full_frame(image)
            padded_mask[start + i] = 1
    return {"hist_images": frames, "hist_mask": padded_mask}


class SplitLeRobotHeadHistory:
    """Separate queried episode-grid frames from the exact current head image."""

    def __init__(self, image_key: str):
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        images = data[self.image_key]
        padding = np.asarray(data[f"{self.image_key}_is_pad"], dtype=bool)
        if len(images) != HISTORY_FRAMES + 1 or padding.shape != (HISTORY_FRAMES + 1,):
            raise ValueError("Expected 20 historical head frames followed by the current frame")
        if padding[-1]:
            raise ValueError("The current head frame must be valid")
        output = dict(data)
        output[self.image_key] = images[-1]
        output.update(pack_history(images[:-1], ~padding[:-1]))
        return output


class HeadHistoryBuffer:
    """Retain the 20 history slots plus the current grid frame; reset each episode."""

    def __init__(self, source_fps: float = RUNTIME_FPS):
        frame_offsets(source_fps)
        self.reset()

    def reset(self) -> None:
        self.frames = deque(maxlen=HISTORY_FRAMES + 1)
        self.next_step = 0

    def observe(self, frame_index: int, image) -> None:
        if frame_index != self.next_step:
            raise ValueError("Observe every consecutive control step starting at zero; reset for a new episode")
        if frame_index % HISTORY_INTERVAL == 0:
            self.frames.append((frame_index, full_frame(image).copy()))
        self.next_step += 1

    def pack(self, frame_index: int) -> dict[str, np.ndarray]:
        if frame_index != self.next_step - 1:
            raise ValueError("Observe the current frame exactly once before packing history")
        indices, valid = history_indices(frame_index)
        frames = dict(self.frames)
        selected = []
        for index, exists in zip(indices, valid, strict=True):
            if exists and int(index) not in frames:
                raise RuntimeError(f"Missing history frame {index}; observe every control step")
            selected.append(frames[int(index)] if exists else None)
        return pack_history(selected, valid)
