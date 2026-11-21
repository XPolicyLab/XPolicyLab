from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistorySampler:
    """Sample strided observation history ending at an anchor frame."""

    num_frames: int = 8
    stride: int = 24

    def __post_init__(self) -> None:
        if self.num_frames < 0:
            raise ValueError("num_frames cannot be negative")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def sample(
        self, head_frames: np.ndarray, *, anchor_index: int | None = None
    ) -> np.ndarray:
        frames = np.asarray(head_frames)
        if frames.ndim != 4:
            raise ValueError("head_frames must have shape [T,H,W,C]")
        if self.num_frames == 0:
            return frames[:0].copy()
        if frames.shape[0] == 0:
            raise ValueError("head_frames cannot be empty")
        anchor = frames.shape[0] - 1 if anchor_index is None else anchor_index
        if not 0 <= anchor < frames.shape[0]:
            raise IndexError("anchor_index is outside head_frames")
        indices = np.arange(
            anchor - self.stride * (self.num_frames - 1),
            anchor + 1,
            self.stride,
            dtype=np.int64,
        )
        return frames[indices[indices >= 0]].copy()


__all__ = ["HistorySampler"]
