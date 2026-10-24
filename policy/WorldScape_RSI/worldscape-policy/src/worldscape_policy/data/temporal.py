"""Native temporal indexing used by the DreamZero/WSP data path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalPackingIndices:
    """Aligned indices for one language-consistent temporal sample."""

    video: np.ndarray
    action: np.ndarray
    state: np.ndarray
    anchors: np.ndarray

    def validate(
        self,
        max_chunk_size: int,
        *,
        action_steps_per_chunk: int = 24,
        video_frames_per_chunk: int = 8,
    ) -> None:
        chunks = len(self.anchors)
        if not 0 < chunks <= max_chunk_size:
            raise ValueError("temporal packing must contain at least one chunk")
        if self.video.shape != (chunks * video_frames_per_chunk + 1,):
            raise ValueError(
                "video indices must contain the configured frames per chunk "
                "plus endpoint"
            )
        if self.action.shape != (chunks * action_steps_per_chunk,):
            raise ValueError(
                "action indices do not match the configured steps per chunk"
            )
        if self.state.shape != (chunks,):
            raise ValueError("state indices must contain one anchor per chunk")


@dataclass(frozen=True)
class LanguageTemporalPacker:
    """Build language-consistent 24- or 48-action causal blocks."""

    max_chunk_size: int = 4
    chunk_size: int = 24
    video_offsets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if self.chunk_size not in {24, 48}:
            raise ValueError("native temporal packing requires chunk_size 24 or 48")
        observation_interval = self.chunk_size // 8
        expected_offsets = tuple(
            range(0, self.chunk_size, observation_interval)
        )
        if self.video_offsets is None:
            object.__setattr__(self, "video_offsets", expected_offsets)
        elif tuple(self.video_offsets) != expected_offsets:
            raise ValueError(
                "video_offsets must sample eight uniformly spaced frames "
                f"for chunk_size={self.chunk_size}: {expected_offsets}"
            )

    def indices(
        self,
        anchor_index: int,
        language: np.ndarray,
        *,
        trajectory_ids: np.ndarray | None = None,
    ) -> TemporalPackingIndices:
        labels = np.asarray(language).reshape(-1)
        length = len(labels)
        if length == 0:
            raise ValueError("language trajectory cannot be empty")
        anchor = int(np.clip(anchor_index, 0, length - 1))
        trajectories = (
            np.zeros(length, dtype=np.int64)
            if trajectory_ids is None
            else np.asarray(trajectory_ids).reshape(-1)
        )
        if len(trajectories) != length:
            raise ValueError("trajectory_ids must match language length")
        target_language = labels[anchor]
        target_trajectory = trajectories[anchor]

        def compatible(candidate: int) -> bool:
            # Include the endpoint at candidate + chunk_size.
            return (
                candidate >= 0
                and candidate + self.chunk_size < length
                and np.all(
                    labels[candidate : candidate + self.chunk_size + 1]
                    == target_language
                )
                and np.all(
                    trajectories[candidate : candidate + self.chunk_size + 1]
                    == target_trajectory
                )
            )

        anchors: list[int] = []
        if compatible(anchor):
            anchors.append(anchor)
        step = 1
        backward_done = not anchors
        forward_done = not anchors
        while len(anchors) < self.max_chunk_size and not (
            backward_done and forward_done
        ):
            backward = anchor - self.chunk_size * step
            if not backward_done:
                if backward < 0 or not compatible(backward):
                    backward_done = True
                else:
                    anchors.append(backward)
            if len(anchors) >= self.max_chunk_size:
                break
            forward = anchor + self.chunk_size * step
            if not forward_done:
                if forward >= length or not compatible(forward):
                    forward_done = True
                else:
                    anchors.append(forward)
            step += 1
        if not anchors:
            raise ValueError(
                "anchor has no complete "
                f"{self.chunk_size}-step chunk within its "
                "language/trajectory boundary"
            )

        anchor_array = np.asarray(sorted(set(anchors)), dtype=np.int64)
        action = np.concatenate(
            [np.arange(item, item + self.chunk_size, dtype=np.int64) for item in anchor_array]
        )
        assert self.video_offsets is not None
        video = np.concatenate(
            [
                item + np.asarray(self.video_offsets, dtype=np.int64)
                for item in anchor_array
            ]
        )
        video = np.append(
            video,
            anchor_array[-1] + self.chunk_size,
        ).astype(np.int64)
        result = TemporalPackingIndices(
            video=video,
            action=action,
            state=anchor_array.copy(),
            anchors=anchor_array,
        )
        result.validate(
            self.max_chunk_size,
            action_steps_per_chunk=self.chunk_size,
            video_frames_per_chunk=len(self.video_offsets),
        )
        return result


@dataclass(frozen=True)
class ContextSampler:
    """Select native visual context with legacy-compatible index arithmetic."""

    mode: str = "none"
    length: int = 50
    ctx_head_only: bool = False

    def __post_init__(self) -> None:
        mode = self.mode.lower()
        if mode not in {"none", "last", "uniform"}:
            raise ValueError("context mode must be none, last, or uniform")
        if self.length <= 0:
            raise ValueError("context length must be positive")
        if mode == "last" and self.length != 1:
            raise ValueError("last context mode requires length=1")

    def indices(self, context_length: int) -> np.ndarray:
        if self.mode.lower() == "none":
            return np.empty(0, dtype=np.int64)
        if context_length <= 0:
            raise ValueError(
                f"context mode {self.mode!r} requires at least one real context frame"
            )
        if self.mode.lower() == "last":
            return np.asarray([context_length - 1], dtype=np.int64)
        if context_length == 1:
            return np.zeros(self.length, dtype=np.int64)
        return np.linspace(0, context_length - 1, self.length).round().astype(np.int64)

    def sample(self, frames: np.ndarray) -> np.ndarray | None:
        values = np.asarray(frames)
        indices = self.indices(len(values))
        return None if len(indices) == 0 else values[indices].copy()


@dataclass(frozen=True)
class VLMHistorySampler:
    """Fixed-length history sampling with deterministic left-edge padding."""

    num_frames: int = 8
    stride: int = 24
    window: int = 192

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.stride <= 0 or self.window <= 0:
            raise ValueError("history num_frames, stride, and window must be positive")

    def indices(self, anchor_index: int, trajectory_length: int) -> np.ndarray:
        if trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        anchor = int(anchor_index)
        max_back = min(self.window, self.stride * self.num_frames)
        history = np.arange(anchor - max_back, anchor, self.stride, dtype=np.int64)
        if len(history) > self.num_frames:
            history = history[-self.num_frames :]
        if len(history) < self.num_frames:
            pad_value = history[0] if len(history) else anchor
            history = np.concatenate(
                [
                    np.full(self.num_frames - len(history), pad_value, dtype=np.int64),
                    history,
                ]
            )
        return np.clip(history, 0, trajectory_length - 1).astype(np.int64)

    def sample(self, frames: np.ndarray, anchor_index: int) -> np.ndarray:
        values = np.asarray(frames)
        return values[self.indices(anchor_index, len(values))].copy()


__all__ = [
    "ContextSampler",
    "LanguageTemporalPacker",
    "TemporalPackingIndices",
    "VLMHistorySampler",
]
