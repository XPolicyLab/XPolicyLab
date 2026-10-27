"""Which observed RGB frames condition the model.

Indices are raw control steps within one episode; ``t`` is the current step.

    dense   ``num_frames`` frames ending at t, ``stride`` steps apart
            (5 frames, stride 4: t-16, t-12, t-8, t-4, t)
    sparse  the episode's first frame, the frame ``past_offset`` steps ago, and t

Steps before the episode start are clamped to its first frame. Training datasets
and the deployment buffer both take their indices from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DENSE = "dense"
SPARSE = "sparse"
_SPARSE_CONTEXT = "anchor_recent_current"


@dataclass(frozen=True)
class RGBHistory:
    mode: str = DENSE
    num_frames: int = 1
    stride: int = 4
    past_offset: int = 32

    def __post_init__(self) -> None:
        if self.mode not in (DENSE, SPARSE):
            raise ValueError(f"Unknown RGB history mode: {self.mode!r}")
        if self.mode == DENSE and (self.num_frames < 1 or self.stride < 1):
            raise ValueError("Dense RGB history needs num_frames >= 1 and stride >= 1.")
        if self.mode == SPARSE and self.past_offset < 1:
            raise ValueError("Sparse RGB history needs past_offset >= 1.")

    @classmethod
    def dense(cls, num_frames: int = 1, stride: int = 4) -> "RGBHistory":
        return cls(DENSE, num_frames=int(num_frames), stride=int(stride))

    @classmethod
    def sparse(cls, past_offset: int = 32) -> "RGBHistory":
        return cls(SPARSE, past_offset=int(past_offset))

    @classmethod
    def from_config(
        cls, data: Mapping[str, Any], model: Mapping[str, Any] | None = None
    ) -> "RGBHistory":
        """Read the history of a recipe or exported model.

        ``data`` is the training data config (``data.train``, or one real-world
        source); ``model`` is the model config, whose sparse RGB memory settings
        are cross-checked when present.
        """
        # Any rgb_memory section means the sparse-memory model; its mode and offset
        # default to the same values as RGBMemoryConfig.
        memory = (model or {}).get("rgb_memory")
        model_offset = None
        if isinstance(memory, Mapping):
            if memory.get("sparse_context_mode", _SPARSE_CONTEXT) == _SPARSE_CONTEXT:
                model_offset = memory.get("sparse_recent_action_offset", 32)
        data_offset = data.get("sparse_recent_action_offset")
        if data_offset is None and data.get("sparse_frame_encoding") is not None:
            data_offset = 32  # RoboDojoSparseRGBMemoryDataset default
        if data_offset is not None or model_offset is not None:
            if None not in (data_offset, model_offset) and data_offset != model_offset:
                raise ValueError(
                    f"Data and model disagree on the sparse RGB offset: "
                    f"{data_offset} vs {model_offset}."
                )
            return cls.sparse(data_offset if data_offset is not None else model_offset)
        history = cls.dense(
            data.get("rgb_history_num_frames", 1),
            data.get("rgb_history_frame_stride", 4),
        )
        model_frames = (model or {}).get("rgb_history_num_frames")
        if model_frames is not None and int(model_frames) != history.num_frames:
            raise ValueError(
                f"Data and model disagree on the RGB history length: "
                f"{history.num_frames} vs {model_frames}."
            )
        return history

    @property
    def lookback(self) -> int:
        """How many steps back the oldest non-anchor frame can be."""
        if self.mode == SPARSE:
            return self.past_offset
        return (self.num_frames - 1) * self.stride

    def offsets(self) -> tuple[int, ...]:
        """Dense offsets relative to t, oldest first, e.g. (-16, -12, -8, -4, 0)."""
        if self.mode != DENSE:
            raise ValueError("Sparse RGB history has no fixed offsets.")
        return tuple(range(-(self.num_frames - 1) * self.stride, 1, self.stride))

    def indices(self, t: int, *, first: int = 0, last: int | None = None) -> list[int]:
        """Frame indices for current step ``t``, clamped to ``[first, last]``."""
        t = int(t)
        if t < 0:
            raise ValueError("The current step must be non-negative.")
        if self.mode == SPARSE:
            indices = [first, max(first, t - self.past_offset), t]
        else:
            indices = [max(first, t + offset) for offset in self.offsets()]
        if last is not None:
            indices = [min(last, index) for index in indices]
        return indices
