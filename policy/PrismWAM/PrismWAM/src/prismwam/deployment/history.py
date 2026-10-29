"""RGB history of one episode at inference time."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import torch

from prismwam.rgb_history import RGBHistory


class RGBHistoryBuffer:
    """Collects the observed RGB frames of an episode and builds the history tensor.

    Append one frame per executed action (the native control rate), not only at
    replans; otherwise the frame spacing differs from training.
    """

    def __init__(self, history: RGBHistory):
        self.history = history
        self.frames: deque[torch.Tensor] = deque(maxlen=history.lookback + 1)
        self.first: torch.Tensor | None = None
        self.step = -1

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RGBHistoryBuffer":
        """Build from an exported model's (or training run's) config."""
        train = config["data"]["train"]
        sources = train.get("sources")
        if sources is not None:
            if len(sources) != 1:
                raise ValueError(
                    "Mixed real-robot config: export one model folder per task (--task-name)."
                )
            train = sources[0]
        return cls(RGBHistory.from_config(train, config.get("model")))

    def reset(self) -> None:
        self.frames.clear()
        self.first = None
        self.step = -1

    def append(self, frame: torch.Tensor, *, step: int) -> None:
        if step != self.step + 1:
            raise ValueError(
                "Append exactly one observation per executed action, starting at step 0."
            )
        if frame.ndim != 3 or frame.shape[0] != 3 or not frame.is_floating_point():
            raise ValueError("Expected normalized CHW RGB.")
        if not torch.isfinite(frame).all():
            raise ValueError("RGB contains nonfinite values.")
        if self.first is not None and self.first.shape != frame.shape:
            raise ValueError("Image size changed within the episode.")
        owned = frame.detach().cpu().clone()
        if self.first is None:
            self.first = owned
        self.frames.append(owned)
        self.step = step

    def indices(self) -> list[int]:
        if self.step < 0:
            raise RuntimeError(
                "Append the first observation before reading the history."
            )
        return self.history.indices(self.step)

    def tensor(self) -> torch.Tensor:
        """``[1, 3, T, H, W]`` history for ``policy.predict(rgb_history_images=...)``."""
        start = self.step - len(self.frames) + 1
        frames = [
            self.first if index == 0 else self.frames[index - start]
            for index in self.indices()
        ]
        return torch.stack(frames, dim=1)[None]
