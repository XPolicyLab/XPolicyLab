from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from worldscape_policy.data.schema import EventSample


@dataclass(frozen=True)
class EventChunkSampler:
    """Sample a bounded contiguous window from one event."""

    max_chunks: int

    def __post_init__(self) -> None:
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

    def sample(
        self,
        chunks: Sequence[EventSample],
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[EventSample, ...]:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        event_ids = {chunk.event_id for chunk in chunks}
        if len(event_ids) != 1:
            raise ValueError("EventChunkSampler only accepts chunks from one event")
        count = min(len(chunks), self.max_chunks)
        if len(chunks) == count:
            start = 0
        else:
            generator = rng or np.random.default_rng()
            start = int(generator.integers(0, len(chunks) - count + 1))
        return tuple(chunks[start : start + count])


__all__ = ["EventChunkSampler"]
