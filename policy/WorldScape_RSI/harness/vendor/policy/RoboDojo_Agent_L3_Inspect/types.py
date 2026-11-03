"""RoboDojo L3 inspect-inspired adapter types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# The arx_x5 websocket action dict, in channel order.
JOINT_CHANNELS: tuple[tuple[str, int], ...] = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


@dataclass(frozen=True)
class Observation:
    """One environment observation as the policy sees it."""

    images: Mapping[str, np.ndarray]
    state: Mapping[str, np.ndarray]
    instruction: str | None = None
    step: int = 0
    remaining_steps: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """One environment action, already in websocket channel form."""

    data: Mapping[str, np.ndarray]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionChunk:
    """Open-loop actions executed before the policy is consulted again."""

    actions: Sequence[Action]
    reasoning: str | None = None
    latency_s: float | None = None
    control_hz: float | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("An action chunk must contain at least one action.")

    def __len__(self) -> int:
        return len(self.actions)


@runtime_checkable
class Policy(Protocol):
    """What the RoboDojo harness drives for the L3 inspect condition."""

    def reset(self) -> None:
        """Clear per-episode state before the first observation."""
        ...

    def act(self, observation: Observation) -> ActionChunk:
        """Return a non-empty open-loop action chunk for this observation."""
        ...


class ActionSpace:
    """Joint channel layout used to encode/decode websocket action dicts."""

    def __init__(self, channels: Sequence[tuple[str, int]] = JOINT_CHANNELS) -> None:
        self.channels = tuple(channels)

    @property
    def width(self) -> int:
        return sum(size for _, size in self.channels)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.channels)

    def offsets(self) -> dict[str, tuple[int, int]]:
        """Where each channel sits in the flat vector, as ``(start, size)``.

        Per-dimension declarations -- labels, bounds, per-step limits -- are
        indexed by flat position, so anything reading one channel's slice out
        of them has to locate it from the layout rather than count by hand.
        """
        located: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, size in self.channels:
            located[name] = (offset, size)
            offset += size
        return located

    def flat_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for name, size in self.channels:
            if size == 1:
                labels.append(name)
            else:
                labels.extend(f"{name}[{i}]" for i in range(size))
        return tuple(labels)

    def decode(self, vector: Sequence[float]) -> Action:
        """Split one flat vector into the websocket action dict."""
        values = np.asarray(list(vector), dtype=np.float32).reshape(-1)
        if values.size != self.width:
            raise ValueError(
                f"Action has {values.size} numbers, expected {self.width} "
                f"({', '.join(self.flat_labels())})."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Action contains a non-finite number.")
        data: dict[str, np.ndarray] = {}
        offset = 0
        for name, size in self.channels:
            data[name] = values[offset : offset + size].copy()
            offset += size
        return Action(data=data)

    def encode(self, state: Mapping[str, Any]) -> list[float]:
        """Read the current state back out in flat channel order."""
        values: list[float] = []
        for name, size in self.channels:
            channel = np.asarray(state.get(name, np.zeros(size)), dtype=np.float32)
            channel = channel.reshape(-1)
            if channel.size < size:
                channel = np.pad(channel, (0, size - channel.size))
            values.extend(float(v) for v in channel[:size])
        return values
