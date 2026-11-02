"""Thread-safe event bus with per-consumer cursors.

Follows the split in KRA, our team's in-house ReAct robot-agent framework,
between wake events (they reach the next model turn) and telemetry (recorded
only). Each agent session reads the wake events it has not seen yet, so
completions of asynchronous robot work show up exactly once.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class EventKind(str, Enum):
    TRACK_STARTED = "track_started"
    TRACK_FINISHED = "track_finished"
    TRACK_FAILED = "track_failed"
    TRACK_CANCELLED = "track_cancelled"
    SKILL_STARTED = "skill_started"
    SKILL_FINISHED = "skill_finished"
    DELEGATED = "delegated"
    RETURNED = "returned"
    NOTE = "note"
    BUDGET = "budget"
    EPISODE_OVER = "episode_over"


WAKE_KINDS = {
    EventKind.TRACK_FINISHED,
    EventKind.TRACK_FAILED,
    EventKind.TRACK_CANCELLED,
    EventKind.EPISODE_OVER,
}


@dataclass
class Event:
    kind: EventKind
    source: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    tick: Optional[int] = None
    time: float = field(default_factory=time.time)
    id: int = 0

    @property
    def wakes(self) -> bool:
        return self.kind in WAKE_KINDS

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind.value, "source": self.source, "summary": self.summary,
                "tick": self.tick, "data": self.data}


class EventBus:
    def __init__(self, history: int = 2000):
        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._history = history
        self._ids = itertools.count(1)
        self._cursors: dict[str, int] = {}
        self._subscribers: list[Callable[[Event], None]] = []

    def publish(self, kind: EventKind, source: str, summary: str, *, tick: Optional[int] = None, **data) -> Event:
        with self._lock:
            event = Event(kind=kind, source=source, summary=summary, data=data, tick=tick, id=next(self._ids))
            self._events.append(event)
            if len(self._events) > self._history:
                del self._events[: len(self._events) - self._history]
            subscribers = list(self._subscribers)
        for callback in subscribers:
            callback(event)
        return event

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unseen(self, consumer: str, *, wake_only: bool = True) -> list[Event]:
        """Events published since this consumer last asked; advances its cursor."""
        with self._lock:
            start = self._cursors.get(consumer, 0)
            fresh = [e for e in self._events if e.id > start]
            if self._events:
                self._cursors[consumer] = self._events[-1].id
        return [e for e in fresh if e.wakes] if wake_only else fresh

    def mark_seen(self, consumer: str) -> None:
        with self._lock:
            self._cursors[consumer] = self._events[-1].id if self._events else 0

    def recent(self, limit: int = 20) -> list[Event]:
        with self._lock:
            return list(self._events[-limit:])
