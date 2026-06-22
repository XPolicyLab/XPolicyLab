"""In-process stop signaling for long-running env client trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Literal

StopRequestResult = Literal["accepted", "already_stopping", "not_found"]


@dataclass
class TrialControlRegistry:
    """Maps active (evaluation_id, trial_index) pairs to stop events."""

    _active: dict[tuple[str, int], Event] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def register(self, evaluation_id: str, trial_index: int) -> Event:
        key = (evaluation_id, trial_index)
        event = Event()
        with self._lock:
            self._active[key] = event
        return event

    def register_if_idle(self, evaluation_id: str, trial_index: int) -> Event | None:
        key = (evaluation_id, trial_index)
        event = Event()
        with self._lock:
            if self._active:
                return None
            self._active[key] = event
        return event

    def is_active(self, evaluation_id: str, trial_index: int) -> bool:
        with self._lock:
            return (evaluation_id, trial_index) in self._active

    def has_active_trials(self) -> bool:
        with self._lock:
            return bool(self._active)

    def request_stop(self, evaluation_id: str, trial_index: int) -> StopRequestResult:
        with self._lock:
            event = self._active.get((evaluation_id, trial_index))
            if event is None:
                return "not_found"
            if event.is_set():
                return "already_stopping"
            event.set()
            return "accepted"

    def clear(self, evaluation_id: str, trial_index: int) -> None:
        with self._lock:
            self._active.pop((evaluation_id, trial_index), None)
