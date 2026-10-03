"""UI-independent runtime events, usable by a CLI, recorder or dashboard."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Event:
    kind: str
    data: dict[str, Any]


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullEvents:
    def emit(self, event: Event) -> None:
        pass


class RecordedEvents:
    def __init__(self):
        self.events = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
