"""Training event system.

Defines training lifecycle events and an EventBus for publish-subscribe
communication between the trainer and callbacks.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Tuple


class Event(Enum):
    """Training lifecycle events."""

    # Run lifecycle
    INIT = auto()
    FIT_START = auto()
    FIT_END = auto()

    # Epoch lifecycle
    EPOCH_START = auto()
    EPOCH_END = auto()

    # Batch lifecycle
    BATCH_START = auto()
    BEFORE_FORWARD = auto()
    AFTER_FORWARD = auto()
    BEFORE_BACKWARD = auto()
    AFTER_BACKWARD = auto()
    BEFORE_OPTIMIZER_STEP = auto()
    AFTER_OPTIMIZER_STEP = auto()
    BATCH_END = auto()

    # Evaluation
    EVAL_START = auto()
    EVAL_BATCH = auto()
    EVAL_END = auto()

    # RL-specific
    ROLLOUT_START = auto()
    ROLLOUT_END = auto()
    REWARD_COMPUTED = auto()

    # Checkpointing
    BEFORE_SAVE = auto()
    AFTER_SAVE = auto()
    BEFORE_LOAD = auto()
    AFTER_LOAD = auto()


class EventBus:
    """Publish-subscribe event bus for training lifecycle events.

    Callbacks subscribe handlers to specific events with an optional priority.
    Lower priority values execute first.
    """

    def __init__(self) -> None:
        self._handlers: Dict[Event, List[Tuple[int, Callable]]] = {
            event: [] for event in Event
        }
        self._sorted: Dict[Event, bool] = {event: True for event in Event}

    def subscribe(self, event: Event, handler: Callable, priority: int = 0) -> None:
        """Register a handler for an event.

        Args:
            event: The event to subscribe to.
            handler: Callable that accepts (state: TrainingState) as argument.
            priority: Lower values execute first. Default 0.
        """
        self._handlers[event].append((priority, handler))
        self._sorted[event] = False

    def unsubscribe(self, event: Event, handler: Callable) -> None:
        """Remove a handler from an event."""
        self._handlers[event] = [
            (p, h) for p, h in self._handlers[event] if h is not handler
        ]

    def fire(self, event: Event, state: Any) -> None:
        """Fire an event, calling all subscribed handlers in priority order.

        Args:
            event: The event to fire.
            state: The TrainingState object passed to each handler.
        """
        if not self._sorted[event]:
            self._handlers[event].sort(key=lambda x: x[0])
            self._sorted[event] = True
        for _, handler in self._handlers[event]:
            handler(state)

    def clear(self, event: Event | None = None) -> None:
        """Clear handlers for a specific event, or all events if None."""
        if event is not None:
            self._handlers[event] = []
            self._sorted[event] = True
        else:
            for e in Event:
                self._handlers[e] = []
                self._sorted[e] = True
