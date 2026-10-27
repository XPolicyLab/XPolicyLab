"""Callback base class.

Callbacks subscribe to training events by defining methods whose names
match event names (lowercase). The base class auto-registers these methods
with the EventBus.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from leap.core.event import Event, EventBus

if TYPE_CHECKING:
    from leap.core.state import TrainingState


# Mapping from method name to Event enum
_EVENT_METHOD_MAP = {event.name.lower(): event for event in Event}


class Callback(ABC):
    """Base class for training callbacks.

    Subclasses define methods named after events (lowercase) to handle them.
    For example, defining `def batch_end(self, state)` will auto-register
    the method to handle Event.BATCH_END.

    Attributes:
        priority: Execution order. Lower values execute first. Default 0.
    """

    priority: int = 0

    def register(self, event_bus: EventBus) -> None:
        """Auto-register event handler methods with the EventBus.

        Scans this instance for methods matching event names and subscribes
        them to the corresponding events.
        """
        for method_name, event in _EVENT_METHOD_MAP.items():
            method = getattr(self, method_name, None)
            if method is not None and callable(method):
                # Skip if it's the base class no-op (not overridden)
                if method_name not in type(self).__dict__:
                    # Check parent classes (but not Callback itself)
                    found = False
                    for cls in type(self).__mro__:
                        if cls is Callback:
                            break
                        if method_name in cls.__dict__:
                            found = True
                            break
                    if not found:
                        continue
                event_bus.subscribe(event, method, priority=self.priority)

    # Default no-op implementations for all events.
    # Subclasses override only the events they care about.
    def init(self, state: TrainingState) -> None:
        pass

    def fit_start(self, state: TrainingState) -> None:
        pass

    def fit_end(self, state: TrainingState) -> None:
        pass

    def epoch_start(self, state: TrainingState) -> None:
        pass

    def epoch_end(self, state: TrainingState) -> None:
        pass

    def batch_start(self, state: TrainingState) -> None:
        pass

    def before_forward(self, state: TrainingState) -> None:
        pass

    def after_forward(self, state: TrainingState) -> None:
        pass

    def before_backward(self, state: TrainingState) -> None:
        pass

    def after_backward(self, state: TrainingState) -> None:
        pass

    def before_optimizer_step(self, state: TrainingState) -> None:
        pass

    def after_optimizer_step(self, state: TrainingState) -> None:
        pass

    def batch_end(self, state: TrainingState) -> None:
        pass

    def eval_start(self, state: TrainingState) -> None:
        pass

    def eval_batch(self, state: TrainingState) -> None:
        pass

    def eval_end(self, state: TrainingState) -> None:
        pass

    def rollout_start(self, state: TrainingState) -> None:
        pass

    def rollout_end(self, state: TrainingState) -> None:
        pass

    def reward_computed(self, state: TrainingState) -> None:
        pass

    def before_save(self, state: TrainingState) -> None:
        pass

    def after_save(self, state: TrainingState) -> None:
        pass

    def before_load(self, state: TrainingState) -> None:
        pass

    def after_load(self, state: TrainingState) -> None:
        pass
