"""Typed, checkpointable lifecycle callbacks for native training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TrainerState(Protocol):
    """Trainer state exposed to callbacks without coupling to an implementation."""

    step: int
    data_batches_consumed: int


@runtime_checkable
class TrainingCallback(Protocol):
    """Lifecycle extension point for the native training loop.

    ``on_checkpoint_save`` runs immediately before the trainer snapshots any
    state. Mutations made by the hook are therefore included in that
    checkpoint. ``path`` identifies the destination; callbacks must not assume
    the new checkpoint has been written yet.
    """

    def on_train_start(self, trainer: TrainerState) -> None: ...

    def on_train_end(self, trainer: TrainerState) -> None: ...

    def on_before_step(self, trainer: TrainerState, batch: Any) -> None: ...

    def on_after_step(
        self,
        trainer: TrainerState,
        batch: Any,
        metrics: Mapping[str, float],
    ) -> None: ...

    def on_checkpoint_save(self, trainer: TrainerState, path: Path) -> None: ...

    def on_checkpoint_load(self, trainer: TrainerState, path: Path) -> None: ...

    def on_metrics(
        self,
        trainer: TrainerState,
        metrics: Mapping[str, float],
    ) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class NoOpTrainingCallback:
    """Default callback preserving the trainer's callback-free behavior."""

    def on_train_start(self, trainer: TrainerState) -> None:
        del trainer

    def on_train_end(self, trainer: TrainerState) -> None:
        del trainer

    def on_before_step(self, trainer: TrainerState, batch: Any) -> None:
        del trainer, batch

    def on_after_step(
        self,
        trainer: TrainerState,
        batch: Any,
        metrics: Mapping[str, float],
    ) -> None:
        del trainer, batch, metrics

    def on_checkpoint_save(self, trainer: TrainerState, path: Path) -> None:
        del trainer, path

    def on_checkpoint_load(self, trainer: TrainerState, path: Path) -> None:
        del trainer, path

    def on_metrics(
        self,
        trainer: TrainerState,
        metrics: Mapping[str, float],
    ) -> None:
        del trainer, metrics

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("no-op callback state must be empty")


class CallbackList(NoOpTrainingCallback):
    """Fan out lifecycle events and namespace each callback's state."""

    def __init__(
        self,
        callbacks: TrainingCallback | Sequence[TrainingCallback] | None = None,
    ) -> None:
        if callbacks is None:
            values: tuple[TrainingCallback, ...] = (NoOpTrainingCallback(),)
        elif isinstance(callbacks, Sequence):
            values = tuple(callbacks) or (NoOpTrainingCallback(),)
        else:
            values = (callbacks,)
        for callback in values:
            if not isinstance(callback, TrainingCallback):
                raise TypeError(
                    "callbacks must implement the complete TrainingCallback protocol"
                )
        self.callbacks = values

    def on_train_start(self, trainer: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_train_end(self, trainer: TrainerState) -> None:
        for callback in reversed(self.callbacks):
            callback.on_train_end(trainer)

    def on_before_step(self, trainer: TrainerState, batch: Any) -> None:
        for callback in self.callbacks:
            callback.on_before_step(trainer, batch)

    def on_after_step(
        self,
        trainer: TrainerState,
        batch: Any,
        metrics: Mapping[str, float],
    ) -> None:
        for callback in self.callbacks:
            callback.on_after_step(trainer, batch, metrics)

    def on_checkpoint_save(self, trainer: TrainerState, path: Path) -> None:
        for callback in self.callbacks:
            callback.on_checkpoint_save(trainer, path)

    def on_checkpoint_load(self, trainer: TrainerState, path: Path) -> None:
        for callback in self.callbacks:
            callback.on_checkpoint_load(trainer, path)

    def on_metrics(
        self,
        trainer: TrainerState,
        metrics: Mapping[str, float],
    ) -> None:
        for callback in self.callbacks:
            callback.on_metrics(trainer, metrics)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            str(index): dict(callback.state_dict())
            for index, callback in enumerate(self.callbacks)
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {str(index) for index in range(len(self.callbacks))}
        actual = set(state)
        if actual != expected:
            raise ValueError(
                "callback state keys failed validation: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for index, callback in enumerate(self.callbacks):
            callback_state = state[str(index)]
            if not isinstance(callback_state, Mapping):
                raise TypeError("individual callback state must be a mapping")
            callback.load_state_dict(callback_state)


Callback = TrainingCallback
NoOpCallback = NoOpTrainingCallback

__all__ = [
    "Callback",
    "CallbackList",
    "NoOpCallback",
    "NoOpTrainingCallback",
    "TrainerState",
    "TrainingCallback",
]
