# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Generic metric container and config surfaces.

This module keeps the evaluator-agnostic metric contracts: a minimal runtime
protocol, a named metric container, and config helpers that construct metric
instances without importing evaluator-specific dispatch logic.
"""

from __future__ import annotations
import copy
from typing import Any, Protocol, cast, runtime_checkable

from robo_orchard_core.utils.config import ClassConfig
from typing_extensions import TypeVar

from robo_orchard_lab.utils.state import (
    ConstructableStateApplyMode,
    State,
    StateSaveLoadMixin,
    validate_recovery_state,
)

__all__ = [
    "MetricProtocol",
    "MetricDict",
    "MetricConfig",
    "MetricDictConfig",
]


@runtime_checkable
class MetricProtocol(Protocol):
    """Minimal runtime contract for metrics stored in ``MetricDict``.

    This protocol intentionally stays evaluator-agnostic. It covers only the
    lifecycle operations that generic metric containers can forward safely:
    reset, update, compute, and device/dtype transfer.
    """

    def reset(self, **kwargs: Any):
        """Reset the metric state to its default value."""
        ...

    def compute(self) -> Any:
        """Compute the final metric value based on state."""
        ...

    def update(self, *args: Any, **kwargs: Any):
        """Update the metric state with new data."""
        ...

    def to(self, *args, **kwargs):
        """Move the metric state to a specified device or dtype."""
        ...


class MetricDict(StateSaveLoadMixin, dict[str, MetricProtocol]):
    """Dictionary-like container for named metric objects.

    `MetricDict` owns only generic metric container behavior: storing metrics,
    forwarding `reset` / `compute` / `to`, providing raw `update(...)`
    fan-out for homogeneous metric collections, and exposing `State`-based
    capture and restore. Members that support the State API recovery seam are
    captured as nested ``State`` payloads. Members that do not support that
    seam are captured as raw metric values inside ``State.state``. Restore is
    replace-only: decoded entries must already be live ``MetricProtocol``
    instances, and the current container contents are replaced with the
    decoded members.

    The main public methods are ``get_state()``, ``load_state(...)``,
    ``reset(...)``, ``update(...)``, and ``compute()``.

    Example:
        ``metrics = MetricDict({"score": score_metric, "loss": loss_metric})``
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) > 1:
            raise TypeError(
                "MetricDict expected at most 1 positional argument, "
                f"got {len(args)}."
            )
        if args:
            source = args[0]
            items = source.items() if hasattr(source, "items") else source
            for key, value in items:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(value, MetricProtocol):
            raise TypeError(
                "MetricDict members must implement MetricProtocol "
                "(`reset(...)`, `compute()`, `update(...)`, `to(...)`). "
                f"Got {type(value).__name__}."
            )
        super().__setitem__(key, value)

    def _get_state(self) -> State:
        """Capture member metric state or snapshot members for replacement."""
        state_by_name: dict[str, object] = {}
        for name, metric in self.items():
            try:
                if isinstance(metric, StateSaveLoadMixin):
                    member_state = metric.get_state()
                    if not isinstance(member_state, State):
                        raise TypeError(
                            "Metrics must return a State payload from "
                            "`get_state()`. "
                            f"Got {type(member_state).__name__} from "
                            f"{type(metric).__name__}."
                        )
                    validate_recovery_state(
                        member_state,
                        context=f"{type(metric).__name__}.get_state()",
                    )
                else:
                    member_state = metric
            except Exception as exception:
                raise type(exception)(
                    f"MetricDict member '{name}' failed state capture: "
                    f"{exception}"
                ) from exception

            if isinstance(member_state, State):
                state_by_name[name] = member_state.model_copy(deep=True)
            else:
                state_by_name[name] = copy.deepcopy(member_state)

        return State(
            state=state_by_name,
            class_type=MetricDict,
            config=None,
        )

    def load_state(
        self,
        path_or_state: str | State,
        *,
        constructable_state_apply_mode: ConstructableStateApplyMode
        | None = None,
    ) -> None:
        """Restore this container from a runtime ``State`` payload or path.

        ``MetricDict`` keeps replace-only semantics: decoded entries must
        materialize into live ``MetricProtocol`` objects before installation.
        Callers may keep the default decode mode or pass
        ``ConstructableStateApplyMode.MATERIALIZE`` explicitly. Preserve-state
        apply is rejected because this container does not apply nested member
        ``State`` payloads back into existing metrics.

        Args:
            path_or_state (str | State): Runtime ``State`` payload to apply, or
                a persisted State directory path.
            constructable_state_apply_mode (ConstructableStateApplyMode):
                Optional decode override. Only ``None`` and
                ``ConstructableStateApplyMode.MATERIALIZE`` are accepted.
                Default is ``None``.
        """
        if constructable_state_apply_mode not in (
            None,
            ConstructableStateApplyMode.MATERIALIZE,
        ):
            raise ValueError(
                "MetricDict restore is replace-only. "
                "constructable_state_apply_mode must be None or "
                "ConstructableStateApplyMode.MATERIALIZE."
            )
        super().load_state(
            path_or_state,
            constructable_state_apply_mode=constructable_state_apply_mode,
        )

    def _set_state(self, state: State) -> None:
        """Restore decoded metric state by replacing container entries."""
        payload = state.state
        if not isinstance(payload, dict):
            raise TypeError(
                "MetricDict state must decode to a dict keyed by metric name. "
                f"Got {type(payload).__name__}."
            )

        state_dict = cast(dict[str, object], dict(payload))
        invalid_entries: dict[str, str] = {}
        for name, item in state_dict.items():
            if not isinstance(item, MetricProtocol):
                invalid_entries[name] = type(item).__name__

        if invalid_entries:
            raise TypeError(
                "MetricDict restore is replace-only. State entries must "
                "decode to MetricProtocol instances before installation. "
                "Capture recoverable metrics as constructable State payloads "
                "or snapshot plain metric objects. "
                f"Got invalid entries: {invalid_entries}."
            )

        self.clear()
        for name, metric in cast(
            dict[str, MetricProtocol], state_dict
        ).items():
            self[name] = metric

    def to(self, *args: Any, **kwds: Any) -> None:
        for metric in self.values():
            metric.to(*args, **kwds)

    def update(self, *args: Any, **kwds: Any) -> None:
        for metric in self.values():
            metric.update(*args, **kwds)

    def reset(self, **kwargs: Any) -> None:
        for metric in self.values():
            metric.reset(**kwargs)

    def compute(self) -> dict[str, Any]:
        return {name: metric.compute() for name, metric in self.items()}


T = TypeVar("T")


class MetricConfig(ClassConfig[T]):
    """Config wrapper that instantiates one metric via ``class_type``."""

    def __call__(self) -> T:
        return self.class_type(self)  # type: ignore


class MetricDictConfig(dict):
    """Dictionary-like container that instantiates a ``MetricDict``.

    Each value must be a metric config whose ``__call__()`` returns one live
    metric instance. Calling this config returns a ``MetricDict`` keyed like
    the original mapping.
    """

    def __call__(self) -> MetricDict:
        return MetricDict({name: cfg() for name, cfg in self.items()})
