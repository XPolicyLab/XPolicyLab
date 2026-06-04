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

from __future__ import annotations
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from robo_orchard_lab.metrics.base import MetricDict, MetricProtocol
from robo_orchard_lab.utils.state import State, validate_recovery_state

__all__ = [
    "EvaluatorMetrics",
    "MetricUpdateTiming",
    "MergeableMetricProtocol",
    "capture_metric_state",
    "load_metric_state",
]

_DEFAULT_SINGLE_METRIC_NAME = "metric"
_MISSING = object()


class MetricUpdateTiming(str, Enum):
    """Evaluator-owned runtime dispatch timing."""

    TERMINAL = "TERMINAL"
    STEP = "STEP"


@runtime_checkable
class MergeableMetricProtocol(Protocol):
    """Metric aggregation contract for policy evaluation."""

    def merge(self, metrics: Iterable[Any]) -> None:
        """Merge runtime state from peer metrics."""
        ...


def capture_metric_state(metrics: MetricDict) -> State:
    """Capture recoverable metric state as a canonical State payload."""

    state = metrics.get_state()
    validate_recovery_state(state, context="Metric state")
    return state


def load_metric_state(metrics: MetricDict, state: State) -> None:
    """Restore recoverable metric state from a canonical State payload."""

    if not isinstance(state, State):
        raise TypeError(
            "Metric state must be a State payload. "
            f"Got {type(state).__name__}."
        )
    metrics.load_state(state)


class EvaluatorMetrics:
    """Evaluator-owned metric surface for PolicyEvaluator and remote wrappers.

    `EvaluatorMetrics` keeps generic metric objects grouped by evaluator-owned
    update timing while exposing a familiar metric-style lifecycle:
    `reset(...)`, `update(...)`, `compute()`, `to(...)`, `get_state()`, and
    `load_state(...)`.

    Use `from_channels(...)` for explicit timing ownership, or use
    `from_metric(...)` / `from_metric_dict(...)` as terminal-only shorthand.

    `EvaluatorMetrics` owns reconstruction metadata such as metric names and
    timing-channel membership. Canonical runtime metric state still flows
    through the delegated `MetricDict + State` seam exposed by `get_state()`
    and `load_state(...)`.

    For example::

        evaluator_metrics = EvaluatorMetrics.from_channels(
            terminal={"success_rate": success_metric},
            step={"reward_trace": reward_metric},
        )
    """

    _metric_dict: MetricDict
    _terminal_metric_names: tuple[str, ...]
    _step_metric_names: tuple[str, ...]

    def __init__(
        self,
        *,
        metric_dict: MetricDict,
        terminal_metric_names: tuple[str, ...],
        step_metric_names: tuple[str, ...],
    ) -> None:
        self._metric_dict = metric_dict
        self._terminal_metric_names = terminal_metric_names
        self._step_metric_names = step_metric_names

    @classmethod
    def from_channels(
        cls,
        *,
        terminal: Mapping[str, MetricProtocol] | None = None,
        step: Mapping[str, MetricProtocol] | None = None,
    ) -> EvaluatorMetrics:
        """Build an evaluator-owned metric surface from explicit channels.

        Args:
            terminal: Metrics updated once at episode termination. Names must
                be unique across all channels.
            step: Metrics updated on every environment step. Names must be
                unique across all channels.

        Returns:
            EvaluatorMetrics: Wrapper with explicit evaluator-owned timing.

        Raises:
            TypeError: If any metric does not implement `MetricProtocol`.
            ValueError: If channel names are duplicated, or one live metric
                instance is registered more than once.
        """
        terminal_items = [] if terminal is None else list(terminal.items())
        step_items = [] if step is None else list(step.items())

        metric_dict = MetricDict()
        terminal_metric_names: list[str] = []
        step_metric_names: list[str] = []
        metric_name_by_id: dict[int, str] = {}

        def add_channel_items(
            items: list[tuple[str, MetricProtocol]],
            *,
            target_names: list[str],
        ) -> None:
            for name, metric in items:
                if not isinstance(metric, MetricProtocol):
                    raise TypeError(
                        "Evaluator metrics must implement MetricProtocol "
                        "(`reset(...)`, `update(...)`, `compute()`, "
                        "`to(...)`). "
                        f"Got {type(metric).__name__} for '{name}'."
                    )
                if name in metric_dict:
                    raise ValueError(
                        "EvaluatorMetrics channel names must be unique across "
                        f"all timings. Duplicate metric name: '{name}'."
                    )
                metric_id = id(metric)
                previous_name = metric_name_by_id.get(metric_id)
                if previous_name is not None:
                    raise ValueError(
                        "One live metric instance cannot be registered under "
                        "multiple evaluator metric names. "
                        f"Metric '{name}' reuses the same instance as "
                        f"'{previous_name}'."
                    )

                metric_name_by_id[metric_id] = name
                metric_dict[name] = metric
                target_names.append(name)

        add_channel_items(
            terminal_items,
            target_names=terminal_metric_names,
        )
        add_channel_items(
            step_items,
            target_names=step_metric_names,
        )

        return cls(
            metric_dict=metric_dict,
            terminal_metric_names=tuple(terminal_metric_names),
            step_metric_names=tuple(step_metric_names),
        )

    @classmethod
    def from_metric(
        cls,
        metric: MetricProtocol,
        *,
        name: str = _DEFAULT_SINGLE_METRIC_NAME,
    ) -> EvaluatorMetrics:
        """Build a terminal-only evaluator surface for one metric.

        This is shorthand for `from_channels(terminal={name: metric})`.
        """
        return cls.from_channels(terminal={name: metric})

    @classmethod
    def from_metric_dict(
        cls,
        metrics: MetricDict,
    ) -> EvaluatorMetrics:
        """Build a terminal-only evaluator surface from one MetricDict.

        This is shorthand for `from_channels(terminal=dict(metrics))`.
        """
        return cls.from_channels(terminal=dict(metrics))

    @property
    def requires_step_callback(self) -> bool:
        return bool(self._step_metric_names)

    def get_metric(self, name: str) -> MetricProtocol:
        """Return one live child metric by name."""

        return self._metric_dict[name]

    def reset(self, **kwargs: Any) -> None:
        """Reset all registered metrics."""

        self._metric_dict.reset(**kwargs)

    def update(
        self,
        *,
        timing: MetricUpdateTiming,
        action: Any = _MISSING,
        step_ret: Any = _MISSING,
    ) -> None:
        """Dispatch one evaluator-owned metric update by timing.

        Required payloads:
            - `TERMINAL`: `action` and `step_ret`
            - `STEP`: `action` and `step_ret`
        """

        if timing is MetricUpdateTiming.TERMINAL:
            if action is _MISSING or step_ret is _MISSING:
                raise TypeError(
                    "TERMINAL metric updates require `action` and `step_ret`."
                )
            for name in self._terminal_metric_names:
                self.get_metric(name).update(action, step_ret)
            return
        if timing is MetricUpdateTiming.STEP:
            if action is _MISSING or step_ret is _MISSING:
                raise TypeError(
                    "STEP metric updates require `action` and `step_ret`."
                )
            for name in self._step_metric_names:
                self.get_metric(name).update(action, step_ret)
            return
        raise TypeError(
            "Evaluator metric updates require a valid MetricUpdateTiming. "
            f"Got {timing!r}."
        )

    def compute(self) -> dict[str, Any]:
        """Compute all registered metrics as one dict-shaped result."""

        return self._metric_dict.compute()

    def to(self, *args: Any, **kwargs: Any) -> None:
        """Move all registered metrics to a device or dtype."""

        self._metric_dict.to(*args, **kwargs)

    def get_state(self) -> State:
        """Capture canonical runtime state through the delegated State seam.

        This `State` payload is the canonical runtime recovery object. It does
        not replace evaluator-owned reconstruction metadata such as names or
        timing-channel membership.
        """

        return capture_metric_state(self._metric_dict)

    def load_state(self, path_or_state: str | State) -> None:
        """Restore runtime state through the delegated MetricDict seam.

        `State` is the canonical runtime recovery payload. A `str` path is
        supported only for persistence compatibility with existing
        `StateSaveLoadMixin` surfaces.
        """

        if isinstance(path_or_state, str):
            self._metric_dict.load_state(path_or_state)
        else:
            load_metric_state(self._metric_dict, path_or_state)
