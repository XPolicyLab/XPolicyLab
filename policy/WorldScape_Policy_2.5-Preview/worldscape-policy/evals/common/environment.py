from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.common.suite import EvaluationTask


@dataclass(frozen=True)
class EnvironmentStep:
    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] | None = None
    frames: tuple[np.ndarray, ...] = ()

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class SimulatorEvaluationEnvironment:
    """Normalize Gym-like simulators to ``EvaluationEnvironment``."""

    def __init__(
        self,
        environment: Any,
        *,
        success_fn: Callable[[Any, Mapping[str, Any]], bool] | None = None,
        capture_frames: bool = False,
    ) -> None:
        self.environment = environment
        self._success_fn = success_fn
        self._capture_frames = capture_frames
        self._last_info: Mapping[str, Any] = {}
        self._last_step: Any = None

    def reset(self, task: EvaluationTask, *, seed: int | None = None) -> Any:
        kwargs = dict(task.metadata.get("reset_kwargs", {}))
        if seed is not None:
            kwargs.setdefault("seed", seed)
        try:
            value = self.environment.reset(**kwargs)
        except TypeError:
            # A number of benchmark releases predate Gym's seeded reset API.
            kwargs.pop("seed", None)
            value = self.environment.reset(**kwargs)
        self._last_info = (
            value[1]
            if isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], Mapping)
            else {}
        )
        self._last_step = value
        return value

    def step(self, action: Any) -> EnvironmentStep:
        value = self.environment.step(action)
        step = normalize_environment_step(value)
        frames = step.frames
        if self._capture_frames and not frames:
            render = getattr(self.environment, "render", None)
            if callable(render):
                rendered = render()
                if rendered is not None:
                    frames = (np.asarray(rendered),)
        step = EnvironmentStep(
            observation=step.observation,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=step.info,
            frames=frames,
        )
        self._last_info = step.info or {}
        self._last_step = value
        return step

    def success(self) -> bool:
        if self._success_fn is not None:
            return bool(self._success_fn(self._last_step, self._last_info))
        for key in ("success", "is_success", "task_success"):
            if key in self._last_info:
                return bool(self._last_info[key])
        for name in ("success", "is_success", "check_success", "_check_success"):
            candidate = getattr(self.environment, name, None)
            if callable(candidate):
                return bool(candidate())
            if candidate is not None:
                return bool(candidate)
        return False

    def metrics(self) -> dict[str, float]:
        """Return common success and subgoal metrics when a backend exposes them."""

        metrics: dict[str, float] = {"success": float(self.success())}
        aliases = {
            "subgoal_success": "subgoal_success",
            "subgoal_completion": "subgoal_completion",
            "subgoal_progress": "subgoal_completion",
            "num_subgoals_completed": "subgoals_completed",
            "subgoals_completed": "subgoals_completed",
            "num_subgoals": "subgoals_total",
            "subgoals_total": "subgoals_total",
        }
        for source, target in aliases.items():
            value = self._last_info.get(source)
            if isinstance(value, (bool, int, float, np.number)):
                metrics[target] = float(value)
        for name in ("subgoal_success", "subgoal_completion"):
            candidate = getattr(self.environment, name, None)
            if callable(candidate):
                candidate = candidate()
            if isinstance(candidate, (bool, int, float, np.number)):
                metrics[name] = float(candidate)
        return metrics

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


def normalize_environment_step(value: Any) -> EnvironmentStep:
    if isinstance(value, EnvironmentStep):
        return value
    if not isinstance(value, tuple):
        return EnvironmentStep(observation=value)
    if len(value) == 5:
        observation, reward, terminated, truncated, info = value
    elif len(value) == 4:
        observation, reward, done, info = value
        terminated, truncated = done, False
    else:
        raise ValueError(
            "Environment step must return an observation, a Gym 4/5-tuple, "
            "or EnvironmentStep"
        )
    if not isinstance(info, Mapping):
        info = {}
    raw_frames = info.get("frames", ())
    if isinstance(raw_frames, np.ndarray):
        raw_frames = (raw_frames,)
    return EnvironmentStep(
        observation=observation,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
        frames=tuple(np.asarray(frame) for frame in raw_frames),
    )
