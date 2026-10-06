"""Small, benchmark-neutral Gymnasium interface for simulator tasks.

The simulator adapter is deliberately a normal :class:`gymnasium.Env`:

* ``reset(seed=...)`` returns ``(observation, info)``;
* ``step(code)`` accepts the agent's Python source and returns Gymnasium's
  five values;
* ``render()`` and ``close()`` are the only lifecycle methods the harness
  needs in addition to those two calls.

The host-side ``StepResult`` and ``execute`` helper are bookkeeping only.  An
agent never sees them and no second action protocol is introduced.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

from gymnasium import Env, spaces

MAX_CODE_LENGTH = 8 * 1024 * 1024
CODE_CHARSET = frozenset(string.printable)


def _code_space() -> spaces.Text:
    """Return a fresh action space for one environment instance."""

    return spaces.Text(
        min_length=1,
        max_length=MAX_CODE_LENGTH,
        charset=CODE_CHARSET,
    )


class Environment(Env):
    """Common Gymnasium base for every EmbodiedRSI simulator environment.

    Subclasses construct their native simulator and call ``configure`` once
    it is available.  They implement normal Gymnasium ``reset`` plus a small
    private ``_step`` adapter; callers only see the validated ``step(code)``
    method and no benchmark-specific action method.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = _code_space()
        # Native environments provide the actual public observation space.
        # This neutral space keeps construction valid for lightweight test
        # doubles that do not declare one.
        self.observation_space: spaces.Space = spaces.Space()
        self._step_index = 0
        # Only the host PlayGround runner grants scene-reset capability.
        self.execution_phase = "test"

    def configure(self, native: Any) -> None:
        """Adopt a native environment's public observation space.

        The action space is intentionally always ours.  A native code env may
        expose an equivalent ``Text`` space, but allowing it to replace this
        object would make the bridge differ by simulator.
        """

        observation_space = getattr(native, "observation_space", None)
        if isinstance(observation_space, spaces.Space):
            self.observation_space = observation_space

    def begin_episode(self, seed: int | None) -> None:
        """Reset Gymnasium bookkeeping before delegating to a native env."""

        super().reset(seed=seed)
        self._step_index = 0

    def validate_action(self, code: str) -> str:
        """Validate the one and only action type at the bridge boundary."""

        if not isinstance(code, str):
            raise TypeError(f"action must be Python source as str, got {type(code).__name__}")
        if not code:
            raise ValueError("action must contain non-empty Python source")
        if len(code) > MAX_CODE_LENGTH:
            raise ValueError(f"action exceeds {MAX_CODE_LENGTH} characters")
        if not self.action_space.contains(code):
            raise ValueError("action contains characters outside the declared code space")
        return code

    def next_step_index(self) -> int:
        index = self._step_index
        self._step_index += 1
        return index

    def finish_step(
        self,
        observation: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        *,
        frames: dict[str, list[Any]] | None = None,
        success: bool | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Complete one common transition after a native step.

        Adapters supply only their native transition and, where necessary, a
        task-specific success predicate.  The returned shape and host fields
        are identical for every simulator.
        """

        info = dict(info)
        if success is None:
            completed = info.get("task_completed")
            success = bool(completed) if completed is not None else bool(terminated)
        info.update(
            task_completed=bool(success),
            step_index=self.next_step_index(),
            frames=dict(frames or {}),
        )
        return (
            observation,
            float(reward),
            bool(terminated or success),
            bool(truncated and not (terminated or success)),
            info,
        )

    def step(self, code: str) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Validate the source action before handing it to an adapter."""

        return self._step(self.validate_action(code))

    def _step(self, code: str) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("an environment adapter must implement _step(code)")

    def render(self, *args: Any, **kwargs: Any) -> Any:
        native = getattr(self, "env", None)
        render = getattr(native, "render", None)
        if not callable(render):
            raise NotImplementedError("the native environment does not implement render()")
        return render(*args, **kwargs)

    def close(self) -> None:
        native = getattr(self, "env", None)
        close = getattr(native, "close", None)
        if callable(close):
            close()


@dataclass
class StepResult:
    """Host-side view of one Gymnasium transition.

    ``execute`` only runs on the host. Its frames are consumed there when
    writing raw feedback videos; its reward stays in the private ledger.
    """

    index: int
    code: str
    ok: bool
    stdout: str
    stderr: str
    reward: float
    success: bool
    terminated: bool = False
    truncated: bool = False
    frames: dict[str, list[Any]] = field(default_factory=dict)


def execute(environment: Any, code: str, *, index: int | None = None) -> StepResult:
    """Convert a standard step(code) transition into private host bookkeeping."""
    transition = environment.step(code)

    if not isinstance(transition, tuple) or len(transition) != 5:
        raise TypeError("environment.step(code) must return Gymnasium's five values")
    _observation, reward, terminated, truncated, raw_info = transition
    info = dict(raw_info or {})
    frames = dict(info.get("frames", {}) or {})
    ok = bool(info["ok"])
    success = bool(info.get("task_completed", terminated))
    return StepResult(
        index=index if index is not None else int(info.get("step_index", 0)),
        code=code,
        ok=ok,
        stdout=str(info.get("stdout", "")),
        stderr=str(info.get("stderr", "")),
        reward=float(reward),
        success=success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        frames=frames,
    )


__all__ = [
    "CODE_CHARSET",
    "MAX_CODE_LENGTH",
    "Environment",
    "StepResult",
    "execute",
]
