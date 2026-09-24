"""WorldScape-owned robot transport contracts for AgileX evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class AgileXReadRequest:
    use_history: bool = False
    num_history_frames: int = 4
    action_mode: str = "eef"


@dataclass(frozen=True)
class AgileXRobotObservation:
    high: np.ndarray
    left: np.ndarray
    right: np.ndarray
    state: dict[str, np.ndarray]
    timestamp: Any

    def as_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], Any]:
        return self.high, self.left, self.right, self.state, self.timestamp


@dataclass(frozen=True)
class AgileXActionCommand:
    timestamp: Any
    rate: int
    left: list[list[float]]
    right: list[list[float]]
    action_mode: str = "eef"


@dataclass(frozen=True)
class AgileXExecutionResult:
    """Definitive transport outcome for an action command."""

    accepted: bool
    timed_out: bool = False
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.timed_out:
            raise ValueError("An accepted action cannot also be timed out")


@runtime_checkable
class AgileXRobot(Protocol):
    """Transport-neutral contract consumed by native WorldScape evaluation."""

    def reset_episode(self) -> None: ...

    def observe(self, request: AgileXReadRequest) -> AgileXRobotObservation: ...

    def try_observe(
        self, request: AgileXReadRequest
    ) -> AgileXRobotObservation | None: ...

    def execute(
        self,
        command: AgileXActionCommand,
        *,
        timeout_s: float | None = None,
    ) -> AgileXExecutionResult: ...

    def close(self) -> None: ...


class LegacyAgileXRobotAdapter:
    """The sole boundary translating historical robot method names."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def reset_episode(self) -> None:
        reset = getattr(self._robot, "reset_episode", None)
        if callable(reset):
            reset()

    def observe(self, request: AgileXReadRequest) -> AgileXRobotObservation:
        value = self._robot.read_observation(
            use_history=request.use_history,
            num_history_frames=request.num_history_frames,
            action_mode=request.action_mode,
        )
        return _observation(value)

    def try_observe(
        self, request: AgileXReadRequest
    ) -> AgileXRobotObservation | None:
        read = getattr(self._robot, "try_read_observation", None)
        if not callable(read):
            return self.observe(request)
        value = read(
            use_history=request.use_history,
            num_history_frames=request.num_history_frames,
            action_mode=request.action_mode,
        )
        return None if value is None else _observation(value)

    def execute(
        self,
        command: AgileXActionCommand,
        *,
        timeout_s: float | None = None,
    ) -> AgileXExecutionResult:
        joint = command.action_mode == "joint"
        base_name = "send_joint_state_action" if joint else "send_end_pose_action"
        timeout_name = f"{base_name}_with_timeout"
        if timeout_s is None:
            getattr(self._robot, base_name)(
                command.timestamp,
                command.rate,
                command.left,
                command.right,
            )
            return AgileXExecutionResult(accepted=True)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        send = getattr(self._robot, timeout_name, None)
        if not callable(send):
            raise RuntimeError(  # noqa: TRY004 - missing transport capability
                "Configured send_timeout_s requires the robot transport "
                f"to implement {timeout_name}()"
            )
        result = send(
            command.timestamp,
            command.rate,
            command.left,
            command.right,
            timeout_s=timeout_s,
        )
        if isinstance(result, AgileXExecutionResult):
            return result
        if all(hasattr(result, name) for name in ("accepted", "timed_out")):
            return AgileXExecutionResult(
                accepted=bool(result.accepted),
                timed_out=bool(result.timed_out),
                detail=getattr(result, "detail", None),
            )
        raise TypeError(f"{timeout_name}() must return an execution result")

    def read_context_video(
        self, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        read = getattr(self._robot, "read_context_video", None)
        return read(num_frames) if callable(read) else None

    def try_read_context_video(
        self, num_frames: int
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, bool]:
        poll = getattr(self._robot, "try_read_context_video", None)
        if not callable(poll):
            raise TypeError("Robot transport does not support context polling")
        return poll(num_frames)

    def default_max_steps(self, rollout_steps: int) -> int:
        infer = getattr(self._robot, "default_max_steps", None)
        if not callable(infer):
            raise RuntimeError(  # noqa: TRY004 - missing transport capability
                "Robot transport cannot infer a replay horizon"
            )
        return int(infer(rollout_steps))

    def close(self) -> None:
        close = getattr(self._robot, "close", None)
        if callable(close):
            close()


class HDF5ReplayRobot:
    """Read-only HDF5 transport implementing the WorldScape robot contract."""

    def __init__(
        self,
        path: str | Path,
        *,
        video_columns: str | None = None,
    ) -> None:
        from evals.agilex.replay import NativeHDF5Robot

        self.path = Path(path)
        kwargs = (
            {"video_columns": video_columns}
            if video_columns is not None
            else {}
        )
        self._transport = LegacyAgileXRobotAdapter(
            NativeHDF5Robot(path, **kwargs)
        )

    def reset_episode(self) -> None:
        self._transport.reset_episode()

    def observe(self, request: AgileXReadRequest) -> AgileXRobotObservation:
        return self._transport.observe(request)

    def try_observe(
        self, request: AgileXReadRequest
    ) -> AgileXRobotObservation | None:
        return self.observe(request)

    def execute(
        self,
        command: AgileXActionCommand,
        *,
        timeout_s: float | None = None,
    ) -> AgileXExecutionResult:
        del command, timeout_s
        return AgileXExecutionResult(
            accepted=True, detail="HDF5 replay: command not executed"
        )

    def default_max_steps(self, rollout_steps: int) -> int:
        return self._transport.default_max_steps(rollout_steps)

    def read_context_video(
        self, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._transport.read_context_video(num_frames)

    def try_read_context_video(
        self, num_frames: int
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, bool]:
        del num_frames
        return None, False

    def close(self) -> None:
        self._transport.close()


def ensure_agilex_robot(robot: Any) -> AgileXRobot:
    """Return a native transport, adapting historical implementations once."""

    if isinstance(robot, AgileXRobot):
        return robot
    return LegacyAgileXRobotAdapter(robot)


def _observation(value: Any) -> AgileXRobotObservation:
    if isinstance(value, AgileXRobotObservation):
        return value
    high, left, right, state, timestamp = value
    return AgileXRobotObservation(
        high=np.asarray(high),
        left=np.asarray(left),
        right=np.asarray(right),
        state={str(key): np.asarray(item) for key, item in state.items()},
        timestamp=timestamp,
    )
