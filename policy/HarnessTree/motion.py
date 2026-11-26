"""Step 2f: Own motion path for HarnessTree.

``state`` / ``move_to`` / ``set_gripper`` / ``home`` call the local colour toolkit
directly (``view_env_state``, ``move_pose``, ``grip``, ``home``) and return
policy-local results.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .exceptions import EpisodeFinished, MotionUnacknowledged, raise_for_error, unwrap
from .types import (
    TCP_OFFSET,
    ArmState,
    Gripper,
    Home,
    Move,
    State,
    _rotation,
    _tuple3,
    gripper_angles,
)


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _arm(name: Any) -> str:
    if name not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return str(name)


def _command(value: Any, *, allow_keep: bool) -> int:
    allowed = (-1, 0, 1) if allow_keep else (-1, 1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if number not in allowed:
        words = "1 (close), -1 (open) or 0 (keep)" if allow_keep else "1 (close) or -1 (open)"
        raise ValueError(f"gripper must be {words}, got {value!r}")
    return int(number)


def _scalar(value: Any, default: float = 0.0) -> float:
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return ()


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees or None, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _arm_state(raw: dict[str, Any], name: str) -> ArmState:
    pose = _sequence(raw.get(f"{name}_ee_pose"))
    if len(pose) != 7:
        pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    r = _rotation(pose[3:])
    tcp = tuple(pose[i] + r[i][0] * TCP_OFFSET for i in range(3))
    pitch, yaw, roll = gripper_angles(pose[3:])
    return ArmState(
        tcp_pos=tcp,  # type: ignore[arg-type]
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        eef_pos=pose[:3],  # type: ignore[arg-type]
        eef_quat=pose[3:],  # type: ignore[arg-type]
        gripper=_scalar(raw.get(f"{name}_ee_joint_state")),
        joints=_sequence(raw.get(f"{name}_arm_joint_state")),
    )


def _state_from_envelope(envelope: dict[str, Any]) -> State:
    summary = envelope.get("state") or {}
    raw = summary.get("state") or {}
    return State(
        step=int(envelope.get("step", -1)),
        task=str(envelope.get("task_language") or summary.get("instruction") or ""),
        left=_arm_state(raw, "left"),
        right=_arm_state(raw, "right"),
        truncated=bool(envelope.get("truncated")),
    )


class MotionSession:
    """Owns state + motion against a local NativeToolkit."""

    def __init__(self, toolkit: Any) -> None:
        self._toolkit = toolkit
        self._last_state: State | None = None
        self._truncated = False
        self._motion_error: str | None = None

    def _guard(self) -> None:
        if self._motion_error is not None:
            raise MotionUnacknowledged(self._motion_error)
        if self._truncated:
            raise EpisodeFinished("truncated")

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        if envelope is None:
            return
        state = _state_from_envelope(envelope)
        self._last_state = state
        if state.truncated:
            self._truncated = True

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _call(self, tool: str, kwargs: dict[str, Any], *, stateful: bool) -> dict[str, Any]:
        if stateful:
            self._guard()
        try:
            raw = self._toolkit.execute_tool(tool, kwargs)
            payload, envelope = unwrap(raw, tool)
            self._absorb(envelope)
            if envelope is not None and tool == "view_env_state":
                payload = envelope
            if payload.get("interrupted") or payload.get("code") == "tool_cancelled":
                self._truncated = True
                raise EpisodeFinished("cancelled")
            raise_for_error(tool, payload)
            if stateful and envelope is None:
                raise RuntimeError("missing stateful motion receipt")
        except Exception as exc:
            if not stateful:
                raise
            self._motion_error = f"{tool} failed without a usable receipt: {exc}"
            raise MotionUnacknowledged(self._motion_error) from exc
        return payload

    def state(self, step: int = -1) -> State:
        payload = self._call("view_env_state", {"step": int(step)}, stateful=False)
        state = _state_from_envelope(payload)
        if step == -1:
            self._last_state = state
            if state.truncated:
                self._truncated = True
        return state

    def move_to(
        self,
        xyz: Sequence[float],
        arm: str = "right",
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
    ) -> Move:
        arm_n = _arm(arm)
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "arm": arm_n,
            "gripper": command,
            "path": path,
        }
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        p = self._call("move_pose", kwargs, stateful=True)
        commanded = p.get("target_xyz")
        angles = p.get("final_angles") or {}
        return Move(
            arm=str(p.get("arm", arm_n)),
            target_xyz=_tuple3(commanded if commanded is not None else target),
            start_xyz=_tuple3(p.get("start_xyz")),
            final_xyz=_tuple3(p.get("final_xyz")),
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            reached=bool(p.get("reached")),
            path=str(p["path"]) if p.get("path") else None,
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            blocked=bool(p.get("blocked")),
            steps_used=int(p.get("steps_used", 0)),
            truncated=self._latest_state().truncated,
        )

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        arm_n = _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {"arm": arm_n, "gripper": command}
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
            kwargs["opening"] = fraction
        p = self._call("grip", kwargs, stateful=True)
        status = p.get("status") or {}
        state = self._latest_state()
        width = p.get("width_m")
        return Gripper(
            arm=arm_n,
            commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
            opening=float(p.get("opening", state.arm(arm_n).gripper)),
            holding=bool(p.get("holding")),
            width_m=float(width) if width is not None else None,
            steps_used=int(p.get("steps_used", 0)),
            step=int(status["step"]) if status.get("step") is not None else None,
            step_limit=(
                int(status["step_limit"]) if status.get("step_limit") is not None else None
            ),
            truncated=state.truncated,
        )

    def home(self, arm: str = "both") -> Home:
        if arm not in ("left", "right", "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        which = str(arm)
        p = self._call("home", {"arm": which}, stateful=True)
        return Home(
            arm=str(p.get("arm", which)),
            reached=bool(p.get("reached")),
            max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
            steps_used=int(p.get("steps_used", 0)),
            truncated=self._latest_state().truncated,
        )
