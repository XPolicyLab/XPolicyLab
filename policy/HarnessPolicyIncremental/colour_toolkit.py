# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Adapted for HarnessPolicyIncremental; see NOTICE.
"""A RoboDojo toolkit that needs nothing but an environment.

RPent's ``RoboDojoToolkit`` wraps RPent's simulator bridge and records RPent's artifacts.  Under
RoboDojo's own evaluation client there is no RPent: there is an observation (three colour
images and the robot's state, with a depth image and the cameras' matrices when the client's
configuration switches them on) and an action that takes joint targets.
:class:`NativeToolkit` is what the robot class needs on top of that and no more: the tools of
:mod:`.colour_tools` (through the depth image where the observation has one, from colour
alone where it has not), ``segment``, the recorded steps (``view_env_state``, the images and
the cameras' calibration of each step) and the envelope RPent's toolkit wraps a
state-changing tool's result in, which the policy unwraps.

``env`` is anything with ``get_obs() -> dict`` (``vision`` / ``state`` / ``instruction``),
``step(action) -> (obs, reward, done, info)`` with ``info["status"]`` holding ``step`` and
``step_limit``, and ``get_status()``.  ``sam3_client`` segments an image by words or by a pixel
(RPent's ``Sam3Client`` interface).
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import colour_tools as tools

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATEFUL = frozenset({"move_pose", "grip", "home"})


class Cancelled(Exception):
    """The episode was stopped while a tool was running."""


class _Primitives:
    """What the tools of :mod:`.colour_tools` are bound to."""

    def __init__(self, env: Any, sam3_client: Any, colour_only: bool = True) -> None:
        self.env = env
        self.sam3_client = sam3_client
        #: no depth image is read, even where the observation has one
        self.colour_only = colour_only
        self._last_obs: dict[str, Any] | None = None
        self.cancelled = threading.Event()

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()


def _png(image: Any) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _plain(value: Any) -> Any:
    """A state entry as plain lists and numbers (it is written to JSON and read back)."""

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Steps:
    """The recorded steps of an episode: the state in memory, the images on disk."""

    def __init__(self, out_dir: Path) -> None:
        self.root = Path(out_dir) / "steps"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def record(self, obs: dict[str, Any], status: dict[str, Any]) -> int:
        step = len(self.records)
        folder = self.root / f"{step:04d}"
        folder.mkdir(exist_ok=True)
        for name, cam in (obs.get("vision") or {}).items():
            color = (cam or {}).get("color")
            if color is not None:
                (folder / f"{name}.png").write_bytes(_png(color))
        try:
            spent = int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30))
        except (TypeError, ValueError):
            spent = False
        cameras = {}
        for name, cam in (obs.get("vision") or {}).items():
            cam = cam or {}
            pose = tools.extrinsic_of(cam)
            if cam.get("intrinsic_matrix") is None and pose is None:
                continue  # an observation without calibration: nothing to record
            cameras[name] = {
                "shape": _plain(cam.get("shape")),
                "has_depth": tools.depth_image(cam) is not None,
                "intrinsic_matrix": _plain(cam.get("intrinsic_matrix")),
                "extrinsic_matrix": _plain(pose),
            }
        self.records.append(
            {
                "state": _plain(obs.get("state") or {}),
                "instruction": str(obs.get("instruction") or obs.get("instructions") or ""),
                "truncated": bool(spent or status.get("over")),
                "cameras": cameras,
            }
        )
        return step

    def index(self, step: int) -> int:
        if not self.records:
            raise IndexError("no step recorded")
        index = len(self.records) + step if step < 0 else step
        if not 0 <= index < len(self.records):
            raise IndexError(f"step {step} is not recorded (0 .. {len(self.records) - 1})")
        return index

    def artifact_path(self, name: str, step: int = -1) -> Path:
        return self.root / f"{self.index(step):04d}" / name

    def load_bytes(self, name: str, step: int = -1) -> bytes:
        return self.artifact_path(name, step).read_bytes()


def segment(
    primitives: Any, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
) -> dict[str, Any]:
    """Segment an object by words in the current image of one camera (RPent's tool, as is)."""

    obs = tools._obs(primitives)
    cam = (obs.get("vision") or {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return {"error": "sam3_client not configured"}
    res = sam3.segment(color, text_prompt=text_prompt, min_score=float(min_score))
    if not res.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    count = int(rows.size)
    return {
        "camera": camera,
        "found": True,
        "score": float(res.score) if res.score is not None else None,
        "box_px": [float(v) for v in res.box] if res.box is not None else None,
        "mask_shape": list(mask.shape),
        "mask_moments": [count, int(rows.sum()), int(cols.sum())],
        "centroid_rc": [int(rows.sum()) // count, int(cols.sum()) // count] if count else None,
    }


class NativeToolkit:
    """The robot's tools over a bare environment, with the steps they record."""

    def __init__(
        self,
        env: Any,
        sam3_client: Any,
        out_dir: Path,
        verdict: Callable[[], bool] | None = None,
        colour_only: bool | None = None,
    ) -> None:
        obs = env.get_obs()
        if colour_only is None:
            # what the observation offers decides: depth where the client sends it
            head = (obs.get("vision") or {}).get("cam_head") or {}
            colour_only = tools.depth_image(head) is None
        self._primitives = _Primitives(env, sam3_client, colour_only)
        self.state = Steps(out_dir)
        self._verdict = verdict
        self._lock = threading.Lock()
        self._primitives._last_obs = obs
        start = {arm: tools._joints(obs.get("state") or {}, arm) for arm in tools.ARMS}
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "move_pose": partial(tools.move_pose, self._primitives, start),
            "grip": partial(tools.grip, self._primitives),
            "home": partial(tools.home, self._primitives, start),
            "locate": partial(tools.locate, self._primitives),
            "back_project": partial(tools.back_project, self._primitives),
            "segment": partial(segment, self._primitives),
        }
        self.state.record(obs, self._status())

    def _status(self) -> dict[str, Any]:
        try:
            status = self._primitives.env.get_status()
        except Exception:  # noqa: BLE001 - optional metadata
            return {}
        return status if isinstance(status, dict) else {}

    def _envelope(self, step: int, tool: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        index = self.state.index(step)
        record = self.state.records[index]
        return {
            "step": index,
            "task_language": record["instruction"],
            "terminated": False,
            "truncated": record["truncated"],
            "state": {
                "state": record["state"],
                "instruction": record["instruction"],
                "cameras": record["cameras"],
            },
            "log": {"tool": tool, "result": payload},
        }

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Run one tool; a state-changing one answers with the step it recorded."""

        kwargs = dict(input_dict or {})
        if name == "view_env_state":
            try:
                return self._envelope(int(kwargs.get("step", -1)), name, {})
            except IndexError as exc:
                return {"error": str(exc)}
        handler = self._tools.get(name)
        if handler is None:
            return {"error": f"this RoboDojo host has no tool {name!r}"}
        with self._lock:
            try:
                payload = handler(**kwargs)
            except Cancelled:
                payload = {"interrupted": True, "code": "tool_cancelled"}
            except TypeError as exc:
                return {"error": f"{name}: {exc}"}
            if name not in STATEFUL:
                return payload
            obs = tools._obs(self._primitives)
            step = self.state.record(obs, self._status())
            return self._envelope(step, name, payload)

    def conclude(self) -> None:
        """The policy has finished: let the environment judge the scene as it stands."""

        conclude = getattr(self._primitives.env, "conclude", None)
        if callable(conclude):
            conclude()

    def solved(self) -> bool:
        return bool(self._verdict()) if self._verdict is not None else False

    def episode_over(self) -> bool:
        """Whether the environment's owner has ended the episode (a client that judges it)."""

        over = getattr(self._primitives.env, "episode_over", None)
        return bool(over()) if callable(over) else False

    def cancel_active_and_wait(self) -> None:
        self._primitives.cancelled.set()

    def close(self) -> None:
        self._primitives.cancelled.set()


class NativeBackend:
    """A :class:`NativeToolkit` as a backend adapter."""

    def __init__(self, toolkit: NativeToolkit) -> None:
        self._toolkit = toolkit

    @property
    def toolkit(self) -> NativeToolkit:
        return self._toolkit

    @property
    def colour_only(self) -> bool:
        return bool(self._toolkit._primitives.colour_only)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def conclude(self) -> None:
        self._toolkit.conclude()

    def solved(self) -> bool:
        return self._toolkit.solved()

    def episode_over(self) -> bool:
        return self._toolkit.episode_over()

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.state.load_bytes(name, step=step)

    def env_steps(self) -> int | None:
        step = self._toolkit._status().get("step")
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    def cancel(self) -> None:
        self._toolkit.cancel_active_and_wait()


__all__ = ["CAMERAS", "Cancelled", "NativeBackend", "NativeToolkit", "Steps", "segment"]
