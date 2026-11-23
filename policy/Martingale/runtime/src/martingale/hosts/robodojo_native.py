# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import io
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import robodojo_tools as tools

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
from ..inference import recipe_handlers
STATEFUL = frozenset({"move_pose", "grip", "home"}) | recipe_handlers.STATEFUL


class Cancelled(Exception):
    """The episode was stopped while a tool was running."""


class _Primitives:
    """What the tools of :mod:`.robodojo_tools` are bound to."""

    def __init__(self, env: Any, sam3_client: Any, colour_only: bool = True) -> None:
        self.env = env
        self.sam3_client = sam3_client
        self.colour_only = colour_only
        self._last_obs: dict[str, Any] | None = None
        self.cancelled = threading.Event()

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()


def _png(image: Any, *, fast: bool = True) -> bytes:
    from PIL import Image

    rgb = np.asarray(image)[..., :3].astype(np.uint8)
    if fast and rgb.ndim == 3 and rgb.shape[-1] == 3:
        try:
            import cv2
        except ImportError:
            pass
        else:
            ok, encoded = cv2.imencode('.png', np.ascontiguousarray(rgb[..., ::-1]),
                [cv2.IMWRITE_PNG_COMPRESSION, 1,
                 cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE])
            if not ok:
                raise RuntimeError('PNG encoding failed')
            return encoded.tobytes()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1 if fast else 6)
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
        stopped = tools._spent(status)
        cameras = {}
        for name, cam in (obs.get("vision") or {}).items():
            cam = cam or {}
            pose = tools.extrinsic_of(cam)
            if cam.get("intrinsic_matrix") is None and pose is None:
                continue
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
                "truncated": stopped,
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
    primitives: Any, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2,
    include_contours: bool = False,
) -> dict[str, Any]:
    """Segment an object by words in the current image of one camera."""

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
    extra = {}
    if include_contours:
        import cv2
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        extra["contours_rc"] = [c.reshape(-1, 2)[:, ::-1].tolist() for c in contours]
    return {
        **extra,
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
            "fit_blocks": partial(tools.fit_blocks, self._primitives),
            "fit_tube_openings": partial(tools.fit_tube_openings, self._primitives),
            "check_tube_alignment": partial(tools.check_tube_alignment, self._primitives),
            "back_project": partial(tools.back_project, self._primitives),
            "segment": partial(segment, self._primitives),
        }
        self._tools.update({'fit_mugs': partial(tools.fit_mugs, self._primitives), 'measure_disks': partial(tools.measure_disks, self._primitives), 'measure_key': partial(tools.measure_key, self._primitives), 'measure_wedge': partial(tools.measure_wedge, self._primitives), 'fit_basket': partial(tools.fit_basket, self._primitives), 'track_basket': partial(tools.track_basket, self._primitives)})
        self._tools.update({'fit_pan': partial(tools.fit_pan, self._primitives), 'fit_peg_landmarks': partial(tools.fit_peg_landmarks, self._primitives), 'fit_peg': partial(tools.fit_peg, self._primitives)})
        self._tools.update(recipe_handlers.register(self._primitives, start))
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
    """A :class:`NativeToolkit` as a :class:`martingale.Backend`."""

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
