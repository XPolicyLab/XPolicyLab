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
import time
from collections.abc import Callable
from typing import Any, ClassVar

from ._backend import Backend, EpisodeFinished, Ledger, ToolError, raise_for_error, unwrap


class RobotBase:
    """Shared machinery of every robot class (see the module docstring)."""

    NAME: ClassVar[str] = ""
    CAMERAS: ClassVar[tuple[str, ...]] = ()
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ()
    SHOW_RESOLUTION: ClassVar[str] = "high"
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset()
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "terminated",
        "truncated",
        "success",
        "final_dist_m",
        "final_eef_pos",
        "steps_used",
        "found",
        "score",
        "world_xyz",
        "center_xyz",
        "step",
    )
    STATE_TOOL: ClassVar[str] = "view_env_state"
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, backend: Backend, *, ledger: Ledger | None = None):
        self._backend = backend
        self._ledger = ledger if ledger is not None else Ledger()
        self._last_state: Any = None
        self._done = False
        self._truncated = False
        self._stop_reason: str | None = None
        self._show_requests: list[str] = []


    def _take_show_requests(self) -> list[str]:
        """Return and clear the cameras requested through :meth:`show`."""
        requests, self._show_requests = self._show_requests, []
        return requests

    def _show_artifacts(self, cameras: list[str]) -> list[str]:
        """Artifact names to attach for ``show`` requests."""
        return [
            self.IMAGE_ARTIFACTS[(camera, self.SHOW_RESOLUTION)]
            for camera in cameras
            if (camera, self.SHOW_RESOLUTION) in self.IMAGE_ARTIFACTS
        ]

    def _reported_success(self) -> bool:
        """Whether the benchmark has reported task success during the episode.

        The hosts ask this instead of ``done``: a robot whose benchmark judges
        only the final scene has no ``done`` at all and answers False here.
        """
        return self._done or bool(self._backend.solved())

    def _stop_episode(self, reason: str) -> None:
        """Make every further motion call raise ``EpisodeFinished``."""
        self._stop_reason = reason
        try:
            self._backend.cancel()
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass


    def _state_from_envelope(self, envelope: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _summary_dict(self) -> dict[str, Any]:
        """A few numbers describing the current state (host-facing, shown per turn)."""
        raise NotImplementedError


    @property
    def task(self) -> str:
        """The task instruction given by the benchmark (authoritative)."""
        return str(getattr(self._current_state(), "task", ""))

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate has fired (sticky)."""
        return self._reported_success()

    def state(self, step: int = -1) -> Any:
        """Read the recorded proprioceptive state of step ``step`` (``-1`` = latest)."""
        payload = self._readonly(self.STATE_TOOL, {"step": int(step)})
        state = self._state_from_envelope(payload)
        if step == -1:
            self._last_state = state
        return state

    def image(self, camera: str, *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (H, W, 3), RGB.

        Pixel coordinates are (row, col) with row 0 at the top and col 0 at the
        left; pass the same ``camera`` and ``resolution`` to the back-projection
        helpers.  ``step`` selects the recorded step (``-1`` = latest).
        """
        data = self.artifact(self._image_name(camera, resolution), step=step)
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))

    def world_map(self, camera: str, *, resolution: str = "high", step: int = -1) -> Any:
        """Load the per-pixel world coordinates of a recorded image.

        Returns a ``numpy`` float array (H, W, 3): ``world_map[row, col]`` is
        the [x, y, z] of the surface seen at that pixel (zeros / non-finite
        where depth is invalid).  Load it to reason about many pixels at once.
        """
        key = (camera, resolution)
        if key not in self.WORLD_MAP_ARTIFACTS:
            raise ValueError(
                f"no world map for camera={camera!r} resolution={resolution!r}; "
                f"available: {sorted(self.WORLD_MAP_ARTIFACTS)}"
            )
        data = self.artifact(self.WORLD_MAP_ARTIFACTS[key], step=step)
        import numpy as np

        loaded = np.load(io.BytesIO(data))
        if isinstance(loaded, np.ndarray):
            return loaded
        with loaded as archive:
            name = "array" if "array" in archive.files else archive.files[0]
            return np.asarray(archive[name])

    def show(self, camera: str) -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically: call ``show(<camera>)`` when
        *you* (the author) need to look at the scene before writing the next
        turn.  Costs tokens: look only when a decision depends on it.
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        if camera not in self._show_requests:
            self._show_requests.append(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact by name (advanced)."""
        record = self._ledger.open("artifact", {"name": name, "step": step}, stateful=False)
        started = time.perf_counter()
        try:
            data = self._backend.artifact(name, step=int(step))
            record.ok = True
            record.summary = {"bytes": len(data)}
            return data
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started


    @classmethod
    def _image_name(cls, camera: str, resolution: str) -> str:
        key = (camera, resolution)
        if key not in cls.IMAGE_ARTIFACTS:
            raise ValueError(
                f"camera must be one of {cls.CAMERAS} and resolution one of "
                f"{sorted({r for _, r in cls.IMAGE_ARTIFACTS})}"
            )
        return cls.IMAGE_ARTIFACTS[key]

    def _current_state(self) -> Any:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _guard(self) -> None:
        if self._stop_reason is not None:
            raise EpisodeFinished(self._stop_reason)
        if self._done:
            raise EpisodeFinished("terminated")
        if self._truncated:
            raise EpisodeFinished("truncated")

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Record the latest state and latch the episode flags.

        ``terminated`` / ``truncated`` come from the envelope and from the
        parsed state when it has attributes of those names, so a robot whose
        success flag the host names differently (``eval_success``) maps them in ``_state_from_envelope``
        instead of overriding this method.
        """
        if envelope is None:
            return
        state = self._state_from_envelope(envelope)
        self._last_state = state
        if envelope.get("terminated") or getattr(state, "terminated", False):
            self._done = True
        if envelope.get("truncated") or getattr(state, "truncated", False):
            self._truncated = True

    def _summary(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: payload[key] for key in self.SUMMARY_KEYS if key in payload}

    def _readonly(
        self, tool: str, kwargs: dict[str, Any], *, allow_error_key: str | None = None
    ) -> dict[str, Any]:
        record = self._ledger.open(tool, kwargs, stateful=False)
        started = time.perf_counter()
        try:
            payload, envelope = unwrap(self._backend.execute_tool(tool, kwargs), tool)
            if envelope is not None:
                self._absorb(envelope)
                payload = envelope
            if allow_error_key is None or allow_error_key not in payload:
                raise_for_error(tool, payload)
            record.ok = True
            record.summary = self._summary(tool, payload)
            return payload
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started

    def _stateful(
        self,
        tool: str,
        kwargs: dict[str, Any],
        build: Callable[[dict[str, Any]], Any],
    ) -> Any:
        self._guard()
        record = self._ledger.open(tool, kwargs, stateful=True)
        steps_before = self._backend.env_steps()
        started = time.perf_counter()
        try:
            payload, envelope = unwrap(self._backend.execute_tool(tool, kwargs), tool)
            self._absorb(envelope)
            if payload.get("interrupted") or payload.get("code") == "tool_cancelled":
                raise EpisodeFinished(self._stop_reason or "cancelled")
            raise_for_error(tool, payload)
            result = build(payload)
            record.ok = True
            record.summary = self._summary(tool, payload)
            return result
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started
            steps_after = self._backend.env_steps()
            if steps_before is not None and steps_after is not None:
                record.env_steps = max(0, steps_after - steps_before)


__all__ = ["RobotBase", "ToolError"]
