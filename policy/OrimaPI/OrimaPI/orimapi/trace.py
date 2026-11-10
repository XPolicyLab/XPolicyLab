# coding=utf-8
"""Append-only events and keyframe files for one run; no database required."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .state import RGBImage, SupervisorRequest, image_media_type, json_value


class TraceWriter:
    def __init__(self, root: str | Path, *, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=resume)
        events = self.root / "events.jsonl"
        sequence = 0
        if resume:
            with events.open() as stream:
                for line in stream:
                    if json.loads(line)["sequence"] != sequence:
                        raise ValueError("Invalid event sequence in resumed trace")
                    sequence += 1
        self._stream = events.open("a" if resume else "x", encoding="utf-8")
        self._lock = threading.Lock()
        self._sequence = sequence
        self._context = {}
        self._closed = False

    def write(self, event: str, **fields) -> None:
        if not event or {"event", "sequence", "time"} & fields.keys():
            raise ValueError("Invalid trace event fields")
        payload = json_value(fields)
        with self._lock:
            if self._closed:
                raise RuntimeError("The trace writer is closed")
            row = {
                **self._context,
                "event": event,
                "sequence": self._sequence,
                "time": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            self._stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
            self._stream.flush()
            self._sequence += 1

    def set_context(self, **fields):
        if fields.keys() - {"suite_id", "job_id", "attempt_id", "task", "seed"}:
            raise ValueError("Unknown trace context fields")
        with self._lock:
            self._context = json_value(fields)

    def write_json(self, name: str, value: dict) -> None:
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("A JSON filename without directories is required")
        payload = json_value(value)
        with self._lock:
            if self._closed:
                raise RuntimeError("The trace writer is closed")
            with (self.root / name).open("x", encoding="utf-8") as stream:
                json.dump(
                    payload, stream, ensure_ascii=False, indent=2, allow_nan=False
                )
                stream.write("\n")

    def record_request(self, request: SupervisorRequest, *, save_frames=True) -> dict:
        """Persist precisely the image bytes submitted to the high-level model."""
        key = hashlib.sha256(request.request_id.encode()).hexdigest()
        directory = self.root / "frames" / key
        if save_frames:
            directory.mkdir(parents=True, exist_ok=False)
        frames = {}
        for index, (camera, image) in enumerate(request.observation.images.items()):
            if not save_frames:
                continue
            if isinstance(image, RGBImage):
                image = image.to_png()
            extension = ".png" if image_media_type(image) == "image/png" else ".jpg"
            path = directory / f"camera-{index:03d}{extension}"
            path.write_bytes(image)
            frames[camera] = {
                "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(image).hexdigest(),
            }
        self.write(
            "supervisor_requested",
            request_id=request.request_id,
            episode_id=request.episode_id,
            env_id=request.observation.env_id,
            observation_step=request.observation.step,
            plan_version=request.plan_version,
            instruction=request.observation.instruction,
            robot_state=request.observation.robot_state,
            memory=request.memory,
            request_kind=(
                "policy_selection" if request.candidates is not None else "subgoal"
            ),
            active_policy=request.policy,
            available_policies=request.candidates,
            frames=frames,
        )
        return frames

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.close()
                self._closed = True

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
