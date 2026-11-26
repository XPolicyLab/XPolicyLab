# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared, read-only projection of live and recorded agent trajectories."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    known = [r["usage"] for r in records if isinstance(r.get("usage"), dict)]
    totals = {
        key: sum(u.get(key) or 0 for u in known)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_output_tokens",
        )
    }
    cache_known = bool(known) and all(
        u.get("cache_read_tokens") is not None for u in known
    )
    return {
        **totals,
        "total_tokens": totals["input_tokens"] + totals["output_tokens"],
        "cache_ratio": totals["cache_read_tokens"] / totals["input_tokens"]
        if cache_known and totals["input_tokens"]
        else None,
        "attempts": len(records),
        "reported_usage_attempts": len(known),
        "unknown_usage_attempts": len(records) - len(known),
        "cost_usd": sum(u["cost_usd"] for u in known)
        if known and all(u.get("cost_usd") is not None for u in known)
        else None,
    }


class TrajectoryProjection:
    """Project each event once; attempt updates replace previous usage."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.agents: dict[str, dict[str, Any]] = {}
        self.turns: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}
        self.attempts: dict[tuple[str, int], dict[str, Any]] = {}
        self._turn_usage_seq: dict[str, int] = {}
        self._seen: set[str] = set()
        self.run_id: str | None = None
        self.last_seq = 0
        self.status = "partial"

    def apply(self, event: dict[str, Any]) -> None:
        """Apply one versioned event without summing cumulative SDK snapshots."""
        identity = str(event.get("event_id", event.get("seq")))
        if identity in self._seen:
            return
        self._seen.add(identity)
        self.events.append(event)
        self.last_seq = max(self.last_seq, int(event.get("seq", 0)))
        self.run_id = event.get("run_id", self.run_id)
        kind, payload = event.get("type"), event.get("payload") or {}
        agent_id, turn_id = event.get("agent_id"), event.get("turn_id")
        request_id = event.get("request_id")
        if kind == "run_end":
            self.status = payload.get("status", "completed")
        if agent_id:
            agent = self.agents.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "parent_agent_id": event.get("parent_agent_id"),
                    "name": payload.get("name", agent_id),
                    "status": "running",
                },
            )
            if kind == "agent_start":
                agent.update(payload)
            elif kind == "agent_end":
                agent["status"] = payload.get("status", "completed")
        if turn_id:
            turn = self.turns.setdefault(
                turn_id,
                {
                    "turn_id": turn_id,
                    "agent_id": agent_id,
                    "seq": event.get("seq"),
                    "status": "running",
                    "events": [],
                    "request_ids": [],
                },
            )
            turn["events"].append(event)
            turn["last_seq"] = event.get("seq", 0)
            if request_id and request_id not in turn["request_ids"]:
                turn["request_ids"].append(request_id)
            if kind == "turn_end":
                turn["status"] = payload.get("status", "completed")
            if kind == "turn_start":
                turn["elapsed_s"] = event.get("elapsed_s")
        if kind == "model_request_start" and request_id:
            self.requests[request_id] = {
                **payload,
                "agent_id": agent_id,
                "turn_id": turn_id,
            }
        if kind in {"model_attempt_start", "model_attempt_end"} and request_id:
            key = (request_id, int(event.get("attempt") or 1))
            attempt = {
                **payload,
                "request_id": request_id,
                "agent_id": agent_id,
                "turn_id": turn_id,
                "purpose": payload.get("purpose", event.get("purpose")),
            }
            if kind == "model_attempt_start":
                self.attempts.setdefault(
                    key, {**attempt, "status": "running", "usage": None}
                )
            else:
                self.attempts[key] = {**self.attempts.get(key, {}), **attempt}
                if turn_id:
                    self._turn_usage_seq[turn_id] = int(event.get("seq", 0))

    def summary(self) -> dict[str, Any]:
        """Return totals and per-agent/model/purpose breakdowns for both UIs."""
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for attempt in self.attempts.values():
            request = self.requests.get(attempt["request_id"], {})
            key = (
                attempt.get("agent_id") or "unassigned",
                request.get("model") or attempt.get("model") or "unknown",
                attempt.get("purpose") or request.get("purpose") or "agent",
            )
            groups.setdefault(key, []).append(attempt)
        return {
            "run_id": self.run_id,
            "last_seq": self.last_seq,
            "status": self.status,
            "n_turns": len(self.turns),
            "n_requests": len(self.requests),
            "usage": _usage(list(self.attempts.values())),
            "groups": [
                {
                    "agent_id": key[0],
                    "model": key[1],
                    "purpose": key[2],
                    "usage": _usage(values),
                }
                for key, values in groups.items()
            ],
        }

    def index(self, after_seq: int = 0) -> dict[str, Any]:
        """Return updated turn summaries, leaving message bodies lazy."""
        turns = []
        cumulative = 0
        cumulative_seq = 0
        for turn in self.turns.values():
            usage = _usage(
                [
                    a
                    for a in self.attempts.values()
                    if a.get("turn_id") == turn["turn_id"]
                ]
            )
            cumulative += usage["total_tokens"]
            # A parallel request may complete after a later turn was already
            # sent to the UI. Its usage changes every subsequent cumulative point.
            cumulative_seq = max(
                cumulative_seq, self._turn_usage_seq.get(turn["turn_id"], 0)
            )
            if max(turn["last_seq"], cumulative_seq) > after_seq:
                turns.append(
                    {
                        **{k: v for k, v in turn.items() if k != "events"},
                        "usage": usage,
                        "cumulative_tokens": cumulative,
                    }
                )
        return {
            "summary": self.summary(),
            "agents": list(self.agents.values()),
            "turns": turns,
        }


class TrajectoryReader:
    """Incrementally consume complete JSONL lines from one run directory."""

    def __init__(
        self, output_dir: str | Path, *, trace_manifest: str = "trace/manifest.json"
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self._manifest_path = self.artifact_path(trace_manifest)
        self._events_path = self._manifest_path.parent / "events.jsonl"
        self.projection = TrajectoryProjection()
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._run_id: str | None = None

    def artifact_path(self, reference: str) -> Path:
        """Resolve a run-relative artifact, rejecting absolute or escaped paths."""
        relative = Path(reference)
        path = (self.output_dir / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.output_dir):
            raise ValueError("artifact reference is outside the run directory")
        return path

    def update(self) -> TrajectoryProjection:
        """Read appended complete events; retain unfinished tails for later."""
        path = self._events_path
        if not path.exists():
            return self.projection
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        manifest = self.manifest()
        run_id = manifest.get("run_id")
        if (
            self._identity not in (None, identity)
            or stat.st_size < self._offset
            or (self._run_id is not None and run_id != self._run_id)
        ):
            self._offset = 0
            self.projection = TrajectoryProjection()
        self._identity, self._run_id = identity, run_id
        with path.open("rb") as stream:
            stream.seek(self._offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(
                        event.get("type"), str
                    ):
                        raise ValueError("expected event object with type")
                    self.projection.apply(event)
                except (UnicodeError, ValueError, TypeError) as exc:
                    raise ValueError(
                        f"invalid trajectory event at byte {self._offset}"
                    ) from exc
                self._offset = stream.tell()
        return self.projection

    def manifest(self) -> dict[str, Any]:
        """Read an atomically published manifest when available."""
        path = self._manifest_path
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def turn(self, turn_id: str) -> dict[str, Any]:
        """Load a turn's actual request/response/tool payload artifacts."""
        turn = copy.deepcopy(self.projection.turns[turn_id])
        for event in turn["events"]:
            payload = event.get("payload") or {}
            for key, reference in list(payload.items()):
                if key.endswith("_ref") and isinstance(reference, str):
                    path = self.artifact_path(reference)
                    if path.suffix == ".json":
                        payload[key.removesuffix("_ref")] = (
                            json.loads(path.read_text(encoding="utf-8"))
                            if path.exists()
                            else {"unavailable": reference}
                        )
        return turn

    def media(self) -> list[dict[str, Any]]:
        """Project existing robot artifacts without guessing video timestamps."""
        path = self.output_dir / "states.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        items = []
        for name in state.get("run_artifacts", []):
            items.append({"artifact_ref": name})
        for step in state.get("steps", []):
            extras = step.get("extras", {})
            for name in step.get("artifacts", []):
                items.append(
                    {
                        "artifact_ref": f"{name}/{step['step_idx']:02d}{Path(name).suffix}",
                        "step_idx": step["step_idx"],
                        "command": step.get("command"),
                        **extras,
                    }
                )
            for video in extras.get("videos", []):
                items.append(
                    {
                        "artifact_ref": video["video_ref"],
                        "segments": [
                            {
                                **video,
                                "step_idx": step["step_idx"],
                                "tool_call_id": extras.get("tool_call_id"),
                                "turn_id": extras.get("turn_id"),
                            }
                        ],
                    }
                )
        for event in self.projection.events:
            if event.get("type") in {"artifact_written", "artifact_failed"}:
                payload = event.get("payload") or {}
                reference = payload.get("artifact_ref")
                if not reference and payload.get("name"):
                    name = payload["name"]
                    step = payload.get("step_idx", event.get("step_idx"))
                    reference = (
                        name
                        if step is None
                        else f"{name}/{step:02d}{Path(name).suffix}"
                    )
                if reference:
                    items.append(
                        {
                            **payload,
                            "artifact_ref": reference,
                            "tool_call_id": event.get("tool_call_id"),
                            "turn_id": event.get("turn_id"),
                            "status": "failed"
                            if event["type"] == "artifact_failed"
                            else "ready",
                            "error": payload.get("error"),
                        }
                    )
                segment_fields = ("video_ref", "frame_start", "frame_end", "fps")
                if all(payload.get(field) is not None for field in segment_fields):
                    items.append(
                        {
                            "artifact_ref": payload["video_ref"],
                            "segments": [
                                {
                                    **{
                                        field: payload[field]
                                        for field in segment_fields
                                    },
                                    "step_idx": payload.get(
                                        "step_idx", event.get("step_idx")
                                    ),
                                    "tool_call_id": event.get("tool_call_id"),
                                    "turn_id": event.get("turn_id"),
                                }
                            ],
                        }
                    )
        unique = {}
        for item in items:
            reference = item["artifact_ref"]
            artifact = self.artifact_path(reference)
            if artifact.suffix.lower() not in {
                ".mp4",
                ".webm",
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            } and item.get("media_type") not in {
                "image/png",
                "image/jpeg",
                "image/webp",
                "video/mp4",
                "video/webm",
            }:
                continue
            previous = unique.get(reference, {})
            failed = item.get("status", previous.get("status")) == "failed"
            available = artifact.is_file() and not failed
            if failed:
                status = "failed"
            elif available:
                status = "ready"
            elif (
                artifact.suffix.lower() in {".mp4", ".webm"}
                and self.projection.status == "partial"
            ):
                status = "pending"
            else:
                status = "unavailable"
            unique[reference] = {
                **previous,
                **item,
                "available": available,
                "segments": previous.get("segments", []) + item.get("segments", []),
                "status": status,
            }
        return list(unique.values())

    def index(self, after_seq: int = 0) -> dict[str, Any]:
        """Return the common trajectory response for export and Dashboard."""
        self.update()
        result = self.projection.index(after_seq)
        manifest = self.manifest()
        if manifest.get("status") == "incomplete":
            result["summary"]["status"] = "incomplete"
        return {**result, "manifest": manifest, "media": self.media()}
