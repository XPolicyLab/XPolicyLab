"""Shared task context: notes written by the Task-Level Agent and geometric references.

References keep their observation provenance and do not track object motion.
The store only flags references a gripper interacted near, so the agent knows
when a measurement should be refreshed.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

REFERENCE_KINDS = ("object", "point", "pose", "axis_tip", "support_plane", "opening", "relation", "grasp")


@dataclass
class Reference:
    ref_id: str
    kind: str
    label: str
    values: dict[str, Any]
    provenance: dict[str, Any]
    created_tick: int
    possibly_moved: bool = False
    moved_reason: str = ""
    invalidated: bool = False

    def position(self) -> Optional[np.ndarray]:
        for key in ("center_xyz", "tip_xyz", "xyz", "grasp_xyz"):
            if key in self.values:
                return np.asarray(self.values[key], dtype=float)
        return None

    def summary(self, now_tick: int) -> dict:
        out = {"ref": self.ref_id, "kind": self.kind, "label": self.label, "age_ticks": now_tick - self.created_tick,
               "source": {k: self.provenance[k] for k in ("camera", "obs_id", "method") if k in self.provenance}}
        out.update(_compact(self.values))
        if self.possibly_moved:
            out["warning"] = "possibly moved since measured: " + self.moved_reason
        if self.invalidated:
            out["warning"] = "invalidated: " + self.moved_reason
        return out


def _compact(values: dict) -> dict:
    keep = ("center_xyz", "top_z", "height_m", "length_m", "width_m", "grasp_yaw_deg", "long_axis_deg",
            "tip_xyz", "axis_dir", "z", "radius_m", "rim_z", "xyz", "quat_wxyz", "relation", "status",
            "grasp_xyz", "yaw_deg", "pitch_deg", "arm")
    out = {}
    for key in keep:
        if key in values:
            value = values[key]
            if isinstance(value, float):
                value = round(value, 4)
            elif isinstance(value, (list, tuple)):
                value = [round(v, 4) if isinstance(v, float) else v for v in value]
            out[key] = value
    return out


class ReferenceStore:
    def __init__(self, interaction_radius_m: float = 0.06):
        self._lock = threading.Lock()
        self._refs: dict[str, Reference] = {}
        self._ids = itertools.count(1)
        self.interaction_radius_m = interaction_radius_m

    def add(self, kind: str, label: str, values: dict, provenance: dict, tick: int, ref_id: Optional[str] = None) -> Reference:
        if kind not in REFERENCE_KINDS:
            raise ValueError(f"unknown reference kind {kind!r}")
        with self._lock:
            rid = ref_id or f"r{next(self._ids)}"
            ref = Reference(rid, kind, label, dict(values), dict(provenance), tick)
            self._refs[rid] = ref
            return ref

    def get(self, ref_id: str) -> Reference:
        with self._lock:
            if ref_id not in self._refs:
                raise KeyError(f"unknown reference {ref_id!r}; known: {sorted(self._refs)[-12:]}")
            return self._refs[ref_id]

    def resolve_point(self, target: Any) -> np.ndarray:
        """A reference id or an explicit [x, y, z] -> world point."""
        if isinstance(target, str):
            point = self.get(target).position()
            if point is None:
                raise ValueError(f"reference {target!r} has no position")
            return point
        array = np.asarray(target, dtype=float).reshape(-1)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError("target must be a reference id or [x, y, z]")
        return array

    def note_interaction(self, xyz: Sequence[float], reason: str) -> list[str]:
        """Flag references near a gripper open/close or contact; returns the flagged ids."""
        point = np.asarray(xyz, dtype=float)
        flagged = []
        with self._lock:
            for ref in self._refs.values():
                pos = ref.position()
                if pos is None or ref.invalidated:
                    continue
                if float(np.linalg.norm(pos[:2] - point[:2])) <= self.interaction_radius_m:
                    ref.possibly_moved = True
                    ref.moved_reason = reason
                    flagged.append(ref.ref_id)
        return flagged

    def invalidate(self, ref_id: str, reason: str) -> None:
        ref = self.get(ref_id)
        ref.invalidated = True
        ref.moved_reason = reason

    def summaries(self, now_tick: int, limit: int = 24) -> list[dict]:
        with self._lock:
            refs = [r for r in self._refs.values() if not r.invalidated]
        return [r.summary(now_tick) for r in refs[-limit:]]


@dataclass
class Note:
    key: str
    text: str
    author: str
    tick: int


class SharedNotes:
    """Task-level notes. Keys overwrite; order is kept."""

    def __init__(self, max_chars: int = 4000):
        self._lock = threading.Lock()
        self._notes: dict[str, Note] = {}
        self.max_chars = max_chars

    def write(self, key: str, text: str, author: str, tick: int) -> Note:
        key = str(key).strip()[:64] or "note"
        text = str(text).strip()
        if len(text) > 1200:
            raise ValueError("a note is limited to 1200 characters; split it or summarise")
        with self._lock:
            self._notes.pop(key, None)
            note = Note(key, text, author, tick)
            self._notes[key] = note
            while sum(len(n.text) for n in self._notes.values()) > self.max_chars and len(self._notes) > 1:
                self._notes.pop(next(iter(self._notes)))
            return note

    def items(self) -> list[Note]:
        with self._lock:
            return list(self._notes.values())

    def render(self) -> str:
        notes = self.items()
        if not notes:
            return "(none)"
        return "\n".join(f"- [{n.key}] (tick {n.tick}) {n.text}" for n in notes)


@dataclass
class TaskContext:
    """Everything both agents share during one episode."""

    instruction: str
    notes: SharedNotes = field(default_factory=SharedNotes)
    references: ReferenceStore = field(default_factory=ReferenceStore)
