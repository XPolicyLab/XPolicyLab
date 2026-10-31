"""Deterministic, bounded evidence for one unreviewed execution interval.

This is a telemetry summary, not an object/state estimator or an LLM scratchpad.
Candidate diagnostics describe commands; only optional measured endpoints describe
observations. No native outcome, task label, free text or hidden simulator field is
copied. The owner records completed chunks and drains at each review packet; none
mode must not create or feed this object. There are no model calls or file writes.
"""
from __future__ import annotations

import copy
import math
import re
from collections import deque
from collections.abc import Mapping

SCHEMA = "oeh.interval-memory.v1"
_EVENT_TYPES = ("partial_execution", "no_execution", "source_gap",
                "commanded_gripper_change", "proposed_gripper_transition")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label):
    try:
        finite = not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite numeric telemetry")
    return float(value)


def _optional_number(value, label, minimum=None):
    if value is None:
        return None
    value = _number(value, label)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _vector(value, label):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-coordinate endpoint")
    return [_number(v, label) for v in value]


class IntervalMemory:
    """Summarize one interval with constant-size event aggregates and samples.

    ``record(entry, request_id=..., requested_steps=...)`` accepts the legacy
    digest entry and optionally ``measured_before`` / ``measured_after`` mappings
    from arm name to measured XYZ. Coordinates must use the integration's same
    metre frame. A missing endpoint is unknown, never reconstructed from commands.
    Unknown input keys are ignored. Invalid known telemetry rejects the complete
    record without mutating memory. ``drain`` clears provenance and baselines too.
    """

    def __init__(self, max_events=16, max_recent=4, arms=("left", "right"), motion_epsilon_m=1e-4):
        self.max_events = _integer(max_events, "max_events", 1)
        self.max_recent = _integer(max_recent, "max_recent", 0)
        if self.max_events > 128 or self.max_recent > 32:
            raise ValueError("sample bounds exceed max_events=128 / max_recent=32")
        self.arms = tuple(arms)
        if (not self.arms or len(self.arms) > 8 or len(set(self.arms)) != len(self.arms)
                or any(not isinstance(a, str) or not _ID.fullmatch(a) for a in self.arms)):
            raise ValueError("arms must be 1..8 distinct bounded identifiers")
        self.motion_epsilon_m = _number(motion_epsilon_m, "motion_epsilon_m")
        if self.motion_epsilon_m < 0:
            raise ValueError("motion_epsilon_m must be nonnegative")
        self.reset()

    def reset(self):
        """Forget the complete interval, including command baselines and event refs."""
        self._count = self._rows = self._requested = self._executed = 0
        self._partial = self._zero = self._gap_steps = 0
        self._start = self._end = None
        self._last_ref = None
        self._event_count = 0
        self._first_events = []
        self._recent_events = deque(maxlen=self.max_events - (self.max_events + 1) // 2)
        self._recent = deque(maxlen=self.max_recent)
        self._events = {}
        self._arm = {a: {
            "proposal_chunks": 0, "proposed_path_m": 0.0,
            "commanded_low_motion_chunks": 0, "commanded_low_motion_run": 0,
            "longest_commanded_low_motion_run": 0, "commanded_low_motion_start_native_step": None,
            "longest_commanded_low_motion_span": None,
            "measured_chunks": 0, "measured_endpoint_displacement_m": 0.0,
            "low_endpoint_displacement_chunks": 0, "low_endpoint_run": 0,
            "longest_low_endpoint_run": 0, "low_endpoint_start_native_step": None,
            "longest_low_endpoint_span": None,
            "ledger_chunks": 0, "commanded_gripper_changes": 0,
            "first_gripper_command": None, "last_gripper_command": None,
        } for a in self.arms}

    def _normalized(self, entry, request_id, requested_steps):
        if not isinstance(entry, Mapping):
            raise ValueError("entry must be a mapping")
        start = _integer(entry.get("native_step"), "native_step")
        rows = _integer(entry.get("rows"), "rows", 1)
        requested = rows if requested_steps is None else _integer(requested_steps, "requested_steps")
        executed = _integer(entry.get("executed_steps"), "executed_steps")
        if requested > rows or executed > requested:
            raise ValueError("executed_steps <= requested_steps <= rows is required")
        if self._end is not None and start < self._end:
            raise ValueError("native_step overlaps or precedes already recorded execution")
        if request_id is not None and (not isinstance(request_id, str) or not _ID.fullmatch(request_id)):
            raise ValueError("request_id must be an opaque bounded identifier")
        if request_id is not None and request_id == self._last_ref:
            raise ValueError("duplicate consecutive request_id")
        def mapping(key):
            value = entry.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping")
            return value
        diagnostics, ledger = mapping("diagnostics"), mapping("gripper_ledger")
        before, after = mapping("measured_before"), mapping("measured_after")
        arms = {}
        for arm in self.arms:
            diag = diagnostics.get(arm) or {}
            if not isinstance(diag, Mapping):
                raise ValueError("arm diagnostics must be a mapping")
            path = _optional_number(diag.get("path_length_m"), "path_length_m", 0)
            command = _optional_number(ledger.get(arm), "gripper_ledger")
            transitions = {}
            for key in ("closes_at_row", "opens_at_row"):
                value = diag.get(key)
                if value is not None:
                    value = _integer(value, key)
                    if value >= rows:
                        raise ValueError(f"{key} must name a candidate row within rows")
                    transitions[key] = value
            xyz0, xyz1 = _vector(before.get(arm), "measured_before"), _vector(after.get(arm), "measured_after")
            displacement = math.dist(xyz0, xyz1) if xyz0 is not None and xyz1 is not None else None
            for amount, total, label in ((path, self._arm[arm]["proposed_path_m"], "proposed path sum"),
                                          (displacement, self._arm[arm]["measured_endpoint_displacement_m"],
                                           "measured endpoint displacement sum")):
                if amount is not None and not math.isfinite(amount + total):
                    raise ValueError(f"{label} overflows finite telemetry")
            arms[arm] = {"proposed_path_m": path, "gripper_command": command,
                         "proposed_transitions": transitions, "measured_endpoint_displacement_m": displacement}
        return {"native_step": start, "end_native_step_exclusive": start + executed,
                "rows": rows, "requested_steps": requested, "executed_steps": executed,
                "request_id": request_id, "arms": arms}

    def _event(self, kind, record, arm=None, **evidence):
        assert kind in _EVENT_TYPES
        source = {"chunk_index": self._count - 1, "native_step": record["native_step"],
                  "end_native_step_exclusive": record["end_native_step_exclusive"],
                  "request_id": record["request_id"]}
        event = {"kind": kind, "arm": arm, "source": source, "evidence": evidence}
        key = (kind, arm)
        if key not in self._events:
            self._events[key] = {"kind": kind, "arm": arm, "count": 0,
                                 "first": copy.deepcopy(event), "last": None}
        aggregate = self._events[key]
        aggregate["count"] += 1
        aggregate["last"] = copy.deepcopy(event)
        self._event_count += 1
        if len(self._first_events) < (self.max_events + 1) // 2:
            self._first_events.append(event)
        else:
            self._recent_events.append(event)

    def record(self, entry, *, request_id=None, requested_steps=None):
        """Record completed execution once; all input validation precedes mutation."""
        record = self._normalized(entry, request_id, requested_steps)
        previous_end = self._end
        gap = 0 if previous_end is None else record["native_step"] - previous_end
        if self._start is None:
            self._start = record["native_step"]
        self._end = record["end_native_step_exclusive"]
        self._last_ref = request_id
        self._count += 1
        self._rows += record["rows"]
        self._requested += record["requested_steps"]
        self._executed += record["executed_steps"]
        if gap:
            self._gap_steps += gap
            self._event("source_gap", record, missing_native_steps=gap, previous_end_exclusive=previous_end)
        if record["executed_steps"] < record["requested_steps"]:
            self._partial += 1
            self._event("partial_execution", record, requested_steps=record["requested_steps"],
                        executed_steps=record["executed_steps"], cause="unknown")
        if record["executed_steps"] == 0:
            self._zero += 1
            self._event("no_execution", record, requested_steps=record["requested_steps"], cause="unknown")
        for arm, values in record["arms"].items():
            totals = self._arm[arm]
            if gap:
                totals["commanded_low_motion_run"] = totals["low_endpoint_run"] = 0
            path = values["proposed_path_m"]
            if path is not None:
                totals["proposal_chunks"] += 1
                totals["proposed_path_m"] += path
            low = path is not None and path <= self.motion_epsilon_m
            totals["commanded_low_motion_chunks"] += int(low)
            if low and totals["commanded_low_motion_run"] == 0:
                totals["commanded_low_motion_start_native_step"] = record["native_step"]
            totals["commanded_low_motion_run"] = totals["commanded_low_motion_run"] + 1 if low else 0
            if not low:
                totals["commanded_low_motion_start_native_step"] = None
            if totals["commanded_low_motion_run"] > totals["longest_commanded_low_motion_run"]:
                totals["longest_commanded_low_motion_run"] = totals["commanded_low_motion_run"]
                totals["longest_commanded_low_motion_span"] = {
                    "start_native_step": totals["commanded_low_motion_start_native_step"],
                    "end_native_step_exclusive": record["end_native_step_exclusive"]}
            distance = values["measured_endpoint_displacement_m"]
            if distance is not None:
                totals["measured_chunks"] += 1
                totals["measured_endpoint_displacement_m"] += distance
            low_endpoint = distance is not None and distance <= self.motion_epsilon_m
            totals["low_endpoint_displacement_chunks"] += int(low_endpoint)
            if low_endpoint and totals["low_endpoint_run"] == 0:
                totals["low_endpoint_start_native_step"] = record["native_step"]
            totals["low_endpoint_run"] = totals["low_endpoint_run"] + 1 if low_endpoint else 0
            if not low_endpoint:
                totals["low_endpoint_start_native_step"] = None
            if totals["low_endpoint_run"] > totals["longest_low_endpoint_run"]:
                totals["longest_low_endpoint_run"] = totals["low_endpoint_run"]
                totals["longest_low_endpoint_span"] = {
                    "start_native_step": totals["low_endpoint_start_native_step"],
                    "end_native_step_exclusive": record["end_native_step_exclusive"]}
            command = values["gripper_command"]
            previous = totals["last_gripper_command"]
            if command is not None:
                totals["ledger_chunks"] += 1
                if totals["first_gripper_command"] is None:
                    totals["first_gripper_command"] = command
                if previous is not None and not gap and command != previous:
                    totals["commanded_gripper_changes"] += 1
                    self._event("commanded_gripper_change", record, arm, before=previous, after=command,
                                meaning="command ledger only; physical gripper/object state unknown")
            # Missing telemetry/gaps break comparison with an earlier command.
            totals["last_gripper_command"] = command
            if values["proposed_transitions"]:
                self._event("proposed_gripper_transition", record, arm, **values["proposed_transitions"],
                            executed_steps=record["executed_steps"],
                            meaning="candidate row indices; not proof of executed grasp or release")
        self._recent.append(record)

    def snapshot(self):
        """Return an independent, bounded JSON-ready view without clearing it."""
        samples = list(self._first_events) + list(self._recent_events)
        return copy.deepcopy({
            "schema": SCHEMA,
            "coverage": {"chunks": self._count, "start_native_step": self._start,
                         "end_native_step_exclusive": self._end, "unrecorded_native_steps": self._gap_steps,
                         "scope": "completed pass-through chunks since last drain/reset"},
            "counts": {"candidate_rows": self._rows, "requested_steps": self._requested,
                       "executed_steps": self._executed, "partial_execution_chunks": self._partial,
                       "zero_execution_chunks": self._zero, "events": self._event_count},
            "arms": self._arm,
            "event_aggregates": list(self._events.values()),
            "event_samples": samples,
            "recent_chunks": list(self._recent),
            "omissions": {"event_details": self._event_count - len(samples),
                          "chunk_details": self._count - len(self._recent),
                          "policy": "all event counts/first/last retained; bounded first/recent event details"},
            "unknown": {"task_progress": "not inferred", "object_state": "not inferred",
                        "grasp_or_release_success": "not inferred", "failure_causes": "not inferred",
                        "measured_motion": "unknown where measured_chunks < coverage.chunks; endpoints are not path length",
                        "command_semantics": "candidate path and gripper ledger are commands, not observed effects"},
            "parameters": {"max_events": self.max_events, "max_recent": self.max_recent,
                           "motion_epsilon_m": self.motion_epsilon_m},
        })

    def drain(self):
        """Emit this interval then clear every retained field before the next one."""
        result = self.snapshot()
        self.reset()
        return result
