"""OEH hybrid v4: when the review mode's GPT teacher takes the slow brain's place, and when it hands back.

The review mode's default schedule (``every_chunk``) shows the teacher every OpenWAM chunk. Under ``on_stall`` the
teacher is the slow brain of OEH hybrid v4: OpenWAM's chunks run unchanged and unreviewed while it makes progress,
and the teacher is consulted only when

- the arms stall (no arm travels ``stall_min_travel_m`` within ``stall_window_s``),
- a gripper oscillates (``stall_gripper_reversals`` turn-backs of at least ``gripper_reversal_delta`` in the window),
- OEH's workspace check refuses rows of OpenWAM's chunk (the candidate is truncated), or
- optionally, OpenWAM has just closed or opened a gripper (``look_after_gripper``), the decisive moments of a grasp
  or a release.

Once consulted the teacher reviews every chunk, under the same prompt and gate as the review mode, until it accepts
a full student chunk while judging the last chunk progressing (or recovered) and the next one aligned; then OpenWAM
runs alone again and no stall or gripper look is raised for ``handback_cooldown_s``. A refused chunk is raised
whatever the cooldown. ``max_teacher_decisions`` bounds the teacher's packets per episode: when they are spent the
teacher hands back and OpenWAM finishes the episode alone.

Why: in the 2026-09-27 slow-brain lines every loss came from taking over while OpenWAM was progressing, and the one
win from acting while it was idle (docs/slow-brain-memory/report); in the Luna review round the teacher took over
almost every decision (228 of 249 calls `eef`) and scored 5.0 against direct's 20.6 (docs/openwam-gpt-hybrid/
formal-run-log.md). The stall detector is the memory line's (escalation.py with gripper_reversal_delta).

All timing is simulated time (native steps / control rate). Inputs are only commanded and observed robot state.
"""
from __future__ import annotations

import math
from typing import Mapping

from .escalation import Escalation

SCHEDULES = ("every_chunk", "on_stall", "segment")
# The teacher prompt's addendum per schedule (oeh_policy/teacher/); every_chunk, the review mode's, has none.
SCHEDULE_FILES = {"on_stall": "on_stall.md", "segment": "segment.md"}
# The on_stall block's keys, their [low, high] bounds, and which are integers or booleans.
ON_STALL_BOUNDS = {"stall_window_s": (0.2, 30.0), "stall_min_travel_m": (0.001, 0.2),
                   "stall_gripper_reversals": (1, 20), "gripper_reversal_delta": (0.05, 1.0),
                   "handback_cooldown_s": (0.0, 120.0), "max_teacher_decisions": (0, 500)}
ON_STALL_INTEGERS = {"stall_gripper_reversals", "max_teacher_decisions"}
ON_STALL_FLAGS = {"look_after_gripper"}
TRIGGERS = ("stall_no_motion", "stall_gripper_oscillation", "candidate_rejected", "after_grasp", "after_release")
# What the teacher judged when it accepted a full student chunk: the hand-back condition.
HANDBACK_EXECUTION = ("progressing", "recovered")
HANDBACK_INTENT = ("aligned",)
# G5 (llm-capabilities-and-oeh-modules.md §2.4): the trigger block a packet carries when review.digest is set. Every
# reason carries the same three evidence keys; a reason whose detector measured nothing has them null.
TRIGGER_EVIDENCE_KEYS = ("window_travel_m", "reversals", "window_s")
# G13: the segment schedule's own wake reasons beside TRIGGERS. episode_start: decision 0; next_look: the
# after_chunks the LLM component named have passed; segment_expired: it named none and the default cadence
# (review.segment.default_after_chunks) has passed.
SEGMENT_REASONS = ("episode_start", "next_look", "segment_expired")
# review.segment: its keys and bounds. max_segment_chunks is required (P4: the bound, settled by the G13 call trace;
# oeh_policy.mechanisms.SCHEDULE_SETTINGS has the same bounds). default_after_chunks answers "what cadence applies
# when the LLM component names no next_look?"; absent, it is max_segment_chunks [proposal: the compiler writes only
# max_segment_chunks on this commit, so the fallback keeps a compiled config valid; the matrix should state it].
SEGMENT_BOUNDS = {"max_segment_chunks": (1, 64), "default_after_chunks": (1, 64)}
SEGMENT_REQUIRED = {"max_segment_chunks"}


def validate_on_stall(value) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(ON_STALL_BOUNDS) | ON_STALL_FLAGS:
        raise ValueError(f"review.on_stall requires exactly {sorted(set(ON_STALL_BOUNDS) | ON_STALL_FLAGS)}")
    for key, (low, high) in ON_STALL_BOUNDS.items():
        number = value[key]
        integer = key in ON_STALL_INTEGERS
        if (isinstance(number, bool) or not isinstance(number, int if integer else (int, float))
                or not math.isfinite(number) or not low <= number <= high):
            raise ValueError(f"review.on_stall.{key} must be {'an integer' if integer else 'a number'} "
                             f"in [{low}, {high}]")
    for key in ON_STALL_FLAGS:
        if type(value[key]) is not bool:
            raise ValueError(f"review.on_stall.{key} must be a boolean")
    return dict(value)


def validate_segment(value) -> dict:
    """review.segment (G13): {max_segment_chunks, default_after_chunks?}, integers in bounds, the default at most the
    bound. Returned as given (no key is added, so the frozen block keeps the compiler's bytes)."""
    if not isinstance(value, Mapping) or not SEGMENT_REQUIRED <= set(value) or set(value) - set(SEGMENT_BOUNDS):
        raise ValueError(f"review.segment takes {sorted(SEGMENT_REQUIRED)} and optionally "
                         f"{sorted(set(SEGMENT_BOUNDS) - SEGMENT_REQUIRED)}")
    for key, (low, high) in SEGMENT_BOUNDS.items():
        if key in value and (type(value[key]) is not int or not low <= value[key] <= high):
            raise ValueError(f"review.segment.{key} must be an integer in [{low}, {high}]")
    if value.get("default_after_chunks", value["max_segment_chunks"]) > value["max_segment_chunks"]:
        raise ValueError("review.segment.default_after_chunks must not exceed max_segment_chunks")
    return dict(value)


def trigger_view(reason: str, evidence: Mapping | None) -> dict:
    """The packet's ``takeover.trigger`` (G5): the wake reason and the detector's measurements, the same three keys
    for every reason (null where the detector measured nothing)."""
    evidence = evidence or {}
    return {"reason": reason, "evidence": {key: evidence.get(key) for key in TRIGGER_EVIDENCE_KEYS}}


class GripperSwings:
    """Committed open/close swings of each arm's commanded opening, with hysteresis: a swing is a move of at least
    ``delta`` away from the last extreme, so command jitter (OpenWAM's commands hover around 0.5) is never one."""

    def __init__(self, delta: float):
        self.delta = delta
        self.turns: list[dict] | None = None

    def update(self, commands) -> list[str]:
        """Feed one step's commanded openings (1 open, 0 closed); returns the swings it completed per arm, each
        ``"close"`` or ``"open"``."""
        if self.turns is None:
            self.turns = [{"extreme": value, "direction": 0} for value in commands]
            return []
        swings = []
        for turn, value in zip(self.turns, commands):
            # While opening the extreme follows the highest command, while closing the lowest; before the first
            # swing it stays the episode's first command.
            if turn["direction"] == 1 and value > turn["extreme"]:
                turn["extreme"] = value
            elif turn["direction"] == -1 and value < turn["extreme"]:
                turn["extreme"] = value
            elif turn["direction"] != 1 and value - turn["extreme"] >= self.delta:
                turn.update(direction=1, extreme=value)
                swings.append("open")
            elif turn["direction"] != -1 and turn["extreme"] - value >= self.delta:
                turn.update(direction=-1, extreme=value)
                swings.append("close")
        return swings


class Takeover:
    """The on_stall schedule's state for one episode. The runtime feeds it every executed native step and asks it at
    every chunk boundary whether the teacher takes over; it tells it what the teacher decided."""

    def __init__(self, cfg: Mapping, control_hz: float):
        self.cfg = validate_on_stall(cfg)
        self.hz = float(control_hz)
        # The memory line's stall detector, with periodic reviews switched off: it never raises periodic_review.
        self.stall = Escalation({"control_hz": self.hz, "review_period_s": 1e9, "cooldown_s": 1.0,
                                 "stall_window_s": cfg["stall_window_s"],
                                 "stall_min_travel_m": cfg["stall_min_travel_m"],
                                 "stall_gripper_reversals": cfg["stall_gripper_reversals"],
                                 "gripper_reversal_delta": cfg["gripper_reversal_delta"],
                                 "stall_cooldown_s": cfg["handback_cooldown_s"]})
        self.swings = GripperSwings(cfg["gripper_reversal_delta"])
        self.engaged = False
        self.reason = None
        self.evidence = None
        self.decisions = 0  # teacher packets this episode
        self.handed_back_s = None
        self.pending_swing = None  # the latest gripper swing since the last chunk boundary ("close" | "open")
        self.handovers: list[dict] = []

    def sim_time(self, step: int) -> float:
        return step / self.hz

    @property
    def spent(self) -> bool:
        limit = self.cfg["max_teacher_decisions"]  # explicit 0: no LLM-call cap, native/time gates still apply
        return limit > 0 and self.decisions >= limit

    def _cooling(self, now: float) -> bool:
        return self.handed_back_s is not None and now - self.handed_back_s < self.cfg["handback_cooldown_s"]

    def observe(self, step: int, positions, grippers) -> None:
        """One executed native step: arm positions (m) and the raw commanded openings in [0, 1]."""
        commands = [min(1.0, max(0.0, float(g))) for g in grippers]
        self.stall.observe(step, positions, commands)
        swings = self.swings.update(commands)
        if swings and not self.engaged:
            self.pending_swing = swings[-1]

    def decide(self, step: int, *, rejected: bool = False) -> str | None:
        """At a chunk boundary while OpenWAM runs alone: the reason the teacher takes over now, or None.
        ``rejected``: OEH refused rows of the chunk OpenWAM just proposed for this boundary."""
        if self.engaged:
            raise RuntimeError("the teacher already has control")
        if self.spent:
            self.pending_swing = None
            return None
        now = self.sim_time(step)
        self.evidence = None
        if rejected:
            return "candidate_rejected"
        reason, _ = self.stall.decide(step)  # stall cooldown = handback_cooldown_s since the last hand-back
        if reason is not None:
            self.evidence = self.stall.evidence
            return reason
        swing, self.pending_swing = self.pending_swing, None
        if swing is not None and self.cfg["look_after_gripper"] and not self._cooling(now):
            return "after_grasp" if swing == "close" else "after_release"
        return None

    def engage(self, step: int, reason: str) -> dict:
        """The teacher takes over at this boundary. Returns the handover note its first packet carries."""
        if reason not in TRIGGERS:
            raise ValueError(f"unknown takeover trigger {reason!r}")
        self.engaged, self.reason = True, reason
        self.stall.escalated(step)  # restart the stall window
        self.pending_swing = None
        note = {"reason": reason, "native_step": step, "sim_time_s": round(self.sim_time(step), 2),
                "teacher_decisions_left": (self.cfg["max_teacher_decisions"] - self.decisions
                                           if self.cfg["max_teacher_decisions"] else None)}
        if self.evidence:
            note["evidence"] = dict(self.evidence)
        self.handovers.append(dict(note))
        return note

    def packet(self) -> None:
        """The teacher gets one more packet."""
        self.decisions += 1

    def teacher_executed(self, step: int, record: Mapping, full_steps: int) -> str | None:
        """After one of the teacher's decisions executed: hand back (returning why) or keep control (None).
        ``full_steps``: the rows of the whole candidate the decision was made on (at most the student cap)."""
        if not self.engaged:
            return None
        assessment = record.get("assessment") or {}
        if (record.get("mode") == "student" and full_steps > 0 and record.get("requested_steps") == full_steps
                and assessment.get("execution_status") in HANDBACK_EXECUTION
                and assessment.get("intent_status") in HANDBACK_INTENT):
            return self.hand_back(step, "teacher_accepted_full_chunk")
        if self.spent:
            return self.hand_back(step, "teacher_decisions_spent")
        return None

    def hand_back(self, step: int, why: str) -> str:
        self.engaged, self.reason = False, None
        self.handed_back_s = self.sim_time(step)
        self.stall.escalated(step)  # the stall cooldown runs from the hand-back
        self.pending_swing = None
        return why

    def summary(self) -> dict:
        return {"teacher_decisions": self.decisions, "handovers": list(self.handovers), "engaged": self.engaged}


class Segment(Takeover):
    """The segment schedule's state for one episode (G13; review.schedule segment). Decision 0 is a packet. After each
    executed decision of the LLM component the next ``k`` chunk boundaries pass OpenWAM's chunk through whole, where
    ``k`` is the decision's ``next_look.after_chunks`` (1 <= k <= max_segment_chunks) or, when it named none,
    ``default_after_chunks``; the boundary after them is a packet (reason next_look or segment_expired). An on_stall
    detector (the review.on_stall block: stall, gripper oscillation, a refused chunk, the gripper looks, with its
    cooldown since the last decision) wakes it earlier. The LLM-call budget is review.max_decisions (the runtime's), not
    the on_stall block's max_teacher_decisions, which this schedule ignores."""

    def __init__(self, on_stall: Mapping, segment: Mapping, control_hz: float):
        super().__init__(on_stall, control_hz)
        self.segment_cfg = validate_segment(segment)
        self.max_chunks = self.segment_cfg["max_segment_chunks"]
        self.default_chunks = self.segment_cfg.get("default_after_chunks", self.max_chunks)
        self.remaining = None  # chunks still to pass in the current segment; None before decision 0
        self.named = False  # whether the current segment's length was the LLM component's own next_look
        self.engaged = True  # decision 0 is a packet: the LLM component holds the first boundary
        self.segments: list[dict] = []

    @property
    def spent(self) -> bool:
        return False  # the budget is review.max_decisions, enforced by the runtime when a packet is due

    def boundary(self, step: int, *, rejected: bool = False) -> str | None:
        """At a chunk boundary while no packet is outstanding: the wake reason, or None (OpenWAM's chunk passes)."""
        if self.engaged:
            # Decision 0, or a packet that executed nothing (abandoned for an empty candidate): the LLM component
            # still has the boundary, for the same reason.
            return self.reason or "episode_start"
        if self.remaining <= 0:
            self.pending_swing = None
            return "next_look" if self.named else "segment_expired"
        reason = self.decide(step, rejected=rejected)
        if reason is not None:
            return reason
        self.remaining -= 1
        return None

    def engage(self, step: int, reason: str) -> dict:
        if reason not in TRIGGERS and reason not in SEGMENT_REASONS:
            raise ValueError(f"unknown segment wake reason {reason!r}")
        if reason in TRIGGERS:
            return super().engage(step, reason)
        self.engaged, self.reason = True, reason
        self.evidence = None
        self.stall.escalated(step)
        self.pending_swing = None
        note = {"reason": reason, "native_step": step, "sim_time_s": round(self.sim_time(step), 2)}
        self.handovers.append(dict(note))
        return note

    def start(self, step: int, after_chunks: int | None) -> int:
        """A decision for the current packet executed: the segment of ``after_chunks`` (or the default) begins."""
        if after_chunks is not None and (type(after_chunks) is not int or not 1 <= after_chunks <= self.max_chunks):
            raise ValueError(f"next_look.after_chunks must be an integer in [1, {self.max_chunks}]")
        self.named = after_chunks is not None
        self.remaining = after_chunks if after_chunks is not None else self.default_chunks
        self.segments.append({"native_step": step, "after_chunks": self.remaining, "named": self.named,
                              "wake": self.reason})
        self.hand_back(step, "segment_started")
        return self.remaining

    def teacher_executed(self, step: int, record: Mapping, full_steps: int) -> str | None:
        raise RuntimeError("the segment schedule starts a segment (Segment.start), it never hands back on a chunk")

    def summary(self) -> dict:
        return {**super().summary(), "segments": list(self.segments)}


def shadow_rates(positions, grippers, cfg: Mapping, control_hz: float, *, chunk_steps: int = 32,
                 rejected_steps=()) -> dict:
    """Shadow detector rates (G5's validation artifact): replay one recorded episode's executed steps through the
    on_stall detectors and count how often each trigger would have fired at a chunk boundary. Nothing intervenes:
    each firing is taken as an immediate hand-back (its window restarts and the cooldown runs), the closest replay
    of "looked once, OpenWAM went on". ``positions[i]``: the arms' measured positions after step i+1 (m);
    ``grippers[i]``: the raw commanded openings of that step; ``chunk_steps``: native steps per chunk boundary
    [proposal: OpenWAM's 32]; ``rejected_steps``: boundaries where OEH refused rows (candidate_rejected cannot be
    recovered from positions alone). The on_stall block's max_teacher_decisions does not bound a shadow count."""
    if len(positions) != len(grippers):
        raise ValueError("positions and grippers must have one entry per executed step")
    if type(chunk_steps) is not int or chunk_steps < 1:
        raise ValueError("chunk_steps must be a positive integer")
    shadow = Takeover({**cfg, "max_teacher_decisions": ON_STALL_BOUNDS["max_teacher_decisions"][1]}, control_hz)
    rejected = set(rejected_steps)
    fires = {reason: 0 for reason in TRIGGERS}
    events, boundaries = [], 0
    for index, (position, grip) in enumerate(zip(positions, grippers)):
        step = index + 1
        shadow.observe(step, position, grip)
        if step % chunk_steps:
            continue
        boundaries += 1
        reason = shadow.decide(step, rejected=step in rejected)
        if reason is None:
            continue
        fires[reason] += 1
        events.append({"native_step": step, **trigger_view(reason, shadow.evidence)})
        shadow.engage(step, reason)
        shadow.hand_back(step, "shadow")
        shadow.decisions = 0
    seconds = len(positions) / float(control_hz)
    return {"steps": len(positions), "boundaries": boundaries, "sim_time_s": round(seconds, 3),
            "fires": fires, "per_minute": {reason: round(count * 60 / seconds, 3) if seconds else None
                                          for reason, count in fires.items()},
            "events": events}
