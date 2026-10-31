"""When the hybrid condition hands one decision from the fast path to the slow brain.

All timing is simulated time (native steps / control rate), so the schedule is
the same whatever the wall clock or model latency. Inputs are only commanded and
observed robot state; evaluator-only data such as native success never enters.
"""
from __future__ import annotations

import math

import numpy as np

FIELDS = {"control_hz", "review_period_s", "cooldown_s", "stall_window_s",
          "stall_min_travel_m", "stall_gripper_reversals"}
# Provenance of the control rate (the pinned file it was read from). Frozen with the
# configuration; the schedule never reads it.
NOTES = {"control_hz_source"}
# Memory-hybrid options (docs/slow-brain-memory; analysis finding F4). Absent, the schedule is the one the oeh-v1
# campaign froze, and so are the configurations' hashes.
#   gripper_reversal_delta: count a gripper reversal only when the commanded opening turns back by at least this much
#     from its last extreme. The benchmark contract then passes raw commands instead of 0/1: RoboDojo's OpenWAM
#     commands jitter across 0.5 (33 of 34 counted flips in the recorded escalations were inside [0.4, 0.6]) and
#     close at 0.29-0.54, so neither the 0.5 cut nor a fixed closed threshold separates a real re-grasp from noise.
#   stall_cooldown_s: a stall escalation needs this much simulated time since the last escalation of any kind.
#   relative_review: the next review is review_period_s after the last slow decision, or when the slow brain asked
#     to look again (request_look); the slow brain may also declare expected stillness (quiet), during which a lack
#     of motion is not a stall.
OPTIONS = {"gripper_reversal_delta", "stall_cooldown_s", "relative_review"}
REASONS = ("gate_rejection", "stall_no_motion", "stall_gripper_oscillation", "periodic_review", "requested_look")


def validate_escalation(value) -> dict:
    if not isinstance(value, dict) or not FIELDS <= set(value) or set(value) - FIELDS - NOTES - OPTIONS:
        raise ValueError(f"hybrid escalation requires exactly {sorted(FIELDS)} "
                         f"(and optionally control_hz_source, {', '.join(sorted(OPTIONS))})")
    if "control_hz_source" in value and (not isinstance(value["control_hz_source"], str)
                                         or not value["control_hz_source"].strip()):
        raise ValueError("control_hz_source must cite the pinned file the rate comes from")
    for key in FIELDS - {"stall_gripper_reversals"}:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 < number < math.inf:
            raise ValueError(f"finite positive escalation {key} required")
    if type(value["stall_gripper_reversals"]) is not int or value["stall_gripper_reversals"] < 1:
        raise ValueError("positive integer stall_gripper_reversals required")
    if value["cooldown_s"] >= value["review_period_s"]:
        raise ValueError("cooldown must be shorter than the review period")
    if "gripper_reversal_delta" in value:
        delta = value["gripper_reversal_delta"]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not 0 < delta <= 1:
            raise ValueError("gripper_reversal_delta must be in (0, 1]")
    if "stall_cooldown_s" in value:
        cooldown = value["stall_cooldown_s"]
        if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or not 0 <= cooldown < math.inf:
            raise ValueError("stall_cooldown_s must be finite and nonnegative")
    if "relative_review" in value and type(value["relative_review"]) is not bool:
        raise ValueError("relative_review must be a boolean")
    return dict(value)


class Escalation:
    def __init__(self, cfg: dict):
        self.cfg = validate_escalation(cfg)
        self.hz = float(cfg["control_hz"])
        self.window = max(2, round(cfg["stall_window_s"] * self.hz))
        self.delta = cfg.get("gripper_reversal_delta")
        self.stall_cooldown_s = cfg.get("stall_cooldown_s", 0)
        self.relative = cfg.get("relative_review", False)
        self.next_review_s = cfg["review_period_s"]
        self.requested = False  # the pending review is one the slow brain asked for
        self.quiet_until_s = None
        self.last_escalation_s = None
        # (step, arm positions, gripper values, reversals this step per arm)
        self.history: list[tuple[int, list[np.ndarray], list[float], list[int]]] = []
        self.turns: list[dict] = []  # per arm: last extreme and direction of the commanded opening
        self.rejection = False
        self.evidence = None  # what the last stall decision measured, for the slow brain's escalation note

    def sim_time(self, step: int) -> float:
        return step / self.hz

    def observe(self, step: int, positions, grippers) -> None:
        """Record one executed native step: arm positions (m) and the commanded grippers. Without
        gripper_reversal_delta the benchmark contract binarizes them (1 open, 0 closed) so continuous jitter never
        counts as a change of direction; with it they are raw openings in [0, 1] and a reversal needs that swing."""
        commands = [float(g) for g in grippers]
        if self.delta is None:
            if any(g not in (0.0, 1.0) for g in commands):
                raise ValueError("gripper commands must be binarized by the benchmark contract")
        elif any(not 0.0 <= g <= 1.0 for g in commands):
            raise ValueError("raw gripper commands must be openings in [0, 1]")
        reversals = self._reversals(commands)
        self.history.append((step, [np.asarray(p, dtype=float)[:3] for p in positions], commands, reversals))
        del self.history[:-self.window]

    def _reversals(self, commands) -> list[int]:
        if self.delta is None:
            if not self.history:
                return [0] * len(commands)
            previous = self.history[-1][2]
            return [int(a != b) for a, b in zip(previous, commands)]
        if not self.turns:
            self.turns = [{"extreme": value, "direction": 0} for value in commands]
            return [0] * len(commands)
        counted = []
        for turn, value in zip(self.turns, commands):
            reversal = 0
            if turn["direction"] == 0:  # the first swing sets a direction; only a turn back counts
                if value - turn["extreme"] >= self.delta:
                    turn.update(direction=1, extreme=value)
                elif turn["extreme"] - value >= self.delta:
                    turn.update(direction=-1, extreme=value)
            elif turn["direction"] == 1:
                if value > turn["extreme"]:
                    turn["extreme"] = value
                elif turn["extreme"] - value >= self.delta:
                    turn.update(direction=-1, extreme=value)
                    reversal = 1
            else:
                if value < turn["extreme"]:
                    turn["extreme"] = value
                elif value - turn["extreme"] >= self.delta:
                    turn.update(direction=1, extreme=value)
                    reversal = 1
            counted.append(reversal)
        return counted

    def rejected(self) -> None:
        self.rejection = True

    def request_look(self, at_s: float) -> None:
        """The slow brain asked to be consulted again at simulated time at_s (relative_review only)."""
        if not self.relative:
            raise ValueError("requested looks need relative_review")
        self.next_review_s = at_s
        self.requested = True

    def quiet(self, until_s: float) -> None:
        """The slow brain expects the arms to stay still until until_s: no stall_no_motion before then."""
        if not self.relative:
            raise ValueError("declared stillness needs relative_review")
        self.quiet_until_s = until_s

    def _stalled(self, now: float) -> str | None:
        if len(self.history) < self.window:
            return None
        if self.last_escalation_s is not None and now - self.last_escalation_s < self.stall_cooldown_s:
            return None
        first = self.history[0][1]
        travel = [max(float(np.linalg.norm(pos[arm] - first[arm])) for _, pos, _, _ in self.history)
                  for arm in range(len(first))]
        reversals = [sum(counts[arm] for _, _, _, counts in self.history[1:])
                     for arm in range(len(self.history[0][2]))]
        window_s = (self.history[-1][0] - self.history[0][0] + 1) / self.hz
        if max(travel) < self.cfg["stall_min_travel_m"] and not (
                self.quiet_until_s is not None and now < self.quiet_until_s):
            self.evidence = {"arm": "both" if len(travel) > 1 else "arm", "window_travel_m": round(max(travel), 4),
                             "reversals": max(reversals), "window_s": round(window_s, 3)}
            return "stall_no_motion"
        for arm, count in enumerate(reversals):
            # Each counted change of direction of the commanded gripper is one reversal.
            if count >= self.cfg["stall_gripper_reversals"]:
                name = ("left", "right")[arm] if len(reversals) == 2 else "arm"
                self.evidence = {"arm": name, "window_travel_m": round(travel[arm], 4),
                                 "reversals": count, "window_s": round(window_s, 3)}
                return "stall_gripper_oscillation"
        return None

    def decide(self, step: int) -> tuple[str | None, str | None]:
        """Return (escalation reason, suppressed periodic review reason) for this decision."""
        self.evidence = None
        if self.rejection:
            return "gate_rejection", None
        now = self.sim_time(step)
        stall = self._stalled(now)
        if stall:
            return stall, None
        if now >= self.next_review_s:
            if self.relative:
                return ("requested_look" if self.requested else "periodic_review"), None
            while self.next_review_s <= now:
                self.next_review_s += self.cfg["review_period_s"]
            if self.last_escalation_s is not None and now - self.last_escalation_s < self.cfg["cooldown_s"]:
                return None, "periodic_review_in_cooldown"
            return "periodic_review", None
        return None, None

    def escalated(self, step: int) -> None:
        self.rejection = False
        self.last_escalation_s = self.sim_time(step)
        self.history.clear()
        self.turns = []
        if self.relative:
            self.next_review_s = self.last_escalation_s + self.cfg["review_period_s"]
            self.requested = False
