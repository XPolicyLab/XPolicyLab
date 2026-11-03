"""Budget governance and in-program guards.

BudgetGovernor: what the remaining native actions can still buy, from public robot state only. It keeps the
episode-start flange poses, estimates the actions back there, refuses (mode 'enforce') a move whose planned
actions plus the way back plus a reserve exceed what is left (unless final=True, or the move itself goes
home), and accounts native actions per finished sub-goal.

Checkpoints: host-evaluated guards (mark, native_reserve, pose). A failed guard raises CheckpointFailed, which
stops the rest of the program that called it.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from typing import Optional

import numpy as np

from knowin_agent.core.geometry import Pose

RESERVE = 10
HOME_STEP_M, HOME_STEP_RAD, HOME_SETTLE = 0.03, 0.15, 2
ORIGIN_M, ORIGIN_RAD = 0.15, math.radians(20)  # RoboDojo all_robot_back_to_origin
MODES = ("off", "panel", "enforce")


class BudgetGovernor:
    def __init__(self, runtime, mode: str = "enforce", reserve: int = RESERVE):
        if mode not in MODES:
            raise ValueError(f"budget mode must be one of {MODES}")
        self.rt = runtime
        self.mode = mode
        self.reserve = reserve
        self.origin = {arm: state.ee_pose for arm, state in runtime.obs.arms.items()}
        self.subgoals: list[dict] = []
        self.attempt_start = runtime.backend.budget().ticks_used
        self.refusals = 0

    def reset(self) -> None:
        """A new attempt in the same scene (practice reset)."""
        self.subgoals.clear()
        self.attempt_start = self.rt.backend.budget().ticks_used

    def remaining(self) -> Optional[int]:
        return self.rt.backend.budget().remaining

    def origin_distance(self, poses: Optional[dict] = None) -> dict:
        poses = poses or {arm: state.ee_pose for arm, state in self.rt.obs.arms.items()}
        rows = {}
        for arm, home in self.origin.items():
            current = poses[arm]
            rows[arm] = {"position_m": float(np.linalg.norm(current.position - home.position)),
                         "max_axis_m": float(np.max(np.abs(current.position - home.position))),
                         "angle_rad": current.errors_to(home)[1]}
        return rows

    @staticmethod
    def home_actions(rows: dict) -> int:
        moves = sum(math.ceil(max(r["position_m"] / HOME_STEP_M, r["angle_rad"] / HOME_STEP_RAD) - 1e-9)
                    for r in rows.values())
        return moves + HOME_SETTLE if moves else 0

    def at_origin(self, rows: Optional[dict] = None) -> bool:
        rows = rows or self.origin_distance()
        return all(r["max_axis_m"] <= ORIGIN_M and r["angle_rad"] <= ORIGIN_RAD for r in rows.values())

    def check_move(self, arm: str, target: Pose, planned: int, *, final: bool = False) -> Optional[dict]:
        """None when the move and the way back fit; otherwise the numbers (the move must not run)."""
        if self.mode != "enforce" or final:
            return None
        remaining = self.remaining()
        if remaining is None:
            return None
        poses = {a: s.ee_pose for a, s in self.rt.obs.arms.items()}
        poses[arm] = target
        rows = self.origin_distance(poses)
        if rows[arm]["position_m"] < 0.01:
            return None  # the way back itself is always allowed
        back = self.home_actions(rows)
        if planned + back + self.reserve <= remaining:
            return None
        self.refusals += 1
        return {"planned_actions": planned, "return_after": back, "reserve": self.reserve,
                "native_steps_remaining": remaining,
                "advice": "not enough native actions for this move and the way back: finish what scores, or pass "
                          "final=True for the last scoring move"}

    def subgoal_done(self, name: str) -> dict:
        if not isinstance(name, str) or not 1 <= len(name) <= 40:
            raise ValueError("subgoal name must be 1..40 characters")
        now = self.rt.backend.budget().ticks_used
        last = self.subgoals[-1]["native_steps"] if self.subgoals else self.attempt_start
        self.subgoals.append({"name": name, "native_steps": now, "spent": now - last})
        return {"status": "recorded", "subgoals_done": len(self.subgoals), "native_steps_spent": now - last}

    def panel(self) -> dict:
        budget = self.rt.backend.budget()
        remaining = budget.remaining
        rows = self.origin_distance()
        home = self.home_actions(rows)
        usable = None if remaining is None else remaining - home - self.reserve
        out = {"native_steps_used": budget.ticks_used, "native_steps_remaining": remaining,
               "return_to_origin_estimate": home, "safety_reserve": self.reserve, "usable_native_steps": usable,
               "at_origin_for_scoring": self.at_origin(rows), "mode": self.mode,
               "origin_distance": {a: {k: round(v, 3) for k, v in r.items()} for a, r in rows.items()}}
        if self.subgoals:
            costs = [row["spent"] for row in self.subgoals]
            per = statistics.median(costs)
            out["subgoals"] = {"done": [row["name"] for row in self.subgoals], "native_steps_each": costs,
                               "median_native_steps": per,
                               "more_affordable": None if usable is None or not per else max(0, int(usable // per)),
                               "note": "mark each finished sub-goal with subgoal_done(name); when more_affordable "
                                       "is 0, finish what already scores and head back"}
        window = getattr(self.rt, "wall_window", None)
        if window:
            left = max(0.0, window[0] + window[1] - time.time())
            out["wall_seconds_left"] = int(left)
            if left < window[1] / 4:
                # Tests that ran past 4000 of their 5400 s all failed (7 of 7): the agent never saw the clock.
                out["wall_advice"] = ("Under a quarter of the wall clock is left: secure the tiers already held, "
                                      "then head back; no new sub-goals.")
        if usable is not None and usable <= 0:
            out["advice"] = "Return to origin now: the remaining native actions only cover the way back and the reserve."
        return out


class CheckpointFailed(RuntimeError):
    pass


class Checkpoints:
    KINDS = ("mark", "native_reserve", "pose")

    def __init__(self, runtime):
        self.rt = runtime
        self.begin()

    def begin(self) -> None:
        """Marks never cross programs: evidence must come from the program that uses it."""
        self.marks: dict = {}
        self.log: list = []

    def run(self, name: str, kind: str, **params) -> dict:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            raise ValueError("checkpoint name must be 1..40 characters of [A-Za-z0-9_-]")
        if kind not in self.KINDS:
            raise ValueError(f"checkpoint kind must be one of {self.KINDS}")
        entry = {"name": name, "kind": kind, "native_steps": self.rt.backend.budget().ticks_used}
        try:
            passed, detail = getattr(self, "_" + kind)(name, **params)
        except (ValueError, KeyError, TypeError) as exc:
            passed, detail = False, {"error": str(exc)}
        entry.update(passed=bool(passed), detail=detail)
        self.log.append(entry)
        if not passed:
            raise CheckpointFailed(f"checkpoint_failed: {name} ({kind}) {detail}")
        return entry

    def _mark(self, name):
        if name in self.marks:
            raise ValueError(f"checkpoint mark {name!r} already exists in this program")
        self.marks[name] = {"tick": self.rt.tick, "poses": {a: s.ee_pose for a, s in self.rt.obs.arms.items()}}
        return True, {}

    def _native_reserve(self, name, *, min_remaining):
        if type(min_remaining) is not int or min_remaining < 0:
            raise ValueError("min_remaining must be a non-negative integer")
        remaining = self.rt.backend.budget().remaining
        return remaining is None or remaining >= min_remaining, {"native_steps_remaining": remaining,
                                                                 "min_remaining": min_remaining}

    def _pose(self, name, *, arm, xyz, tolerance, frame="flange"):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        target = np.asarray(xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("xyz must be three finite metres")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < tolerance <= 0.1:
            raise ValueError("tolerance must be in (0, 0.1] metres")
        actual = self.rt.grasp_point(arm) if frame == "grasp" else self.rt.obs.arms[arm].ee_pose.position
        error = float(np.linalg.norm(actual - target))
        return error <= tolerance, {"arm": arm, "frame": frame, "actual_xyz": [round(v, 4) for v in actual],
                                    "error_m": round(error, 4), "tolerance_m": tolerance}
