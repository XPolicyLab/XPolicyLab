"""Bounded physical-time stability checks, without inferring object grasp."""

from __future__ import annotations

from collections import deque
import math


def action_targets(env, action):
    result = {}
    # An env that cannot name its grippers yields no targets at all, which the
    # monitor then reports as "unavailable". That is the safe direction: an
    # absent measurement is documented as not being grasp evidence, so it can
    # only withhold a candidate, never invent one.
    manager = getattr(env, "robot_manager", None)
    if manager is None:
        return result
    for robot in manager.robot_list:
        if robot.type != "target" or getattr(robot, "ee_type", "gripper") != "gripper":
            continue
        key = manager.process_name(robot.gripper_name)
        if key in action:
            value = float(action[key][0])
            if not math.isfinite(value):
                raise ValueError("gripper action target must be finite")
            result[robot.arm_name.split("_")[0]] = min(max(value, 0.0), 1.0)
    return result


class GripperSettle:
    def __init__(self, cfg, control_dt, previous_targets):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.min_s = float(cfg.get("min_s", 0.5))
        self.stable_s = float(cfg.get("stable_s", 0.2))
        self.max_s = float(cfg.get("max_s", 2.5))
        self.tolerance = float(cfg.get("tolerance", 0.002))
        self.dt = control_dt
        if (not all(math.isfinite(v) and v > 0 for v in (self.min_s, self.stable_s, self.max_s, self.tolerance))
                or self.max_s < max(self.min_s, self.stable_s, self.dt)):
            raise ValueError("invalid gripper_settle durations or tolerance")
        self.previous_targets = previous_targets
        self.arms = {}

    def changed(self, targets):
        return {arm: target for arm, target in targets.items()
                if arm in self.previous_targets and abs(target - self.previous_targets[arm]) > 1e-9}

    def expect(self, targets):
        for arm in self.changed(targets):
            self.arms[arm] = {"status": "not_executed", "steps": 0, "moved": False, "added_steps": 0}

    @property
    def pending(self):
        return any(state["status"] == "pending" for state in self.arms.values())

    def needs_measurement(self, targets):
        return self.enabled and (bool(self.changed(targets)) or self.pending)

    def advance(self, targets, before, after, floors, *, added=False):
        for arm, target in self.changed(targets).items():
            initial, floor = before.get(arm), floors.get(arm)
            state = {"status": "pending" if self.enabled else "disabled", "steps": 0,
                     "moved": False, "added_steps": 0, "target": target,
                     "initial": initial, "floor": floor, "samples": deque()}
            if self.enabled:
                if initial is None or floor is None or not all(math.isfinite(float(v)) for v in (initial, floor)):
                    state["status"] = "unavailable"
                else:
                    state["initial"] = float(initial)
                    state["floor"] = float(floor)
                    state["samples"].append((0.0, float(initial)))
                    state["initial_near_target"] = abs(float(initial) - max(target, float(floor))) <= self.tolerance
            self.arms[arm] = state
        self.previous_targets.update(targets)
        for arm, state in self.arms.items():
            if state["status"] != "pending":
                continue
            state["steps"] += 1
            state["added_steps"] += int(added)
            elapsed = state["steps"] * self.dt
            value = after.get(arm)
            floor = floors.get(arm)
            if value is None or floor is None or not all(math.isfinite(float(v)) for v in (value, floor)):
                state["status"] = "unavailable"
                continue
            value, floor = float(value), float(floor)
            state["floor"] = floor
            state["moved"] |= abs(value - state["initial"]) > self.tolerance
            samples = state["samples"]
            samples.append((elapsed, value))
            cutoff = elapsed - self.stable_s
            while len(samples) > 1 and samples[1][0] <= cutoff + 1e-9:
                samples.popleft()
            stable = (elapsed - samples[0][0] >= self.stable_s - 1e-9
                      and max(v for _, v in samples) - min(v for _, v in samples) <= self.tolerance)
            responsive = state["moved"] or state["initial_near_target"]
            if elapsed >= self.min_s - 1e-9 and stable and responsive:
                at_limit = (abs(value - floor) <= self.tolerance
                            and max(state["target"], floor) <= floor + self.tolerance)
                state["status"] = "closed_at_limit" if at_limit else "settled"
            elif elapsed + self.dt > self.max_s + 1e-9:
                state["status"] = "timeout" if responsive else "unresponsive"

    def interrupt(self, reason):
        for state in self.arms.values():
            if state["status"] == "pending":
                state.update(status="interrupted", reason=reason)

    def report(self):
        # Numerical finger measurements stay private to this monitor. This
        # report is safe to pass to a policy with measure_gripper disabled.
        return {arm: {"status": state["status"], "sim_elapsed_s": round(state["steps"] * self.dt, 9),
                      "moved": state["moved"], "added_steps": state["added_steps"],
                      **({"reason": state["reason"]} if "reason" in state else {})}
                for arm, state in self.arms.items()}
