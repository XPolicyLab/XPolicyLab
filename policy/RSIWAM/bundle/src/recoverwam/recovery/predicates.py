"""Local evaluation predicates; these never control the deployed policy."""
from dataclasses import dataclass, field
from typing import Mapping


def category_mixing(objects):
    """Input containment must be computed with the native task geometry."""
    baskets = {}
    for obj in objects:
        if (obj.get("released") is True and obj.get("below_rim") is True
                and obj.get("speed_m_s", float("inf")) < .01 and obj.get("native_container") is not None):
            baskets.setdefault(obj["native_container"], []).append(obj)
    return sorted(name for name, rows in baskets.items() if len({r["category"] for r in rows}) > 1)


def wrong_digit_placements(objects):
    return sorted(obj["id"] for obj in objects if obj.get("released") is True
                  and obj.get("native_orientation_accepted") is True
                  and obj.get("speed_m_s", float("inf")) < .01
                  and obj.get("native_mat") is not None
                  and obj["native_mat"] != obj["required_mat"])


def lost_structure(previous_subgoals, current_subgoals, all_parts_reachable):
    if all_parts_reachable is not True:
        return []
    return sorted(k for k,v in previous_subgoals.items() if v is True and current_subgoals.get(k) is False)


def failed_tool_acquisition(attempts):
    failed = [a for a in attempts if all(a.get(k) is True for k in
              ("approach_observed", "contact_observed", "lift_attempted", "retreat_observed",
               "tool_left_on_table", "gripper_empty_after_lift"))]
    # Disjoint action intervals prevent one stalled attempt being counted twice.
    failed.sort(key=lambda a:a["start_step"])
    count, end = 0, -1
    for a in failed:
        if a["start_step"] > end and a["end_step"] >= a["start_step"]:
            count += 1
            end = a["end_step"]
    return count >= 2


@dataclass
class LocalErrorTracker:
    """Track one fixed error and preserve completed subgoals across repair."""
    persistence_actions: int = 24
    clear_actions: int = 12
    last_step: int | None = None
    active_key: tuple = ()
    error_start: int | None = None
    qualified_step: int | None = None
    clear_start: int | None = None
    repaired_step: int | None = None
    protected: set = field(default_factory=set)
    damaged: set = field(default_factory=set)

    def observe(self, step, error_entities, native_subgoals: Mapping[str, bool], *, terminal=False):
        if type(step) is not int or step < 0 or (self.last_step is not None and step != self.last_step + 1):
            raise ValueError("A contiguous native control-step trace is required")
        if any(type(v) is not bool for v in native_subgoals.values()):
            raise ValueError("Unknown subgoals cannot be counted as preserved")
        self.last_step = step
        key = tuple(sorted(error_entities))
        if self.qualified_step is None:
            if key != self.active_key:
                self.active_key, self.error_start = key, step if key else None
                self.protected = {k for k,v in native_subgoals.items() if v} if key else set()
                self.damaged = set()
            self.damaged.update(k for k in self.protected if native_subgoals.get(k) is not True)
            if key and not terminal and step-self.error_start+1 >= self.persistence_actions:
                self.qualified_step = step
        else:
            self.damaged.update(k for k in self.protected if native_subgoals.get(k) is not True)
            # A different error is not evidence that the original was repaired.
            if not key:
                if self.clear_start is None:
                    self.clear_start = step
                if step-self.clear_start+1 >= self.clear_actions and self.repaired_step is None:
                    self.repaired_step = step
            else:
                self.clear_start = None
        return dict(failure_persistence_verified=self.qualified_step is not None,
                    first_qualified_step=self.qualified_step, repaired_step=self.repaired_step,
                    local_error_cleared=self.clear_start is not None and not key and step-self.clear_start+1 >= self.clear_actions,
                    protected_subgoals_preserved=not self.damaged, damaged_subgoals=sorted(self.damaged),
                    recoverability_verified=False)
