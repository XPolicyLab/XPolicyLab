"""The declared handover (OURS_HANDOVER, acts with OURS_ARM_GUARD): a declared handover relaxes F-g between the two
grippers only.

Declaration (the runtime side, agp_core/runtime_ext.py DeclareMixin): an arm action whose evidence has `handover`, a short
text that names the other gripper ("right gripper" in the left arm's evidence, "left gripper" in the right arm's;
names_other_gripper). The runtime refuses a `handover` evidence that does not name the other gripper before dispatch
(INVALID_EVIDENCE, HANDOVER_TEXT) and otherwise adds ARGS_KEY: true to that arm's dispatched args (the worker sees no
evidence; the key is not a model-facing argument). A stage is a declared handover when both arms are in it and both
carry the key (declared).

What changes in a declared stage (HandoverMixin over agp_core/speed_profiles.py's SpeedProfilesMixin; every other stage
goes to the unchanged F-g code):
  * the pairs of shapes split into two classes (agp_core/arm_guard.py clearance(pairs=...)): `relaxed` = a finger or
    wrist-camera link of one gripper (HANDOVER_LINKS: link7, link8, camera_base, camera) against one of the other
    gripper; `usual` = every other pair (link6, the gripper body, included);
  * layer 1 (guard_stage_handover, the same stage timeline, interpolation and delay search as guard_stage): the usual
    class keeps F-g's margin (GUARD_MARGIN_M 25 mm, or GUARD_V2_MARGIN_M 15 mm with OURS_GUARD_V2) or its start minus
    2 mm, the relaxed class HANDOVER_CLEARANCE_M (5 mm, a fixed value) or its start minus 2 mm; the stage passes
    when both hold (with a delay of one arm when one is needed and found, as before);
  * layer 2 (before every control step): the usual class keeps GUARD_HARD_M (10 mm) or its start minus 2 mm, the
    relaxed class HANDOVER_CLEARANCE_M or its start minus 2 mm. One value for both layers: the relaxed class has no
    layer-1 allowance above its layer-2 limit (a lag that brings two finger links below 5 mm stops both arms, the same
    stop as F-g's; it is never sent closer).
A refused undeclared stage in which only one arm declared gets one more sentence (ONE_ARM_HINT). Records: arm_guard.jsonl
as before, a declared stage's line with handover true and both classes; handover.json in the session (the constants).
"""
from __future__ import annotations

import math
import re
import time

import numpy as np

from agp_core import speed_profiles as sp

HANDOVER_LINKS = ("link7", "link8", "camera_base", "camera")
HANDOVER_CLEARANCE_M = 0.005          # fixed value, layers 1 and 2, relaxed pairs only
ARGS_KEY = "ours_handover"            # set by the runtime in an arm's dispatched args (never by the model)
EVIDENCE_KEY = "handover"
HANDOVER_TEXT = ("evidence.handover must name the other gripper ({other} gripper), for example "
                 "'{other} gripper takes it'")
ONE_ARM_HINT = (" A handover relaxes this only between the two grippers' fingers and wrist cameras, and only when both "
                "arms are in the step and each declares it (evidence handover naming the other gripper).")


def other_arm(arm):
    return "right" if arm == "left" else "left"


def names_other_gripper(arm, text):
    """True when `text` (an evidence value) names the gripper of the arm other than `arm`."""
    return (isinstance(text, str) and arm in ("left", "right")
            and re.search(rf"\b{other_arm(arm)}[\s_-]+gripper\b", text, re.I) is not None)


def relaxed_pair(link_a, link_b):
    return link_a in HANDOVER_LINKS and link_b in HANDOVER_LINKS


def usual_pair(link_a, link_b):
    return not relaxed_pair(link_a, link_b)


def flagged(plans):
    """The arms of the prepared plans whose dispatched args carry the declaration."""
    return sorted(arm for arm, plan in plans.items() if bool((plan.get("args") or {}).get(ARGS_KEY)))


def declared(plans, arms):
    """A declared handover: every arm of the worker is in the stage and carries the declaration."""
    return set(arms) <= set(plans) and set(flagged(plans)) >= set(arms)


def _classes(geometry, timeline, fingers, held, margins, report=None, stop_below=None):
    """{class: (per_time, detail)} of the usual and relaxed pairs."""
    out = {}
    for name, pairs in (("usual", usual_pair), ("relaxed", relaxed_pair)):
        out[name] = geometry.clearance(timeline, fingers, held, threshold=margins[name] + sp.GUARD_EXACT_ABOVE_M,
                                       report=report, pairs=pairs,
                                       stop_below=None if stop_below is None else stop_below[name])
    return out


def _class_record(per_time, detail, threshold):
    return {"start_clearance_m": float(per_time[0]), "threshold_m": float(threshold),
            "min_clearance_m": float(per_time.min()), "closest": detail,
            "ok": bool(float(per_time.min()) >= threshold)}


def guard_stage_handover(geometry, arms, plans, measured, fingers, held, margin=sp.GUARD_MARGIN_M,
                         delay_max=sp.GUARD_DELAY_MAX, v2=False):
    """Layer 1 of F-g for a declared handover stage (module docstring): guard_stage with two classes of pairs. Returns
    the record of guard_stage (ok, checked, start_clearance_m = the usual class's, threshold_m, min_clearance_m,
    closest, step_of_min, delay, error) plus handover true and usual / relaxed {start_clearance_m, threshold_m,
    min_clearance_m, closest, ok}."""
    moving = [arm for arm, p in plans.items() if p["command"] in sp.MOTIONS]
    record = {"ok": True, "moving": moving, "handover": True, "handover_clearance_m": HANDOVER_CLEARANCE_M}
    if not moving:
        record["checked"] = False
        return record
    started = time.monotonic()
    stats = {}
    margins = {"usual": margin, "relaxed": HANDOVER_CLEARANCE_M}
    timeline, steps = sp.stage_timeline(arms, plans, measured)
    found = _classes(geometry, timeline, fingers, held, margins, report=stats)
    thresholds = {name: min(margins[name], float(found[name][0][0]) - sp.GUARD_START_TOLERANCE_M) for name in found}
    classes = {name: _class_record(*found[name], thresholds[name]) for name in found}
    overall = np.minimum(found["usual"][0], found["relaxed"][0])
    worst = min(found, key=lambda name: float(found[name][0].min()) - thresholds[name])     # the tighter class
    detail = found[worst][1]
    record.update(checked=True, steps=steps, fingers_m=fingers, held=held,
                  start_clearance_m=classes["usual"]["start_clearance_m"], threshold_m=thresholds["usual"],
                  min_clearance_m=float(overall.min()), closest=detail,
                  step_of_min=int(math.ceil(detail["time"] / sp.INTERP_SUBSTEPS)), delay=None,
                  usual=classes["usual"], relaxed=classes["relaxed"])
    if all(c["ok"] for c in classes.values()):
        record.update(seconds=round(time.monotonic() - started, 3), **stats)
        return record
    options = []
    end_conflict = False
    if len(moving) == 2:
        limits = {arm: delay_max for arm in moving}
        if v2:
            limits = {arm: int(plans[other]["run_samples"]) for arm in moving for other in moving if other != arm}
            end_conflict = any(float(found[name][0][-1]) < thresholds[name] for name in found)
            record.update(v2=True, delay_limits=limits, end_conflict=end_conflict)
        for arm in moving:
            for d in range(1, limits[arm] + 1):
                length = max(p["run_samples"] + (d if a == arm else 0) for a, p in plans.items())
                options.append((length, d, arm))
        options.sort()
        if end_conflict:
            options = []
    elif v2:
        record.update(v2=True)
    tried = 0
    for length, d, arm in options:
        delayed, _ = sp.stage_timeline(arms, plans, measured, {arm: d})
        again = _classes(geometry, delayed, fingers, held, margins, report=stats, stop_below=thresholds)
        tried += 1
        if all(float(again[name][0].min()) >= thresholds[name] for name in again):
            record.update(delay={"arm": arm, "steps": d, "stage_steps": length,
                                 "min_clearance_m": float(min(float(again[n][0].min()) for n in again)),
                                 "closest": {n: again[n][1] for n in again}},
                          tried=tried, seconds=round(time.monotonic() - started, 3), **stats)
            return record
    step = int(math.ceil(detail["time"] / sp.INTERP_SUBSTEPS))
    names = " and ".join(f"the {arm} arm's {name}" for arm, name in sorted(detail["links"].items()))
    which = ("the grippers' fingers and wrist cameras may come within "
             f"{HANDOVER_CLEARANCE_M * 1000:.0f} mm of each other" if worst == "relaxed" else
             f"every part other than the grippers' fingers and wrist cameras keeps {margin * 1000:.0f} mm")
    reason = (f"Two-arm clearance in a declared handover ({which}): {names} would come within "
              f"{detail['distance_m'] * 1000:.0f} mm of each other at control step {step} of {steps} (the stage must "
              f"keep {thresholds[worst] * 1000:.0f} mm)")
    if v2 and len(moving) == 2:
        reason += ("; no delay of either arm keeps it. " if end_conflict else
                   "; delaying either arm until the other has finished does not keep it. ") + sp.GUARD_V2_PAIR
    elif v2:
        reason += "; the other arm does not move in this step. " + sp.GUARD_V2_SINGLE
    elif len(moving) == 2:
        reason += f"; delaying either arm by up to {delay_max} steps does not keep it"
    else:
        reason += "; the other arm does not move in this step"
    record.update(ok=False, error=reason, tried=tried, seconds=round(time.monotonic() - started, 3), **stats)
    return record


class HandoverMixin:
    """In front of a worker class with SpeedProfilesMixin and F-g on (module docstring)."""

    def __init__(self, env, session):
        super().__init__(env, session)
        from migration import worker as base
        base.atomic(self.session / "handover.json", {
            "switch": "OURS_HANDOVER", "declaration": "both arms of the stage carry the runtime's args key "
                                                      f"{ARGS_KEY} (evidence {EVIDENCE_KEY} naming the other gripper)",
            "relaxed_links": list(HANDOVER_LINKS), "relaxed_clearance_m": HANDOVER_CLEARANCE_M,
            "layers": "1 and 2", "usual_margin_m": (sp.GUARD_V2_MARGIN_M if self._sp_flags.get("guard_v2")
                                                     else sp.GUARD_MARGIN_M),
            "usual_hard_m": sp.GUARD_HARD_M, "start_tolerance_m": sp.GUARD_START_TOLERANCE_M,
            "source": "agp_core/handover.py"})

    def _sp_guard_stage(self, plans):
        if not declared(plans, self.arms):
            record = super()._sp_guard_stage(plans)
            if not record.get("ok") and flagged(plans):
                error = str(record.get("error") or "").rstrip()
                record["error"] = error + ("" if error.endswith(".") else ".") + ONE_ARM_HINT
                record["handover_declared_by"] = flagged(plans)
            return record
        measured = {arm: np.asarray(self.rm.get_joint(c["robot"])[0], float) for arm, c in self.arms.items()}
        fingers, held = self._sp_stage_state(plans)
        v2 = bool(self._sp_flags.get("guard_v2"))
        record = guard_stage_handover(self._sp_guard, tuple(self.arms), plans, measured, fingers, held,
                                      margin=sp.GUARD_V2_MARGIN_M if v2 else sp.GUARD_MARGIN_M, v2=v2)
        record["sequence"] = self.motion_sequence
        if record.get("checked"):
            self._sp_stage = {"start_clearance_m": record["usual"]["start_clearance_m"], "fingers": fingers,
                              "held": held, "handover": {"start_clearance_m": record["relaxed"]["start_clearance_m"]}}
        delay = record.get("delay")
        if record["ok"] and delay:                           # as SpeedProfilesMixin._sp_guard_stage
            plan = plans[delay["arm"]]
            plan["ours_guard_delay"] = delay["steps"]
            hold = [np.asarray(plan["start"], float).copy() for _ in range(delay["steps"])]
            plan["points"] = hold + list(plan["points"])
            plan["full_samples"] += delay["steps"]
            plan["run_samples"] += delay["steps"]
        return record

    def _sp_guard_step(self, targets):
        stage = self._sp_stage or {}
        if "handover" not in stage:
            return super()._sp_guard_step(targets)
        measured = {arm: np.asarray(self.rm.get_joint(c["robot"])[0], float) for arm, c in self.arms.items()}
        fractions = np.arange(0, sp.INTERP_SUBSTEPS + 1) / sp.INTERP_SUBSTEPS
        timeline = {arm: np.asarray([measured[arm] + f * (np.asarray(targets[arm], float) - measured[arm])
                                     for f in fractions]) for arm in self.arms}
        hard = {"usual": sp.GUARD_HARD_M, "relaxed": HANDOVER_CLEARANCE_M}
        starts = {"usual": float(stage.get("start_clearance_m", np.inf)),
                  "relaxed": float(stage["handover"].get("start_clearance_m", np.inf))}
        found = _classes(self._sp_guard, timeline, stage.get("fingers"), stage.get("held") or {}, hard)
        smallest = float(min(found[name][0].min() for name in found))
        for name in ("usual", "relaxed"):
            per_time, detail = found[name]
            threshold = min(hard[name], starts[name] - sp.GUARD_START_TOLERANCE_M)
            if float(per_time.min()) < threshold:
                names = " and ".join(f"{arm} {link}" for arm, link in sorted(detail["links"].items()))
                return (f"two-arm clearance below {threshold * 1000:.0f} mm in a declared handover ({names}: "
                        f"{detail['distance_m'] * 1000:.1f} mm between the measured joints and the next targets): "
                        f"both arms stopped", dict(detail, handover_class=name)), smallest
        return None, smallest


def worker_class(base_cls):
    """`base_cls` with HandoverMixin in front (official_worker composes it only with OURS_HANDOVER and OURS_ARM_GUARD
    on, after agp_core/speed_profiles.py)."""
    if not issubclass(base_cls, sp.SpeedProfilesMixin):
        raise ValueError("OURS_HANDOVER acts on F-g: the worker class needs SpeedProfilesMixin (OURS_ARM_GUARD)")
    return type(f"Handover{base_cls.__name__}", (HandoverMixin, base_cls),
                {"__module__": __name__, "__doc__": f"{base_cls.__qualname__} with a declared handover (G6)."})
