"""The nine public reusable skills.

pick_and_hold, stage_near, lift_upright, transport_held, place_for_regrasp,
seat_probe, press_to_contact, knob_cycle, tool_stroke.

Bodies compose motion programs directly (never the runtime's blocking APIs,
which would contend for the arm this skill already leases).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from knowin_agent.core.geometry import Pose, axis_angle_matrix, flange_for_point, matrix_to_quat, unit, wrap180
from knowin_agent.runtime.motion import limits_with, move_flange, set_gripper
from knowin_agent.skills.base import SkillSpec, arm_binding

OK = ("reached",)
OK_OR_CONTACT = ("reached", "blocked")
ARM = {"type": "string", "enum": ["left", "right"], "description": "arm that executes the skill"}
POINT = {"description": "reference id or [x, y, z] in metres (world)"}


# -- shared helpers ------------------------------------------------------------------------------

def _rotation(rt, arm, yaw=None, pitch=None):
    return rt._rotation(arm, yaw, pitch, None, None)


def _working_offset(rt, arm, frame):
    if frame == "tip":
        if arm not in rt.tip_offsets:
            raise ValueError("frame 'tip' needs inspect(mode='axis_tip', arm=...) of the held object first")
        return np.asarray(rt.tip_offsets[arm]["tip_in_flange"], dtype=float)
    return np.asarray(rt.profile.grasp_offset, dtype=float)


def _goto(ctx, rt, arm, point, rotation, *, frame="grasp", **kw):
    flange = flange_for_point(point, rotation, _working_offset(rt, arm, frame))
    return (yield from move_flange(ctx, arm, flange, limits=rt.limits, **kw))


def _working_point(ctx, rt, arm, frame="grasp"):
    return ctx.obs.arms[arm].ee_pose.transform(_working_offset(rt, arm, frame))


def _spin_about(ctx, rt, arm, axis, degrees, pivot):
    flange = ctx.obs.arms[arm].ee_pose
    r = axis_angle_matrix(axis, math.radians(degrees))
    return Pose.make(pivot + r @ (flange.position - pivot), matrix_to_quat(r @ flange.rotation))


def _tool_axis(ctx, rt, arm):
    return unit(ctx.obs.arms[arm].ee_pose.rotation @ np.asarray(rt.profile.tool_axis, dtype=float))


def _status(stage):
    return stage.get("status") if isinstance(stage, dict) else None


def _resolve_grasp(rt, bindings):
    """Turn an object or grasp reference into concrete grasp point, yaw, pitch and arm."""
    target = bindings["target"]
    ref = rt.ctx.references.get(target) if isinstance(target, str) else None
    if ref is not None and ref.kind == "object":
        arm = bindings.get("arm", "auto")
        candidates = rt.grasp_candidates(ref.ref_id, arm=arm, max_candidates=1,
                                         depth_m=bindings.get("depth_m")).data["candidates"]
        if not candidates:
            raise ValueError("no grasp candidate for that object")
        ref = rt.ctx.references.get(candidates[0]["ref"])
    if ref is not None and ref.kind == "grasp":
        v = ref.values
        bindings.setdefault("yaw_deg", v["yaw_deg"])
        bindings.setdefault("pitch_deg", v["pitch_deg"])
        if bindings.get("arm") in (None, "auto"):
            bindings["arm"] = v["arm"]
        bindings["target"] = list(v["grasp_xyz"])
    if bindings.get("arm") in (None, "auto"):
        raise ValueError("arm must be given unless target is a grasp reference")
    return bindings


# -- guards --------------------------------------------------------------------------------------

def arm_is_free(rt, b):
    """the arm has no running track"""
    busy = rt.executor.busy_arms()
    if b["arm"] in busy:
        return f"{b['arm']} arm is busy with {busy[b['arm']].label}"
    return None


def gripper_closed(rt, b):
    """the arm's gripper is closed (holding)"""
    if float(rt.obs.arms[b["arm"]].gripper_command) > 0.5:
        return f"{b['arm']} gripper is open; this skill expects a held object"
    return None


def target_fresh(rt, b):
    """the target reference is not flagged as moved"""
    target = b.get("target")
    if isinstance(target, str):
        ref = rt.ctx.references.get(target)
        if ref.invalidated:
            return f"reference {target} was invalidated ({ref.moved_reason})"
    return None


# -- bodies --------------------------------------------------------------------------------------

def pick_and_hold(ctx, rt, arm, target, yaw_deg=None, pitch_deg=90.0, hover_m=0.06, lift_m=0.06,
                  close_ticks=8, depth_m=None):
    point = rt._point(target)
    rotation = _rotation(rt, arm, yaw_deg, pitch_deg)
    out = {"grasp_xyz": [round(v, 4) for v in point], "yaw_deg": yaw_deg, "pitch_deg": pitch_deg}
    out["open"] = yield from set_gripper(ctx, arm, 1.0, settle_ticks=4)
    out["hover"] = yield from _goto(ctx, rt, arm, point + [0, 0, hover_m], rotation, gripper=1.0)
    if _status(out["hover"]) not in OK:
        return {**out, "status": "hover_failed"}
    out["descend"] = yield from _goto(ctx, rt, arm, point, rotation, gripper=1.0, precision=True, stop_on_contact=True)
    if _status(out["descend"]) not in OK_OR_CONTACT:
        return {**out, "status": "descend_failed"}
    out["close"] = yield from set_gripper(ctx, arm, 0.0, settle_ticks=close_ticks)
    out["lift"] = yield from _goto(ctx, rt, arm, _working_point(ctx, rt, arm) + [0, 0, lift_m], rotation, gripper=0.0)
    out["status"] = "lifted" if _status(out["lift"]) in OK else "lift_failed"
    out["note"] = "a reached lift does not prove the object came along; check the wrist view or gripper width"
    return out


def stage_near(ctx, rt, arm, target, standoff_m=0.03, approach=(0.0, 0.0, -1.0), frame="grasp", label=None):
    point = rt._point(target)
    d = unit(approach)
    goal = point - d * float(standoff_m)
    rotation = ctx.obs.arms[arm].ee_pose.rotation
    move = yield from _goto(ctx, rt, arm, goal, rotation, frame=frame, precision=True, stop_on_contact=True)
    reached = _working_point(ctx, rt, arm, frame)
    residual = reached - goal
    ref = rt.ctx.references.add("point", label or f"staged near {target}", {
        "xyz": [round(v, 4) for v in reached], "target_xyz": [round(v, 4) for v in point],
        "residual_mm": [round(v * 1000, 2) for v in residual], "approach": [round(v, 3) for v in d], "arm": arm},
        {"method": "stage_near", "tick": ctx.obs.tick}, ctx.obs.tick)
    return {"status": "staged" if _status(move) in OK_OR_CONTACT else "stage_failed", "move": move,
            "contact_ref": ref.ref_id, "residual_mm": [round(v * 1000, 2) for v in residual]}


def lift_upright(ctx, rt, arm, clearance_m=0.08, upright=(0.0, 0.0, 1.0)):
    rotation = ctx.obs.arms[arm].ee_pose.rotation
    out = {"lift": (yield from _goto(ctx, rt, arm, _working_point(ctx, rt, arm) + [0, 0, clearance_m], rotation))}
    if _status(out["lift"]) not in OK:
        return {**out, "status": "lift_failed"}
    if arm not in rt.tip_offsets:
        return {**out, "status": "lifted", "note": "no held axis measured; orientation unchanged"}
    flange = ctx.obs.arms[arm].ee_pose
    axis = unit(flange.rotation @ np.asarray(rt.tip_offsets[arm]["axis_in_flange"], dtype=float))
    goal = unit(upright)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ goal)))))
    if angle > 1.0:
        rot_axis = np.cross(axis, goal)
        target = _spin_about(ctx, rt, arm, unit(rot_axis), min(angle, 120.0), _working_point(ctx, rt, arm))
        out["reorient"] = yield from move_flange(ctx, arm, target, limits=rt.limits)
    out["axis_error_deg"] = round(angle, 2)
    out["status"] = "upright" if _status(out.get("reorient", {"status": "reached"})) in OK else "reorient_failed"
    return out


def transport_held(ctx, rt, arm, target, clearance_m=0.08, lower=True, yaw_deg=None, frame="grasp"):
    point = rt._point(target)
    start = _working_point(ctx, rt, arm, frame)
    carry_z = max(float(point[2]), float(start[2])) + float(clearance_m)
    keep = ctx.obs.arms[arm].ee_pose.rotation
    rotation = keep if yaw_deg is None else rt._rotation(arm, yaw_deg, 90.0, None, None)
    out = {"raise": (yield from _goto(ctx, rt, arm, [start[0], start[1], carry_z], keep, frame=frame, gripper=0.0))}
    if _status(out["raise"]) not in OK:
        return {**out, "status": "raise_failed"}
    out["carry"] = yield from _goto(ctx, rt, arm, [point[0], point[1], carry_z], rotation, frame=frame, gripper=0.0)
    if _status(out["carry"]) not in OK:
        return {**out, "status": "carry_failed"}
    if lower:
        out["lower"] = yield from _goto(ctx, rt, arm, point, rotation, frame=frame, gripper=0.0, precision=True,
                                        stop_on_contact=True)
        if _status(out["lower"]) not in OK_OR_CONTACT:
            return {**out, "status": "lower_failed"}
    out["status"] = "transported"
    return out


def place_for_regrasp(ctx, rt, arm, target, yaw_deg=None, clearance_m=0.05, press_m=0.0, retreat_m=0.06,
                      open_ticks=6):
    point = rt._point(target)
    carried = yield from transport_held(ctx, rt, arm, point, clearance_m=clearance_m, lower=True, yaw_deg=yaw_deg)
    out = {"transport": carried}
    if carried.get("status") != "transported":
        return {**out, "status": "release_refused", "reason": "placement pose not reached; still holding"}
    if press_m > 0:
        rotation = ctx.obs.arms[arm].ee_pose.rotation
        out["press"] = yield from _goto(ctx, rt, arm, point - [0, 0, press_m], rotation, gripper=0.0, precision=True,
                                        stop_on_contact=True)
    out["open"] = yield from set_gripper(ctx, arm, 1.0, settle_ticks=open_ticks)
    rotation = ctx.obs.arms[arm].ee_pose.rotation
    out["retreat"] = yield from _goto(ctx, rt, arm, _working_point(ctx, rt, arm) + [0, 0, retreat_m], rotation, gripper=1.0)
    rt.ctx.references.note_interaction(point, f"{arm} released an object at tick {ctx.obs.tick}")
    rt.tip_offsets.pop(arm, None)
    out["status"] = "released"
    return out


def seat_probe(ctx, rt, arm, target, axis=(0.0, 0.0, -1.0), depth_m=0.03, standoff_m=0.01, step_m=0.002,
               frame="tip"):
    if frame == "tip" and arm not in rt.tip_offsets:
        frame = "grasp"
    point = rt._point(target)
    d = unit(axis)
    rotation = ctx.obs.arms[arm].ee_pose.rotation
    out = {"frame": frame}
    out["align"] = yield from _goto(ctx, rt, arm, point - d * float(standoff_m), rotation, frame=frame, gripper=0.0,
                                    precision=True)
    if _status(out["align"]) not in OK:
        return {**out, "status": "align_failed"}
    start = _working_point(ctx, rt, arm, frame)
    limits = limits_with(translation_step_m=float(step_m), precision_step_m=float(step_m), stall_window=6)
    goal = flange_for_point(point + d * float(depth_m), rotation, _working_offset(rt, arm, frame))
    out["probe"] = yield from move_flange(ctx, arm, goal, gripper=0.0, limits=limits, stop_on_contact=True)
    travelled = float((_working_point(ctx, rt, arm, frame) - start) @ d)
    out["travelled_m"] = round(travelled, 4)
    out["stopped_by"] = {"blocked": "contact", "reached": "depth_limit"}.get(_status(out["probe"]), _status(out["probe"]))
    out["status"] = "probed"
    return out


def press_to_contact(ctx, rt, arm, target, depth_m=0.01, hover_m=0.05, gripper=0.0, yaw_deg=None):
    point = rt._point(target)
    rotation = _rotation(rt, arm, yaw_deg, 90.0) if yaw_deg is not None else ctx.obs.arms[arm].ee_pose.rotation
    out = {"hover": (yield from _goto(ctx, rt, arm, point + [0, 0, hover_m], rotation, gripper=gripper))}
    if _status(out["hover"]) not in OK:
        return {**out, "status": "hover_failed"}
    out["press"] = yield from _goto(ctx, rt, arm, point - [0, 0, depth_m], rotation, gripper=gripper, precision=True,
                                    stop_on_contact=True)
    out["retract"] = yield from _goto(ctx, rt, arm, point + [0, 0, hover_m], rotation, gripper=gripper)
    out["status"] = "pressed" if _status(out["press"]) in OK_OR_CONTACT else "press_failed"
    out["contact"] = _status(out["press"]) == "blocked"
    return out


def knob_cycle(ctx, rt, arm, cycles=1, degrees=90.0, withdraw_m=0.0, descend_per_cycle_m=0.0, close_ticks=6,
               open_ticks=4):
    out = {"cycles": []}
    for _ in range(max(1, int(cycles))):
        row = {}
        row["close"] = yield from set_gripper(ctx, arm, 0.0, settle_ticks=close_ticks)
        pivot = _working_point(ctx, rt, arm)
        axis = _tool_axis(ctx, rt, arm)
        turned = _spin_about(ctx, rt, arm, axis, float(degrees), pivot)
        if descend_per_cycle_m:
            turned = turned.with_position(turned.position + axis * float(descend_per_cycle_m))
        row["turn"] = yield from move_flange(ctx, arm, turned, gripper=0.0, limits=rt.limits, precision=True,
                                             stop_on_contact=True)
        row["open"] = yield from set_gripper(ctx, arm, 1.0, settle_ticks=open_ticks)
        if withdraw_m:
            back = ctx.obs.arms[arm].ee_pose
            row["withdraw"] = yield from move_flange(ctx, arm, back.with_position(back.position - axis * withdraw_m),
                                                     gripper=1.0, limits=rt.limits)
        reset = _spin_about(ctx, rt, arm, axis, -float(degrees), _working_point(ctx, rt, arm))
        row["reset"] = yield from move_flange(ctx, arm, reset, gripper=1.0, limits=rt.limits)
        if withdraw_m:
            ahead = ctx.obs.arms[arm].ee_pose
            row["approach"] = yield from move_flange(ctx, arm, ahead.with_position(ahead.position + axis * withdraw_m),
                                                     gripper=1.0, limits=rt.limits, precision=True, stop_on_contact=True)
        row["grasp_z"] = round(float(_working_point(ctx, rt, arm)[2]), 4)
        out["cycles"].append({k: (_status(v) if isinstance(v, dict) else v) for k, v in row.items()})
        if _status(row["turn"]) not in OK_OR_CONTACT:
            out["status"] = "turn_failed"
            return out
    out["status"] = "cycled"
    return out


def tool_stroke(ctx, rt, arm, start, direction, length_m, depth_m=0.0, hover_m=0.04, frame="tip"):
    if frame == "tip" and arm not in rt.tip_offsets:
        frame = "grasp"
    p0 = rt._point(start)
    d = unit(direction)
    rotation = ctx.obs.arms[arm].ee_pose.rotation
    out = {"frame": frame}
    out["hover"] = yield from _goto(ctx, rt, arm, p0 + [0, 0, hover_m], rotation, frame=frame, gripper=0.0)
    if _status(out["hover"]) not in OK:
        return {**out, "status": "hover_failed"}
    out["lower"] = yield from _goto(ctx, rt, arm, p0 - [0, 0, depth_m], rotation, frame=frame, gripper=0.0,
                                    precision=True, stop_on_contact=True)
    end = _working_point(ctx, rt, arm, frame) + d * float(length_m)
    out["stroke"] = yield from _goto(ctx, rt, arm, end, rotation, frame=frame, gripper=0.0)
    out["retract"] = yield from _goto(ctx, rt, arm, _working_point(ctx, rt, arm, frame) + [0, 0, hover_m], rotation,
                                      frame=frame, gripper=0.0)
    out["status"] = "stroked" if _status(out["stroke"]) in OK_OR_CONTACT else "stroke_failed"
    return out


# -- registry ------------------------------------------------------------------------------------

def builtin_skills() -> list[SkillSpec]:
    return [
        SkillSpec("pick_and_hold", "Select and execute a grasp, then lift while maintaining wrist orientation.",
                  {"arm": {**ARM, "enum": ["left", "right", "auto"]},
                   "target": {"description": "object reference, grasp reference from grasp_candidates, or [x,y,z] grasp point"},
                   "yaw_deg": {"type": "number"}, "pitch_deg": {"type": "number"},
                   "hover_m": {"type": "number"}, "lift_m": {"type": "number"}, "depth_m": {"type": "number"},
                   "close_ticks": {"type": "integer"}},
                  ["target"], pick_and_hold, arm_binding, [arm_is_free, target_fresh],
                  outputs="status lifted|hover_failed|descend_failed|lift_failed; per-stage receipts",
                  termination="returns after the lift or at the first failed stage (gripper state is reported)"),
        SkillSpec("stage_near", "Approach a target with the held working point, measure the reached pose, and create a contact reference.",
                  {"arm": ARM, "target": POINT, "standoff_m": {"type": "number"},
                   "approach": {"type": "array", "items": {"type": "number"}}, "frame": {"enum": ["grasp", "tip"]},
                   "label": {"type": "string"}},
                  ["arm", "target"], stage_near, arm_binding, [arm_is_free, target_fresh],
                  outputs="contact_ref (point reference of the reached pose) and residual_mm",
                  termination="after one precise approach; stops early on contact"),
        SkillSpec("lift_upright", "Lift a held object to provide clearance, then reorient its measured axis upright.",
                  {"arm": ARM, "clearance_m": {"type": "number"}, "upright": {"type": "array", "items": {"type": "number"}}},
                  ["arm"], lift_upright, arm_binding, [arm_is_free, gripper_closed],
                  outputs="axis_error_deg before reorientation", termination="after lift and one reorientation"),
        SkillSpec("transport_held", "Lift, translate laterally, and lower while maintaining orientation and the grasp.",
                  {"arm": ARM, "target": POINT, "clearance_m": {"type": "number"}, "lower": {"type": "boolean"},
                   "yaw_deg": {"type": "number"}, "frame": {"enum": ["grasp", "tip"]}},
                  ["arm", "target"], transport_held, arm_binding, [arm_is_free, gripper_closed, target_fresh],
                  outputs="status transported|raise_failed|carry_failed|lower_failed",
                  termination="after lowering (stops on contact) or at the first failed stage"),
        SkillSpec("place_for_regrasp", "Reach a placement pose, open the gripper subject to release checks, and retreat.",
                  {"arm": ARM, "target": POINT, "yaw_deg": {"type": "number"}, "clearance_m": {"type": "number"},
                   "press_m": {"type": "number"}, "retreat_m": {"type": "number"}},
                  ["arm", "target"], place_for_regrasp, arm_binding, [arm_is_free, gripper_closed, target_fresh],
                  outputs="status released|release_refused", termination="releases only after the placement pose is reached or touched"),
        SkillSpec("seat_probe", "Align a held working point and perform bounded axial probing while maintaining the grasp.",
                  {"arm": ARM, "target": POINT, "axis": {"type": "array", "items": {"type": "number"}},
                   "depth_m": {"type": "number"}, "standoff_m": {"type": "number"}, "step_m": {"type": "number"},
                   "frame": {"enum": ["grasp", "tip"]}},
                  ["arm", "target"], seat_probe, arm_binding, [arm_is_free, gripper_closed],
                  outputs="travelled_m and stopped_by contact|depth_limit", termination="after the probe stops"),
        SkillSpec("press_to_contact", "Approach from a hovering pose, press toward a surface, and retract.",
                  {"arm": ARM, "target": POINT, "depth_m": {"type": "number"}, "hover_m": {"type": "number"},
                   "gripper": {"type": "number"}, "yaw_deg": {"type": "number"}},
                  ["arm", "target"], press_to_contact, arm_binding, [arm_is_free],
                  outputs="status pressed|press_failed and whether contact stopped the press",
                  termination="always retracts to the hover height"),
        SkillSpec("knob_cycle", "Close the gripper, rotate, open, and reset; optionally withdraw before resetting.",
                  {"arm": ARM, "cycles": {"type": "integer"}, "degrees": {"type": "number"},
                   "withdraw_m": {"type": "number"}, "descend_per_cycle_m": {"type": "number"},
                   "close_ticks": {"type": "integer"}},
                  ["arm"], knob_cycle, arm_binding, [arm_is_free],
                  outputs="per-cycle stage statuses and grasp height", termination="after the requested cycles or a failed turn",
                  local_tool=True),
        SkillSpec("tool_stroke", "Position a held tool's working point, execute a bounded stroke, and retract.",
                  {"arm": ARM, "start": POINT, "direction": {"type": "array", "items": {"type": "number"}},
                   "length_m": {"type": "number"}, "depth_m": {"type": "number"}, "hover_m": {"type": "number"},
                   "frame": {"enum": ["grasp", "tip"]}},
                  ["arm", "start", "direction", "length_m"], tool_stroke, arm_binding, [arm_is_free, gripper_closed],
                  outputs="status stroked|stroke_failed", termination="always retracts"),
    ]


PREPARE = {"pick_and_hold": _resolve_grasp}
