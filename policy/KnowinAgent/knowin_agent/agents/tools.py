"""Agent tool interfaces for task planning and local interaction.

The full robot set is 16 APIs plus remember / run_skill / skill_info / interact /
finish (and cancel_track). Scoring episodes (hybrid) expose only the non-moving
queries plus exec; tools-mode ablation still sees the full set. Local has shift /
rotate / advance / gripper / observe plus return_to_task, and knob_cycle when its
contract allows rotation and gripper actions. Arguments are validated before
anything moves; errors return to the calling agent as tool results so it can
correct itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from knowin_agent.runtime.api import ApiError, RobotRuntime, ToolOutcome
from knowin_agent.runtime.contract import ContractViolation
from knowin_agent.runtime.executor import EpisodeOver
from knowin_agent.skills.library import SkillError, SkillLibrary

ARM = {"type": "string", "enum": ["left", "right"]}
ARM_AUTO = {"type": "string", "enum": ["left", "right", "auto"]}
TARGET = {"description": "reference id such as 'r3', or [x, y, z] in metres (world frame)"}
VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
CAMERA = {"type": "string", "enum": ["head", "left_wrist", "right_wrist"]}
WAIT = {"type": "boolean", "description": "default true. false returns at once with a track id; the result arrives "
                                          "as an event before your next decision (the other arm can act meanwhile)"}
ORIENT = {
    "yaw": {"type": "number", "description": "degrees; sets the finger closing direction (use grasp_candidates' yaw_deg)"},
    "pitch": {"type": "number", "description": "degrees; 90 = gripper pointing straight down (default when yaw is given)"},
    "roll": {"type": "number", "description": "degrees about the tool axis"},
}


class ControlTransfer(Exception):
    """Raised by control tools whose effect is handled by the episode controller."""

    def __init__(self, kind: str, payload: dict):
        super().__init__(kind)
        self.kind = kind
        self.payload = payload


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict
    required: list
    handler: Callable[[dict], ToolOutcome]
    control: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.properties, "required": self.required,
                           "additionalProperties": False}}}


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict
    images: list = field(default_factory=list)

    def text(self, limit: int = 6000) -> str:
        body = json.dumps(self.data, ensure_ascii=False, default=str)
        return body if len(body) <= limit else body[:limit] + "...<truncated>"


def validate(spec: ToolSpec, args: dict) -> Optional[str]:
    unknown = sorted(set(args) - set(spec.properties))
    if unknown:
        return f"unknown argument(s) {unknown}; allowed: {sorted(spec.properties)}"
    missing = [k for k in spec.required if k not in args]
    if missing:
        return f"missing required argument(s) {missing}"
    for key, value in args.items():
        schema = spec.properties[key]
        kind = schema.get("type")
        if "enum" in schema and value not in schema["enum"]:
            return f"{key} must be one of {schema['enum']}"
        if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"{key} must be a number"
        if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{key} must be an integer"
        if kind == "boolean" and not isinstance(value, bool):
            return f"{key} must be true or false"
        if kind == "string" and not isinstance(value, str):
            return f"{key} must be a string"
        if kind == "array" and not isinstance(value, list):
            return f"{key} must be an array"
    return None


def execute(spec: ToolSpec, args: dict) -> ToolResult:
    problem = validate(spec, args)
    if problem:
        return ToolResult(spec.name, False, {"error": problem})
    try:
        outcome = spec.handler(args)
    except ControlTransfer:
        raise
    except EpisodeOver as exc:
        return ToolResult(spec.name, False, {"error": str(exc), "episode_over": True})
    except (ApiError, ContractViolation, SkillError, ValueError, KeyError, TypeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return ToolResult(spec.name, False, {"error": f"{type(exc).__name__}: {message}"})
    if isinstance(outcome, ToolOutcome):
        return ToolResult(spec.name, "error" not in outcome.data, outcome.data, outcome.images)
    return ToolResult(spec.name, True, dict(outcome or {}))


def robot_tools(rt: RobotRuntime) -> dict[str, ToolSpec]:
    """All 16 robot APIs; the role toolsets pick from these."""
    s = ToolSpec
    return {t.name: t for t in [
        s("locate", "Ground objects by text in a camera view; returns object references with centre, top height, "
                    "footprint and grasp heading (world frame). Wrist views measure more precisely than the head view.",
          {"text": {"type": "string"}, "camera": CAMERA, "hint_px": {"type": "array", "items": {"type": "number"},
                                                                       "description": "[u, v] pixel on the object"},
           "max_objects": {"type": "integer"}, "label": {"type": "string"}},
          ["text"], lambda a: rt.locate(**a)),
        s("inspect", "Measure geometry: shape (object box), axis_tip (held/elongated object axis and its tip; pass arm "
                     "when held), support_plane (surface height), opening (rim or hole centre and radius).",
          {"mode": {"type": "string", "enum": ["shape", "axis_tip", "support_plane", "opening"]},
           "target": {"description": "object description or reference id"}, "camera": CAMERA, "arm": ARM,
           "radius_m": {"type": "number"}},
          ["mode"], lambda a: rt.inspect(**a)),
        s("grasp_candidates", "Grasp poses for an object reference and arm (or auto), ranked by reachability and "
                              "joint travel; each becomes a grasp reference usable by move_to or pick_and_hold.",
          {"target": {"type": "string"}, "arm": ARM_AUTO, "pitch": {"type": "number"},
           "max_candidates": {"type": "integer"}, "depth_m": {"type": "number"}},
          ["target"], lambda a: rt.grasp_candidates(**a)),
        s("check_relation", "Measured geometric relation between references: distance, on_top, inside, aligned. "
                            "Reports 'unknown' when evidence is missing. It is not the task evaluator.",
          {"relation": {"type": "string", "enum": ["distance", "on_top", "inside", "aligned"]},
           "a": {"type": "string"}, "b": {"type": "string"}, "tolerance_m": {"type": "number"}},
          ["relation", "a"], lambda a: rt.check_relation(**a)),
        s("reach_query", "Reachability of a pose for one or both arms without moving, with reachable pitch/height "
                         "variants when it is not reachable.",
          {"arm": {"type": "string", "enum": ["left", "right", "both"]}, "target": TARGET, **ORIENT,
           "frame": {"type": "string", "enum": ["grasp", "flange", "tip"]}},
          ["arm", "target"], lambda a: rt.reach_query(**a)),
        s("check_reach", "Reachability of a pose without moving (0 native actions; same as the program API). "
                         "Reports reachable / unknown plus pitch/height variants and joint_travel_deg. "
                         "Of two reachable choices take the one with less joint travel.",
          {"target": TARGET, "arm": {"type": "string", "enum": ["left", "right", "both"]}, **ORIENT,
           "frame": {"type": "string", "enum": ["grasp", "flange", "tip"]}},
          ["target"], lambda a: rt.reach_query(**{**a, "arm": a.get("arm", "both")})),
        s("move_to", "Move an arm so its grasp point (default), flange or measured held tip reaches a pose. "
                     "A grasp reference supplies arm and orientation. Without orientation the current one is kept.",
          {"arm": ARM_AUTO, "target": TARGET, **ORIENT, "frame": {"type": "string", "enum": ["grasp", "flange", "tip"]},
           "offset": {**VEC3, "description": "world offset added to the target, e.g. [0,0,0.06] to hover"},
           "precision": {"type": "boolean", "description": "slow millimetre-accurate final approach"},
           "stop_on_contact": {"type": "boolean"}, "gripper": {"type": "number", "description": "0 closed .. 1 open"},
           "speed": {"type": "string", "enum": ["normal", "slow"]}, "wait": WAIT, "queue": {"type": "boolean"}},
          ["arm", "target"], lambda a: rt.move_to(**a)),
        s("shift", "Correct position in the image plane of a reference camera: right_m / down_m as seen in that "
                   "image, or du_px / dv_px pixels (scaled by depth).",
          {"arm": ARM, "camera": CAMERA, "right_m": {"type": "number"}, "down_m": {"type": "number"},
           "du_px": {"type": "number"}, "dv_px": {"type": "number"}, "depth_m": {"type": "number"}, "wait": WAIT},
          ["arm", "camera"], lambda a: rt.shift(**a)),
        s("rotate", "Rotate an arm about an axis in the world, tool or a camera frame, pivoting at the grasp point, "
                    "flange or measured held tip.",
          {"arm": ARM, "degrees": {"type": "number"}, "axis": {"description": "'x','y','z' (optionally '-z') or a vector"},
           "frame": {"type": "string", "enum": ["world", "tool", "camera"]}, "camera": CAMERA,
           "pivot": {"type": "string", "enum": ["grasp", "flange", "tip"]}, "wait": WAIT},
          ["arm", "degrees"], lambda a: rt.rotate(**a)),
        s("advance", "Move a bounded distance along a direction: 'tool' (approach axis), 'down', 'up', 'camera' "
                     "(optical axis) or a vector. Stops on contact by default.",
          {"arm": ARM, "distance_m": {"type": "number"}, "direction": {"description": "'tool','down','up','camera' or [x,y,z]"},
           "camera": CAMERA, "stop_on_contact": {"type": "boolean"}, "wait": WAIT},
          ["arm", "distance_m"], lambda a: rt.advance(**a)),
        s("gripper", "Open or close a gripper ('open', 'close' or 0..1 with 1 open). Reports measured width when the "
                     "robot provides it.",
          {"arm": ARM, "command": {"description": "'open', 'close' or a number 0..1"}, "settle_ticks": {"type": "integer"},
           "wait": WAIT},
          ["arm", "command"], lambda a: rt.gripper(**a)),
        s("observe", "Move a physical wrist camera to view a target from a direction ('down', 'front', 'back', "
                     "'left_side', 'right_side', 'oblique' or a vector) at a distance. Returns the new view.",
          {"arm": ARM, "target": TARGET, "view": {"description": "named view or [x,y,z] viewing direction"},
           "distance_m": {"type": "number"}, "wait": WAIT},
          ["arm", "target"], lambda a: rt.observe(**a)),
        s("wait", "Advance time while arms hold their commands: a number of ticks or seconds, or until given "
                  "tracks finish.",
          {"ticks": {"type": "integer"}, "seconds": {"type": "number"},
           "tracks": {"type": "array", "items": {"type": "string"}}},
          [], lambda a: rt.wait(**a)),
        s("orient_held", "Bounded controller: rotate so the measured held-object axis points along target_direction "
                         "(needs inspect axis_tip with arm). Pass camera to re-measure between iterations.",
          {"arm": ARM, "target_direction": VEC3, "pivot": {"type": "string", "enum": ["tip", "grasp", "flange"]},
           "tolerance_deg": {"type": "number"}, "max_iterations": {"type": "integer"}, "camera": CAMERA},
          ["arm", "target_direction"], lambda a: rt.orient_held(**a)),
        s("carry_tip_to", "Bounded controller: translate so the measured held working point (tip) reaches a target, "
                          "keeping orientation; approach_offset_m stops short along the held axis.",
          {"arm": ARM, "target": TARGET, "approach_offset_m": {"type": "number"}, "precision": {"type": "boolean"},
           "wait": WAIT},
          ["arm", "target"], lambda a: rt.carry_tip_to(**a)),
        s("center_over", "Bounded controller: reduce the lateral error between the grasp point and an object seen in "
                         "this arm's wrist camera by re-locating and correcting.",
          {"arm": ARM, "target": {"type": "string"}, "camera": CAMERA, "height_m": {"type": "number"},
           "tolerance_mm": {"type": "number"}, "max_iterations": {"type": "integer"}},
          ["arm", "target"], lambda a: rt.center_over(**a)),
        s("advance_until", "Bounded controller: advance in small steps until contact stops the arm or max_distance_m "
                           "is reached; reports travelled distance.",
          {"arm": ARM, "direction": {"description": "'down','tool','camera' or [x,y,z]"}, "max_distance_m": {"type": "number"},
           "step_m": {"type": "number"}, "camera": CAMERA},
          ["arm"], lambda a: rt.advance_until(**a)),
    ]}


TASK_ROBOT_TOOLS = ("locate", "inspect", "grasp_candidates", "check_relation", "reach_query", "move_to", "rotate",
                    "advance", "gripper", "observe", "wait", "orient_held", "carry_tip_to", "center_over",
                    "advance_until")
LOCAL_ROBOT_TOOLS = ("shift", "rotate", "advance", "gripper", "observe")


def task_toolset(rt: RobotRuntime, library: SkillLibrary, *, remember, interact, finish) -> list[ToolSpec]:
    robot = robot_tools(rt)
    tools = [robot[name] for name in TASK_ROBOT_TOOLS]

    def run_skill(a):
        return ToolOutcome(library.run(rt, a["skill_id"], a.get("bindings") or {}, owner="task",
                                       wait=a.get("wait", True)))

    def cancel_track(a):
        track = rt.executor.cancel(a["track"], reason=a.get("reason", "cancelled by task"))
        return ToolOutcome(track.summary())

    tools += [
        ToolSpec("remember", "Write a shared note (targets, arm roles, measured facts) that both agents see. "
                             "Writing an existing key replaces it.",
                 {"key": {"type": "string"}, "text": {"type": "string"}}, ["key", "text"], remember),
        ToolSpec("skill_info", "Input contract, guards and termination of a skill.",
                 {"skill_id": {"type": "string", "enum": library.ids()}}, ["skill_id"],
                 lambda a: ToolOutcome(library.info(a["skill_id"]))),
        ToolSpec("run_skill", "Run a reusable procedure to its endpoint without intermediate decisions; returns an "
                              "execution receipt (not task success). Skills: " +
                              "; ".join(f"{c['skill_id']}: {c['summary']}" for c in library.catalog()),
                 {"skill_id": {"type": "string", "enum": library.ids()}, "bindings": {"type": "object"}, "wait": WAIT},
                 ["skill_id", "bindings"], run_skill),
        ToolSpec("cancel_track", "Cancel a running or queued asynchronous track.",
                 {"track": {"type": "string"}, "reason": {"type": "string"}}, ["track"], cancel_track),
        ToolSpec("interact", "Delegate a bounded local goal to the Local Interaction Agent. The contract fixes the goal, "
                             "arm roles (actor/observer/holder/idle), permitted actions (shift, rotate, advance, gripper, "
                             "observe, knob_cycle), limits and the completion condition. You pause until it returns "
                             "a report.",
                 {"goal": {"type": "string"}, "completion": {"type": "string"},
                  "arm_roles": {"type": "object", "description": "{'left': role, 'right': role}"},
                  "permitted": {"type": "array", "items": {"type": "string"}},
                  "reference_view": {"type": "string"}, "advance_direction": VEC3,
                  "max_step_translation_m": {"type": "number"}, "max_total_translation_m": {"type": "number"},
                  "max_step_rotation_deg": {"type": "number"}, "max_rounds": {"type": "integer"},
                  "release_allowed": {"type": "boolean"}, "notes": {"type": "string"}},
                 ["goal", "completion", "arm_roles"], interact, control=True),
        ToolSpec("finish", "End the episode with your outcome report (what was achieved, what was verified).",
                 {"outcome": {"type": "string", "enum": ["completed", "partial", "failed"]},
                  "summary": {"type": "string"}}, ["outcome", "summary"], finish, control=True),
    ]
    return tools


def local_toolset(rt: RobotRuntime, library: SkillLibrary, contract, *, return_to_task) -> list[ToolSpec]:
    robot = robot_tools(rt)
    tools = [robot[name] for name in LOCAL_ROBOT_TOOLS if name in contract.permitted]
    if "knob_cycle" in contract.permitted:
        def knob(a):
            arm = a["arm"]
            contract.authorize("knob_cycle", arm, rotation_deg=abs(float(a.get("degrees", 90.0))))
            if not contract.release_allowed:
                raise ContractViolation("knob_cycle opens the gripper; the contract must set release_allowed")
            return ToolOutcome(library.run(rt, "knob_cycle", {"arm": arm, "cycles": 1, "degrees": a.get("degrees", 90.0),
                                                             "withdraw_m": 0.0}, owner="local"))
        tools.append(ToolSpec("knob_cycle", "One close-rotate-open-reset cycle on the actor arm.",
                              {"arm": ARM, "degrees": {"type": "number"}}, ["arm"], knob))
    tools.append(ToolSpec("return_to_task", "End this local interaction and report to the Task-Level Agent: whether "
                                            "the completion condition holds, the evidence, and anything that needs "
                                            "task-level reconsideration.",
                          {"status": {"type": "string", "enum": ["completed", "unresolved", "needs_replan"]},
                           "report": {"type": "string"}, "evidence": {"type": "string"}},
                          ["status", "report"], return_to_task, control=True))
    return tools
