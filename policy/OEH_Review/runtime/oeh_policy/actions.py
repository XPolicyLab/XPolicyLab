"""Native EE action validation and bounded analytic primitives.

Model actions already went through the official OpenWAM coordinate conversion.
This layer never re-normalizes, reorders or silently clips the fast condition.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .contracts import PHASES, STATE, project_state


# Phase checkpoints shared by the dual- and single-arm methods.
TRANSITIONS = {
    "grasp": {"grasp", "verify_grasp", "home"},
    "verify_grasp": {"verify_grasp", "grasp", "carry", "home"},
    "carry": {"carry", "place", "verify_grasp", "home"},
    "place": {"place", "verify_place", "carry", "home"},
    "verify_place": {"verify_place", "place", "retreat", "home"},
    "retreat": {"retreat", "grasp", "home"}, "home": {"home"},
}


# The fields each mode adds to the common ones. check_intent requires exactly these, and the schema the brain is
# shown states them: the kernel's schema subset (type, properties, required, additionalProperties) cannot express a
# per-mode field set, so it is written into descriptions, which the kernel ignores. On the pod (2026-09-25) the flat
# schema let Codex send `arm` with a vla intent, schema-valid and refused here.
MODE_FIELDS = {"vla": ("subgoal",), "move_relative": ("arm", "delta_xyz"), "gripper": ("arm", "opening"),
               "wait": (), "home": ()}


def mode_rules(mode_fields) -> tuple[str, dict[str, str]]:
    """The schema's description of the per-mode field sets, and each mode-specific field's own description."""
    rule = "; ".join(f"{mode}: {', '.join(fields) or 'no other field'}" for mode, fields in mode_fields.items())
    users: dict[str, list[str]] = {}
    for mode, fields in mode_fields.items():
        for field in fields:
            users.setdefault(field, []).append(mode)
    return (f"The mode decides the fields besides the common ones ({rule}). An intent with a field its mode does "
            "not use, or without one it needs, is refused.",
            {field: f"Only with mode {' or '.join(modes)}." for field, modes in users.items()})


def mode_fields(modes=None, *, pose_control=False) -> dict:
    """MODE_FIELDS restricted to ``modes`` (brain.intent_modes, G4/M3), in MODE_FIELDS order; None: all of them."""
    available = {**MODE_FIELDS, **({"pose": ("targets",)} if pose_control else {})}
    if modes is None:
        return available
    unknown = [mode for mode in modes if mode not in available]
    if unknown or not modes or len(set(modes)) != len(modes):
        raise ValueError(f"brain.intent_modes must list distinct modes of {list(available)}, got {list(modes)}")
    return {mode: fields for mode, fields in available.items() if mode in modes}


def intent_schema(max_steps: int, modes=None, *, pose_control=False) -> dict:
    """The slow brain's ``oeh_manipulate`` schema. ``modes`` (brain.intent_modes): only these modes are in the enum and
    only their fields are properties (the authority check: what the LLM component is shown is what it may issue);
    None gives the frozen schema byte for byte."""
    fields = mode_fields(modes, pose_control=pose_control)
    rule, only = mode_rules(fields)
    properties = {
        "observation_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "mode": {"type": "string", "enum": list(fields)},
        "phase": {"type": "string", "enum": list(PHASES)},
        "target": {"type": "string", "minLength": 1, "maxLength": 128},
        "execute_steps": {"type": "integer", "minimum": 1, "maximum": max_steps},
        "subgoal": {"type": "string", "minLength": 1, "maxLength": 512, "description": only.get("subgoal")},
        "arm": {"type": "string", "enum": ["left", "right"], "description": only.get("arm")},
        "delta_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                      "description": only.get("delta_xyz")},
        "opening": {"type": "number", "minimum": 0, "maximum": 1, "description": only.get("opening")}}
    for field in ("subgoal", "arm", "delta_xyz", "opening"):
        if field not in only:
            del properties[field]
    if "targets" in only:
        from .q1_actions import targets_schema
        properties["targets"] = targets_schema()
    return {"type": "object", "additionalProperties": False, "description": rule,
            "required": ["observation_id", "mode", "phase", "target", "execute_steps"],
            "properties": properties}


# The slow role's brain-row settings (G4, oeh_eval.designs.SLOW_KEYS), as frozen into cfg["brain"]. Each is optional
# here: absent, the runtime's literals (isolated camera turns, one tool step, one attempt) and today's slow-mode
# behaviour hold, so a hand-written slow config keeps its hash and its bundle.
SLOW_ON_NO_CALL = ("fast_chunk", "hold", "stop")
# brain.on_call_failure (M3's fallback; docs/agent-icl-study/acceptance/m3-fallback-analysis.md option A in its small
# form [decision: run session under the user's mandate, chat 2026-10-19]): what a slow turn that ends after a failed
# model call, or with an intent refused by admission, yields. stop (default, today's behaviour): the attempt stops
# (decision_failed). fill_once: the same fill as on_no_call (fast_chunk or hold), once per decision; the failed
# decision counts against max_calls, the fill does not. Question: does an LLM-side failure cost M3 one chunk of
# OpenWAM-α-alone execution, as it costs the review designs, or the episode?
SLOW_ON_CALL_FAILURE = ("fill_once", "stop")


def check_slow_settings(brain: dict) -> dict:
    """Refuse a malformed slow-role setting in cfg["brain"]; returns the settings present."""
    found = {}
    for key in ("pose_control", "execution_feedback", "agent_termination"):
        if key in brain:
            if type(brain[key]) is not bool:
                raise ValueError(f"brain.{key} must be a boolean")
            found[key] = brain[key]
    if brain.get("agent_termination") and brain.get("pose_control") is not True:
        raise ValueError("brain.agent_termination requires brain.pose_control true")
    if "task" in brain:
        if "context" not in brain:
            raise ValueError("brain.task requires brain.context")
        if not isinstance(brain["task"], str) or not 0 < len(brain["task"]) <= 512:
            raise ValueError("brain.task must be a nonempty task identifier of at most 512 characters")
        found["task"] = brain["task"]
    if "context" in brain:
        from .slow_reference import validate_static_context
        found["context"] = validate_static_context(brain["context"], task=brain.get("task"))
    if "notes_store" in brain:
        from .mechanisms import _notes_store
        found["notes_store"] = _notes_store("oeh_slow", brain["notes_store"])
        if brain.get("pose_control") is not True or not isinstance(brain.get("task"), str) or not brain["task"].strip():
            raise ValueError("brain.notes_store requires pose_control true and a public task")
        if brain["notes_store"]["scope"] != "task":
            raise ValueError("brain.notes_store supports only task scope")
    if (brain.get("pose_control") or brain.get("execution_feedback") or "context" in brain) and brain.get("isolated_camera_turns", True):
        raise ValueError("brain.pose_control / execution_feedback / context need brain.isolated_camera_turns false "
                         "so full control and context information reaches the model")
    if "phase_policy" in brain:
        if brain["phase_policy"] not in ("checkpoints", "annotate"):
            raise ValueError("brain.phase_policy must be checkpoints or annotate")
        if brain["phase_policy"] == "annotate" and brain.get("pose_control") is not True:
            raise ValueError("brain.phase_policy annotate needs brain.pose_control true")
        found["phase_policy"] = brain["phase_policy"]
    if "isolated_camera_turns" in brain:
        if type(brain["isolated_camera_turns"]) is not bool:
            raise ValueError("brain.isolated_camera_turns must be a boolean")
        found["isolated_camera_turns"] = brain["isolated_camera_turns"]
    for key, (low, high) in (("max_steps", (1, 8)), ("retry_max_attempts", (1, 5)), ("memory_last_n", (0, 32))):
        if key in brain:
            if type(brain[key]) is not int or not low <= brain[key] <= high:
                raise ValueError(f"brain.{key} must be an integer in [{low}, {high}]")
            found[key] = brain[key]
    for key in ("show_decisions_left", "scan_payload"):
        if key in brain:
            if type(brain[key]) is not bool:
                raise ValueError(f"brain.{key} must be a boolean")
            found[key] = brain[key]
    modes = brain.get("intent_modes")
    if modes is not None:
        if not isinstance(modes, list):
            raise ValueError("brain.intent_modes must be a list")
        found["intent_modes"] = list(mode_fields(modes, pose_control=brain.get("pose_control", False)))
    if "on_no_call" in brain:
        value = brain["on_no_call"]
        if value not in SLOW_ON_NO_CALL:
            raise ValueError(f"brain.on_no_call must be one of {list(SLOW_ON_NO_CALL)}")
        # The fill goes through admission as an intent of the capability, so its mode must be one the schema shows.
        needed = {"fast_chunk": "vla", "hold": "wait"}.get(value)
        if needed is not None and modes is not None and needed not in modes:
            raise ValueError(f"brain.on_no_call {value} is admitted as a {needed} intent, which brain.intent_modes "
                             f"{list(modes)} does not allow")
        found["on_no_call"] = value
    if "on_call_failure" in brain:
        value = brain["on_call_failure"]
        if value not in SLOW_ON_CALL_FAILURE:
            raise ValueError(f"brain.on_call_failure must be one of {list(SLOW_ON_CALL_FAILURE)}")
        if value == "fill_once" and brain.get("on_no_call") not in ("fast_chunk", "hold"):
            raise ValueError("brain.on_call_failure fill_once takes the on_no_call fill: brain.on_no_call must be "
                             "fast_chunk or hold")
        found["on_call_failure"] = value
    # The brain lane's isolated camera turn (crates/oeh-brain/src/native/vision.rs isolate_policy_observation) rebuilds
    # the camera payload from a fixed key list: recent_decisions and decisions_left would be dropped before the model
    # sees them. A design that declares them therefore needs isolated_camera_turns false (the lane then renders the
    # whole observation, the kernel's state snapshot included) until that list carries them; refused here rather than
    # run as a design whose additions never reach the LLM component.
    if brain.get("isolated_camera_turns", True) and (brain.get("memory_last_n") or brain.get("show_decisions_left")):
        raise ValueError("brain.memory_last_n / brain.show_decisions_left need brain.isolated_camera_turns false: the "
                         "isolated camera turn (oeh-brain vision.rs) passes only its fixed payload keys, so "
                         "recent_decisions and decisions_left would never reach the model")
    return found


def validate_action(action: dict, limits: dict) -> dict:
    if not isinstance(action, dict) or set(action) != set(STATE):
        raise ValueError("native EE action requires exactly the four arm fields")
    result = project_state(action)
    for arm in ("left", "right"):
        pose = result[arm + "_ee_pose"]
        bounds = np.asarray(limits[arm + "_xyz_bounds"])
        if np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1]):
            raise ValueError("native action outside frozen workspace bounds")
        if abs(np.linalg.norm(pose[3:]) - 1) > 1e-3:
            raise ValueError("native action needs a unit wxyz quaternion")
        opening = result[arm + "_ee_joint_state"][0]
        if not 0 <= opening <= 1:
            raise ValueError("native gripper action outside [0,1]")
    return result


class Method:
    def __init__(self, config: dict):
        self.config = config
        self.phase = "grasp"
        self.target = "unspecified"
        self.home = None
        self.gripper_commands = {}
        brain = config.get("brain") or {}
        self.pose_control = brain.get("pose_control", False)
        if "pose_control" in brain or "phase_policy" in brain:
            if config.get("mode") != "slow":
                raise ValueError("brain.pose_control / phase_policy are only supported in slow mode")
            check_slow_settings(brain)
        # In hybrid mode every slow intent answers an escalation, itself an observation checkpoint, and the
        # fast path never advances the phase: the brain may name any phase there (the runtime logs it).
        self.any_phase = config.get("mode") == "hybrid" or brain.get("phase_policy") == "annotate"

    def seed(self, state: dict):
        # Only the first observation, with explicit previous-command semantics,
        # initializes the command ledger. Later measurements never overwrite it.
        if self.home is None:
            self.home = project_state(state)
            self.gripper_commands = {arm: float(state[arm + "_ee_joint_state"][0])
                                     for arm in ("left", "right")}

    def acknowledged(self, action: dict):
        for arm in ("left", "right"):
            self.gripper_commands[arm] = float(action[arm + "_ee_joint_state"][0])

    def check_intent(self, params: dict):
        modes = (self.config.get("brain") or {}).get("intent_modes")
        fields = mode_fields(modes, pose_control=self.pose_control)
        allowed = intent_schema(self.config["max_chunk_steps"], modes, pose_control=self.pose_control)["properties"]
        required = {"observation_id", "mode", "phase", "target", "execute_steps"}
        if modes is not None and params.get("mode") in MODE_FIELDS and params["mode"] not in modes:
            # The authority check (G4/M3): a mode the design's vocabulary leaves out is refused by name.
            raise ValueError(f"mode {params['mode']} is not among brain.intent_modes {list(modes)}")
        if set(params) - allowed.keys() or required - params.keys():
            raise ValueError("unexpected or missing intent fields")
        if params["phase"] not in PHASES or params["mode"] not in allowed["mode"]["enum"]:
            raise ValueError("unknown phase or capability mode")
        if type(params["execute_steps"]) is not int or not 1 <= params["execute_steps"] <= self.config["max_chunk_steps"]:
            raise ValueError("invalid execution prefix")
        for key in ("observation_id", "target"):
            if not isinstance(params[key], str) or not 0 < len(params[key]) <= 128:
                raise ValueError("invalid intent identity")
        if set(params) != required | set(fields[params["mode"]]):
            raise ValueError("parameters do not match selected capability")
        if not self.any_phase and params["phase"] not in TRANSITIONS[self.phase]:
            raise ValueError("phase transition skips an observation checkpoint")
        if params["mode"] == "vla" and (not isinstance(params["subgoal"], str) or
                                         not 0 < len(params["subgoal"]) <= 512):
            raise ValueError("invalid VLA subgoal")
        if params["mode"] == "pose":
            from .q1_actions import validate_targets
            validate_targets(params["targets"])

    def primitives(self, params: dict, state: dict) -> list[dict]:
        self.check_intent(params)
        if self.home is None or len(self.gripper_commands) != 2:
            raise ValueError("command ledger has not been initialized")
        current = project_state(state)
        for arm in ("left", "right"):
            current[arm + "_ee_pose"] = current[arm + "_ee_pose"].astype(float)
        for arm in ("left", "right"):
            current[arm + "_ee_joint_state"] = np.array([self.gripper_commands[arm]], dtype=np.float32)
        goal = copy.deepcopy(current)
        mode, count = params["mode"], params["execute_steps"]
        if mode == "pose":
            from .q1_actions import pose_goal, pose_prefix
            limits = self.config["action_limits"]
            if limits.get("frame") != "env_relative_world" or limits.get("quaternion") != "wxyz":
                raise ValueError("pose control requires native env_relative_world and wxyz action limits")
            goal = pose_goal(current, params["targets"])
            # Check the entire requested target even when only a prefix will run.
            validate_action(goal, limits)
            return [validate_action(row, limits) for row in pose_prefix(current, goal, count)]
        if mode == "move_relative":
            arm = params["arm"]
            if arm not in ("left", "right"):
                raise ValueError("invalid arm")
            delta = np.asarray(params["delta_xyz"], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .030000001:
                raise ValueError("relative transport must be finite and at most 3 cm")
            goal[arm + "_ee_pose"][:3] += delta
        elif mode == "gripper":
            arm, opening = params["arm"], params["opening"]
            if arm not in ("left", "right") or isinstance(opening, bool) or not isinstance(opening, (int, float)) or not 0 <= opening <= 1:
                raise ValueError("invalid gripper command")
            goal[arm + "_ee_joint_state"] = np.array([opening], dtype=np.float32)
        elif mode == "home":
            if any(command < .8 for command in self.gripper_commands.values()):
                raise ValueError("open grippers after placement before returning home")
            for arm in ("left", "right"):
                goal[arm + "_ee_pose"] = self.home[arm + "_ee_pose"].copy()
        elif mode != "wait":
            raise ValueError("VLA inference belongs to the model adapter")
        # Cartesian interpolation is a new EE implementation, not the source
        # joint-speed interpolation. Its fixed bounds are part of method identity.
        required_steps = 1
        for arm in ("left", "right"):
            a, b = current[arm + "_ee_pose"], goal[arm + "_ee_pose"]
            angle = 2 * math.acos(float(np.clip(abs(np.dot(a[3:], b[3:])), 0, 1)))
            required_steps = max(required_steps, math.ceil(float(np.linalg.norm(b[:3] - a[:3])) / .005),
                                 math.ceil(angle / .025))
        horizon = max(count, required_steps)
        rows = []
        for step in range(1, count + 1):
            fraction = step / horizon
            row = {}
            for arm in ("left", "right"):
                key = arm + "_ee_pose"
                a, b = current[key].astype(float), goal[key].astype(float)
                if np.dot(a[3:], b[3:]) < 0:
                    b[3:] *= -1
                pose = a + fraction * (b - a)
                pose[3:] /= np.linalg.norm(pose[3:])
                row[key] = pose.astype(np.float32)
                grip = arm + "_ee_joint_state"
                # Speed limits apply to pose; explicit gripper ramps use their
                # requested duration and held channels retain exact commands.
                row[grip] = (current[grip] + step / count * (goal[grip] - current[grip])).astype(np.float32)
            rows.append(validate_action(row, self.config["action_limits"]))
        return rows
