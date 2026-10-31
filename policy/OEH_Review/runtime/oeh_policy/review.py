"""Review mode: the GPT-as-Policy hybrid decision contract over OpenWAM's EE chunks.

Every decision the harness infers one fresh OpenWAM chunk (the *candidate*) on the current observation and shows it to
the teacher with the camera views. The teacher answers with one decision that becomes one ``oeh_manipulate`` proposal:

``student``  the first 1..32 candidate rows, unchanged (OpenWAM's native output);
``edit``     1..5 candidate rows with a per-arm position/rotation offset ramped over the prefix, gripper keep/open/closed;
``eef``      1..5 rows toward explicit absolute per-arm targets within 5 cm / 0.35 rad of the measured pose;
``home``     1..32 rows back toward the episode-start poses (both grippers must already be commanded open);
``hold``     one row at the measured pose with the previous gripper commands (the harness's own, while the teacher thinks).

Rows of ``edit``/``eef``/``home`` track their targets with bounded increments per native step, as the paper's damped
least-squares IK did (<= 2 cm / 0.1 rad per step): RoboDojo's official EE path reaches each row's target within one
0.04 s control step, so an unbounded row would move the arm at metres per second. Student rows are never modified.

The teacher-side schema, its validation and the gate-consistency check follow the paper's `schema.py`,
`validation.py` and `gate_assessment.py` (GPT-as-Policy 79f8be5); the limits are its limits unless noted.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np

from .action_space import RoboDojoEE
from .actions import Method, validate_action
from .contracts import plain

ARMS = ("left", "right")
# The arm keys of a review packet, decision and row: RoboDojo's (and RoboTwin's) two arms by default; a single-arm
# embodiment passes its own, ("left",) (oeh_policy.action_space: a single arm fills the left block). Every function
# below that takes `arms` gives RoboDojo's frozen output when it is left at the default. Likewise `gripper`: arms
# without a gripper slot (oeh_policy.action_space.ActionSpace.gripper False, GR1's hands) pass False, and their
# decisions, rows, packets and records have no gripper field; the default is RoboDojo's.
TEACHER_MODES = ("student", "edit", "eef", "home")
INTENT_MODES = TEACHER_MODES + ("hold",)
EXECUTION_STATUS = ("not_started", "progressing", "failed", "uncertain", "recovered")
INTENT_STATUS = ("aligned", "misaligned", "uncertain")
GRIPPER_EDIT = ("keep", "open", "closed")

# The packet reports poses rounded to 0.1 mm and 1e-5 (quaternion), and hold rows may pass between a decision and its
# execution: targets are validated with this slack and clipped onto the exact limit when their rows are built.
TARGET_SLACK_M = 0.002
TARGET_SLACK_RAD = 0.01
# review.guard.hold_grip: a last command at or below this is a grip the arm holds (OpenWAM's closes land at
# 0.29-0.54, oeh_policy.escalation; it carried objects at 0.5-0.6 in both formal rounds).
HOLD_MAX = 0.6

DEFAULT_LIMITS = {
    "student_max_steps": 32,
    "correction_max_steps": 5,
    "home_max_steps": 32,
    "correction_max_translation_m": 0.05,
    "correction_max_rotation_rad": 0.35,
    "row_max_translation_m": 0.02,
    "row_max_rotation_rad": 0.1,
    "home_open_threshold": 0.8,
}
LIMIT_SOURCES = {
    "correction_max_translation_m": "GPT-as-Policy 79f8be5 validation.py / eef_control.md: 0.05 m per arm",
    "correction_max_rotation_rad": "GPT-as-Policy 79f8be5 validation.py / eef_control.md: 0.35 rad per arm",
    "correction_max_steps": "GPT-as-Policy 79f8be5 validation.py: edit and eef 1-5 steps",
    "row_max_translation_m": "GPT-as-Policy 79f8be5 kinematics.py: DLS task error clipped to 0.02 m per step",
    "row_max_rotation_rad": "GPT-as-Policy 79f8be5 kinematics.py: DLS task error clipped to 0.1 rad per step",
    "student_max_steps": "OpenWAM-Alpha-Sim-RoboDojo inference horizon (32), OEH max_chunk_steps",
    "home_open_threshold": "RoboDojo ee67a14 reward manager is_all_gripper_open(open_threshold=0.8)",
}


def max_steps(limits: Mapping) -> int:
    """The kernel schema's step cap: the largest of the three per-mode caps (each mode is checked again exactly)."""
    return max(limits["student_max_steps"], limits["correction_max_steps"], limits["home_max_steps"])


def validate_limits(value: Mapping | None) -> dict:
    limits = dict(DEFAULT_LIMITS)
    for key, item in (value or {}).items():
        if key not in DEFAULT_LIMITS:
            raise ValueError(f"unknown review limit {key}")
        limits[key] = item
    for key in ("student_max_steps", "correction_max_steps", "home_max_steps"):
        if type(limits[key]) is not int or limits[key] < 1:
            raise ValueError(f"positive integer review limit {key} required")
    for key in ("correction_max_translation_m", "correction_max_rotation_rad", "row_max_translation_m",
                "row_max_rotation_rad", "home_open_threshold"):
        if isinstance(limits[key], bool) or not isinstance(limits[key], (int, float)) or not 0 < limits[key] < math.inf:
            raise ValueError(f"finite positive review limit {key} required")
    return limits


# --------------------------------------------------------------------------------------------------------------
# Rotation helpers (wxyz quaternions), numpy only.

def _unit(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or n < 1e-9:
        raise ValueError("quaternion must be finite and non-zero")
    return q / n


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_from_rotvec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def rotvec_between(a, b) -> np.ndarray:
    """The rotation vector r with R(b) = R(r) * R(a), shortest path."""
    a, b = _unit(a), _unit(b)
    d = quat_mul(b, np.array([a[0], -a[1], -a[2], -a[3]]))
    if d[0] < 0:
        d = -d
    w = float(np.clip(d[0], -1.0, 1.0))
    angle = 2 * math.acos(w)
    s = math.sqrt(max(1 - w * w, 0.0))
    if s < 1e-9:
        return np.zeros(3)
    return d[1:] / s * angle


def angle_between(a, b) -> float:
    return float(np.linalg.norm(rotvec_between(a, b)))


def rpy_deg(q) -> list[float]:
    """Extrinsic xyz Euler angles in degrees (scipy 'xyz' convention, as the paper reports)."""
    w, x, y, z = _unit(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [round(math.degrees(v), 1) for v in (roll, pitch, yaw)]


# --------------------------------------------------------------------------------------------------------------
# Kernel capability schema (the vocabulary stage) and the teacher's tool schema.

def _without_gripper(schema: dict, gripper: bool, field: str) -> dict:
    """An arm's schema, without its gripper ``field`` for arms that have no gripper slot."""
    if not gripper:
        schema["required"].remove(field)
        del schema["properties"][field]
    return schema


def _arm_edit_schema(gripper: bool = True) -> dict:
    return _without_gripper({"type": "object", "additionalProperties": False,
                             "required": ["delta_position", "delta_rotation_vector", "gripper"],
                             "properties": {"delta_position": {"type": "array", "items": {"type": "number"}},
                                            "delta_rotation_vector": {"type": "array", "items": {"type": "number"}},
                                            "gripper": {"type": "string", "enum": list(GRIPPER_EDIT)}}},
                            gripper, "gripper")


def _arm_target_schema(gripper: bool = True) -> dict:
    return _without_gripper({"type": "object", "additionalProperties": False,
                             "required": ["position", "quaternion_wxyz", "gripper_closed"],
                             "properties": {"position": {"type": "array", "items": {"type": "number"}},
                                            "quaternion_wxyz": {"type": "array", "items": {"type": "number"}},
                                            "gripper_closed": {"type": "boolean"}}},
                            gripper, "gripper_closed")


def _both(arm_schema: dict, arms: tuple[str, ...] = ARMS) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(arms),
            "properties": {arm: copy.deepcopy(arm_schema) for arm in arms}}


def intent_schema(max_steps: int, arms: tuple[str, ...] = ARMS, gripper: bool = True) -> dict:
    """``oeh_manipulate`` parameters in review mode. Mode-specific fields: edit needs ``edit``, eef needs ``target``,
    the other modes carry neither; the motion plugin refuses any other combination."""
    return {"type": "object", "additionalProperties": False,
            "description": "Review-mode decision: student prefix, bounded EEF edit/target correction, home, or hold.",
            "required": ["observation_id", "candidate_id", "mode", "steps", "reason"],
            "properties": {
                "observation_id": {"type": "string"},
                "candidate_id": {"type": "string"},
                "mode": {"type": "string", "enum": list(INTENT_MODES)},
                "steps": {"type": "integer", "minimum": 1, "maximum": max_steps},
                "reason": {"type": "string"},
                "edit": _both(_arm_edit_schema(gripper), arms),
                "target": _both(_arm_target_schema(gripper), arms)}}


def _assessment_schema() -> dict:
    string = {"type": "string"}
    return {"type": "object", "additionalProperties": False,
            "required": ["task_progress", "current_subgoal", "execution_status", "execution_evidence",
                         "expected_next_intent", "predicted_next_intent", "intent_status", "intent_evidence"],
            "properties": {
                "task_progress": {"type": "object", "additionalProperties": False,
                                  "required": ["verified_completed", "currently_attempting", "remaining"],
                                  "properties": {"verified_completed": {"type": "array", "items": string},
                                                 "currently_attempting": string,
                                                 "remaining": {"type": "array", "items": string}}},
                "current_subgoal": string,
                "execution_status": {"type": "string", "enum": list(EXECUTION_STATUS)},
                "execution_evidence": string, "expected_next_intent": string, "predicted_next_intent": string,
                "intent_status": {"type": "string", "enum": list(INTENT_STATUS)}, "intent_evidence": string}}


def _next_look_schema(max_chunks: int | None = None) -> dict:
    """G13 (review.schedule segment): how many of OpenWAM's chunks pass whole before the next packet."""
    after = {"type": "integer"}
    if max_chunks is not None:
        after.update(minimum=1, maximum=max_chunks)
    return {"type": "object", "additionalProperties": False, "required": ["after_chunks"],
            "properties": {"after_chunks": after}}


def response_schema(arms: tuple[str, ...] = ARMS, gripper: bool = True, schedule: str | None = None) -> dict:
    """The teacher's ``robodojo_execute`` argument (paper schema.py, OpenWAM EE edition). Strict function calling
    needs every field; ``edit`` and ``target`` are therefore always present and ignored by modes that do not use
    them, as in the paper. Under ``schedule`` segment (G13) it also takes ``next_look`` (an object or null); under
    every other schedule, None included, it is the frozen schema byte for byte."""
    vec = {"type": "array", "items": {"type": "number"}}
    edit_arm = _without_gripper({"type": "object", "additionalProperties": False,
                                 "required": ["delta_position", "delta_rotation_vector", "gripper"],
                                 "properties": {"delta_position": vec, "delta_rotation_vector": vec,
                                                "gripper": {"type": "string", "enum": list(GRIPPER_EDIT)}}},
                                gripper, "gripper")
    target_arm = _without_gripper({"type": "object", "additionalProperties": False,
                                   "required": ["position", "quaternion_wxyz", "gripper_closed"],
                                   "properties": {"position": vec, "quaternion_wxyz": vec,
                                                  "gripper_closed": {"type": "boolean"}}},
                                  gripper, "gripper_closed")
    schema = {"type": "object", "additionalProperties": False,
              "required": ["request_id", "mode", "steps", "reason", "edit", "target", "assessment"],
            "properties": {
                "request_id": {"type": "string"},
                "mode": {"type": "string", "enum": list(TEACHER_MODES)},
                "steps": {"type": "integer"},
                "reason": {"type": "string"},
                "edit": {"type": "object", "additionalProperties": False, "required": list(arms),
                         "properties": {arm: copy.deepcopy(edit_arm) for arm in arms}},
                "target": {"type": "object", "additionalProperties": False, "required": list(arms),
                           "properties": {arm: copy.deepcopy(target_arm) for arm in arms}},
                "assessment": _assessment_schema()}}
    if schedule == "segment":
        schema["required"].append("next_look")
        schema["properties"]["next_look"] = {"anyOf": [_next_look_schema(), {"type": "null"}]}
    return schema


VIEW_WORDS = {1: "one RGB view", 2: "two RGB views", 3: "three RGB views", 4: "four RGB views"}


def rgb_views(count: int = 3) -> str:
    """How the first packet's text names its camera views: "three RGB views" for RoboDojo's three (its frozen
    wording), the count of views actually attached on every other benchmark."""
    return VIEW_WORDS.get(count, f"{count} RGB views")


def tool_spec(arms: tuple[str, ...] = ARMS, gripper: bool = True, schedule: str | None = None) -> dict:
    """The teacher's one tool. Its name is the GPT-as-Policy hybrid's on every benchmark: the teacher transports (the
    Rust brain lane among them) and the frozen RoboDojo prompt name it; only its arms (and whether they have a
    gripper) differ by embodiment, and under the segment schedule its argument takes ``next_look`` (G13)."""
    return {"type": "function", "name": "robodojo_execute", "strict": True,
            "description": ("Submit your gate assessment and one decision for the current fresh OpenWAM proposal. "
                            "The harness validates it, OEH admits and executes it, and the result is the next "
                            "observation with the next fresh proposal. Invalid decisions execute nothing and are "
                            "returned to you for correction."),
            "parameters": response_schema(arms, gripper, schedule)}


# --------------------------------------------------------------------------------------------------------------
# Teacher response validation (before any proposal is made).

def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float) if isinstance(value, (list, tuple)) else None
    if array is None or array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a list of {size} finite numbers")
    return array


def validate_assessment(response: Mapping, decision_index: int, *, first: bool | None = None,
                        free: str | None = None) -> str | None:
    """Paper gate_assessment.py: complete structured assessment; not_started only before the first execution; an
    edit/eef/home takeover requires an observed failure or a misaligned next intent. Returns the takeover trigger.
    ``first``: whether nothing has executed yet; by default the first decision (every_chunk). Under on_stall the
    teacher's first packet usually comes after OpenWAM has run alone, so the packet's native step decides.
    ``free``: the trigger recorded for a takeover that needs neither (review.guard.final_home_steps)."""
    a = response.get("assessment")
    if not isinstance(a, Mapping):
        raise ValueError("assessment is required")
    progress = a.get("task_progress")
    if not isinstance(progress, Mapping):
        raise ValueError("assessment.task_progress is required")
    for key in ("verified_completed", "remaining"):
        if not isinstance(progress.get(key), list) or not all(isinstance(v, str) for v in progress[key]):
            raise ValueError(f"assessment.task_progress.{key} must be a list of strings")
    if not isinstance(progress.get("currently_attempting"), str):
        raise ValueError("assessment.task_progress.currently_attempting must be a string")
    for key in ("current_subgoal", "execution_evidence", "expected_next_intent", "predicted_next_intent",
                "intent_evidence"):
        if not isinstance(a.get(key), str) or not a[key].strip():
            raise ValueError(f"assessment.{key} must be a non-empty string")
    if a.get("execution_status") not in EXECUTION_STATUS:
        raise ValueError(f"assessment.execution_status must be one of {list(EXECUTION_STATUS)}")
    if a.get("intent_status") not in INTENT_STATUS:
        raise ValueError(f"assessment.intent_status must be one of {list(INTENT_STATUS)}")
    if (a["execution_status"] == "not_started") != (decision_index == 0 if first is None else first):
        raise ValueError("execution_status not_started is used for the first decision only, and only there")
    if response.get("mode") == "student":
        return None
    failed, misaligned = a["execution_status"] == "failed", a["intent_status"] == "misaligned"
    if not (failed or misaligned):
        if free is not None:
            return free
        raise ValueError(f"a {response.get('mode')} takeover requires execution_status=failed or "
                         "intent_status=misaligned; uncertainty alone is not a takeover reason")
    return "both" if failed and misaligned else ("execution_failed" if failed else "intent_misaligned")


def _arm_list(arms: tuple[str, ...]) -> str:
    """How a validation message names the arms ("left and right", RoboDojo's frozen wording)."""
    return " and ".join(arms)


def validate_teacher_response(response: Any, packet: Mapping, limits: Mapping, *, final_home_steps: int = 0,
                              arms: tuple[str, ...] = ARMS, gripper: bool = True) -> dict:
    """Check a teacher decision against its packet; return the normalized decision or raise ValueError with a message
    the teacher can act on. Nothing is executed on failure. ``final_home_steps`` (review.guard): in that many last
    native steps a ``home`` needs no failed or misaligned assessment."""
    if not isinstance(response, Mapping):
        raise ValueError("the decision must be a JSON object")
    if response.get("request_id") != packet["request_id"]:
        raise ValueError(f"request_id must equal {packet['request_id']!r} (the current proposal)")
    mode = response.get("mode")
    if mode not in TEACHER_MODES:
        raise ValueError(f"mode must be one of {list(TEACHER_MODES)}")
    steps = response.get("steps")
    if type(steps) is not int:
        raise ValueError("steps must be an integer")
    rows = packet["student_proposal"]["rows"]
    if mode in ("student", "edit") and rows == 0:
        raise ValueError(f"the current proposal has no admissible rows, so {mode} is unavailable; use eef or home")
    cap = {"student": min(limits["student_max_steps"], rows), "edit": min(limits["correction_max_steps"], rows),
           "eef": limits["correction_max_steps"], "home": limits["home_max_steps"]}[mode]
    if not 1 <= steps <= cap:
        raise ValueError(f"{mode} takes 1-{cap} steps")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ValueError("reason must be a non-empty English sentence")
    final = mode == "home" and final_home_steps > 0 and packet.get("remaining_steps", math.inf) <= final_home_steps
    trigger = validate_assessment(response, packet["decision"],
                                  first=(not packet["task_started"]) if "task_started" in packet else
                                        (packet["native_step"] == 0 if "takeover" in packet else None),
                                  free="final_return" if final else None)
    decision = {"mode": mode, "steps": steps, "reason": response["reason"].strip()[:1000]}
    if mode == "edit":
        edit = response.get("edit")
        if not isinstance(edit, Mapping) or set(edit) != set(arms):
            raise ValueError(f"edit requires explicit {_arm_list(arms)} arm{'s' if len(arms) > 1 else ''}")
        decision["edit"] = {}
        for arm in arms:
            item = edit[arm]
            if not isinstance(item, Mapping):
                raise ValueError(f"edit.{arm} must be an object")
            dp = _vector(item.get("delta_position"), 3, f"edit.{arm}.delta_position")
            dr = _vector(item.get("delta_rotation_vector"), 3, f"edit.{arm}.delta_rotation_vector")
            if np.linalg.norm(dp) > limits["correction_max_translation_m"] + 1e-9:
                raise ValueError(f"edit.{arm}.delta_position exceeds {limits['correction_max_translation_m']} m")
            if np.linalg.norm(dr) > limits["correction_max_rotation_rad"] + 1e-9:
                raise ValueError(f"edit.{arm}.delta_rotation_vector exceeds {limits['correction_max_rotation_rad']} rad")
            decision["edit"][arm] = {"delta_position": dp.tolist(), "delta_rotation_vector": dr.tolist()}
            if not gripper:
                if "gripper" in item:
                    raise ValueError(f"edit.{arm} has no gripper field: this robot's hands follow OpenWAM's rows")
                continue
            if item.get("gripper") not in GRIPPER_EDIT:
                raise ValueError(f"edit.{arm}.gripper must be keep, open or closed")
            decision["edit"][arm]["gripper"] = item["gripper"]
    elif mode == "eef":
        target = response.get("target")
        if not isinstance(target, Mapping) or set(target) != set(arms):
            raise ValueError(f"eef requires explicit {_arm_list(arms)} target{'s' if len(arms) > 1 else ''}")
        decision["target"] = {}
        for arm in arms:
            item = target[arm]
            if not isinstance(item, Mapping):
                raise ValueError(f"target.{arm} must be an object")
            position = _vector(item.get("position"), 3, f"target.{arm}.position")
            quaternion = _vector(item.get("quaternion_wxyz"), 4, f"target.{arm}.quaternion_wxyz")
            norm = float(np.linalg.norm(quaternion))
            if abs(norm - 1) > 1e-3:
                raise ValueError(f"target.{arm}.quaternion_wxyz must be a unit quaternion (norm {norm:.4f})")
            if not gripper and "gripper_closed" in item:
                raise ValueError(f"target.{arm} has no gripper_closed field: this robot's hands follow OpenWAM's rows")
            if gripper and type(item.get("gripper_closed")) is not bool:
                raise ValueError(f"target.{arm}.gripper_closed must be true or false")
            current = packet["current_eef"][arm]
            distance = float(np.linalg.norm(position - np.asarray(current["position"], dtype=float)))
            if distance > limits["correction_max_translation_m"] + TARGET_SLACK_M:
                raise ValueError(f"target.{arm} is {distance:.3f} m from the measured pose; the limit is "
                                 f"{limits['correction_max_translation_m']} m")
            rotation = angle_between(current["quaternion_wxyz"], quaternion)
            if rotation > limits["correction_max_rotation_rad"] + TARGET_SLACK_RAD:
                raise ValueError(f"target.{arm} rotates {rotation:.3f} rad from the measured pose; the limit is "
                                 f"{limits['correction_max_rotation_rad']} rad")
            decision["target"][arm] = {"position": position.tolist(), "quaternion_wxyz": (quaternion / norm).tolist()}
            if gripper:
                decision["target"][arm]["gripper_closed"] = item["gripper_closed"]
    decision["trigger"] = trigger
    look = response.get("next_look")
    if look is not None:
        # The authority check (G13): next_look is the segment schedule's, whose packets state its bound; under any
        # other schedule it is refused, never ignored.
        offered = (packet.get("takeover") or {}).get("next_look") if isinstance(packet.get("takeover"), Mapping) \
            else None
        if not isinstance(offered, Mapping):
            raise ValueError("next_look is accepted only under the segment schedule; this packet offers none "
                             "(omit it or give null)")
        bound = offered["max_after_chunks"]
        if (not isinstance(look, Mapping) or set(look) != {"after_chunks"} or type(look["after_chunks"]) is not int
                or not 1 <= look["after_chunks"] <= bound):
            raise ValueError(f"next_look must be {{\"after_chunks\": k}} with an integer k in [1, {bound}], or null")
        decision["next_look"] = {"after_chunks": look["after_chunks"]}
    return decision


def intent_from_decision(decision: Mapping, observation_id: str, candidate_id: str) -> dict:
    """The ``oeh_manipulate`` proposal for a validated teacher decision."""
    intent = {"observation_id": observation_id, "candidate_id": candidate_id, "mode": decision["mode"],
              "steps": decision["steps"], "reason": decision["reason"]}
    if decision["mode"] == "edit":
        intent["edit"] = copy.deepcopy(decision["edit"])
    if decision["mode"] == "eef":
        intent["target"] = copy.deepcopy(decision["target"])
    if "next_look" in decision:  # G13: the segment the decision asks for travels with it to the runtime
        intent["next_look"] = copy.deepcopy(decision["next_look"])
    return intent


# --------------------------------------------------------------------------------------------------------------
# Rows.

def _pose(row: Mapping, arm: str) -> np.ndarray:
    return np.asarray(row[arm + "_ee_pose"], dtype=float)


def _grip(row: Mapping, arm: str) -> float:
    return float(np.asarray(row[arm + "_ee_joint_state"], dtype=float).reshape(-1)[0])


def _row(poses: Mapping[str, np.ndarray], grips: Mapping[str, float] | None, arms: tuple[str, ...] = ARMS) -> dict:
    """A review-layout row; ``grips`` None for arms without a gripper slot."""
    row = {}
    for arm in arms:
        pose = np.asarray(poses[arm], dtype=float).copy()
        pose[3:] = _unit(pose[3:])
        row[arm + "_ee_pose"] = pose.astype(np.float32)
        if grips is not None:
            row[arm + "_ee_joint_state"] = np.array([float(np.clip(grips[arm], 0.0, 1.0))], dtype=np.float32)
    return row


def _inside(pose: np.ndarray, bounds) -> tuple[np.ndarray, bool]:
    """The pose with its position clipped into the workspace bounds as the float32 row stores them (rounded inward,
    so the stored row passes validate_action), and whether it moved."""
    bounds = np.asarray(bounds, dtype=float)
    low, high = bounds[:, 0].astype(np.float32), bounds[:, 1].astype(np.float32)
    low = np.where(low < bounds[:, 0], np.nextafter(low, np.float32(np.inf)), low)
    high = np.where(high > bounds[:, 1], np.nextafter(high, np.float32(-np.inf)), high)
    position = np.asarray(pose[:3], dtype=np.float32)
    clipped = np.clip(position, low, high)
    if np.array_equal(clipped, position):
        return pose, False
    result = np.asarray(pose, dtype=float).copy()
    result[:3] = clipped
    return result, True


def _between(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """The pose ``fraction`` of the way from ``a`` to ``b``: linear in position, along the shortest rotation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rotation = quat_mul(quat_from_rotvec(fraction * rotvec_between(a[3:], b[3:])), _unit(a[3:]))
    return np.concatenate([a[:3] + fraction * (b[:3] - a[:3]), _unit(rotation)])


def track(start: Mapping[str, np.ndarray], targets: list[Mapping[str, np.ndarray]], grips: list[Mapping[str, float]],
          max_translation: float, max_rotation: float, arms: tuple[str, ...] = ARMS) -> list[dict]:
    """Commanded rows that follow per-row targets, moving at most max_translation / max_rotation per row from the
    previous commanded pose (the paper's clipped IK tracking, in EE space)."""
    commanded = {arm: np.asarray(start[arm], dtype=float).copy() for arm in arms}
    rows = []
    for target, grip in zip(targets, grips):
        for arm in arms:
            current, goal = commanded[arm], np.asarray(target[arm], dtype=float)
            delta = goal[:3] - current[:3]
            distance = float(np.linalg.norm(delta))
            if distance > max_translation:
                delta = delta * (max_translation / distance)
            rv = rotvec_between(current[3:], goal[3:])
            angle = float(np.linalg.norm(rv))
            if angle > max_rotation:
                rv = rv * (max_rotation / angle)
            commanded[arm] = np.concatenate([current[:3] + delta, _unit(quat_mul(quat_from_rotvec(rv), current[3:]))])
        rows.append(_row(commanded, grip, arms))
    return rows


class ReviewMethod(Method):
    """The OEH method object for review mode: the episode-start poses, the gripper command ledger and the rows of
    every review intent mode, all in the review layout (oeh_policy.action_space).

    The review port: a benchmark runs review mode with this class and its own ``space`` (the native <-> review layout
    conversion), plus the hooks below where its native rows or bounds differ from RoboDojo's (``bounds``,
    ``check_row``, ``student_rows``, ``executed_rows``; docs/benchmark-adapters/README.md). The defaults are
    RoboDojo's, whose native rows and state are the review layout: its rows, packets and records are the ones the
    formal rounds froze."""

    space = RoboDojoEE()

    def __init__(self, config: dict):
        super().__init__(config)
        self.limits = validate_limits((config.get("review") or {}).get("limits"))
        self.clip = None  # the arms whose rows started at the workspace edge, not their measured pose, for the record
        # review.guard.hold_grip / even_eef / home_lift: how a correction's rows are built (oeh_policy.review_guard).
        # Off without the guard block.
        guard = (config.get("review") or {}).get("guard") or {}
        self.hold_grip = bool(guard.get("hold_grip"))
        self.even_eef = bool(guard.get("even_eef"))
        self.home_lift = bool(guard.get("home_lift"))

    @property
    def arms(self) -> tuple[str, ...]:
        return self.space.arms

    @property
    def gripper(self) -> bool:
        """Whether the arms have a gripper (``ActionSpace.gripper``). Without one there is no gripper ledger, and no
        decision, row, packet or record carries a gripper field: the hands are the space's to carry."""
        return self.space.gripper

    # -- the review port's hooks (RoboDojo's defaults) ------------------------------------------------------------

    def seed(self, state):
        """The first observation fixes the episode-start poses and the command ledger (previous-command semantics:
        the observed opening is the last command); later observations never overwrite them."""
        if self.home is None:
            self.home = self.space.review_state(state)
            self.gripper_commands = ({arm: float(self.home[arm + "_ee_joint_state"][0]) for arm in self.arms}
                                     if self.gripper else {})

    def acknowledged(self, action):
        self.gripper_commands.update(self.space.openings(action))

    def bounds(self, arm: str):
        """The arm's frozen workspace box (3x2, review frame) its correction rows start inside."""
        return self.config["action_limits"][arm + "_xyz_bounds"]

    def check_row(self, row: Mapping) -> dict:
        """A correction or hold row in the review layout, checked against the frozen workspace before it leaves."""
        return validate_action(row, self.config["action_limits"])

    def student_rows(self, candidate: list, steps: int) -> list:
        """The first ``steps`` native candidate rows, unchanged (OpenWAM's own output)."""
        return [self.check_row(copy.deepcopy(dict(row))) for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        """An offered decision's native rows in the review layout, for the previous-result record (its gripper
        column); ``state`` is the observation the rows were built on."""
        return rows

    def review_schema(self) -> dict:
        """The ``oeh_manipulate`` schema the motion plugin registers in review mode. Under the segment schedule (G13)
        a decision may carry ``next_look``; under every other schedule the schema is the frozen one."""
        schema = intent_schema(max_steps(self.limits), self.arms, self.gripper)
        review = self.config.get("review") or {}
        if review.get("schedule") == "segment":
            schema["properties"]["next_look"] = _next_look_schema(review["segment"]["max_segment_chunks"])
        return schema

    def describe_executed(self, decision_index, intent: Mapping, rows: list, executed: int, source: str,
                          trigger: str | None, assessment: Mapping | None) -> dict:
        """The previous-result record of an executed decision (``executed_record``); ``rows`` are ``executed_rows``'s.
        The runtime's review state calls it (oeh_policy.runtime.ReviewState)."""
        return executed_record(decision_index, intent, rows, executed, source, trigger, assessment, self.arms,
                               self.gripper)

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        """The packet's robot fields, in packet order: the measured poses and gripper ledger, the episode-start poses,
        each arm's offset from them and the candidate as the teacher reads it. ``state`` is the current observation's
        projected state."""
        measured = self.space.review_state(state)
        view = (proposal_view(self.space.review_rows(candidate, state), measured, self.arms,
                              gripper=self.gripper) if candidate else
                {"rows": 0, "table": [], "columns": [], "diagnostics": {}})
        if truncated:
            view["truncated"] = truncated
        return {"current_eef": eef_view(measured, self.gripper_commands if self.gripper else None, self.arms),
                "episode_start_eef": eef_view(self.home, None, self.arms),
                "home_offset": home_offset(measured, self.home, self.arms),
                "student_proposal": view}

    # -- intents and rows -----------------------------------------------------------------------------------------

    def check_intent(self, params: Mapping):
        allowed = self.review_schema()["properties"]
        required = {"observation_id", "candidate_id", "mode", "steps", "reason"}
        if set(params) - allowed.keys() or required - set(params):
            raise ValueError("unexpected or missing review intent fields")
        mode = params["mode"]
        if mode not in INTENT_MODES:
            raise ValueError("unknown review mode")
        extra = {"edit": {"edit"}, "eef": {"target"}}.get(mode, set())
        look = "next_look" in params and mode != "hold"
        if set(params) - ({"next_look"} if look else set()) != required | extra:
            raise ValueError(f"review mode {mode} takes exactly {sorted(required | extra)}")
        if look:
            bound = allowed["next_look"]["properties"]["after_chunks"]["maximum"]
            item = params["next_look"]
            if (not isinstance(item, Mapping) or set(item) != {"after_chunks"} or type(item["after_chunks"]) is not int
                    or not 1 <= item["after_chunks"] <= bound):
                raise ValueError(f"next_look takes after_chunks in [1, {bound}]")
        cap = {"student": self.limits["student_max_steps"], "edit": self.limits["correction_max_steps"],
               "eef": self.limits["correction_max_steps"], "home": self.limits["home_max_steps"], "hold": 1}[mode]
        if type(params["steps"]) is not int or not 1 <= params["steps"] <= cap:
            raise ValueError(f"{mode} takes 1-{cap} steps")
        for key in ("observation_id", "candidate_id", "reason"):
            if not isinstance(params[key], str) or not 0 < len(params[key]) <= 1000:
                raise ValueError(f"invalid review intent {key}")

    def rows(self, params: Mapping, candidate: list | None, state: Mapping) -> list:
        """The native rows of one admitted review intent. ``candidate`` holds OpenWAM's native rows and ``state`` is
        the current observed robot state. Corrections are built in the review layout and leave through the action
        space; student rows leave unchanged."""
        self.check_intent(params)
        arms, gripper = self.arms, self.gripper
        if self.home is None or len(self.gripper_commands) != (len(arms) if gripper else 0):
            raise ValueError("command ledger has not been initialized")
        measured = self.space.review_state(state)
        start = {arm: measured[arm + "_ee_pose"].astype(float) for arm in arms}
        # Without a gripper there is no gripper ledger, and every row of the harness and the teacher is pose only.
        ledger = {arm: float(self.gripper_commands[arm]) for arm in arms} if gripper else None
        mode, steps = params["mode"], params["steps"]
        limits = self.limits
        # The harness's own rows start inside the frozen workspace: an arm measured outside it (pushed out under
        # student rows, ATTR-6, replication round 2026-10-03) holds at the nearest pose inside, and a correction
        # tracks from there, so neither the harness's hold nor a teacher's in-bounds correction is refused for where
        # the arm already is. Targets are not clipped: one outside the workspace is still refused.
        origin, clip = {}, {}
        for arm in arms:
            origin[arm], moved = _inside(start[arm], self.bounds(arm))
            if moved:
                clip[arm] = {"measured": [round(float(v), 5) for v in start[arm][:3]],
                             "held": [round(float(v), 5) for v in origin[arm][:3]]}
        self.clip = {"mode": mode, "arms": clip} if clip and mode != "student" else None
        if mode == "hold":
            rows = [_row(origin, ledger, arms)]
        elif mode == "student":
            if not candidate or steps > len(candidate):
                raise ValueError("student prefix exceeds the candidate")
            return self.student_rows(candidate, steps)
        elif mode == "edit":
            if not candidate or steps > len(candidate):
                raise ValueError("edit prefix exceeds the candidate")
            proposal = self.space.review_rows(candidate, state)
            targets, grips = [], []
            for index in range(steps):
                alpha = (index + 1) / steps
                target, grip = {}, {}
                for arm in arms:
                    edit = params["edit"][arm]
                    base = _pose(proposal[index], arm)
                    position = base[:3] + alpha * np.asarray(edit["delta_position"], dtype=float)
                    rotation = quat_mul(quat_from_rotvec(alpha * np.asarray(edit["delta_rotation_vector"],
                                                                            dtype=float)), _unit(base[3:]))
                    target[arm] = np.concatenate([position, rotation])
                    if gripper:
                        own = _grip(proposal[index], arm)
                        grip[arm] = {"keep": own, "open": 1.0,
                                     "closed": self._closed(own, ledger[arm])}[edit["gripper"]]
                targets.append(target)
                grips.append(grip if gripper else None)
            rows = track(origin, targets, grips, limits["row_max_translation_m"], limits["row_max_rotation_rad"], arms)
        elif mode == "eef":
            target = {}
            grip = {}
            for arm in arms:
                item = params["target"][arm]
                position = np.asarray(item["position"], dtype=float)
                quaternion = _unit(item["quaternion_wxyz"])
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError("eef target position must be three finite numbers")
                offset = position - start[arm][:3]
                distance = float(np.linalg.norm(offset))
                if distance > limits["correction_max_translation_m"] + TARGET_SLACK_M:
                    raise ValueError("eef target beyond the translation limit from the measured pose")
                if distance > limits["correction_max_translation_m"]:  # rounding or hold drift: onto the limit
                    position = start[arm][:3] + offset * (limits["correction_max_translation_m"] / distance)
                rv = rotvec_between(start[arm][3:], quaternion)
                angle = float(np.linalg.norm(rv))
                if angle > limits["correction_max_rotation_rad"] + TARGET_SLACK_RAD:
                    raise ValueError("eef target beyond the rotation limit from the measured pose")
                if angle > limits["correction_max_rotation_rad"]:
                    rv = rv * (limits["correction_max_rotation_rad"] / angle)
                    quaternion = _unit(quat_mul(quat_from_rotvec(rv), start[arm][3:]))
                target[arm] = np.concatenate([position, quaternion])
                if not gripper:
                    continue
                if type(item["gripper_closed"]) is not bool:
                    raise ValueError("eef gripper_closed must be a boolean")
                grip[arm] = self._closed(None, ledger[arm]) if item["gripper_closed"] else 1.0
            targets = [target] * steps
            if self.even_eef:
                # Evenly spaced targets from where the rows start (the measured pose, or its clip into the bounds):
                # each row moves distance / steps (at most the row limits) instead of the row limit until the target
                # and then nothing.
                targets = [{arm: _between(origin[arm], target[arm], (index + 1) / steps) for arm in arms}
                           for index in range(steps)]
            rows = track(origin, targets, [grip if gripper else None] * steps, limits["row_max_translation_m"],
                         limits["row_max_rotation_rad"], arms)
        elif mode == "home":
            if gripper and any(value < limits["home_open_threshold"] for value in ledger.values()):
                raise ValueError("open both grippers (eef with gripper_closed=false) before returning home")
            target = {arm: self.home[arm + "_ee_pose"].astype(float) for arm in arms}
            targets = [target] * steps
            if self.home_lift:
                # An arm below its start height first rises straight up to it, keeping its orientation.
                raised = {arm: np.concatenate([origin[arm][:2], target[arm][2:3], origin[arm][3:]]) for arm in arms}
                rise = {arm: max(0, math.ceil((target[arm][2] - origin[arm][2]) / limits["row_max_translation_m"]
                                              - 1e-9)) for arm in arms}
                targets = [{arm: raised[arm] if index < rise[arm] else target[arm] for arm in arms}
                           for index in range(steps)]
            rows = track(origin, targets, [ledger] * steps, limits["row_max_translation_m"],
                         limits["row_max_rotation_rad"], arms)
        else:  # pragma: no cover - check_intent refuses it
            raise ValueError("unknown review mode")
        return self.space.native_rows([self.check_row(row) for row in rows], state)

    def _closed(self, own: float | None, ledger: float) -> float:
        """The opening a "closed" command gets: 0.0, or under review.guard.hold_grip, when the arm already grips (its
        last command, the ledger, is HOLD_MAX or less), that grip: an eef keeps it, and an edit may tighten it to
        OpenWAM's own row but never loosens it. Whether the arm grips is read from the ledger, never from the
        proposal, so from an open gripper "closed" still closes fully."""
        if self.hold_grip and round(float(ledger), 3) <= HOLD_MAX:  # the last command as the packet shows it
            return float(ledger if own is None else min(own, ledger))
        return 0.0


# --------------------------------------------------------------------------------------------------------------
# What the teacher sees about robot state and the candidate.

def eef_view(state: Mapping, ledger: Mapping[str, float] | None = None, arms: tuple[str, ...] = ARMS) -> dict:
    view = {}
    for arm in arms:
        pose = np.asarray(state[arm + "_ee_pose"], dtype=float)
        item = {"position": [round(float(v), 4) for v in pose[:3]],
                "quaternion_wxyz": [round(float(v), 5) for v in _unit(pose[3:])],
                "rpy_deg": rpy_deg(pose[3:])}
        if ledger is not None:
            item["gripper_command"] = round(float(ledger[arm]), 3)
        view[arm] = item
    return view


def home_offset(state: Mapping, home: Mapping, arms: tuple[str, ...] = ARMS) -> dict:
    offset = {}
    for arm in arms:
        pose = np.asarray(state[arm + "_ee_pose"], dtype=float)
        start = np.asarray(home[arm + "_ee_pose"], dtype=float)
        offset[arm] = {"distance_m": round(float(np.linalg.norm(pose[:3] - start[:3])), 4),
                       "rotation_deg": round(math.degrees(angle_between(start[3:], pose[3:])), 1)}
    return offset


def proposal_columns(arms: tuple[str, ...] = ARMS, gripper: bool = True) -> list[str]:
    fields = ("x", "y", "z", "roll", "pitch", "yaw", "opening")
    return ["i"] + [f"{arm}_{field}" for arm in arms for field in (fields if gripper else fields[:-1])]


PROPOSAL_COLUMNS = proposal_columns()
PROPOSAL_FRAME = "env-relative world; position m; roll/pitch/yaw deg (extrinsic xyz); opening 1=open 0=closed"


def proposal_view(rows: list[Mapping], state: Mapping, arms: tuple[str, ...] = ARMS,
                  frame: str = PROPOSAL_FRAME, gripper: bool = True) -> dict:
    """The candidate as the teacher reads it: every row's EE position (m), orientation (deg, xyz Euler) and gripper
    opening (1 open, 0 closed; none without a gripper), plus plain arithmetic diagnostics. Not a safety, collision,
    intent or success verdict."""
    table = []
    for index, row in enumerate(rows):
        line = [index]
        for arm in arms:
            pose = _pose(row, arm)
            line += [round(float(v), 4) for v in pose[:3]] + rpy_deg(pose[3:])
            if gripper:
                line.append(round(_grip(row, arm), 3))
        table.append(line)
    measured = {arm: np.asarray(state[arm + "_ee_pose"], dtype=float) for arm in arms}
    diagnostics = {}
    for arm in arms:
        positions = np.asarray([_pose(row, arm)[:3] for row in rows])
        steps = np.linalg.norm(np.diff(np.vstack([measured[arm][:3], positions]), axis=0), axis=1)
        diagnostics[arm] = {
            "net_displacement_m": [round(float(v), 4) for v in positions[-1] - measured[arm][:3]],
            "path_length_m": round(float(steps.sum()), 4),
            "max_step_m": round(float(steps.max()), 4)}
        if not gripper:
            continue
        openings = [_grip(row, arm) for row in rows]
        diagnostics[arm].update({
            "opening_range": [round(min(openings), 3), round(max(openings), 3)],
            "closes_at_row": next((i for i, v in enumerate(openings) if v < 0.5), None),
            "opens_at_row": next((i for i, v in enumerate(openings) if v >= 0.5 and i and openings[i - 1] < 0.5),
                                 None)})
    return {"rows": len(rows), "columns": proposal_columns(arms, gripper), "table": table, "diagnostics": diagnostics,
            "frame": frame}


def executed_record(decision_index: int, intent: Mapping, rows: list[Mapping], executed: int, source: str,
                    trigger: str | None, assessment: Mapping | None, arms: tuple[str, ...] = ARMS,
                    gripper: bool = True) -> dict:
    """The previous-result record the next packet carries (paper history record, trimmed to what matters)."""
    record = {"decision": decision_index, "mode": intent["mode"], "requested_steps": intent["steps"],
              "executed_steps": executed, "source": source, "takeover_trigger": trigger}
    if gripper:
        record["executed_gripper_closed"] = [{arm: bool(_grip(row, arm) < 0.5) for arm in arms}
                                             for row in rows[:executed]]
    record["reason"] = intent.get("reason")
    if assessment is not None:
        record["assessment"] = plain(assessment)
    return record
