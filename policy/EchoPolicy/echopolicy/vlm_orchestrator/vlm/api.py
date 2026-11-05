# SPDX-License-Identifier: Apache-2.0

"""Minimal VLM API for task decomposition and one-way progress checks."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np
from PIL import Image
import requests

from vlm_orchestrator.vlm.task_descriptions import find_task_reference

logger = logging.getLogger(__name__)

VLM_REQUEST_MAX_ATTEMPTS = 3
VLM_REQUEST_RETRY_DELAYS_S = (0.5, 1.0)

UNORDERED_TASKS = frozenset(
    {
        "stack-bowls",
        "push-t",
        "pack-objects-into-box",
        "fold-clothes",
        "hang-mugs",
        "sweep-blocks",
        "pour-liquid-into-cup",
        "make-toast",
        "store-laptop-and-headphones",
        "stack-blocks",
        "fasten-screws",
        "plug-in-charger",
        "insert-tubes",
        "pour-balls-into-vase",
        "play-xylophone",
        "deposit-coin",
        "insert-key",
        "build-tower",
        "fill-pen-holder",
        "put-bottles-into-dustbin",
        "play-tic-tac-toe",
        "fill-egg-holder",
        "organize-table",
        "make-kong",
        "play-stacking-toy",
    }
)

ATOMIC_ACTIONS = {
    "observe",
    "pick",
    "place",
    "insert",
    "handover",
    "press",
    "turn",
    "throw",
    "hang",
    "fold",
    "push",
    "close",
    "strike",
    "pour",
    "sweep",
    "reorient",
    "lift_up",
}

# Shared semantic context for progress checks. These are action meanings, not
# completion rules: the VLM still decides completion from the current evidence.
ATOMIC_ACTION_DESCRIPTIONS = {
    "observe": "Inspect the requested object, region, or state and report what is visually present.",
    "pick": "Grasp the specified object securely and lift it clear of the ground.",
    "place": "Carry the held object to the specified destination, release it, and leave it stably placed.",
    "insert": "Align the held object with the specified slot or opening and move it fully into the receptacle.",
    "handover": "Transfer the object from one gripper to the other while maintaining control of the object.",
    "press": "Close the gripper, press the designated control device to its lowest position, and then lift it slightly.",
    "turn": "Rotate the specified object or control through the required angle while maintaining contact.",
    "throw": "Release or propel the held object toward the specified disposal or target region.",
    "hang": "Place the held object onto the specified hook, rack, or support so it remains hanging.",
    "fold": "Manipulate the specified deformable item into the requested folded arrangement.",
    "push": "Maintain contact with the specified object and translate it to the requested location or alignment.",
    "close": "Move the specified lid, drawer, cover, or mechanism into its closed state.",
    "strike": "Use the specified tool to make the required contact strike on the target.",
    "pour": "Orient the held container and transfer its contents into the specified destination.",
    "sweep": "Use the specified tool to contact and move the target material into the requested region.",
    "reorient": "Change the held object's orientation to the requested pose while retaining control.",
    "lift_up": "Raise the specified object or end effector vertically to the requested height or state.",
}

_ATOMIC_ACTION_ALIASES = {
    "hand over": "handover",
    "hand-over": "handover",
    "pick up": "pick",
    "put down": "place",
    "lift up": "lift_up",
    "lift-up": "lift_up",
    "liftup": "lift_up",
}
_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_PLACEHOLDER_POINT_RE = re.compile(r"\[\s*x\s*,\s*y\s*\]", re.IGNORECASE)
_ARM_RE = re.compile(r"\b(left|right)\s+arm\b", re.IGNORECASE)


def encode_image_b64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(
        buffer, format="JPEG", quality=90
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_json(text: str) -> dict:
    """Parse a JSON object from a clean response, code fence, or prose."""
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"Cannot parse JSON from: {text[:240]}")


def normalize_point(value) -> list[int] | None:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return [x, y] if 0 <= x <= 255 and 0 <= y <= 255 else None


def normalize_coordinates(value) -> list[list[int]]:
    point = normalize_point(value)
    if point is not None:
        return [point]
    values = value.values() if isinstance(value, Mapping) else value
    if not isinstance(values, (list, tuple)) and not isinstance(value, Mapping):
        return []
    return [point for item in values if (point := normalize_point(item)) is not None]


def normalize_google_coordinates(value) -> list[list[int]]:
    """Accept requested 0..255 points or Gemini's occasional 0..1000 points."""
    values = value.values() if isinstance(value, Mapping) else value
    if not isinstance(values, (list, tuple)) and not isinstance(value, Mapping):
        return []
    is_point = (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple, Mapping))
        and not isinstance(value[1], (list, tuple, Mapping))
    )
    items = [value] if is_point else list(values)
    raw_points = []
    for item in items:
        if isinstance(item, Mapping):
            item = (item.get("x"), item.get("y"))
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            raw_points.append([int(item[0]), int(item[1])])
        except (TypeError, ValueError):
            continue
    if raw_points and all(0 <= value <= 1000 for point in raw_points for value in point):
        if any(value > 255 for point in raw_points for value in point):
            return [
                [round(x * 255 / 1000), round(y * 255 / 1000)]
                for x, y in raw_points
            ]
    return normalize_coordinates(value)


def extract_coordinates_from_text(text: str) -> list[list[int]]:
    return [
        point
        for match in _POINT_RE.finditer(str(text or ""))
        if (point := normalize_point((match.group(1), match.group(2)))) is not None
    ]


def extract_arm(value=None, text: str = "") -> str:
    arm = str(value or "").strip().lower()
    if arm in {"left", "right"}:
        return arm
    match = _ARM_RE.search(str(text or ""))
    return match.group(1).lower() if match else ""


def apply_coordinate_variables(instruction: str, coordinates=None, arm: str = "") -> str:
    text = str(instruction or "")
    points = normalize_coordinates(coordinates)
    if arm in {"left", "right"}:
        text = re.sub(r"\{arm\}", arm, text, flags=re.IGNORECASE)
    if not points:
        return text
    replacements = {"x": str(points[0][0]), "y": str(points[0][1])}
    for index, (x, y) in enumerate(points, start=1):
        replacements[f"x{index}"] = str(x)
        replacements[f"y{index}"] = str(y)

    def replace_named(match: re.Match[str]) -> str:
        return replacements.get(match.group(1).lower(), match.group(0))

    text = re.sub(r"\{(x\d*|y\d*)\}", replace_named, text, flags=re.IGNORECASE)
    unused = iter(points)

    def replace_point(match: re.Match[str]) -> str:
        point = next(unused, None)
        return f"[{point[0]}, {point[1]}]" if point else match.group(0)

    return _PLACEHOLDER_POINT_RE.sub(replace_point, text)


def vla_coordinate_payload(points) -> list[int] | list[list[int]] | None:
    normalized = normalize_coordinates(points)
    if not normalized:
        return None
    return normalized[0] if len(normalized) == 1 else normalized


def normalize_atomic_action(value) -> str:
    action = str(value or "").strip().lower().replace("_", " ")
    action = _ATOMIC_ACTION_ALIASES.get(action, action)
    return action if action in ATOMIC_ACTIONS else ""


def infer_atomic_action(text: str) -> str:
    patterns = (
        ("observe", r"\bobserve\b|\bwatch\b|\binspect\b"),
        ("handover", r"\bhand\s*-?\s*over\b|\bhandover\b"),
        ("reorient", r"\breorient\b|\brotate\b"),
        ("insert", r"\binsert\b|\bplug\b"),
        ("lift_up", r"\blift\s*[-_ ]\s*up\b|\bliftup\b"),
        ("pick", r"\bpick(?:\s+up)?\b|\bgrab\b|\bgrasp\b|\blift\b"),
        ("place", r"\bplace\b|\bput\b|\bset\s+down\b|\bstack\b"),
        ("press", r"\bpress\b"),
        ("turn", r"\bturn\b|\btighten\b"),
        ("throw", r"\bthrow\b"),
        ("hang", r"\bhang\b"),
        ("fold", r"\bfold\b"),
        ("push", r"\bpush\b"),
        ("close", r"\bclose\b|\bshut\b"),
        ("strike", r"\bstrike\b|\bhit\b"),
        ("pour", r"\bpour\b"),
        ("sweep", r"\bsweep\b"),
    )
    matches = []
    for action, pattern in patterns:
        match = re.search(pattern, str(text or ""), re.IGNORECASE)
        if match:
            matches.append((match.start(), action))
    return min(matches)[1] if matches else ""


def requires_object_target_order(instruction: str) -> bool:
    reference = find_task_reference(instruction)
    return reference is None or reference.slug not in UNORDERED_TASKS


@dataclass(frozen=True)
class DecompositionResult:
    subgoals: list[str]
    ordered: bool = True
    raw: str = field(default="", repr=False)
    adapter_ids: list[str] = field(default_factory=list)
    atomic_actions: list[str] = field(default_factory=list)
    coordinates: list[list[list[int]]] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProgressResult:
    decision: str
    reason: str = ""
    raw: str = field(default="", repr=False)
    grasp_secure: bool = False
    object_lifted: bool = False
    memory: str = ""
    before_to_now_summary: str = ""
    decision_reason: str = ""


class VLMClient(Protocol):
    def select_trajectory_group(
        self, instruction, subgoal, groups, image, extra_images=None, *,
        next_subgoal="",
        before_images=None, previous_available=False, memory="",
        last_executed_ee_trajectory=None, selection_only=False,
        task_description="", full_score_condition="",
    ) -> dict: ...

    def decompose(
        self, instruction: str, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
        *, task_description: str = "", full_score_condition: str = "",
    ) -> DecompositionResult: ...

    def assess_progress(
        self, instruction: str, current_subgoal: str, remaining_subgoals: list[str],
        last_advanced_subgoal: str | None, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
        *, before_images: list[np.ndarray] | None = None, memory: str = "",
        last_executed_ee_trajectory=None,
        task_description: str = "", full_score_condition: str = "",
    ) -> ProgressResult: ...

    def replan(
        self, instruction: str, current_subgoal: str, remaining_subgoals: list[str],
        reason: str, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
        *, before_images: list[np.ndarray] | None = None, memory: str = "",
        task_description: str = "", full_score_condition: str = "",
        last_executed_ee_trajectory=None,
    ) -> DecompositionResult: ...


class VLMBackend(ABC):
    def select_trajectory_group(
        self, instruction, subgoal, groups, image, extra_images=None, *,
        next_subgoal="",
        before_images=None, previous_available=False, memory="",
        last_executed_ee_trajectory=None, selection_only=False,
        task_description="", full_score_condition="",
    ):
        raise NotImplementedError("This VLM backend does not support trajectory group selection")

    @abstractmethod
    def decompose(
        self, instruction, image, extra_images=None, *, task_description="",
        full_score_condition="",
    ): ...

    @abstractmethod
    def assess_progress(
        self, instruction, current_subgoal, remaining_subgoals,
        last_advanced_subgoal, image,
        extra_images=None, *, before_images=None, memory="",
        last_executed_ee_trajectory=None,
        task_description="", full_score_condition="",
    ): ...

    @abstractmethod
    def replan(
        self, instruction, current_subgoal, remaining_subgoals, reason, image,
        extra_images=None, *, before_images=None, memory="",
        task_description="", full_score_condition="",
        last_executed_ee_trajectory=None,
    ): ...


DECOMPOSE_PROMPT_SPEC = (
    Path(__file__).resolve().parents[2]
    / "subtask_template"
    / "ROBODOJO_VLM_DECOMPOSE_PROMPT_SPEC.md"
).read_text(encoding="utf-8")
PROGRESS_PROMPT_SPEC = (
    Path(__file__).resolve().parents[2]
    / "subtask_template"
    / "ROBODOJO_VLM_PROGRESS_PROMPT_SPEC.md"
).read_text(encoding="utf-8")
REPLAN_PROMPT_SPEC = (
    Path(__file__).resolve().parents[2]
    / "subtask_template"
    / "ROBODOJO_VLM_REPLAN_PROMPT_SPEC.md"
).read_text(encoding="utf-8")

DECOMPOSE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "subgoals": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "atomic_action": {
                        "type": "STRING",
                        "enum": sorted(ATOMIC_ACTIONS),
                    },
                    "arm": {"type": "STRING", "enum": ["left", "right"]},
                    "coordinates": {
                        "type": "ARRAY",
                        "items": {
                            "type": "ARRAY",
                            "items": {"type": "INTEGER"},
                        },
                    },
                },
                "required": ["text", "atomic_action", "arm"],
            },
        },
    },
    "required": ["subgoals"],
}
PROGRESS_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {
            "type": "STRING",
            "enum": ["stay", "advance", "replan"],
        },
        "reason": {"type": "STRING"},
        "decision_reason": {"type": "STRING"},
        "memory": {"type": "STRING"},
        "before_to_now_summary": {"type": "STRING"},
    },
    "required": [
        "decision", "reason", "decision_reason", "memory", "before_to_now_summary",
    ],
}


def _generate_content_text(response: Mapping) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("VLM generateContent response has no candidates")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        raise ValueError("VLM generateContent candidate has no content parts")
    text = "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, Mapping)
    ).strip()
    if not text:
        raise ValueError("VLM generateContent candidate contains no text")
    return text


class GoogleVLM(VLMBackend):
    def select_trajectory_group(
        self, instruction, subgoal, groups, image, extra_images=None, *,
        next_subgoal="",
        before_images=None, previous_available=False, memory="",
        last_executed_ee_trajectory=None, selection_only=False,
        task_description="", full_score_condition="",
    ):
        before = before_images or []
        atomic_action = next(
            (
                normalize_atomic_action(group.get("atomic_action"))
                for group in groups
                if group.get("atomic_action")
            ),
            infer_atomic_action(subgoal),
        )
        atomic_action_description = ATOMIC_ACTION_DESCRIPTIONS.get(
            atomic_action, "(none)"
        )
        now_views = 1 + len(extra_images or []) - len(groups)
        current_ee = next(
            (group.get("current_ee") for group in groups if group.get("current_ee")),
            None,
        )
        group_summaries = [
            {key: value for key, value in group.items() if key != "current_ee"}
            for group in groups
        ]
        selection_options = (
            "Choose the displayed group whose averaged action best performs the atomic action. "
            "The controller will execute that group's precomputed mean action unchanged. "
            "Always select the best available group, even if none is ideal; do not select an individual "
            "candidate or invent a new trajectory. "
        )
        allowed_group_ids = [group["group_id"] for group in groups]
        task_reference = find_task_reference(instruction)
        task_context = {
            "description": task_description,
            "full_score_condition": full_score_condition,
        } if task_description or full_score_condition else (
            {
                "description": task_reference.description,
                "full_score_condition": task_reference.full_score_condition,
            }
            if task_reference is not None else {}
        )
        progress_instructions = (
            "This is a selection-only retry for a new subgoal using the same observation. "
            "Do not reassess progress, update memory, or request replanning; only select a group. "
            if selection_only else
            "The check memory summarizes verified progress from earlier observations. Use it with the current images "
            "and do not repeat an already completed state change unless the current images contradict that memory. "
            "Use the recent executed EE trajectory window (up to 20 steps, assembled from actual 10-step action chunks) "
            "to understand the motion since the previous observation. When two chunks are present, "
            "the window labels previous_executed_chunk as the 10 steps sent by the preceding inference and "
            "latest_executed_chunk as the 10 steps sent by the most recent inference. "
            "but do not treat commanded motion alone as proof of contact. Assess progress before selecting a group. "
            "Completion decisions (decision, decision_reason, progress_reason, memory, and before_to_now_summary) "
            "MUST be based only on the already executed EE trajectory and gripper states, the BEFORE/NOW images, "
            "and the current atomic action. Candidate group trajectories are future predictions and MUST NOT be "
            "used as evidence that a press, grasp, placement, or any other atomic action has completed. A predicted "
            "future motion or release cannot advance the subgoal. "
            "Return before_to_now_summary as a concise factual comparison grounded jointly in the BEFORE views, "
            "the NOW views, the recent executed EE trajectory window (including gripper state and chunk boundaries), and the current "
            "atomic action. State what changed, what motion was actually executed, and how that evidence relates "
            "to the current atomic action; do not infer the summary from memory alone. Return memory as a "
            "self-contained integration of relevant prior memory and the verified "
            "change. Retain exact requirements, completed and remaining quantities, phase, object states, outcomes, "
            "and relevant failed attempts; never reset or make known progress vague. Use decision=stay when incomplete, "
            "decision=advance when the current subgoal is visibly complete, and decision=replan when the plan is wrong. "
            "Return decision_reason as the explicit evidence-based explanation for the selected decision, separate "
            "from reason, which explains the selected trajectory group/candidate. "
            "Make both reason fields numeric whenever the supplied evidence contains numbers: cite the relevant "
            "candidate/group id, step number or executed-step count, EE XYZ or displacement in meters, "
            "target distance in meters, gripper value, or observed quantity. Include units "
            "and distinguish measured values from visual estimates. Do not use vague explanations such as only "
            "'looks closer' or 'not complete'. If a required comparison has no usable numeric value, explicitly "
            "say 'no numeric evidence available' for that comparison rather than inventing a number. "
            "Do not apply a code-defined special completion rule. If decision is advance or replan, group_id is a "
            "placeholder and the controller will discard that old-subgoal trajectory. "
        )
        replan_instruction = (
            "" if selection_only else
            "Set replan=true when the selected motion should be followed by replanning from the next observation "
            "and explain that choice in the reason. "
        )
        text = (
            f"Overall task: {instruction}\nCurrent subgoal: {subgoal}\n"
            f"Next subgoal after the current one (planning context only): {next_subgoal or '(none; final subgoal)'}\n"
            f"Recent executed EE trajectory (up to 20 steps; each action chunk is normally 10 steps; "
            "previous_executed_chunk=preceding inference, latest_executed_chunk=most recent inference): "
            f"{last_executed_ee_trajectory or '(none)'}\n"
            f"Official task context: {task_context or '(none)'}\n"
            f"Current atomic action: {atomic_action}\n"
            f"Current atomic action description: {atomic_action_description}\n"
            f"Current left/right EE poses: {current_ee}\n"
            f"Accumulated check memory: {memory or '(none)'}\n"
            f"Groups: {group_summaries}\n"
            f"Images: first {len(before)} BEFORE views, then {now_views} NOW views "
            "(main/left/right when available), "
            "then one labeled main-view trajectory overlay per group in group order. "
            f"{progress_instructions}"
            "Judge every displayed candidate group against the CURRENT atomic action. "
            "Choose whichever arm best continues the task from the current scene and motion. Any arm named in the "
            "subgoal is a non-binding planning hint, not a requirement. Compare whether each trajectory acts on "
            "the correct object or target and "
            "makes visually meaningful progress toward completing that atomic action. Merely moving toward or "
            "crossing a target pixel is insufficient. Infer the required motion semantics from the current atomic "
            f"action and scene. When the target is in the central workspace between the two arms and the left and "
            "right candidates are otherwise comparable, prefer the left-arm trajectory; this is only a soft "
            "tie-breaker and must not override clear target alignment, action progress, or safety evidence. "
            f"{selection_options} "
            "Each group is represented by one precomputed mean action. The numeric group data contains the "
            "complete mean left/right EE XYZ trajectory for the whole available action horizon. "
            "The complete EE trajectory is a step-indexed list in world-coordinate meters; use it to inspect "
            "the planned motion and later continuation. The first trajectory_steps entries are the immediate "
            "executable prefix that will be sent now. "
            "The corresponding group image is an overlay of that same mean action over 30 steps. "
            "Do not infer or request any individual member trajectory. "
            "Use both representations for trajectory selection: the first trajectory_steps entries describe the "
            "immediate prefix that will be executed now, while the 30-step mean overlay shows the predicted continuation after that "
            "prefix. When groups have similar 10-step prefixes, use the 30-step continuation to judge whether the "
            "arm remains on a coherent route for the current subgoal and is well positioned to continue into the "
            "next subgoal shown above. Prefer the group whose full continuation preserves access and task progress, "
            "not merely the group with the smallest immediate 10-step difference. The 30-step continuation is "
            "selection context only and is never evidence that an action has already completed. "
            "A group with contains_previous_remaining=true is the standalone unexecuted remainder of the previously "
            "selected trajectory and is labeled PREVIOUS in its overlay. "
            "The complete group mean EE trajectory is in world-coordinate meters. "
            "The executable prefix may show only an initial approach and need not reach the "
            "target within the displayed horizon. Prefer a trajectory that makes clear "
            "directional progress toward the current target over a hold or retreat trajectory. "
            "A 2D crossing is not proof of grasp. "
            "The trajectory-selection reason must be concise but quantitative when possible: cite the selected "
            "group and at least one relevant trajectory step and XYZ (m), trajectory displacement (m), "
            "target distance (m), or executable-prefix length. "
            f"{replan_instruction}"
            "target_distance_m and far_distance_m are world-coordinate distances in meters; use them for near/far context. "
            "Use overlays only to interpret trajectories; the numeric XYZ trajectory includes height. "
            "Consider approach direction, overshoot, both arms and wrist evidence. "
            "After a subgoal changes, judge trajectories only against the new current subgoal. "
            "A trajectory that only retracts from the previous target is insufficient if another "
            "candidate makes clearer progress toward the current target. Do not justify a "
            "previous-target clearing motion as progress toward the new target. "
            "Return every field required by the response schema."
        )
        images = [*before, image, *(extra_images or [])]
        properties = {
            "group_id": {"type": "INTEGER", "description": "the displayed candidate group id"},
            "reason": {"type": "STRING"},
        }
        required = ["group_id", "reason"]
        if not selection_only:
            properties.update({
                "decision": {"type": "STRING", "enum": ["stay", "advance", "replan"]},
                "progress_reason": {"type": "STRING"},
                "decision_reason": {"type": "STRING"},
                "memory": {"type": "STRING"},
                "before_to_now_summary": {"type": "STRING"},
                "replan": {"type": "BOOLEAN", "description":
                    "true when the current scene or selected motion requires VLM replanning after execution"},
            })
            required.extend([
                "decision", "progress_reason", "decision_reason", "memory", "before_to_now_summary",
                "replan",
            ])
        raw = self.generate(
            "Select one provided robot trajectory group. Do not change the task or invent actions.",
            text, images[0], images[1:],
            {"type": "OBJECT", "properties": properties, "required": required},
        )
        data = parse_json(raw)
        group_id = data.get("group_id")
        valid_ids = set(allowed_group_ids)
        if type(group_id) is not int or group_id not in valid_ids:
            raise ValueError(f"VLM returned invalid trajectory group: {group_id!r}")
        result = {
            "group_id": group_id,
            "reason": str(data.get("reason", "")),
            "raw": raw,
        }
        if selection_only:
            return result
        result.update({
            "decision": str(data.get("decision", "stay")).strip().lower(),
            "progress_reason": str(data.get("progress_reason", "")).strip(),
            "decision_reason": str(
                data.get("decision_reason", data.get("progress_reason", ""))
            ).strip(),
            "memory": str(data.get("memory", "")).strip(),
            "before_to_now_summary": str(data.get("before_to_now_summary", "")).strip(),
            "replan": data.get("replan") is True,
        })
        if result["decision"] not in {"stay", "advance", "replan"}:
            result["decision"] = "stay"
        return result

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        system_prompt: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: str | None = None,
        thinking_level: str = "medium",
        timeout: float = 120.0,
        proxy_url: str | None = None,
    ):
        normalized_thinking = thinking_level.strip().lower()
        if normalized_thinking not in {"minimal", "low", "medium", "high"}:
            raise ValueError("thinking_level must be one of: minimal, low, medium, high")
        self.api_key = api_key or os.environ.get("VLM_API_KEY")
        if not self.api_key:
            raise ValueError("VLM_API_KEY or api_key is required for GoogleVLM")
        self.session = requests.Session()
        self.session.trust_env = False
        proxy = proxy_url or os.environ.get("VLM_PROXY_URL")
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.thinking_level = normalized_thinking
        self.timeout = timeout

    @staticmethod
    def _content(
        text: str, image: np.ndarray, extra_images: list[np.ndarray] | None
    ) -> list[dict]:
        content = [{"text": text}]
        for value in [image, *(extra_images or [])]:
            content.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": encode_image_b64(value),
                    }
                }
            )
        return content

    def generate(
        self,
        system: str,
        text: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
        response_schema: dict | None = None,
    ) -> str:
        endpoint = f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": self.system_prompt or system}]},
            "contents": [
                {"role": "user", "parts": self._content(text, image, extra_images)}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": self.thinking_level.upper()},
            },
        }
        if response_schema is not None:
            payload["generationConfig"]["responseSchema"] = response_schema
        last_error: Exception | None = None
        for attempt in range(1, VLM_REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    endpoint,
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, Mapping):
                    raise TypeError("Google generateContent response must be an object")
                return _generate_content_text(data)
            except (requests.RequestException, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == VLM_REQUEST_MAX_ATTEMPTS:
                    break
                delay = VLM_REQUEST_RETRY_DELAYS_S[attempt - 1]
                logger.warning(
                    "VLM request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    VLM_REQUEST_MAX_ATTEMPTS,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Google generateContent failed after {VLM_REQUEST_MAX_ATTEMPTS} attempts"
        ) from last_error

    @staticmethod
    def _parse_subgoals(data: Mapping, fallback: str):
        values = data.get("subgoals")
        if not isinstance(values, list):
            values = []
        subgoals = []
        actions = []
        coordinates = []
        arms = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            text = str(value.get("text") or value.get("subtask_prompt") or "").strip()
            points = normalize_google_coordinates(value.get("coordinates"))
            if not points:
                text_points = [
                    [int(match.group(1)), int(match.group(2))]
                    for match in _POINT_RE.finditer(text)
                ]
                points = normalize_google_coordinates(text_points)
            arm = extract_arm(value.get("arm"), text)
            if text:
                if points:
                    replacements = iter(points)
                    text = _POINT_RE.sub(
                        lambda match: (
                            f"[{point[0]}, {point[1]}]"
                            if (point := next(replacements, None)) is not None
                            else match.group(0)
                        ),
                        text,
                    )
                subgoals.append(apply_coordinate_variables(text, points, arm))
                actions.append(
                    normalize_atomic_action(value.get("atomic_action"))
                    or infer_atomic_action(text)
                )
                coordinates.append(points)
                arms.append(arm)
        if not subgoals:
            subgoals = [fallback]
            actions = [infer_atomic_action(fallback)]
            coordinates = [extract_coordinates_from_text(fallback)]
            arms = [extract_arm(text=fallback)]
        return subgoals, [""] * len(subgoals), actions, coordinates, arms

    def _plan(self, system, user_prompt, instruction, image, extra_images=None):
        raw = self.generate(
            system,
            user_prompt,
            image,
            extra_images,
            DECOMPOSE_RESPONSE_SCHEMA,
        )
        data = parse_json(raw)
        plan = self._parse_subgoals(data, instruction)
        subgoals, adapters, actions, coordinates, arms = plan
        return DecompositionResult(
            subgoals,
            requires_object_target_order(instruction),
            raw,
            adapters,
            actions,
            coordinates,
            arms,
        )

    def decompose(
        self, instruction, image, extra_images=None, *, task_description="",
        full_score_condition="",
    ):
        prompt = f"Task instruction: {json.dumps(str(instruction), ensure_ascii=False)}"
        if task_description:
            prompt += (
                "\nOfficial RoboDojo task description: "
                + json.dumps(str(task_description), ensure_ascii=False)
            )
        if full_score_condition:
            prompt += (
                "\nOfficial RoboDojo full-score condition: "
                + json.dumps(str(full_score_condition), ensure_ascii=False)
            )
        return self._plan(
            DECOMPOSE_PROMPT_SPEC,
            prompt,
            instruction,
            image,
            extra_images,
        )

    def assess_progress(
        self, instruction, current_subgoal, remaining_subgoals,
        last_advanced_subgoal, image,
        extra_images=None, *, before_images=None, memory="",
        last_executed_ee_trajectory=None,
        task_description="", full_score_condition="",
    ):
        text = (
            f'Overall task: "{instruction}"\n'
            f'Current subgoal: "{current_subgoal}"\n'
            f'Remaining subgoals: {remaining_subgoals}\n'
            f'Last advanced subgoal to validate: {last_advanced_subgoal or "(none)"}\n'
            f'Official task description: {task_description or "(none)"}\n'
            f'Official full-score condition: {full_score_condition or "(none)"}\n'
            "Canonical atomic action descriptions (semantic guidance only):\n"
            f"{json.dumps(ATOMIC_ACTION_DESCRIPTIONS, ensure_ascii=False, sort_keys=True)}\n"
            f'Previous check memory: {memory or "(none)"}\n'
            f'Recent executed EE trajectory (up to 20 steps; each action chunk is normally 10 steps; '
            f'previous_executed_chunk=preceding inference, latest_executed_chunk=most recent inference): '
            f'{last_executed_ee_trajectory or "(none)"}\n'
            "This is the progress check. Any advance decision MUST be justified only by the already executed "
            "recent up-to-20-step EE/gripper trajectory window, the BEFORE/NOW images, and the current atomic action. "
            "The window is assembled only from actually dispatched 10-step chunks, with the preceding inference "
            "marked previous_executed_chunk and the most recent inference marked latest_executed_chunk. Do not use any "
            "candidate or predicted trajectory as completion evidence; a predicted Z minimum, endpoint, or future "
            "release is not an executed event. Treat the executed EE trajectory as commanded-motion evidence, "
            "not proof of contact by itself. Return before_to_now_summary using the BEFORE and NOW views, the last executed "
            "recent executed EE trajectory including gripper state and chunk boundaries, and the current atomic action together. It must be a "
            "concise factual comparison of what changed, what motion was actually executed, and how it relates to "
            "the atomic action; do not derive it from memory alone. Then return memory as a self-contained integration "
            "of the previous memory and this new verified "
            "change. Both fields are required; do not merely repeat the old memory. Preserve exact task-relevant "
            "Return decision_reason as the explicit evidence-based explanation for the selected decision "
            "(stay, advance, or replan), separate from the trajectory-selection reason. Preserve exact task-relevant "
            "Make decision_reason and reason quantitative whenever possible: cite concrete values from the supplied "
            "EE trajectory (step number, displacement or endpoint in meters), gripper state/value, target distance, "
            "height/Z change, or observed/completed quantity, with units. State whether each value is measured from "
            "the input or visually estimated. Do not write only 'looks complete', 'not enough', or similar vague "
            "claims. If no numeric value is available for a comparison, explicitly say 'no numeric evidence "
            "available' instead of inventing one. "
            "progress including observed requirements, completed and remaining quantities, current phase, object "
            "identity/state, confirmed outcomes, and relevant failed attempts. Never replace known numeric or ordered "
            "progress with vague wording, reset it, or mark a multi-step requirement complete after only one step."
        )
        if before_images:
            text += (
                f"\nImages: first {len(before_images)} are BEFORE views "
                "(main, left wrist, right wrist when available); the following "
                "views are NOW in the same order. NOW is authoritative."
            )
            images = [*before_images, image, *(extra_images or [])]
            image, extra_images = images[0], images[1:]
        raw = self.generate(
            PROGRESS_PROMPT_SPEC,
            text,
            image,
            extra_images,
            PROGRESS_RESPONSE_SCHEMA,
        )
        data = parse_json(raw)
        decision = str(data.get("decision", "stay")).strip().lower()
        if decision not in {"stay", "advance", "replan"}:
            decision = "stay"
        return ProgressResult(
            decision=decision,
            reason=str(data.get("reason", "")).strip(),
            decision_reason=str(
                data.get("decision_reason", data.get("reason", ""))
            ).strip(),
            raw=raw,
            grasp_secure=data.get("grasp_secure") is True,
            object_lifted=data.get("object_lifted") is True,
            memory=str(data.get("memory", "")).strip(),
            before_to_now_summary=str(data.get("before_to_now_summary", "")).strip(),
        )

    def replan(
        self, instruction, current_subgoal, remaining_subgoals, reason, image,
        extra_images=None, *, before_images=None, memory="",
        task_description="", full_score_condition="",
        last_executed_ee_trajectory=None,
    ):
        prompt = (
            f'Overall task: "{instruction}"\n'
            f'Current plan from this point: {[current_subgoal, *remaining_subgoals]}\n'
            f'Replanning reason: "{reason}"\n'
            f'Accumulated check memory: {memory or "(none)"}\n'
            f'Recent executed EE trajectory (up to 20 steps; each action chunk is normally 10 steps; '
            f'previous_executed_chunk=preceding inference, latest_executed_chunk=most recent inference): '
            f'{last_executed_ee_trajectory or "(none)"}\n'
            "Return a corrected complete plan for all work that remains in the current scene."
        )
        if task_description:
            prompt += (
                "\nOfficial RoboDojo task description: "
                + json.dumps(str(task_description), ensure_ascii=False)
            )
        if full_score_condition:
            prompt += (
                "\nOfficial RoboDojo full-score condition: "
                + json.dumps(str(full_score_condition), ensure_ascii=False)
            )
        if before_images:
            prompt += (
                f"\nImages: first {len(before_images)} are BEFORE views "
                "(main, left wrist, right wrist when available); the following "
                "views are NOW in the same order. Ground targets from NOW only."
            )
            images = [*before_images, image, *(extra_images or [])]
            image, extra_images = images[0], images[1:]
        return self._plan(
            REPLAN_PROMPT_SPEC,
            prompt,
            instruction,
            image,
            extra_images,
        )


class PassthroughVLM(VLMBackend):
    def decompose(
        self, instruction, image, extra_images=None, *, task_description="",
        full_score_condition="",
    ):
        return DecompositionResult([instruction], True, "passthrough")

    def assess_progress(
        self, instruction, current_subgoal, remaining_subgoals,
        last_advanced_subgoal, image,
        extra_images=None, *, before_images=None, memory="",
        last_executed_ee_trajectory=None,
        task_description="", full_score_condition="",
    ):
        return ProgressResult("stay", "passthrough", "passthrough")

    def replan(
        self, instruction, current_subgoal, remaining_subgoals, reason, image,
        extra_images=None, *, before_images=None, memory="",
        task_description="", full_score_condition="",
        last_executed_ee_trajectory=None,
    ):
        return self.decompose(instruction, image, extra_images)


__all__ = [
    "DecompositionResult",
    "GoogleVLM",
    "PassthroughVLM",
    "ProgressResult",
    "VLMBackend",
    "VLMClient",
    "apply_coordinate_variables",
    "encode_image_b64",
    "extract_arm",
    "extract_coordinates_from_text",
    "infer_atomic_action",
    "normalize_atomic_action",
    "normalize_coordinates",
    "normalize_point",
    "parse_json",
    "vla_coordinate_payload",
]
