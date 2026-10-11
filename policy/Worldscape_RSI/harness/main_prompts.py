"""Prompts and memory of the discrete-command policy.

The RoboDojo counterpart of ``harness/agent/prompts.py`` and
``harness/agent/memory.py``. The benchmark-specific knowledge lives in a *profile* (see
``profiles/robodojo_x5_main.yaml``): coordinate frame conventions, workspace,
camera descriptions, gripper facts and a handful of tips. Nothing in this module
is task specific, and nothing here knows about the simulator: the policy server
renders the prompt from the state dict the agent builds (``main_route.arm_state``)
and the per-command results the simulation client reported.

State schema expected by :func:`state_text`::

    {"steps_used": int, "table_z_cm": float,
     "left":  {"position_cm": [x, y, z],      # fingertip centre, world frame, centimetres
               "approach": [ax, ay, az],       # unit vector the fingers point along
               "finger_axis": [fx, fy, fz],    # unit vector the fingers open/close along
               "opening": float | None,        # measured opening, 0 closed .. 1 open
               "closed_floor": float | None,   # what an empty closed gripper reads
               "held": bool | None},           # opening clearly above the closed floor
     "right": {...same...}}
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _sibling(name: str):
    if __package__:
        return importlib.import_module(f".{name}", __package__)
    spec = importlib.util.spec_from_file_location(f"vlm_agent_{name}", Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAMMAR_HELP = _sibling("commands").GRAMMAR_HELP

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE_NAME = "robodojo_x5_main.yaml"

DEFAULT_PROFILE = {
    # Robot-agnostic placeholders. Every deployment overrides these in its profile YAML.
    "robot": "a robot manipulator with a parallel-jaw gripper (describe arms, reach and gripper opening here).",
    "frame": "World frame in centimetres: state +x/+y/+z directions as seen in the overview image and the table height.",
    "workspace": "Reachable region of the fingertips and any rules for choosing an arm.",
    "cameras": "Describe every image the model receives and any markers drawn on it.",
    "gripper": (
        "The 'gripper position' reported below is the tool centre point between the fingertips. 'approach' is the "
        "unit vector the fingers point along; 'finger axis' is the direction along which the fingers open and close. "
        "Gripper opening is reported 0 (closed) .. 1 (open); after a 'close' command an opening clearly above the "
        "empty-closed reading means an object is held."
    ),
    "tips": [
        "Plan: orient the gripper, move above the target, align x/y using the overview and then the wrist camera, "
        "descend to grasp height, close, lift, transport at a safe height, descend, open.",
        "Each command is executed by a motion planner; if a command reports FAILED the arm did NOT move.",
        "Look at the images every turn and verify the effect of your last commands before continuing.",
    ],
}


def load_profile(path: Optional[str | Path]) -> dict:
    """Read a profile YAML; ``None`` loads the default RoboDojo X5 profile."""
    import yaml

    file = Path(path) if path else PROFILE_DIR / DEFAULT_PROFILE_NAME
    if not file.is_absolute() and not file.is_file():
        file = PROFILE_DIR / file
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"profile {file} must be a mapping")
    for key, value in data.items():
        if key == "tips":
            if not isinstance(value, list) or not all(isinstance(tip, str) for tip in value):
                raise ValueError(f"profile {file}: tips must be a list of strings")
        elif not isinstance(value, str):
            raise ValueError(f"profile {file}: {key} must be a string")
    data["_source"] = str(file)
    return data


_TEMPLATE_RE = re.compile(r"\{([a-z_]+)([+-]\d+(?:\.\d+)?)?\}")


def render_template(text: str, context: Optional[dict]) -> str:
    """Substitute ``{name}`` / ``{name+5}`` / ``{name-2}`` placeholders with numbers from ``context``.

    Placeholders whose name is not in the context are left untouched, so a profile written with
    literal numbers keeps working and a deployment can add its own keys.
    """
    if not context:
        return text

    def sub(match):
        name, delta = match.group(1), match.group(2)
        if name not in context:
            return match.group(0)
        value = float(context[name]) + (float(delta) if delta else 0.0)
        return str(int(value)) if value.is_integer() else f"{value:g}"

    return _TEMPLATE_RE.sub(sub, text)


def profile_text(profile: Optional[dict], context: Optional[dict] = None) -> str:
    merged = {**DEFAULT_PROFILE, **{k: v for k, v in (profile or {}).items() if not k.startswith("_")}}
    rendered = {k: (render_template(v, context) if isinstance(v, str) else [render_template(t, context) for t in v])
                for k, v in merged.items()}
    tips = "\n".join(f"- {tip}" for tip in rendered.get("tips", []))
    return (
        f"ROBOT: {rendered['robot']}\n\n"
        f"COORDINATE FRAME: {rendered['frame']}\n\n"
        f"WORKSPACE: {rendered['workspace']}\n\n"
        f"CAMERAS: {rendered['cameras']}\n\n"
        f"GRIPPER: {rendered['gripper']}\n\n"
        f"TIPS:\n{tips}"
    )


DEMO_NOTE = (
    "One or more DEMONSTRATIONS are shown before your first turn: successful episodes of the same kind of task, recorded "
    "in reference scenes (object positions, colours, lighting and table height may differ). Each contains fingertip "
    "reference states (same frame as CURRENT STATE) and commands; captions identify what any images show and whether "
    "commands were converted from an existing expert recording. Copy its strategy "
    "(order of sub-goals, how it aligned, grasp and place heights relative to the table), NOT its numbers: read "
    "the positions for YOUR scene from your own images and state."
)


def system_prompt(profile: Optional[dict], max_commands_per_turn: int, memory_enabled: bool,
                  context: Optional[dict] = None, extra_note: str = "") -> str:
    memory_field = (
        '  "memory": "<rewrite your running notes: what you have achieved, what you learned (e.g. grasp height that '
        'worked, commands that failed), and what remains; keep it under 120 words>",\n'
        if memory_enabled
        else ""
    )
    return (
        "You are the controller of a robot in a physics simulator. You receive camera images and the robot state, "
        "and you reply with a few discrete commands that are executed in order. Then you get new images.\n\n"
        + profile_text(profile, context)
        + "\n\n"
        + GRAMMAR_HELP
        + "\n" + DEMO_NOTE + "\n"
        + (("\n" + extra_note.strip() + "\n") if extra_note.strip() else "")
        + "\nRESPONSE FORMAT: reply with ONE JSON object and nothing else:\n"
        "{\n"
        '  "scene": "<one or two sentences: where the relevant objects and the grippers are, in cm>",\n'
        '  "progress": "<which sub-goal you are on and whether the last commands had the intended effect>",\n'
        + memory_field
        + '  "plan": "<the next few steps in words>",\n'
        f'  "commands": ["<command>", ...]   // 1 to {max_commands_per_turn} commands, executed in order\n'
        "}\n"
        "Keep every text field short (at most ~40 words each); the whole reply must stay well under 400 words. "
        "The episode ends automatically as soon as the benchmark checker registers success, so as long as you keep "
        "receiving turns the task is NOT complete yet. Only send \"done\" if you are sure nothing more can be done."
    )


def _vec(values: Any, digits: int = 1) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def state_text(state: dict, grasp_facts: Optional[dict] = None) -> str:
    def arm(name: str) -> str:
        a = state[name]
        opening = a.get("opening")
        opening_text = "unknown" if opening is None else f"{float(opening):.2f}"
        floor = a.get("closed_floor")
        floor_text = f" (an empty closed gripper reads {float(floor):.2f})" if floor is not None else ""
        return (
            f"{name.upper()} gripper position (fingertip centre) = {_vec(a['position_cm'])} cm, "
            f"approach = {_vec(a['approach'], 2)}, finger axis = {_vec(a['finger_axis'], 2)}, "
            f"opening = {opening_text}{floor_text}"
        )

    arms = [name for name in ("left", "right") if isinstance(state.get(name), dict)]
    table_z = state.get("table_z_cm")
    lines = [arm(name) for name in arms]
    if table_z is not None:
        lines.append(f"table top at z = {float(table_z):g}")
    lines.append(f"steps used: {state.get('steps_used', '?')}")
    for name, fact in (grasp_facts or {}).items():
        if name not in state:
            continue
        held = state[name].get("held")
        if held is None:
            held_text = "the fingers cannot be measured right now"
        elif held:
            held_text = "still holding it"
        else:
            held_text = "opening is near the empty baseline; retention is uncertain for thin objects, so verify visually"
        tz = f"{float(table_z):g}" if table_z is not None else "the table height"
        lines.append(
            f"{name.upper()} arm closed on an object with the fingertips at z = {fact['grasp_z']} ({held_text}). "
            f"To set it down on a surface at height h, lower the fingertips to z = {fact['grasp_z']} + (h - {tz}) before opening."
        )
    return "\n".join(lines)


def results_text(last_results: list[dict]) -> str:
    lines = []
    for result in last_results:
        tag = "ok" if result.get("ok") else "FAILED"
        note = result.get("note") or ""
        if (os.environ.get("ROBODAWN_RSI2_HARNESS_V1") == "1" or os.environ.get("ROBODAWN_CONVEYOR_V1") == "1") and "steps_cost" in result:
            note += f"; cost {result['steps_cost']} native steps"
        lines.append(f"- {result.get('command', '?')}: {tag}{(' (' + note + ')') if note else ''}")
    return "\n".join(lines)


_RESET_CLAUSE = re.compile(r"\s*,?\s*(?:and\s+|then\s+)?(?:then\s+)?reset\s+the\s+robot\s+arms?\s*\.?", re.I)


def strip_reset_clause(instruction: str) -> str:
    """Drop 'then reset the robot arm' from a task instruction.

    RoboDojo's checkers expect both arms back at their origin at the end: the
    requirement is the harness's job, not the model's. The policy server sends
    both arms home when the model declares done, so the model never reasons
    about it.
    """
    if os.environ.get("ROBODAWN_RSI2_HARNESS_V1") == "1":
        return str(instruction or "").strip()
    text = _RESET_CLAUSE.sub("", str(instruction or "")).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def turn_text(turn: int, instruction: str, state: dict, last_results: list[dict], memory_text: str,
              image_captions: list[str], grasp_facts: Optional[dict] = None) -> str:
    parts = [f"TASK: {instruction}", f"TURN {turn}."]
    if last_results:
        parts.append("RESULT OF YOUR LAST COMMANDS:\n" + results_text(last_results))
    parts.append("CURRENT STATE:\n" + state_text(state, grasp_facts))
    if memory_text:
        parts.append(memory_text)
    if image_captions:
        parts.append("IMAGES ATTACHED (in order): " + "; ".join(image_captions))
    parts.append("Reply with the JSON object.")
    return "\n\n".join(parts)


DONE_RESULT = {
    "command": "done", "kind": "done", "arm": None, "ok": False,
    "note": ("NOT finished: the benchmark checker has not registered success, so the episode continues. "
             "Re-read the instruction; typical reasons: the object is not where/how the task wants it "
             "(wrong spot, not high enough, not far enough to the side, wrong orientation), a gripper "
             "must be opened at the end, or a second object/arm is still required. Keep acting."),
}


# ----------------------------------------------------------------------------- memory
@dataclass
class TurnRecord:
    turn: int
    commands: list[str]
    outcomes: list[str]
    steps_after: Any
    arms: dict = field(default_factory=dict)   # arm name -> {"pos": [x, y, z], "grip": float | None}


@dataclass
class AgentMemory:
    """``history`` is written by the harness, ``scratchpad`` by the model; both switchable for ablations."""

    use_history: bool = True
    use_scratchpad: bool = True
    history_turns: int = 12
    scratchpad: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    planner_failures: list[str] = field(default_factory=list)

    def record_turn(self, turn: int, commands: list[str], results: list[dict], state: dict) -> None:
        outcomes = []
        for result in results:
            tag = "ok" if result.get("ok") else "FAILED"
            note = (result.get("note") or "").strip()
            if note and len(note) > 90:
                note = note[:87] + "..."
            outcomes.append(f"{tag}{(' - ' + note) if note else ''}")
            if result.get("kind") in ("move", "rotate", "point", "home") and not result.get("ok"):
                self.planner_failures.append(str(result.get("command")))
        self.turns.append(TurnRecord(
            turn=turn, commands=list(commands), outcomes=outcomes, steps_after=state.get("steps_used"),
            arms={name: {"pos": state[name]["position_cm"], "grip": state[name].get("opening")}
                  for name in ("left", "right") if isinstance(state.get(name), dict)},
        ))

    def update_scratchpad(self, text: Any) -> None:
        if text is None:
            return
        text = str(text).strip()
        if text:
            self.scratchpad = text[:2500]

    def render(self) -> str:
        parts = []
        if self.use_scratchpad:
            parts.append("YOUR NOTES FROM PREVIOUS TURNS (you wrote these; update them in the `memory` field):\n"
                         + (self.scratchpad if self.scratchpad else "(empty - this is the first turn)"))
        if self.use_history:
            recent = self.turns[-self.history_turns:] if self.history_turns > 0 else []
            if recent:
                lines = []
                if len(self.turns) > len(recent):
                    lines.append(f"... {len(self.turns) - len(recent)} earlier turns omitted ...")
                for record in recent:
                    cmds = "; ".join(f"{c} -> {o}" for c, o in zip(record.commands, record.outcomes))
                    after = ", ".join(
                        f"{name[0].upper()}@{_fmt(arm['pos'])} grip {_grip(arm['grip'])}"
                        for name, arm in record.arms.items())
                    lines.append(f"turn {record.turn}: {cmds} | after: {after}, steps used {record.steps_after}")
                parts.append("HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES:\n" + "\n".join(lines))
            else:
                parts.append("HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES:\n(none yet)")
        return "\n\n".join(parts)

    def to_json(self) -> dict:
        return {"scratchpad": self.scratchpad, "turns": [t.__dict__ for t in self.turns],
                "planner_failures": self.planner_failures}


def _fmt(point: Any) -> str:
    return "(" + ", ".join(f"{float(v):.0f}" for v in point) + ")"


def _grip(value: Any) -> str:
    return "?" if value is None else f"{float(value):.2f}"
