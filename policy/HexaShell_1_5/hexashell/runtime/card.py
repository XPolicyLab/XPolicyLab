"""The system prompt and the task card the model starts from.

The system prompt is ``prompts/<bench>/*.md`` in file-name order (``*_env_icl.md`` only in an
episode with Env_ICL, :mod:`hexashell.robot.env_icl`).  The task card is built from
the first observation: instruction, step budget, the start pose of both arms (the pose the
evaluator calls "origin"), the three start pictures, and - when enabled - the task's score ladder
(``prompts/ladders``) and its skill note (``skills/robodojo/<task>.md``).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..motion.chain import slide
from ..motion.driver import display_angles
from ..state import CAMERAS, Snapshot
from ..world import ARMS, GRASP_REACH

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
SKILLS = ROOT / "skills" / "robodojo"
LADDER_SECTIONS = ("Capability dimension", "Description", "Scoring")


def base_task(task: str | None) -> str | None:
    """``stack_bowls_random`` -> ``stack_bowls``: variants share their base task's notes."""
    if not task:
        return None
    return re.sub(r"_random$", "", str(task))


def system_prompt(bench: str, env_icl: bool = False) -> str:
    folder = PROMPTS / bench
    parts = [p for p in sorted(folder.glob("*.md")) if env_icl or not p.stem.endswith("_env_icl")]
    if not parts:
        raise FileNotFoundError(f"no prompt files in {folder}")
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts) + "\n"


def ladder(task: str | None) -> str | None:
    """The wiki sections of the task's score ladder, or None when there is none."""
    name = base_task(task)
    if not name:
        return None
    folder = PROMPTS / "ladders"
    path = folder / f"{name}.md"
    if not path.exists():
        return None
    data = path.read_bytes()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["files"].get(path.name) != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"{path} does not match prompts/ladders/manifest.json")
    sections: dict[str, list[str]] = {}
    current = None
    for line in data.decode("utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    out = []
    for name_ in LADDER_SECTIONS:
        body = "\n".join(sections.get(name_, [])).strip()
        if body:
            out.append(f"### {name_}\n\n{body}")
    return "\n\n".join(out) or None


def skill(task: str | None) -> str | None:
    name = base_task(task)
    if not name:
        return None
    path = SKILLS / f"{name}.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def task_card(first: Snapshot, *, with_ladder: bool, with_skill: bool) -> tuple[str, dict]:
    """The card's text and what went into it (for the record)."""
    task = first.info.get("task_name")
    lines = ["# Task card", "",
             f"Instruction (from the benchmark; it is authoritative): {first.instruction!r}"]
    if task:
        lines.append(f"Task: {task}")
    if first.step_limit is not None:
        lines.append(f"Step budget: {first.step_limit} steps for the whole episode "
                     f"(used so far: {first.step or 0}).")
    lines.append("Start pose of each arm (the evaluator's \"origin\"; robot.home() returns to it):")
    for arm in ARMS:
        reading = first.arms[arm]
        pose = slide(reading.flange, GRASP_REACH)
        p, y, r = display_angles(pose[:3, :3])
        xyz = ", ".join(f"{v:.3f}" for v in pose[:3, 3])
        lines.append(f"- {arm}: grasp point ({xyz}), pitch {p:.0f} yaw {y:.0f} roll {r:.0f}, "
                     f"gripper {reading.gripper:.2f}")
    shown = [c for c in CAMERAS if c in first.images]
    lines.append(f"Pictures 1-{len(shown)} below: {', '.join(shown)} at the start; they stay in the conversation.")
    used = {"task": task, "ladder": False, "skill": False}
    if with_ladder:
        text = ladder(task)
        if text:
            lines += ["", "## How this task is scored (RoboDojo wiki)", "",
                      "The episode ends by itself when the last row is met. The other rows are the partial "
                      "credit the scene earns if the episode ends earlier.", "", text]
            used["ladder"] = True
    if with_skill:
        text = skill(task)
        if text:
            lines += ["", "## Notes for this task", "", text]
            used["skill"] = True
    return "\n".join(lines) + "\n", used
