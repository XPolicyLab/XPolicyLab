"""The system prompt and the task card the model starts from.

The system prompt is ``prompts/<bench>/*.md`` in file-name order.  The task card is built from the
first observation: the instruction, RoboDojo's official task text (``obs["extra_info"]
["detailed_instruction"]``: Description and public Scoring, verbatim), the step budget, the start pose
of both arms, and the three start pictures.  Under the Agent Track rules of 2026-11-14 the official
text is the only task-specific text in the card; the task's name is not shown.
"""

from __future__ import annotations

from pathlib import Path

from ..motion.chain import slide
from ..motion.driver import display_angles
from ..state import CAMERAS, Snapshot
from ..world import ARMS, GRASP_REACH

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"


def system_prompt(bench: str) -> str:
    folder = PROMPTS / bench
    parts = sorted(folder.glob("*.md"))
    if not parts:
        raise FileNotFoundError(f"no prompt files in {folder}")
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts) + "\n"


def task_card(first: Snapshot) -> tuple[str, dict]:
    """The card's text and what went into it (for the record)."""
    task = first.info.get("task_name")  # for the record only; never shown
    lines = ["# Task card", "",
             f"Instruction (from the benchmark; it is authoritative): {first.instruction!r}"]
    official = first.detailed_instruction
    if official:
        lines += ["", "The benchmark's official description of this task, verbatim:", "", official.strip(), ""]
    if first.step_limit is not None:
        lines.append(f"Step budget: {first.step_limit} steps for the whole episode "
                     f"(used so far: {first.step or 0}).")
    lines.append("Start pose of each arm (its initial pose; robot.home() returns to it):")
    for arm in ARMS:
        reading = first.arms[arm]
        pose = slide(reading.flange, GRASP_REACH)
        p, y, r = display_angles(pose[:3, :3])
        xyz = ", ".join(f"{v:.3f}" for v in pose[:3, 3])
        lines.append(f"- {arm}: grasp point ({xyz}), pitch {p:.0f} yaw {y:.0f} roll {r:.0f}, "
                     f"gripper {reading.gripper:.2f}")
    shown = [c for c in CAMERAS if c in first.images]
    lines.append(f"Pictures 1-{len(shown)} below: {', '.join(shown)} at the start; they stay in the conversation.")
    used = {"task": task, "detailed_instruction": bool(official)}
    return "\n".join(lines) + "\n", used
