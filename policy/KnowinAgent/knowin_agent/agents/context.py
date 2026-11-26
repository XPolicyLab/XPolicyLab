"""Per-decision context: h_t = instruction, current images and measurements, shared notes and
references, unseen events, and (for Local) the interaction contract.

Images are attached only to the newest observation message; older ones keep their
text, so the history stays readable without re-sending pixels (history sanitising as in
KRA, our team's in-house ReAct robot-agent framework). Tool results never carry images in
the chat protocol, so images a tool acquired (e.g. observe) are attached to the next
observation message.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from knowin_agent.agents.llm import image_part
from knowin_agent.runtime.api import RobotRuntime

DEFAULT_CAMERAS = ("head", "left_wrist", "right_wrist")


def load_prompt(name: str) -> str:
    return resources.files("knowin_agent.agents").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


class ContextBuilder:
    def __init__(self, runtime: RobotRuntime, library, *, task_brief: str = "", image_side: int = 448,
                 cameras=DEFAULT_CAMERAS, recorder=None, task_sections=(), local_sections=()):
        self.rt = runtime
        self.library = library
        self.task_brief = task_brief.strip()
        self.image_side = image_side
        self.cameras = cameras
        self.recorder = recorder
        self.task_sections = list(task_sections)
        self.local_sections = list(local_sections)

    def system_prompt(self, role: str, contract=None) -> str:
        parts = [load_prompt(role)]
        notes = self.rt.profile.notes.strip()
        if notes:
            parts.append("## Robot\n" + notes)
        sections = self.task_sections if role == "task" else self.local_sections
        parts += [text if text.lstrip().startswith("#") else f"## {title}\n{text}" for title, text in sections if text]
        if role == "task":
            if self.task_brief:
                parts.append("## Task brief\n" + self.task_brief)
        else:
            parts.append("## Your contract\n" + json.dumps(contract.describe(), ensure_ascii=False, indent=1))
        return "\n\n".join(parts)

    def observation(self, role: str, consumer: str, *, decision: int, contract=None, extra_images=(),
                    note: str = "") -> dict:
        rt = self.rt
        obs = rt.executor.refresh()
        budget = rt.budget()
        lines = [f"[Decision {decision}] tick {obs.tick} | action budget: {budget['ticks_used']} used"
                 + (f" of {budget['ticks_limit']} ({budget['ticks_remaining']} left)" if budget["ticks_limit"] else "")]
        if note:
            lines.append(note)
        if role == "task":
            lines.append("[Instruction] " + (obs.instruction or rt.ctx.instruction))
        else:
            lines.append(f"[Contract goal] {contract.goal} | completion: {contract.completion} | "
                         f"roles: {contract.arm_roles} | round {decision} of {contract.max_rounds}")
        lines.append("[Robot] " + json.dumps(rt.arm_status(), ensure_ascii=False))
        lanes = rt.executor.lanes()
        if any(lanes[k] for k in ("left", "right", "both")) or lanes["recently_finished"]:
            lines.append("[Tracks] " + json.dumps(lanes, ensure_ascii=False))
        events = rt.bus.unseen(consumer)
        if events:
            lines.append("[Events since your last decision]")
            lines += [f"- tick {e.tick}: {e.summary}" for e in events[-12:]]
        lines.append("[Shared notes]\n" + rt.ctx.notes.render())
        refs = rt.ctx.references.summaries(obs.tick)
        if refs:
            lines.append("[References] " + json.dumps(refs, ensure_ascii=False))
        content = [{"type": "text", "text": "\n".join(lines)}]
        for name in self.cameras:
            if name in obs.cameras:
                rgb = obs.cameras[name].rgb
                content.append({"type": "text", "text": f"[Image] {name} (current; {self.size_note(rgb)})"})
                content.append(image_part(rgb, max_side=self.image_side))
                if self.recorder is not None:
                    self.recorder.save_image(f"{obs.obs_id}_{name}", rgb)
        for label, rgb in extra_images:
            content.append({"type": "text", "text": f"[Image] {label} ({self.size_note(rgb)})"})
            content.append(image_part(rgb, max_side=self.image_side))
        return {"role": "user", "content": content, "_meta": {"obs_id": obs.obs_id, "tick": obs.tick}}

    def size_note(self, rgb) -> str:
        """Shown versus native size: pixel arguments (locate(pixel=...), hint_px) are native camera pixels."""
        height, width = rgb.shape[:2]
        scale = min(1.0, self.image_side / max(height, width))
        if scale >= 1.0:
            return f"{width}x{height} native pixels"
        return (f"shown {round(width * scale)}x{round(height * scale)} of native {width}x{height}; pixel "
                f"arguments are native: multiply a pixel read off this image by {1 / scale:.3f}")


def strip_images(message: dict) -> dict:
    """Keep only the text of an older observation message."""
    if message.get("role") != "user" or not isinstance(message.get("content"), list):
        return message
    texts = [part["text"] for part in message["content"] if part.get("type") == "text"
             and not part["text"].startswith("[Image]")]
    return {"role": "user", "content": "\n".join(texts) + "\n(images of this decision omitted)",
            "_meta": message.get("_meta", {})}
