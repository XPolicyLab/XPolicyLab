"""Skill contract: declared inputs, applicability guards, arms, and a program body.

A skill body is a generator `body(ctx, rt, **bindings)` composed from motion
programs with `yield from`. It runs as one executor track without any agent
deliberation, may use perception and feedback control, and returns a receipt.
Task success is never claimed by a skill; the receipt reports execution status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

Guard = Callable[[Any, dict], Optional[str]]


@dataclass
class SkillSpec:
    skill_id: str
    summary: str
    inputs: dict
    required: list
    body: Callable
    arms: Callable[[dict], tuple]
    guards: list = field(default_factory=list)
    outputs: str = ""
    termination: str = ""
    source: dict = field(default_factory=lambda: {"origin": "builtin"})
    local_tool: bool = False

    def info(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "summary": self.summary,
            "inputs": self.inputs,
            "required": self.required,
            "outputs": self.outputs,
            "termination": self.termination,
            "guards": [getattr(g, "__doc__", None) or g.__name__ for g in self.guards],
            "source": self.source,
        }

    def schema(self) -> dict:
        return {"type": "object", "properties": self.inputs, "required": self.required}


def arm_binding(bindings: dict) -> tuple:
    return (bindings["arm"],)
