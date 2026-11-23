# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ._robot import RobotBase

_HOST_ONLY = frozenset({"ledger", "backend"})
_LINE = 88


def _annotation(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return value.__name__
    return str(value).replace("typing.", "")


def _format_parameters(sig: inspect.Signature) -> list[str]:
    parts: list[str] = []
    seen_kw_only = False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        text = name
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not inspect.Parameter.empty:
            text += f" = {param.default!r}" if annotation else f"={param.default!r}"
        parts.append(text)
    return parts


def _render_callable(owner: str, name: str, fn: Any) -> str:
    sig = inspect.signature(fn)
    params = _format_parameters(sig)
    ret = _annotation(sig.return_annotation)
    suffix = f" -> {ret}" if ret else ""
    head = f"{owner}.{name}({', '.join(params)}){suffix}"
    if len(head) > _LINE:
        joined = ",\n    ".join(params)
        head = f"{owner}.{name}(\n    {joined},\n){suffix}"
    doc = inspect.getdoc(fn) or ""
    return head + "\n" + textwrap.indent(doc, "    ") if doc else head


def _render_property(owner: str, name: str, prop: property) -> str:
    fn = prop.fget
    ret = ""
    if fn is not None:
        ret = _annotation(inspect.signature(fn).return_annotation)
    head = f"{owner}.{name}" + (f" -> {ret}" if ret else "") + "  (read-only attribute)"
    doc = inspect.getdoc(fn) if fn is not None else ""
    return head + ("\n" + textwrap.indent(doc, "    ") if doc else "")


def _render_result_type(cls: type) -> str:
    parts = []
    for item in fields(cls):
        parts.append(f"{item.name}: {_annotation(item.type)}")
    head = f"class {cls.__name__}({', '.join(parts)})"
    if len(head) > _LINE:
        head = f"class {cls.__name__}(\n    " + ",\n    ".join(parts) + ",\n)"
    extra = []
    for name, member in vars(cls).items():
        if isinstance(member, property) and not name.startswith("_"):
            ret = _annotation(inspect.signature(member.fget).return_annotation)
            doc = inspect.getdoc(member.fget) or ""
            extra.append(f"    .{name} -> {ret}: {doc}")
    doc = inspect.getdoc(cls) or ""
    body = textwrap.indent(doc, "    ") if doc else ""
    return "\n".join(part for part in [head, body, *extra] if part)


def _public_members(robot_cls: type) -> dict[str, Any]:
    """Public methods and properties in definition order, base classes first.

    An override keeps the position of the member it overrides, so a robot
    that re-documents ``image`` for its own cameras does not reorder the
    reference.  ``RobotBase``'s configuration attributes never show, and
    neither do the members the robot declares absent (``ABSENT_MEMBERS``).
    """

    absent = frozenset(getattr(robot_cls, "ABSENT_MEMBERS", ()))
    members: dict[str, Any] = {}
    for klass in reversed(robot_cls.__mro__):
        if klass is object:
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") or name in _HOST_ONLY or name in absent:
                continue
            if isinstance(member, property) or inspect.isfunction(member):
                members[name] = member
    return members


def api_reference(robot_cls: type, *, var: str = "robo") -> str:
    """Render the policy-facing API of ``robot_cls`` from its live definition."""

    sections: list[str] = []
    class_doc = inspect.getdoc(robot_cls) or ""
    sections.append(f"class {robot_cls.__name__}\n{textwrap.indent(class_doc, '    ')}")
    hidden = set(vars(RobotBase)) | {"RESULT_TYPES"}
    lineage = [cls for cls in reversed(robot_cls.__mro__) if not issubclass(RobotBase, cls)]
    defined: dict[str, Any] = {}
    for cls in lineage:
        hidden |= {
            name
            for name, annotation in inspect.get_annotations(cls).items()
            if str(annotation).replace("typing.", "").startswith("ClassVar")
        }
        defined.update(vars(cls))
    constants = [
        f"{var}.{name} = {value!r}"
        for name, value in defined.items()
        if name.isupper() and not name.startswith("_") and name not in hidden
    ]
    if constants:
        sections.append("Constants:\n" + "\n".join(f"    {c}" for c in constants))
    members: list[str] = []
    for name, member in _public_members(robot_cls).items():
        if isinstance(member, property):
            members.append(_render_property(var, name, member))
        else:
            members.append(_render_callable(var, name, member))
    sections.append("Methods and attributes of `robo`:\n\n" + "\n\n".join(members))
    helper_types = getattr(robot_cls, "_api_helpers", lambda: {})()
    for name, helper_cls in helper_types.items():
        owner = f"{var}.{name}"
        methods = [
            _render_callable(owner, method, fn)
            for method, fn in _public_members(helper_cls).items()
            if inspect.isfunction(fn)
        ]
        sections.append(
            f"Pre-built functions on `{owner}` (already available):\n"
            + (inspect.getdoc(helper_cls) or "")
            + "\n\n" + "\n\n".join(methods)
        )
    result_types = tuple(getattr(robot_cls, "RESULT_TYPES", ()))
    results = [_render_result_type(cls) for cls in result_types if is_dataclass(cls)]
    sections.append(
        "Result types (frozen dataclasses; access fields as attributes):\n\n" + "\n\n".join(results)
    )
    absent = frozenset(getattr(robot_cls, "ABSENT_MEMBERS", ()))
    reasons = "truncated | timeout | cancelled"
    if "done" not in absent:
        reasons = "terminated | " + reasons
    sections.append(
        "Exceptions:\n"
        "    ToolError(RuntimeError): a call was rejected or a service failed; "
        "catch it if you have a fallback.\n"
        "    ValueError: an argument was invalid (e.g. a planar move over "
        "0.30 m) - nothing was executed.\n"
        "    EpisodeFinished(BaseException): the episode is over "
        f"(reason: {reasons}); return."
    )
    return "\n\n".join(sections)


def knowledge_text(name: str) -> str:
    """Return the packaged operating-knowledge markdown ``knowledge/<name>.md`` of a robot."""

    return resources.files("martingale").joinpath(f"knowledge/{name}.md").read_text(encoding="utf-8")


@dataclass
class TaskCard:
    """What the policy author is told about one evaluation cell.

    Everything here is available to a tool-calling agent too (through its first
    ``view_env_state`` call); object poses are not
    included.
    """

    suite: str
    task: int | str
    seed: int
    task_language: str
    object_names: list[str]
    images: dict[str, str] = field(default_factory=dict)
    transfer: bool = False
    robot: str = "robodojo_xpl_rgb"
    facts: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> TaskCard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def render(self) -> str:
        """Card text: identifiers, instruction, facts, images."""

        lines = [
            "# Task card",
            f"- robot: {self.robot}",
            f"- task: {self.task}",
        ]
        if self.suite:
            lines.append(f"- variant: {self.suite}")
        lines += [
            f"- seed: {self.seed}",
            f"- task instruction (authoritative): {self.task_language!r}",
        ]
        if self.object_names:
            lines.append(f"- objects named by the benchmark: {', '.join(self.object_names)}")
        for key, value in self.facts.items():
            lines.append(f"- {key}: {value}")
        lines.append("- do not waste actions")
        if self.images:
            names = ", ".join(f"{k} ({Path(v).name})" for k, v in self.images.items())
            cameras = ", ".join(f"`robo.image({k!r})`" for k in self.images)
            if self.transfer:
                lines.append(
                    f"- attached images: {names}. They show ONE example layout "
                    f"(seed {self.seed}) of this task; your program will run on "
                    "other seeds where objects are placed differently, so do "
                    "not hard-code pixels or world positions read off them."
                )
            else:
                lines.append(
                    f"- attached images: {names}. They are step 0 of your run, "
                    f"exactly what {cameras} return before any motion."
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class PromptBundle:
    """System text, task text and image attachments for one generation."""

    system: str
    user: str
    images: tuple[str, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def text(self) -> str:
        return self.system + "\n\n" + self.user


CONTRACT = """\
# What you are asked to do

You control {robot_blurb} by WRITING A
PROGRAM, not by chatting with the robot.  Reply with one complete Python file
that defines

    def run(robo):

The host imports the file, constructs `robo` (a `{cls}` bound to the
live simulator{services}) and calls
`run(robo)` exactly once.  No language model runs while the program executes:
every decision has to be made in code, from the return values and images
documented below.

# How the episode is scored

{scoring_head}
- The return value of `run` is ignored.
{scoring_limits}
{scoring_tail}

# Output format

Reply with the complete source of `policy.py` in a single ```python fenced
block and nothing else.  The program may import `math`, `time` and `numpy`
(as `np`); do not import anything else, read files or start threads.  Keep it
self-contained: helper functions live in the same file.  Print short progress
lines with `print(...)`; stdout is saved with the episode.
"""


SCORING_NOTES = {
    True: (
        "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
        "  moment while `run` executes.  The simulator checks it, not you.",
        "- The episode counts as a failure when `run` raises an uncaught exception,\n"
        "  when the host's time limit of {budget_s} s runs out, or when the\n"
        "  benchmark ends the episode. do not waste actions.",
        "- Once the task terminates, every further motion call raises\n"
        "  `EpisodeFinished` (a BaseException): just return.",
    ),
    False: (
        "- Success is judged by the benchmark from the scene `run` leaves behind.\n"
        "  Nothing reports it while `run` executes: no result carries a task-success\n"
        "  flag, so the program must decide from its own observations when the\n"
        "  instruction is satisfied, and return.",
        "- `run` is stopped by an uncaught exception, by the host's time limit\n"
        "  of {budget_s} s or when the benchmark ends the episode; the\n"
        "  scene is judged as it is then.",
        "- do not waste actions. Once the episode ends, every further motion call\n"
        "  raises `EpisodeFinished` (a BaseException): just return.",
    ),
}


def render_prompt(
    card: TaskCard | None = None,
    *,
    budget_s: int = 1200,
    knowledge: str | None = None,
    robot_cls: type | None = None,
) -> PromptBundle:
    """Assemble the generation prompt for one cell (or a cell-less preview).

    The robot comes from ``card.robot`` (``robodojo_xpl_rgb`` when there is no
    card): its class supplies the API reference, its knowledge file the
    operating knowledge and its adapter the wording of the contract.
    """

    from .robots import get_adapter

    adapter = get_adapter(card.robot if card is not None else "robodojo_xpl_rgb")
    services = adapter.services
    scoring_head, scoring_limits, scoring_tail = SCORING_NOTES[bool(adapter.live_success)]
    contract = CONTRACT.format(
        budget_s=budget_s,
        robot_blurb=adapter.blurb,
        cls=(robot_cls or adapter.robot_cls).__name__,
        services=f", {services}" if services else "",
        scoring_head=scoring_head,
        scoring_limits=scoring_limits.format(budget_s=budget_s),
        scoring_tail=scoring_tail,
    )
    reference = api_reference(robot_cls or adapter.robot_cls)
    know = knowledge if knowledge is not None else knowledge_text(adapter.knowledge)
    system = "\n\n".join(
        [
            contract.rstrip(),
            "# API reference (generated from the library the host injects)\n\n" + reference,
            know.strip(),
        ]
    )
    if card is None:
        user = "# Task card\n(no cell selected; render with a TaskCard for a real run)"
        images: tuple[str, ...] = ()
    else:
        user = card.render()
        images = tuple(card.images.values())
    return PromptBundle(system=system, user=user, images=images)


__all__ = [
    "CONTRACT",
    "PromptBundle",
    "TaskCard",
    "api_reference",
    "knowledge_text",
    "render_prompt",
]
