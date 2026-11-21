"""Task brief: the supplied task knowledge, each part gated by a ``knowledge.*`` switch.

Order: scoring notes (``_common.md`` + the task note; ``code`` also appends the official task
source), recipe, official task source, object models. The recipe and reference copies are
checksum-pinned by their ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from knowin_agent.knowledge.tasks import assets, base_task

SCORING_LEVELS = ("off", "docs", "code")
DEFAULTS: dict[str, Any] = {
    "knowledge.scoring": "docs",
    "knowledge.recipe": True,
    "knowledge.task_source": True,
    "knowledge.object_models": False,
}

SCORING_HEADING = "TASK SCORING:"
SOURCE_HEADING = "OFFICIAL TASK SOURCE (authoritative where any note above differs):"
RECIPE_HEADING = "TASK RECIPE:"
RECIPE_RULE = "Where the recipe disagrees with the scoring notes, the scoring notes win."
RECIPE_TOOLS = ("A recipe may name unavailable tools; only the API described in "
                "this prompt is available.")
OBJECTS_HEADING = "OBJECT MODELS:"


@dataclass(frozen=True)
class Brief:
    task: str
    text: str
    parts: tuple[str, ...]
    digests: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    switches: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def manifest(self) -> dict:
        """What the run manifest records about the brief: switches, included and missing parts, hashes."""
        return {"task": self.task, "switches": dict(self.switches), "parts": list(self.parts),
                "missing": list(self.missing), "digests": dict(self.digests),
                "sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()}


def _flag(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("on", "off"):
        return value == "on"
    raise ValueError(f"{name} must be on/off (or a bool), got {value!r}")


def knowledge_switches(switches: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The four knowledge switches from a flat ``{'knowledge.x': v}`` or nested ``{'knowledge': {...}}`` map."""
    values = dict(DEFAULTS)
    switches = dict(switches or {})
    nested = switches.pop("knowledge", None)
    flat = {f"knowledge.{k}": v for k, v in (nested or {}).items()}
    flat.update({k: v for k, v in switches.items() if k.startswith("knowledge.")})
    unknown = set(flat) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown knowledge switches {sorted(unknown)}")
    values.update(flat)
    if values["knowledge.scoring"] not in SCORING_LEVELS:
        raise ValueError(f"knowledge.scoring must be one of {SCORING_LEVELS}, got {values['knowledge.scoring']!r}")
    for name in ("knowledge.recipe", "knowledge.task_source", "knowledge.object_models"):
        values[name] = _flag(name, values[name])
    return values


def _manifest(directory: str) -> dict[str, str]:
    path = assets().joinpath(directory, "manifest.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["files"] if "files" in data else data


class _Reader:
    def __init__(self):
        self.parts: list[str] = []
        self.digests: dict[str, str] = {}
        self.missing: list[str] = []
        self._pins = {"recipe": _manifest("recipe"), "reference": _manifest("reference")}

    def read(self, relative: str) -> Optional[str]:
        """Asset text (checksum-checked when pinned), or None and a ``missing`` entry if absent."""
        if relative in self.parts:
            return None
        path = assets().joinpath(*relative.split("/"))
        if not path.is_file():
            self.missing.append(relative)
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        top, _, rest = relative.partition("/")
        pinned = self._pins.get(top, {}).get(rest)
        if top in self._pins and pinned is not None and pinned != digest:
            raise ValueError(f"checksum mismatch for {relative}: the pinned copy was modified")
        self.parts.append(relative)
        self.digests[relative] = digest
        return data.decode("utf-8").strip()


def task_brief(task: str, switches: Optional[Mapping[str, Any]] = None) -> Brief:
    """Compose the brief for ``task`` (``*_random`` uses its base task's files); ``str(brief)`` is the text."""
    name = base_task(task)
    values = knowledge_switches(switches)
    reader = _Reader()
    sections: list[str] = []

    def source() -> None:
        text = reader.read(f"reference/code/{name}.md")
        if text:
            sections.append(f"{SOURCE_HEADING}\n{text}")

    scoring = values["knowledge.scoring"]
    if scoring != "off":
        notes = [reader.read("scoring/_common.md"), reader.read(f"scoring/{name}.md")]
        notes = [n for n in notes if n]
        if notes:
            sections.append(SCORING_HEADING + "\n" + "\n\n".join(notes))
        if scoring == "code":
            source()
    if values["knowledge.recipe"]:
        text = reader.read(f"recipe/{name}.md")
        if text:
            rules = [RECIPE_RULE] if scoring != "off" else []
            sections.append("\n".join([RECIPE_HEADING, *rules, RECIPE_TOOLS, "", text]))
    if values["knowledge.task_source"]:
        source()
    if values["knowledge.object_models"]:
        text = reader.read(f"reference/objects/{name}.md")
        if text:
            sections.append(f"{OBJECTS_HEADING}\n{text}")
    return Brief(task=task, text="\n\n".join(sections), parts=tuple(reader.parts), digests=reader.digests,
                 missing=tuple(reader.missing), switches=values)
