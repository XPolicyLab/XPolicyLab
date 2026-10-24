"""System prompts assembled from a prompt pack for the skills a run enables.

A pack is a directory of Markdown sections (prompts/robodojo, prompts/robotwin). Its
pack.toml lists the sections in order; a section with `skills` is included only when
one of them is enabled:

    sections = [
      { file = "core.md" },
      { file = "learned.md", skills = ["vla", "wam"] },
      { file = "rules_learned.md", skills = ["vla"], optional = true },
    ]

Inside a section, lines can be kept for some skills only:

    <!-- if: pointing, segmentation -->      kept when all named skills are enabled
    ...
    <!-- else -->                            kept otherwise (optional)
    ...
    <!-- end -->

Markers may also name features of the robot connection (FEATURES): measured_gripper
holds when gripper readings are measured openings, not under a contract that reports
only the last command, and step_limit when the episode's step limit is reported
(core/contract.py). The marker lines themselves never reach the
model, so a prompt with every skill and feature enabled is the pack's text without them. The API documentation stays in the pack
because it describes one embodiment (its frames, reach and numbers); the skills'
code is the same for every embodiment.
"""

from pathlib import Path
import re
import tomllib


# Sections of a pack without pack.toml (the RoboDojo layout before packs had one).
DEFAULT_SECTIONS = (
    {"file": "core.md"},
    {"file": "learned.md", "skills": ["vla", "wam"]},
    {"file": "reactive.md", "skills": ["reactive"]},
    {"file": "events.md"},
    {"file": "handlers.md", "skills": ["handlers"]},
    {"file": "rules.md"},
    {"file": "rules_learned.md", "skills": ["vla"], "optional": True},
    {"file": "rules_reactive.md", "skills": ["reactive"], "optional": True},
    {"file": "rules_handlers.md", "skills": ["handlers"], "optional": True},
)
# Features of the robot connection that prompt text may depend on (not skills).
FEATURES = ("measured_gripper", "step_limit")


def features_for(runtime: dict) -> set[str]:
    """The prompt features of a runtime configuration (its observation contract)."""
    return set(FEATURES) if runtime.get("contract", "native") == "native" else set()


_MARK = re.compile(r"^\s*<!--\s*(if:\s*(?P<skills>[\w,\s]+?)|(?P<else>else)|(?P<end>end))\s*-->\s*$")


def pack(directory: Path) -> dict:
    """The pack's settings: sections and the exported names it leaves undocumented."""
    path = Path(directory) / "pack.toml"
    if not path.is_file():
        return {"sections": [dict(section) for section in DEFAULT_SECTIONS], "undocumented": []}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    unknown = set(data) - {"sections", "undocumented"}
    if unknown or not data.get("sections"):
        raise ValueError(f"{path}: needs `sections` (and may have `undocumented`); unknown keys {sorted(unknown)}")
    return {"sections": [dict(section) for section in data["sections"]],
            "undocumented": list(data.get("undocumented", []))}


def sections(directory: Path) -> list[dict]:
    """The pack's sections in order ({"file", and optionally "skills", "optional"}):
    pack.toml's, or DEFAULT_SECTIONS without one."""
    return pack(directory)["sections"]


def select(text: str, skills) -> str:
    """Keep the lines of `text` whose if/else markers hold for the enabled `skills`."""
    enabled = set(skills)
    stack = []  # (condition holds, in the else branch)
    out = []
    for number, line in enumerate(text.split("\n"), 1):
        match = _MARK.match(line)
        if match is None:
            if all(holds != in_else for holds, in_else in stack):
                out.append(line)
            continue
        if match.group("skills") is not None:
            names = [name.strip() for name in match.group("skills").split(",") if name.strip()]
            stack.append((all(name in enabled for name in names), False))
        elif match.group("else"):
            if not stack or stack[-1][1]:
                raise ValueError(f"line {number}: <!-- else --> without an open <!-- if: ... -->")
            stack[-1] = (stack[-1][0], True)
        else:
            if not stack:
                raise ValueError(f"line {number}: <!-- end --> without an open <!-- if: ... -->")
            stack.pop()
    if stack:
        raise ValueError("an <!-- if: ... --> block is not closed with <!-- end -->")
    return "\n".join(out)


def marked_skills(text: str) -> set[str]:
    """Skills named in the markers of `text`."""
    names = set()
    for line in text.split("\n"):
        match = _MARK.match(line)
        if match and match.group("skills"):
            names |= {name.strip() for name in match.group("skills").split(",") if name.strip()}
    return names


def assemble_prompt(directory: Path, skills, overrides: dict | None = None,
                    features=("measured_gripper",)) -> str:
    """The pack's sections for the enabled `skills` and connection `features`, joined.
    `overrides` maps a section name (file stem) to another file, e.g. another
    embodiment's core section."""
    directory = Path(directory)
    enabled = set(skills) | set(features)
    parts = []
    for section in sections(directory):
        name = Path(section["file"]).stem
        wanted = section.get("skills")
        if wanted and not set(skills) & set(wanted):
            continue
        path = Path((overrides or {}).get(name) or directory / section["file"])
        if not path.is_file():
            if section.get("optional"):
                continue
            raise ValueError(f"Prompt section {path} is missing")
        parts.append(select(path.read_text(), enabled).rstrip("\n"))
    return "\n".join(parts) + "\n"


def mentions(text: str, name: str) -> bool:
    """Whether the prompt refers to an API name: a call for functions, the bare word
    for constants (upper case)."""
    pattern = rf"(?<![\w.]){re.escape(name)}\b" if name.isupper() else rf"(?<![\w.]){re.escape(name)}\("
    return re.search(pattern, text) is not None


def documentation_gaps(directory: Path, skills, exported: dict, features=("measured_gripper",)) -> dict:
    """What the assembled prompt gets wrong about the API: `missing` are functions of
    enabled skills it never shows being called (except those the pack lists as
    undocumented), `stray` are names of disabled skills it still mentions.
    `exported` maps each exported name to its skill, or to the skills that export it."""
    text = assemble_prompt(directory, skills, features=features)
    enabled = set(skills)
    skip = set(pack(directory)["undocumented"])
    public = {name: {owner} if isinstance(owner, str) else set(owner)
              for name, owner in exported.items() if not name.startswith("_")}
    missing = sorted(name for name, owners in public.items()
                     if owners & enabled and not name.isupper() and name not in skip and not mentions(text, name))
    stray = sorted(name for name, owners in public.items() if not owners & enabled and mentions(text, name))
    return {"missing": missing, "stray": stray}
