"""Disable host skills for the session-only robot experiment prompt."""

import json
from pathlib import Path


def skill_overrides() -> str:
    roots = (Path.home() / ".agents/skills", Path.home() / ".codex/skills")
    paths = sorted({str(path) for root in roots for path in root.rglob("SKILL.md")})
    entries = ["{path=" + json.dumps(path) + ",enabled=false}" for path in paths]
    return "skills.config=[" + ",".join(entries) + "]"


if __name__ == "__main__":
    print(skill_overrides())
