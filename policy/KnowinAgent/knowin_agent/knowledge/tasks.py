"""RoboDojo task names, their base task (``*_random`` variants share it) and capability dimension."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from typing import Optional

DIMENSIONS = ("Generalization", "Long-Horizon", "Memory", "Open", "Precision")
RANDOM_SUFFIX = "_random"

_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_DIMENSION_HEADING = "## Capability dimension"


def assets():
    return resources.files("knowin_agent.assets").joinpath("robodojo")


def check_task(task: str) -> str:
    if not isinstance(task, str) or not _TASK_NAME.fullmatch(task):
        raise ValueError(f"invalid RoboDojo task name {task!r}")
    return task


def base_task(task: str) -> str:
    check_task(task)
    return task[:-len(RANDOM_SUFFIX)] if task.endswith(RANDOM_SUFFIX) else task


@lru_cache(maxsize=1)
def task_names() -> tuple[str, ...]:
    """The 42 base tasks (one scoring note each)."""
    scoring = assets().joinpath("scoring")
    return tuple(sorted(p.name[:-3] for p in scoring.iterdir() if p.name.endswith(".md") and not p.name.startswith("_")))


@lru_cache(maxsize=None)
def task_dimension(task: str) -> Optional[str]:
    """RoboDojo capability dimension of a task, read from its recipe; None if unknown."""
    recipe = assets().joinpath("recipe", f"{base_task(task)}.md")
    if not recipe.is_file():
        return None
    lines = recipe.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _DIMENSION_HEADING:
            for following in lines[index + 1:]:
                if following.strip():
                    name = following.split("—")[0].strip()
                    return name if name in DIMENSIONS else None
    return None
