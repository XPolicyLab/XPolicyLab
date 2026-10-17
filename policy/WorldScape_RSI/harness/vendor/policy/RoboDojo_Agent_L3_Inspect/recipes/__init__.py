"""Per-task wiki capability dimension, Description and Scoring, as recipes.

The dimension is the benchmark's own grouping of its 42 simulation tasks --
Generalization, Memory, Precision, Long-Horizon, Open -- and says what the task
is built to test, which is the closest thing the model gets to being told how
much care a task wants. The paper's sixth group, DLC, is an auxiliary
domain-randomized training-data directory rather than an evaluation task, so no
recipe carries it.

A few tasks add a trailing ``## Notes`` section that is not from the wiki: the
sequence a scripted opponent forces, or a scene fact the Description leaves
out. Advice that holds whatever the task is belongs in the embodiment notes
instead, which ride every prompt; a recipe is for what only its own task needs.
"""

from __future__ import annotations

from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parent


def task_recipe(task_name: str) -> tuple[Path, str] | None:
    """Load the wiki recipe for one task, or None when it has none.

    Random-layout variants share the base task's recipe. ``TASK RECIPE:`` is
    appended to the Goal turn when ``L3_INSPECT_USE_RECIPE`` is on.
    """
    if Path(task_name).name != task_name:
        raise ValueError(f"invalid task name for recipe lookup: {task_name!r}")
    name = task_name.removesuffix("_random")
    path = RECIPE_DIR / f"{name}.md"
    if not path.is_file():
        return None
    return path, path.read_text(encoding="utf-8")
