"""Append the selected public RoboDojo recipe to newly copied instructions."""

import hashlib
import json
import os
from pathlib import Path
import re


RECIPE_DIR = Path(__file__).resolve().parents[1] / "tasks/RoboDojo/recipe"


def append_recipe(inputs, task, *, directory=None):
    """EMBODIEDEVO_USE_RECIPE=0 disables recipes; existing run inputs are never rebuilt."""
    enabled = os.environ.get("EMBODIEDEVO_USE_RECIPE", "1")
    if enabled not in {"0", "1"}:
        raise ValueError("EMBODIEDEVO_USE_RECIPE must be 0 or 1")
    if enabled == "0":
        return {"enabled": False}
    if not re.fullmatch(r"[A-Za-z0-9_]+", task):
        raise ValueError(f"Invalid recipe task name: {task!r}")
    directory = Path(directory) if directory is not None else RECIPE_DIR
    name = task.removesuffix("_random") + ".md"
    manifest = json.loads((directory / "manifest.json").read_text())
    if name not in manifest["files"]:
        raise ValueError(f"No recipe for {task!r}; set EMBODIEDEVO_USE_RECIPE=0 to omit recipes")
    recipe = (directory / name).read_bytes()
    digest = hashlib.sha256(recipe).hexdigest()
    if digest != manifest["files"][name]:
        raise ValueError(f"Recipe checksum mismatch: {name}")
    source = f"{manifest['repository']}/blob/{manifest['commit']}/{manifest['path']}/{name}"
    section = (f"\n\n<!-- TASK RECIPE BEGIN -->\n<!-- Source: {source}; {manifest['license']} -->\n"
               "TASK RECIPE:\n").encode() + recipe + b"\n<!-- TASK RECIPE END -->\n"
    for name in ("instruction.md", "playground.md", "test.md"):
        path = Path(inputs) / name
        if path.is_file():
            original = path.read_bytes()
            if original.endswith(section):
                continue
            if b"<!-- TASK RECIPE BEGIN -->" in original:
                raise ValueError(f"A different recipe is already pinned in {path}")
            path.write_bytes(original + section)
    return {"enabled": True, "task": task.removesuffix("_random"),
            "source": source, "sha256": digest}
