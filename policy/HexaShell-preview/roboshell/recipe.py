#!/usr/bin/env python3
"""The acting agent's prompt: the official instruction, then the task recipe, then the task notes.

    recipe.py prompt --task TASK --instruction-file FILE   prints the prompt
    recipe.py info --task TASK                             prints which recipe and notes the prompt carries, as JSON

The recipes in agent/recipes/ are copied unchanged from RoboProbe (see agent/recipes/README.md) and appended after
"TASK RECIPE:" as the GPT-6 Astra L3 Inspect EEF baseline and EmbodiedEvo append them. ROBOSHELL_RECIPE=0 leaves it out,
the condition of every result before the recipes were added.
The notes are tasks/<task>/notes.md, written from earlier runs of the task's frozen tools on the development layouts
and checked by eval/check_notes.py (no object positions, sizes or layouts); a task without notes.md gets none.
ROBOSHELL_NOTES=0 leaves them out.
"""
import argparse, hashlib, importlib.util, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from roboshell.server.tools import base_task  # noqa: E402

RECIPES = os.path.join(ROOT, "agent", "recipes")
CHECK_NOTES = os.path.join(ROOT, "eval", "check_notes.py")
MARKER = "\n\nTASK RECIPE:\n"
NOTES_MARKER = "\n\nTASK NOTES (from earlier runs of the same tools in other scenes):\n"


def switch(name):
    value = os.environ.get(name, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, not {value!r}")
    return value == "1"


def enabled():
    return switch("ROBOSHELL_RECIPE")


def load(task):
    """(file name, text, sha256) of the task's recipe. A missing or edited file is an error, never a silent fallback."""
    name = f"{base_task(task)}.md"
    with open(os.path.join(RECIPES, "manifest.json")) as handle:
        manifest = json.load(handle)
    if name not in manifest["files"]:
        raise ValueError(f"no recipe for task {task!r}")
    with open(os.path.join(RECIPES, name), "rb") as handle:
        data = handle.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["files"][name]:
        raise ValueError(f"agent/recipes/{name} differs from manifest.json")
    return name, data.decode("utf-8"), digest


def notes_problems(text):
    """eval/check_notes.py on the text; loaded by path because eval/ is not a package."""
    spec = importlib.util.spec_from_file_location("check_notes", CHECK_NOTES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.problems(text)


def load_notes(task):
    """(path relative to the repository, text, sha256) of tasks/<task>/notes.md, or None when the task has no notes.
    Notes that fail eval/check_notes.py (for example after a hand edit) are an error, never passed on."""
    path = os.path.join("tasks", base_task(task), "notes.md")
    if not os.path.exists(os.path.join(ROOT, path)):
        return None
    with open(os.path.join(ROOT, path), "rb") as handle:
        data = handle.read()
    text = data.decode("utf-8")
    found = notes_problems(text)
    if found:
        raise ValueError(f"{path} fails eval/check_notes.py: {found[0]}")
    return path, text, hashlib.sha256(data).hexdigest()


def prompt(task, instruction):
    text = instruction.rstrip("\n")
    if enabled():
        text += MARKER + load(task)[1].rstrip()
    if switch("ROBOSHELL_NOTES"):
        notes = load_notes(task)
        if notes:
            text += NOTES_MARKER + notes[1].rstrip()
    return text


def info(task):
    result = {"recipe": False}
    if enabled():
        with open(os.path.join(RECIPES, "manifest.json")) as handle:
            manifest = json.load(handle)
        name, _, digest = load(task)
        result = {"recipe": True, "file": name, "sha256": digest, "source": manifest["repository"], "commit": manifest["commit"]}
    notes = load_notes(task) if switch("ROBOSHELL_NOTES") else None
    result["notes"] = {"file": notes[0], "sha256": notes[2]} if notes else False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prompt", "info"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction-file")
    args = parser.parse_args()
    try:
        if args.action == "prompt":
            with open(args.instruction_file) as handle:
                print(prompt(args.task, handle.read()))
        else:
            print(json.dumps(info(args.task)))
    except (OSError, ValueError) as error:
        sys.exit(f"recipe: {error}")
