#!/usr/bin/env python3
"""Assemble the manual the agent reads.

    docs.py build --task TASK      writes agent/build/AGENTS_<TASK>.md and prints its path
    docs.py build                  the base manual only (agent/AGENTS.md)
    docs.py check --task TASK      the manual of a task must not name the task or its objects
    docs.py banned --task TASK     the words that check rejects

AGENTS.md = agent/INTERFACE.md + the interface.md of every tool enabled in tasks/<task>/enabled_tools.txt
(tools/<name>/interface.md or the task's own tools/<name>/interface.md), in that order.
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from roboshell.server.tools import base_task, enabled_tools, find_tool  # noqa: E402


def tool_doc(name, task):
    path = os.path.join(os.path.dirname(find_tool(name, task)), "interface.md")
    with open(path) as handle:
        return handle.read().strip()


def build(task):
    task = base_task(task)
    with open(os.path.join(ROOT, "agent", "INTERFACE.md")) as handle:
        text = handle.read().rstrip() + "\n"
    if task and os.environ.get("ROBOSHELL_TOOLS", "").strip() in ("none",):
        task = None  # base tools only
    if task:
        names = enabled_tools(task)
        if names:
            text += "\n## Extra tools for this task\n"
        for name in names:
            text += "\n" + tool_doc(name, task) + "\n"
    out = os.path.join(ROOT, "agent", "build", f"AGENTS_{task}.md") if task else os.path.join(ROOT, "agent", "build", "AGENTS_base.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(text)
    return out


# Words every manipulation manual needs; an instruction containing them does not make them task words.
GENERIC = set("""align aligned alignment place placed put move moved push pushed pull pick picked grasp grip hold held lift lifted lower rotate
turn slide open close closed release stack insert pour press with into onto from that this then them they their there
precisely exactly carefully slowly gently left right front back center centre middle side sides upper lower above below
position pose target goal object objects item items surface table shape shaped each every both same first second third
until while after before using make sure keep task robot gripper hand hands""".split())


def banned_words(task):
    """Words that would name the task or its objects: from the instruction templates, the object labels and the task name."""
    task = base_task(task)
    words = set()
    src = os.path.join(os.environ.get("ROBODOJO_REPO", ""), "task", "RoboDojo", "tasks", f"{task}.py")
    if os.path.exists(src):
        text = open(src).read()
        quoted = re.findall(r'"([^"\n]{12,})"', text)                    # instruction sentences
        labels = re.findall(r'label(?:_[A-Z])?\s*=\s*"([^"]+)"', text)  # object labels
        for chunk in quoted + labels:
            if "<" in chunk or "/" in chunk:
                continue
            words |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", chunk)}
    words |= {w for w in task.lower().split("_") if len(w) >= 4}
    return sorted(words - GENERIC)


def check(task):
    """Interface only: the tool descriptions must not name the task or its objects (whole words, plural included)."""
    task = base_task(task)
    text = open(build(task)).read().lower()
    base = open(os.path.join(ROOT, "agent", "INTERFACE.md")).read().lower()
    extra = text[len(base.rstrip()):] if text.startswith(base.rstrip()) else text
    hits = [w for w in banned_words(task) if re.search(rf"\b{re.escape(w)}s?\b", extra) and not re.search(rf"\b{re.escape(w)}s?\b", base)]
    print("task words found in the manual:", hits or "none")
    return 0 if not hits else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "check", "banned"])
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.action == "build":
        print(build(args.task))
    elif args.action == "banned":
        print(" ".join(banned_words(args.task)))
    else:
        sys.exit(check(args.task))
