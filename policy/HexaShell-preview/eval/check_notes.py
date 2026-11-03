#!/usr/bin/env python3
"""Check tasks/<task>/notes.md, the notes the acting agent reads after the task recipe.

    check_notes.py TASK          a task without notes.md passes
    check_notes.py --file PATH   check one file

The notes hold only what the acting agent could see itself: commands and their options, failure reasons, simulation
seconds, counts of runs, what the images show. They must hold no position, size or height of an object and no layout,
because they are written from runs on the development layouts and read in scenes the agent has not seen.
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from roboshell.server.tools import base_task  # noqa: E402

HEADINGS = ["## Time", "## When a command fails", "## Before you stop", "## Known problems"]
MAX_LINES = 20          # lines of content, headings not counted
MAX_CHARS = 240         # per line
NUM = r"-?\d+(?:\.\d+)?"
TIME = re.compile(r"\d+(?:\.\d+)?\s*s\b")    # simulation seconds may have any number of decimals
QUOTED = re.compile(r"`[^`]*`")               # a tool's own message, quoted: its constants are design values
RULES = [   # (pattern, what it is, applies to quoted text too)
    (rf"{NUM}\s*(?:m|cm|mm|meters?|metres?|centimet\w*|millimet\w*)\b", "a length", False),
    (r"\b(?:centimet\w*|millimet\w*)\b", "a length", False),
    (r"\d\.\d{2,}|(?<![\w.])\.\d", "a number with two or more decimals (a coordinate or a size)", True),
    (rf"[\[(]\s*{NUM}\s*,\s*{NUM}(?:\s*,\s*{NUM})*\s*[\])]", "a coordinate", True),
    (r"(?<![A-Za-z0-9])(?:[xyz]|d[xyz]|[uv]\d?|p[uv]|c[xyz])\s*[=:]?\s*-?\.?\d", "a coordinate", True),
    (r"\b\d{2,4}\s*,\s*\d{2,4}\b", "a pixel position", True),
    (r"\blayouts?\b|\bscene \d", "a layout", True),
    (r"\b(?:true (?:positions?|poses?|locations?|heights?|sizes?)|ground[- ]truth|server_only|human_notes|playbook)\b",
     "a source the acting agent cannot see", True),
]


def problems(text):
    out = []
    lines = text.splitlines()
    headings = [line.strip() for line in lines if line.startswith("#")]
    if headings != HEADINGS:
        out.append(f"headings must be exactly {HEADINGS} in this order, found {headings}")
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    if not body:
        out.append("no content")
    if len(body) > MAX_LINES:
        out.append(f"{len(body)} lines of content, at most {MAX_LINES}")
    for n, line in enumerate(lines, 1):
        if len(line) > MAX_CHARS:
            out.append(f"line {n}: longer than {MAX_CHARS} characters")
        whole = TIME.sub("T", line)
        unquoted = QUOTED.sub("Q", whole)
        for pattern, what, in_quotes in RULES:
            if re.search(pattern, whole if in_quotes else unquoted, re.IGNORECASE):
                out.append(f"line {n}: {what}: {line.strip()[:120]}")
    return out


def main(argv):
    if argv[:1] == ["--file"]:
        path = argv[1]
    else:
        path = os.path.join(ROOT, "tasks", base_task(argv[0]), "notes.md")
        if not os.path.exists(path):
            print("notes: none")
            return 0
    found = problems(open(path, encoding="utf-8").read())
    print("notes: ok" if not found else "notes: " + "; ".join(found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
