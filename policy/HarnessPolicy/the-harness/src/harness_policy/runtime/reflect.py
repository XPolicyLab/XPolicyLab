"""Reflection: turn a development episode into one candidate library entry.

The model reads a compact trajectory (programs, events, outcome on the dev
seed) and the current library, and may propose one entry:

    <note name="...">one or two sentences</note>             -> notes/<name>.md
    <memory name="...">situation and what worked</memory>   -> memory/<name>.md
    <skill name="...">def f(...): ...</skill>                -> skills/<name>.py
    <reactive name="...">def g(obs): ...</reactive>          -> reactive/<name>.py
    <none/>

A candidate only enters the library after it improves the score on held-out
development seeds (scripts/self_improve.py).
"""

import ast
import json
from pathlib import Path
import re


# Notes go into every prompt, memory entries only when recall finds them: a longer entry is
# discarded (the prompt says so; before it did not, and half the proposals were lost).
NOTE_MAX_CHARS, MEMORY_MAX_CHARS = 600, 1200
REFLECT_SYSTEM = (
    "You improve a robot-control harness between episodes. A coding agent controls a dual-arm robot "
    "by writing Python against a skill API. You read one finished development episode and the "
    "current library of lessons and helper functions, and propose at most ONE addition that would "
    "help future episodes on any task: a short note (a general lesson, not about this layout's "
    "coordinates), a memory entry (a specific situation and what worked or failed, which the agent "
    "can look up with recall(query)), a helper skill (a Python function using the API names the "
    "agent uses), or a reactive helper (a pure, fast predicate or controller of obs). Reply with "
    'exactly one of <note name="snake_case">...</note>, <memory name="snake_case">...</memory>, '
    '<skill name="snake_case">...</skill>, '
    '<reactive name="snake_case">...</reactive>, or <none/>. Code must define one public function '
    f"with a one-line docstring. A note is at most {NOTE_MAX_CHARS} characters, a memory entry at most "
    f"{MEMORY_MAX_CHARS}; a longer one is discarded."
)
_ENTRY = re.compile(r'<(note|memory|skill|reactive)\s+name="([a-z][a-z0-9_]{1,40})">(.*?)</\1>', re.DOTALL)
SECTION = {"note": ("notes", ".md"), "memory": ("memory", ".md"), "skill": ("skills", ".py"),
           "reactive": ("reactive", ".py")}


def summarize_episode(run: Path, max_chars: int = 12000) -> str:
    lines = []
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    for line in (run / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        kind = event["event"]
        if kind == "reset":
            lines.append(f"Instruction: {event['observation']['instruction']}")
        elif kind == "model_reply":
            lines.append(f"[turn {event['call']}] {event['text'].strip()[:1500]}")
        elif kind == "cell":
            result = event["result"]
            text = f"  -> {result['status']}"
            if result.get("escalation"):
                ev = result["escalation"]["event"]
                text += f" {ev['kind']} from {ev['source']}: {json.dumps(ev['detail'])[:300]}"
            if result.get("error"):
                text += " " + result["error"].strip().splitlines()[-1][:300]
            if result.get("stdout"):
                text += " | out: " + result["stdout"].strip().replace("\n", " / ")[:300]
            lines.append(text)
        elif kind == "verify":
            lines.append(f"  verify {event['claim'][:200]!r}: {event['answer'][:200]}")
    lines.append(f"Outcome on this development seed: success={summary.get('success')} "
                 f"score={summary.get('reward')} status={summary.get('status')}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars // 2] + "\n...\n" + text[-max_chars // 2:]
    return text


def library_text(directory: Path) -> str:
    parts = []
    for section in ("notes", "memory", "skills", "reactive"):
        for path in sorted((directory / section).glob("*")) if (directory / section).is_dir() else []:
            parts.append(f"--- {section}/{path.name}\n{path.read_text().strip()}")
    return "\n".join(parts) or "(empty)"


def propose(chat, run: Path, library: Path):
    """Ask the model for one candidate; returns (section, filename, text) or None."""
    prompt = (f"Current library:\n{library_text(library)}\n\nEpisode:\n{summarize_episode(run)}\n\n"
              "Propose at most one addition.")
    reply = chat.complete(REFLECT_SYSTEM, [{"role": "user", "content": prompt}])
    return parse_candidate(reply.text), reply.text


def parse_candidate(text: str):
    match = _ENTRY.search(text or "")
    if not match:
        return None
    kind, name, body = match.group(1), match.group(2), match.group(3).strip()
    section, suffix = SECTION[kind]
    if kind in ("skill", "reactive"):  # code; a note and a memory entry are text
        body = re.sub(r"^```[a-z]*\n|\n```$", "", body).strip()
        try:
            tree = ast.parse(body)
        except SyntaxError:
            return None
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")]
        if len(functions) != 1 or not ast.get_docstring(functions[0]):
            return None
        body += "\n"
    elif not body or len(body) > (NOTE_MAX_CHARS if kind == "note" else MEMORY_MAX_CHARS):
        return None
    return section, name + suffix, body if body.endswith("\n") else body + "\n"
