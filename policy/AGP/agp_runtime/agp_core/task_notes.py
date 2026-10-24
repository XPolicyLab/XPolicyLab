"""Task notes: a short note per official task, appended to the launch message.

With OURS_TASK_NOTES on (agp_core/features.py, opt-in), agp_core/run_agent.py appends section(task) right after the rendered
main prompt (agp_core/PROMPT.md) and before the episode part that the preload adds (the task text and the first
observation). Off, nothing of this module runs and the launch message is unchanged.

Where the task name comes from: the official harness passes --task_name to the evaluator (src/eval_client/main.py
eval_cfg["task_name"], the EvalEnv's task_name) and the same value to the policy server's model_cfg (the policy folder's
setup_eval_policy_server.sh --overrides task_name=...). AGP's agent runs on the evaluator side, where the adapter reads
TASK_ENV.task_name once in the setup phase through the whitelist (agp_core/official_env.py METADATA,
TaskEnvWhitelist.episode_metadata) and writes it into the session's episode.json (`task`) before the policy starts.
launch_section reads that field; no other TASK_ENV access is involved.

The notes live in agp_core/task_notes.json: `notes` keyed by the 42 official task names (each states the task's official
scoring requirements, from the official success checks), and `length_review`, the source of every number with a
length unit (official thresholds with their file:line, advised motion amounts, values from our runs; the one exclusion
is a geometric fact about an object). A _random configuration gets its base task's note. A task without a note gets no
section.

Task hints: the notes hold only the official scoring requirements; the hints are a separate layer behind their own
opt-in switch OURS_TASK_HINTS. They live in agp_core/task_hints.json (`hints`: a hint for 22 tasks, one entry per base task,
appended verbatim; a _random configuration gets its base task's hint, a task without an entry gets no section). With
the switch on, agp_core/run_agent.py appends hint_section(task) (heading HINT_HEADING) right after the notes section (right
after the main prompt when OURS_TASK_NOTES is off) and records launch_hint_section's record in launch.json `task_hint`.
Off, nothing of it runs.
"""
import hashlib
import json
from pathlib import Path
import re

NOTES_FILE = Path(__file__).with_name("task_notes.json")
HINTS_FILE = Path(__file__).with_name("task_hints.json")
MAX_CHARS = 1500
HINT_MAX_CHARS = 1500
RANDOM_SUFFIX = "_random"
HEADING = "# Notes on this task"
INTRO = "The benchmark's official scoring for this task:"
HINT_HEADING = "# Hints for this task"
# a number with a length unit (a range such as 1-4 cm or 12 to 14 cm included), or a bare decimal in metres
LENGTH = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|m)\b"
                    r"|(?<![\w.])0\.\d+\b")
KINDS = {"official": ("cite", "code", "why"), "action": ("why",), "record": ("source", "why")}
ALLOWED_NON_ASCII = set("°≥≤–—×’")


def base_task(task):
    """The task whose note a configuration gets: stack_bowls_random -> stack_bowls."""
    task = str(task)
    return task[:-len(RANDOM_SUFFIX)] if task.endswith(RANDOM_SUFFIX) else task


def load(path=NOTES_FILE):
    return json.loads(Path(path).read_text())


def note_for(task, notes=None):
    """(base task, note text or None)."""
    notes = load()["notes"] if notes is None else notes
    base = base_task(task)
    text = notes.get(base)
    return base, (text if text else None)


def section(task, notes=None):
    """The launch-message section for this task: empty when the task has no note."""
    _, text = note_for(task, notes)
    return f"\n{HEADING}\n\n{INTRO}\n\n{text}\n" if text else ""


def session_task(session):
    """The official task name of the episode: episode.json `task`, written by the adapter from TASK_ENV.task_name."""
    path = Path(session) / "episode.json"
    try:
        task = json.loads(path.read_text()).get("task")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"task notes: no task name, {path} unreadable ({type(exc).__name__})")
    if not isinstance(task, str) or not task:
        raise SystemExit(f"task notes: {path} has no task name")
    return task


def launch_section(session, notes_file=NOTES_FILE):
    """(text to append after the main prompt, record for launch.json)."""
    task = session_task(session)
    notes = load(notes_file)["notes"]
    base, text = note_for(task, notes)
    out = section(task, notes)
    record = {"task": task, "note_of": base if text else None, "chars": len(text) if text else 0,
              "section_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
              "source": "episode.json task (TASK_ENV.task_name, setup phase)"}
    if not text:
        record["reason"] = f"no note for {base}"
    return out, record


# ----------------------------------------------------------------------------- task hints (OURS_TASK_HINTS)
def load_hints(path=None):
    """`hints` of agp_core/task_hints.json (base task -> hint text)."""
    return json.loads(Path(HINTS_FILE if path is None else path).read_text())["hints"]


def hint_for(task, hints=None):
    """(base task, hint text or None): a _random configuration gets its base task's hint."""
    hints = load_hints() if hints is None else hints
    base = base_task(task)
    text = hints.get(base)
    return base, (text if text else None)


def hint_section(task, hints=None):
    """The launch-message section with the hint for this task: empty when the task has none."""
    _, text = hint_for(task, hints)
    return f"\n{HINT_HEADING}\n\n{text}\n" if text else ""


def launch_hint_section(session, hints_file=None):
    """(text to append after the notes section, record for launch.json `task_hint`)."""
    task = session_task(session)
    hints = load_hints(hints_file)
    base, text = hint_for(task, hints)
    out = hint_section(task, hints)
    record = {"task": task, "hint_of": base if text else None, "chars": len(text) if text else 0,
              "section_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
              "source": "episode.json task (TASK_ENV.task_name, setup phase)"}
    if not text:
        record["reason"] = f"no hint for {base}"
    return out, record


def hint_problems(text):
    """Problems of a hint text (the checks below; no content rule)."""
    if not isinstance(text, str) or not text.strip():
        return [f"not a non-empty text: {text!r}"]
    return [f"{len(text)} characters (limit {HINT_MAX_CHARS})"] if len(text) > HINT_MAX_CHARS else []


def length_mentions(text):
    """Every number with a length unit (or bare decimal in metres) in a note, whitespace normalised, in order."""
    return [" ".join(m.group(0).split()) for m in LENGTH.finditer(text)]


def note_problems(text):
    """Problems of a note text (the checks below)."""
    problems = []
    if len(text) > MAX_CHARS:
        problems.append(f"{len(text)} characters (limit {MAX_CHARS})")
    other = sorted({c for c in text if ord(c) > 127 and c not in ALLOWED_NON_ASCII})
    if other:
        problems.append(f"characters outside English text: {other}")
    return problems


def review_problems(text, reviewed):
    """Every length mention of the note has a length_review entry of a known kind, and every entry names a mention."""
    problems = note_problems(text)
    mentions = set(length_mentions(text))
    texts = [entry.get("text") for entry in reviewed]
    for mention in sorted(mentions - set(texts)):
        problems.append(f"{mention!r} not reviewed")
    for entry in reviewed:
        if entry.get("text") not in mentions:
            problems.append(f"review entry {entry.get('text')!r} names no mention")
        kind = entry.get("kind")
        if kind not in KINDS:
            problems.append(f"{entry.get('text')!r}: unknown kind {kind!r}")
            continue
        missing = [key for key in KINDS[kind] if not entry.get(key)]
        if missing:
            problems.append(f"{entry.get('text')!r} ({kind}): missing {missing}")
    if len(texts) != len(set(texts)):
        problems.append("duplicate review entries")
    return problems
