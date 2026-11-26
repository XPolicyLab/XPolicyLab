"""The realized levels of the declared context manifest (review.context; docs/agent-icl-study/declared-factor-mode.md
§1, experiment-plan-round1.md §3 G3): what the LLM component reads under ``task.note``, ``state.embodiment`` and
``behavior.lessons_file``. Every level is opt-in: with the manifest absent or these switches at their off values,
``teacher.prompt_text``, ``teacher.prompt_identity`` and ``teacher.task_context`` never call this module and their
bytes are today's (tests/test_frozen_review_parity.py, tests/test_review_context.py).

``task.note`` (the first packet's ``task_context``)
    ``brief`` (off): task_context.json's note, as today. ``none``: the first packet carries no ``task_context``, and
    SKILL.md loses the statement describing it (``NOTE_STATEMENT``, removed exactly once). ``none_process``:
    ``task_context`` without ``process`` (``task`` and ``requires_arm_return`` stay); the prompt is unchanged.
    ``recipe`` (with ``task.recipe_file``): ``task_context`` additionally carries ``recipe``, the text of the file's
    ``## <task>`` section for the packet's task (``_random`` stripped); the prompt is unchanged. The file is markdown,
    one ``## <task>`` heading per task; the text before the first heading is the file's header and is never sent.
    A file without a section for the configured task is refused at validation (contracts.validate_context, through
    ``check_recipe``) and again when the packet is built.
``state.embodiment`` ``rich`` (with ``state.embodiment_file``)
    The developer prompt gains, right after teacher_context.md (before any schedule addendum), ``"\\n\\n"`` +
    ``EMBODIMENT_HEADING`` + ``"\\n\\n"`` + the file's text, verbatim.
``behavior.lessons_file`` (a path, or None: off)
    The developer prompt gains, last (after any schedule addendum), ``"\\n\\n"`` + ``LESSONS_HEADING`` + ``"\\n\\n"`` +
    the file's text, verbatim. Only the read path: the file is written elsewhere (G7).

Every file a switch names (recipe, embodiment, lessons) is a path under the benchmark's teacher prompt directory
(contracts._context_file) and enters ``prompt_identity["files"]`` under that relative path, hashed as the other
prompt files are (``contracts.digest`` of its text); ``prompt_sha256`` is the hash of the prompt as built.
"""
from __future__ import annotations

import re
from typing import Mapping

from .contracts import CONTEXT_SWITCHES, PACKET_FORBIDDEN_KEYS, digest

EMBODIMENT_HEADING = "# Embodiment record"
LESSONS_HEADING = "# Lessons from earlier episodes"
# task.note none: SKILL.md's statement of task_context and requires_arm_return (RoboDojo's SKILL.md:24-26), removed
# exactly once; the paragraph then starts with "After the object-level goal".
NOTE_STATEMENT = ("The first packet supplies this task's `task_context`. When `requires_arm_return=true`, success also "
                  "requires both arms\nback near their episode-start end-effector positions and orientations with both "
                  "grippers open (the benchmark's own\ncheck is roughly within 0.15 m and 20 degrees, openings at least "
                  "0.8). ")
RECIPE_HEADING = re.compile(r"^## (.+?)[ \t]*$", re.M)
# The switches whose files enter the prompt identity, in this order.
FILE_SWITCHES = (("task", "recipe_file"), ("state", "embodiment_file"), ("behavior", "lessons_file"))


def setting(manifest: Mapping | None, namespace: str, key: str):
    """One switch's value; its off value where the manifest or the key is left out."""
    return ((manifest or {}).get(namespace) or {}).get(key, CONTEXT_SWITCHES[namespace][key][0])


def note_level(manifest: Mapping | None) -> str:
    return setting(manifest, "task", "note")


def prompt_moved(manifest: Mapping | None) -> bool:
    """Whether the manifest changes a developer-prompt byte."""
    return (note_level(manifest) == "none" or setting(manifest, "state", "embodiment") == "rich"
            or setting(manifest, "behavior", "lessons_file") is not None)


def identity_moved(manifest: Mapping | None) -> bool:
    """Whether the manifest changes the prompt identity (the prompt, or a file it adds: the recipe file too)."""
    return prompt_moved(manifest) or any(setting(manifest, *switch) is not None for switch in FILE_SWITCHES)


def _read(path: str, benchmark: str) -> str:
    from .teacher import prompt_dir
    return (prompt_dir(benchmark) / path).read_text(encoding="utf-8")


def prompt_text_for(manifest: Mapping | None, schedule: str | None = None, benchmark: str = "robodojo") -> str:
    """The developer prompt under the manifest: today's prompt (teacher.prompt_text without a manifest) with the
    note statement removed (note none), the embodiment record after teacher_context.md and the lessons last."""
    from .teacher import prompt_files, prompt_text
    base = prompt_text(None, benchmark)  # SKILL.md, the gate prompt and teacher_context.md
    addendum = prompt_text(schedule, benchmark)[len(base):]  # the schedule's addendum ("" without one)
    if note_level(manifest) == "none":
        skill = prompt_files(benchmark)["SKILL.md"].read_text(encoding="utf-8")
        if skill.count(NOTE_STATEMENT) != 1 or base.count(NOTE_STATEMENT) != 1:
            raise ValueError("review.context.task.note none: SKILL.md no longer holds exactly once the statement it "
                             "removes (oeh_policy.review_context.NOTE_STATEMENT)")
        base = base.replace(NOTE_STATEMENT, "")
    embodiment = setting(manifest, "state", "embodiment_file")
    if setting(manifest, "state", "embodiment") == "rich":
        if embodiment is None:
            raise ValueError("review.context.state.embodiment: rich and embodiment_file go together")
        base += "\n\n" + EMBODIMENT_HEADING + "\n\n" + _read(embodiment, benchmark)
    text = base + addendum
    lessons = setting(manifest, "behavior", "lessons_file")
    if lessons is not None:
        text += "\n\n" + LESSONS_HEADING + "\n\n" + _read(lessons, benchmark)
    return text


def prompt_identity_for(manifest: Mapping | None, schedule: str | None = None, benchmark: str = "robodojo",
                        arms: tuple[str, ...] | None = None, gripper: bool = True) -> dict:
    """Today's prompt identity plus the hash of every file a switch names (under its relative path), and the hash of
    the prompt as built under the manifest."""
    from .teacher import prompt_identity
    from .review import ARMS
    identity = prompt_identity(schedule, benchmark, ARMS if arms is None else arms, gripper)
    for namespace, key in FILE_SWITCHES:
        path = setting(manifest, namespace, key)
        if path is not None:
            identity["files"][path] = digest(_read(path, benchmark))
    identity["prompt_sha256"] = digest(prompt_text_for(manifest, schedule, benchmark))
    return identity


def recipe_sections(text: str) -> dict[str, str]:
    """A recipe file's sections: task -> the text under its ``## <task>`` heading up to the next heading, stripped.
    The text before the first heading is the file's header. A task given twice is refused."""
    matches = list(RECIPE_HEADING.finditer(text))
    sections = {}
    for match, after in zip(matches, matches[1:] + [None]):
        task = match.group(1).strip()
        if task in sections:
            raise ValueError(f"the recipe file gives task {task} twice")
        sections[task] = text[match.end():after.start() if after is not None else len(text)].strip()
    return sections


def base_task(task: str) -> str:
    return task[:-len("_random")] if task.endswith("_random") else task


def check_recipe(path: str, task: str | None, benchmark: str = "robodojo") -> None:
    """Refuse a recipe file without a non-empty section for ``task`` (``_random`` stripped); no task: nothing to
    check (a review without a task sends no task_context)."""
    if not task:
        return
    sections = recipe_sections(_read(path, benchmark))
    if not sections.get(base_task(task)):
        raise ValueError(f"review.context.task.recipe_file {path} has no section '## {base_task(task)}' for the "
                         "configured task")


def task_context_for(manifest: Mapping | None, task: str | None, benchmark: str = "robodojo") -> dict | None:
    """The first packet's ``task_context`` under ``task.note``: none, no task_context; none_process, today's
    without ``process``; brief, today's; recipe, today's plus ``recipe`` (the file's section for this task)."""
    from .teacher import task_context
    level = note_level(manifest)
    context = task_context(task, benchmark)
    if level == "none" or context is None:
        return None
    if level == "none_process":
        context.pop("process")
    elif level == "recipe":
        path = setting(manifest, "task", "recipe_file")
        if path is None:
            raise ValueError("review.context.task.note: recipe and recipe_file go together")
        check_recipe(path, task, benchmark)
        context["recipe"] = recipe_sections(_read(path, benchmark))[base_task(task)]
    return context


def forbidden_words(text: str) -> list[str]:
    """The packet's oracle keys (contracts.PACKET_FORBIDDEN_KEYS) found in ``text`` as whole words, case-insensitive,
    a word being a run of letters, digits and underscores, or any underscore-separated part of one (so
    ``task_success`` hits ``success``). A key scan is for JSON; this is its counterpart for context files' text."""
    hits = []
    for word in re.findall(r"[a-z0-9_]+", text.lower()):
        for token in {word, *word.split("_")}:
            if token in PACKET_FORBIDDEN_KEYS and token not in hits:
                hits.append(token)
    return hits
