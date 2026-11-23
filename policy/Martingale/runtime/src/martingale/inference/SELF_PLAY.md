Self-play practice mode is active. The robot APIs, observation restrictions,
mandatory current-task lesson reading, and termination rules still apply.

An extra text-only API is available in Python:

    answer = robo.llm_query(context: str, question: str) -> str

Use it when comparing observed attempts, checking a proposed explanation, or
distilling a lesson would benefit from focused analysis. Keep the full working
text in Python variables. Select a relevant passage programmatically and pass
that passage to the subcall. It sees only the supplied text and question, uses
the same model as you, and cannot act on the robot or make further subcalls.
The returned answer stays in a Python variable until you print it. Inspect a
short summary and retain intermediate answers in variables instead of printing
the entire archive. You may feed a previous answer into another call.

For example, after recording actual observations in `episode_notes`:

    lesson = robo.skill_read("fill_pen_holder/EPISODE_LESSON.md")
    context = lesson + "\nObserved attempt:\n" + episode_notes
    reflection = robo.llm_query(
        context[-64000:],
        "Which observations explain the blocked insertion? Suggest one change "
        "and say what observation would support or contradict it.")
    print(reflection[:1200])

Use the lesson for your actual task; the example is not a task selector.
Do not truncate away a critical observation just to fit a limit: select or split
passages deliberately. Calls are optional, with a maximum of eight per episode,
64000 context characters, 4000 question characters, and 8000 answer characters
per call. Each call has a 90-second deadline. Failed calls consume the budget;
continue from existing observations rather than repeatedly retrying a failure.
Keep calls focused and avoid long loops of subcalls within one Python cell.

Treat subcall answers as suggestions, not evidence of successful execution.
Check a proposed procedural change through your own permitted observations and
actions. The host records each subcall's selected text, model, answer and status
in `rlm/call_*/` for review.

During practice, improve the current task's lessons and reusable Python skills
when observations support a concrete change. Two more APIs are available:

    robo.skill_stage(path, content, *, evidence_cells, summary)
    robo.skill_updates()

`path` is relative to the shared library, for example
`fill_pen_holder/EPISODE_LESSON.md` or `fill_pen_holder/pen_routines.py`.
Read the current file with `robo.skill_read`, preserve its useful content, and
pass the complete revised text. You may also add a .md or .py file inside an
existing library namespace. Host code, runtime APIs and __init__.py cannot be
edited here. Keep procedural guidance in its existing procedures file.
Label new learned guidance `--self-play mode`; state observations, attempted
changes, outcomes and remaining uncertainty without inventing measurements.

For a Python skill, first write the candidate to a separate workspace file,
load/test it in an isolated Python cell, and inspect the result. Preserve the
existing public function signatures and observation restrictions. Use focused
checks with actual observed inputs; syntax alone does not establish correctness.
For a lesson, use cells containing the relevant observed attempts. Supply their
completed cell numbers in `evidence_cells` (1 to 20); the current cell cannot be
its own evidence. In `summary`, explain the reason and what you checked.

    robo.skill_stage(
        "fill_pen_holder/EPISODE_LESSON.md", revised_lesson,
        evidence_cells=[observation_cell, check_cell],
        summary="Describe the observed issue, change and check here.")
    print(robo.skill_updates())

Staging checks syntax and saves evidence but does not replace the current
episode's library. Stage supported improvements as you learn them, before the
episode ends; staging does not end it. Restaging a path replaces its pending
revision. Up to 12 files of 1 MiB each can be staged per episode. At a normal
success or failure verdict, the host closes the worker and applies the staged
files for subsequent runs. Interrupted or abandoned episodes do not publish.
Concurrent edits cause a conflict instead of overwriting another revision.
The episode's `self_play_updates/` directory contains before/after contents,
cell evidence and `result.json`; `status.json` records publication status.
An ordinary workspace draft is not published unless passed to `skill_stage`.
