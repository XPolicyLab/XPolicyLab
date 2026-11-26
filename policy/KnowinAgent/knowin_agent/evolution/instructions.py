"""Instruction texts for the simulator-free coding-agent sessions (review merge, shared-library curation).

Ported from our H4 console (training v2 review, cross-task curation) and rewritten for KnowinAgent's interfaces:
agents submit Python programs through the `exec` tool (code + include list) into a program namespace.
"""

REVIEW_PROMPT = ("Read instruction.md and follow it. You have no robot or simulator. Write only under out/. "
                 "Stop when out/skills, out/lessons (with PLAN.md) and out/REPORT.md are complete.")

CURATE_PROMPT = ("Read instruction.md and follow it. You have no robot or simulator. Write only under out/. "
                 "Stop when out/library, out/tests and out/CHANGES.md are complete and the tests pass.")

HOW_AGENTS_ACT = """\
The robot agent acts by submitting Python programs through its `exec` tool: `exec(code, include=[...])`.
`include` lists `skills/<name>.py`, `skills/shared/<name>.py` or built-in `knowin_agent:<module>` files; they are loaded
into the same program namespace before `code` runs. The namespace already provides the robot functions
(`move_to`, `set_gripper`, `check_reach`, `checkpoint`, `subgoal_done`, `get_observation`, `step`, `reset`,
`grasp_point`, `pick`, `place`, `look_down`, `screw_down`, `press_at`, `release_yaw`, `best_yaw`) and the KnowinAgent
extras (`locate`, `inspect`, `observe`, `grasp_candidates`, `run_skill`, `wait`, asynchronous
`move_to(..., wait=False)` so one arm moves while the other works, and delegation of a fine alignment to the
Local agent); see `reference/api.md`. A failed `checkpoint` stops the rest of that submission, and every
receipt carries the budget panel (native actions left, estimated cost of returning home)."""

REVIEW_INSTRUCTION = """# Review: merge practice knowledge for {task}

You are reviewing practice runs of the robot task `{task}`. You cannot run the robot. Your output is the
only knowledge a fresh agent will get before it solves an unseen scene of this task with a limited budget
(about {test_budget} `exec` submissions, and the task's native action limit). Make it as useful as possible.

## How the agent acts

{how_agents_act}

## Inputs (read-only)

- `practice/run*_seed*/`: one folder per practice run on a different training scene. Start with its
  `timeline.md`: what the run started from, whether it played out, its attempts (the run split at each reset,
  with each attempt's best Score), its budget, then one line per execution (Score after it, native actions,
  early stop, failed checkpoints, how its moves ended, gripper openings, the agent's stated intent), per
  delegation to the Local agent and per single tool call, then the agent's notes and every delegation's goal,
  report and Local calls. `executions/<n>.py` is every program the run submitted, with its include list, the
  agent's intent and the receipt (native actions, error, checkpoints, motion, printed output). The views after
  each execution are where `timeline.md` says (all of them in the read-only run directory when it is mounted,
  else `views/` for the key executions marked `*`). Open the views of an execution before trusting its code, a
  lesson or a Score about it. `summary.json` holds the same numbers for scripts (Scores, `attempts`,
  `motion_health`, `native_steps_per_execution`, `delegations`). The run's `skills/` and `lessons/` are its
  knowledge at the end. `skills/shared/` (if present) is the read-only cross-task library the run started with.
  It is restored unchanged in your merge: do not write `out/skills/shared/`; put task-specific variants or
  wrappers in `out/skills/`. Each run's `lessons/shared_proposals.md` (if present) proposes changes to that
  library; merge them, deduplicated and with their evidence, into `out/lessons/shared_proposals.md`. A run whose
  timeline says it started from your previous merge (`previous/`) began on a new scene from it: it shows whether
  that knowledge transferred and is the main evidence for the later stages of the task. A run that started from
  other knowledge says so; it is no test of `previous/`.
- `practice/mined_candidates.json`: call sequences that recur across the runs' structured traces (tool calls
  made outside `exec`, Local corrections), with the arguments that varied. A sequence that recurs and is backed
  by Score is a candidate for a skill; otherwise ignore it.
{dev_section}{previous_section}{oracle_section}- `reference/scoring.md`: the evaluator's exact scoring rule for this task.
- `reference/api.md`: the program namespace and the `exec` tool, including batched submissions.
{extra_references}
## Output (write only under `out/`)

1. `out/skills/`: merged reusable code (`.py`, includable through the `include` list of `exec`; definitions
   only, no top-level robot actions; only `numpy` and `math` imports; no names starting with `_`) plus a short
   `.md` per skill: parameters, preconditions, evidence (which run and Score backs it).
2. `out/lessons/`: merged lessons. Start with `out/lessons/PLAN.md`: one page, a three-column table of
   sub-goals in order (`What` / `Verification` / `Recovery`). What: arm(s), object, from → to, heading,
   skill/`include`, native-action estimate. Verification: the image or measurement that shows it worked.
   Recovery: what to change if it fails. Batch a sub-goal into one submission once its motion is proven;
   where the evidence shows the two arms can work at once, say so in What. Include the return to the
   starting pose when the scoring requires it. Keep the lesson files that carry evidence, merged across
   runs: `EVIDENCE.md` (which run, execution and Score backs each claim), `RECOVERY.md` (failure signatures
   and what recovered from them), `PERCEPTION.md` (measurement procedures and the values that must be
   re-measured) and `shared_proposals.md`.
3. `out/REPORT.md`: what you kept, merged, rewrote or dropped and why; conflicts between runs and how you
   resolved them; open risks. It must contain a `## Changes` table with one row per PLAN row or file you kept,
   rewrote, deleted or added: `| PLAN row or file | kept / rewritten / deleted / added | evidence: run, execution,
   what its views or receipt show |`. A merge without this table is rejected.

## Rules

- Evidence decides: prefer what reached higher Scores; when runs disagree, keep the better-scoring
  approach and record the other as a fallback only if it has its own evidence.
- Locate a failure before you fix it. Change or delete a step only after finding the execution where that step
  failed, in a run that carried it out, and checking its views. A Score below the top tier does not say which
  item failed: tiers often count items ("any 2 of 4 correct"), so read which item is wrong from the views and
  measurements, never from the Score alone. If you cannot find where a step failed, keep it and write the doubt
  under open risks. Never rewrite a step that the views show in place at the end of a run.
- What a run verified stays verified. A step whose result the views show in place right after the Score rose,
  or at the end of a run, is proven for that run; keep it with that evidence. Facts proven this way about an
  object model (which axis the scoring uses, which end is the front, a grasp that held) are knowledge, not
  scene constants: keep them in PERCEPTION.md with the run and execution that proved them.
- A run whose timeline says it did not play out (stopped by the console or an operator, the agent died) is no
  evidence that a step failed: the tiers it held are a lower bound, and the steps it never reached were not
  tested. Do not delete or rewrite anything because of such a run.
- Learn from the attempt that worked, for the first attempt. When a run reached its best only after one or more
  resets, compare the attempt that failed with the one that succeeded and write what changed so that the next
  agent, who has one attempt and no Score, does it right the first time. Where the success relied on what an
  earlier failure in that scene showed (where the liquid landed, how far a part slid), replace that by a
  measurement the agent takes before the irreversible step.
- Assign arms by reach in the scene at hand (`check_reach` on the targets), not as fixed left/right roles,
  unless the scoring names an arm. A scene with the object on the other side must still be covered.
- Scene-specific constants (object positions, heights measured in one scene) must become observation-driven
  parameters or clearly marked defaults to re-measure. The next scene is different.
- When the instruction (`get_instruction()`) differs between scenes (which object, colour, number, order,
  equation or reference image it names), keep how to read it and act on it: parse it, find what it names,
  check the result against it. Never keep one scene's answer (a colour, an object, a result) as a default.
- Do not invent behaviour that no run tested; mark untested ideas as hypotheses in lessons.
- Remove duplicates and stale notes. Keep it short enough to read in a few minutes: at most {max_files}
  files and {max_kb} KB under `out/`.
- Every step you keep must serve a scoring condition in `reference/scoring.md`, directly or because the
  physics needs it to reach that condition (a nut reaches its depth only by being screwed down). A step that
  only happened to precede a success in one run, with no such link, is marked "optional" with its
  native-action cost. Where a recipe and the scoring disagree, follow the scoring. Record the native actions
  each subgoal took (`native_steps_per_execution` in each `summary.json`), so the next agent sees where the
  budget goes; failed tests mostly ran out of native actions, so a cheaper proven step beats a safer slow one.
- Test agents get no score. Each Verification entry in PLAN.md must be checkable from images and measurements
  alone (which camera or measurement, and the threshold from `reference/scoring.md`), never "the score rose".
  End PLAN.md with a short "Done when" checklist: every condition of the top tier, each with its check. Take
  these from what the views after the successful runs' last programs show.
- PLAN.md: at most 12 rows and about 2 KB, no run names, seeds or execution numbers (those go to EVIDENCE.md),
  and only the numbers the next agent must re-measure or budget against.
- Use `motion_health` to find the sub-goal that fails mechanically: the same move stalling or dropping the part,
  or closes finding nothing, across runs means its approach must change in PLAN.md (another grasp pose, pitch or
  arm, a lower carry height, a shorter move, re-measuring first), not the same code again. Write each such
  signature and what recovered from it in RECOVERY.md.
- Revise, do not accumulate. When `previous/` exists, the runs that started from it are the test of each of
  its steps: keep a step they carried out successfully, rewrite a step they failed at (located as above), and
  delete content they contradicted. A step they skipped because its condition did not hold in their scene, or
  never reached, is untested, not wrong: keep it. A merge that is longer than the previous one needs a reason
  in REPORT.md; every deletion is a row of the `## Changes` table.
- Runs whose `summary.json` has `score_feedback_during_run: false` practised without score feedback, as a test
  does: their final Score is the honest measure of how well the knowledge they started from transfers.
"""

REVIEW_DEV_SECTION = ("- `dev/seed*/`: tests of your previous merge (`previous/`) on unseen validation scenes, laid "
                      "out as the practice runs (`timeline.md`, `executions/`, views; `scratch_notes/` when the "
                      "agent wrote any). The test agent saw no Score; the Scores in its timeline were recorded by "
                      "the host and read only now. Find in the views where each failed tier was lost and fix "
                      "that; keep what worked. A folder ending in `-cut_off` was stopped before the agent "
                      "finished: it is no evidence that a step failed.\n")

REVIEW_ORACLE_SECTION = ("- `practice/run*/ground_truth.md`, `dev/seed*/ground_truth.md` (when present): the "
                         "evaluator's own record of which scoring predicates held after each execution. It names "
                         "the item that failed; still check its views to see why.\n")

REVIEW_PREVIOUS_SECTION = ("- `previous/`: the merged knowledge you produced last round, with its REPORT.md. "
                           "Start from it and revise; write the complete new set to `out/`.\n")

CURATE_INSTRUCTION = """# Curate the next cross-task skill library ({base} -> next version)

Coding agents solve RoboDojo robot tasks (dual-arm ARX X5 in Isaac Sim) by submitting Python programs through
an `exec` tool whose include list loads helper files into the program namespace. Every new practice run
starts with a read-only copy of this shared library in `skills/shared/` and keeps its own task skills in
`skills/`. Your output replaces the library for all tasks trained from now on. You cannot run the robot.

{how_agents_act}

## Inputs (read-only)

- `current/`: the library every task currently starts from ({base}).
- `tasks/<task>/skills/`, `tasks/<task>/lessons/`: the frozen knowledge of the best run of each task so far.
  `lessons/shared_proposals.md` (if present) holds changes that task's agents proposed to the library, with
  evidence. `skills/shared/` inside a task is the copy it ran with; a difference from `current/` is a
  proposal too.
- `tasks/<task>/usage.json`: for that knowledge's practice runs and its tests: executions, Scores, and how
  many `exec` submissions included each `skills/...` file. This is the evidence of what is used and what works.
- `reference/api.md`: the program namespace the library builds on. Its functions are fixed; the library may
  call them, it may not redefine them. A helper that re-implements one of them (hand-built down quaternions,
  per-episode fingertip offsets, wrist aiming for a grasp) should be removed or rewritten on top of it, keeping
  only what the namespace does not cover (task-neutral grasp strategies, recovery, budgeting).

## Output (write only under `out/`)

1. `out/library/`: the complete next library, flat (no subfolders): `.py` helpers, a short `.md` per helper
   (parameters, preconditions, failure modes, the tasks and Scores that back it) and `README.md` indexing
   every helper. At most {max_files} files and {max_kb} KB.
2. `out/tests/test_*.py`: `unittest` tests of every helper's computations, offline: load a helper with
   `exec(Path(...).read_text(), namespace)` from `../library` (relative to the test file's folder) after
   putting stubs for the namespace functions into `namespace`. Python {python} with numpy and the standard
   library only. Run them yourself: `cd out && python -m unittest discover -s tests`. They are run again the
   same way after you stop.
3. `out/CHANGES.md`: for each helper: kept / changed / added / removed, why, and the evidence (task, run,
   Scores, usage counts, proposal).

## Rules

- Evidence decides. A helper or a change enters when a task reached a good Score using it, preferably two or
  more tasks. Remove helpers no task used or that the evidence shows failing, and say so in CHANGES.md.
- Task-neutral: no task names, object names, scene constants or one task's step sequence in library code
  (docs cite tasks only as evidence). Measured values become parameters. Sequences stay in the task's skills.
- Backward compatible: tasks keep including helpers by name. Keep names and signatures; add optional
  parameters whose defaults keep the old behaviour. For a breaking change add a new function instead.
- Library code is a robot snippet: only `numpy` and `math` imports; no name or attribute starting with `_`;
  no `open`, `eval`, `exec`, `getattr`, `setattr`, `globals`, `vars`, `dir`, `input`; no top-level robot
  actions (definitions and constants only); no redefinition of namespace functions.
"""

API_REFERENCE = """# KnowinAgent program namespace (reference for offline review)

## The `exec` tool
`exec(code: str, include: list[str] = [])` runs `code` in a persistent, restricted Python namespace (numpy as
`np`, `math`; no other imports, no names starting with `_`, no file access). `include` entries are loaded first,
in order: `skills/<name>.py`, `skills/shared/<name>.py`, `knowin_agent:<module>`. The receipt reports `ok`, `error`,
`stdout`, `stop_reason`, `interruption` (stalled motion, failed checkpoint, over budget), per-arm `tracks`,
`budget_panel` and `native_steps`; in practice also the native Score feedback.

## H4/H5-compatible functions (same names and parameters)
- `move_to(arm, position, quaternion=None, *, frame='grasp', yaw=None, pitch=None, roll=None, grip=None,
  max_steps=None, label=None, wait=True)`: one bounded motion; unreachable targets are refused before moving.
  `wait=False` returns a track id at once so the other arm can act; join with `wait(tracks=[...])`.
- `set_gripper(arm, opening)`, `check_reach(position, yaw=..., pitch=..., frame='grasp', arms=[...])`.
- `checkpoint(name, condition, evidence=None)`: a False condition stops the rest of the submission.
- `subgoal_done(name)`: records a finished subgoal and its native-action cost in the budget panel.
- `get_observation()`, `get_instruction()`, `step(action)` (one native action), `reset()` (practice only).
- `grasp_point(...)`, `pick(...)`, `place(...)`, `look_down(arm)`, `screw_down(...)`, `press_at(...)`,
  `release_yaw(...)`, `best_yaw(...)`.

## KnowinAgent extras
- `locate(text, view=None)`: segmentation + depth -> object with centre, top height, size, axis, grasp yaw.
- `inspect(...)`, `grasp_candidates(ref)`, `observe(arm, target, direction=..., distance=...)` (move a wrist
  camera to look; the free arm can watch the working arm).
- `run_skill(skill_id, **inputs)`: a library skill as one execution track with guards and a receipt.
- `wait(seconds=None, tracks=None)`; delegation of a fine alignment to the Local agent through `interact`.

Signatures above are a summary; the authoritative ones are those of the running program namespace.
"""
