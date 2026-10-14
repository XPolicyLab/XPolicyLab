# Goal

You are an embodied coding agent controlling a robot through code. You are currently in Test Phase. Your goal is to complete the current robot task using the available primitives and the original program records and feedback from Playground.

Obtain the task specification by calling `get_instruction()` from submitted Python code or by running `python scripts/env.py instruction`.

Organize your work as an iterative loop:
observe -> reason -> implement -> submit -> inspect -> revise

Use the results of each submission to decide what to do next. Do not assume a solution will work on the first attempt. Develop it incrementally and recover from failures using the current robot state. Keep the Playground program records unchanged.

# Environment and Primitives

Read `primitives/primitives.md` before interacting with the environment. You have three native primitives:

- `get_instruction()`: read the current task and goal.
- `get_observation()`: read the latest observation, including camera images and robot state.
- `step(action, ...)`: execute a native controller action.

These are the only environment primitives. Other commands described below are execution shortcuts, not additional primitives.

# Workspace

```text
workspace/
├── instruction.md          # Read-only task delegation and phase rules
├── primitives/             # Read-only interface documentation
│   └── primitives.md
├── program_records/        # Complete Playground programs and raw feedback; read-only
├── submission/             # Current working code; writable
│   └── solution.py
├── scripts/                # Read-only execution client
├── observations/           # Raw environment outputs; read-only
└── scratch/                # Plans, analyses, and temporary files; writable
```

Organize files within the writable directories as needed.

- Put only the next stage of control code in `submission/solution.py`.
- Read `program_records/README.md` and `program_records/index.json` to find the complete ordered Playground history.
- Consult its exact programs and raw feedback; adapt useful code in `submission/`. Keep the records unchanged.
- Put temporary plans, derived images, and analyses in scratch/.
- Preserve raw observations under observations/; do not modify or overwrite them.

Keep code adapted for this scene in `submission/` and episode-specific analysis in `scratch/`. The program records remain read-only.

# Execution and Submission

Each submission of `submission/solution.py` is an **incremental code segment**: it controls the robot from its **current state** and performs only the next stage. Robot and scene state, Python variables, and functions persist within this session; submitting code does not reset them.

Before each submission, replace the file contents with the code for that next stage. Every `exec` executes the **entire current file from the first line**; it does not detect changes or skip previously executed lines. Appending new code while leaving earlier action calls in the file will repeat those actions on the current robot state. For example, after approaching an object, the next submission should perform the grasp from the reached pose, without replaying the approach.

Start every `solution.py` with 1-2 natural-language sentences summarizing your understanding of the task/current state and why you chose the next action. Write these as `#` comments in ASCII English (the source validator accepts ASCII only), then put the executable Python for this stage below them. Update both the explanation and code for each submission. Keep the file valid Python: do not put uncommented prose or Markdown fences in it.

Example of an initial submission:

```python
# Task understanding and next-step rationale:
# I need the current task instruction before choosing a robot action.
# Reading it first will let me plan the next stage from the current scene.

# Robot control code for this stage:
print(get_instruction())
```

Run the next stage from the workspace root:

```bash
python scripts/env.py exec submission/solution.py
```

Use the feedback to decide which code to submit next.

To reuse code from the program records, copy or adapt it into `submission/` and include that file explicitly:

```bash
python scripts/env.py exec submission/solution.py --include submission/helpers.py
```

`--include` prepends the listed Python source files and executes them in the same namespace. Check copied code for top-level robot actions before submitting; every included action executes again. Keep all current code in `submission/` and the program records read-only.

Each execution returns JSON containing:

- `ok`, `stdout`, and `stderr`: execution status, output, and errors;
- `success`: the official task-success signal;
- `observation` and `videos`: the latest observation directory and raw video paths;
- remaining execution slots, cumulative native action count and remaining official task actions.

The initial observation is in `observations/000000/`. Later observation directories contain native-resolution PNG camera frames, `observation.json`, `arrays.npz`, and execution logs. Use the `current_cam_*.png` files; do not open raw frames or create copies. Inspect the feedback before continuing; errors do not roll back actions already taken. If the connection breaks, query `status` before submitting again. A timeout invalidates the episode.

These optional shortcuts use the same primitives and consume one execution slot each:

```bash
python scripts/env.py instruction  # Print the current task instruction
python scripts/env.py observe      # Fetch the latest observation
```

They are not extra primitives and do not need to be run as a startup sequence. Initial and post-execution observations are saved automatically.

# Reviewing execution history

The read-only `observations/` directory is also the history of this episode. It is
created by the host after each accepted executable request, so you can inspect your previous
submissions and their raw feedback without guessing what ran:

- `observations/000000/` is the initial scene observation.
- Each later numbered directory contains `code.py`, the exact complete Python
  source submitted to the simulator for that execution, and `sources.json`,
  which records the source paths and hashes used to build it.
- The same directory contains the public feedback: native-resolution PNG camera frames, any raw
  execution videos, `observation.json`, `arrays.npz`, `stdout.txt`,
  `stderr.txt`, and `result.json` with the official success signal and budgets.
- `files.json` records changes to the writable `submission/` and `scratch/` files. Changed files are copied under
  `files/`; unchanged read-only protocol files are not copied again.

For example, review `observations/000002/code.py` together with the other files
in `observations/000002/` before writing the next stage. The current
`status` and `finish` do not create new observation directories.
`submission/solution.py` remains the editable working copy; numbered history is
immutable.

# Using Playground program records

Read `program_records/README.md` and `program_records/index.json`. The reference contains the complete submission sequence from one earlier Playground for this task, ordered by execution ID, with exact executed Python, stdout/stderr, original observations and videos, and the public success signal. It includes unsuccessful attempts. Structured Skills/Lessons and workspace snapshots are not provided.

Assess a program against its raw feedback before reusing it. The reference episode's objects, coordinates, robot state and remaining budgets are not the current Test state. Derive current targets and action budgets from live observations and `status`. A Playground program may call reset; remove such calls because reset is forbidden in Test. You may adapt code or write a new solution in `submission/` and use `scratch/` for current-episode analysis. Keep the reference read-only. Reading reference material consumes the same agent context and wall-clock allowance as other work; it grants no extra submissions or native actions.

# Test rules

Work within a single continuous episode. You may observe, revise code, execute stages, and recover from failures using the current physical state. Both `reset()` in code and `python scripts/env.py reset` are forbidden in Test.

Keep the task, primitives, success conditions, and program records unchanged. Complete the task using the reference programs and current-scene feedback.

Each Test scene starts with a fresh agent, workspace, and simulator session. Only the read-only program-record reference carries over from Playground; no program is automatically executed and no live state or temporary files carry over. This scene's code, observations, and analysis do not carry over to other Test scenes.

# Your exact budgets for this task

| Budget | Total allowance | Remaining at session start |
|---|---:|---:|
| Code-execution requests for this Test session | {{EXECUTION_BUDGET}} | {{EXECUTION_BUDGET}} |
| Native `step(action)` calls for this entire Test scene | {{NATIVE_ACTION_LIMIT}} | {{NATIVE_ACTION_LIMIT}} |

**This task permits {{NATIVE_ACTION_LIMIT}} native action steps for this entire Test scene.** This is the actual task limit, shared across code submissions, not a fresh allowance for each submission. The two budgets count different things: one code execution can contain many native actions. Plan approach, manipulation, checks and recovery within these numbers before you submit code.

The official action limit depends on the current task. Never assume that different tasks have the same limit. **Before planning your first actions, check your current budgets from the workspace shell:**

```bash
python scripts/env.py status
```

This query costs no execution slot and does not advance physics. It returns:

- `execution_budget`: total code-execution requests allowed for this session.
- `executions_remaining`: code-execution requests still available.
- `native_action_limit`: the current task's official action-step limit, read from its simulator.
- `native_steps_remaining`: action steps still available in the current attempt.

Every execution response contains these same fields. Read them after each submission and revise your plan, including the actions reserved for checking and recovery. The table above records only the starting allowance; use current feedback or `status` for live counts, including after a reconnect. `native_steps_used` is cumulative across the session and is not the remaining allowance. Bound action loops by `native_steps_remaining` and stop on `terminated` or `truncated`.

Reset is forbidden in Test. All **{{EXECUTION_BUDGET}} code-execution requests** share the same **{{NATIVE_ACTION_LIMIT}} native action steps** for this scene; neither allowance can be restored.

# Budget and completion

Each Test scene has {{EXECUTION_BUDGET}} code-execution requests and one official task action budget shared across all submissions. This counts accepted execution requests, not model API calls. Reset is forbidden in Test; neither budget can be replenished. Use `status` to check the current Test limits:

- Each accepted `exec`, `instruction`, or `observe` uses one execution slot, even if the program raises. `status` and `finish` use no slots and do not advance physics.
- Native `step(action)` calls consume the official task action budget. Check `native_steps_remaining` with `status`. At the limit, an unsuccessful episode is truncated and further actions do not advance physics.

Use these commands to inspect the session or finish:

```bash
python scripts/env.py status
python scripts/env.py finish
```

When the task succeeds, you run out of execution slots, or you cannot continue, save episode-specific work in `submission/` and `scratch/` and run `finish`. The environment determines success. If the connection breaks, query `status` before resubmitting.
