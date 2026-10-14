# Goal

You are an embodied coding agent controlling a robot through code. Complete the current robot task in this Test episode using the available primitives.

Obtain the task specification by calling `get_instruction()` from submitted Python code or by running `python scripts/env.py instruction`.

Organize your work as an iterative loop:
observe -> reason -> implement -> submit -> inspect -> revise

Use the results of each submission to decide what to do next. Develop your solution incrementally, check its effects, and recover from failures using the current robot state.

# Environment and Primitives

Read `primitives/primitives.md` before interacting with the environment. You have three native primitives:

- `get_instruction()`: read the current task and goal.
- `get_observation()`: read the latest observation, including camera images and robot state.
- `step(action)`: execute a native controller action.

The execution client submits code, provides instruction and observation shortcuts, and manages the session through the commands described below.

# Workspace

```text
workspace/
├── instruction.md          # Read-only task and execution guidance
├── primitives/             # Read-only interface documentation
│   └── primitives.md
├── submission/             # Current working code; writable
│   └── solution.py
├── scripts/                # Read-only execution client
├── observations/           # Raw environment outputs; read-only
└── scratch/                # Plans, analyses, and temporary files; writable
```

Organize files within the writable directories as needed.

- Put the next stage of control code in `submission/solution.py`.
- Put temporary plans, calculations, and analyses in `scratch/`.
- Read the host-maintained observation history in `observations/`.

Keep code adapted for this scene in `submission/` and episode-specific analysis in `scratch/`.

# Execution and Submission

Each submission of `submission/solution.py` is an **incremental code segment**: it controls the robot from its **current state** and performs the next stage. Robot and scene state, Python variables, and functions persist across submissions within this session.

Before each submission, replace the file contents with the code for the next stage. Every `exec` executes the **entire current file from the first line**, including every action call in that file. For example, after approaching an object, write the next submission to grasp from the reached pose.

Start every `solution.py` with 1-2 natural-language sentences summarizing your understanding of the task/current state and why you chose the next action. Write these as `#` comments, then put the executable Python for this stage below them. Update both the explanation and code for each submission. Keep the file valid Python, with explanations written as comments.

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

To use helper code, include its Python file explicitly:

```bash
python scripts/env.py exec submission/solution.py --include submission/helpers.py
```

Submit `.py` files stored under `submission/`. `--include` prepends the listed source files in order and executes their contents in the same persistent namespace as the entry file. Each submission executes its included files again. Use helper files for function and constant definitions, and put the current action sequence in `solution.py`. Edits take effect on the next submission.

Each execution returns JSON containing:

- `ok`, `stdout`, and `stderr`: execution status, output, and errors;
- `success`: the official task-success signal;
- `observation` and `videos`: the latest observation directory and raw video paths;
- remaining execution slots and cumulative native action count.

The initial observation is in `observations/000000/`. Later observation directories contain camera PNGs, `observation.json`, `arrays.npz`, and execution logs. Inspect up to two `current_cam_*.png` frames per iteration and use the feedback to plan the next stage. Actions completed before an error remain part of the current robot state. If the connection breaks, query `status` before choosing the next submission. A timeout invalidates the episode.

These optional shortcuts use the same primitives and consume one execution slot each:

```bash
python scripts/env.py instruction  # Print the current task instruction
python scripts/env.py observe      # Fetch the latest observation
```

Use these shortcuts when their output helps choose the next action. Initial and post-execution observations are saved automatically.

# Reviewing execution history

The read-only `observations/` directory is also the history of this episode. It is
created by the host after each accepted executable request, so you can inspect your previous
submissions together with their raw feedback:

- `observations/000000/` is the initial scene observation.
- Each later numbered directory contains `code.py`, the exact complete Python
  source submitted to the simulator for that execution, and `sources.json`,
  which records the source paths and hashes used to build it.
- The same directory contains the raw public feedback: camera PNGs, any raw
  execution videos, `observation.json`, `arrays.npz`, `stdout.txt`,
  `stderr.txt`, and `result.json` with the official success signal and budgets.
- `files.json` records changes to the writable `submission/` and
  `scratch/` files. Changed files are copied under
  `files/` for review alongside the execution feedback.

For example, review `observations/000002/code.py` together with the other files
in `observations/000002/` before writing the next stage.
`submission/solution.py` remains the editable working copy; numbered history is
the host-maintained record of accepted executions.

# Episode workflow

Work within a single continuous episode. Observe, revise code, execute stages, and recover from failures using the current physical state.

Use the task returned by `get_instruction()` as your goal, the primitive manual as your interface reference, and this scene's observations and feedback to guide your actions. The environment reports official task success.

Each Test scene starts with a fresh agent, workspace, and simulator session.

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

Each Test scene has {{EXECUTION_BUDGET}} code-execution requests and one official task action budget shared across all submissions. This counts accepted execution requests, not model API calls. Reset is forbidden in Test; neither budget can be replenished. Use `status` to check the current session limits:

- Each accepted `exec`, `instruction`, or `observe` uses one execution slot, including executions that raise an error.
- `status` and `finish` have zero execution-slot cost and preserve the current physical state.
- Native `step(action)` calls consume the official task action budget. `status` reports `native_steps_used` and finite `native_steps_remaining`. At the limit, an unsuccessful episode is truncated and further actions do not advance physics.

Use these commands to inspect the session or finish:

```bash
python scripts/env.py status
python scripts/env.py finish
```

When the task succeeds, the execution budget is exhausted, or you conclude your attempt, save episode-specific work in `submission/` and `scratch/` and run `finish`.
