# Simulator session client

Run commands from the workspace root. No copy or marker file submits code.

```bash
python scripts/env.py status
python scripts/env.py instruction
python scripts/env.py observe
python scripts/env.py exec submission/solution.py
python scripts/env.py exec submission/solution.py --include skills/helper.py
python scripts/env.py reset     # Playground only
python scripts/env.py finish
```

`exec` reads the listed files once, prepends `--include` files in command-line order,
and sends their combined Python source to the host's `env.step(code)`. Includes
execute in the same persistent namespace as the entry file; they are not Python
module imports. Keep reusable function definitions free of unintended top-level
robot actions. Only `.py` files under `submission/` or `skills/` may be submitted.
The host archives the exact submitted bytes in the read-only
`observations/<execution>/code.py` history. Edits after submission apply to the
next call.

Robot state, variables, functions and numerical arrays persist between executions.
Only `numpy` (the numerical subset) and `math` imports are supported in submitted
code. Source must fit the environment's 8 MiB ASCII code space. Submitted code
cannot read workspace or host files; use workspace tools for local file and image
analysis, then submit the resulting numerical logic. `--include` is the explicit
way to load helper source. Workspace Python has its own separate process state.

`instruction`, `observe`, and `reset` are convenience requests executing
`print(get_instruction())`, `get_observation()`, and `reset()` respectively through
that same `env.step(code)` boundary. They are not additional robot primitives.
Each accepted execution, including these convenience requests and programs that
raise, uses one execution slot. For RoboDojo, native `step(action, ...)`
also consumes the benchmark's native action budget. `native_steps_used` counts
consumed actions and `native_steps_remaining` reports the remaining allowance.
For RoboDojo, Playground reset renews the official native action allowance while
preserving the session's total execution-request count and elapsed time. Its
`native_steps_used` remains cumulative across resets. Other benchmarks keep their
existing action-budget rules. Test reset is forbidden. `status` and `finish` do not use slots or
advance physics. Invalid transport requests and a Test `reset` command are rejected
before execution. A Test program that calls `reset()` raises and uses a slot.

The JSON response contains `ok`, `stdout`, `stderr`, official `success`, budget
counters, and paths relative to the workspace. Every execution response and `status`
include `execution_budget` (total allowed requests), `executions_remaining`,
`native_action_limit` (the selected task's action limit, read from its simulator),
and `native_steps_remaining` (the current attempt's remaining actions).
These are distinct budgets; plan with the current values rather than assuming a
fixed action limit across tasks. Playground reset renews RoboDojo's remaining
actions but does not change either total limit or restore used execution requests.
`observation` points to a folder
under `observations/` containing native-resolution PNG camera frames, `observation.json`,
and `arrays.npz`. Entries with `npz_key` in the JSON refer to arrays in that
NPZ; load them with `numpy.load(path, allow_pickle=False)`. Executions also
save `result.json`, logs, and raw camera videos there. The same native-resolution PNGs are also retained in the host run's
`raw_observations/` directory for replay and audit.
Initial observations are in `observations/000000/`.
`status` reports the latest completed observation even after execution slots run out.
JSON stdout/stderr fields contain at most the last 256 KiB each; the observation
directory keeps the full output. Private outer rewards and the ledger are not
returned.

Simulation advances only during native control calls. Agent thinking and editing
consume wall time, but no physics steps. Reset restores the same scene; execution
slots, native action usage, wall time and Python variables remain consumed/present.
Test forbids reset, including calls through aliases or helper functions.

Calls are synchronous and serialized. Do not submit concurrently. Errors can occur
after physical actions; nothing is rolled back. Inspect the result before retrying.
On a connection failure, do not automatically resubmit: execution may already have
occurred. Query `status` first. A program timeout ends the episode as invalid and
never starts a replacement scene. `finish` closes the session; save reusable work
before calling it. Ending the agent also ends its simulator session.
