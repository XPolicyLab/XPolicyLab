# Contract

What a task program is given, what it may do, and where its authority stops. Read
this before writing one; everything in [stages.md](stages.md) assumes it.

## What the program sees

The runtime hands each program a context with five accessors and nothing else:

| Accessor | Returns | Where it comes from |
| --- | --- | --- |
| `ctx.q(arm)` | the arm's joint angles | the observation the policy server already decoded |
| `ctx.g(arm)` | the gripper reading, one scalar | same |
| `ctx.ee(arm)` | the end-effector pose | same |
| `ctx.rgb(cam)` | one camera image as an RGB array | same |
| `ctx.instruction` | the task instruction text | same |

`arm` is `left` or `right`; `cam` is `cam_head`, `cam_left_wrist` or `cam_right_wrist`.
There is no depth channel, no object pose, no scene graph and no reward.

Two more accessors exist for records, not for decisions: `ctx.log(...)` writes a
line into the episode tape and `ctx.snap(...)` saves a frame. Use them to make a
failure explainable after the fact. A program must not branch on anything the log
returned.

## What the program produces

A program is a generator. Each `yield` is one control step: the runtime takes the
action the program has written into `ctx.cmd_q` / `ctx.cmd_g`, sends it to the
simulator, and resumes the program with fresh observations.

The helpers in `gm/lib/motion` wrap the common cases (`move`, `goto`, `grip`,
`hold`, `move_joints`), and each of them is itself a generator you delegate to with
`yield from`. Writing raw joint targets is allowed and occasionally necessary.

## Giving up: the one thing a program may decide alone

A program may hand the rest of the episode to the learned branch by yielding a
request:

```python
yield ctx.switch_to_wam(
    SwitchReason.PLAN_EXHAUSTED,
    detail="both approaches to the lid exhausted the caller's retries",
    source="<task>.<a name registered for that task>",
)
```

Three properties make this safe, and all three are enforced:

1. **The `source` string must be registered for that task.** The table lives in
   `gm/runtime/switch_sources.py`. An unregistered string raises. This is a typo
   guard and a review handle: every place a program can give up is enumerable by
   reading one file.
2. **The yield point is a checkpoint, not a state.** The runtime holds the current
   joint and gripper commands and hands that state to the learned branch. A program
   must therefore yield *before* it opens a gripper, goes home or otherwise leaves
   the arm somewhere it would not want to resume. The runtime refuses a request whose
   checkpoint is not clean.
3. **It is one-way and per-episode.** Once an environment has handed over it stays on
   the learned branch until the episode ends. A reset clears it. There is no path
   back to code mid-episode, which means a program cannot use the learned branch as a
   retry inside its own plan.

Giving up is a statement about the program — it has run out of plan, retries or
budget — not a claim about the task. Nothing in the runtime records whether the
episode then succeeded, and a program never sees that afterwards.

## What a program must never read

- **No reward, no success flag, no score.** The runtime does not expose them. A
  program that tries to infer them from a log line is reading its own bookkeeping.
- **No layout identifier, no seed number, no scene name.** A program receives a task
  name and an instruction. It does not receive which arrangement of the scene it is
  looking at.
- **No file that was produced by the simulator.** Data files under
  `gm/programs/data/` hold shapes, colours and geometry that a program needs; they
  do not hold scene layouts or object placements. Anything in a program that names
  an arrangement rather than a property is wrong even when it happens to work.

## The constants a program carries

Every module-level constant falls into one of three classes. The class decides what
you owe a reader.

| Class | Examples | What it needs |
| --- | --- | --- |
| Environment description | robot link lengths, camera intrinsics, table height, gripper geometry | Nothing beyond its source, which is the device. These are not task-derived. |
| Measured property | a cup's radius, a block's edge, a garment's fold line, a template scale | An entry in `evidence/` showing it recomputes from recorded robot data within the consumer's tolerance. |
| Budget and reserve | a step limit, a hold-check reserve, a retry count | A reason in a comment and, if it moves a score, a paired comparison behind it. |

A constant with no class is a defect. This is the single rule that the whole
`evidence/` folder exists to satisfy, and the reason the third phase in the overview
is not optional.

## Division of labour with the learned branch

The adapter runs a code program first and can hand over to a learned model. That
model is one component of this system, not the system's fallback for everything: a
program that gives up immediately is legal but it is a program that contributes
nothing, and the documentation has to say so plainly rather than describe it as a
choice between two policies.

Where the hand-over happens is a property of the program: it is the point at which
the program has no further plan. It is not a property of the task's name, and the
code must not become a table of which tasks get which branch.
