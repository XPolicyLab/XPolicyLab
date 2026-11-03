# Shape of a program

One worked skeleton, so the contract has something concrete next to it. Nothing here
is task-specific.

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...lib.geometry import camera
from ...lib.motion import plan as m
from ...runtime.session import SwitchReason
from .support import <task>_see as S
from .support import pb_kin as K

DATA_DIR = Path(__file__).resolve().parent / "data"

STEP_LIMIT = 900          # from the task's own budget, not chosen to fit a run
RESERVE = 60              # the steps a clean ending needs


def run(ctx):
    try:
        target = S.locate(ctx)            # measure; never assume a position
        if target is None:
            raise m.PlanError("no candidate in view")

        for arm, xy in plan_hands(ctx, target):
            yield from reach_and_grasp(ctx, arm, xy)
            yield from carry_and_place(ctx, arm, xy)

    except m.PlanError as exc:
        # The one legal exit that is not a normal finish. The checkpoint below is
        # where the learned branch picks up, so it must be a clean pose.
        yield from abandoned(ctx, "PLAN_EXHAUSTED", detail=exc, source="<task>.plan_exhausted")
    finally:
        yield from m.home(ctx, ("left", "right"), reserve=RESERVE)
```

## Why it is shaped this way

**`run` is a generator.** Every `yield from` is a sequence of control steps, and the
helpers in `gm/lib/motion` are generators for the same reason. A helper that returns a
target rather than yielding one is a computation; a helper that moves the arm is a
generator.

**Measurement comes before motion, and again after contact.** `S.locate` reads the
images. Anything it returns is a hypothesis with an error bar, not a coordinate. The
second look, after the arm has moved, is the one that matters — a pose measured
through a different viewpoint is the only check on the first.

**Failures raise `m.PlanError`.** That is the type the motion layer already raises, so
a program does not need a second error vocabulary. A generic exception escaping `run`
ends the episode without a hand-over, which is correct for an infrastructure fault
and wrong for an exhausted plan.

**The hand-over is the last thing in the failure path**, and its source string is
registered for that task. See [contract.md](contract.md).

**The reserve is subtracted, never checked afterwards.** A program that discovers at
step 890 that it has 10 steps left cannot go home. Every long loop should test
`ctx.step` against `STEP_LIMIT - RESERVE` before it starts a new attempt.

## What is deliberately absent

- No global state between episodes: the runtime builds a fresh context per episode,
  and a module-level cache that survives that is a bug.
- No branch on the task's own name. A program may read `ctx.instruction`; it may not
  read which task it is, because the program *is* the task.
- No writes outside `ctx.out`, and no reads of anything under `evidence/`: a program
  runs in deployment, where that folder does not exist.
- No timing-dependent behaviour. The deadline is the controller's business; a program
  that measures wall-clock time to decide something cannot be compared across runs.

## The step budget

`STEP_LIMIT` is a property of the task — how many control steps the environment grants
— not a tuning knob. Copy it from the task, and record where it came from. If a
program ends early, that is a program ending, not a budget reduction; if it ends at
the limit with no motion, the plan failed and the hand-over path is what runs.
