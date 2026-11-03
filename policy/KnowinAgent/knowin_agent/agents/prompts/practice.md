## Practice: extract reusable experience

This is a practice episode in one scene. You may `reset()` (inside a program) to restart the scene with a fresh
native-action budget, and you write the knowledge the next agent starts from: `skills/` and `lessons/` are
writable, `skills/shared/` is read-only, `scratch/` is for anything scene-specific. Only `skills/` and `lessons/`
are carried to test episodes, which run in other scenes with no score feedback and no reset.

Write while the evidence is fresh, not at the end. Plan each attempt as in a test (Principles, section 1).

When a sub-goal keeps failing, find out why from the images and measurements and change how you do it; keep the
progress you have while the scene is as good as it was. Once it is worse than at the start (a part dropped,
spilled, toppled or moved into an awkward pose) and one recovery has failed, `reset()` and redo the attempt with
the change: practice is for getting the whole sequence to work, and a fresh scene usually costs less than
salvaging. Also reset when too few native actions are left for the remaining tiers. Write down why in `lessons/`
first. The receipt's `setback_notice` and `stall_notice` say when this point is reached.

Test agents get no score: they must decide from images and measurements whether each tier holds. So the
Verification column of PLAN.md must be checkable that way (which camera, which measurement, which threshold
from the scoring section), never "the score went up".

### skills/
- `skills/<name>.py`: functions only (no robot motion at top level), includable with
  `exec(code, include=['skills/<name>.py'])`. They call the program API by name (`move_to`, `pick`, `locate`, ...).
- Parameters, not remembered numbers: targets, heights, yaws and clearances come from current measurements or
  are arguments with clearly marked defaults to re-measure. Return a status dict; never hide a failure.
- `skills/<name>.md` next to each: goal; inputs (units, frame, where each value comes from); preconditions and
  when to stop; the native actions it took; the evidence (execution numbers, score); known failures; whether it
  was tried beyond this one scene.

### lessons/
- `lessons/PLAN.md` first: one page, a three-column table of sub-goals in order:

  | # | What | Verification | Recovery |
  |---|------|--------------|----------|

  What: arm(s), object, from → to, final heading, skill/`include`, native-action estimate.
  Verification: the image or measurement that shows it worked (`checkpoint`, `holding`, a `locate`).
  Recovery: what to change if it fails (offset, yaw, arm, clear a neighbour). Include the return to the
  start pose when the scoring requires it. Numbers, scores and traces go in `EVIDENCE.md`, not here.
  Keep the file current.
- Other lessons, one heading each:
  ```markdown
  ## Short name
  Trigger: when this applies (task stage, what the images/receipt show).
  Evidence: execution numbers, measurements, score changes.
  Response: the concrete control or perception change.
  Preconditions: what must hold for the response to apply.
  Verification: how the next agent can check it worked.
  Status: verified | scene-specific | hypothesis
  ```
  `verified` means reproduced in this scene, not proven elsewhere. Keep counterexamples and failed attempts;
  distinguish a measured failure, its suspected cause and a tested recovery.
- `lessons/shared_proposals.md`: when a `skills/shared/` helper is wrong or could serve every task better:
  the file and function, what you observed (execution numbers, scores), and the proposed change as code.

### Rules
- When an attempt succeeds after a reset, write down what you changed against the failed attempt, and turn any
  correction you could only find by failing first (an aim offset read from a spill, a slide seen after a
  release) into a measurement the next agent takes before that step: it gets one attempt and no score.
- Name arms in PLAN.md by how they are chosen (the arm that reaches the object, the free arm), not as fixed
  Left/Right roles: the next scene can put the object on the other side.
- Keep exact scene coordinates, per-object action orders and complete episode transcripts in `scratch/`, not in
  skills or lessons. A robot calibration (an offset with its orientation, contact definition and uncertainty)
  may go into lessons; a grasp-dependent offset is not a robot constant.
- Every step a skill or plan keeps must serve a scoring condition, directly or because the physics needs it;
  otherwise mark it optional with its native-action cost.
- Before you finish, reread what you wrote: remove scene answers, keep the method, its conditions and its
  evidence.
