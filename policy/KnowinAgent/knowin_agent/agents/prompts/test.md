## Test: turn the frozen experience into this scene's plan

This is a test episode. `skills/` and `lessons/` hold knowledge frozen from practice; they are read-only, and
`reset()` is not available. There is no score feedback.

- Start from `lessons/PLAN.md` and the skills it names, then write this scene's plan (Principles, section 1) by
  mapping each of its steps onto the objects measured here: which object, which arm, where to, and the action
  estimate. Steps whose conditions do not hold here (nothing is in the way, an object is already in place) are
  left out of the plan with the reason.
- Numbers in skills and lessons (positions, heights, yaws, action counts, orders) come from other scenes.
  Re-measure targets, container geometry, object headings, clearances and category assignments here and pass
  them to the skills. Include skills as they are: `exec(code, include=['skills/<name>.py'])`.
- When a skill's preconditions do not hold here, adapt it inside your program (define a modified function; it
  stays in the namespace), not in the frozen files. Do not force the old trajectory.
- Reuse a calibration only when its robot, orientation and contact assumptions match.
- Assign arms by reach here (`check_reach`), even where the lessons name a side.
- There is one attempt: before an irreversible step (pouring, releasing over a target, closing onto placed
  parts), measure where the stream or part will land and keep it clearly inside the target, as the
  Principles say.
- Judge each sub-goal from the images and measurements, using the signs the lessons describe.
- Before `finish`, check every condition of the tier you claim, one by one, from the scoring section and the
  Verification column of PLAN.md (measure; do not rely on how the last motion looked). If one fails and native
  actions remain, fix it first; `finish` reports `completed` only when every condition was checked and held.
- Success ends the episode by itself. If every step of the plan is done (arms back where the scoring wants
  them) and the episode is still running, at least one condition is false: find it by measurement and fix it
  while actions remain. `finish(outcome='completed')` is refused while the episode runs.
- A measurement within its uncertainty of a threshold (head camera about 1 cm, wrist camera about 3 mm) is not
  a pass: move it clearly inside before finishing.
