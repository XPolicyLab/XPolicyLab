## Principles: plan the scene, act, check

The score comes only from task progress in the simulator, and only actions that move the arms spend the native
action budget. Programs are limited too (`executions_used` / `executions_limit` in every receipt): up to two
programs in a row that move nothing (only measure or check) are not counted (`measure_only_uncounted`); from the
third on they are. Each program also costs time to write. So when a measurement only sets the numbers of the next
motion, measure and move in the same program (guard the motion with `checkpoint`); keep a separate measuring
program for choices you must look at first. Compare candidate poses in one program (a loop over yaws, pitches or
arms with `check_reach`, taking the first reachable), not one program per candidate.
Most failed attempts did not lack precision: they ran out of actions or programs, worked in the wrong order, or
knocked into something that should have been moved first. So plan
the whole scene before moving, keep the plan in view, and spend actions only on what the plan needs.

### 1. Before the first motion: write the scene plan
1. Read the task scoring section: the exact rule for every tier, including open grippers and the return to the
   start pose. Plan for the tiers. Where the recipe or the lessons disagree with it, the scoring wins. Keep the
   steps a scoring condition needs, including those the physics needs to reach it (a nut reaches its depth only
   by being screwed down).
2. Read `lessons/PLAN.md` and the skills it names. They are a method from other scenes; their numbers are not
   this scene's.
3. Measure the scene, free of native actions: `locate` every object the task uses (head camera), and note each
   one's position, footprint (length and width, not only its centre), heading and height.
4. Save the plan with `remember('plan', ...)`, at most about 15 lines:
   - **Layout**: where each object ends up (build site, slots, container) and why that place is clear: nothing
     inside the site's footprint or where a grasp or placement will put the fingers. Moving a piece costs
     40-100 native actions, so move only what actually blocks: something on the site, or on a path that a
     `check_reach` / `path_blocked` result or a lower approach cannot avoid. Each such piece gets its own first
     sub-goal: move it out of the way. When the task builds or stacks
     (towers, stacks, nested or assembled parts), choose the build site first, in reach of both arms (usually
     the open middle of the table), clear it of every piece that is not its first layer, then build in the
     order the scoring checks the layers, bottom up, checking each layer before the next.
   - **Sub-goals in order**, one line each: arm(s), object, from → to, final heading, how you will check it
     worked, and the native actions it should take (the price list in the Program API: `pick` about 20, `place`
     about 20 plus the carry). Order them so that earlier placements do not block later ones, and so the
     sub-goals that score come first.
   - **Arms**: which arm does what, chosen by reach in this scene (`check_reach` on each sub-goal's grasp and
     target poses), not by the left/right roles of a lesson: the same object can sit on the other side here.
     Which sub-goals the two arms can do at the same time (`both(...)`).
   - **Budget**: the sum of the estimates plus the way back must fit `usable_native_steps` in the budget panel,
     with about 10 % to spare. If it does not, drop or postpone the sub-goals worth least.

### 2. Executing
Motion only through `exec`. The task tools (`locate`, `inspect`, `check_reach`, `remember`) do not move the arms.
1. One sub-goal per program until its motion is seen to work; then run whole sub-goals, each guarded by
   `checkpoint(...)` (a failed guard stops the rest of the program, nothing after it moves). Mark each finished
   sub-goal with `subgoal_done(name)`.
2. After each program read `ok`, `stop_reason`, `error.line` / `interruption`, `stdout`, the images and the budget
   panel. Compare what the sub-goal cost with its estimate.
3. Check every placement before building on it: look at the images (and re-measure) after release.
4. A retry must change something based on what you saw: the target or offset, the grasp face or yaw, the arm, the
   view. A move that stalled or returned `path_blocked` is refused if repeated unchanged.

### 3. When to re-plan
Update the `plan` note, and only then continue, when:
- a sub-goal failed twice;
- a sub-goal cost more than 1.5 times its estimate, or the remaining plan no longer fits the budget;
- an object moved, fell or was not where the plan assumed;
- a precondition of the next sub-goal turned out false.
Otherwise follow the plan; do not improvise new sub-goals between programs.

### 4. Budget
An attempt that runs out of native actions scores only the tiers already held. `usable_native_steps` is what is
left after the estimated way back and a 10-action reserve; `subgoals.more_affordable` says how many more
sub-goals of the size so far still fit. A move that would leave no actions for the way back returns
`over_budget` without moving; pass `final=True` only for the last scoring move. When an object is knocked out of
reach or tipped over, first finish what still scores, then return to it if actions remain. While actions remain
and a scoring sub-goal is still within reach, keep going; finishing early forfeits the score.

### 5. Manipulation rules
- Grasp and place with the preloaded library (`pick`, `place`, `press_at`, `screw_down`; pouring with
  `rim_transfer`). They move the grasp point (`frame='grasp'`), take `grasp_yaw` and the footprint from
  `locate`'s `object_pose`, approach from straight above and stop on contact. Decide each object's final
  heading before grasping it.
- Head-camera positions are about 1 cm off. For placing or inserting within a few millimetres, re-measure the
  target from the acting arm's wrist camera 15-25 cm above it (`look_down`), correct for where the part sits in
  the gripper, and descend with `precision=True, stop_on_contact=True`. When the last millimetres need repeated
  look-and-correct, delegate them (`interact`) instead of guessing in code. Delegate only a short, checkable
  correction (its contract names the measurement that says it is done) with `max_rounds` sized to it; when a
  delegation of the same kind already came back without success, change the approach in code instead of
  delegating it again (in 10-31 practice most delegations ended unresolved and still spent their actions).
- Irreversible steps (pouring, releasing a part over a target, closing a lid onto placed parts, a final push)
  cannot be taken back, and a test has no reset. Before one, measure what will happen, not only where things
  are: where the stream or the released part will land relative to the target, with the margin left to the
  target's edge, and correct the pose until that prediction is clearly inside. Tilt or open in a small first
  increment, look, then continue. Never let the first attempt at an irreversible step be the calibration.
- Two arms work at the same time; the cost is the longer motion, not the sum. Keep their paths apart.
- Check `check_reach` before committing an arm to a pose; of two reachable choices take the one with less joint
  travel.

### 6. Finishing
When the scoring needs it, bring both arms back to the start pose before the native actions run out
(`budget_panel.at_origin_for_scoring` says whether the current pose already counts), then `finish`.
Once a tier's conditions hold, do not run an action that can disturb them (pressing, pushing or closing
something onto placed parts, sweeping an arm past them) unless a higher tier needs it; if it does, look at the
lower tier's conditions again right after it, before anything else.
