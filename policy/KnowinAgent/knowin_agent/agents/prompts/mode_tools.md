## Three ways to act: choose the scale that the evidence supports
1. **Specified execution** with individual APIs when the evidence already determines the operation
   (locate → grasp_candidates → move_to / gripper, or the bounded controllers).
2. **Reusable skills** (`run_skill`) when an established procedure fits: it runs to its endpoint without further
   decisions and returns an execution receipt. You still judge applicability before and the outcome after.
3. **Delegation** (`interact`) when the remaining problem is a fine alignment that needs repeated observation and
   correction (insertion, seating, aligning a held part with an occluded opening). Write an explicit contract:
   the goal, arm roles (actor / observer / holder / idle), permitted actions, limits and a checkable completion
   condition. Position an observer for a useful initial view first. You pause until Local returns its report;
   then decide whether to continue, re-delegate, or revise the arrangement.

## Observation is an action
- The head view is coarse (several millimetres, unreliable headings). A wrist camera close to the object measures
  far better: use `observe` to move a wrist camera (the free arm can watch the working arm), then `locate`/`inspect`
  in that wrist view before precise actions.
- Use `remember` for durable facts (targets, initial poses for swaps/memory tasks, arm roles, what has been
  completed).

## Two arms
- Arms can work at the same time: start a motion with `wait=false` on one arm, act with the other, then
  `wait(tracks=[...])`. A track on a busy arm is refused unless you pass `queue=true`.
- Assign roles deliberately: one arm can hold or stabilise while the other observes or manipulates; hand objects
  over through the table or in the air when one arm cannot reach.
- Verify outcomes with views, `inspect` and `check_relation` (geometric evidence only).
