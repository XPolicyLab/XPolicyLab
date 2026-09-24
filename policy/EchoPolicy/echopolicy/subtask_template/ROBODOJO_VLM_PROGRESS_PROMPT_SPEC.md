# RoboDojo Subgoal Progress

Inspect the current camera images and choose exactly one decision for the
current plan.

Use the last actually executed 10-step EE trajectory to help interpret what
happened since the previous observation. Treat it as commanded-motion evidence,
not proof of physical contact or task completion.

Use the previous check memory together with the current observation to produce
two outputs:

- `before_to_now_summary`: a concise, factual description of what changed from
  the BEFORE views to the NOW views. Do not omit this field when BEFORE views
  are supplied; if no reliable change is visible, state that explicitly.
- `memory`: a concise, self-contained replacement for the previous memory.
  Preserve still-relevant task facts and failed attempts, add the verified
  before-to-now change and current result, and remove only details directly
  contradicted by current evidence. Always retain task-relevant progress:
  observed requirements, exact completed and remaining quantities, current
  phase, object identities and states, confirmed outcomes, execution order,
  and relevant failed attempts. Never replace a known number or ordered fact
  with vague wording, reset accumulated progress, or declare a multi-step
  requirement complete after only one step.

When BEFORE views are supplied, compare BEFORE to NOW to verify object motion,
grasp, lift, and release. Views within each time group are main, left wrist,
then right wrist when available. NOW determines the current completion state;
historical success alone does not prove that the object is still held.

- `stay`: the current subgoal and its target remain correct but are incomplete
  or uncertain.
- `advance`: the current subgoal is visibly complete and execution should move
  to the next listed subgoal.
- `replan`: the active subgoal, object identity, target point, or
  remaining decomposition is visibly wrong for the current scene.

Do not use a code-defined special completion rule for `pick` or any other
atomic action. Decide `stay`, `advance`, or `replan` from the current images,
the ordered EE motion, the current subgoal, and the accumulated memory. Do not
treat commanded motion alone as proof of a physical result.

Use `replan` for an actual planning misjudgment, not merely slow execution.
When a last advanced subgoal is supplied, verify that its requested result is
still visibly satisfied. If that advancement was incorrect, choose `replan`.
For an active `observe` subgoal, use `stay` while the requested observation is
incomplete, `advance` when it is complete and the existing next subgoal remains
valid, and `replan` when the observed information changes object identity,
execution order or target coordinates. Arm choice is non-binding and may change
when another arm can continue the task more effectively.
Provide a short reason identifying what must be corrected. Do not select a
replacement target or rewrite the plan in this response.

Return JSON only:

```json
{
  "decision": "stay",
  "reason": "The current subgoal is not visibly complete.",
  "before_to_now_summary": "The gripper moved closer to the red block, but contact is not visually confirmed.",
  "memory": "The red block is still on the table; the latest grasp attempt did not secure it."
}
```
