# RoboDojo Task Replanning

The current plan has been judged incorrect for the current visual scene.
Generate a corrected complete sequence for all work that remains.

Use the current images as authoritative. Correct mistaken object identity,
subgoal ordering, and target points. Do not include work that is
already visibly complete.
Use the accumulated check memory to avoid repeating failed attempts and to
preserve task-relevant facts that remain consistent with the current images.
Use the supplied official task description and full-score condition to retain
every still-incomplete required action. Do not drop repeated confirmations,
final placements, gripper release, or return-to-origin steps during replanning.

For every returned subgoal:

- `text` must be one direct action instruction for the VLA.
- Each subgoal must contain exactly one atomic action. Never combine `pick`
  with `place`, `insert`, or another later action in the same subgoal.
- `atomic_action` must be one canonical action name. Use `observe` when the
  robot should inspect or wait for visible scene information before acting.
- `arm` is required for every subgoal and must be either `left` or `right`.
  Choose the primary arm that should execute the subgoal from the current
  images, target geometry, reachability, and the preceding subgoals. This is a
  non-binding planning hint for downstream trajectory selection, not a hard
  execution constraint; the selector may use the other arm when its observed
  motion is more appropriate. Do not choose an arm merely to balance work
  between the two arms.
- `coordinates` is optional. Include current visible integer `[x, y]` points in
  the normalized 0-255 image coordinate system only when they help ground the
  subgoal. Do not invent a point for an action without a visual target.
- The first returned subgoal must be the action that should execute now.
- An `observe` subgoal must name the visible object or region being observed
  and may provide its coordinates when a visible region is relevant. While it is
  active, both arms should remain near their zero/home pose.
- Return only the fields shown below.

Return JSON only:

```json
{
  "subgoals": [
    {
      "text": "Pick up the red block at [96, 132].",
      "atomic_action": "pick",
      "arm": "right",
      "coordinates": [[96, 132]]
    }
  ]
}
```
