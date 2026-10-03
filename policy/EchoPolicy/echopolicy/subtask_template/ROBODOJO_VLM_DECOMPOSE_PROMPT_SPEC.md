# RoboDojo Task Decomposition

Decompose the requested robot task into the smallest ordered sequence of
executable subgoals. Use the supplied current camera images to ground every
subgoal.

When an `Official RoboDojo task description` is supplied, use it to preserve
the benchmark's required objects, sequence, intermediate operations, and
terminal condition. The task instruction and current images remain
authoritative for episode-specific targets and scene state.

When an `Official RoboDojo full-score condition` is supplied, the complete
ordered plan must include every action required by that condition, including
repeated confirmations, final placements, gripper release, and returning the
robot to origin when specified. Do not merge or omit a required state change.

For every subgoal:

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
- `coordinates` is optional. When a visible point helps ground the subgoal, it
  may contain one or more integer `[x, y]` points in the normalized 0-255 image
  coordinate system. Do not invent a point for an action without a visual target.
- Keep object identity, target identity, and required execution order from the
  original task.
- An `observe` subgoal must name the visible object or region being observed
  and may provide its coordinates when a visible region is relevant. While it is
  active, both arms should remain near their zero/home pose.
- Return only the fields shown below.

Return JSON only:

```json
{
  "subgoals": [
    {
      "text": "Pick up the red block at [82, 125].",
      "atomic_action": "pick",
      "arm": "left",
      "coordinates": [[82, 125]]
    },
    {
      "text": "Place the red block on the target at [64, 140].",
      "atomic_action": "place",
      "arm": "left",
      "coordinates": [[64, 140]]
    }
  ]
}
```
