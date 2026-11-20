---
name: policy_action_step
description: "Runs one learned-policy action step on the latest prepared RGB image and 16-value robot state, then formats the returned vector as left/right end-effector poses and normalized gripper commands."
---

# Policy action step

## Function

Runs one learned-policy action step on the latest prepared RGB image and 16-value robot state, then formats the returned vector as left/right end-effector poses and normalized gripper commands.

## Inputs

- Adapter instance with self._latest and a loaded self.policy implementing act(image, state).
- Calling adapter constants for the 16-value action layout, quaternion starts and gripper indices.

## Outputs

- One action dictionary in a one-element list, containing both seven-value EE poses and one normalized jaw command per side.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `get_action(self)`

The skill library assembles this callable header with the operations in [workflow.json](workflow.json).

## Constraints

- The skill library assembles the callable body from workflow.json; implementation.py supplies its signature and is not a standalone no-op implementation.
- Raises RuntimeError before any observation and ValueError for a policy output with a length other than 16.
- Copies the float32 action, normalizes each quaternion only when its norm exceeds 1e-8, and clips gripper commands to [0, 1].
