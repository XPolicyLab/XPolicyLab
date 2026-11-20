---
name: policy_observation_update
description: "Prepares the current three-camera RGB composition and robot state, performs a pending episode reset with the current task text, and retains the prepared pair for the next action."
---

# Policy observation update

## Function

Prepares the current three-camera RGB composition and robot state, performs a pending episode reset with the current task text, and retains the prepared pair for the next action.

## Inputs

- Current public observation consumed by self._stitch_image, self._state_from_obs and self._task_text.
- Adapter instance with self.policy and the self._pending_reset flag.

## Outputs

- Updates self._latest to (image, state); no action is returned.
- On the first observation after reset, calls policy.reset(image, state, instruction) and clears the pending flag.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `update_obs(self, obs)`

The skill library assembles this callable header with the operations in [workflow.json](workflow.json).

## Constraints

- The skill library assembles the callable body from workflow.json; implementation.py supplies its signature and is not a standalone no-op implementation.
- The current instruction is passed through the adapter's task-text method; evaluator labels and reward are not inputs to this graph.
- Does not construct or reload the policy model.
