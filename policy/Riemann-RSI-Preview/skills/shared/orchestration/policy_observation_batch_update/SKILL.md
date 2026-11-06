---
name: policy_observation_batch_update
description: "Passes the sole batch observation to the learned adapter's ordinary update_obs method."
---

# Policy observation batch update

## Function

Passes the sole batch observation to the learned adapter's ordinary update_obs method.

## Inputs

- obs_list containing exactly one public observation and the calling adapter instance.

## Outputs

- The state effects of self.update_obs(obs_list[0]); no action is returned.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `update_obs_batch(self, obs_list)`

The skill library assembles this callable header with the operations in [workflow.json](workflow.json).

## Constraints

- The skill library assembles the callable body from workflow.json; implementation.py supplies its signature and is not a standalone no-op implementation.
- Raises NotImplementedError when len(obs_list) is not one; there is no multi-environment state collection in this skill.
