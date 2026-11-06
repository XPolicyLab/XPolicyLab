---
name: policy_episode_reset
description: "Marks the learned episode for initialization on its next observation and clears the cached image/state pair while retaining the loaded policy."
---

# Policy episode reset

## Function

Marks the learned episode for initialization on its next observation and clears the cached image/state pair while retaining the loaded policy.

## Inputs

- The calling adapter instance.

## Outputs

- Sets self._pending_reset=True and self._latest=None; emits the existing reset log message.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `reset(self)`

The skill library assembles this callable header with the operations in [workflow.json](workflow.json).

## Constraints

- The skill library assembles the callable body from workflow.json; implementation.py supplies its signature and is not a standalone no-op implementation.
- The next update_obs supplies fresh RGB, state and instruction to policy.reset.
- Reset neither returns an action nor constructs another model.
