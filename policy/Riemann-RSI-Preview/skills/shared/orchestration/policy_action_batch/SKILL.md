---
name: policy_action_batch
description: "Adapts the learned backend's single action result to the one-active-environment batch interface by wrapping self.get_action() in a list."
---

# Policy action batch

## Function

Adapts the learned backend's single action result to the one-active-environment batch interface by wrapping self.get_action() in a list.

## Inputs

- Adapter instance whose get_action() produces one action chunk.
- Optional env_idx_list argument accepted by the signature.

## Outputs

- A one-element list containing the single-environment action chunk.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `get_action_batch(self, env_idx_list=None)`

The skill library assembles this callable header with the operations in [workflow.json](workflow.json).

## Constraints

- The skill library assembles the callable body from workflow.json; implementation.py supplies its signature and is not a standalone no-op implementation.
- The graph does not inspect env_idx_list or construct independent environment models. The outer official adapter owns active-environment validation.
