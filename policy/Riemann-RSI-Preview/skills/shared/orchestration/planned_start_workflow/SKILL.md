---
name: planned_start_workflow
description: "Defines WorkflowStopped and DeterministicTaskWorkflow, which validates a frozen native task definition and advances through declared stages using adapter-supplied requests, receipts, and bounded recovery without model reselection. Completion means declared commands finished; official success remains unclaimed."
---

# Planned start workflow

## Function

Defines WorkflowStopped and DeterministicTaskWorkflow, which validates a frozen native task definition and advances through declared stages using adapter-supplied requests, receipts, and bounded recovery without model reselection. Completion means declared commands finished; official success remains unclaimed.

## Inputs

- definition: mapping with execution, stages, and completion_is_official_success
- adapter: object exposing allowed_skills, recovery_skills, initialise, request, recover, recovery_completed, and boundary_completed
- api: public API used by the adapter and for issued_action_count()
- event: skill_outcome receipt with request, skill, actions, execution_feedback, public_action_epoch, and official_success_claimed

## Outputs

- next_request(api): deep-copied native request dict, or None when the cursor is complete
- accept(event): mutates cursor, pending, recovery, attempts, state, and completed_stages, or raises WorkflowStopped
- snapshot(): dict containing executor, state, cursor, stage, completed_stages, recovery_attempts, active_skill, failure, and official_success_claimed false

## Callable interface

Implementation: [implementation.py](implementation.py).

- `WorkflowStopped: __init__(self, code, stage, reason, phase=None)`
- `DeterministicTaskWorkflow: __init__(self, definition, adapter); stage(self); stop(self, code, reason, phase=None); next_request(self, api); accept(self, event); snapshot(self)`

## Constraints

- Construction raises ExecutionError unless execution equals deterministic_native with maximum_recovery_attempts_per_stage 2 and partial_irreversible_replay false, completion_is_official_success is false, and stages length is 1 through 64.
- Each stage must be a dict with only id, skill, side, and move_index; id must be a unique string of length 1 through 128; move_index, when present, must be an int from 1 through 64; side must be left or right; skill must be in adapter.allowed_skills and ACTION_BINDINGS.
- A pending boundary must finish before advancement; receipts must match the exact active request, action count, and public action epoch, and both event and execution_feedback must report official_success_claimed false.
- Recovery is limited to two attempts per stage and each recovery list must contain one through three requests using adapter.recovery_skills; partial irreversible stages stop without replay, and completion is not official success.
