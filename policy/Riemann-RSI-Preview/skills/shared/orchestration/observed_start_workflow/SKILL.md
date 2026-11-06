---
name: observed_start_workflow
description: "Defines DeterministicTaskWorkflow, which executes a frozen native skill workflow as a cursor over declared stages, dispatching one adapter request at a time and advancing only on a validated skill_outcome receipt. It stops with WorkflowStopped on invalid contracts, receipts, partial irreversible stages, or exhausted recovery, and snapshots state without claiming official success."
---

# Observed start workflow

## Function

Defines DeterministicTaskWorkflow, which executes a frozen native skill workflow as a cursor over declared stages, dispatching one adapter request at a time and advancing only on a validated skill_outcome receipt. It stops with WorkflowStopped on invalid contracts, receipts, partial irreversible stages, or exhausted recovery, and snapshots state without claiming official success.

## Inputs

- definition: dict with execution exactly {'executor': 'deterministic_native', 'maximum_recovery_attempts_per_stage': 2, 'partial_irreversible_replay': False}, completion_is_official_success is False, and 1-64 stages
- adapter: object exposing allowed_skills, recovery_skills, initialise(api), request(stage, context, api), boundary_completed(stage, context, event), recovery_completed(event), and recover(stage, context, attempt, event)
- api: public API object whose issued_action_count() is used to bind the pending request epoch
- event: skill_outcome receipt containing event, request, skill, actions, execution_feedback.actions_executed, public_action_epoch, and official_success_claimed

## Outputs

- next_request returns a deep-copied request dict with skill and integer steps >= 1, or None when no stage remains
- WorkflowStopped exception carrying failure dict with code, stage, reason truncated to 4096 characters, phase, and official_success_claimed False
- snapshot() returns a dict with executor, state, cursor, stage, completed_stages, recovery_attempts, active_skill, failure, and official_success_claimed False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `WorkflowStopped: __init__(self, code, stage, reason, phase=None)`
- `DeterministicTaskWorkflow: __init__(self, definition, adapter); stage(self); stop(self, code, reason, phase=None); next_request(self, api); accept(self, event); snapshot(self)`

## Constraints

- The class is adapter-bound: dispatch, qualification, and recovery depend on the supplied adapter methods and the imported ACTION_BINDINGS; no standalone invocation is defined
- Requires adapter.allowed_skills and adapter.recovery_skills to be subsets of ACTION_BINDINGS/allowed_skills, and stages to use unique bounded IDs and only id, skill, side, move_index keys
- A pending request blocks advancement; accept requires exact request/skill match, actions equal feedback actions_executed and <= steps, public_action_epoch equal pending epoch + actions, nondecreasing epoch, and no official success claim
- If actions > 0 and boundary_completed is false, stops with partial_irreversible_stage; recovery is limited to two attempts per stage and 1-3 requests using adapter.recovery_skills
