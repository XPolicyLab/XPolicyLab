---
name: task_workflow
description: "Runs a declared deterministic_native task as a finite cursor over native skills, requiring exact structured receipts before advancing and using adapter-supplied requests, completion checks, and bounded recovery. AcknowledgedCommandWorkflow additionally validates prepared programs and enforces finite tool-boundary budgets without model or policy reselection."
---

# Task workflow

## Function

Runs a declared deterministic_native task as a finite cursor over native skills, requiring exact structured receipts before advancing and using adapter-supplied requests, completion checks, and bounded recovery. AcknowledgedCommandWorkflow additionally validates prepared programs and enforces finite tool-boundary budgets without model or policy reselection.

## Inputs

- definition: execution contract with executor deterministic_native, maximum_recovery_attempts_per_stage 2, partial_irreversible_replay false, completion_is_official_success false, and 1..64 unique stages
- adapter: allowed_skills, recovery_skills, initialise, request, boundary_completed, recover, recovery_completed, and optional prepare/position/advance/zero_action_observation hooks
- action_bindings: optional mapping from native skill name to action limit; defaults to ACTION_BINDINGS
- task_name: identifier used by configured_workflow to load tasks/<name>/workflow.json and controller execution

## Outputs

- next_request: deep-copied native request dict with skill and steps, or None when the cursor is complete
- accept: consumes an exact skill_outcome receipt and advances state or raises WorkflowStopped
- snapshot: dict with executor, state, cursor, stage, completed_stages, recovery_attempts, active_skill, failure, official_success_claimed false, and command-boundary counters

## Callable interface

Implementation: [implementation.py](implementation.py).

- `WorkflowStopped: __init__(self, code, stage, reason, phase=None)`
- `DeterministicTaskWorkflow: __init__(self, definition, adapter, *, action_bindings=None); stage(self); stop(self, code, reason, phase=None); next_request(self, api); select_command(self, api); verified_receipt(self, event); accept(self, event); snapshot(self)`
- `AcknowledgedCommandWorkflow: prepare_program(self, program); __init__(self, definition, adapter, *, action_bindings=None); position(self); select_command(self, api); accept(self, event); accept_command(self, event); snapshot(self)`
- `configured_workflow(task_name)`

## Constraints

- Only the exact active request, action count, public action epoch, and official_success_claimed false may advance; partial irreversible commands stop without replay.
- At most two public recovery attempts per stage; recovery must be a declared native recovery list of 1..3 requests, and no policy takeover is permitted.
- A command workflow requires a frozen maximum_tool_boundaries of 1..4096 and cannot rewrite the request or issue unacknowledged actions during prepare_program.
- Completion means declared commands finished, not environment success; the official evaluator alone decides success.
