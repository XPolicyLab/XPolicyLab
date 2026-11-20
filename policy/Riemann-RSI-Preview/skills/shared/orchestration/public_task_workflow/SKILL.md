---
name: public_task_workflow
description: "Dispatches existing public task workflows by declaration adapter_family through the official policy protocol, using PublicSession as the acknowledged-action counter and public frame store. It supports observation loops plus stack, toast, and command workflow families, with task-specific source bindings and local policy-only preparation."
---

# Public task workflow

## Function

Dispatches existing public task workflows by declaration adapter_family through the official policy protocol, using PublicSession as the acknowledged-action counter and public frame store. It supports observation loops plus stack, toast, and command workflow families, with task-specific source bindings and local policy-only preparation.

## Inputs

- mode and declaration with adapter_family and official_action_horizon
- initial public observation and exchange callback
- output_root for diagnostics and policy assets
- optional public_parameters; only sort_nesting_dolls_by_size accepts public_shape_profile random

## Outputs

- policy-diagnostics.json with policy_error, traceback, adapter_family, acknowledged_actions, phases, actor diagnostics, and official_success_claimed false
- own_command_budget_reached flag when PublicActionBudgetReached is raised
- task-family return value, such as actor.run() for stack workflows; no official score or success

## Callable interface

Implementation: [implementation.py](implementation.py).

- `prepare(mode, source)`
- `PublicSession: __init__(self, initial, rpc, root, horizon, evidence, *, robots=None, task_name=''); public_frame(self)`
- `run(mode, declaration, initial, exchange, output_root, *, prepare_runtime=prepare, public_parameters=None)`

## Constraints

- adapter_family must be one of public_observation_policy, public_command_workflow, public_stack_workflow, or public_toast_workflow
- output_root is created with exist_ok false and preparation is policy-local, not evaluator hooks
- PublicSession counts acknowledged commands with dt .004 and step_lim horizon; this count does not reveal official termination or success
- exceptions are recorded and enter finite last-command holding; original task sampling, release waits, and recovery remain in existing skills
