---
name: public_mature_workflow
description: "Constructs an existing public task controller from public RGB/state, its unchanged profile, and the public command clock, then runs the original external workflow through the policy-side action clock. It writes policy diagnostics and converts ordinary completion or policy exceptions into finite last-command holding rather than official success."
---

# Public mature workflow

## Function

Constructs an existing public task controller from public RGB/state, its unchanged profile, and the public command clock, then runs the original external workflow through the policy-side action clock. It writes policy diagnostics and converts ordinary completion or policy exceptions into finite last-command holding rather than official success.

## Inputs

- controller module or callable with create_public_environment and create
- profile with parameters and optional environment entries, plus initial public observation containing instruction
- exchange callback that submits one action and returns the next acknowledged observation
- output_root and horizon, with optional prepare function

## Outputs

- policy-diagnostics.json containing policy_error, traceback, acknowledged_actions, phases, context diagnostics, and official_success_claimed false
- own_command_budget_reached flag when PublicActionBudgetReached is raised
- no official score or success result

## Callable interface

Implementation: [implementation.py](implementation.py).

- `save(path, value)`
- `configure_planners(full_warmup)`
- `run(controller, profile, initial, exchange, output_root, *, horizon, prepare=configure_planners)`

## Constraints

- profile parameters must have official_eval true; environment entries cannot alter HOME, PATH, PYTHONPATH, LD_PRELOAD, or CUDA_VISIBLE_DEVICES
- environment step_lim must equal horizon and dt must equal .004
- planner configuration is local to this policy process and does not modify the official evaluator process
- this runner is one adapter family; observation-native and generator policies need their own declared bindings, and an exception is recorded and enters policy finite hold
