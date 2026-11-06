---
name: planned_start_action_dispatch
description: "Validates and dispatches bounded task-tool action requests by importing the ACTION_BINDINGS module, building a PublicRobotAPI, and requiring the module to return a PhasedProgram. It also validates returned phase feedback against skill-specific schedules and declared contracts."
---

# Planned start action dispatch

## Function

Validates and dispatches bounded task-tool action requests by importing the ACTION_BINDINGS module, building a PublicRobotAPI, and requiring the module to return a PhasedProgram. It also validates returned phase feedback against skill-specific schedules and declared contracts.

## Inputs

- request: dict with skill, steps, reason, and exactly the parameters declared in ACTION_PARAMETERS for that skill
- observation plus optional home_targets, geometry_memory, trajectory, home_joints, robot_bases, command_memory, issued_action_count, and workflow_factory for action_program
- feedback: phase feedback dict with status, actions_executed, required_minimum_actions, phase_reports, and image_measurements
- skill name used to look up ACTION_BINDINGS, ACTION_LIMITS, ACTION_MINIMUM_STEPS, ACTION_PARAMETERS, and ACTION_CONTRACTS

## Outputs

- deep-copied validated request from validate_action_parameters
- PhasedProgram returned by the bound module's run(api, request), or by workflow_factory for the two declared rule workflows
- checked feedback dict after validate_phase_feedback and skill-specific schedule checks
- action_contract dict containing parameters, budget, statuses, and run_interface

## Callable interface

Implementation: [implementation.py](implementation.py).

- `action_contract(name)`
- `validate_action_parameters(request)`
- `action_program(observation, request, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0, workflow_factory=None)`
- `validate_action_feedback(request, feedback)`

## Constraints

- Request keys must equal declared parameters plus skill, steps, and reason; invalid types, enums, non-finite numbers, out-of-range values, delta norm > 0.15, non-unit axis, or equal reference_colour/colour raise ExecutionError.
- number_empty_approach steps must be an int within its declared shared action allowance; t_transfer_stage side must match the stage suffix (right for right stages, left otherwise).
- workflow_factory is allowed only for instrument_rule_boundary and equation_rule_boundary; other uses raise ExecutionError.
- Feedback required_minimum_actions must match exact per-skill schedules, including special number_empty_approach measurement validation, otherwise ExecutionError.
