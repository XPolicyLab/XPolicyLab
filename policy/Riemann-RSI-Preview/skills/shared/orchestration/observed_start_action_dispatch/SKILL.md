---
name: observed_start_action_dispatch
description: "Defines task-tool contracts and dispatch for observed-start actions: validates declared parameters, imports the bound skill module, constructs PublicRobotAPI, and requires run(...) to return a PhasedProgram. Also validates phase feedback against shared phase rules and skill-specific minimum or expected command schedules."
---

# Observed start action dispatch

## Function

Defines task-tool contracts and dispatch for observed-start actions: validates declared parameters, imports the bound skill module, constructs PublicRobotAPI, and requires run(...) to return a PhasedProgram. Also validates phase feedback against shared phase rules and skill-specific minimum or expected command schedules.

## Inputs

- request: mapping containing exactly skill, steps, reason, and the skill's declared parameters
- observation: public observation passed into PublicRobotAPI
- optional context: home_targets, geometry_memory, trajectory, home_joints, robot_bases, command_memory, issued_action_count, workflow_factory
- feedback: mapping for validate_action_feedback, including image_measurements and phase reports

## Outputs

- validated request deepcopy from validate_action_parameters
- PhasedProgram from action_program; a non-PhasedProgram raises ExecutionError
- action_contract returns a contract dict with parameters, budget, statuses, and run_interface
- checked feedback mapping from validate_action_feedback

## Callable interface

Implementation: [implementation.py](implementation.py).

- `action_contract(name)`
- `validate_action_parameters(request)`
- `action_program(observation, request, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0, workflow_factory=None)`
- `validate_action_feedback(request, feedback)`

## Constraints

- ACTION_BINDINGS maps action names to module paths; action_program imports the bound module and calls run(api, deepcopy(request)), except instrument_rule_boundary may use workflow_factory.
- Requests must contain only declared parameters plus skill, steps, reason; number_empty_approach steps must be an int in 1..ACTION_LIMITS[name]; invalid values raise ExecutionError.
- delta magnitude must be <= 0.15; axis magnitude must be within 0.99..1.01; reference_colour must differ from colour; t_transfer_stage side must match the stage's left/right arm.
- Feedback validation applies validate_phase_feedback and skill-specific schedules; mismatches raise ExecutionError. action_contract budgets use shared_episode_budget True and statuses completed, uncertain, precondition_unmet, budget. Bound skills run through PublicRobotAPI, not standalone.
