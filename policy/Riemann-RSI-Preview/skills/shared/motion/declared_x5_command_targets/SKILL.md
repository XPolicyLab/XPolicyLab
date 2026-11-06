---
name: declared_x5_command_targets
description: "Supplies static X5 robot target descriptions to an existing actor environment when robot_manager.robot_list is missing, preserving the actor's step_lim. Existing robot descriptions are returned unchanged."
---

# Declared x5 command targets

## Function

Supplies static X5 robot target descriptions to an existing actor environment when robot_manager.robot_list is missing, preserving the actor's step_lim. Existing robot descriptions are returned unchanged.

## Inputs

- environment with robot_manager and step_lim
- configs/public-x5-numeric-tools.json manifest with public_robot_parameters and planner_robot_names

## Outputs

- original environment when robot_manager already has robot_list
- otherwise SimpleNamespace(robot_manager=manager, step_lim=environment.step_lim) with target robots and process_name mapping

## Callable interface

Implementation: [implementation.py](implementation.py).

- `clock_environment(environment)`

## Constraints

- Declarations encode gripper_scale, sign, mimic, arm_name, gripper_name, and _joint_state command keys; they are not measurements of fingers or contact.
- Initial targets still come from the same public normalized feedback; encoded physical joint targets are command hypotheses.
- No planner or native manager is constructed; original observation updates are retained.
- Factories with existing robot descriptions use the unchanged original route.
