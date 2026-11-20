---
name: public_phase_execution
description: "Provides a restricted public robot API and a PhasedProgram executor for bounded motion phases, returning proposed own-arm targets and honest phase and visual feedback. Validates task-tool feedback, including skill-specific public measurement contracts, without claiming grasp or official success."
---

# Public phase execution

## Function

Provides a restricted public robot API and a PhasedProgram executor for bounded motion phases, returning proposed own-arm targets and honest phase and visual feedback. Validates task-tool feedback, including skill-specific public measurement contracts, without claiming grasp or official success.

## Inputs

- observation: projected public instruction, RGB image and own left/right arm joint and end-effector state
- request: phase-program request containing skill, side, steps and optional colour, support_colour, reference_colour, count, stage, cursor, slot, turn, cell and source fields
- phases: 1-40 phase dicts with name, 7-element pose, opening, minimum_steps and require_pose
- optional bounded episode facts: home_targets, geometry_memory, command_memory, home_joints, robot_bases, trajectory and issued_action_count

## Outputs

- proposal: deep-copied targets for the declared side with the current phase pose and normalized opening command
- feedback: status, current phase, recent phase reports, action counts, normalized gripper values, image measurements and visual lift feedback
- validated feedback: deep copy after checking declared fields, budgets, poses, gripper ranges and skill-specific measurement contracts

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PublicRobotAPI: __init__(self, observation, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0); observe(self, observation); targets(self); instruction(self); image(self, camera='cam_head'); regions(self, colour, camera='cam_head'); home_targets(self); geometry_memory(self); issued_action_count(self); command_memory(self); trajectory_available(self); home_joint_states(self); own_joint_states(self); own_robot_bases_available(self); own_robot_bases(self); plan_trajectory(self, side, pose, support_plane_m, remaining_actions, time_scale, operation='pose', joint_target=None); own_inverse_kinematics(self, side, pose); preview_trajectory(self, side, poses, support_plane_m, remaining_actions, time_scale); seeded_own_inverse_kinematics(self, side, pose, seed, reference); preview_joint_route(self, side, poses, support_plane_m, remaining_actions, time_scale)`
- `validate_geometry_memory(value)`
- `pose_phase(name, pose, opening, minimum_steps=2, require_pose=True)`
- `translated(pose, delta)`
- `relative_rotation(pose, axis, angle)`
- `downward_tool_pose(pose)`
- `source_measurements(api, colour=None, support_colour=None)`
- `PhasedProgram: __init__(self, api, request, phases, precondition=True, measurements=None); record_action(self); observe(self, observation); proposal(self, observation); feedback(self, observation, budget=False)`
- `validate_phase_feedback(request, value, automatic_side=False)`

## Constraints

- PublicRobotAPI exposes only public-observation-derived data and optional own-robot kinematics/planning callables; it has no simulator, task, scorer, contact, layout, file or network interface.
- issued_action_count must be an integer from 0 to 3600 and is an own issued-command count, not progress, contact or success; PhasedProgram permits 1-40 phases with minimum_steps 1-16 and pose tolerances of 0.004 m and 0.035 rad.
- own_inverse_kinematics and seeded_own_inverse_kinematics have a 1024-query per-tool limit; preview_joint_route requires exactly five poses and has a 96-preview per-tool limit.
- Gripper values are normalized public commands, not measured finger width or contact; feedback always sets visual_review_required true, object_grasp_confirmed false and official_success_claimed false.
