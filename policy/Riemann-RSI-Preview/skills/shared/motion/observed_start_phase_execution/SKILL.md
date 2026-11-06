---
name: observed_start_phase_execution
description: "Provides a restricted PublicRobotAPI over projected instruction/RGB/own robot state, bounded command memory and optional own-robot trajectory, FK and IK capabilities. Provides PhasedProgram to execute 1-40 bounded pose/opening phases through proposal, observe and feedback, reporting pose arrival, command dwell, stalls and budget/precondition outcomes without claiming grasp or official success."
---

# Observed start phase execution

## Function

Provides a restricted PublicRobotAPI over projected instruction/RGB/own robot state, bounded command memory and optional own-robot trajectory, FK and IK capabilities. Provides PhasedProgram to execute 1-40 bounded pose/opening phases through proposal, observe and feedback, reporting pose arrival, command dwell, stalls and budget/precondition outcomes without claiming grasp or official success.

## Inputs

- Public observation containing instruction, RGB images, 7-element end-effector poses, six-joint arm states and one normalized gripper value per side
- Request dict with skill, side, steps and optional colour, support_colour, reference_colour, count, stage, slot, cursor, turn, cell and source fields
- Phase list of 1-40 dicts with name, finite 7-element pose whose quaternion norm is 0.99-1.01, opening in [0,1], minimum_steps 1-16 and require_pose bool
- Optional trajectory and robot_bases callables, home_targets, home_joints, command_memory of at most 32 dicts, issued_action_count 0-3600 and geometry_memory

## Outputs

- proposal returns a deep copy of left/right ee_pose and ee_joint_state targets, using the current phase pose and normalized opening command when not finished
- feedback returns a validated phase report dict with status, latest phase reports, action counts, normalized public gripper values, image measurements, visual lift feedback and explicit false grasp/success flags
- own_inverse_kinematics returns status, one joint solution or None, query_count, robot_actions and is_motion_command
- dense_own_trajectory and preview_trajectory return validated trajectory result dicts with status, positions and waypoint counts

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PublicRobotAPI: __init__(self, observation, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0); observe(self, observation); targets(self); instruction(self); image(self, camera='cam_head'); regions(self, colour, camera='cam_head'); home_targets(self); geometry_memory(self); issued_action_count(self); command_memory(self); trajectory_available(self); home_joint_states(self); own_joint_states(self); own_robot_bases_available(self); own_robot_bases(self); plan_trajectory(self, side, pose, support_plane_m, remaining_actions, time_scale, operation='pose', joint_target=None); own_inverse_kinematics(self, side, pose); dense_own_trajectory(self, side, pose, operation='original_dense_pose', joint_target=None); preview_trajectory(self, side, poses, support_plane_m, remaining_actions, time_scale)`
- `validate_geometry_memory(value)`
- `pose_phase(name, pose, opening, minimum_steps=2, require_pose=True)`
- `translated(pose, delta)`
- `relative_rotation(pose, axis, angle)`
- `downward_tool_pose(pose)`
- `source_measurements(api, colour=None, support_colour=None)`
- `PhasedProgram: __init__(self, api, request, phases, precondition=True, measurements=None); record_action(self); observe(self, observation); proposal(self, observation); feedback(self, observation, budget=False)`
- `validate_swap_placement_review(request, value)`
- `validate_phase_feedback(request, value, automatic_side=False)`

## Constraints

- PhasedProgram requires type(api) is PublicRobotAPI; if minimum_actions exceeds request['steps'] it sets precondition_unmet and issues no actions, and an ended tool cannot record another action
- A required pose phase completes only after minimum_steps with position error <=0.004 m and quaternion angle <=0.035 rad and no pending trajectory; otherwise a non-required phase completes as command_dwell
- Stall aborts after 64 phase actions or after at least 16 actions with 12 error samples spanning <0.00015, optionally inserting one 0.025 m upward uncertain_retreat unless fewer than 3 actions remain or already retreating
- Feedback validation rejects undeclared measurements, non-boolean visibility flags, non-finite or negative errors, rotation error > pi, normalized gripper values outside [0,1] and any claimed grasp, contact or official success; visual lift feedback is uncertain public-image evidence with displacement fraction <=sqrt(2) and robot translation <=0.3 m
