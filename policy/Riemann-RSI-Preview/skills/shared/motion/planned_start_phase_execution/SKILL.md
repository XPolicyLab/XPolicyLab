---
name: planned_start_phase_execution
description: "Defines a restricted PublicRobotAPI for projected observation and own-robot state, plus a PhasedProgram that advances bounded pose/opening phases and reports public pose errors and uncertainty. It also validates phase feedback and swap placement reviews against declared public fields."
---

# Planned start phase execution

## Function

Defines a restricted PublicRobotAPI for projected observation and own-robot state, plus a PhasedProgram that advances bounded pose/opening phases and reports public pose errors and uncertainty. It also validates phase feedback and swap placement reviews against declared public fields.

## Inputs

- Public observation containing instruction, RGB images, own left/right EE poses and six joint states, and normalized gripper values; optional callbacks for trajectory, own FK, command memory, home targets/joints, and issued_action_count.
- PhasedProgram request with side, skill, steps, optional colour/support_colour/reference_colour/count, and 1-40 phase dictionaries of name, pose (7,), opening, minimum_steps, require_pose.
- validate_phase_feedback(request, value, automatic_side=False) feedback dictionaries and validate_swap_placement_review(request, value) occupancy-review dictionaries.
- Optional geometry memory limited to support_plane_m and raised_colours.

## Outputs

- PublicRobotAPI copies including targets, instruction, image, regions, home targets/joints, geometry/command memory, own joint states, issued action count, and trajectory/IK/dense/preview results with status and positions.
- PhasedProgram.proposal returns deep-copied targets with the phase EE pose and normalized opening; feedback returns status, phase reports, action counts, public gripper values, image measurements, and visual lift feedback.
- Validators return deep copies of validated feedback/review data or raise ExecutionError/ValueError for malformed fields.
- Phase reports contain position_error_m and rotation_error_radians; program statuses include running, completed, uncertain, precondition_unmet, and budget.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PublicRobotAPI: __init__(self, observation, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0); observe(self, observation); targets(self); instruction(self); image(self, camera='cam_head'); regions(self, colour, camera='cam_head'); home_targets(self); geometry_memory(self); issued_action_count(self); command_memory(self); trajectory_available(self); home_joint_states(self); own_joint_states(self); own_robot_bases_available(self); own_robot_bases(self); plan_trajectory(self, side, pose, support_plane_m, remaining_actions, time_scale, operation='pose', joint_target=None); own_inverse_kinematics(self, side, pose); dense_own_trajectory(self, side, pose, operation='original_dense_pose', joint_target=None, start_state=None); preview_trajectory(self, side, poses, support_plane_m, remaining_actions, time_scale)`
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

- PublicRobotAPI accepts issued_action_count only as an integer from 0 to 3600; command memory is at most 32 dict events; home joints require six finite values per side; home targets require finite 7-vector EE poses with near-unit quaternion and gripper values from 0 to 1.
- PhasedProgram requires 1-40 phases, finite 7-vector poses with quaternion norm 0.99-1.01, opening 0-1, minimum_steps 1-16, and a minimum action sum no greater than request steps or status becomes precondition_unmet; pose tolerance is 0.004 m and rotation tolerance is 0.035 rad.
- Feedback must set visual_review_required True, object_grasp_confirmed False, and official_success_claimed False; gripper values are normalized public commands, not measured finger width/contact; phase reports and actions are bounded; image-axis angles are limited to 0..pi/2.
- The sampled trajectory API rejects READ_ONLY_DENSE_OPERATIONS, own IK is limited to 1024 queries, dense calls to 512, and downward_tool_pose aligns the own observed quaternion's local +X tool axis to world downward as robot geometry only; implementation relies on imported shared helpers/contract validators and optional adapter-supplied callbacks, not standalone invocation.
