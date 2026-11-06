---
name: observed_rigid_transfer
description: "Validates a closed same-instruction public rigid-transfer state, including semantic goal, current scene/anchor, job/recipe progress, bounded histories, support memory, and optional own execution receipts; measurements() validates visual review and native-command completion receipts."
---

# Observed rigid transfer

## Function

Validates a closed same-instruction public rigid-transfer state, including semantic goal, current scene/anchor, job/recipe progress, bounded histories, support memory, and optional own execution receipts; measurements() validates visual review and native-command completion receipts.

## Inputs

- rigid-transfer state dict with schema riemann-public-rigid-transfer-v3, instruction_fingerprint, goal, phase, counters, scene, completed_descriptions, visited_targets, last_source_epoch, home_leg, planned_sources, planned_targets, observed_source_count, recipe, recipe_index, job, support_memory, view_attempts, view_hint
- goal dict with kind, sources, receiver, receiver_kind, ordering, matching, own_forward_m, close_commands, release_commands, release_strategy, after_release, axis_alignment, source_side_rule and optional close_settle_commands, release_settle_commands, pre_release_hold_commands, gripper_schedule
- scene and anchor dicts with current_public_rgb_visual_hull or current_public_rgb_tracked_reference geometry, source_reference, target_reference, target_outline, support_memory, side, route, matching_confidence, observation_epoch
- measurements dict with public_rigid_transfer_state, rigid_visual_review, rigid_command_phases_completed, optional own_feedback_motion, rigid_motion_execution, normalized_command_clock; status and action_count

## Outputs

- Deep-copied validated rigid-transfer state dict
- initial(instruction) state with sha256 instruction_fingerprint, phase clear, zero counters, None goal/scene/recipe, and empty planned/completed/visited lists
- recorded(memory, instruction) returns the latest validated state only if its fingerprint matches, otherwise initial(instruction)
- measurements(...) returns the set of validated optional receipt keys present

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size, maximum=4.0)`
- `pose(value)`
- `goal(value)`
- `anchor(value)`
- `scene(value)`
- `initial(instruction)`
- `initial_job()`
- `validate(value)`
- `recorded(memory, instruction)`
- `measurements(value, status, action_count)`

## Constraints

- Closed field sets and ValueError on unknown/private fields, invalid schema/phase/goal kinds, nonfinite or out-of-bound values, non-unit quaternions, invalid coverage/outlines, or fingerprint mismatch.
- Counters are ints bounded view_leg<=2, item<=32, attempts<=3, home_leg<=2, observed_source_count<=32, view_attempts<=4; recipe_index<=48 and within recipe steps; planned/completed/visited lists each <=32; goal own_forward_m in (.145,.155), close_commands/release_commands 1-40, settle/hold 0-16.
- rigid_visual_review status is not_reviewed, uncertain, current_surface_follows, or current_image_placement with is_contact_measurement false; position_error_m 0-4 and colour_difference 0-450 are uncertain evidence; rigid_command_phases_completed requires status completed/uncertain and action_count>=1.
- Optional receipts are closed records with native_actions_issued equal to action_count; own_feedback_motion source current_public_own_pose_joint_feedback_and_declared_own_fk, rigid_motion_execution backend declared_public_own_curobo_trajectory, normalized_command_clock command_source explicit_already_sampled_own_normalized_commands, and all contact/finger/official-success flags false.
