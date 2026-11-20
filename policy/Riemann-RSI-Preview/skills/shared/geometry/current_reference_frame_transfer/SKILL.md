---
name: current_reference_frame_transfer
description: "Checks three finite 7-vector poses and computes target_reference @ inverse(source_reference) @ tool_contact_command as a transferred pose. It returns the pose with explicit hypothesis and no-action flags."
---

# Current reference frame transfer

## Function

Checks three finite 7-vector poses and computes target_reference @ inverse(source_reference) @ tool_contact_command as a transferred pose. It returns the pose with explicit hypothesis and no-action flags.

## Inputs

- source_reference: finite 7-vector public numeric XYZ/WXYZ pose
- target_reference: finite 7-vector public numeric XYZ/WXYZ pose
- tool_contact_command: finite 7-vector own contact command pose in XYZ/WXYZ

## Outputs

- dict with pose: 7-vector transferred pose
- dict flags: source='current_reference_frame_transport_of_own_contact_command', attachment_is_hypothesis=True, true_object_pose_or_asset_geometry_read=False, contact_measurement_claimed=False, robot_actions_issued=0, official_success_claimed=False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `checked_pose(value)`
- `run(source_reference, target_reference, tool_contact_command)`

## Constraints

- checked_pose requires shape (7), finite values, |XYZ| <= 5, and quaternion norm in [0.99, 1.01], raising ExecutionError otherwise.
- Uses matrix_pose/pose_matrix from urdf_pose_kinematics; no environment, object asset, layout, scorer, or true-contact capability.
- Transported attachment remains a hypothesis; do not substitute stored geometry for an unqualified source or destination.
- Pose and orientation uncertainty must be qualified before task scheduling; no robot command is issued.
