---
name: garment_mounting_datum
description: "Validates the exact v45 model-based Fold mounting datum, recomputes left/right own base poses from current EEF poses and six arm joints using imported FK, and checks consistency with supplied roots, mean height, and disagreement. Returns a deep copy of the validated value."
---

# Garment mounting datum

## Function

Validates the exact v45 model-based Fold mounting datum, recomputes left/right own base poses from current EEF poses and six arm joints using imported FK, and checks consistency with supplied roots, mean height, and disagreement. Returns a deep copy of the validated value.

## Inputs

- value: dict with exact fields source, reference_semantics, kinematic_model_sha256, current_public_action_epoch, public_own_state, own_base_poses, mount_z_m, root_height_disagreement_m, height_is_observed_surface, surface_height_error_bound_m
- public_own_state: left/right _arm_joint_state (6) and _ee_pose (7) arrays
- own_base_poses: left/right 7-element pose arrays

## Outputs

- deepcopy of value after validation
- Raises ValueError on field/constant/shape/finite/quaternion/root/FK/height inconsistency

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`

## Constraints

- Requires source public_own_robot_mounting_datum_v45, the declared reference semantics and URDF SHA256, height_is_observed_surface=False, and surface_height_error_bound_m=None
- current_public_action_epoch must be an int in [0,500]; independent root heights must agree within 0.005 m; supplied mean/disagreement must match computation within 1e-8
- Reads configs/mature-trajectory.json robot_urdf_sha256 and depends on the repository layout plus imported own-model FK and pose_matrix helpers
- The datum is a static model-based prior; it does not measure the actual surface, camera alignment, physical fingers, or contact
