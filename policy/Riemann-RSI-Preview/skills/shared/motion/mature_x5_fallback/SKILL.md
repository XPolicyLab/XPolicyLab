---
name: mature_x5_fallback
description: "Builds an own-arm X5 joint trajectory for x5_pose, x5_place, x5_place_calibrated, or x5_retreat from public own joints/EE pose and a target pose. Place operations add a high waypoint and the calibrated x/y offset [0.012, 0.010]; retreat copies observed x/y and orientation."
---

# Mature x5 fallback

## Function

Builds an own-arm X5 joint trajectory for x5_pose, x5_place, x5_place_calibrated, or x5_retreat from public own joints/EE pose and a target pose. Place operations add a high waypoint and the calibrated x/y offset [0.012, 0.010]; retreat copies observed x/y and orientation.

## Inputs

- payload with observation, side, operation, pose (7-vector), and public observation state entries side_arm_joint_state (6 joints) and side_ee_pose (7-vector)
- interval used to set upsample count ceil(interval * 1.8)
- own URDF supplied to OriginalX5Fallback, which constructs X5MeasuredFallbackKinematics

## Outputs

- numpy.float32 array of shape (N, 6) of bounded own-joint control commands in radians
- no task-success or contact result is returned

## Callable interface

Implementation: [implementation.py](implementation.py).

- `principal(values)`
- `pose_matrix(pose)`
- `OriginalX5Fallback: __init__(self, urdf); path(self, payload, interval)`

## Constraints

- Supports only the named operations; uses EEF_FROM_TCP and the nominal fingertip transform for FK, not a finger measurement.
- Applies principal-angle wrapping to joint index 5 during upsampling.
- Uses strict TCP tolerances for place/calibrated/retreat and relaxed tolerances for pose; branch_jump_tolerance is 0.50.
- Reads no environment, object registry, layout, scorer, contact, or reward state.
