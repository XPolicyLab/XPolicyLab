---
name: rigid_tool_command_tracking_observed_start
description: "Provides the observed-start variant of pure rigid-transform helpers for held-tool tracking: checked poses, predicted tool pose, wrist-for-tool goal, and bounded translation/rotation correction status. It returns a command correction or a diverged/within-tolerance classification without executing motion."
---

# Rigid tool command tracking observed start

## Function

Provides the observed-start variant of pure rigid-transform helpers for held-tool tracking: checked poses, predicted tool pose, wrist-for-tool goal, and bounded translation/rotation correction status. It returns a command correction or a diverged/within-tolerance classification without executing motion.

## Inputs

- observed_wrist, attachment, tool_goal, and wrist_command: 7-element [x,y,z,qx,qy,qz,qw] poses
- tolerance: translation tolerance in meters, 0 < tolerance <= 0.04
- rotation_tolerance_degrees: 0 < value <= 20

## Outputs

- predicted_tool_pose(observed_wrist, attachment): 7-element predicted tool pose
- wrist_for_tool_goal(tool_goal, attachment): 7-element wrist command pose
- tracking_correction(...): dict with status, translation_m, rotation_degrees, command_pose, and prediction_is_true_object_pose=False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `checked_pose(value)`
- `predicted_tool_pose(observed_wrist, attachment)`
- `wrist_for_tool_goal(tool_goal, attachment)`
- `tracking_correction(goal, predicted_tool, wrist_command, tolerance, rotation_tolerance_degrees)`

## Constraints

- Imports matrix_pose and pose_matrix from urdf_pose_kinematics_observed_start.
- checked_pose rejects shape != (7,), nonfinite values, max |xyz| > 3, and quaternion norm outside [0.99,1.01] with ValueError.
- tracking_correction returns within_tolerance with a copied wrist_command when distance <= tolerance and angle <= rotation_tolerance_degrees; it returns diverged with command_pose None when distance > 0.04 m or angle > 20 degrees.
- Predicted tool pose is a public attachment belief, not a true object pose or contact measurement; no retries or feedback guards are changed.
