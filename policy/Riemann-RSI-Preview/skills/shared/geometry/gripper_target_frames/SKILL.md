---
name: gripper_target_frames
description: "Provides arrange path resampling, wrist/tip pose construction, lateral park and look poses, repair thresholds, and a supported-place stop predicate. Openings are public finger-mesh hypotheses for clearance, not measured finger width or contact."
---

# Gripper target frames

## Function

Provides arrange path resampling, wrist/tip pose construction, lateral park and look poses, repair thresholds, and a supported-place stop predicate. Openings are public finger-mesh hypotheses for clearance, not measured finger width or contact.

## Inputs

- tip, q, offset for wrist_for_tip; closing_yaw, tilt, outward for downward
- path, current, max_joint_step, speed, min_controls for resample
- frame, rig, xy, yaw, height, normal, tilt_deg, openings for wrist_look_pose; arm for lateral_park_pose
- xy, destination, yaw for needs_pose_repair; actual, target for supported_place_stop

## Outputs

- wrist pose [xyz, wxyz] or quaternion; resampled six-joint path
- target pose for look/park; booleans for repair and stop checks

## Callable interface

Implementation: [implementation.py](implementation.py).

- `wrist_for_tip(tip, q, offset=0.155)`
- `downward(closing_yaw, tilt=0.0, outward=(0.0, 1.0))`
- `resample(path, current, max_joint_step=0.075, speed=1.8, min_controls=5)`
- `lateral_park_pose(frame, arm)`
- `wrist_look_pose(frame, rig, xy, yaw, height=0.22, *, normal=True, tilt_deg=0.0, openings=None)`
- `needs_pose_repair(xy, destination, yaw)`
- `supported_place_stop(actual, target)`

## Constraints

- resample prepends current, uses max absolute joint-step distance, raises RuntimeError above 250 controls, and repeats the last point for zero-length paths.
- wrist_look_pose rejects optical axis with forward z >= -0.2 and public finger-mesh clearance below 0.07 m; openings default to 0.044 each as hypotheses.
- needs_pose_repair uses 0.014 m and 32 degrees; supported_place_stop allows xy <=0.004 m, z in [-0.001, 0.012] m, and norm <=0.014 m.
- supported_place_stop is geometric only and cannot claim a private finger gate or contact; no environment/task state is read.
