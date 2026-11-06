---
name: closing_direction_orientation
description: "Builds a 3x3 arm-dependent downward gripper orientation from a measured closing yaw and one allowed tilt. run validates the exact payload and returns the rotation with zero robot actions and no GT object-pose claim."
---

# Closing direction orientation

## Function

Builds a 3x3 arm-dependent downward gripper orientation from a measured closing yaw and one allowed tilt. run validates the exact payload and returns the rotation with zero robot actions and no GT object-pose claim.

## Inputs

- closing_yaw: finite current measured closing-axis yaw in radians
- tilt_degrees: one of 15, 25, 35, or 0
- arm: 'left' or 'right'
- payload for run: dict with exactly closing_yaw, tilt_degrees, arm

## Outputs

- 3x3 rotation matrix whose columns are forward, closing, and upward
- run result dict with rotation, robot_actions_issued=0, object_pose_is_GT=False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `closing_orientation(closing_yaw, tilt_degrees=15, arm='left')`
- `run(payload)`

## Constraints

- Rejects nonfinite or nonnumeric closing_yaw, unsupported tilt_degrees, or arm outside left/right with ValueError.
- Reorients lateral direction according to arm and projects closing axis perpendicular to tilted forward; normalizes closing vector.
- Returns only own gripper rotation; it is not an object pose measurement, does not access the environment, and does not claim grasp success.
- run rejects payloads with keys other than the three declared parameters.
