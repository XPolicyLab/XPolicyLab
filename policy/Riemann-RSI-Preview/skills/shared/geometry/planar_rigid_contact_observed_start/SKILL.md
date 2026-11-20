---
name: planar_rigid_contact_observed_start
description: "Observed-start variant of planar rigid-contact commands using matrix pose conversions; computes xy/orientation errors, a manual capped planar step, object-in-tool capture, and wrist targets. contact_wrist is the same commanded pose construction as the base skill."
---

# Planar rigid contact observed start

## Function

Observed-start variant of planar rigid-contact commands using matrix pose conversions; computes xy/orientation errors, a manual capped planar step, object-in-tool capture, and wrist targets. contact_wrist is the same commanded pose construction as the base skill.

## Inputs

- current, destination, and initial poses as seven values [x,y,z,qw,qx,qy,qz] with finite position components and quaternion norm in [0.99,1.01]
- translation cap in meters (default 0.018, must be >0 and <=2) and yaw cap in radians (default 6 degrees, must be >0 and <=pi)
- public shape pose and observed own wrist pose for capture, plus a shape-in-wrist transform for own_wrist_target
- contact_wrist turn in {-1,1}, shape pose, and forward_offset exactly 0.154 m

## Outputs

- errors returns {'xy_m': Euclidean XY distance in meters, 'orientation_deg': rotation magnitude in degrees}
- planar_step returns a seven-value pose with copied initial orientation and adjusted XY/yaw
- capture returns the captured object-in-tool transform; own_wrist_target returns the wrist target pose
- contact_wrist returns a commanded seven-value wrist pose [x,y,z,qw,qx,qy,qz]

## Callable interface

Implementation: [implementation.py](implementation.py).

- `yaw(value)`
- `wrapped(value)`
- `errors(actual, destination)`
- `planar_step(current, destination, initial, translation=0.018, angle=math.radians(6))`
- `capture(public_shape_pose, observed_own_wrist)`
- `own_wrist_target(destination, shape_in_wrist)`
- `contact_wrist(shape_pose, turn, forward_offset=0.154)`

## Constraints

- Invalid pose shape, non-finite values, position magnitude >3, or quaternion norm outside 0.99-1.01 raises ValueError
- planar_step moves current XY toward destination by min(1, translation/distance); zero XY distance keeps current XY
- planar_step applies a clipped wrapped yaw increment about world Z to the initial orientation
- contact_wrist requires turn in {-1,1} and forward_offset 0.154; its output is a commanded hypothesis, not measured contact
