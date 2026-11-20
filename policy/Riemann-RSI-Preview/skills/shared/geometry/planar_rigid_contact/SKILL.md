---
name: planar_rigid_contact
description: "Validates bounded seven-value public pose hypotheses and composes planar rigid-contact commands: wrapped yaw/error helpers, a capped planar step, object-in-tool capture, and wrist targets. contact_wrist builds a commanded wrist pose from a shape pose, a ±1 closing direction, and a fixed own TCP offset."
---

# Planar rigid contact

## Function

Validates bounded seven-value public pose hypotheses and composes planar rigid-contact commands: wrapped yaw/error helpers, a capped planar step, object-in-tool capture, and wrist targets. contact_wrist builds a commanded wrist pose from a shape pose, a ±1 closing direction, and a fixed own TCP offset.

## Inputs

- current, destination, and initial poses as seven values [x,y,z,qw,qx,qy,qz] with finite position components and quaternion norm in [0.99,1.01]
- translation cap in meters (default 0.018, must be >0 and <=2) and yaw cap in radians (default 6 degrees, must be >0 and <=pi)
- public shape pose and observed own wrist pose for capture, plus a shape-in-wrist transform for own_wrist_target
- contact_wrist turn in {-1,1}, shape pose, and forward_offset exactly 0.154 m

## Outputs

- checked_pose returns a copied seven-value pose with normalized quaternion
- planar_step returns the bounded planar increment pose
- capture returns the captured object-in-tool transform; own_wrist_target returns the wrist target pose
- contact_wrist returns a commanded seven-value wrist pose [x,y,z,qw,qx,qy,qz]

## Callable interface

Implementation: [implementation.py](implementation.py).

- `checked_pose(value)`
- `yaw(value)`
- `wrapped(value)`
- `errors(actual, destination)`
- `planar_step(current, destination, initial, translation=0.018, angle=math.radians(6))`
- `capture(public_shape_pose, observed_own_wrist)`
- `own_wrist_target(destination, shape_in_wrist)`
- `contact_wrist(shape_pose, turn, forward_offset=0.154)`

## Constraints

- Invalid pose shape, non-finite values, position magnitude >3, or quaternion norm outside 0.99-1.01 raises ValueError
- Invalid translation/yaw bounds or unsupported contact_wrist turn/offset raise ValueError
- Forward direction is [0,0,-1], closing direction is the shape's world-Y projected to the XY plane and scaled by turn
- The captured attachment is a command-tracking hypothesis, not measured contact or ground-truth object pose; table height and T dimensions are not constants here
