---
name: own_finger_sweep_support
description: "Computes the lowest world Z of two supplied static finger hulls over their full declared prismatic joint ranges at a finite own-wrist pose. The endpoints suffice because translation along each prismatic axis is affine."
---

# Own finger sweep support

## Function

Computes the lowest world Z of two supplied static finger hulls over their full declared prismatic joint ranges at a finite own-wrist pose. The endpoints suffice because translation along each prismatic axis is affine.

## Inputs

- wrist_pose: finite 7D pose with quaternion norm within 1e-3 of 1
- fingers: exactly two dicts with link values link7 and link8
- each finger: origin 4x4 rigid transform, unit axis 3-vector, limits [lower, upper] with lower < upper, vertices Nx3 with 4..512 finite rows

## Outputs

- float minimum world Z across both finger hulls at both declared limit openings

## Callable interface

Implementation: [implementation.py](implementation.py).

- `minimum_z(fingers, wrist_pose)`

## Constraints

- Raises ValueError for invalid wrist pose, missing/extra links, or malformed nonfinite finger geometry
- Uses authored link-local vertices, joint origins, axes and limits plus the public own-wrist pose; receives no environment, observation, gripper feedback, object pose, reward or action interface
- Full independent joint-range sweep is an uncertainty envelope, not measured finger separation or contact
- Result is a nominal CAD lower bound in the caller's coordinate frame and cannot certify a real surface, collision-free trajectory, grasp, release or task completion
