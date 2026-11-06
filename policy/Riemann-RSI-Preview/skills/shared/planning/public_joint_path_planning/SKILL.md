---
name: public_joint_path_planning
description: "Planner wraps injected left/right CuRobo planners to update collision worlds and plan/check joint, Cartesian, and pose paths; cartesian performs sequential fixed-URDF IK and validates sampled configurations with native collision spheres. It does not accept an environment or task handle."
---

# Public joint path planning

## Function

Planner wraps injected left/right CuRobo planners to update collision worlds and plan/check joint, Cartesian, and pose paths; cartesian performs sequential fixed-URDF IK and validates sampled configurations with native collision spheres. It does not accept an environment or task handle.

## Inputs

- planners: mapping keyed by arm with plan_joint, plan_path, robot_cfg, device_cfg, motion_planner, and _build_cspace_joint_values
- frame: object with bases, mounting_z, kinematics.fk, ik, and root_pose
- digits: iterable of dicts with digit, xy, origin_to_bottom_m, height_m, and optional support_z; excluded digit label
- arm, current, target, path: arm name, joint arrays, target pose, and sampled joint path

## Outputs

- world(...): SceneCfg with table, non-excluded digit cuboids, and other_robot_base cuboid
- validate(...): bool from RobotCollisionChecker over all sampled configurations
- cartesian(...): joint path array including current, or None on IK/collision failure
- joint(...): joint path array; pose(...): path array or None on non-Success

## Callable interface

Implementation: [implementation.py](implementation.py).

- `Planner: __init__(self, planners); world(self, arm, frame, digits, excluded=None); validate(self, arm, path); cartesian(self, arm, current, target, frame); joint(self, arm, current, target); pose(self, arm, current, target, frame)`

## Constraints

- world excludes the selected excluded digit and always adds a 3x3x0.05 m table and 0.12 m other-robot-base cuboid; updates motion_planner, motion_planner_batch, and ik_solver worlds
- validate uses collision_activation_distance=0 and self_collision_activation_distance=0, builds full cspace joints, and returns all().item()
- cartesian samples by max 0.008 m translation or 0.055 rad rotation, rejects any IK solution with joint change >0.4, and returns None if validation fails
- joint raises RuntimeError if CuRobo status is not Success; pose returns None if status is not Success; fixed graph dispatch only, no GT/hidden retry/environment capability; CPU parity is not native performance verification
