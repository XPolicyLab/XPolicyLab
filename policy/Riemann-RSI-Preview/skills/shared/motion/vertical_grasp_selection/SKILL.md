---
name: vertical_grasp_selection
description: "PublicVerticalGraspSelector validates a restricted public robot API and current RGB surface geometry, then searches available open arms in order of inferred base XY distance. For each arm it preflights two approach heights and four yaw offsets through api.plan_trajectory and returns the first accepted approach/grasp pair."
---

# Vertical grasp selection

## Function

PublicVerticalGraspSelector validates a restricted public robot API and current RGB surface geometry, then searches available open arms in order of inferred base XY distance. For each arm it preflights two approach heights and four yaw offsets through api.plan_trajectory and returns the first accepted approach/grasp pair.

## Inputs

- api: instance of PublicRobotAPI adapter exposing own_robot_bases(), targets() and plan_trajectory()
- request: side in auto/left/right and steps action allowance
- surface: point (3), edge_m > 0 and yaw_radians, all finite
- support_plane_m: support-plane height passed to planning

## Outputs

- select() returns side, approach (7), grasp (7), engagement, approach_plan and event
- engagement = surface edge_m * 0.5 - 0.015
- successful_plans stores side, target pose and planner response; events record reachable_grasp with rejected candidates

## Callable interface

Implementation: [implementation.py](implementation.py).

- `quat_mul_wxyz(first, second)`
- `yaw_quat(angle)`
- `VisionSwapBlocksTeacher: exception or data type`
- `PublicVerticalGraspSelector: __init__(self, api, request, surface, support_plane_m); select(self)`

## Constraints

- Raises ExecutionError for wrong API type, invalid surface/request, or Budget status; raises RuntimeError when cuRobo plan_path fails or no candidate is reachable.
- Arm filter uses targets[side+'_ee_joint_state'][0] >= 0.9; auto considers both sides and explicit side restricts to that side.
- Approach heights are 0.04 and 0.02; yaw offsets are 0, pi/2, -pi/2 and pi; wrist pose is shifted 0.155 along the quaternion x axis.
- Selected approach must match the last-second successful plan; selection returns an approach plan and does not execute descent.
