---
name: public_conveyor_workspace
description: "Builds planner-only conveyor collision proxies for deck, fixed camera mount and table from world AABBs and robot root poses, reserves named cuboid slots before cuRobo solver construction, and activates per-arm collision worlds on associated solvers. It also returns legal one-turn base-yaw goal alternatives with the original goal first."
---

# Public conveyor workspace

## Function

Builds planner-only conveyor collision proxies for deck, fixed camera mount and table from world AABBs and robot root poses, reserves named cuboid slots before cuRobo solver construction, and activates per-arm collision worlds on associated solvers. It also returns legal one-turn base-yaw goal alternatives with the original goal first.

## Inputs

- world AABB lower/upper 3-vectors in meters, robot_origin_pose 7-vector [x,y,z,qw,qx,qy,qz], env_origin and frame_bias 3-vectors, nonnegative padding in meters
- env.robot_manager target robots, planners, standalone IK wrappers, robot roots by arm name, and belt_height
- joint goal q, current joint vector, and ordered (N,2) joint limits in radians for base-yaw alternatives
- optional planner_class for install_conveyor_planner_capacity dependency injection

## Outputs

- deck proxy dict with name DECK_PROXY_NAME, 7-element pose and 3-element dims
- ConveyorPlannerWorld with activate(arm, force=False) returning whether a reload occurred and a mutable report
- installation report dict with version, installed_now and reserved_names
- list of joint goal vectors: original first, then legal q[0] +/- 2*pi alternatives ordered by distance to current base yaw

## Callable interface

Implementation: [implementation.py](implementation.py).

- `equivalent_base_yaw_goals(q, current, limits)`
- `install_conveyor_planner_capacity(planner_class=None)`
- `deck_proxy_from_bounds(lower, upper, robot_origin_pose, *, env_origin=(0.0, 0.0, 0.0), frame_bias=(0.0, 0.0, 0.0), padding=0.002)`
- `ConveyorPlannerWorld: __init__(self, specs, report); activate(self, arm, *, force=False)`
- `unproject(pixels, height)`
- `vision_collision_world(env, roots, belt_height)`

## Constraints

- Planner-only: placeholders are not a usable collision world and no physical USD prim, physics state or scene manager is changed
- Capacity hook must run before solver/cache/graph creation; repeated installation is idempotent and raises on unsupported hook, different installed version, or reserved-name conflict
- activate(arm) should be called before every IK/plan entry; controller is for serial single-environment use, not concurrent calls into shared planners
- deck_proxy_from_bounds subtracts env_origin, applies inverse robot root, adds frame_bias, rotates pose without re-axis-aligning dims; padding is conservative planner clearance only
