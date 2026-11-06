---
name: continuous_joint_descent
description: "Plans a bounded local six-joint descent from current own joints toward a public target pose converted through own_base and frame_bias into the own root frame using a worker-owned forward-kinematics callback and least_squares. Returns the joint planner's path when it reports Success, otherwise a linear interpolation fallback with at least 30 intervals and joint step <= 0.01."
---

# Continuous joint descent

## Function

Plans a bounded local six-joint descent from current own joints toward a public target pose converted through own_base and frame_bias into the own root frame using a worker-owned forward-kinematics callback and least_squares. Returns the joint planner's path when it reports Success, otherwise a linear interpolation fallback with at least 30 intervals and joint step <= 0.01.

## Inputs

- current: finite 6-vector of own-arm joints
- pose and own_base: finite 7-vectors (position plus unit quaternion in w,x,y,z order); frame_bias: finite 3-vector
- forward_kinematics(joints) -> (position 3, quaternion 4) and joint_planner(current, solution) -> dict with status/position
- unwrap_path: finite N x 6 positions with 1 <= N <= 20000 and a finite 6-vector current

## Outputs

- float32 N x 6 own-joint command path
- joint_planner result is accepted only when status is Success and its position is a finite N x 6 path with 1 <= N <= 20000
- ValueError for nonfinite/wrong-size vectors, non-unit quaternions, unbounded paths, or non-callable callbacks; RuntimeError when residual acceptance fails

## Callable interface

Implementation: [implementation.py](implementation.py).

- `finite(values, size)`
- `local_goal(pose, own_base, frame_bias)`
- `unwrap_path(positions, current)`
- `plan_local_descent(current, pose, own_base, frame_bias, forward_kinematics, joint_planner)`

## Constraints

- Solver bounds each joint to current +/- 0.8 clipped to [-3.14, 3.14], diff_step=0.001, max_nfev=80, ftol/xtol/gtol=1e-7.
- Acceptance requires position residual norm < 0.003 m and orientation residual norm < 0.035 rad after the 0.12 rotation weight.
- unwrap_path unwraps the first five joints toward previous values within 3.14 rad and leaves the sixth joint unchanged.
- Does not receive environment, object pose, collision/contact measurement, task layout, score, or teacher; outputs are own robot commands, not proof of insertion or grasp.
