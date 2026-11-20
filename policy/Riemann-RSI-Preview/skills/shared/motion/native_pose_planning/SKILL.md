---
name: native_pose_planning
description: "plan_pose_result tries planner.plan_path for a 7-vector pose target, then falls back to IK-to-joint plus plan_joint; it rejects a pose path whose endpoint differs from current by more than pi on any joint. plan_pose_sequence chains route phases by using each trajectory's final joint as the next start."
---

# Native pose planning

## Function

plan_pose_result tries planner.plan_path for a 7-vector pose target, then falls back to IK-to-joint plus plan_joint; it rejects a pose path whose endpoint differs from current by more than pi on any joint. plan_pose_sequence chains route phases by using each trajectory's final joint as the next start.

## Inputs

- planner object exposing plan_path, solve_ik_to_joint, and plan_joint
- current own joint vector, target 7-vector pose, base pose, and limits array shape (len(current), 2) with finite bounds
- route list of (phase, target) for plan_pose_sequence

## Outputs

- plan_pose_result returns (result or None, diagnostics dict) with mode, acceptance, control count, goal joint, and failure reason where applicable
- plan_pose_sequence returns (list of results or None, diagnostics dict with per-step rows and total_controls)

## Callable interface

Implementation: [implementation.py](implementation.py).

- `plan_pose_result(planner, current, target, base, limits)`
- `plan_pose_sequence(planner, current, route, base, limits)`

## Constraints

- Raises RuntimeError for invalid cuRobo joint trajectory, invalid observed joint limits, or invalid IK joint solution.
- Canonical IK fallback chooses each joint's 2*pi winding within limits nearest current; no in-limit choice returns accepted False with reason no_in_limit_ik_solution.
- Does not run physics or execute actions.
- Uses cuRobo planner methods supplied by the caller.
