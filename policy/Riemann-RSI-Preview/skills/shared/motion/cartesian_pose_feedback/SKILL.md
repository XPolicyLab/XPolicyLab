---
name: cartesian_pose_feedback
description: "Adapter-bound mixin that provides home-joint planning and control merging plus pose planning and bounded feedback for arm motion. It retries only exact cuRobo pose-planning or distant-winding failures through canonical IK, then executes merged pose and gripper control sequences."
---

# Cartesian pose feedback

## Function

Adapter-bound mixin that provides home-joint planning and control merging plus pose planning and bounded feedback for arm motion. It retries only exact cuRobo pose-planning or distant-winding failures through canonical IK, then executes merged pose and gripper control sequences.

## Inputs

- Injected adapter state: self.home_joints, self.robots, self.args, self.phase, self.rm, self._plan_pose_for(robot, pose), self._ik(arm, pose), self._actual_ee_pose(robot), and self._official_bridge
- self.rm must provide get_joint(robot, env_idx_list=[0]), planner[robot.robot_name].plan_joint(current, goal), plan_ee(env_idx, arm_tag, result), process_name(arm), and immutable joint_limits[arm] in arm order
- _move(arm, target, phase, grip, precise=False, insertion=False, initial_plan=None): target pose is converted to float32; precise and insertion modes alter attempts and tolerances

## Outputs

- _plan_joint_home and _plan return lists of control dictionaries; _merge_controls returns merged dictionaries with last-point extension
- _move returns None on tolerance success or raises PosePlanningError, PoseFeedbackError, or a propagated RuntimeError

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PosePlanningError: exception or data type`
- `PoseFeedbackError: __init__(self, phase, position_error, rotation_error)`
- `NeutralArmMotion: exception or data type`

## Constraints

- Only RuntimeErrors beginning with 'cuRobo plan_path failed for pose=' or equal to 'Pose plan uses a distant joint winding; retry canonical IK' trigger IK fallback; CUDA, recorder, and other faults propagate.
- Canonical IK enumerates +/-2pi endpoints for k in range(-3, 4), filters against immutable joint limits, and selects the endpoint nearest the current joint value.
- Precise feedback allows args.pose_corrections + 1 attempts; position tolerance is args.insertion_tolerance for insertion otherwise args.pose_tolerance, rotation tolerance is 0.5 deg for insertion otherwise 3 deg; errors above 0.04 m or 20 deg stop feedback.
- The code relies on adapter globals and self plus the injected public planner; it adds no initializer, controller, geometry, speed, budget, recovery, SDK, or ground-truth dependency, and does not use the align_blocks servo.
