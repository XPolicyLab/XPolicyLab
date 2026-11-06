---
name: single_arm_curobo_planner
description: "FrozenSingleArmPlanner loads a contained owned-robot YAML/URDF configuration, builds cuRobo MotionPlanner and InverseKinematics instances, and plans pose or joint paths and IK solutions for the active joints. Returned trajectories are active-joint position/velocity (and acceleration for plan_joint) arrays."
---

# Single arm curobo planner

## Function

FrozenSingleArmPlanner loads a contained owned-robot YAML/URDF configuration, builds cuRobo MotionPlanner and InverseKinematics instances, and plans pose or joint paths and IK solutions for the active joints. Returned trajectories are active-joint position/velocity (and acceleration for plan_joint) arrays.

## Inputs

- robot_origin_pose, active_joints_name, all_joints, yml_path, optional dt and table_height
- plan_path: curr_joint_pos, target_ee_pose, real_robot_pose, optional 6D constraint_pose
- plan_joint: start_joint_pos and goal_joint_pos
- solve_ik_to_joint: curr_joint_pos, target_ee_pose, real_robot_pose, optional num_seeds

## Outputs

- plan_path/plan_joint return {'status': 'Success' or 'Fail'} plus position and velocity; plan_joint also acceleration
- solve_ik_to_joint returns {'status': 'Success', 'joint_names': active_joints_name, 'joint_value': active-joint solution} or {'status': 'Fail', 'joint_names': active_joints_name}
- successful trajectories are numpy arrays extracted for active joints; missing velocity/acceleration are zero-filled where implemented

## Callable interface

Implementation: [implementation.py](implementation.py).

- `owned_configuration(filename)`
- `FrozenSingleArmPlanner: __init__(self, robot_origin_pose, active_joints_name, all_joints, dt=None, yml_path=None, table_height=None); plan_path(self, curr_joint_pos, target_ee_pose, real_robot_pose, constraint_pose=None); plan_joint(self, start_joint_pos, goal_joint_pos); solve_ik_to_joint(self, curr_joint_pos, target_ee_pose, real_robot_pose, num_seeds: int=32)`

## Constraints

- yml_path is required; owned_configuration requires asset_root_path 'owned_robot_model' and a non-symlink URDF contained in the YAML directory
- constraint_pose must be 6D or ValueError is raised; table_height None clears the scene model; planner uses CUDA graph, warmup 5, max_attempts 5, 32 IK seeds, position tolerance .001, orientation tolerance .02
- joint inputs may be active-joint or all-joint length; unexpected joint or IK solution dimensions raise ValueError
- callers retain their own planner instances, table-height choice, and execution clock; instance methods depend on self and imported cuRobo modules with no standalone invocation
