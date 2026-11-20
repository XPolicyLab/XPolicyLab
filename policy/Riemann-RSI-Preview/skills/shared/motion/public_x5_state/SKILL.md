---
name: public_x5_state
description: "Validates current public RGB/proprioception into a copied observation and builds an X5 cuRobo-backed PublicRobotManager and PublicPolicyEnvironment. create_environment wires args.public_instruction and horizon into that environment."
---

# Public x5 state

## Function

Validates current public RGB/proprioception into a copied observation and builds an X5 cuRobo-backed PublicRobotManager and PublicPolicyEnvironment. create_environment wires args.public_instruction and horizon into that environment.

## Inputs

- observation: state fields left/right arm_joint_state (6), ee_joint_state (1), ee_pose (7), and vision cameras cam_head/cam_left_wrist/cam_right_wrist color
- recorder, asset resolver, optional planner, args.public_instruction, and horizon
- env_idx_list for get_joint/get_real_endpose must be None or [0]
- plan_endeffector_joint target_val normalized to [0,1] and control_step_num

## Outputs

- public_observation returns {'state': copied finite fields, 'vision': {'color': RGB copies}}
- PublicRobotManager exposes get_joint, get_real_endpose, plan_ee, plan_endeffector_joint, update, and raises on measured jaw requests
- PublicPolicyEnvironment stores robot_manager, dt=0.004, step_lim, take_action_cnt, instruction and returns instruction for env 0
- create_environment returns a PublicPolicyEnvironment

## Callable interface

Implementation: [implementation.py](implementation.py).

- `as_numpy(value)`
- `public_observation(observation)`
- `PublicRobotManager: __init__(self, recorder, observation, *, asset, planner=None); process_name(name); get_joint(self, robot, env_idx_list=None); get_real_endpose(self, robot, env_idx_list=None, is_relative=True); get_end_effector_real_val(self, robot, env_idx_list=None); plan_ee(self, env_idx, arm_tag, result); plan_endeffector_joint(self, env_idx, arm_tag, target_val, need_plan=True, control_step_num=1); update(self, observation)`
- `PublicPolicyEnvironment: __init__(self, manager, instruction, horizon); gen_instruction(self, env_idx)`
- `create_environment(observation, recorder, args, *, asset, horizon)`

## Constraints

- Rejects wrong field shapes, nonfinite values, normalized jaw outside [0,1], and non-uint8 RGB not shaped (480,640,3) or (480,640,4) with ValueError.
- get_end_effector_real_val raises RuntimeError because measured jaw positions are not exposed by official observations.
- Only current public episode and episode-relative pose are supported; other env_idx_list or is_relative=False raises ValueError.
- plan_endeffector_joint interpolates normalized current/target through -0.01+0.054*x and requires curobo.yml/X5A.urdf assets; missing assets raise FileNotFoundError.
