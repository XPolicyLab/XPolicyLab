---
name: public_dense_execution
description: "Projects the official public observation into detached state/RGB inputs, retains two-frame public memory, maps normalized gripper commands through declared X5 command geometry, and executes dense joint controls through an explicit exchange while preserving sparse fill, fractional sampling, gripper preload and observation-after-ACK ordering. It does not read simulator, reward, success or contact data."
---

# Public dense execution

## Function

Projects the official public observation into detached state/RGB inputs, retains two-frame public memory, maps normalized gripper commands through declared X5 command geometry, and executes dense joint controls through an explicit exchange while preserving sparse fill, fractional sampling, gripper preload and observation-after-ACK ordering. It does not read simulator, reward, success or contact data.

## Inputs

- official observation with exactly six declared state fields and cam_head, cam_left_wrist, cam_right_wrist decoded RGB uint8 images
- static PublicRobotDescription per arm with arm_name, gripper_scale, gripper_sign, gripper_mimic, and gripper_name
- owned planners and 6x2 finite ordered joint_limits per arm
- controls list and phase, explicit exchange(action) callable, time_scale, and integer step_limit

## Outputs

- project_observation dict with copied six state fields and three RGB frames
- PublicObservationMemory state access and two latest retained frames per field/camera
- normalized action dict plus validated PublicStepReply execution updating memory, action logs and gripper_seed_audit
- plan_ee list of joint target dicts on status Success, or gripper controls from plan_endeffector_joint

## Callable interface

Implementation: [implementation.py](implementation.py).

- `NativeEpisodeEnded: exception or data type`
- `PublicObservationSeries: __init__(self); append(self, frame)`
- `PublicObservationMemory: __init__(self, observation, output_root); update(self, observation, *, sim_step); state(self, key)`
- `project_observation(observation)`
- `PublicRobotDescription: command_pair(self, normalized_value)`
- `PublicRobotManager: __init__(self, memory, robots, planners, joint_limits); process_name(name); get_joint(self, robot, env_idx_list=None); plan_ee(self, env_idx, arm_tag, result); plan_endeffector_joint(self, env_idx, arm_tag, target_val, need_plan=True, control_step_num=100)`
- `PublicStepReply: exception or data type`
- `PublicExecutionClock: __init__(self, robot_manager, step_limit)`
- `PublicCommandBridge: __init__(self, memory, robot_manager, exchange: Callable, *, time_scale=1.4, step_limit=1400); execute(self, controls, phase)`

## Constraints

- Gripper feedback must be normalized [0,1] and is treated as a limited command target, never as measured physical finger width or contact; audit records actual_width=None and contact_evidence=unknown
- Only env index 0/current public episode is available; need_plan must be True; gripper interpolation uses 90% of control_step_num steps then holds the target
- step_limit must be an int from 1 to 1400; exchange must return PublicStepReply with boolean episode_ended; local action counter is not official termination
- State keys must match exactly six declared fields; images must be uint8 with 3 or 4 channels; memory retains only two latest frames and copies inputs

## Provenance

Project dense-command implementation previously used at Git source `3ccc9458df1f22b1ff415ef13a424dd2e8730559`.
