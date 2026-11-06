---
name: normalized_pair_joint_execution
description: "Projects and retains detached public observations, validates X5 robot command geometry and normalized gripper plans, and replays dense joint controls through an explicit public exchange with sparse fill, fractional sampling, gripper preload, and observation-after-ACK ordering."
---

# Normalized pair joint execution

## Function

Projects and retains detached public observations, validates X5 robot command geometry and normalized gripper plans, and replays dense joint controls through an explicit public exchange with sparse fill, fractional sampling, gripper preload, and observation-after-ACK ordering.

## Inputs

- observation: mapping with exactly six finite robot state fields and three uint8 RGB cameras (cam_head, cam_left_wrist, cam_right_wrist)
- robots: exactly two PublicRobotDescription entries for left_arm/right_arm; planners: owned numeric planners keyed by robot_name; joint_limits: 6x2 public URDF limits
- normalized gripper target in [0,1] plus control_step_num; controls: list of joint-control dicts; phase: recorded phase label
- exchange: callable returning PublicStepReply; time_scale: finite positive; step_limit: integer 1..1400

## Outputs

- project_observation returns a detached dict with copied state and RGB vision arrays
- PublicObservationMemory retains only the two latest frames per series and returns copies from state()/indexing; len() reports total appended frame ordinal
- plan_ee returns joint controls only for status 'Success'; plan_endeffector_joint returns normalized gripper command controls with 90% interpolation and final hold
- PublicCommandBridge.execute returns None but appends normalized_actions, action_sim_steps, and action_phases, updates memory after each ACK, and raises NativeEpisodeEnded on episode end or budget exhaustion

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

- Only env_idx 0 is accepted; exactly two arms and only the six declared state widths are allowed; gripper feedback must be normalized in [0,1]; PublicExecutionClock requires step_limit 1..1400; bridge requires callable exchange and finite positive time_scale.
- Raises ValueError for malformed observations, missing planners/limits, invalid controls, unknown keys, or non-finite/shape-mismatched targets; TypeError if exchange reply is not PublicStepReply with bool episode_ended.
- PublicStepReply carries public observations and the native episode-ended bit only, not reward, success, score, collision, or articulation data; gripper observations are normalized limited command targets, never measured physical fingers or contact.
- Memory retains no metadata and only two frames; joint limits are read-only copies; episode end or native action-budget exhaustion raises NativeEpisodeEnded.

## Provenance

Project dense-command implementation previously used at Git source `3ccc9458df1f22b1ff415ef13a424dd2e8730559`.
