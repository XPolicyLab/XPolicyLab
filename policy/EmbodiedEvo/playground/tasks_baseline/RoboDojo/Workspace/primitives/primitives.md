# RoboDojo native primitives

Submitted Python runs inside one persistent simulator session. Use
`get_instruction()`, `get_observation()` and `step(action)` to read the task,
observe the scene and control the robot. Robot state, Python variables and helper
functions persist across submissions within the episode.

`np` provides array operations, numerical functions, linear algebra and random
sampling. Submitted code supports `numpy` and `math` imports. You can define helper
functions for calculations and control. Local workspace Python can inspect exported
observations.

## get_instruction()

Returns the native episode instruction as a string. Use this instruction to
discover the selected task and its goal. Reading it preserves simulation time.

## get_observation()

Returns a snapshot of the latest native policy observation at the current
simulation time. Snapshot arrays are independent copies suitable for local analysis.

- `instruction`: the native task instruction.
- `vision`: camera dictionaries, normally `cam_head`, `cam_left_wrist` and
  `cam_right_wrist`. Each `color` is a 480 by 640 by 3 uint8 RGB image.
  `shape` records the native dimensions. Camera resolution and names come from the
  task configuration.
- `state`: measured arm joints and absolute end-effector poses, plus the native
  controller's normalized gripper command state.
- `action`: native joint/gripper command dictionary.
- `additional_info.frequency`: native policy action frequency (25 Hz).

The default dual ARX X5 has six joints per arm. Its state keys are
`left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`,
`right_ee_joint_state`, `left_ee_pose`, and `right_ee_pose`. Read the live keys and
array shapes for the current task's embodiment. Poses are
`[x, y, z, qw, qx, qy, qz]` in metres and scalar-first quaternion order in the native
world frame (+z is up). The reported end-effector frame is the target frame accepted
by EE actions. Image coordinates are measured in pixels.

## step(action)

Accepts one complete native action dictionary. Supply finite, one-dimensional
arrays/lists for every controlled arm and gripper. Choose joint mode or EE mode
for each action. Single-arm embodiments use `arm_joint_state`, `ee_joint_state`
and `ee_pose` keys.

Joint mode, dual-arm example (hold measured joints):

```python
obs = get_observation()
action = {k: v for k, v in obs['state'].items() if k.endswith('joint_state')}
obs, reward, terminated, truncated, info = step(action)
```

EE mode, dual-arm example (hold current end-effector poses):

```python
obs = get_observation()
s = obs['state']
action = {
    'left_ee_pose': s['left_ee_pose'],
    'right_ee_pose': s['right_ee_pose'],
    'left_ee_joint_state': s['left_ee_joint_state'],
    'right_ee_joint_state': s['right_ee_joint_state'],
}
obs, reward, terminated, truncated, info = step(action)
```

Arm joint targets are absolute radians. EE targets are absolute world poses.
Gripper entries are one-element normalized arrays: 0 closed, 1 open; upstream
clamps them to this range. Native IK handles EE targets; use the returned pose
to check the achieved motion. One call interpolates/holds the command through the
native 25 Hz control interval (10 physics ticks at 250 Hz).

Returns `(observation, reward, terminated, truncated, info)`:

- `observation`: the resulting observation in the format described above.
- `reward`: the binary official completion signal.
- `terminated`, `truncated`: episode state flags.
- `info['success']`: the same official completion signal as `reward`.

Native success is reported when the official episode-ending check confirms
completion. The episode may also end in failure. After termination, subsequent
calls return the terminal state.

Each scene provides 100 execution slots and the official per-task native action
limit, read from the selected task at runtime. Query `status` for
`native_action_limit`; do not assume a fixed number across tasks.
`status` reports cumulative `native_steps_used` and finite `native_steps_remaining`.
At the limit, the native final check determines success; otherwise `truncated` is
true. Further actions after termination or truncation do not advance physics. Observations and raw per-action RGB
video are exported automatically after each submitted code segment.
