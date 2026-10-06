# RoboDojo native primitives

Submitted Python runs inside one persistent host simulator. Only the four functions
below are environment primitives. `np` provides the documented restricted NumPy
operations; imports, files, networking and simulator internals are unavailable in
submitted code. Local workspace Python may inspect exported observations.

## get_instruction()

Returns the native episode instruction as a string. It does not advance physics.
Use this instruction to discover the selected task; this workspace is generic.

## get_observation()

Returns a detached copy of the latest native policy observation without advancing
physics. Arrays may be edited locally without changing simulator state.

- `instruction`: the native task instruction.
- `vision`: camera dictionaries, normally `cam_head`, `cam_left_wrist` and
  `cam_right_wrist`. Each `color` is a 480 by 640 by 3 uint8 RGB image.
  `shape` records the native dimensions. Camera resolution and names come from the
  task configuration. Depth and camera calibration are not enabled by this surface.
- `state`: measured arm joints and absolute end-effector poses, plus the native
  controller's normalized gripper command state.
- `action`: native joint/gripper command dictionary.
- `additional_info.frequency`: native policy action frequency (25 Hz).

The default dual ARX X5 has six joints per arm. Its state keys are
`left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`,
`right_ee_joint_state`, `left_ee_pose`, and `right_ee_pose`. Read live shapes instead
of assuming the same embodiment for every task. Poses are
`[x, y, z, qw, qx, qy, qz]` in metres and scalar-first quaternion order in the native
world frame (+z is up). The reported end-effector frame is the target frame accepted
by EE actions. Images are pixels, not world coordinates.

## step(action)

Accepts one complete native action dictionary. Supply finite, one-dimensional
arrays/lists for every controlled arm and gripper. The dictionary selects one mode;
do not mix joint and EE targets. Single-arm embodiments use the same names without
`left_`/`right_` prefixes.

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

Arm joint targets are absolute radians. EE targets are absolute world poses,
not displacements. Gripper entries are one-element normalized arrays: 0 closed,
1 open; upstream clamps them to this range. Native IK handles EE targets and may
leave an arm unchanged if it cannot solve the target. One call interpolates/holds
the command through the native 25 Hz control interval (10 physics ticks at 250 Hz).
There is no automatic planner or grasp primitive exposed to you.

Returns `(observation, reward, terminated, truncated, info)`. `reward` is the binary
official completion signal, and `info['success']` is the same signal. No private
shaped reward or object ground truth is exposed. Native success is reported only
when the official episode-ending check confirms completion. A native failure may
terminate the episode without success. Native actions obey the official per-task
`step_lim`, read from the selected task at runtime. Query `status` for
`native_action_limit`; do not assume a fixed number across tasks.
The Harness also limits accepted code submissions to 100. `status` reports cumulative
`native_steps_used` and finite `native_steps_remaining`. At the action limit, the
native final check determines success; otherwise `truncated` is true. Further actions
after termination or truncation leave physics unchanged. Observations and raw per-action RGB video are exported
automatically after each submitted code segment.

## reset()

Playground only: restores this session's same native layout and instruction.
Returns `(observation, info)`. Python variables and cumulative action counts persist.
Reset replenishes the official native action allowance for a new attempt, including
after action-budget truncation. Used execution slots and elapsed time are not reset: the
entire Playground session shares 100 execution slots by default. A standalone reset
request uses one slot; a reset inside submitted code shares that execution's slot.
`native_steps_used` remains cumulative telemetry; `native_steps_remaining` returns
to the official task limit.
This operation is forbidden in Test, including calls through aliases.
No seed or scene-selection arguments are accepted.
