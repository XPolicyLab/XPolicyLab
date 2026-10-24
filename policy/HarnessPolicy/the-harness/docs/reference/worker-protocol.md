# Worker protocol

A worker is a robot, simulated or real, behind HTTP: `harness serve --factory
module:create_environment [--options FILE] [--port P] [--contract C --embodiment E]`
(`src/harness_policy/core/worker.py`). One request at a time; loopback by default
(binding another interface needs `HARNESS_WORKER_TOKEN`, sent as a bearer token).
`src/harness_policy/core/robot.py` (`RemoteEnvironment`) is the client.

```text
POST /rpc   {"method": "...", "arguments": {...}}   ->   {"result": ...} | {"error": "..."}
```

| Method | Arguments | Result |
|---|---|---|
| `describe` | | `{protocol_version: 1, name, tools}` |
| `reset` | `task`, `seed` | the first observation: `{episode_id, step, instruction, state, images}` |
| `call` | `name`, `arguments` | the result of one of the environment's methods, below |

## The robot API (`call`)

| Name | Arguments | Result |
|---|---|---|
| `profile` | | static facts: joint groups, action limits, control rate, cameras, arm bases, gripper constants, planner limits, shield options, `return_to_start`, the step limit (native contract) |
| `state` | `episode_id` | `step`, `ended`, `joint_position` (14 on the dual ARX X5: left 6 joints, left gripper, right 6 joints, right gripper), `left_ee_pose`, `right_ee_pose` (link6, `[x, y, z, qw, qx, qy, qz]`, env frame) |
| `tick` | `episode_id`, `step`, `command` (absolute joint targets), optional `cameras` | `{executed, shield: {verdict, command, reasons}, state[, observation]}` |
| `observe` | `episode_id`, `cameras`, `image_format`, `quality` | `{state, images (data URLs), intrinsics, camera_to_world (OpenCV convention), image_size}` |
| `fk` | `command` (joints) | `{arm: link6 pose}` |
| `ik` | `episode_id`, `arm`, `pose` | `{success, joints}` from the current joints |
| `plan` | `episode_id`, `arm`, `pose` or `joints`, `raw` | `{success, waypoints}` at the control rate (`raw`: the planner's own path) |
| `policy_obs` | `episode_id` | the observation RoboDojo's evaluation client hands a policy (for `run_policy`) |
| `claim_outcome` | `episode_id` | a token, claimed once by the driver after reset |
| `outcome` | `token` | `{success, reward, process_score, terminated, truncated, steps}` |
| `action_spec` | | RoboDojo only: per arm its mount pose, joint limits and velocity limits; control rates |

Rules every worker keeps:

- **Episode and step.** `tick` names the episode and the step it was computed for; a
  stale one is rejected and never executed again. A possibly executed motion is never
  retried silently.
- **The shield runs here.** Every `tick` command is checked before it reaches the robot
  ([The episode runtime](../concepts/runtime.md#the-shield)); programs cannot bypass it.
- **The outcome is guarded.** Only the holder of the token (the driver) reads it;
  programs and the model never see it.

## Contracts

`--contract xpolicylab --embodiment arx_x5` filters every result to what XPolicyLab's
official client hands a policy: grippers as their last commands, intrinsics and
`camera_to_world` from the embodiment, no step limit in the profile or the state
(`src/harness_policy/core/contract.py`). The profile then says
`observation_contract: "xpolicylab"`, `gripper_feedback: "commanded"`.

## Implementations

| Factory | Bench |
|---|---|
| `benches.robodojo.runtime:create_environment` | RoboDojo, one task per process ([Benches](benches.md)) |
| `harness_policy.connectors.xpolicylab.Bridge` | not served by `harness serve`: the XPolicyLab connector, a worker whose robot is XPolicyLab's client |
