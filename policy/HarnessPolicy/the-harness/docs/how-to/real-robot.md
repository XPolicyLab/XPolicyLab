# Run on a real robot

A real robot is reached the way RoboDojo-Real reaches it: the agent is exported as an
XPolicyLab policy folder, and the evaluator's client drives the robot and calls the
policy. The agent does not change; the robot is an embodiment profile and a kinematics
service for it.

## 1. The embodiment profile

`embodiments/<name>/embodiment.toml` holds what the agent may know about the robot:
joint groups and limits, the fingertip offset and tool axis, gripper widths, control
rate, planner limits, the shield's limits and the cameras. `embodiments/arx_x5` is
RoboDojo's simulated ARX X5; `embodiments/piper` is a pair of AgileX PiPER arms with the
facts of the PiPER URDF and `calibrated = false`.

A profile is calibrated on its rig before a real run. Measure and write:

| Key | What to measure |
|---|---|
| `arm_bases` | each arm's mount pose in the env frame |
| `table_z` | the table top's height in the env frame |
| `[shield] workspace` | the box the fingertips may reach without hitting the rig |
| `[calibration.<camera>]` | intrinsics and camera_to_world (head camera) or camera_on_link6 (wrist cameras), OpenCV convention |
| `gripper_empty_opening`, `gripper_full_stroke_seconds` | the reading of an empty closed gripper, and the time of a full stroke |
| `control_hz` | the rate at which the evaluator's client asks for actions |
| `kinematics_service`, `planner_world` | a service of kind `kinematics` (fk, ik, plan_path) for this robot and its planner's world; `services/kinematics.toml` is the ARX X5's (RoboDojo's cuRobo planner with `x5/curobo.yml`) |

Then set `calibrated = true`. `harness export xpolicylab --embodiment <name>` refuses an
uncalibrated profile, and a profile other than the ARX X5 without its own
`kinematics_service`: the shield's limits and every camera-based skill depend on these
values, and the shield's workspace and Cartesian checks run the kinematics service's fk.

## 2. Export and run

```bash
harness export xpolicylab --name HarnessPolicy --out XPolicyLab/policy/HarnessPolicy --embodiment piper
```

The folder is run like any XPolicyLab policy on the robot's client. A human holds the
emergency stop during every trial; the run's traces record every shield rejection.

## What works on a new robot

With a calibrated profile and its kinematics service, programs in joint space, guards,
handlers and the shield run on the new robot: the shield's joint bounds are the
profile's `action_low`/`action_high`, its workspace and jump checks go through the
robot's own fk. `run_policy` runs a learned policy on any robot, but RoboDojo's released
checkpoints (Simate, G0.5, ...) were trained on the simulated ARX X5: on another robot a
policy trained for it is needed. The analytic skills (locate, pick, release, push) also
need the robot's tool frame: they assume the ARX X5's tool along link6's x axis, so a
robot whose tool points along another axis (the PiPER's `tool_axis_link6 = [0, 0, 1]`)
runs them once the geometry reads that key. The prompts (`prompts/robodojo`) describe the
ARX X5's workspace.

The identity names the agent, so it is the same for every robot; `agent.json` records the
embodiment the folder was exported for, and the README's commands and `--check` use its
XPolicyLab `env_cfg_type`.
