# Robot interface

You control two robot arms on a table through the `robo` command. This file describes the interface.
The task is given in your prompt.

## What you can do

- Run `robo` commands in the shell. This is the only way to act on the robot.
- Look at the camera images in `obs/` with your image viewing tool.
- Read `obs/state.json`.
- Keep notes in `notes.md`.

## World frame

One right-handed frame is used for both arms and all cameras. Units are meters, degrees and seconds.

- Origin: on the floor, midway between the two arm bases.
- +x: to the right, as seen from the robot. +y: forward, away from the robot. +z: up.
- The table top is at z = 0.74.
- Left arm base: x = -0.30, y = -0.45. Right arm base: x = +0.30, y = -0.45.
- In the head camera image, +x points to the right of the image and +y points to the top of the image.
- `frame.png` shows the frame from above.

## Gripper

- TCP (tool center point): the midpoint between the two fingertips. All positions refer to it.
- Tool frame, attached to the TCP: x is the direction the gripper points to (`approach_dir`), y is the axis along which the fingers open (`open_dir`), z = x cross y.
- Gripper value: 1 is fully open (8.8 cm between the fingers), 0 is fully closed. The reported value is the commanded opening; whether something is held is visible in the wrist camera.
- At the start both grippers are open, point forward (+y), and are 0.18 m above the table.
- Reach: pointing straight down, the TCP reaches forward to about y = 0.00 at table height, and less higher up. Pointing `down45` it reaches to about y = 0.15. Targets beyond that fail with `ik_unreachable`.

## Commands

| Command | Effect | Costs budget |
|---|---|---|
| `robo obs` | Refresh `obs/` | no |
| `robo move <arm> [--dx M] [--dy M] [--dz M]` | Translate the TCP in the world frame, orientation unchanged. Each axis is clipped to +-0.20 | yes |
| `robo rotate <arm> [--roll D] [--pitch D] [--yaw D] [--frame world\|tool]` | Rotate about the TCP, position unchanged. Roll, pitch, yaw are rotations about x, y, z of the chosen frame, applied in that order. Each is clipped to +-90. Default frame: world | yes |
| `robo point <arm> <down\|forward\|down45> --open <x\|y\|z>` | Make the gripper point down (-z), forward (+y) or 45 degrees between them, with the fingers opening along the given world axis. `--open` is required. Position unchanged | yes |
| `robo gripper <arm> <open\|close\|0..1>` | Set the gripper opening | yes |
| `robo home <arm\|both>` | Return to the start pose | yes |
| `robo wait <sec>` | Let the simulation run, at most 5 s | yes |
| `robo status` | Print the feedback of the last command again | no |
| `robo done` | Declare the task finished. Ends the episode | no |

`<arm>` is `left` or `right`.

Examples:

- `robo move left --dz -0.05` lowers the left TCP by 5 cm.
- `robo point right down --open x` turns the right gripper to point straight down, fingers opening left-right.
- `robo rotate left --yaw 30` turns the left gripper by 30 degrees about the vertical axis through its TCP.
- `robo gripper left 0.5` opens the left gripper halfway.

## Limits

- Command budget: 60 commands per episode.
- Simulation time: limited per task. Every command that moves the robot or waits uses simulation time. `sim_time_left_s` tells how much is left.
- The simulation only runs while a command runs. Thinking costs no simulation time.
- The episode ends when you run `robo done`, when the budget or the simulation time is used up, or when the environment ends it. After that every command returns exit code 3.
- A command can take up to a few minutes of real time to return.

## Output of a command

Every command prints one JSON object.

| Field | Meaning |
|---|---|
| `requested` | What you asked for |
| `clipped` | true if a value was cut to its limit |
| `workspace_limited` | true if the target was moved back inside the workspace |
| `plan_ok` | false if no motion to the target was found; the arm did not move |
| `plan_fail_reason` | `ik_unreachable`, `self_collision`, `table_collision` or `workspace_limit` |
| `reached_tcp` | Pose of the TCP after the motion: `pos`, `quat` [qw, qx, qy, qz], `rpy` |
| `error_m`, `error_deg` | Distance and angle between the target and the reached pose |
| `settled` | true if the arm came to rest |
| `gripper` | Gripper value after the command |
| `sim_time_s`, `sim_time_left_s` | Simulation time used and left |
| `budget_left` | Commands left |
| `episode_over` | Present and true once the episode has ended |

Exit codes: 0 ok, 1 wrong arguments (no budget used), 2 the command failed (budget used), 3 the episode is over, 4 the client timed out (use `robo status`).

Motion planning knows the arm and the table. It does not know the objects on the table.

## Observations

Commands that move the robot refresh `obs/` before they return.

| File | Content |
|---|---|
| `obs/head.png` | Head camera, 640x480, fixed above the robot, looking forward and down |
| `obs/wrist_l.png` | Camera on the left wrist, 640x480 |
| `obs/wrist_r.png` | Camera on the right wrist, 640x480 |
| `obs/state.json` | State of both arms and the camera parameters |

There is **no depth image**. Positions in 3D come from the colour images and the camera parameters below: a pixel defines a ray from the camera; where that ray meets a plane of known height (the table, see `robo table_plane`) or the ray from a second camera to the same point is where the point is.

`obs/state.json`:

- `left`, `right`: `tcp_pos`, `tcp_quat`, `tcp_rpy` (extrinsic x-y-z, degrees), `approach_dir`, `open_dir`, `gripper`.
- `cameras.<name>.intrinsics`: 3x3 matrix K in pixels.
- `cameras.<name>.extrinsics_world`: 4x4 matrix T that maps camera coordinates to world coordinates. Camera axes: x right, y down, z forward (along the view).
- `cameras.<name>.size`: [width, height].
- `sim_time_left_s`, `budget_left`, `step`.

A world point p is seen at pixel (u, v) with p_cam = inverse(T) * p, u = K[0][0] * p_cam.x / p_cam.z + K[0][2], v = K[1][1] * p_cam.y / p_cam.z + K[1][2].
The other way round, pixel (u, v) is the ray from the camera centre T[0:3, 3] along R * [(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1], 1], with R = T[0:3, 0:3]; on the plane z = h it is the point at distance (h - c.z) / d.z along the ray (c the centre, d the direction).

## Extra tools

Some tasks add commands to `robo`. They are described at the end of this file under "Extra tools for this task"; `robo --help` lists them too.

## Rules

- When the task is finished, return both arms to the start pose with `robo home both`, then run `robo done`.
- Run `robo done` only when you are sure the task is finished.
- If a command fails, read its output before you decide what to do next.
- You are not told the positions of the objects. Work them out from the images and the camera parameters.

## Extra tools for this task

`robo transfer auto|left|right --x X --y Y --z Z --to_x X --to_y Y --to_z Z [--open y|x] [--align_yaw 1|0] [--grasp_depth .022] [--clearance .035] [--margin 0] [--lift_scale 2] [--transit_scale 1] [--transit_speed .20] [--arch 0] [--close_mid .5] [--other home|keep]`
Transfers between world TCP grasp-centre coordinates in metres: refines source xy within 15 mm using calibrated depth, aligns rectangular-top yaw, adjusts insertion/support clearance and destination height, then grasps, lifts, transports, lowers, releases and retracts. Head geometry precedes independent wrist fallback; absent observation API leaves centring unavailable. Auto requires model access and executes one cheapest feasible arm route; retained preflight checks kinematics, not collisions.
Transfer bounds: grasp_depth 0 or [.010,.030] m; clearance [.035,.20] m; margin/arch [0,.10] m; close_mid [.05,.95] (legacy 0 becomes .5); scales [1,4], effective lift/lower scale at least 2; transit_speed 0 or [.05,.50] m/s bounds mean endpoint speed. Observed support imposes 5 mm tip clearance and observed corridor geometry can raise transport height; unobserved clearance is uncertified.
Closure ramps over 16 ticks; sampled drift over 3 mm/.5 degrees rejects. Closure acceptance requires 1 mm/.2 degrees and stationarity, with 2–8 settling ticks (up to 16 for measured relaxation). A supported stationary deflection permits one partial opening for at most six ticks, retaining that aperture on recovery; rejected finite closure attempts full opening without lift. Lift/release stability failures retain closed jaws. Matching RGB/depth at the original source rejects source_not_lifted; absent witnesses never certify retention.
`robo tap left|right --x X --y Y --z Z [--open x|y] [--clearance .045] [--travel .0085] [--dwell .16] [--tip_offset .013] [--contact_scale 3] [--release_scale 3] [--unload .32] [--surface_speed .02] [--finish retract|home] [--other home|keep]`
Tap closes jaws, approaches supplied surface z, targets TCP z+tip_offset-travel, dwells, unloads and retracts; finish=home also returns active joints. travel (0,.025] m; dwell [.04,.4] s; tip_offset [0,.03] m; unload [0,.6] s; scales [1,4]; surface_speed 0 or [.005,.05] m/s. Model-based contact/unload retains a .04 m/s mean-speed floor; instantaneous speed is not bounded.
Both motion commands default other=home; a displaced inactive arm must have open jaws for return. other=keep preserves its pose. Initial unloaded aperture changes can overlap planned approach for up to eight ticks; closure/release remain serial. Execution consumes action steps; estimated route costs omit some overhead.
`robo rest left|right|both` returns initial joints concurrently, with 2–8 settling ticks and .02 rad tolerance; the joint route must be clear.
Returns plan_ok/plan_fail_reason, stages and reached_tcp; transfer adds selected_arm/arm_candidates, effective geometry/timing, source_camera, source_departure, closure/cleanup diagnostics and released; tap adds surface_reached, contact_tip_world, tip_penetration_m and returned_home. Geometry flags do not certify activation; grasp_verified/contact_verified remain false.
Fails on invalid arguments, unsuitable/occluded source geometry, unsafe support correction, infeasible/stale route, inactive_gripper_not_open, source_not_lifted, closure deflection/instability, lift/release instability, tracking error, executor failure or termination. No automatic motion retries; contact failures attempt guarded retraction while execution remains available.

`robo press_surface left|right --u U --v V [--radius 6] [--open x|y] [--travel .012] [--min_displacement .004] [--dwell .16] [--finish home|retract]`
U,V select a horizontal planar surface in the current head image; radius [2,15] pixels fits depth before motion. The witness expands over the connected plane within .06 m, excluding silhouette edges and matching baseline RGB chromaticity.
Closes jaws, approaches vertically, targets calibrated tip travel below the surface, dwells, unloads and retracts. finish=home returns active joints before release observation; retract omits that return. Displaced inactive joints return only with open jaws.
travel [.006,.025] m; min_displacement [.002,travel-.002] m; dwell [.08,.4] s; calibrated tip extension .013 m. Contact uses .04 m/s mean timing; motion consumes action steps.
Returns plan_ok/plan_fail_reason, surface_world, displacement_verified, release_verified, stroke_cycle_verified, per-camera samples, witness appearance and motion diagnostics. Travel needs two consecutive observations from one head/wrist camera; release needs matching appearance within .001 m of the initial plane. Neither certifies identity or an external event.
Fails on invalid selection, missing/misaligned RGB, ambiguous appearance, unsuitable depth, insufficient visible support, unverified travel/release, motion failure or termination. Occlusion stays unverified; motion errors attempt retraction, observation errors return after retraction; no extra holds or retries.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.
