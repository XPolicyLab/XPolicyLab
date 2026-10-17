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
| `robo done` | End the episode now (see Rules) | no |

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

- The episode ends by itself the moment the task is judged complete. While it is still running, the task is not complete.
- When you think the task is finished, return both arms to the start pose with `robo home both`. If the episode is still running after that, compare the scene with every condition in the TASK RECIPE, fix what is missing, and home again.
- Partial results count (see the score table in the TASK RECIPE). While simulation time is left, keep working on the parts that are not done.
- Do not end your reply while the episode is running and simulation time is left.
- `robo done` stops the episode where it is. Run it only when the time left cannot cover one more attempt.
- Before you give up on an attempt, compare `sim_time_left_s` with the time the last similar attempt took (the change in `sim_time_s`).
- If a command fails, read its output, run `robo obs` and look at the new images before you try again. Measure again instead of repeating the same command with the same numbers.
- You are not told the positions of the objects. Work them out from the images and the camera parameters.

## Extra tools for this task

`robo surface --u U --v V [--camera head|wrist_l|wrist_r]`: free RGB silhouette stereo measurement of a compact colored patch within 12 px; default head. Tests complete components touching the pixel neighborhood, with locally selected hues.
Returns `plan_ok`, `plan_fail_reason`, `center_xy`, `top_z`, `extent_xyz`, `grasp_xyz`, `grasp_offset_xy`, `open_axis`, `cross_section_xy`, hue and pixel count. Requires calibrated overlapping views; weak, inconsistent, occluded or non-horizontal silhouettes fail. Failure detail distinguishes absent local color from rejected stereo height.
`grasp_offset_xy` is grasp minus footprint center; rotate this offset by yaw, then add it to desired center xy for destination TCP xy. Heights are measured by cross-view silhouette alignment, without depth input.
`robo locate --u U --v V [--camera head|wrist_l|wrist_r] [--z Z]`: free pixel-ray intersection with a horizontal plane; omitted z uses the RGB-measured table plane. Returns `xyz`, `plane_z`; invalid rays or unavailable plane measurements fail. Supplied z must be a known surface height.
`robo transfer <left|right> --x X --y Y --z Z --to_x X --to_y Y --to_z Z [--open x|y] [--approach down|down45] [--clearance M] [--seat M] [--landing auto|off] [--source_plane Z] [--yaw DEG]`
Absolute world TCP endpoints in metres; active gripper starts open. Defaults: open=x, approach=down, clearance=0.045 (0.025–0.15), seat=0.003 (0–0.004), landing=auto. Down45 tilts forward 45 degrees. Yaw defaults to 0; nonzero yaw requires down and source_plane, range ±180°, positive about world +z.
Executes bounded idle-arm clearance, orientation, approach, grasp, lift, carry, release and retraction. Yaw turns at elevated destination TCP after verified carry; rotated RGB verification must pass before lowering. Motions consume action steps. Observed seating lowers both endpoints equally; auto landing adjusts release only with measured support evidence, retaining explicit endpoints and reporting unverified if support measurement fails.
Returns `stages`, `visual_checks`, `released`, `plan_ok`/`plan_fail_reason`, effective endpoints and reached TCP. Checks distinguish `visible_at_tcp`, `lost`, `unverified`; absent perception never certifies retention.
Source stereo starts at the projected grasp pixel; excessive stereo work fails without returning partial geometry. Missing source evidence stops before contact; uncertain lift permits one 15 mm verification rise, then stops closed if unverified. Visible loss stops; wrist-only carry evidence requires a released-placement check.
After an unmoved carry IK refusal, one measured-clear lower route may be tried; otherwise restores/releases at source once and reports `plan_detail.source_release`. Other motion failures or episode end stop; no automatic regrasp.
Optional `source_plane` supplies a measured static source height within 12 mm of grasp z; captures a complete RGB silhouette and checks rigid-motion projections against stationary/grounded alternatives. Lift/carry may fit a shared ≤6 mm translation with two calibrated views and equally fitted stationary/grounded competitors; `registration` reports evidence. Release checks stay strict; ambiguity stops, and projected height is not measured height.

`robo stereo_point --u U --v V --other_u U2 --other_v V2 [--camera head|wrist_l|wrist_r] [--other_camera head|wrist_l|wrist_r]`: free RGB triangulation of the same physical point in two current views; defaults head/wrist_l.
Returns world `xyz` in metres, `ray_gap_m`, `ray_angle_deg`, `reprojection_error_px`, `one_pixel_sensitivity_m`, and `plan_ok`/`plan_fail_reason`. Caller supplies corresponding pixels; identity is not verified. Silhouette extrema in differing views need not correspond.
Rejects invalid pixels, weak baseline, rays behind cameras, gap >4 mm, reprojection error >2 px or one-pixel sensitivity >8 mm. No plane assumption or motion.
`robo plane_patch --u U --v V --z Z [--camera head|wrist_l|wrist_r] [--target_x X --target_y Y]`: free compact color-component measurement by ray intersection with an explicitly supplied measured horizontal surface height.
Returns `center_xy`, `extent_xyz`, `grasp_xyz`, `grasp_offset_xy`, `open_axis`, `cross_section_xy`, hue, pixel count and status. `top_z` repeats supplied Z; this command does not measure height. Default camera=head. Also returns geometric `axis_angle_deg` from world +x modulo 180°, `axis_anisotropy`, and `axis_period_deg`; angle is null below anisotropy 1.2, and does not identify semantic uprightness.
Optional paired target coordinates return `center_correction_xy` (target minus observed center) and `center_error_m`. Destination TCP xy equals desired center plus `grasp_offset_xy`.
Rejects nonfinite arguments, missing paired target, absent color, broad components and image-clipped components. Requires a fully visible isolated silhouette on the supplied plane; occlusion, side walls and connected same-color surfaces can bias the result. No motion or retention certification.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

`robo transfer_reach --x X --y Y --z Z --to_x X --to_y Y --to_z Z [--arm both|left|right] [--approach both|down|down45] [--open x|y] [--clearance M] [--yaw DEG]`: free read-only IK route query; no motion or action steps.
Endpoints are absolute world TCP coordinates in metres. Defaults: arm=both, approach=both, open=x, clearance=0.045 (0.025–0.15).
Checks orientation, elevated approach, descent, lift, carry, lowering and retraction from current joints for each requested arm/preset; down45 includes elevated empty departure when needed. Active gripper must be open.
Returns `candidates` with `estimate_ok`, failed stage/reason or nominal `transfer_action_steps`, `stage_costs`, remaining steps and scene reference; `cheapest_reachable` selects the lowest estimated cost among reachable candidates.
Returns `plan_ok=false` if no route passes or arguments/calculation fail. Reachability is IK only: no collision, grasp, retention or placement certificate.
Literal endpoints only: no seating or landing adjustment. Optional yaw defaults to 0 (±180° about world +z); nonzero yaw excludes down45 and adds elevated destination rotation plus rotated lowering/retraction. Cost includes close/open but excludes parking, visual probes, recovery and home. Any physical command, failure or reset invalidates the estimate.

`robo transfer_patch --u U --v V --source_plane Z --target_x X --target_y Y --to_z Z [--camera head|wrist_l|wrist_r] [--arm auto|left|right] [--approach auto|down|down45] [--open auto|x|y] [--clearance M] [--grasp_dz M] [--yaw DEG]`
Selects a compact static RGB silhouette near a pixel on an explicitly supplied measured horizontal height; camera defaults to head. Source height is supplied, never inferred.
Target xy is desired footprint center in world metres; measured grasp offset is rotated by yaw and added automatically. `to_z` is absolute release TCP height. Yaw defaults to 0; ±180° about world +z, counterclockwise viewed from above. Nonzero yaw requires down (auto excludes down45), executes at elevated destination TCP, and requires RGB rotation verification before release.
Defaults: arm/approach/open=auto, clearance=0.045 (0.025–0.15), grasp_dz=0 (±0.010). Auto opening uses measured cross-section geometry; grasp_dz offsets source TCP from source_plane.
Checks complete nominal routes for open candidate arms before motion; auto prefers reachable down over down45, then lower estimated action cost. No reachable route or missing source geometry stops before motion.
Executes one visually checked grasp/lift/carry/release/retract via `transfer`, with seating and automatic landing correction disabled. Retention, clearance, probe, recovery and released-placement checks match `transfer`; no automatic second grasp or arm retry.
Costs one command plus all executed action steps. IK estimates exclude collision, parking, probes and recovery; they do not certify retention or sufficient remaining budget.
Returns transfer status/stages/released plus `patch_transfer` containing measured patch, TCP endpoints, reach candidates and selected route. Invalid input, uncertain retention, motion refusal or placement uncertainty returns `plan_ok=false` with `plan_fail_reason`.
