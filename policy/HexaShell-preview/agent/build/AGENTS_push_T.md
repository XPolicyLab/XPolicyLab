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

`robo planar_match --u U --v V --ref_u U --ref_v V [--camera head|wrist_l|wrist_r] [--color_tol 45] --surface_z Z --reference_z Z [--height_fit auto|off]`: free registration of two similarly sized horizontal color regions.
color_tol accepts 5–100. Both finite measured world-Z planes are required; RGB silhouettes are projected with calibrated rays, without depth or guessed heights. With reference_z, auto refines source height within 4 cm/15% scale assuming equal-size outlines; off fixes supplied planes. Returns plane_sources, reference_z, height_fit diagnostics, fit_rms_m, color_tol_used, centers, seed_xyz/seed_goal_xy/open_deg and yaw. contact_available adds contact_xyz/contact_goal_xy/contact_open_deg/contact_width_m and coherent transfer_args with outline; invalid regions, unequal areas or fit >5 mm fail.
`robo planar_height --u U --v V --other_u U --other_v V --z Z [--camera head|wrist_l|wrist_r] [--other_camera wrist_l|head|wrist_r] [--color_tol 45]`: free height fit of the same complete region in two simultaneous calibrated views over Z±.06 m. Returns surface_z, height_uncertainty_m, cross_view_rms_m, plane_source, surface_areas_m2 and area_ratio from calibrated pixel footprints; rejects weak parallax, color/area mismatch, RMS >2.5 mm, uncertainty >12 mm or boundary minima.
`robo planar_transfer left|right --x X --y Y --z Z --to_x X --to_y Y --open_deg A --yaw A [--contact_z Z] [--outline TOKEN] [--clearance .07] [--inset .006] [--relay auto|off] [--verify auto|off]`: closes, translates/rotates in <=90° sweeps, releases and retracts; motions cost simulation time.
Coordinates are world meters; angles are degrees. open_deg is the XY finger-opening axis; yaw is relative counterclockwise rotation about contact. Clearance .03–.15, inset 0–.012, translation <=.6 m, yaw ±180°. z is the geometry/verification plane; contact_z defaults to z and sets execution height within ±.04 m. Contact is approached at contact_z-inset; a 2–12 mm upward descent error permits one unload and transport at reached Z.
relay=auto splits crossings at the observed TCP XY bisector and measures handoff slip; off uses one arm. Returns stages, reached_tcp, relay_completed, final_arm and handoff diagnostics. One unexecuted setup IK rejection permits bounded recovery.
verify=auto tracks complete RGB outlines projected on the supplied z using compatible calibrated views and bounded color/seed searches; acceptance is <=3 mm/2°. Opaque outline preserves material geometry and camera origin; initial freshness requires <=6 mm/3°. Shape-validated tracking tolerates bounded lighting changes; alternate views require <=3 mm parallax disagreement under .04 m plane uncertainty (legacy tokens omit this gate). Unavailable released views permit one guarded entry-pose return or short-displacement upward clearance, then recheck; off skips verification.
A measured residual <=12 mm/6° with >=8 s remaining permits one refinement from measured pose directly to the original goal. Returns alignment_verified, alignment_scope=requested_transform (not independent destination validation), registration_z/contact_z, residual_contact_m/residual_yaw_deg, correction_args preserving outline/goal and refinement diagnostics. Transfers with verify=auto also return check_args on release-visibility failure.
`robo planar_check left|right --x X --y Y --z Z --to_x X --to_y Y --open_deg A --yaw A --outline TOKEN [--contact_z Z] [--clearance .07] [--inset .006]`: free observation-only verification accepting check_args; returns alignment_verified, residuals and correction_args against the carried original transform. Same tracking/alignment gates; no motion, clearance, recapture or refinement.
All commands return plan_ok/plan_fail_reason. Invalid arguments, unavailable/incompatible measurements, alignment residual, motion/position failure or exhaustion fail; missing initial visibility fails before motion. Motion failure stops execution and may leave fingers closed.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.
