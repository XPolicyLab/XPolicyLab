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

`robo rigid-place {left,right} --cx FLOAT --cy FLOAT --cz FLOAT --ax FLOAT --ay FLOAT --az FLOAT --length FLOAT --x FLOAT --y FLOAT --z FLOAT [--clearance 0.025] [--depth 0] [--tcp_clearance 0.04] [--release 0] [--release_engagement 0.25] [--retreat 0.06] [--speed 2] [--gripper_steps 6]`
Moves a held elongated feature upright and lowers its negative-axis end; world coordinates are metres. cx/cy/cz is the current full midpoint; signed ax/ay/az maps to +Z; length is full extent, 0.01–0.5 m.
x/y/z is the destination surface point; depth is 0–length. clearance (0.005–0.2 m) margins rotation/transfer; tcp_clearance (0–0.2 m) limits lowering by minimum TCP height above z.
release=0 retains grasp; 1 requests max(depth, release_engagement×length), opens after sufficient reached rigid engagement, then withdraws opposite approach. release_engagement: 0–0.5; retreat: 0.03–0.15 m; speed: 0.5–2; gripper_steps: integer 6–12 at 25 Hz.
Calibrated RGB silhouettes estimate yaw/transit clearance; the rear-hand target is 65 mm. Empty evidence may trigger one post-lift refresh; unverified monocular rays rank yaw only, with reached-state original-path fallback if preferred paths reject. Clearance is not certified.
An eligible commanded-open peer may withdraw outward up to 0.15 m before active motion; peer failure or active drift stops. A failed command may already have moved the peer.
Available IK estimates preflight through lowering with bounded source/interior/partial-turn alternatives and cost ranking within the first reachable family. Motionless rejection permits bounded remainder/split recovery; executed failure stops at errors over 8 mm/0.08 rad or exhaustion.
Terminal RGB stereo permits one 3–25 mm lateral correction; resolved residual over 25 mm lateral or 15 mm axial fails. Temporal fallback requires >=20 mm virtual baseline, >=10° parallax and <=10° TCP reorientation; new current color references require simultaneous stereo.
Missing/ambiguous terminal evidence preserves supplied geometry. Full extent, endpoint identity, rigid retention and physical seating remain unverified; accepted IK does not certify collision freedom.
Returns plan_ok/plan_fail_reason, stages, planned_depth_m, effective_depth_m, minimum_release_depth_m, released, reached_tcp and yaw_selection with clearance/preflight/terminal diagnostics; reached predictions include center, axis and lower end under rigid retention.
Invalid arguments, unreachable chains, tracking errors or insufficient reached engagement fail. Failure after opening can return released=true; engagement and predicted geometry do not measure slip or contact.

`robo axis-grasp {left,right} --x FLOAT --y FLOAT --z FLOAT --ax FLOAT --ay FLOAT [--clearance 0.035] [--rotation_clearance 0.10] [--speed 2] [--gripper_steps 6]`
Approaches from above, aligns finger opening perpendicular to a horizontal feature axis, descends and closes; no final lift.
x/y/z is the world grasp point in metres; ax/ay is a nonzero horizontal vector of either sign. clearance is 0.02–0.2 m; rotation_clearance is 0.06–0.2 m.
Low starts withdraw vertically before turning; descent preserves orientation. Available IK estimation checks complete approach/descent chains for both finger orientations and split paths before motion; no reachable chain returns grasp_chain_unreachable.
Without estimation, motionless approach rejection permits bounded alternate-orientation/split recovery. Estimation certifies IK only; execution failure stops.
An accepted descent stopping over 2 mm high with at most 3 mm lateral error makes one fixed-orientation unload to 3 mm above reached height before closing.
Stops before closing on planner failure, position error over 8 mm (2 mm during unload), rotation error over 0.08 rad or exhaustion; invalid arguments fail without motion.
speed is 0.5–2 (default 2), preserving settling; gripper_steps is integer 6–12 (default 6 at 25 Hz) for opening/closing holds.
Returns plan_ok/plan_fail_reason, stages, preflight when available, closed, reached grasp_point_world, requested_grasp_point_world and supplied axis_world. grasp_verified=false: closure does not certify retention.

`robo return-home {left,right,both} [--speed 2]`
Returns selected arms simultaneously to their recorded initial joint angles using eased joint interpolation and an eight-step final hold.
`speed`: motion speed multiplier relative to base home (0.5–2, default 2); nominal joint duration rate is 1.2 times speed rad/s at 25 Hz.
Requires selected hands already commanded open; preserves gripper commands. Joint interpolation does not plan around obstacles.
Returns plan_ok, plan_fail_reason, joint_error_rad per arm, planned_steps, planned_duration_s and episode_over after execution.
Fails without motion for invalid arguments, unavailable initial angles, closed hand commands or an ended episode; fails after motion if any final joint error exceeds 0.03 rad.

`robo stereo-feature {point,segment,body,line,horizontal,plane} --peer CAMERA --u FLOAT --v FLOAT --pu FLOAT --pv FLOAT [--camera head] [--u2 FLOAT --v2 FLOAT --pu2 FLOAT --pv2 FLOAT] [--color_tolerance 65]`
Read-only calibrated RGB stereo; no depth input, assumed height, motion or action steps. Coordinates are world metres and pixels; cameras accept head, wrist_l, wrist_r, their .png names or observation source names.
point matches u/v with pu/pv and returns point_world; segment additionally matches u2/v2 with pu2/pv2, returning endpoints_world, center_world, length_m and input-order signed axis_world. Selections must identify the same current physical features.
body seeds the same uniform-color elongated span in both views and uniquely pairs fitted endpoints. line seeds the same straight colored axis, requiring an unclipped primary span; peer may be shorter/clipped, with extrapolation capped at half its span.
Body/line/horizontal return center_world, endpoints_world, arbitrary-sign axis_world, visible_length_m and region diagnostics; full_geometry_verified=false. Horizontal assumes a level axis, fits height jointly without endpoint pairing, and returns horizontal_height_m; peer may be shorter/clipped. Supplying all u2/v2/pu2/pv2 instead marks two centerline points per view, bypassing segmentation; points must lie on the same physical axis, but need not correspond across views. Returns axis_selection=explicit_centerlines and silhouette_verified=false; RGB support is unverified. Line/horizontal measure primary-view extent only.
color_tolerance is 5–150: RGB distance for body/line/horizontal, normalized-color distance min(0.25, color_tolerance/500) for plane. Automatic body/line/horizontal require aspect ratio ≥3, span ≥16 pixels and regular silhouettes; body requires unique pairing. Explicit horizontal axes require all endpoints in bounds and spans ≥16 pixels; color_tolerance does not fit their pixels.
plane uses median 3×3 chromatic seeds on the same complete colored rectangular outer surface in both views; triangulates boundary corners, excluding interiors. Rejects clipped/small/nonrectangular/nonplanar or ambiguous boundaries; common full-boundary visibility is assumed.
Plane returns plane_point_world, normal_world, plane_offset_m, corners_world, plane_max_error_m, height_span_m and horizontal_height_m (null if tilt >5°).
Diagnostics include ray_gap_m, ray_angle_deg, reprojection_error_px, one_pixel_sensitivity_m and extent/axis sensitivity where applicable; line perturbs peer endpoints by ±1 pixel; horizontal reports maximum endpoint shift from independent ±1-pixel coordinate perturbations (not a joint bound).
Returns plan_ok/plan_fail_reason; invalid input or baseline <20 mm fails. Point-based triangulation rejects ray angle <5°, behind-camera intersections, gap >10 mm, reprojection >2 pixels or sensitivity >15 mm. Line rejects epipolar angles <15°. Horizontal projects fitted endpoints into the peer view and returns peer_span_fractions; both reject extrapolation beyond half the peer span. Horizontal rejects >2-pixel line residual, unstable height or >15-mm endpoint sensitivity.
Sensitivity excludes segmentation/calibration bias and incorrect matches; identity, hidden geometry, retention and free space remain unverified.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

`robo surface-circles --u FLOAT --v FLOAT [--z FLOAT] [--camera head] [--color_tolerance 0.12]`
Read-only calibrated RGB measurement of complete circular boundaries enclosed by a connected colored surface; no motion or action steps.
u/v seed a median 3×3 chromatic surface patch, not its interior cutouts. Camera accepts head, wrist_l, wrist_r, or observation names.
Omitted z fits height by matching complete rectangular outer boundaries across current cameras; requires a horizontal surface and unambiguous stereo geometry. Returns height_measurement with cameras, sensitivity and height span.
Explicit z supplies a caller-measured horizontal plane height in world metres and bypasses stereo height fitting; supplied height is unverified.
Returns plane_z, plane_height_verified, circles with center_world, radius_m, fit_rms_m, one_pixel_scale_m and boundary_center_pixel, sorted by world coordinates.
color_tolerance is normalized RGB distance, 0.02–0.25. Requires distinct chromatic color (neutral pixels excluded), complete enclosed boundaries and sufficient pixel resolution; hidden boundaries are not filled.
Returns plan_ok/plan_fail_reason and surface_clipped; with explicit z the outer surface may be cropped, but measured boundaries must remain enclosed.
Rejects invalid input/calibration, unresolved or ambiguous height, oblique rays, noncircular boundaries or no resolved circles.
Assumes circular coplanar geometry; physical identity, occupancy and clearance remain unverified. Pixel scale excludes segmentation, correspondence and height bias.
