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

Read-only RGB geometry; no motion or action steps. Cameras: head, wrist_l, wrist_r or observation keys. Coordinates are world meters; pixels are [u,v]. All commands return plan_ok/plan_fail_reason and plan_detail on errors.
`robo surface-points --camera NAME --pixels JSON [--roi '[u0,v0,u1,v1]'] (--plane_z M | --plane_points JSON) [--frame_arm left|right]` intersects 1..32 rays with an explicitly supplied plane; ROI only bounds pixels, it does not estimate height. Returns points_world, midpoint_world and separation_m for two points.
`plane_z` is a measured horizontal surface height (table_plane reports table height only); `plane_points` is 3..32 measured noncollinear world points on one surface, e.g. stereo-points output. Exactly one is required; never substitute table height for an elevated surface. Plane membership is assumed, not verified.
`robo stereo-points --camera NAME --pixels JSON --other_camera NAME (--other_pixels JSON | --other_roi JSON | --other_segments JSON) [--tolerance M] [--frame_arm left|right] [--base_mode none|endpoints|surface|lines] [--refine_radius PX]` triangulates 1..32 corresponding physical points in distinct current RGB views. Default base_mode=none preserves all points, including three-point plane samples.
Stereo tolerance=0.002 (0.0005..0.005); rejects ray gaps above tolerance, reprojection >1 pixel, angles within 10 degrees of parallel, invalid pixels/calibration, and points behind cameras. Returns ray_gap_m, triangulation_angle_deg and stereo_views; correspondence and visibility are caller assumptions. Optional other_segments=[[[u0,v0],[u1,v1]],...] supplies one physical straight edge containing each selected point in the second view: intersects each segment with its epipolar line, then triangulates; requires length >=5 px, crossing angle >=20 degrees, intersection inside segment, refine_radius=0 and base_mode=none|endpoints|surface. Returns other_pixels_intersected and edge_diagnostics. Edge membership/straightness/visibility are assumed, not RGB-verified; view-dependent silhouettes are not corresponding physical edges, and zero epipolar residual is not independent evidence.
Stereo endpoints/surface mode requires four correspondences: two forward ends, then their two rear points. Returns the forward pair, axis_world and source_geometry={points,axis,frame,arm}; rejects inconsistent segments or baseline <5 mm. Surface mode permits bounded transverse rear offsets. Lines mode also takes four pixels per view: two corresponding forward ends then one rear sample on each same straight axis; rear samples need not correspond across views. It intersects image-line planes (>=10 degrees), requires >=5 px line lengths, <=5 degree axis disagreement and perpendicularity error, explicit other_pixels and refine_radius=0. Returns axis_world and the same bundle; straight-line membership and endpoint identity are assumed.
`robo plane-pair --camera NAME --roi JSON --separation M (--plane_z M | --plane_points JSON) [--tolerance M] [--contrast N] [--pixels JSON] [--radius PX] [--roi_margin PX] [--frame_arm left|right]` detects enclosed dark RGB regions and projects their centers onto the supplied plane. No foreground/occlusion check; returns only a unique spacing-compatible pair, plus diagnostics.
Pair separation=0.003..0.2 m; tolerance=0.002 (0.0005..0.005, below half separation). contrast=0 requires one stable pair across >=2 thresholds and rejects persistent alternatives; explicit contrast=10..200 uses one. Optional two seeds preserve order within radius=4 (1..12 px); roi_margin=8 (integer 0..24 px) adds boundary/contrast context, while centers must remain inside ROI; ambiguity/instability fail without accepted points_world.
frame_arm adds points_local and tcp_at_measurement; axis captures also add axis_local and use frame=tcp in source_geometry. Without it, the bundle uses frame=world. Rigid attachment remains assumed.
`robo feature-pose ARM --points_local JSON [--axis_local JSON]` transforms saved local geometry through current TCP; returns predicted world geometry with attachment_status=unverified. It does not visually validate attachment.
Optional refine_radius=1..12 snaps approximate corner seeds to subpixel RGB corners in both views (default 0 preserves exact clicks); requires exactly one pair with reprojection <=0.5 px per seed and rejects duplicate corners. Returns refined_pixels, refined_other_pixels, seed_offsets_px and corner_candidate_counts. Alternatively other_roi=[u0,v0,u1,v1] replaces other_pixels and searches <=128 corners in a second-view region with sides <=512 px and area <=65536 px with refine_radius=1..12; only unique matches yield geometry, ambiguous/missing matches return correspondence_diagnostics with candidate pixels for narrowing the region. Corners are not region centers; identity remains assumed.

Read-only calibrated RGB geometry across time; no motion or action steps. Every command returns plan_ok/plan_fail_reason, with plan_detail on failure.
`robo capture-rays --arm left|right --camera NAME --pixels JSON` records 1..32 [u,v] pixels, calibration and measured camera-to-TCP transform; returns reference JSON, not 3D points.
`robo temporal-points --arm left|right --reference JSON --camera NAME --pixels JSON [--base_mode none|endpoints|surface|lines] [--tolerance M]` combines the saved rays with current corresponding pixels in the same arm's TCP frame. The camera may be the same at both times; relative camera-to-TCP viewpoint must differ.
Reference means exactly the reference object returned by capture-rays. Samples must belong to one rigid body continuously attached to that arm throughout both observations; release, regrasp, slip or reset invalidates the reference. Attachment and feature identity are unverified.
Defaults: base_mode=none, tolerance=0.002 m (0.0005..0.005). Rejects count mismatch, invalid input, ray gaps above tolerance, reprojection >1 px, behind-camera solutions and triangulation angles <10 degrees from parallel. A camera moving rigidly with the same TCP supplies no temporal baseline.
None mode returns every point; endpoints/surface require two forward ends then two corresponding rear points. Lines requires two corresponding forward ends then a rear sample on each same straight axis in each view; rear samples need not correspond. Axis consistency and baseline guards match stereo-points.
Returns current points_world, points_local, midpoint_world, ray_gap_m and triangulation_angle_deg; two points add separation_m. Axis modes add axis_world, axis_local and source_geometry={points,axis,frame:tcp,arm} compatible with mate-pair. All coordinates are meters; no height or dimension prior is inferred.

`robo stereo-guides --camera NAME --pixels JSON --other_camera NAME [--other_pixels JSON] [--other_roi JSON] [--band PX]` returns calibrated epipolar diagnostics; read-only, no action steps.
Cameras: head, wrist_l, wrist_r or observation keys. pixels: 1..32 [u,v] pairs; optional other_pixels has matching count.
Each guide returns other_line_abc (a*u+b*v+c=0, sqrt(a*a+b*b)=1), other_image_segment clipped to the second image, and optional other_pixel_line_distance_px.
Optional other_roi=[u0,v0,u1,v1], <=256 px per side, returns subpixel RGB corner_candidates within band=3 (0.5..12) pixels of each line, with line_distance_px; >128 ROI corners fails.
Lines constrain correspondence but do not determine depth or identity. Candidates are diagnostic only, never accepted geometry; no world points or attachment claims are returned. An empty segment means the line misses the image.
Returns plan_ok/plan_fail_reason; invalid inputs/calibration, coincident centers or epipole pixels fail with plan_detail. No candidates is a valid diagnostic result.

`robo mate-pair ARM --source JSON --target '[[x,y,z],[x,y,z]]' [--source_frame world|tcp] [--source_axis '[x,y,z]'] [--target_axis '[x,y,z]'] [--clearance M] [--depth M] [--advance_step M] [--tolerance M] [--wrist_lift_deg D] [--correspondence ordered|either] [--park_other 0|1] [--park_clearance M] [--compact 0|1] [--orient_step_deg D]`
Aligns rigid feature pairs with measured feature-to-TCP offset compensation; ARM=left|right, coordinates/distances in meters. Source is two current points or `{points,axis,frame,arm}`; bundle axis/frame override flags and TCP arm must match ARM. Local captures require unchanged rigid attachment.
`source_frame=world`; tcp transforms points/axis through the current TCP. Target points/axis always use world coordinates. Axes default [0,0,-1] in their respective frames, perpendicular to pairs. Target spacing is reconciled to source spacing only within tolerance, retaining target midpoint/direction.
`clearance=0.015` (0..0.08), `depth=0` (0..0.03 beyond target), `tolerance=0.002` (0.0005..0.005). `wrist_lift_deg=8` (permission 0..25, effective cap 8 degrees) permits calibrated midpoint tilt bounded by tip/shaft geometry and half the remaining tolerance; advance follows target_axis.
`correspondence=ordered` preserves order; either permits endpoint exchange. Compact routes compare calibrated wrist travel and, when planner is available, check every angle including zero, trying permitted exact-axis correspondences before tilt permissions in 2-degree increments through the effective cap; the first fully preflighted route is selected before motion. Without compact route planning, alignment stays exact. Preflight covers descent/final advance; it does not verify collisions or physical joint limits.
`park_other=1` requires the other hand commanded open; homes it, then permits <=0.15 m outward lift to satisfy calibrated segment gap `park_clearance=0.12` (0..0.2). Zero clearance means home only; park_other=0 omits parking. The axial gap check includes elevated arrival/descent for large compact rotations; lateral transit and whole-arm clearance remain unverified.
`compact=1` combines small rotations/transfer; larger rotations use elevated coordinated increments then axial descent <=20 mm. Geometric plane-clearance checks gate compact routes. `orient_step_deg=20` (5..45; 0 selects the bounded 20-degree default) bounds rotation increments with calibrated interpolation checks; compact=0 separates orientation/transfer.
`advance_step=0.005` (0.001..0.005 m) bounds each advance, with an explicit checkpoint at the target plane before entering; checks combined predicted tip/shaft path errors against tolerance. One endpoint exchange may follow motionless orientation/transfer IK rejection; no contact retries or exchange after partial failed motion. Stops on invalid geometry, planner/calibration error, tracking/contact or exhausted budget; never releases.
Returns plan_ok/plan_fail_reason, stage details/settle_steps, resolved source geometry, correspondence, preflight_attempts/kinematic_path_preflight_passed when applicable, increment counts/advance_step_m/effective_orient_step_deg, tilt permission/limit/selected limit, applied tilt, feature/shaft errors, predicted_points_world/predicted_depth_m and grip_unchanged/seat_verified=false.
After executed motion fails, cancels active targets at measured joints and reports target_cancelled/cancel_error. Rigid attachment is assumed; tracking and predicted depth do not verify physical engagement. Planned action-step estimates exclude parking/settling.

`robo carry-offset ARM --delta '[x,y,z]' [--lift M] [--up '[x,y,z]'] [--tolerance M]`
Translates a commanded closed grasp by a world displacement while preserving its measured initial TCP orientation; ARM is left/right, distances are meters.
Executes three checked legs: outward to a raised plane, lateral travel on that plane, then axial lowering to the displaced starting pose. Never lowers before lateral travel completes.
`delta` is relative to the TCP at command entry, length <=0.6 m; `up` is a nonzero outward surface normal (default `[0,0,1]`, normalized).
`lift=0.06` (0..0.15 m) is extra clearance beyond the higher of the starting and ending TCP planes. With 0, lateral travel still precedes any lowering; clearance for the held geometry remains caller-dependent.
`tolerance=0.003` (0.0005..0.005 m); stops on larger position error, rotation error >2 degrees, planning failure, or exhausted budget. No retries, opening, or other-arm motion.
Requires commanded gripper opening <=0.5; attachment, collisions, and physical displacement of held geometry are not verified by TCP tracking.
Returns `plan_ok`, `plan_fail_reason`, `plan_detail` on failure, stage errors/details, requested/reached TCP position and achieved displacement after motion, `grip_unchanged=true`, `attachment_verified=false`.
After executed motion fails, cancels the active target with measured joints and reports `target_cancelled`/`cancel_error`; invalid arguments reject before motion.

`robo axial-pick ARM --goal '[x,y,z]' [--up '[x,y,z]'] [--clearance M] [--lift M] [--tolerance M] [--source_geometry JSON]`
Moves to a world TCP grasp position, closes, and lifts while preserving the current orientation; ARM is left/right, distances are meters.
Requires commanded opening >=0.95 and current TCP approach axis within 5 degrees of `-up`; no automatic orientation change or opening.
`goal` is the desired TCP position at closure, not a visible surface point; distance from current TCP <=0.6 m. `up` is a nonzero outward normal, normalized (default `[0,0,1]`).
Raises if needed, travels laterally above the goal without lowering, descends axially to 20 mm above it, then advances in 5 mm increments before closing and lifting.
`clearance=0.04` and `lift=0.10`, each 0.02..0.20 m; lateral travel is at the higher of current height and goal plus clearance. Clearance of surrounding geometry remains caller-dependent.
`tolerance=0.003` (0.0005..0.005 m); stops on larger TCP error, rotation error >2 degrees, planning failure or exhausted budget; no retries or other-arm motion.
Returns `plan_ok`, `plan_fail_reason`, stage errors, `close_commanded`, `pickup_verified=false`; success describes motion only, not retention or collision clearance.
Failures include `plan_detail`; after executed motion, cancels the active target at measured joints and reports `target_cancelled`/`cancel_error`. Never closes after failed approach/descent, or opens after closure.
Optional source_geometry={points:[[x,y,z],[x,y,z]],axis:[x,y,z],frame:world} supplies measured geometry, assuming it stays stationary until closure and rigidly attached afterward; separation 0.003..0.2 m, nonzero axis perpendicular within 5 degrees. Identity, stationarity and attachment are unverified.
On successful lift, returns TCP-frame source_geometry compatible with mate-pair, tcp_at_binding, predicted_points_world/predicted_axis_world and attachment_status=unverified. Binds at measured pre-closure TCP; closure translation above tolerance or rotation >2 degrees fails without a bundle. Failure never returns source geometry; release, regrasp or slip invalidates it.

`robo bounded-rotate ARM [--roll DEG] [--pitch DEG] [--yaw DEG] [--frame world|tcp] [--step_deg DEG] [--tolerance M] [--reserve_steps N]`
Rotates about the initial measured TCP position; ARM is left/right; RPY deltas default to zero, each -180..180 degrees, composed as Rz Ry Rx in world (default) or TCP coordinates.
Executes the shortest resulting rotation in calibrated increments: `step_deg=20` (5..30), reduced further to bound ideal end-link interpolation bow to half `tolerance`.
`tolerance=0.003` (0.0005..0.005 m); stops if measured TCP displacement exceeds it or orientation error exceeds 2 degrees. This does not bound physical motion between measurements.
Preflights every increment and estimates action cost before motion; rejects unreachable paths or estimates leaving fewer than `reserve_steps=40` (integer 0..400). Reserve is a budget allowance, not a return-path guarantee.
Returns `plan_ok`, `plan_fail_reason`, `plan_detail` on failure, `preflight`, estimated rotation steps, increment count, and measured stage errors; zero rotation returns success without motion.
Stops on planning/tracking failure or exhausted budget; cancels executed targets at measured joints and reports `target_cancelled`/`cancel_error`. No retries, hand opening, or other-arm motion.
Requires calibrated TCP-to-end-link transform and chain-cost API; preserves the hand command. Attachment and collision clearance remain unverified.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

`plane-height --camera NAME --other_camera NAME --roi '[u0,v0,u1,v1]' --z_range '[min,max]' [--patches '[[u0,v0,u1,v1],...]' | --polygon '[[u,v],...]']` measures a horizontal surface height from calibrated RGB views; read-only, no action steps.
ROI bounds are integer pixels, 16..128 per side for automatic support or 16..512 with explicit patches; the same textured planar region must be visible in both views. z_range is world meters and must span 0.008..0.2 m.
Uses four quadrants when textured; otherwise requires at least three spatially distributed textured tiles. Every selected patch must agree within 1 mm and have peak correlation >=0.85. Broad peaks may pass with two independently strong patches, contiguous interior near-peak support and a strong equal-weight joint peak; no evidence is discarded.
Optional patches replace automatic support: 3..16 disjoint integer rectangles inside ROI, sides >=4 pixels, each textured, at most 16384 selected pixels; farthest centers must span >=25% of ROI diagonal, and may be collinear. All must depict the same horizontal surface visible in both views. Pixels outside patches are excluded.
Optional polygon: 3..8 ordered convex integer vertices inside a <=128-pixel-per-side ROI, area >=64 pixels. Excludes a one-pixel boundary and selects all textured 4x4-grid tile portions with >=16 pixels; requires >=3 tiles spanning >=25% of ROI diagonal. Selection uses only the reference image, before matching; all selected tiles must pass the same confidence checks. Mutually exclusive with patches.
Returns `plane_z`, `plane_point`, `normal_world`, `support_mode`, confidence_mode, joint_peak_margin, selected_pixel_count, support_polygon, patch bounds/peak scores, height spread, sampling step, and `plan_ok`/`plan_fail_reason`.
Fails on insufficient texture/spatial support, weak baseline, occlusion/out-of-view patches, ambiguous/boundary peaks or >1 mm patch disagreement; returns no accepted height on failure.
Horizontal planarity and physical feature identity are assumptions; returned sampling step and spread are not accuracy guarantees.
