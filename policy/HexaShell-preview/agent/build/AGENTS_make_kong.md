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

`robo expose_extend <left|right> --tip_arm left|right --pieces JSON --source_top JSON --row_end JSON --rest_surface JSON --height H --thickness T` pushes upright pieces, carries a separate raised flat piece through a resting point, turns it upright into the next row position, and opens both hands.
`check_expose_extend` accepts identical arguments for a motionless preview of the entire operation; `tip_pieces` / `check_tip_pieces` are aliases requiring the same complete geometry.
XYZ arrays are measured world metres: pieces contains upper centers; source_top is an upper center with directed long axis +x, which is turned to +z; row_end contains two consecutive upper centers, inner then outer; rest_surface is a floor point. H is the long dimension and T the flat thickness.
Source validation requires two separated raised faces in the same x column and accepts only the one nearer the measured row. Missing or ambiguous RGB-derived range fails without motion.
Options: --clearance M (0.10; 0.045–0.2), --tip_clearance M (0.045), --tip_min_clearance M (0.025), --face_y -1|1 (−1, push direction), --aperture F (0.65), --source_aperture F (1), --release_aperture F (0.65), --carry_guard 0|1 (0). Apertures range 0.1–1. Contacts and withdrawals are vertical; the donor points down and opens along x, the receiver opens along y and finishes pointing −x/down.
`fetch_place` / `check_fetch_place` and aliases `extend_row` / `check_extend_row` accept the same arguments except pieces/tip_arm/tip clearances, and perform only the carry; optional --tip JSON prepends pushes.
`inspect_upright_groups [--camera head|wrist_l|wrist_r --profiles JSON]` returns aligned narrow upper faces and both endpoint pairs. Optional profiles is an array of six [u,v] corners per body: bottom front left/right, upper front right/left, upper rear left/right; requires visible floor contact, y-normal fronts and horizontal tops. Measures the floor from current RGB, then returns upper centers, height/width/thickness and input indices without dense stereo; unknown floor or inconsistent outlines fail. Omitting profiles uses stereo. `inspect_raised_sources --floor_z Z --length L --thickness T [--width W --camera C]` returns measured raised rectangular faces.
`inspect_extension --source_top JSON --row_end JSON --height H --width W --thickness T [--camera C]` returns visible upright geometry in either axis-aligned yaw at the derived destination and residual source evidence.
`--source_views JSON` on raised-face inspection and carry/compound commands supplies {camera,other_camera,faces,other_faces}: two distinct camera names and corresponding arrays of cyclic four-corner pixel polygons of the same complete horizontal x/y-aligned faces. Fits one horizontal rectangle to both sets of fresh RGB rays (RMS <=2 px, each corner <=3 px, center sensitivity <=4 mm/px; estimated per-axis two-sigma error <=8 mm at center and <=10 mm at corners); returns corner_fit residuals, fitted pixels, center_sigma_per_pixel, pixel_sigma_px and center_2sigma_m/corner_2sigma_m (local linear estimates, exact calibration/axes, >=1 px noise); ray_gaps_m describes raw selections; requires current floor agreement, rectangular geometry, measured size/elevation and consistent rays; malformed or ambiguous input fails. Complete-operation selection still requires two separated faces and the nearer one. Six-corner profiles are fitted together to a grounded cuboid; each result includes profile_fit with fitted_pixels, reprojection_rms_px, reprojection_max_px and center_sigma_per_pixel (metres per pixel, assuming exact floor/axes). RMS above 2 px, any corner above 3 px, center sensitivity above 8 mm/px, or estimated per-axis two-sigma error above 8 mm at center / 10 mm at corners fails with profile_index; profile_fit also returns pixel_sigma_px (>=1 px, seven residual degrees of freedom), center_2sigma_m and corner_2sigma_m, assuming exact floor/calibration/axes; requires complete visible outlines.
Returns plan_ok/plan_fail_reason, derived_geometry, selected_relay, relay_search, failed_stages and action_budget; previews are free, execution consumes action steps. For x-axis carries, donor clearance reverses the flat transit without wrist rotation. Pickup RGB-derived range requires at least three samples in an interior strip (half-length 0.4H, half-width 12 mm) within ±min(6 mm,T/4) of the upper face, rechecked before descent; visible end samples feed retention checks. Resting-face RGB-derived range is checked after donor clearance and before receiver approach; post-lift RGB-derived range checks retention. The x-axis carry includes eight settling steps after receiving closure, included in preview cost and returned as receive_settle_steps. Redundant opening targets are omitted; both hands finish at least 60% open. Costs exclude home travel; execution ends after opening and rising. Invalid geometry, unavailable RGB-derived range, blocked paths, insufficient budget, IK or tracking errors stop without retry; physical completion is unverified.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

`robo rgb_points --pixels '[[u,v],...]' [--camera head|wrist_l|wrist_r] [--plane_z table|Z | --other_camera C --other_pixels '[[u,v],...]']` measures selected image points without motion or action cost.
With plane_z=table, measures the current RGB floor and intersects calibrated rays with it; unknown floor fails. Numeric Z uses a supplied world-height plane and refuses values more than 4 mm below a resolved floor; with paired pixels, triangulates corresponding physical points; otherwise estimates ranges from the current RGB cameras using horizontal and vertical local planes.
Returns plan_ok/plan_fail_reason, points, centroid and extent_xyz in world metres, method, plane_z_used/table_measurement for plane intersections and ray_gaps_m for triangulation. Pixels may have fractional coordinates.
Rejects malformed calibration, outside-image pixels, parallel rays, insufficient baseline, behind-camera points, ray gaps over 3 mm and ambiguous automatic correspondences.
Plane height accuracy and correspondence identity are caller responsibilities; floor intersections do not measure elevated surfaces. Selected surface points are not body centers or contact targets; measurements expire after motion.
Automatic reconstruction covers world z=0.45–1.20 m, x=−0.75–0.75 m and y=−0.75–0.60 m; it leaves weakly textured or conflicting areas unknown, including competing plane orientations at different ranges. Other modes require no height-search envelope.
`robo rgb_patch --corners '[[u,v],[u,v],[u,v],[u,v]]' --axis x|y|z [--camera head|wrist_l|wrist_r] [--base_z table|Z | --edge_length L]` measures one rectangular planar face without motion or action cost; corners are cyclic vertices of the complete visible face, axis is its world normal.
Without base_z/edge_length, pools interior RGB texture across the polygon and searches the same coordinate envelope to measure plane_coordinate; no supplied height or paired pixels required. Returns plan_ok/plan_fail_reason, points, centroid, extent_xyz, edge_lengths in metres, camera-pair scores and margins.
Rejects small/nonconvex polygons, grazing rays and nonrectangular reconstructed corners; unconstrained mode also rejects insufficient texture, ambiguous correspondence, camera-plane disagreement over 3 mm and weak single-pair evidence. Results do not populate automatic per-pixel reconstruction.
With base_z=table, measures the floor from current RGB cameras; unknown floor fails. Numeric Z remains available but is rejected when more than 4 mm below the measured floor. Returns base_z_used and table_measurement. With either base_z form, axis must be x or y and the first two cyclic corners must be the visible bottom edge resting at that measured world height; intersects those rays to locate the vertical face and measure all four corners. Rejects bottom-edge plane disagreement over 3 mm or upper corners below the bottom.
With edge_length, L is the independently measured length in metres from corner 0 to corner 1; solves distance to the specified plane from calibrated angular size. Both constrained modes work with a single image and uniform interiors; dimensions, contact height, full-face corners and normal axis are caller assumptions, not verified physical facts.

`robo compare_faces --reference '[camera,u,v]' --faces '[[camera,u,v],...]' [--count N --tolerance M]`: read-only comparison of planar colored fronts across head/wrist_l/wrist_r views, normalizing contrast and color to each pale background; seeds are pale face interiors as [camera,u,v] or [camera,u,v,u0,v0,u1,v1] with exclusive crop bounds; count defaults to 3, more candidates than count required. Returns selected_indices, unresolved seed roles/indices/reasons, retained per-face distances, measured surface centers/normals, background_rgb and separation; selects nothing if any front is unresolved; fails on unresolved patterns, insufficient RGB-derived range or ambiguous separation or overlapping crops (merged_fronts).
`robo inspect_regions [--camera head|wrist_l|wrist_r] [--u0 N --v0 N --u1 N --v1 N] [--min_pixels N] [--top_tolerance M]`
Reads fresh calibrated RGB stereo without motion or command cost; defaults to the full head image and 30 pixels, with exclusive rectangle upper bounds.
Returns pale connected regions, pixel boxes, visible world bounds/midpoints, top_z, upper_patches and multiple_upper_patches. Each upper patch has center_xy, z, extent_xy, narrower_axis, pixels, pixel_box, near_horizontal and touches_roi_edge.
Upper patches are separate connected areas within top_tolerance of the region's upper height after excluding its highest 2% of samples (default 0.001 m, range 0.0005–0.004), with ≥10 pixels and ≥6 mm XY spans. top_center_xy, top_extent_xy and narrower_top_axis are null unless exactly one patch is within 15° of horizontal and clear of the ROI border. Unresolved seams can merge, colors can fragment, lower areas are omitted. Patterns: G=green, R=red, #=dark, .=pale.
`robo inspect_surface --u N --v N [--camera head|wrist_l|wrist_r] [--tolerance M]`
Reads fresh RGB-derived range without motion or cost; fits a plane in a 5×5 seed neighborhood and grows connected coplanar pixels regardless of color.
Tolerance defaults to 0.002 m (range 0.0005–0.005); seed must be at least two pixels from image edges, with valid RGB-derived range and a planar neighborhood.
Returns surface_center, normal toward camera, seed_world, visible bounds, extent_xyz, pixel_box, pixels and touches_image_edge; world distances are meters.
Both return plan_ok/plan_fail_reason; fail on invalid inputs, absent calibration/RGB-derived range or insufficient area. Surface fitting also fails on nonplanar seeds.
Visible centers are not body centers or TCP targets; occlusion can truncate bounds and touching coplanar faces can merge. Measurements expire after motion. Weak texture, occlusion or inconsistent camera pairs leave unknown pixels; no range sensor is read.
