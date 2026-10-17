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

`robo transfer-cycle {left,right} --x X --y Y --z Z --tx X --ty Y --tz Z --tip L [--pitch DEG] [--clearance M] [--dwell S] [--travel_pitch DEG] [--entry_offset M] [--reserve S] [--release_wait S] [--finish_home 0|1]`
`robo transfer-estimate` accepts identical arguments without motion or command-budget cost; returns waypoints, heuristic estimated_seconds/fits_estimate and read-only IK joint_cost_evidence. Estimates omit settling, corrections, split fallback, inactive retraction and concurrent terminal home; neither guarantees time or clearance.
Grasps an upright rigid item, lifts, transports upright, tilts, dwells, reverses, replaces, releases and retreats. Requires an open empty active hand and clear swept corridors; bounded inactive-hand retraction requires that hand open/empty and a clear backward corridor. Attachment and whole-scene clearance remain unverified.
World metres: x/y/z is interior grasp centre; tx/ty/tz is the tip target at full tilt, not TCP; tip is height above grasp (0.02–0.30). Signed world-Y pitch is ±140° (default +140°, toward +X); full-tilt TCP must exceed grasp Z by 0.02 m.
Defaults/ranges: clearance=0.06 m (0.06–0.25); dwell=1.80 s (1.80–2, rounded up to 25 Hz ticks); travel_pitch=0 required; entry_offset=0 or 0.08–0.25 m; reserve=0 nonnegative seconds; release_wait=0.16 s (0.16–0.48); finish_home=0 or 1.
entry_offset=0 gives overhead entry/vertical retreat; positive values give front staging at y-entry_offset. Loaded travel stays upright; outward/reverse arcs compensate tip XY with 8.5 mm nominal chord error plus reported observed-offset bound, without a mandatory horizontal stop.
Supported RGB stereo envelopes permit ≤20 mm early arc lowering plus tip×max(0,-cos(angle)) after horizontal, withheld across a segment crossing 90°. Continuous ring minima and a 50 mm TCP underside allowance constrain lowering; missing grasp radii trigger independent coaxial fits. Fixed entry/terminal poses are preserved; missing support does not certify clearance.
Preflight rejects observed entry obstacles, narrow/tapered grasps and unsupported terminal clearance; suggestions never substitute arguments. insufficient_time may include ≤4 measured higher-grasp alternatives preserving standing tip and target. Lift evidence requires fresh RGB geometry and two consecutive reads agreeing within 2 mm centre/1 mm radius; ≤3 reads, plus one confirmation only after valid third geometry. Correction ≤15 mm; radius change ≤3 mm. Fallbacks use conditioned paired edges or per-view size calibrated at earlier stereo height (unchanged horizontal shape assumed, radius_remeasured=false); ambiguous evidence fails. Accepted offsets correct arcs and recheck workspace, hand clearance and time.
Tracking gates: intermediate 5 mm/1°, continuous dwell 10 mm/1°, recovery 10 mm/2° with ≤15 mm tip bound, staging/release 2 mm/0.5°. Closed-hand settling is bounded at 4/12 ticks; release allows ≤4 extra ticks only within 10% of tolerance and still improving, preserving reserves. Opening overlaps retreat; withdrawal checks measured TCP/wrist geometry.
finish_home=1 withdraws and returns both empty open hands concurrently to recorded base home joints, requiring clear joint paths; reports completion.home_verified, remaining_steps, repeated_home cost/fit and home_result. Home verification covers selected poses only.
Returns plan_ok/plan_fail_reason, failed_stage, stages, lift_evidence (including support_mode for tracked fits and final_confirmation_read when extended), arc_clearance, entry_clearance, grasp_support, tilt_hold_steps and advisory held_axis_evidence/destination_surface_evidence; completion_scope=motion_only, grasp_verified=false, transfer_verified=false. Invalid geometry/time, missing required RGB correspondences or tracking errors stop; failures after closing may leave the hand closed. Only an unmoved combined-aim IK failure permits one time-checked split fallback.

`robo joint-return {left,right,both}` moves selected arms concurrently to their recorded base home joint targets.
`robo joint-return-estimate {left,right,both}` returns the planned duration without motion or command-budget cost.
`robo joint-return-status {left,right,both}` measures home-pose errors without motion or command-budget cost; query success does not imply at_home or stationary joints.
Movement requires open grippers, empty hands and clear straight joint-space paths; preserves gripper commands and performs no collision checking.
Local profiles use ≤2 rad/s and 6 or 12 rad/s²; if infeasible, a public base-home trajectory may be selected before motion using current server limits. Admission uses integer steps and retains a live tick.
Deadline/margin profiles permit early completion after two accurate/stable samples (≤0.02 rad error, ≤0.005 rad change/tick), with at most four settling ticks and at least one live spare tick; preferred headroom never overrides convergence.
Returns trajectory_profile, plan_ok/plan_fail_reason, planned_seconds/steps, minimum_steps, adaptive_settling, acceleration_limit_rad_s2, reserve_steps, available_steps, fits_budget, collision_checked=false and joint_return_verified. Estimates do not verify execution; minimum_steps assumes early convergence.
Execution/status report joint_error_rad, at_home per arm, all_selected_at_home, episode_live and target_reference=base_home_recorded_joints. Execution verifies measured error ≤0.02 rad and a live episode; scene_completion_verified=false.
base_home_followup reports shared targets, pose_already_satisfied and estimated repeat cost/fit; headroom_profile_selected, remaining_steps and preferred_headroom_retained expose margin policy/results. Follow-up timing is model-based.
Invalid state, insufficient time, timeout or failed convergence returns failure without retry; verification covers only selected arm poses.

`robo rim-fit --u PIXEL --v PIXEL [--camera head|wrist_l|wrist_r] [--window 60] [--band 0.04] [--reach 0.12]`
Read-only horizontal circular-rim measurement from calibrated RGB pairs; no motion or command-budget cost. At least two overlapping calibrated PNG views are required.
Pixels are integers; window is patch half-width (8–100 pixels), band is stereo-point vertical half-width (0.01–0.04 m), reach is horizontal distance from the stereo seed (0.03–0.12 m).
Fits the highest supported local circular section of radius 0.015–0.12 m. Sparse stereo falls back to matching horizontal circular contours in two views within window of the seed, above or below either camera, without a guessed plane height.
Returns centre_world (XYZ metres), centre_xy, rim_z, radius_m, arc_degrees, radial_rmse_m, inlier_points, source and camera_source; point fits also return surface_world and stereo_pairs, contour fits stereo_pair plus conditioning_m or image_edge_error_px.
Point fits require 220° coverage and reject supported higher surfaces; rim_search_clipped rejects a fit within 3 mm of the upper height boundary. Paired contours require 240° coverage, centres agreeing within 3 mm and radii within 2 mm. An edge-arc fallback refines XYZ/radius against both images, requiring 80% one-pixel edge support, 10 of 12 supported sectors, mean capped error <0.65 pixels and a local conditioning check; ambiguous associations fail.
search_upper_z is the point-search limit, or null for contour fits with height_search=matched_complete_contours or matched_edge_arcs. Unseen higher geometry remains unverified.
Edges retain channel contrast as well as grayscale contrast. Proposals may start in either view; their fitted centre must project within window of the selected pixel, with fresh support in both views. contour_reference reports the proposal camera; more than 64 contours per proposal view fails as clutter. Small conics may propose a horizontal circle within 0.75 pixels; these proposals require the same two-view edge refinement and cannot enter direct contour measurements.
Returns plan_ok/plan_fail_reason and plan_detail on failure. Invalid arguments, insufficient RGB support or inconsistent geometry fail without motion. The measured plane does not include any stand-off or establish free space, contents or physical clearance.

`robo side-pick {left,right} --x X --y Y --z Z --tip M --radius M [--lift M] [--entry_offset M] [--reserve S]`
Grasps an upright item from +Y with X finger opening, lifts vertically, then withdraws toward -Y while upright; leaves the hand closed.
World metres: x/y/z is the interior grasp centre; tip is top height above it (0.09–0.30); radius is observed horizontal radius (0.008–0.05).
lift is 0.06–0.20 m (default 0.08); entry_offset is 0.10–0.25 m (default 0.12); reserve is nonnegative seconds (default 0).
Requires an open empty active hand, visible upper surface or annular edge in calibrated RGB stereo, and clear front travel, vertical lift and insertion corridors. Paths are not fully collision checked.
Backs out before lateral travel, aligns outside the entry plane, lowers there, inserts at fixed X/Z, closes, lifts and withdraws. Rejects an obstructing inactive hand before motion.
Checks geometry before closing, after vertical lift and after withdrawal; absent, occluded or inconsistent evidence fails. This is geometric evidence, not attachment or stability verification.
Returns plan_ok, plan_fail_reason, plan_detail on failure, stages, failed_stage on failure, evidence, lift_observed, grasp_verified=false, gripper_closed and grasp_pose/reached_tcp on success.
Invalid geometry, insufficient estimated time, motion error or missing RGB geometry fails without retry; a failure after closing leaves the hand closed. Timing is a Cartesian heuristic; joint retiming and settling may take more or less time.

`robo tip-tilt {left,right} --tx X --ty Y --tz Z --tip L [--pitch DEG] [--offset_y M] [--dwell S] [--reserve S] [--clearance M]`
`robo tip-tilt-estimate` accepts the same arguments and returns geometry and heuristic timing without motion or command-budget cost.
Requires an already held, upright rigid item and clear raised travel/rotation corridors. Preserves the gripper command throughout; attachment is not checked.
World metres: tx/ty/tz is the desired tip at full tilt, not the TCP; tip is its vertical distance above the initial TCP (.02–.30); offset_y is its measured world-Y displacement from that TCP (default 0, ±.05).
Pitch is signed world-Y rotation, magnitude 120–140°, default +120°; positive tips toward +X. Initial TCP local Z must be vertical within 2°.
Raises before lateral alignment to at least initial TCP Z + clearance (.02–.15 m, default .06); holds that altitude through 90°, then descends to the full-tilt endpoint. Tip X/Y compensation uses adaptive segments with ≤8.5 mm nominal interpolation error, excluding tracking/geometry error. Reverses to elevated upright; full-tilt TCP below initial altitude fails before motion.
Dwell .12–2 s (default .12) requires measured endpoints within 5 mm/1° before and after every tick; reserve is nonnegative seconds outside this operation. No retries or release.
Returns plan_ok, plan_fail_reason, failed_stage on failure, stages, waypoints, final_upright_tcp, estimated_seconds, travel_clearance_m, nominal_xy_chord_bound_m and grasp_verified=false/transfer_verified=false/scene_clearance_verified=false. Timing is a Cartesian heuristic, not a deadline guarantee or IK check.
Rejects invalid geometry, workspace limits, obstructing inactive hand, or insufficient estimated time; stops held on motion, tracking, dwell or time failure. Caller geometry, rigid attachment and scene clearance remain unverified.

`robo tip-fit --u U --v V --z Z [--camera head|wrist_l|wrist_r] [--window PX] [--band M] [--reach M]`
Read-only upper horizontal annular-endpoint measurement from calibrated RGB pairs; no motion, action steps or command-budget cost. Requires two overlapping calibrated PNG views.
Z is supplied grasp height in world metres; returns tip_m=endpoint_z-Z, centre_world, centre_xy, endpoint_z, radius_m, radial_rmse_m, arc_degrees, inlier_points and source.
Defaults: camera=head, window=24 pixels, band=0.02 m, reach=0.04 m. Limits: window 8–100, band 0.01–0.04, reach 0.03–0.12; fitted radius 0.006–0.04 m, tip_m 0.02–0.30 m.
Local stereo-point fits require 20 samples, 70% radial support and 220° coverage; reject substantial surface above the fit. Sparse evidence falls back to matched horizontal circular contours near the seed in two views, measuring height without an assumed plane.
Contour fits allow features above or below either camera; require 240° coverage, centre agreement within 3 mm, radius agreement within 2 mm and bounded conditioning; return stereo_pair and conditioning_m. A partial-edge fallback jointly refines XYZ/radius with 80% one-pixel support in both images, 10 of 12 supported sectors, mean capped error <0.65 pixels, local conditioning and ambiguity checks; returns image_edge_error_px. band/reach delimit the point search only.
Assumes an upright axis through grasp and endpoint. axis_alignment_verified=false and endpoint_verified=false: fits do not establish identity, attachment, hidden higher surfaces or complete shape.
Edges retain channel contrast as well as grayscale contrast. Proposals may start in either view; their fitted centre must project within window of the selected pixel, with fresh support in both views. contour_reference reports the proposal camera; more than 64 contours per proposal view fails as clutter. Small conics may propose a horizontal circle within 0.75 pixels; these proposals require the same two-view edge refinement and cannot enter direct contour measurements.
Returns plan_ok/plan_fail_reason; invalid arguments, missing RGB support, occlusion, ambiguous matches and unsupported geometry fail without motion.

`robo axis-pose --u1 U --v1 V --u2 U --v2 V [--camera head] [--window 40] [--band 0.015] [--reach 0.07]`
Measures the upward axis through two caller-selected circular surface sections in one calibrated RGB stereo observation; no motion or command-budget cost.
Pixels are integers; camera accepts head, wrist_l, wrist_r or their cam_* source names. window: 8–100 px; band: 0.01–0.04 m vertical half-width; reach: 0.03–0.12 m horizontal search radius.
Requires two nonoverlapping sections on the same approximately axisymmetric surface, with centre heights 0.05–0.30 m apart and inclination at most 10°. Selection order is immaterial; local circular fits require sufficient visible arc and vertical support.
Returns plan_ok/plan_fail_reason, lower_centre_world, upper_centre_world, axis_up_world, tilt_degrees, lateral_offset_m (upper minus lower XY), vertical_span_m and both section fits. Coordinates are world metres; tilt is relative to world +Z.
Measures axis direction only, not rotation about it. Same-item association, attachment and measurement uncertainty are unverified; near-vertical circular-section approximation may be biased by taper, occlusion or asymmetric surfaces.
Missing RGB pairs/calibration, invalid arguments, ambiguous sections or unsupported geometry return perception_failed without motion.

`robo axis-track --x X --y Y --z Z --dx DX --dy DY --dz DZ --radius M [--span 0.05] [--camera head]`
Uses calibrated RGB stereo to measure two cylindrical surface sections around a world-space reference centre and direction, at any attitude; no motion or command-budget cost.
Coordinates and radius are metres; direction must be finite and nonzero and is normalized. Radius: 0.008–0.06; span: 0.05–0.15 between section centres, each sampled over ±0.012. Camera: head, wrist_l, wrist_r or corresponding cam_* source names.
Requires visible circular sections of the same straight cylinder, within 10° and 0.02 m transverse displacement of the reference, with each fitted radius within 0.003 m of the supplied radius.
Returns plan_ok/plan_fail_reason, centre_world at the reference axial plane, axis_world oriented along the supplied direction, reference_angle_deg, transverse_error_m, sections in the reference frame and section_frame_world.
Axial translation, rotation about the axis, same-item association, attachment and uncertainty are unverified. Taper, occlusion, insufficient support, mismatched radius or invalid arguments return perception_failed; failure does not establish displacement.

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

`robo rgb-stereo --u U --v V [--camera head|wrist_l|wrist_r]`
Triangulates a selected visible surface using calibrated RGB image pairs; read-only, no motion or command-budget cost.
Pixels are finite integers. Also accepts cam_head, cam_left_wrist and cam_right_wrist. Requires at least two simultaneous PNGs with paired camera matrices and adequate overlapping texture.
Returns plan_ok/plan_fail_reason, surface_world in metres, source, camera_source, pair sample counts and geometry_verified=false. The point is on the exterior surface, not an interior axis.
Classical stereo rejects inconsistent reverse matches, weak texture, poor photometric agreement and triangulations moving more than 2 mm under a quarter-pixel disparity perturbation; this is a conditioning bound, not calibrated accuracy.
A seed requires at least three supported samples within four pixels and no spread beyond 12 mm from their median. Missing, ambiguous, occluded or malformed evidence returns perception_failed and plan_detail. Holes are never filled and free space is not inferred.
`robo rgb-triangulate --u U --v V --u2 U2 --v2 V2 --camera2 head|wrist_l|wrist_r [--camera head|wrist_l|wrist_r]`
Read-only triangulation of caller-selected corresponding pixels in two distinct current RGB views; accepts fractional pixels and the same camera aliases. Selections must represent the same physical point; silhouette extrema and apparent ellipse centres are not generally corresponding points.
Returns point_world (metres), stereo_pair, ray_gap_m, reprojection_px, conditioning_m and correspondence_verified=false/geometry_verified=false; does not automatically match pixels or infer an interior axis.
Rejects malformed/out-of-image pixels, baseline below 15 mm, ray condition number above 50, ranges outside 0.05–3 m, ray gap above 3 mm, reprojection above 1.5 pixels, or worst simultaneous half-pixel perturbation above 4 mm. These checks bound geometric consistency, not selection correctness; failures return perception_failed without motion.

`robo axis-fit --u U --v V [--camera head] [--window 40] [--band 0.015] [--reach 0.07]`
Motion-free local vertical circular-axis fit from calibrated RGB stereo correspondences; no command-budget or action-step cost. camera accepts head/wrist_l/wrist_r or cam_head/cam_left_wrist/cam_right_wrist; camera_source reports the resolved observation key.
u/v select a visible curved side surface pixel. window is pixel half-width (8–100); band is world-Z half-height (0.01–0.04 m); reach is horizontal distance from the selected surface point (0.03–0.12 m).
Returns plan_ok, plan_fail_reason, centre_xy (world metres), radius_m, surface_world, observed_z_range, surface_points, inlier_points, radial_rmse_m and arc_degrees.
centre_xy estimates the interior axis, unlike surface_world which lies on the visible exterior. No grasp height or top height is inferred.
Requires a predominantly vertical circular surface of radius 0.008–0.06 m, at least 70° visible arc and 0.015 m vertical extent; checks consistency across two height bands.
Fails with perception_failed and plan_detail for invalid inputs, missing RGB correspondences, insufficient support or inconsistent fits. geometry_verified=false: model fit does not establish identity, attachment or collision clearance.
