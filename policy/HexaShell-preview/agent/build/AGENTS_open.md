# Robot interface

You control two robot arms on a table. The task is given in your prompt. You have not seen this task before:
work it out from the instruction, the task description and the camera images.

## What you can do

- Write Python and run it with `robo py` (see "Python"). This is the main way to act: one cell can measure, move,
  check the result and stop when something goes wrong, so a whole grasp is one cell, not ten separate commands.
- Run single `robo` commands in the shell (see "Commands"). They do the same as the Python functions.
- Look at the camera images in `obs/` and at the pictures you make with `show()`, using your image viewing tool.
- Read `obs/state.json`.
- Keep notes in `notes.md`.

## World frame

One right-handed frame is used for both arms and all cameras. Units are meters, degrees and seconds.

- Origin: on the floor, midway between the two arm bases.
- +x: to the right, as seen from the robot. +y: forward, away from the robot. +z: up.
- The table top is near z = 0.76; measure it with `table_z()` (`robo table_plane`).
- Left arm base: x = -0.30, y = -0.45. Right arm base: x = +0.30, y = -0.45.
- In the head camera image, +x points to the right of the image and +y points to the top of the image.
- `frame.png` shows the frame from above.

## Gripper

- TCP (tool center point): the midpoint between the two fingertips. All positions refer to it.
- Tool frame, attached to the TCP: x is the direction the gripper points to (`approach_dir`), y is the axis along
  which the fingers open (`open_dir`), z = x cross y.
- Gripper value: 1 is fully open (8.8 cm between the fingers), 0 is fully closed.
- After a gripper command the feedback tells what the fingers did:
  - `fingers_stopped`: true when the fingers stopped before the commanded opening. After closing, true means
    something is between the fingers; false means they closed on nothing (or on something thinner than about 2 mm).
  - `opening_m`: the opening between the fingers in meters, `opening_range_m` its likely range. When the fingers
    stopped, this is close to the width of what they hold (about +-3 mm).
  - The same fields are in `obs/state.json` for each arm and keep showing whether something is still held.
- At the start both grippers are open, point forward (+y), and are 0.18 m above the table.
- Reach: pointing straight down, the TCP reaches forward to about y = 0.00 at table height, and less higher up.
  Pointing `down45` it reaches to about y = 0.15. Use `reach()` to test a pose before moving.

## Python

`robo py` runs Python code in a kernel that stays alive for the whole episode: variables, functions and imports
from earlier cells are kept. numpy is available as `np`; there is no OpenCV. Pass the code on stdin:

```
robo py <<'EOF'
s = state()
print(s["left"]["tcp_pos"], s["sim_time_left_s"])
EOF
```

The value of a final expression is printed. Each robot function prints one summary line. A failed command
raises `RoboError` (its feedback is in `error.feedback`) so the rest of the cell does not run on a bad assumption;
pass `check=False` to get the feedback instead. When the episode ends, every command raises `EpisodeOver`.
Arms are `"left"` and `"right"`. Cameras are `"head"`, `"wrist_l"` and `"wrist_r"`.

Robot (each returns the command's feedback as a dict; see "Output of a command"):

| Function | Effect |
|---|---|
| `goto(arm, x, y, z, approach=None, open=None, path="line")` | Move the TCP to (x, y, z). `approach`: `"down"`, `"forward"`, `"down45"` or a direction vector, e.g. `tilt(30, 90)`. `open`: `"x"`, `"y"`, `"z"`, an angle in degrees, or a vector. None keeps the current value. Pointing down, `open=a` makes the fingers open along the world-xy direction at angle a from +x (counterclockwise seen from above). `path="line"` moves the TCP on a straight line; `path="joint"` interpolates the joints (the TCP path is curved, use it only in free space) |
| `move(arm, dx=0, dy=0, dz=0)` | Translate the TCP, orientation unchanged. At most 0.20 m per axis |
| `rotate(arm, roll=0, pitch=0, yaw=0, frame="world")` | Rotate about the TCP, position unchanged; `frame="tool"` uses the tool axes. At most 90 degrees per axis |
| `point(arm, "down"\|"forward"\|"down45", open="x"\|"y"\|"z")` | Point the gripper, position unchanged |
| `gripper(arm, value)`, `open_gripper(arm)`, `close_gripper(arm)` | Set the opening (1 open, 0 closed) |
| `home(arm="both")` | Return to the start pose |
| `wait(sec)` | Let the simulation run, at most 5 s |
| `reach(arm, x, y, z, approach=None, open=None)` | Is this pose reachable? Nothing moves and no simulation time is used. If not, `reachable_nearby` lists close poses that are |
| `done()` | End the episode now (see Rules) |
| `tilt(degrees, heading)` | Approach vector tilted from straight down towards the world-xy heading (0 = +x, 90 = +y) |

Observation and geometry (no simulation time):

| Function | Result |
|---|---|
| `obs()`, `state()` | Refresh and/or read `obs/state.json` |
| `tcp(arm)` | Current `pos`, `approach`, `open` and gripper fields of one arm |
| `img(camera)` | The latest image as an array, 480 x 640 x 3, RGB, indexed `[v, u]` |
| `K(camera)`, `T(camera)` | Intrinsics 3x3 and camera-to-world 4x4 |
| `ray(camera, u, v)` | (origin, unit direction) of the ray through a pixel |
| `pixel_to_plane(camera, u, v, z)` | World point where that ray meets the horizontal plane at height z |
| `project(point, camera)` | Pixel (u, v) of a world point, or None if it is behind the camera |
| `triangulate(cam_a, (u, v), cam_b, (u, v))` | (point, gap): the world point seen at both pixels; a gap above about 5 mm means the pixels are not the same point |
| `table_z()` | World z of the table top, measured from the images |
| `show(camera or array, points=[(u, v)], boxes=[(u0, v0, u1, v1)], mask=..., crop=(u0, v0, u1, v1), scale=2)` | Draw numbered marks on an image, optionally crop and zoom, save it under `obs/`, return the path to look at |
| `segment(camera)`, `locate(camera, id=N)` or `locate(camera, u=U, v=V)` | Object regions in one image, and the 3D top centre, size and yaw of one of them (`robo segment`, `robo locate`, see the scene tool at the end) |
| `tool(name, ...)`, `tools()` | Run one of the extra commands at the end of this manual; `tools()` lists them |

Images change only when a command runs. Look again after every motion before you rely on an image.

## Commands

The same actions as shell commands:

| Command | Effect |
|---|---|
| `robo py [FILE]` | Run Python code from FILE or stdin in the kernel |
| `robo obs` | Refresh `obs/` |
| `robo goto <arm> [--x M] [--y M] [--z M] [--approach down\|forward\|down45\|a,b,c] [--open x\|y\|z\|DEG\|a,b,c] [--path line\|joint]` | Move the TCP to an absolute pose; omitted values are kept |
| `robo reach <arm> [--x M] [--y M] [--z M] [--approach ...] [--open ...]` | Reachability check, nothing moves |
| `robo move <arm> [--dx M] [--dy M] [--dz M]` | Translate the TCP |
| `robo rotate <arm> [--roll D] [--pitch D] [--yaw D] [--frame world\|tool]` | Rotate about the TCP |
| `robo point <arm> <down\|forward\|down45> --open <x\|y\|z>` | Point the gripper |
| `robo gripper <arm> <open\|close\|0..1>` | Set the gripper opening |
| `robo home <arm\|both>` | Return to the start pose |
| `robo wait <sec>` | Let the simulation run, at most 5 s |
| `robo pixels [--camera C]` | Write the images as numpy arrays `obs/<camera>_rgb.npy` |
| `robo status` | Print the feedback of the last command again |
| `robo done` | End the episode now (see Rules) |

## Limits

- Simulation time is limited per task. Every command that moves the robot or waits uses simulation time;
  `sim_time_left_s` tells how much is left. Measuring, thinking and Python code without robot commands use none.
- Your session is limited to about 60 minutes of real time. Prefer one cell that does a whole step and checks it
  over many small calls.
- There is no limit on the number of commands.
- The episode ends when you run `done()`, when the simulation time is used up, or when the environment ends it
  (for example because the task is complete). After that every command fails with `EpisodeOver` / exit code 3.
- A command can take up to a few minutes of real time to return.

## Output of a command

Every command returns (prints, in the shell) one JSON object.

| Field | Meaning |
|---|---|
| `requested` | What you asked for |
| `clipped` | true if a value was cut to its limit |
| `workspace_limited` | true if the target was moved back inside the workspace (x -0.75..0.75, y -0.75..0.60, z 0.74..1.45) |
| `plan_ok` | false if no motion to the target was found; the arm did not move |
| `plan_fail_reason`, `plan_detail` | `ik_unreachable`, `self_collision`, `table_collision` or `workspace_limit`, and where it failed |
| `target_reachable`, `reachable_nearby` | After a failed `goto`: whether the target pose itself is reachable, and nearby poses that are |
| `reached_tcp` | Pose of the TCP after the motion: `pos`, `quat` [qw, qx, qy, qz], `rpy` |
| `error_m`, `error_deg` | Distance and angle between the target and the reached pose. A large error after a motion usually means the arm pushed against something |
| `settled` | true if the arm came to rest at the target |
| `gripper`, `fingers_stopped`, `opening_m` | Commanded gripper value, and what the fingers did (see Gripper) |
| `sim_time_s`, `sim_time_left_s` | Simulation time used and left |
| `episode_over` | Present and true once the episode has ended |

Exit codes of the shell command: 0 ok, 1 wrong arguments (nothing ran), 2 the command failed, 3 the episode is
over, 4 the client timed out (use `robo status`).

Motion planning knows the arm and the table. It does not know the objects on the table or the other arm.

## Observations

Commands that move the robot refresh `obs/` before they return.

| File | Content |
|---|---|
| `obs/head.png` | Head camera, 640x480, fixed above the robot, looking forward and down |
| `obs/wrist_l.png` | Camera on the left wrist, 640x480; it moves with the gripper and sees the fingers |
| `obs/wrist_r.png` | Camera on the right wrist, 640x480 |
| `obs/state.json` | State of both arms and the camera parameters |

There is **no depth image**. Positions in 3D come from the colour images and the camera parameters: a pixel
defines a ray from the camera; where that ray meets a plane of known height (the table, or the top of an object
whose height you measured) or the ray from a second camera to the same point is where the point is.

`obs/state.json`:

- `left`, `right`: `tcp_pos`, `tcp_quat`, `tcp_rpy` (extrinsic x-y-z, degrees), `approach_dir`, `open_dir`,
  `gripper`, `gripper_observed`, `fingers_stopped`, `opening_m`, `opening_range_m`.
- `cameras.<name>.intrinsics`: 3x3 matrix K in pixels.
- `cameras.<name>.extrinsics_world`: 4x4 matrix T that maps camera coordinates to world coordinates. Camera axes:
  x right, y down, z forward (along the view).
- `cameras.<name>.size`: [width, height].
- `sim_time_left_s`, `step`.

A world point p is seen at pixel (u, v) with p_cam = inverse(T) * p, u = K[0][0] * p_cam.x / p_cam.z + K[0][2],
v = K[1][1] * p_cam.y / p_cam.z + K[1][2]. The wrist cameras move with the arms: their T changes after every
motion, so use the T of the same observation as the image.

## How to work

These notes hold for tabletop tasks in general.

- Read the instruction and the task description (TASK RECIPE in your prompt, when there is one) first, then look at all three images. Name every object you need,
  where it is and what has to change. Language or picture clues in the instruction (colours, order, a sign, a
  written number, a reference picture) decide which object is meant: check them against the images, zoom in with
  `show(..., crop=..., scale=3)` when objects are small.
- Measure before you act. Get positions in 3D from two cameras or from a ray and a known plane, and check them:
  project the 3D point back into the other cameras with `project()` and look at the marks with `show()`.
  The head camera sees the whole table; a wrist camera held 15-25 cm above an object, pointing down, gives a
  sharper close view of it. Heights are best measured with two views (`triangulate`, or the extra measuring
  commands).
- Grasp from above when you can: point down with the fingers opening across the narrow side of the object, go
  6-10 cm above it, descend straight to the grasp height, close, check `fingers_stopped`, then lift straight up.
  For a low object put the TCP 1-2 cm below its top; for a tall one around its upper middle. A wide opening before
  closing tolerates a few millimetres of position error.
- Motions are straight lines and do not avoid objects. Lift before moving sideways, come down vertically, and
  keep the other arm out of the way (`home` it, or move it aside).
- Each arm works best on its own side (left arm x < 0, right arm x > 0). Far forward targets (large y) need a
  tilted approach (`down45` or `tilt()`); test them with `reach()` first.
- To put something down, lower it until it is about 1 cm above the surface, open, then lift straight up before
  moving away. Objects dropped from higher bounce, roll or tip over.
- To push or slide an object, lower the closed gripper to 1-2 cm above the table beside it and move in a straight
  line through it; push in steps and measure again between steps.
- To turn or tilt a held object, use `rotate(..., frame="tool")` or `goto` with a new approach direction; check
  `reach()` first, many orientations are not reachable everywhere.
- After every important step, check the result in the images (and the gripper feedback) before going on. If it
  failed, measure again and change something; do not repeat the same numbers.
- Simulation time is limited: a grasp-and-place takes about 5-10 s of simulation time. Compare `sim_time_left_s`
  with what the remaining steps need.

## How the result is judged

- The environment checks the scene itself. The episode ends by itself the moment the task is judged complete; while
  it is still running, the task is not complete.
- Positions are judged in the final scene, after you finish, usually to within a few centimetres and with objects
  at rest. Objects that were knocked over or dropped usually count against you.
- Many tasks require both arms to be back at the start pose at the end.

## Rules

- When you think the task is finished, return both arms to the start pose with `home("both")`. If the episode is
  still running after that, compare the scene with every condition in the task description and the instruction,
  fix what is missing, and home again.
- Partial results count (see the score table in the task description, when there is one). While simulation time is
  left, keep working on the parts that are not done.
- Do not end your reply while the episode is running and simulation time is left.
- `done()` stops the episode where it is. Run it only when the time left cannot cover one more attempt.
- If a command fails, read its output, look at fresh images and measure again before you try again.
- You are not told the positions of the objects. Work them out from the images and the camera parameters.

## Extra commands

These are robo commands too; in Python call them with `tool(name, ...)` (see above).

## Tool: table_plane

`robo table_plane` prints `table_z`, the world z of the table top in meters, measured from the camera images, and `conf` (0 to 1) for how reliable it is; the value is kept for the rest of the episode and the command costs no budget.
`table_z` is null when the images show too little of the table: move an arm so that its wrist camera sees a larger part of the table, then run `robo table_plane --refresh 1`.

## Tool: scene

Three commands that find and measure the objects on the table from the camera images. They use no simulation time and the robot does not move.

### `robo segment [--camera head|wrist_l|wrist_r] [--min_area PX]`

Finds the object regions in one camera image: everything that is not table and not robot. The default camera is `head`.
It also writes `obs/seg_<camera>.png`: the image with each region outlined and its id written on it. The robot is tinted blue. The area that does not count (off the table, or seen too flat) is darkened. Look at this image before you trust a region.

The output lists `objects`, sorted by x from left to right. Each object has:

- `id`: the number to use with `robo locate --id`. An id belongs to this camera and this image only. After the robot moves, run `robo segment` again.
- `bbox` [u0, v0, u1, v1], `centroid_px` [u, v], `area_px`: where the region is in the image, in pixels.
- `color`: the main colour; `colors`: every colour that covers at least 15 % of the region; `mean_rgb`.
- `bottom_xy` [x, y]: the point of the region nearest to the camera, on the table. For an object that stands on the table this is a point on its near bottom edge, not its centre.
- `table_xy` [x, y]: where the ray through the region's centre pixel meets the table. It lies behind the object (farther from the camera), more so for tall objects. Use it only as a rough position.
- `touches_border`: the region runs off the image, so part of the object is not seen.
- `near_robot`: the region touches the robot in the image. Part of it may be hidden, or the robot holds it.
- `clipped`: the region ran into the background behind the table and was cut there.

`gripper_seen` gives the opening of each gripper as seen in its own wrist camera (0 closed, 1 open). It is only given when the commanded opening in `obs/state.json` is below 1. If it is clearly larger than the commanded value, the fingers are stopped by something they hold.

Limits:
- Two touching objects of clearly different colours become two regions. Touching objects of the same colour stay one region.
- An object with parts of very different colours can become two regions.
- Objects with the colour of the table (wood, brown) and very dark objects in shadow can be missed or found only in part.
- Objects behind the line of the two arm bases are not listed, and neither is the edge of the table.
- `--min_area` (default 40 pixels) drops smaller regions. Use a smaller value for small objects far from the camera.

### `robo locate --camera C --id N [--camera_b auto|head|wrist_l|wrist_r]`
### `robo locate --camera C --u U --v V [--camera_b ...]`

Measures one object in 3D from two or more cameras. Give the id from the latest `robo segment --camera C`, or a pixel (U, V) on the object in camera C.
It finds the same object in another camera and compares the two views on horizontal planes at many heights. With `--camera_b auto` (default) it uses every other camera that sees the object.

Output:

- `top_z`: world z of the top of the object, in meters.
- `footprint_xy` [x, y]: the centre of the object's top face. For an object with straight sides (a box, a cylinder, a bowl) this is the centre of the object.
- `top_xyz`: [x, y, top_z] of that centre.
- `extents`: `length` and `width` of the top face (meters, length is the longer side) and `height` of the top above the table.
- `yaw_deg`: the direction of the long side in the xy plane, in degrees from +x toward +y, between -90 and 90. It means little when length and width are about the same.
- `cameras`: the cameras that were used. `matched_ids`: the id of the object in each other camera. `pair_heights`: the height found with each other camera.
- `iou`: how well the views agree at the measured height (1 is perfect, below 0.5 is weak). `conf`: 0 to 1, higher with more cameras and better agreement.
- `lifted`: present and true when the object does not reach down to the table: it is held, or it stands on another object. `base_height` is then roughly where its bottom is, above the table.
- `partly_seen`: present when another camera sees only part of the object (image border or robot in the way).
- `notes`: why a camera was not used, or that the cameras disagree on the height.

When it fails (exit code 2), `reason` says why. The most common reason is that only one camera sees the object: in the start pose the wrist cameras look forward and do not see the table near the robot. `hint` then says where to put a gripper so that its wrist camera sees the object: pointing down with the fingers opening along x, a wrist camera sees the table from about 0.04 m ahead of the TCP to 0.30 m behind it (toward the robot). After the move, run `robo segment` and `robo locate` again.

Limits:
- The measurement is best for objects with a flat top. For a round or uneven top, `top_z` is near the highest part, and `footprint_xy` is the centre of the outline seen from above.
- For a thin object lying flat, `top_z` is close to the table and the height is only good to a few millimeters.
- For an object of irregular shape (a mug with a handle, a brush, a tool), `footprint_xy` is the centre of what both cameras see from above, not a grasp point.
- An object partly hidden in one camera gives a smaller `iou` and a less exact result. Check `partly_seen` and `notes`.
- When two cameras see the whole object, `footprint_xy` is usually within 1 cm.

### `robo pixel_world --camera C --u U --v V [--z Z]`

Gives the world point `xyz` where the ray of pixel (U, V) meets the horizontal plane at height Z. Without `--z` the plane is the table top. Use it for a point you know lies at that height.

## Tool: rgb_point

Use it to find the 3D world position of a point that you see in a camera image (a spot on a surface, the top of an object, a corner).

`robo rgb_point --u U --v V [--camera C] [one mode below]`

Free: nothing moves and no simulation time is used. U, V: pixel column and row in the latest image of camera C (default `head`). Cameras: `head`, `wrist_l`, `wrist_r` (`cam_head`, `cam_left_wrist`, `cam_right_wrist` also work).

Modes:

- Automatic (no other arguments): measures the height of the surface around the pixel from the other two cameras. It matches a small square patch around the pixel (`--radius R`, half size in pixels, 2–15, default 6) and assumes that this patch is flat and horizontal. If the patch is plain (almost one colour), it uses the outline of the one-coloured region that contains the pixel instead; that outline must lie in one horizontal plane (for example the flat top of something seen from above) and must not touch the image border. Heights between `--z_min` and `--z_max` are searched (m, default 0.50 and 1.05; the true height must lie inside).
- `--z Z`: the point lies on the horizontal plane at world height Z (m), for example the table height from `robo table_plane`. Nothing is measured; a wrong Z gives a wrong point.
- `--camera2 C2 --u2 U2 --v2 V2`: (U2, V2) is the same physical point in camera C2. The two pixel rays are intersected.
- `--reference R`: R is the `reference` text printed by `robo rgb_ray` for the same physical point, taken earlier from a different camera position (for example before you moved a wrist). The point must not have moved since then. Put R in single quotes.

Output: `point_world` [x, y, z] in m and `method`. Automatic mode also gives `matched_cameras` and, per camera, the result or the reason it failed (`pairs`). The two-ray modes give `ray_gap_m` (at most 0.003 m) and `crossing_sine` (at least 0.05).

`robo rgb_ray --u U --v V [--camera C]`

Free. Prints `reference`: the ray of this pixel (camera origin and direction) as JSON text, for `--reference` later.

Fails with `plan_ok` false, a `plan_fail_reason` and exit code 1 (nothing moved) when the pixel is outside the image, when no other camera gives a clear and unique match (`no_unique_rgb_surface_match`, `ambiguous_surface_match`, `nonplanar_or_ambiguous_patch`, `camera_surface_disagreement`; `diagnostics` gives the reason per camera), when the two rays miss each other by more than 3 mm (`rays_disagree`: the pixels are not the same point) or are nearly parallel.

Tips: pick a pixel well inside one flat face, at least `radius` pixels from its edges; sloped faces, edges and blank areas fail. A pixel on a top face gives the top surface, not the centre of the object. The result is not checked against the arms: a pixel on a gripper measures the gripper.

## Tool: rgb_quad

Use it to measure a flat face with four straight sides (the top of a box, a tray, a pad or a lid) in 3D: its corners, centre, side lengths, direction and tilt.

`robo rgb_quad --regions '[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]]' [--camera head] [--other wrist_r]`

Free: nothing moves and no simulation time is used. The first rectangle is in the image of `--camera` (default `head`), the second in the image of `--other` (default `wrist_r`); the two cameras must differ. Cameras: `head`, `wrist_l`, `wrist_r`. Rectangles are in pixels, at least 16 pixels wide and high.

Each rectangle must contain the whole outline of the same face, with at least 3 pixels of free space between the outline and the rectangle border, and no second complete four-sided outline. The sides must be straight, clearly visible and not covered by anything, at least 12 pixels long in the image, and the corners must not be sharper than 20 degrees. The tool finds the four sides in each image, pairs the corners of the two views and triangulates them.

Output:

- `vertices_xyz`: the four corners [x, y, z] in m, in order around the face.
- `center_xyz`: their mean. `edge_lengths_m`: lengths of the sides 0-1, 1-2, 2-3, 3-0.
- `plane_fit`: `normal`, `max_residual_m`, `tilt_from_horizontal_deg`.
- `matched_pixels`: per corner [u, v, u2, v2] in the two images; `quality`: stereo checks per corner.

The direction of a side in the horizontal plane is atan2(y1 - y0, x1 - x0) of its two corners.

Fails with `plan_fail_reason` `quadrilateral_unavailable` and a `detail` (exit code 2, nothing moved) when a rectangle holds no outline or more than one ("found N"), the outline touches the rectangle border, a side is curved, broken or hidden, the corner pairing is not unique, the 3D corners are not flat within 5 mm, or the stereo geometry is weak (ray angle below 3 degrees, ray gap above 8 mm, reprojection above 2 pixels, or more than 15 mm change per pixel of error). Then draw tighter rectangles, or move a wrist camera so that it sees the face better. It does not measure thickness or hidden parts.

## Tool: boundary_candidates

Use it to find round or oval edges in one camera image (the rim of an opening, the edge of a disc or a hole, the top of a cylinder) and to get exact edge pixels, for example as input for `robo fit-rim`.

`robo boundary_candidates --roi '[left,top,right,bottom]' [--camera head|wrist_l|wrist_r]`

Free: nothing moves and no simulation time is used. `--roi` is a rectangle in image pixels (whole numbers), inside the image, 12–320 pixels wide and high; default camera `head`. Draw it around the whole edge with a few pixels to spare: an edge that comes within 2 pixels of the rectangle border is skipped.

Output: `candidates`, up to 8 closed edges of near-elliptical shape found in the rectangle, the most complete first. Each has:

- `points`: 22–24 pixels [u, v] on the edge, spread around it.
- `ellipse_center_px`, `ellipse_diameters_px`, `ellipse_angle_deg`: the ellipse fitted to the edge in the image.
- `angular_sectors`: how many of 24 sectors around the ellipse the edge covers; `contour_rms_px`: how well it fits; `channel`: the colour channel it was found in.

One object can give several candidates, for example the inside and the outside of a rim: compare `ellipse_center_px` and the diameters with the image to choose. The ellipse centre in the image is not the image of the 3D circle centre; use `robo fit-rim` with the `points` of the same edge in two cameras to measure the circle.

Fails with `no_closed_rgb_boundary` (exit code 2, nothing moved) when the rectangle holds no closed edge with at least 8 pixels small diameter, at most 6:1 axis ratio and at least 22 of 24 sectors covered; an edge that is partly hidden is not found.

## Tool: rim_geometry

Use it to measure a level circle in 3D (the rim of a round container, a round opening or hole, the top edge of a cylinder or disc): its centre, radius and height.

`robo fit-rim --views '{"head":[[u,v],...],"wrist_r":[[u,v],...]}' --zmin Z --zmax Z [--tolerance 0.003] [--snap 4]`

Free: nothing moves and no simulation time is used.

- `--views`: 2 or 3 cameras (`head`, `wrist_l`, `wrist_r`), each with 6–64 different pixels on the same physical circular edge. The pixels do not have to be the same points in the different views, but in each view they must be spread around the circle, with no gap larger than half of it. The `points` of a `robo boundary_candidates` candidate can be used as they are. The cameras must be at least 3 cm apart.
- `--zmin`, `--zmax`: world heights in m between which the circle lies, for example the table height and 0.30 m above it. The best height must lie inside this range, which may be at most 1 m wide.
- `--snap` (0–6 pixels, default 4): first moves each pixel to the nearest strong image edge within this distance, across the edge. Every pixel needs such an edge. 0 uses the pixels as given.
- `--tolerance` (0.0005–0.01 m, default 0.003): largest allowed distance of a point from the fitted circle; also at most 10% of the radius.

Output: `center_world` [x, y, z] in m (z is the height of the edge), `radius_m`, `normal_world`, `max_residual_m`, `rms_residual_m`, `cameras` and, per camera, the snapped pixels (`refinement`).

The circle must be level and have a radius of 5 mm to 20 cm. Fails with `geometry_fit_failed` and a `plan_detail` (exit code 2, nothing moved), for example: the pixels are not on one edge or not on the same edge in each view (`inconsistent_multiview_boundary`, `inconsistent_boundary`); they cover too little of the circle (`insufficient_boundary_coverage`, `insufficient_view_coverage`); the best height is at the end of the range (`height_outside_bracket`: widen the range); two heights fit about equally well (`ambiguous_height`: use cameras that see the circle from more different directions); no edge near a pixel when snapping (`boundary_not_supported_by_rgb`: use better pixels or `--snap 0`).

## Tool: guarded_grasp

Use it to pick up an object at a measured point with a chosen gripper direction (from above, tilted, or from the side), and to put a held object down at a point.

All positions are TCP positions (the point between the fingertips) in world metres. Each command uses the simulation time of its motions. Every motion is checked: the command stops at the first motion that fails or ends more than `--tolerance` (0.002–0.015 m, default 0.01) or 5 degrees from its target. It does not retry, and it does not close or open after a failed approach. A closed gripper does not prove that the object is held: look at the wrist image.

`robo grasp_point <left|right> --x X --y Y --z Z [--yaw 0] [--tilt 0] [--opening 0.5] [--lift 0.06] [--lift_dx 0] [--lift_dy 0] [--clearance 0.06] [--tolerance 0.01]`

Turns the gripper in place to point down (or tilted), rises if needed, moves above (X, Y), opens, goes straight down to (X, Y, Z), closes and lifts.

- `--yaw`: direction in which the fingers open, in degrees from world +x toward +y (0: along x, 90: along y), -180 to 180. Choose it so that the fingers close across the narrow side of the object.
- `--tilt`: 0–60 degrees away from straight down, toward the direction 90 degrees from yaw (yaw 0: leans forward, to +y; yaw 180: backward; yaw 90: to -x; yaw -90: to +x). It helps to reach farther forward; 45 is like `point down45`. The descent stays vertical.
- `--opening`: how far the fingers open before the descent, 0.05–1 (1 = 8.8 cm). The default 0.5 is 4.4 cm: use more for wide objects.
- `--clearance` (0.02–0.15 m): travel height above Z before the descent. The TCP never travels lower than it already is.
- `--lift` (0.02–0.15 m): rise after closing. `--lift_dx`, `--lift_dy`: also move sideways during the lift (m, together at most 0.15).

The gripper turns at its current position first, so start where the hand has free space around it.

`robo grasp_pose <left|right> --x X --y Y --z Z --ax AX --ay AY --az AZ --ox OX --oy OY --oz OZ [--opening 0.5] [--lift 0.06] [--lift_dx 0] [--lift_dy 0] [--clearance 0.06] [--tolerance 0.01] [--direct 0]`

Grasp with any gripper direction, for example from the side. A = (AX, AY, AZ) is the direction the gripper points; it moves along A into the grasp. O = (OX, OY, OZ) is the direction in which the fingers open. A and O need not have length 1, but they must be perpendicular. The gripper turns in place, rises, travels above the start point P = (X, Y, Z) - clearance * A, opens, goes down to P, moves along A to (X, Y, Z), closes and lifts. `--direct 1` goes straight to P without rising first, opens there and moves in 1 cm steps from P to the grasp. Here `--lift` may be 0 (no lift).

Example, a horizontal grasp moving forward (+y) with the fingers opening left-right: `--ax 0 --ay 1 --az 0 --ox 1 --oy 0 --oz 0`.

`robo place_point <left|right> --x X --y Y --z Z [--clearance 0.06] [--tolerance 0.01]`

Keeps the gripper orientation, rises to at least Z + clearance, moves above (X, Y), goes straight down until the TCP is at (X, Y, Z), opens fully and rises by `clearance`. Z is the TCP height at release: for an object held upright, Z = support height + (height of the grasp point above the bottom of the object) + 2 to 5 mm.

Output: `plan_ok`; `plan_fail_reason` (for example `ik_unreachable`, `tracking_error`, `workspace_limited`, `episode_over`, or `tool_failed` with `plan_detail`); `failed_stage`; `stages` (each motion with `error_m` and `rotation_error_deg`); `reached_tcp`; `close_commanded`; `released`; `tcp_lift_m`; `grasp_verified` (always false). Exit code 2 on failure. Paths are not checked for collisions with objects.

## Tool: guarded_press

Use it to press something (a button or a switch) or to push an object along a straight line with the fingertips.

`robo press_pose <left|right> --x X --y Y --z Z --ax AX --ay AY --az AZ --ox OX --oy OY --oz OZ --dx DX --dy DY --dz DZ [--clearance 0.04] [--increment 0.01] [--tolerance 0.008] [--opening 0]`

- (X, Y, Z): TCP position at the first contact, in world metres. The TCP is the point between the fingertips; to press straight down with closed fingers, Z is about the height of the surface.
- A = (AX, AY, AZ): direction the gripper points. O = (OX, OY, OZ): direction in which the fingers open. They must be perpendicular. Pointing down with the fingers along x: `--ax 0 --ay 0 --az -1 --ox 1 --oy 0 --oz 0`.
- D = (DX, DY, DZ): the stroke after contact, in m, 0.005–0.12 long. Press 1 cm down: `--dx 0 --dy 0 --dz -0.01`. Push 5 cm toward +x: `--dx 0.05 --dy 0 --dz 0`.
- `--opening` (0–1, default 0 = closed): gripper opening, set before the motion. Opening a hand that holds something drops it.
- `--clearance` (0.02–0.12 m): the approach starts this far back along A from the contact point, and the travel happens this much above the highest of the current, start and contact heights. `--increment` (0.005–0.02 m): length of each step of the stroke. `--tolerance` (0.002–0.015 m).

Sequence: turn the gripper in place, set the opening, rise, travel above the start point (contact point - clearance * A), move to it, move to the contact point, make the stroke in small steps, then withdraw by `clearance` back along A.

`robo press_feature <left|right> --fx FX --fy FY --fz FZ --x X --y Y --z Z --ax AX --ay AY --az AZ --ox OX --oy OY --oz OZ --dx DX --dy DY --dz DZ [--clearance 0.04] [--increment 0.01] [--tolerance 0.008]`

The same, but the contact is made with a point F = (FX, FY, FZ) that moves with the hand, for example the tip of a held object, measured now in world coordinates, 0.001–0.25 m from the TCP. F is brought to (X, Y, Z) and then moved by D. The gripper opening is not changed. It assumes that the held object does not slip.

Output: `plan_ok`, `plan_fail_reason`, `stages`, `reached_tcp`, `stroke_complete`, `achieved_stroke_m`, `activation_verified` (always false: look at the images to see whether the press had an effect). press_feature also gives `contact_tcp_world` and `reached_feature_world`.

It stops (exit code 2) at the first motion that fails, or when the TCP (or F) ends more than `--tolerance` from its target or more than 5 degrees off, for example when the surface resists. It then stays where it stopped, without withdrawing: move the arm away yourself. Each command costs the simulation time of its motions. Paths are not checked for collisions with objects.

## Tool: bounded_rotate

Use it to turn the gripper, and what it holds, by a large angle without moving the TCP, for example to tilt a held container to pour from it and back, or to turn a held object over.

`robo bounded-rotate <left|right> [--roll D] [--pitch D] [--yaw D] [--frame world|tcp] [--step_deg 20] [--tolerance 0.003] [--reserve_steps 40]`

- `--roll`, `--pitch`, `--yaw`: rotation about x, y and z in degrees (each -180 to 180, default 0), applied in that order, about the axes of the world frame (default) or of the tool frame (`--frame tcp`; this is `--frame tool` in `robo rotate`), with the same convention as `robo rotate`. The arm then turns the shortest way to the resulting orientation, about the TCP.
- The turn is made in steps of at most `--step_deg` degrees (5–30, default 20; smaller when needed to keep the TCP within half the tolerance). Before anything moves, every step is checked to be reachable.
- `--tolerance` (0.0005–0.005 m, default 0.003): it stops when the TCP moves more than this from its start, or when the orientation is more than 2 degrees off.
- `--reserve_steps` (0–400, default 40): it refuses to start when the estimated motion would leave fewer than this many simulation steps (25 steps per second).

Output: `plan_ok`; `plan_fail_reason` `rotation_failed` with a `plan_detail` (for example `preflight_unreachable; no motion attempted`, `insufficient_step_budget; no motion attempted`, `tracking_error_or_contact`, or the planner's reason); `increment_count`; `estimated_rotation_steps`; `preflight`; `stages` (per step: `error_m`, `rotation_error_deg`); `reached_tcp` (4x4 pose). After a failure during the motion the arm is stopped where it is (`target_cancelled`). Exit code 2 on failure.

The gripper opening does not change. It uses the simulation time of the motion. It does not check collisions or whether a held object stays in the hand.

The TCP is the point between the fingertips, so the turn is about that point: an object held at its side turns about that side. To pour, first carry the held container so that its opening is just above and beside the receiving opening, then turn.

## Tool: timed_pick

Use it to grasp an object that moves at a constant speed in a straight line (for example carried by a moving surface). With zero speed it is a top-down grasp at a set moment.

`robo timed_pick <left|right> --x X --y Y --z Z --vx VX --vy VY [--delay 3] [--clearance 0.08] [--lift 0.12] [--descent_lead 1.2] [--top_z 0] [--yaw 0] [--approach auto|down|down45] [--open y|x]`

- (X, Y, Z): world position in m of the grasp point (where the TCP should be when the fingers close) now, that is, in the latest images. The simulation stops between commands, so the latest measurement is "now".
- (VX, VY): measured velocity of the object along world x and y, in m/s; the speed must be at most 0.3 m/s.
- `--delay` (1–8 s, default 3): time from the start of the command to the middle of the closing. The grasp point is predicted as (X + VX * delay, Y + VY * delay, Z). When the arm cannot be there in time, the tool plans a later grasp (up to 4 tries) or fails.
- `--clearance` (0.03–0.20 m): height above Z at which the open gripper waits before it goes down. `--top_z`: measured world height in m of the top of the object (0: not given); the clearance is then raised so that the hand waits at least 3 cm above that top.
- `--descent_lead` (0.3–1.5 s, less than the delay): how long before the closing the descent starts; raised automatically when the descent takes longer.
- `--approach`: gripper direction, `down`, `down45`, or `auto` (`down`, and `down45` when `down` cannot be reached). `--open`: world axis along which the fingers open (default `y`). `--yaw`: then turns the gripper by this angle about the vertical (degrees, -180 to 180).
- `--lift` (0.05–0.25 m): rise after closing.

Sequence: rise, turn, open, move above the predicted point, wait, go down, wait, close at the predicted moment, lift.

To measure the velocity, measure the same point twice (for example with `robo rgb_point`), before and after `robo wait SEC`: velocity = change of position / SEC. Measure the position again just before `timed_pick` and pass that latest value.

Output: `plan_ok`; `plan_fail_reason` (for example `invalid_arguments`, `insufficient_time`, `intercept_late`, `intercept_unreachable`, `ik_unreachable`, `tracking_error`) with `plan_detail`; `stages` (each with `elapsed_s`); `intercept_xyz` (the predicted grasp point); `delay_s`; timing details; `grasp_verified` (always false: look at the wrist image). Exit code 2 on failure; when it failed after moving (`position_refresh_required`), measure the object again. It costs the simulation time of the motions and waits.
