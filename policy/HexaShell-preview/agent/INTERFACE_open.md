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
