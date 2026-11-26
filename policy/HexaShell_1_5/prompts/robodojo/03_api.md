# 3. The `robot` API

Cameras are named 'head', 'left_wrist' and 'right_wrist' ('left' and 'right' work too). Arms are
'left' and 'right'.

Motion - uses steps; each call returns when the motion is done:

| call | what it does | returns |
|---|---|---|
| `robot.goto(arm, xyz, pitch=, yaw=, roll=, grip=, path='auto')` | moves the grasp point to `xyz` with those angles (`path`: 'auto' a straight line, else a joint swing close to it; 'line'; 'swing' any curve); `grip` ('open', 'close' or 0..1) starts as the move starts: open on the approach with it, but let go of a held object with `gripper` once there | Move |
| `robot.goto(arm, by=(dx, dy, dz))` | moves relative to where it is (with `xyz` too: relative to `xyz`) | Move |
| `robot.goto(..., dry_run=True)` | only checks: reachable or not, path, steps, and whether it comes next to the other arm or goes into a measured Thing; nothing moves | Move |
| `robot.goto(..., contact=True)` | for a press, push or insertion meant to meet resistance: it is not stopped early and raises neither `Blocked` nor `Dropped` | Move |
| `robot.goto(..., gentle=True)` | no faster middle on long moves, for loads that slip or swing when moved fast (tall, top-heavy, hanging or loosely held); a 25 cm carry takes about 4 steps more - the default is faster | Move |
| `robot.gripper(arm, opening)` | 'open', 'close' or 0..1 (1 = 8.9 cm between the pads) | Grip |
| `robot.home(arm='both')` | lifts (if low) and swings back to the start pose; grippers unchanged | Homing |
| `robot.wait(n)` | holds everything still for n steps (n <= 200) | Waited |

Seeing and measuring - no steps used:

| call | what it does | returns |
|---|---|---|
| `robot.look(cams='head', grid=, marks=, masks=, crop=, zoom=, tile=)` | attaches the current picture(s) to this call's result (see section 4) | Picture |
| `robot.find(cam, 'red cup')` | outlines every instance the words describe, best first | list of Mask |
| `robot.find(cam, at=(x, y))` | outlines the object under that pixel | list of Mask |
| `robot.find(cam, points=[(x, y), ...], labels=[1, 0, ...])`, `box=(x0, y0, x1, y1)` | outline from several points (1 = on the object, 0 = not on it) or a box | list of Mask |
| `robot.measure(target)` | 3-D position, top, size and grasp angle of one object; `target` is words, a Mask, `(cam, (x, y))`, or a Thing measured before | Thing |
| `robot.to_world(cam, (x, y))` | the world point seen at a pixel; `z=` or `pair=(cam2, (x2, y2))` (section 4) | Spot |
| `robot.to_pixel(xyz, cam)` | where a world point appears in a camera's picture, or None | (x, y) |

Reading - no steps used:

| call | what it gives |
|---|---|
| `robot.task` | instruction, steps used and left, start poses |
| `robot.state()` | both arms: grasp point, angles, gripper, holding, distance from the start pose; steps. One arm: `robot.state().left` - `.at`, `.angles`, `.gripper`, `.holding` |
| `robot.image(cam)` | the current picture as an RGB numpy array (480, 640, 3) |
| `robot.camera(cam)` | a camera's focal length, centre and pose now |
| `robot.things` | every object measured so far (the latest measurement of each) |

Results print as one line or a few, with problems first. Their fields can be used in code:

```
Move left: at (0.112, -0.083, 0.861) pitch 90 yaw 12 roll 0  ±1 mm 0°  14 steps (line)  [986 steps left]
```
`.ok` (reached, not blocked, nothing slipped out), `.reached`, `.blocked`, `.lag_mm` (how far it
ended behind its command), `.dropped` (what it held slipped out on the way; `Homing` has it too),
`.at`, `.angles`, `.steps`. A pose that cannot be reached raises `NotExecuted` (`Move left: NOT
executed` with the reason and `reachable instead: ...`): nothing moved, and the rest of the cell does
not run. `BLOCKED`: the arm stopped short of the target - it is touching something (the note names a
measured Thing at the hand, or the other arm when that one has moved too). A move that stalls against
something stops early (`stopped after k of n actions`); it, and one that ends BLOCKED by more than
25 mm, raises `Blocked` with the Move, so the rest of the cell does not run and you can look. What
it holds slipping out stops the motion and raises `Dropped` (when the fingers end only nearly shut,
`.dropped` and a note say so instead); in the last 2 cm of a descent it only says so.
A note says when the planned path goes into a Thing measured so far, below its top (boxes are off by
1-2 cm), and when what the hand holds has shifted in the fingers (its width changed by over 3 mm).

```
Grip right: closed on something ~3.4 cm wide (holding)  6 steps  [980 steps left]
```
`.holding` (True: the fingers stopped on something; False: open, or closed fully; None: they
stayed about fully open - blocked), `.width_m`, `.opening`. An open that stops well short, less than
2 cm wider than what it held, raises `Blocked` with the Grip: it may still be caught between the fingers.
The first close after an open, empty descent held up over 10 mm short (the open fingertips rest on
something) is not done (`NOT closed`) and raises `Blocked`: grasp another way, or close again.

```
Thing#3 'red cup'  center (0.112, -0.083, 0.801)  top 0.836  size 0.071×0.069×0.071 round  grip_yaw any  on table  2 views (head, left_wrist) ±3mm
```
`.center` (x, y, z: the middle of the object), `.top`, `.size` (length, width, height), `.grip_yaw`
(the yaw that closes down-pointing fingers across its narrow side; None when round: any yaw), `.on` and `.on_z` (what it
stands on, and that surface's height), `.views`, `.err_mm`, `.masks`. A warning comes first when
there is one (`⚠ 1 view: height guessed - ...`). When the object is not found it prints
`NOT FOUND` with the reason and the candidates, and `.center` raises NotFound.

```
Mask#7 head 'red cup' score 0.91  centroid (412, 233)  box (391, 210)-(433, 258)  1520 px
Spot (0.112, -0.083, 0.836) from head (412, 220) on the same surface in head and left_wrist  (~1.6 mm per pixel)
```

Errors are Python exceptions with a message saying what to change: `ValueError` (bad or impossible
arguments; nothing happened), `NotExecuted` (a goto that cannot be done; nothing moved), `Blocked` and
`Dropped` (a goto that ran and was stopped, or an open still on what it held; the message is the Move
or the Grip), `NotFound`, `RobotError`
(`NotExecuted`, `Blocked` and `Dropped` are ones); the names are defined in the session. Catch them
in code when you have a plan B, or check a doubtful pose first with `dry_run=True`, which reports
instead of raising.
