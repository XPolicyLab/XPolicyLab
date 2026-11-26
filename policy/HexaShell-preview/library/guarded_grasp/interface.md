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
