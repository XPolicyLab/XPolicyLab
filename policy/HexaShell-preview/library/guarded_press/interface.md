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
