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
