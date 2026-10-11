## Tool: pick

`robo pick <arm> --x X --y Y --z Z [--open x|y] [--approach down|down45] [--clearance M] [--lift M]`

Grasps from above at the world point (X, Y, Z): the gripper turns to point down (`--approach`, default `down`) with the
fingers opening along `--open` (default x), moves to `--clearance` (default 0.10) above the point, opens, descends to the
point, closes, and lifts by `--lift` (default 0.10). Costs one command. If the point above the grasp is out of reach for
a gripper pointing straight down, the tool retries once with `down45` (tilted 45 degrees forward, which reaches about
0.15 m farther in +y). The output lists the motion stages (point, approach, descend, lift) with `approach`, `plan_ok` and
`error_m` each, and `approach` tells which direction was finally used; if a stage fails the later ones are not executed.
The gripper is closed at the end.
