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
