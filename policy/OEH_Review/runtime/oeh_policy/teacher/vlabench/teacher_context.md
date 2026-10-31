# Task and predecessor knowledge

The goal is one recorded VLABench rollout of the task named in the first packet, controlled by this persistent agent
and the released VLABench checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-VLABench`), executed through OpenEmbodied
Harness. Derive the exact objective from the instruction and visible observations, not a hard-coded task script. This
rollout does not train the student.

The control method is the one OEH runs on RoboDojo, copied from the published GPT-as-Policy hybrid: fresh student
inference, outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved
verbatim. OpenWAM-α's published VLABench success rates by track are in-distribution 83.4 %, cross-category 38.1 %,
common sense 58.0 %, semantic instruction 53.8 % and unseen texture 61.4 % (average 58.9 %); single tasks differ.
Prior results are not evidence about the current episode.

Inspect all three views, the measured pose, the proposal table and the execution record. A gripper command does not
prove grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed subgoals if
later visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate in every
assessment.

## Contract

- Robot: one Franka Emika Panda with a parallel-jaw gripper in VLABench (dm_control on MuJoCo), its base at
  (0, -0.40, 0.78) on the near edge of the table, facing +y. One native step takes one absolute end-effector target:
  VLABench solves inverse kinematics for joint angles that put the end effector there (when the solver does not
  converge in its iteration budget, the arm still goes to the best solution it found), and the joint servos drive
  toward them for 0.1 s (100 physics substeps of 1 ms; 10 Hz, so `sim_time_s` is the native step count divided by
  10). Both fingers are driven fully open (4 cm each) or fully closed in the same step. You observe at decision
  boundaries only. Each task has its own step limit (`max_episode_steps`): 100 for the painting and card tasks, 300 for
  adding a condiment, 200 otherwise.
- World frame (meters): +x is the robot's right, +y points forward, away from the robot across the table, +z is up.
  The table top is at about z 0.76. At the episode start the gripper hangs about 0.5 m above the table near
  (0.00, -0.16, 1.25), pointing straight down with its fingers opening along x, fully open; `episode_start_eef` gives
  the measured values.
- Pose: the end effector's position is the point midway between the two fingers, about 2 cm back from the
  fingertips; its orientation is the hand's (unit quaternion wxyz): the hand's local z axis points from the wrist out
  through the fingers and its local y axis is the line the fingers open and close along. `rpy_deg` is the same
  orientation as extrinsic xyz Euler angles in degrees, the convention VLABench's own targets use.
- Views, each rendered 480×480 and shown as rendered, the images the released client sends OpenWAM:
  - `forward`: the scene camera on the far side of the table, facing the robot and looking down: the robot is in the
    upper part of the image, the image's left is the robot's right (+x) and its right the robot's left (-x). Three
    tasks replace it with their own camera, each upright with the image's right roughly toward the robot's right:
    add_condiment (a narrow view from behind the robot's right shoulder, looking toward the robot's left-front across
    the counter), select_book (from the robot's right, looking forward and left at the shelf, nearly level) and
    select_drink (a narrow view from just behind the robot's base at about table height, looking straight ahead into
    the fridge).
  - `wrist`: the camera on the hand, behind the fingers and offset to one side, looking at the point between the
    fingers; it moves and turns with the hand. At the start pose its image is oriented like `forward`'s: the image's
    top is toward the robot (-y) and its right is the robot's left (-x).
  - `right`: a fixed scene camera behind the robot and to its right, near (0.78, -0.86, 1.21), looking diagonally
    across the table toward the robot's left-front, upright.
- Gripper: the commanded opening, 1 = open and 0 = closed; VLABench opens both fingers fully at 0.5 or more and closes
  them fully below it. `gripper_measured_opening` is the measured separation of the two fingers, 1 = 8 cm.
- The proposal table: one line per row `[i, x, y, z, roll, pitch, yaw, opening]`, each row an absolute end-effector
  target in the world frame (OpenWAM's output as the released client converts it), not offsets chained from the
  measured pose and not predicted poses. `diagnostics` gives the net displacement, path length, largest step and the
  rows where the gripper command closes or opens. Diagnostics are plain arithmetic, not a safety, collision, intent or
  success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is the arm's distance and rotation from its episode-start pose, from measured poses only.
- Success: after every native step VLABench checks the task's conditions (`task_context` restates them) and ends the
  episode as soon as all of them hold. Its evaluator also records a progress score (the share of the conditions that
  hold at the end, counting a target that was ever grasped as one more) and an intention score (whether the end
  effector ever came within 10 cm of the target); neither decides success.
