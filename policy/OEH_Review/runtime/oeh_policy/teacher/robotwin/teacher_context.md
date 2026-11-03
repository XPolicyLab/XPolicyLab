# Task and predecessor knowledge

The goal is one recorded RoboTwin 2.0 rollout of the task named in the first packet, controlled by this persistent
agent and the released RoboTwin checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-RoboTwin-Full`), executed through
OpenEmbodied Harness. Derive the exact objective from the native instruction and visible observations, not a
hard-coded task script. This rollout does not train the student.

The control method is the published hybrid's, unchanged from the other benchmarks: fresh student inference,
outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved verbatim.
The OpenWAM paper reports this checkpoint alone at 93.74% success in clean scenes and 93.46% in randomized scenes,
averaged over these 50 tasks; single tasks differ. Prior results are not evidence about the current episode.

Inspect all three views, the measured poses, the proposal table and the execution record. A gripper command does not
prove grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed subgoals if
later visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate in every
assessment.

## Contract

- Robot: aloha-agilex in RoboTwin 2.0 (SAPIEN): two 6-DoF arms with parallel grippers, facing the table from its near
  edge. One native step executes one end-effector target per arm through RoboTwin's motion planner; nominally 15/250 s
  (16.7 Hz), so `sim_time_s` is nominal too. You observe at decision boundaries only. Each task has its own step limit
  (`max_episode_steps`, 400–1700).
- World frame, shared by both arms (meters): +x is the robot's right, +y points away from the robot across the table,
  +z is up. The table top is at z ≈ 0.74; randomized scenes lower it by up to 3 cm and add clutter, random
  backgrounds and random lighting. The left arm is mounted near (-0.30, -0.42, 0.78) and works at negative x, the
  right arm near (0.31, -0.42, 0.78) at positive x.
- Poses: an arm's reported and commanded pose is its wrist (the last joint), not the grasp point: the grasp centre
  between the fingers lies 0.12 m along the pose's local x axis, the direction the gripper points. At the episode
  start both arms are raised about 0.2 m above the table near x = -0.30 (left) and +0.31 (right), y = -0.31,
  z = 0.94, grippers open and pointing forward (+y) with their local z axis down; `episode_start_eef` gives the
  measured values. `rpy_deg` is the same orientation as extrinsic xyz Euler angles in degrees.
- Views: `cam_head` (the overview, fixed above the robot looking forward and down: image left is the robot's left and
  the image top is farther from the robot), `cam_left_wrist` and `cam_right_wrist` (one on each gripper), 320×240.
- Gripper: the commanded opening, 1 = open and 0 = closed; RoboTwin reads above 0.8 as open and below 0.2 as closed.
- The proposal table: one line per row `[i, left x, y, z, roll, pitch, yaw, opening, right x, y, z, roll, pitch, yaw,
  opening]`; `diagnostics` gives each arm's net displacement, path length, largest step and the rows where the
  gripper closes or opens. Diagnostics are plain arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is each arm's distance and rotation from its episode-start pose, from measured poses only.
