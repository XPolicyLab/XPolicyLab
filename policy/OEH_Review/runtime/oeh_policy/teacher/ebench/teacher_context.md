# Task and predecessor knowledge

The goal is one recorded EBench rollout of the task named in the first packet, on GenManip's own evaluation server,
controlled by this persistent agent and the released EBench checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-EBench`),
executed through OpenEmbodied Harness. Derive the exact objective from the native instruction and visible
observations, not a hard-coded task script. This rollout does not train the student.

The control method is the published hybrid's, unchanged from the other benchmarks: fresh student inference,
outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved verbatim.
The OpenWAM paper reports this checkpoint alone on EBench's generalist split at 30.0% success on the table-top tasks,
67.5% on mobile pick-and-place and 44.3% on the long-horizon tasks, 49.4% overall; single tasks differ. Prior results
are not evidence about the current episode.

Inspect all three views, the measured poses, the base, the proposal table and the execution record. A gripper command
does not prove grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed
subgoals if later visual evidence contradicts them. The conversation is your only memory: keep `task_progress`
accurate in every assessment.

## Contract

- Robot: GenManip's lift2 in Isaac Sim: a mobile base (x, y and yaw), a lift held at a fixed height and two six-joint
  arms with two-finger grippers, each arm on its own mount. One native step sets both arms' joint targets from the
  row's two poses (inverse kinematics on GenManip's server), both grippers' finger targets and the base's next target
  pose, then advances the physics one step: 1/30 s on the mobile tasks, 1/60 s on the table-top tasks (`sim_time_s`
  counts steps at that rate). You observe at decision boundaries only. Each task has its own step limit
  (`max_episode_steps`, 600–5000).
- Arm frames (meters): each arm's poses, measured and commanded, are in that arm's own base frame, fixed to its mount:
  +x forward (the way the robot faces), +y to the robot's left, +z up, as GenManip's own scripted motions use them.
  The two frames have different origins, one per mount (GenManip does not publish the mounts' offsets), so compare
  the arms' numbers by direction, not as positions in one frame. Both frames move with the base.
- Base: `base.pose` is the mobile base's [x m, y m, yaw deg] and `base.start` its value at the episode start, in the
  axes the robot was placed in (at yaw 0 it faces +x, and +y is its left). A row's base displacement [dx m, dy m,
  dyaw deg] is in the same axes, clipped by GenManip to 0.015 m and 1° per step and added to the base's target pose.
- Poses: an arm's pose is its end-effector link, not the grasp point: the centre between the fingertips lies 0.135 m
  along the pose's local +x axis, the direction the gripper points. `rpy_deg` is the same orientation as extrinsic
  xyz Euler angles in degrees.
- Views: `cam_overlook` (1280×720, fixed to the base 1.5 m above it and 0.3 m ahead, looking forward and 75° down:
  the image top is farther ahead of the robot and image left is the robot's left; it moves and turns with the base),
  `cam_left_wrist` and `cam_right_wrist` (640×480, one on each arm's last link, looking along the gripper). These are
  the images the released bridge sends OpenWAM, upright as the cameras deliver them; attachments are downscaled to
  at most 480 px.
- Gripper: an opening of 1 = open and 0 = closed; a command sets both fingers of that arm to opening × 0.044 m.
  `gripper_measured_opening` is the mean measured finger position over 0.044 m.
- The proposal table: one line per row `[i, left x, y, z, roll, pitch, yaw, opening, right x, y, z, roll, pitch, yaw,
  opening, base dx, dy, dyaw]`: each arm's absolute target in its own frame and the base displacement as GenManip
  applies it (after its clip). `diagnostics` gives each arm's net displacement, path length, largest step and the
  rows where the gripper closes or opens, and the base's net displacement over all rows. Diagnostics are plain
  arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is each arm's distance and rotation from its episode-start pose in its own frame, from measured poses
  only; it ignores the base's motion.
