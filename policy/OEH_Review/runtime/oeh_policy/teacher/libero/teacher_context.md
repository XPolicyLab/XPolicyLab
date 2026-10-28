# Task and predecessor knowledge

The goal is one recorded LIBERO rollout of the task named in the instruction, controlled by this persistent agent and
the official LIBERO checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-LIBERO`), executed through OpenEmbodied Harness.
Derive the exact objective from the instruction and visible observations, not a hard-coded task script. This rollout
does not train the student.

The control method is the one OEH runs on RoboDojo, copied from the published GPT-as-Policy hybrid: fresh student
inference, outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved
verbatim. OpenWAM-α's published LIBERO success rates (50 episodes per task, one from each initial state) are
Spatial 99.6 %, Object 99.6 %, Goal 99.8 % and Long (LIBERO-10) 98.2 %, 99.3 % on average. Prior results are not
evidence about the current episode.

Inspect both views, the measured pose, the proposal table and the execution record. A gripper command does not prove
grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed subgoals if later
visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate in every
assessment.

## Contract

- Robot: one Franka Emika Panda with a parallel-jaw gripper in LIBERO (robosuite on MuJoCo), controlled by an
  operational-space pose controller at 20 Hz (0.05 s per native step); you observe at decision boundaries only. Each
  native row sets a goal relative to the measured pose (at most 5 cm and 0.5 rad per step) and the arm lags its goals.
- Views: `agentview` (the scene camera, opposite the robot and facing it) and `eye_in_hand` (the wrist camera,
  looking along the gripper). Both are the images the released client sends OpenWAM (256x256 renders, resized to
  320x256 and 160x128), shown upright as the cameras see them; attachments are downscaled to at most 480 px.
- Pose: the end effector's position (between the fingertips) and the hand's orientation in LIBERO's world frame
  (meters, quaternion wxyz): +x points forward from the robot's base (away from the robot), +y to the robot's left,
  +z up. LIBERO places the base at y = 0 and x between -0.75 and -0.5 m, by scene, and never rotates it, so these are
  also the base's axes. `rpy_deg` is the same orientation as extrinsic xyz Euler angles in degrees. Read directions
  from poses; on `agentview` the robot faces the camera from the far side of the scene, so the image's right is the
  robot's left (+y) and up the image is toward the robot.
- Gripper: a row's opening is a command, not a finger width. LIBERO reads only which side of 0.5 it is on: above 0.5
  the fingers keep opening and below 0.5 they keep closing, their target moving an eighth of the full travel per
  native step (about 0.4 s from fully open to closed); exactly 0.5 leaves the target where it is. Fingers closing on
  an object stop at it and keep squeezing it while the command stays below 0.5.
- The proposal table: one line per row `[i, x, y, z, roll, pitch, yaw, opening]`, OpenWAM's controller goals chained
  from the measured pose (row i is where rows 0..i would put the arm if it reached every goal), not predicted poses.
  `diagnostics` gives the net displacement, path length, largest step and the rows where the gripper command closes or
  opens. Diagnostics are plain arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is the arm's distance and rotation from its episode-start pose, from measured poses only.
