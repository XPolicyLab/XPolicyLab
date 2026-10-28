# Task and predecessor knowledge

The goal is one recorded LIBERO-Pro rollout of the task named in the instruction, controlled by this persistent agent
and the official LIBERO checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-LIBERO`; LIBERO-Pro evaluates policies trained on
LIBERO's original tasks and has no checkpoint of its own), executed through OpenEmbodied Harness. Derive the exact
objective from the goal and the visible observations, not a hard-coded task script. This rollout does not train the
student.

The scene is perturbed in one way, fixed by the episode's suite: the object variant gives the task's objects another
appearance (colour, size or shape; one kind of object can stand in for another), and the position variant exchanges
the starting places of some of the task's objects with those of other objects. The instruction keeps the original
task's wording in every variant, so it may not match the images; go by the goal (`task_context`) and by what you see.

The control method is the one OEH runs on RoboDojo, copied from the published GPT-as-Policy hybrid: fresh student
inference, outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved
verbatim. OpenWAM-α has no published LIBERO-Pro results. On LIBERO-Pro's leaderboard, policies trained on LIBERO
(OpenVLA, Pi0, Pi0.5 and others; its authors report the first three at 0.82-0.99 on the original tasks) keep 0.46-0.98
success under the object variant and score at most 0.06 under the position variant (Pi0.5 0.08-0.38). Prior results
are not evidence about the current episode.

Inspect both views, the measured pose, the proposal table and the execution record. A gripper command does not prove
grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed subgoals if later
visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate in every
assessment.

## Contract

- Robot: one Franka Emika Panda with a parallel-jaw gripper in LIBERO (robosuite on MuJoCo), controlled by an
  operational-space pose controller at 20 Hz (0.05 s per native step); you observe at decision boundaries only. Each
  native row sets a goal relative to the measured pose (at most 5 cm and 0.5 rad per step) and the arm lags its goals.
- Views: `agentview` (the scene camera) and `eye_in_hand` (the wrist camera, looking along the gripper). Both are the
  images the released client sends OpenWAM, shown upright; attachments are downscaled to at most 480 px.
- Pose: the end effector's position (between the fingertips) and the hand's orientation in LIBERO's world frame
  (meters, quaternion wxyz): +x points forward from the robot's base (away from the robot), +y to the robot's left,
  +z up. LIBERO places the base per scene and never rotates it, so these are also the base's axes. `rpy_deg` is the
  same orientation as extrinsic xyz Euler angles in degrees. Read directions from poses, not from image directions.
- The proposal table: one line per row `[i, x, y, z, roll, pitch, yaw, opening]`, OpenWAM's controller goals chained
  from the measured pose (row i is where rows 0..i would put the arm if it reached every goal), not predicted poses.
  `diagnostics` gives the net displacement, path length, largest step and the rows where the gripper command closes or
  opens. Diagnostics are plain arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is the arm's distance and rotation from its episode-start pose, from measured poses only.
