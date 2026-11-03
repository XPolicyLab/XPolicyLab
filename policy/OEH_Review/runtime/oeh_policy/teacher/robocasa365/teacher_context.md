# Task and predecessor knowledge

The goal is one recorded RoboCasa365 rollout of the task named in the first packet, controlled by this persistent agent
and the released RoboCasa365 checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-RoboCasa365`), executed through OpenEmbodied
Harness. Derive the exact objective from the native instruction and visible observations, not a hard-coded task
script. This rollout does not train the student.

The control method is the published hybrid's, unchanged from the other benchmarks: fresh student inference,
outcome/intent review, optional short EEF correction, then fresh observation. The gate text is preserved verbatim.
The OpenWAM paper reports this checkpoint alone at 69.7% success on the 18 atomic target tasks, 32.1% on the 16
composite tasks with pretraining demonstrations and 8.9% on the 16 composite tasks without any, 50 episodes per task;
single tasks differ. Prior results are not evidence about the current episode.

Inspect all three views, the measured pose, the base's pose, the proposal table, its base motion and the execution
record. A gripper command does not prove grasp, and a target does not prove arrival. Keep uncertain results uncertain;
retract completed subgoals if later visual evidence contradicts them. The conversation is your only memory: keep
`task_progress` accurate in every assessment.

## Contract

- Robot: PandaOmron in RoboCasa (robosuite on MuJoCo): a Franka Emika Panda arm with a parallel-jaw gripper on the
  torso of an Omron mobile base; the torso lifts the arm and both agent views together, by up to 34 cm. One native step
  is 0.05 s (20 Hz); you observe at decision boundaries only. Each native row commands the arm as an operational-space
  pose delta in the base's frame (at most 5 cm and 0.5 rad per step), the gripper closed or open, the base's forward
  and sideways velocity (at most 1 m/s each) and turn rate (at most 1.5 rad/s), the torso's height change (at most
  5 cm per step), and a control mode, which only decides whether the arm's next goal starts from its measured pose (arm
  mode) or from its previous goal (base mode). In both modes the arm's goal is held relative to the base, so the base
  carries the arm; the arm lags its goals.
- World frame: RoboCasa's kitchen frame (meters, +z up); x and y are fixed by the kitchen's layout and do not turn with
  the robot. `mobile_base.position` is the arm's mounting point on the torso (it rises with the torso); `heading_deg` is
  the direction the base faces, in degrees from world +x toward +y (the robot's left is 90° counter-clockwise from
  it); `moved_since_start_m` and `turned_since_start_deg` are its travel and turn since the episode start. Read
  directions from poses, not from image directions.
- Pose: the gripper's grip point (between the fingertips) and the hand's orientation in the world frame (meters,
  quaternion wxyz): the base's pose combined with the arm's pose relative to the base. `rpy_deg` is the same orientation
  as extrinsic xyz Euler angles in degrees. The tasks' distance conditions are measured from this grip point.
- Views: `robot0_agentview_left` and `robot0_agentview_right` are fixed to the base above and behind the arm, one on
  each side, looking forward and down past the gripper (image left is the robot's left); they move, turn and rise with
  the base and the torso. `robot0_eye_in_hand` sits on the hand, looking along the gripper. All three are 256×256 and
  upright. OpenWAM receives the same three images (the left agent view resized to 320×256 as its main view, the other
  two at 160×128), the arm's pose relative to the base, the finger joints, the base's pose and the instruction; nothing
  else.
- The proposal table: one line per row `[i, x, y, z, roll, pitch, yaw, opening]`, OpenWAM's arm goals relative to the
  base, chained from the measured pose and placed in the world at the base's current pose (row i is where rows 0..i
  would put the gripper if it reached every goal and the base stood still), not predicted poses. `diagnostics` gives
  the net displacement, path length, largest step and the rows where the gripper command closes or opens. Diagnostics
  are plain arithmetic, not a safety, collision, intent or success verdict.
- `student_proposal.base_motion` is what the proposal's rows command the base to do if every step ran at its command:
  `forward_m` and `side_m` (travel along the base's own forward and left axes, summed without the turn), `turn_deg`
  (positive turns left), `torso_m` (the torso's height change) and `rows_in_base_mode`. All zero means the proposal
  keeps the base and the torso still.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `episode_start_eef` is the arm's episode-start posture relative to the base, placed where the base now stands;
  `home_offset` is the arm's distance and rotation from it, from measured poses only.
