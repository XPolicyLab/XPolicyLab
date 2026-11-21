# Task and predecessor knowledge

The goal is one recorded RoboCasa GR1 tabletop rollout of the task named in the first packet, controlled by this
persistent agent and the released GR1 checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-RoboCasa-GR1`), executed through
OpenEmbodied Harness. Derive the exact objective from the instruction and visible observations, not a hard-coded task
script. This rollout does not train the student.

The control method is the published hybrid's, unchanged from the other benchmarks except that you steer the arms only:
fresh student inference, outcome/intent review, optional short EEF correction, then fresh observation. The gate text is
preserved verbatim. The OpenWAM paper reports OpenWAM-α at 60.5% success averaged over these 24 tasks; single tasks
differ. Prior results are not evidence about the current episode.

Inspect the view, the measured poses and hand closures, the proposal table and the execution record. A hand closure
does not prove a grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed
subgoals if later visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate
in every assessment.

## Contract

- Robot: the GR1 humanoid in RoboCasa GR1 tabletop (robosuite on MuJoCo), standing at a kitchen counter; its legs and
  head are not commanded. It has two 7-joint arms, each ending in a Fourier hand with six finger commands, and a waist
  with yaw, pitch and roll joints. One native step is 0.05 s (20 Hz): the released client solves the row's two hand
  poses into absolute arm-joint targets and sends them, with the row's finger and waist targets, to joint position
  controllers. You observe at decision boundaries only. The step limit is 720 steps (36 s).
- Instruction: `instruction` is the text OpenWAM receives, `unlocked_waist: ` (the dataset's tag for this robot with its
  waist free to move) followed by the task's instruction.
- Frame: every pose is in the robot-base frame, fixed at the robot's pelvis, which never moves: +x points forward from
  the robot toward the counter, +y to the robot's left, +z up; meters. With the waist straight the shoulders sit about
  0.36 m above its origin and 0.13 m to either side (the left one at +y), and the head camera about 0.23 m forward and
  0.57 m up. A waist motion moves the shoulders and the head camera, never this frame.
- Poses: a hand's reported and commanded pose is its wrist site, where the hand attaches to the forearm, not a fingertip
  or the palm centre. In the pose's own axes the finger roots lie about 0.10 m along local -y, the fingers extend
  beyond them, and they curl toward local +z, the palm side, where the thumb sits. Quaternions are wxyz; `rpy_deg` is
  the same orientation as extrinsic xyz Euler angles in degrees. `episode_start_eef` gives the measured start poses:
  the arms start bent about 90 degrees at the elbows with both hands in front of the torso.
- View: `ego_view`, the head camera, looking forward and about 56 degrees down: image left is the robot's left and the
  image top is farther forward. It is rendered at 1280×800, padded with black bands above and below to a square and
  scaled to 256×256, upright: the image OpenWAM receives (stretched to 320×256 for it). The camera turns with the waist.
- Hands: each hand takes six joint targets in radians, 0 straight: the little, ring, middle and index fingers' flexion
  (limit 1.74), the thumb's bend (limit 1.22) and the thumb's rotation. `hand_closure_command` and
  `hand_closure_measured` are the closures defined above (0 open, 1 fully flexed), read from the last command and
  from the measured joints.
- Waist: `waist_deg` gives the last commanded and the measured yaw, pitch and roll in degrees: positive yaw turns the
  torso to the robot's left, positive pitch leans it forward, positive roll tilts it toward the robot's right.
- The proposal table: one line per row `[i, left x, y, z, roll, pitch, yaw, hand closure, right x, y, z, roll, pitch,
  yaw, hand closure, waist yaw, pitch, roll]`, OpenWAM's absolute targets in the robot-base frame, angles in degrees.
  `diagnostics` gives each arm's net displacement from its measured pose, path length, largest step, the range of its
  hand closure and the rows where the hand closes (closure 0.5 or more) or opens again, and the waist's net change.
  Diagnostics are plain arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the hand closures each executed row commanded.
- `home_offset` is each arm's distance and rotation from its episode-start pose, from measured poses only.
