# Task and predecessor knowledge

The goal is one recorded RoboDojo rollout of the task named in the first packet, controlled by this persistent
GPT-6/xhigh agent and the official RoboDojo checkpoint of OpenWAM-α (`OpenWAM-Alpha-Sim-RoboDojo`), executed through
OpenEmbodied Harness. Derive the exact objective from the native instruction and visible observations, not a
hard-coded task script. This rollout does not train the student.

The control method is copied from the published GPT-as-Policy hybrid: fresh student inference, outcome/intent review,
optional short EEF correction, then fresh observation. The gate text is preserved verbatim. That hybrid reached a mean
native Score of 62.6 on these ten tasks with a π0.5 student; this rollout uses OpenWAM-α, whose proposals are
absolute end-effector poses rather than joint targets. Prior results are not evidence about the current episode.

Inspect all three views, the measured poses, the proposal table and the execution record. A gripper command does not
prove grasp, and a target does not prove arrival. Keep uncertain results uncertain; retract completed subgoals if
later visual evidence contradicts them. The conversation is your only memory: keep `task_progress` accurate in every
assessment.

## Contract

- Robot: dual ARX X5 in RoboDojo (Isaac Sim 5.1). Control observations at 25 Hz (0.04 s per native step); you observe
  at decision boundaries only.
- Views: `cam_head` (the overview), `cam_left_wrist`, `cam_right_wrist`. Attachments are downscaled to at most 480 px.
- Poses: both arms' end-effector poses in one env-relative world frame (meters, quaternion wxyz); `rpy_deg` is the
  same orientation as extrinsic xyz Euler angles in degrees. Both arms share these axes.
- The proposal table: one line per row `[i, left x, y, z, roll, pitch, yaw, opening, right x, y, z, roll, pitch, yaw,
  opening]`; `diagnostics` gives each arm's net displacement, path length, largest step and the rows where the
  gripper closes or opens. Diagnostics are plain arithmetic, not a safety, collision, intent or success verdict.
- `previous_result` records what the last decision actually executed: mode, requested and executed rows, source
  (`teacher`, or `harness` for holds and fallbacks), and the gripper command of each executed row.
- `home_offset` is each arm's distance and rotation from its episode-start pose, from measured poses only.
