<!-- if: vla -->
  Learned skills (visuomotor policies trained on RoboDojo demonstrations):
  list_policies() -> [{"name", "description", "arms"}]
      The description says which kinds of manipulation the policy does reliably, partly
      or not at all, and the signs that it is failing. Read it before your first motion.
  start(strategy, kind, policy=None) -> Trace or None
      Your first motion, once: the strategy for this task and the line of the policy's
      description the task matches (kind, in your words). "A": hand it the whole task and
      supervise; "C": the policy first, then finish what it leaves; "B": your own tools
      only. A and C run the policy at once, as run_policy(policy) from the untouched start
      pose (default: the first policy), and return its trace; B returns None. Until start,
      every motion ends at once with SKILL_ERROR ("call start(strategy, kind) first");
      looking runs (get_observation, locate, segment_text, point_to take no episode
      time; watch_scene takes the time it watches).
      Called again with the same strategy it does nothing. After B, start("A" or "C",
      kind) hands the task over after all (from the start pose: home() first); other
      changes are refused.
      After its first run you have a few model calls (the turn says how many); then it
      runs on to the end of the episode by itself.
  run_policy(name, instruction=None, arms=("left_arm", "right_arm"), max_seconds=None,
             region=None, until=None, reset=False, stall_seconds=5.0) -> Trace
      Hands control to the policy. It was trained on whole tasks: demonstrations that
      start at the start pose and finish the task, labelled with the task's own
      instruction. It does best in that setting: from the untouched start pose, with no
      instruction (the task's own is used) and both arms, left to run. A fresh run (the
      first, another instruction or arms, or reset=True) starts only from the start
      pose: elsewhere it ends at once with SKILL_ERROR, detail reason "not at the start
      pose" (home() first; it keeps the grippers). Without max_seconds it runs until the
      episode ends, until(obs) holds, the arms stop moving (once they have moved, no
      joint of its arms moves 0.02 rad for stall_seconds; arms still from the start are
      waiting), or an event stops it. Stopping or using up max_seconds is a normal end:
      the trace ends with COMPLETED and trace.event.detail has "stopped":
      "stalled" or "max_seconds"; your program goes on from there, so
      check the scene in it (PREDICATE_SATISFIED: until held).
<!-- if: measured_gripper -->
      detail "drops" lists jaws that closed fully after holding something steadily.
<!-- end -->
      Called again with the same instruction and arms it continues where it was, with
      the rest of the action chunk it was executing (unless the arms moved in between),
      so slices of max_seconds act like one long run. Other instructions, one arm, or a
      start in the middle of the task are outside what it learned: expect much less
      there. Guards, until, region and the shield apply at every tick.
  home(arms=("left_arm", "right_arm"), speed=2.5) -> [Trace, Trace]
      Moves the arms back to their start-of-episode joints, both at once (grippers as
      they are); about 20-40 ticks.
<!-- end -->
<!-- if: wam -->
  list_world_models() -> [{"name", "description"}]
  predict_future(name, instruction, horizon=None) -> Prediction
      Asks a world-action model what would happen if the robot followed the
      instruction from now; nothing moves. .frames are predicted camera images
      (show() them to look), .actions the predicted end-effector targets per 25 Hz
      step ({"left_ee_pose", "left_ee_joint_state", "right_ee_pose",
      "right_ee_joint_state"}, link6 poses in the env frame). Takes about a minute;
      use it to compare candidate instructions before committing to one.
<!-- end -->
