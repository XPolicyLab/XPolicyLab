- You keep this episode. Your first motion is start("B", kind). Do not use A or C
  to hand the episode away: that does not run the policy and does not lock the arms.
- The learned policy is one call inside a program, like a motion. From the start
  pose, run_policy("wall-oss", max_seconds=30). After you have moved, pass
  from_anywhere=True. When the slice ends, the trace comes back and the program
  continues: look, read the grippers, and either move with your own tools or call
  run_policy again. A call without max_seconds runs until the episode ends, so
  pass max_seconds.
- Check the scene after a grasp or a placement before the next motion. A failed
  check is another try in the same program.
