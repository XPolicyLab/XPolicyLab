Learned skill (a visuomotor policy trained on whole RoboDojo tasks):
  {"type": "policy", "instruction": "..." (optional),
   "arms": ["left_arm", "right_arm"], "seconds": s}
      Runs the learned policy for at most s seconds (default 8, at most 30). It was trained on
      demonstrations that start at the start pose and finish the task, with the task's
      own instruction, which it gets when you give none. It does best in that setting,
      with both arms; calling it again with the same instruction and arms continues
      where it was, so a long task takes several calls in a row. Other instructions,
      one arm, or a start in the middle of the task are outside what it learned.
