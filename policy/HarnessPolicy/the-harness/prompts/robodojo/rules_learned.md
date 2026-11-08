- Your first motion is start(strategy, kind), with the line of the policy's description
  the task matches. The policy was trained on these tasks, each from the start pose with
  its own instruction: hand it the task unless the task needs what it cannot do.
  B, your own tools only, for a read the policy does not do, and only when the
  objects stand still: press buttons a number of times read from number cards (count
  the presses; the confirm button confirms one count, so press it once after each
  card's presses and before the next card's, even when the instruction names it once
  at the end); complete an equation by the missing number or operator on a card;
  arrange digits into the largest number.
  Named 10 cm pickup goes to the policy. Everything else goes to the policy, also
  when the instruction names colours or an
  order ("uncover them in the order red, green, blue", "stack blue, yellow, orange",
  "cars into the left basket"), a size, a precise fit, a picture of a target, and when
  objects are matched to each other by how they look. Objects moving past (a conveyor)
  go to the policy whatever identifies the target.
  The runtime enforces this split on the instruction text, not on a task name. A B on
  anything else is run as A, and an A on one of those reads is run as B.
  To hand it over: A, hand over and supervise, when the description says it usually
  finishes that kind by itself; C, the policy first, then finish what it leaves, for
  every other kind. A condition in a task the policy learned whole (an orientation, a
  place, an order, "by size", "by category"), a precise fit, moving objects, or
  remembering an object seen earlier in the episode (it keeps a memory of past frames)
  is part of what it learned: it does not make the task B.
  Decide from the task and the first images, and hand over in your first program: the
  policy then runs the task uninterrupted until it ends or stops. Do not watch_scene
  before a handover: it holds the arms still for the time it watches, and that time is
  the policy's (a 12 s watch took half of make_kong's 600 steps).
- If you chose B and your own tools fail at the task early (two attempts at the same
  step failed, or the objects cannot be reached) while the task is one the policy was
  trained on, hand it over: home() if the arms are not at the start pose, then
  start("A" or "C", kind). It learned whole tasks from the start: hand over while the
  scene still looks like the start of the task, not after you have moved many objects.
- After A or C the policy runs on to the end of the episode, and later motions in the
  program are refused. Do not grasp what it already placed. A fresh run (reset=True)
  starts only from the start pose: home() first, and only when the scene still looks
  like the start of the task.
- Use your own tools for the reads above. A picture of a target, including one on a
  conveyor, stays with the policy.
