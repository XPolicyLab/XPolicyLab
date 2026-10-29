- Your first motion is start(strategy, kind), with the line of the policy's description
  the task matches. The policy was trained on these tasks, each from the start pose with
  its own instruction: hand it the task unless the instruction asks for a read it does
  not do.   B, your own tools only, only for these instructions: press buttons a number of
  times read from number cards (count the presses; the confirm button confirms one
  count, so press it once after each card, even when the instruction names it once at
  the end); complete an equation by the missing number or operator on a card; arrange
  digits into the largest number; pick one named object up by 10 cm; sort objects by
  category into the three baskets; stack the three blocks with different textures;
  stack blocks from bottom to top in a stated order; arrange nesting dolls from
  smallest to largest; insert a key into the keyhole; sweep with a dustpan; declare a
  kong; place objects into the basket in the same order just observed. Anything else
  is A and runs the Simate base, including a game, a conveyor, a tower, bowls, folding,
  packing, pouring, screws, a laptop, and a picture of a target. The instruction
  decides: a B on anything else is run as A, and an A on one of the reads above is run
  as B.
  A hands the task over and the policy keeps it: after start returns, do not move the
  arms. Calls after that handover are not spent. Do not watch_scene before a handover:
  it holds the arms still for the time it watches, and that time is the policy's (a 12 s
  watch took half of make_kong's 600 steps).
- If you chose B and your own tools fail at the task early (two attempts at the same
  step failed, or the objects cannot be reached) while the task is one the policy was
  trained on, hand it over: home() if the arms are not at the start pose, then
  start("A" or "C", kind). It learned whole tasks from the start: hand over while the
  scene still looks like the start of the task, not after you have moved many objects.
- After A or C the policy runs on to the end of the episode. Do not grasp what it
  already placed. A fresh run (reset=True) starts only from the start pose: home()
  first, and only when the scene still looks like the start of the task.
- Use your own tools for the reads listed above. A picture of a target, including one
  on a conveyor, stays with the policy.
