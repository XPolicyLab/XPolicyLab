How the official evaluator scores an attempt (applies to every task):

- Checks run after every native action. Tiered tasks use transition mode: only the next tier is checked, at
  most one tier is gained per native action, and a reached tier is never lost by later motion.
- The 100 tier also needs the success check list to hold. If the last tier is reached but success does not
  hold, the score falls back to the previous tier. The episode ends as soon as success holds.
- A successful attempt scores 100 by itself: conditions listed only in the tiers (open grippers, objects at
  rest) do not apply to it. Tiers only decide the score of an attempt that does not succeed.
- "Gripper open >= t" is measured on the fingers' actual joint positions (1 = fully open), not on the
  command: a gripper still squeezing an object or blocked by it may not count as open. A tier that requires
  open grippers is only credited on an action where all grippers are open, so release, back away so the
  fingers can open fully, and keep them open before expecting the tier.
- A forbidden event in a task (lifting too high, moving an object that must stay, an extra press, an
  out-of-order release) ends the episode at once; tiers already earned are kept.
- "Robot back to origin" means every arm's end effector is within 0.15 m on each of X, Y and Z and within
  20 degrees of its pose at reset
  (its pose at the start of the episode). Reserve the actions to return there.
- The native action limit (`step_lim`) ends the attempt; unspent actions earn nothing.
