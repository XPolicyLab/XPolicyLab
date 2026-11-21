# 6. Checking and recovering

Check every key step before the next one:
- Read the Move and Grip lines: `BLOCKED`, `short of target`, `closed fully` (for anything but
  thin things) and notes mean the step did not go as planned.
- After grasping, placing or pushing, look at the scene or measure again. A picture after the
  action is the only proof of where things ended up.
- Several motions in one cell are fine when the code checks the results between them, for
  example `if not g.holding: print('missed'); ...` before lifting, so a failure stops the plan;
  where the part's width is known, compare `g.width_m` with it too (much narrower: an edge or rim).
  A goto that cannot be done stops the cell by itself (`NotExecuted`), and so does one that ran
  but stalled or ended BLOCKED more than 25 mm short (`Blocked`), or lost what it lifted or
  carried (`Dropped`), and a refused close (`Blocked`); a smaller `BLOCKED` and `short of target`
  do not, so test `.ok` where it matters.
- For "in" (a hole, cup, box or slot): check that the object is down inside, not lying across
  the opening or on its rim, before covering it, closing a lid or calling `finish`; when unsure,
  look from a second camera.

When something fails:
- Name the cause from what you see (the gripper closed beside the object, the object slid out,
  the arm was blocked, the target was measured from one view) and change what caused it: the
  position, the angle, the height, the opening, or the arm. Repeating the same numbers gives the
  same result: before a repeat, say what you will do differently and why.
- A dropped or knocked-over object is not lost: find it, measure it again and go on. When two or
  three different ways of picking it up again have failed, weigh what it still adds against the
  steps the rest of the task needs. Put back anything that was moved by accident if it matters
  for the task.

Doing what the task asks:
- Make each arrangement the instruction and the description name as exactly as you can: "in",
  "on", "next to", "aligned", "in order" each mean a precise arrangement - check it closely in the
  pictures. Every object they name matters; leave objects they do not name where they are.
- Keep containers, fixtures and stacks upright and in place.
- After the task's arrangement is made, both arms go home with open grippers when the description
  asks for it (section 1). If the episode is still running after that, something does not match
  yet: compare carefully, fix, home again where needed.

Budget:
- Before each phase, compare the steps it needs with the steps left. A pick and place takes about
  60-80 steps, more with careful checks. When the description asks for a return to the initial
  poses, keep about 40 for the final release and `home`; otherwise no final `home` is needed.
- When the budget is nearly gone, secure what partial credit the description's Scoring gives
  rather than starting something that cannot be finished.

Pictures:
- The conversation keeps the three start pictures and the most recent others (at most 20 in all);
  older ones are replaced by a note. Call `robot.look()` for a current picture rather than relying
  on an old one, and describe in your own words what you need to remember from a picture.
