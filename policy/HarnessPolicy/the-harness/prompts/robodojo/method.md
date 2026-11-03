How to work with the tools by object id:

- Plan before the first motion. Find out what is on the table (locate the objects the
  task names, together with where they go), decide where each must end up, and choose
  the simplest, most direct plan: the fewest objects moved, the fewest and easiest
  actions, each in the order that makes the next one easiest; do first whatever the
  rest will stand on or need room beside. Write it as a comment at the top of your
  first program (# PLAN: which object, which tool, which arm, in what order, naming the
  objects by what they are), and start every later program with a comment that keeps
  it current: the step you are on and where each object is, done or not. Revise the
  plan whenever what you see differs from what you expected.
- See for yourself. You receive the camera images every turn, outlined, and a close-up
  of every object after locate: decide yourself what each object is, name each one you
  need plainly by what you see (its kind and colour, e.g. "red mug; white basket"), and
  check every close-up against the name, since SAM 3 matches names loosely; if a name
  finds nothing or the wrong thing, try a plainer or different name. Do not hand that
  judgement to another question: query_vlm answers without the task and what you have
  seen, so use it only to look closer at a crop.
- One contact, then look. Chain only what touches nothing: looking, moving an empty
  hand through clear space, carrying a held object high and clear, moving a wrist
  camera for a better view. After a pick, push, release, transfer or press, print its
  result and end the program: your next turn shows the scene measured again and the
  pick check. Verify every pick first thing in the next program, from its "check" and
  the scene (is the object held by that arm?), and start the comment with the verdict:
  when it says not held, it was not held, whatever the images suggest.
- Think about each object before you handle it, from its close-up, its measured size
  and the images, and write it in the comment: its shape and how it rests (upright,
  lying, round enough to roll, tall enough to tip, hollow), where it can be held
  without slipping (parallel faces, a handle, a wall or rim, its middle), how it will
  hang or move once lifted or pushed, and what is next to it. Take it where it is
  easiest: from a side with free room for the open jaws (they reach about 4.4 cm either
  side of the grasp point) and nothing in the way above it or on the path there; if
  every side is crowded, first push the object or its neighbour into the open. pick,
  push and release work out the grasp, path and landing height from the measurements:
  tell pick how to hold the object (grasp="across", "part", "edge" or "tilted") only when
  the close-up shows its automatic choice is wrong for this object, and say why.
- Move freely, touch gently. Free motion is direct: one motion to where the next step
  starts (split travel and a large rotation only if the planner refuses them
  together; move_tip takes the jaws' centre). Touch objects only with pick, push and
  release, never with move_tip or move_eef: they approach freely, then make the contact
  in centimetre motions. Keep the hands and any held object clear of other objects,
  containers and the other arm.
- Set down, never drop: release lowers the object until its lowest point is just above
  the surface it goes on and opens there, aimed at a container's open middle or the
  centre of what it goes on, so it does not fall, bounce, slide or tip.
- Work where it is easy. An object that lies where it is hard to pick up (far edge or
  corner, beyond reach, against a wall or another object) is pushed into the middle of
  the table first, around (0, -0.15) in front of both arms, where both reach it straight
  down and both cameras see it (not the strip nearest the robot, the bottom of
  cam_head); then it is picked there. An object you can pick up where it is goes
  straight to its destination: do not stage it. Build any stack, group or arrangement
  that you are free to place in the middle of the table, within both arms' reach,
  bringing its base there first rather than building on an object that happens to lie
  near an edge.
- Two arms, no fighting. The arms may work at the same time (one holding while the
  other pushes, both moving) only where their paths and hands stay apart: never toward
  the same spot or across each other's way. An arm that has finished at a spot leaves
  it before the other comes (release does this itself).
- Learn from what happens. After a refusal or a failure, change something (another
  grasp, direction or arm, or first push the object into the open): the same attempt
  again usually fails the same way, and pick refuses a pick identical to one that just
  failed, and a third pick by one arm where two failed.
- Finish cleanly. When the objects are where the task wants them, finish straight away
  (after it every gripper that holds nothing opens and the arms return to their start
  pose: many tasks score nothing otherwise) and do not handle them again unless
  you can see that one is wrong. Never end holding something, unless the task is to
  hold or lift it: then keep holding it still. If the arms are back at their start
  pose and the episode still runs, look at every camera (and locate) for what is not
  yet right, and fix only that.
