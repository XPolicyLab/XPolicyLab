# 5. Moving and handling objects

How motions run:
- `goto` moves the grasp point in a straight line when it can (`path='line'` insists on it).
  Otherwise it swings the joints, but only when the grasp point stays within about 3 cm of that
  line; a wider curve can sweep through objects, so that move is not executed and the result says
  how far off it would go: go via a waypoint (each leg a straight line), or pass `path='swing'`
  to accept the curve - only where nothing stands near it: a swing beside an object shoves it.
  Near objects, use straight lines; to cross the table, lift first.
- A pose that cannot be reached is not executed: the result says why and what is reachable
  instead (another pitch, another height, the other arm). Change the plan rather than resending
  the same numbers. A pose within 2 cm and 8 degrees of reach is accepted with a note;
  a height up to 5 mm too low for the fingertips to clear the table is raised to the lowest that clears it.
- `BLOCKED` means the arm is pressing on something - an object, the table or the other arm. Look
  before going on: something may have been pushed or knocked over. A move that stalls, or ends
  BLOCKED far short, raises `Blocked`, which skips the rest of the cell. Where touching is the
  point - a press, a push, an insertion, a lever - pass `contact=True`: the move runs to its end
  and raises nothing, so check its `.blocked` and `.at` yourself. A pick-up's descent is not one:
  stopped on top of the object, it refuses the close that follows once.
- Only one arm moves at a time; the other holds its pose, and a closed gripper keeps squeezing.
- Rough costs: a 10 cm straight move 5-7 steps (long moves speed up in their middle), from the
  start pose to above an object 10-15, across the table 11-15, a gripper change 6-8, `home`
  15-20. Every Move, Grip and Homing prints what it used and what is left.

The gripper:
- An opening g (0..1) leaves a gap of about 10.8·g - 1.9 cm between the pads: 1 → 8.9 cm,
  0.8 → 6.7 cm, 0.6 → 4.6 cm, 0.5 → 3.5 cm. The pads touch at about 0.18.
- It is position-controlled: `'close'` squeezes whatever is between the pads and keeps squeezing
  while the arm moves. An opening wider than the object does not hold it. Carry things closed.
- After closing, `holding` and the width say what is in the fingers: holding with a width near the
  object's size is a grasp; `closed fully` is a miss - except on cloth, paper and other thin
  things, which the fingers close on completely: look at a wrist picture to tell. `holding` None
  means the fingers stayed about fully open: they are blocked (pressed on the table or against
  something) - look before lifting. Right after a BLOCKED descent it usually means the fingertips
  rest on top of something as wide as the open gripper or wider: the same approach fails again -
  grip lower from the side, by a narrower part, or another way. When what it held slips out
  while it is lifted or carried, the motion stops and raises `Dropped` (`home` only reports it):
  look where it went, and grip it another way the next time - a grasp that slipped once slips
  again. Carry loads that swing or slip (tall, top-heavy, hanging, loosely held) with
  `gentle=True`: a few steps more, no fast middle.

Picking something up from above:
1. Go 8-10 cm above it pointing down with `yaw=thing.grip_yaw`, opening the gripper on the way
   (`grip=`, a gap at least 2 cm wider than the object - cheaper than a separate gripper call),
   and measure it again from there.
2. Descend in a straight line to the grasp height (from the new measurement): 1.5-2 cm below its
   top, but no lower than 1.5 cm above the surface it stands on. Lower is better for a firm grip
   as long as the fingers stay off the table.
3. Close, check the Grip line, lift straight up 8-10 cm, and confirm with a picture or
   `robot.measure(thing)` that the object came up with the gripper.

Other grasps:
- Tall or thin things (bottles, standing tools) are steadier gripped from the side, if they fit
  between the open fingers (about 8.9 cm) at that height: level or tilted pitch, grasp point at
  half to two thirds of the height, approaching along the pointing direction from about 10 cm
  away - or, when other objects stand near that approach, come down onto it: the hand level and
  pointing at it, first 3-5 cm higher than its top, then a straight descent that brings one open
  finger down each side of it.
- Long things: grip across the long side (`grip_yaw` gives that) at the balance point - the middle
  of an even rod, nearer the heavy end of a tool with a head, blade or metal part - on a part that
  fills the fingers. Held by a thin end far from its weight, it turns and slips out; after a slip,
  grip nearer the heavy end.
- Hollow things (cups, bowls) can be gripped by the wall when there is room inside: one finger
  inside, one outside, the grasp point about 1.5-2 cm below the rim (a shallower pinch slips).
  A wall-held object pivots and hangs below the grasp point: carry it beside or behind the
  direction of travel, not ahead, and grip the body across instead when it fits the opening and
  how it sits after release matters.

Putting down:
- Carry high enough to clear everything on the way (lift, move across, then descend), and work
  the height out: the highest top under the path (box walls, standing or placed objects) plus how
  far the load hangs below the grasp point, plus 2-3 cm - a fixed 8-12 cm lift is often less. A
  light object or an empty box that the load brushes slides or tips with no `BLOCKED`: look after
  carrying over or into something.
- Lower in a straight line until the object's bottom is about 1 cm above the surface it goes
  on, open, then lift straight up before moving sideways. Dropping from higher makes things
  bounce, roll or tip; pressing it down drags it out of place.
- A put-down that ends `BLOCKED` short has met something (test `.blocked` in code before the
  open). From `.at`, work out where the object's bottom is: within about 1 cm of the surface it
  goes on, it has landed - open. Higher (on a rim, an edge, a peg), keep the grip - opening there
  drops it - lift 1-2 cm, shift a centimetre or two and lower again.
- Where the task leaves the spot to you, put things down touching nothing and about 5 cm from
  anything still to be picked up: the open fingers reach about 4.5 cm out from the grasp point
  and shove what is nearer.
- Stacking: aim at the lower object's centre - where you released it, if you placed it yourself
  (measuring a stack again often fails: the outlines merge). The object being placed hangs in
  the gripper where it was grasped: if it was grasped off-centre, shift the target by the same
  amount.
- Re-tilting a held object by 20 degrees or more shifts it in the fingers: when the place needs a
  tilt, grasp at that tilt.
- A held object turns with the hand: released, it faces as it did when grasped, turned by the
  release yaw minus the grasp yaw. When how it ends up facing matters (writing, a symbol, an arrow,
  a handle), choose the grasp yaw from the heading you need before picking it up; after the
  release, check a zoomed picture before doubting that choice.
- Thin flat things (tiles, cards, discs) flip easily: carry them with `gentle=True`, release them
  about 1 cm above the surface, and do not push them.
- Into an opening narrower than the object's long side (a cup, hole, slot or holder) it goes in
  end-first: tilt the hand so that the long side points down before the last descent, and lower
  it until its lower end is 2-4 cm inside (less where the bottom is nearer; do not press) before
  opening - let go at the rim and it often tips out. Released level, it bridges the opening.
- Standing up a long thing gripped lying flat: it is upright when the pitch is 90 less than at the
  grasp (level after a grasp from straight above, -30 after one at 60), and the end that lay toward
  the grasp's yaw heading (yaw 0: away from the robot) comes up - with roll 180 the other end.
  When one end must be on top, choose the grasp for it, and check a wrist picture before lowering
  it in.

Pushing, pressing, turning:
- Push with the gripper closed, its lowest point 1-2 cm above the table, in a straight line, a
  few centimetres at a time, with `contact=True`; check after each push. Pushing high tips tall,
  narrow objects over; squat or wide ones slide.
- An object that must not be lifted can often still be gripped low and slid, turning the wrist on
  the way, every held move level: one held slide moves and turns it where pushes take many tries.
- Press (buttons, keys): close the gripper, point down above the centre, go straight down with
  `contact=True` until the fingertips are about 1 cm below where they touch the top (deeper can
  pry the closed fingers apart and fling the arm), then straight back up at least 3 cm.
- To turn a held object, change yaw (or roll) with the arm high above everything. The wrist turns
  only about 178 degrees either way from its middle; `dry_run=True` shows whether a turn is
  possible before you commit.
- Pouring from a held container: its opening is the distance from the grasp point to the opening
  along the container's axis, and moves as `roll` tilts it. Put the opening 3-5 cm above the
  target's rim, with the lip it pours over a little short of the target's centre: tilted past
  level, that lip is a few centimetres beyond the grasp point toward the tilt, so the grasp point
  goes that much farther back (from higher, the stream overshoots the far side or splashes).
  Tilt in stages - a full container starts to flow at about 100 degrees - to well past level
  (130-150), and hold each stage until the flow stops (`robot.wait`, 40-90 steps in all); a fast
  or deep tilt in one move spills, and a spill cannot be undone. Empty it in one pass:
  a second round costs as much as the first. Turn it back upright over the target before moving
  away, and put it down on a free spot.
- Things that move by themselves (a conveyor, a falling or rolling object, another robot arm):
  `robot.wait(n)` lets the time pass without moving the arms; look again to see whether they are
  still moving.

The two arms:
- Each arm reaches its own side best. For something in the middle, check both with `dry_run`.
- Choose the arm for the whole pick-and-place: before picking something up, dry-run the put-down
  pose with the arm you will pick with (a tilted pitch, 30-60, or a level hand reaches farther
  than straight down). A hand-over costs 70-110 steps and often fails; a direct place 20-35.
- When the arm that holds it cannot reach the place, change arms through a still object rather
  than in the air: put it back where it came from, or - if it lies still and can be gripped again
  as it lies - set it on a clear spot both arms reach, move that hand up and away, and pick it up
  there with the other arm. Thin or standing things set down loose often end up lying flat.
- Keep the arm that is not working clear of the other arm's path, the camera views you need and
  the objects: lifted and toward its own side is enough - a short move costs far fewer steps
  than `home`, which is for the end and for another arm's turn.
- A hand-over in the air (for what cannot be set down): one arm holds the object still and high; the other grips a free part of it; the
  first then opens and backs away along its pointing direction. Open the first only in a later
  cell, after the second's Grip shows it holding with a width that fits the part it gripped: when
  it closed empty or on the wrong part, the object falls.
