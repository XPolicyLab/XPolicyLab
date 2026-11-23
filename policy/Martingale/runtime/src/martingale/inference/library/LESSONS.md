# Shared manipulation lessons

## Object/scene geometry

No standalone object/scene geometry notes in this file.

## Grasping/contact/other

`--self-play mode`

### Mandatory reading for the current task

Before the first robot motion, read only the lesson matching the current
instruction with `robo.skill_read("<capability>/EPISODE_LESSON.md")`.
Each file contains that task's procedure text. The entries
below are alternatives for different tasks, not a list to read through: do not read
all 20. Read the matching text completely, plus any additional pre-motion text it
explicitly requires. For example, Hang Mugs uses only its own episode lesson;
Insert Key also requires its existing task card. If no listed lesson applies, use
the shared guidance below and the relevant skill documentation.

- `align_blocks/EPISODE_LESSON.md`
- `build_tower/EPISODE_LESSON.md`
- `deposit_coin/EPISODE_LESSON.md` and `deposit_coin/LESSONS.md`
- `fill_egg_holder/EPISODE_LESSON.md`
- `fill_pen_holder/EPISODE_LESSON.md`
- `fold_clothes/EPISODE_LESSON.md`
- `hang_mugs/EPISODE_LESSON.md` and `hang_mugs/LESSONS.md`
- `imitate_sorting_sequence/LESSONS.md`
- `insert_key/EPISODE_LESSON.md` and `insert_key/TASK_CARD.md`
- `insert_tubes/EPISODE_LESSON.md`
- `make_toast/EPISODE_LESSON.md`
- `pick_from_conveyor_by_image/EPISODE_LESSON.md`
- `play_tic_tac_toe/LESSONS.md`
- `pour_liquid_into_cup/EPISODE_LESSON.md`
- `solve_equation/EPISODE_LESSON.md`
- `sort_nesting_dolls_by_size/EPISODE_LESSON.md`
- `store_laptop_and_headphones/EPISODE_LESSON.md`
- `store_tools_in_toolbox/EPISODE_LESSON.md`
- `swap_blocks/LESSONS.md`
- `sweep_blocks/EPISODE_LESSON.md`

### Shared guidance

Read relevant helper signatures, complete docstrings and current observations before
acting. All dimensions/poses estimated in another scene must be remeasured here.
Use a visible support/retention check after grasp, release, transport or nearby contact.
A task predicate may require uprightness, release and homing as well as placement.
Do not use a whole-task `run()` helper without understanding its preconditions; use
its subskills and recover through the same agent when a measurement or action fails.

- **Tiles and symbolic instructions:** read the actual card/instruction and visually
  identify the target set. After pushing, retained rack geometry is useful while
  unchanged; occlusion calls for a new viewpoint, not an immediate task stop.
- **Liquid transfer:** estimate the held vessel and receiving opening, maintain the
  pouring lip over the opening through wrist rotation, inspect transfer, then return
  the bottle upright. If rotation is unreachable, lift into clear space and preview
  alternate waypoints without spilling. Complete every requested transfer.
- **Balls into a vase:** use an unobstructed assistive wrist view and mark the rim and
  cup lip. A stationary vase estimate may be reused, but the held cup pose changes.
  Rotate about the intended lip trajectory, inspect remaining balls and recover
  accessible spills; do not waste actions; avoid inferring current pose
  by projecting old estimates into a new image.
- **Pen holder:** prefer broad markers before thin pens when accessible. Keep the
  holder stable and verify retention before moving its arm. If a dropped pen is too
  close to the holder, create clearance before regrasping. Scout a crowded opening
  before picking up the last pen if visibility requires it; mark a current free
  region on a close-up and actually inspect the overlay. Maintain the observed rim
  height and held-tip offset. Retain, withdraw, reobserve and shift when insertion
  is blocked. Review orientation, seating and release; never tip the held cup to
  reach another object or reuse an unverified old free-space target.
- **Organizing:** transport the clock above drawer edges and verify upright support
  before release. Carry the figurine through clear space and place gently on its
  small stand. Use short observed keyboard pushes and recheck the figurine afterward.
  Review all completed subgoals instead of assuming the keyboard is the only problem.
- **Charger and screws:** observe the held part and receptacle with calibrated close
  views; neighboring geometry can constrain alignment. Approach slowly near contact,
  remeasure after slip/occlusion and verify support before release. Motion speed does
  not compensate for an incorrect geometric target.
- **Pressing:** read both number cards from current images. Press/release the left
  red button its indicated count, then press/release blue
  ONCE. Press/release the middle red button its indicated count, then press/release
  blue ONCE again. There are TWO confirmations, not one total or a third extra.
  Use PressLedger to track observed depression/release cycles, not motion replies.
  Park the inactive hand clear, verify complete spring return before the next
  cycle, and withdraw both hands before opening/homing to avoid extra presses.
- **Folding:** preserve the grasp during controlled cloth transport, inspect after
  release/settling, and correct a sleeve or hem that unfolds. Do not equate an executed
  waypoint with a completed fold.
- **Laptop and headphones:** first hang the headphones on their stand, then close
  the laptop and carry it to the rack. Grasp an accessible headset part, carry it into open space,
  and use a supported regrasp if needed. Measure headset orientation against the stand,
  including front/back identity, rather than guessing wrist angles. Estimate laptop
  edge clearance from stereo observations; create overhang with small observed nudges.
  Verify a probe lift, clear the clutter with the left arm, present the laptop before
  the right hand approaches, secure the receiver before releasing, then align and
  insert with support checks. These components do not establish whole-task reliability.
- **Xylophone and stacking:** ground the target sequence/geometry in observation,
  use repeatable motion segments, and inspect disturbed objects before continuing.

Alternate grasps, arms, viewpoints, supported setdowns and routes remain available.
These are conditional lessons, not universal bans or guarantees of success.
