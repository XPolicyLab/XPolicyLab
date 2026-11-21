# Manipulation library

Preloaded in every program (no include). Nothing moves until you call a function; every motion is an ordinary
counted `move_to` / `set_gripper` in the grasp-point frame, with `on_stall='return'`, so these functions return a
`status` instead of stopping the program.

## Frames and angles

- **Grasp point**: the middle of the finger pads, `GRASP_OFFSET` from the flange (0.145 m along the pointing
  axis on the RoboDojo ARX arms; fingertips `FINGERTIP_BEYOND_GRASP_M` = 1.3 cm beyond). `frame='grasp'` moves
  this point; no per-episode fingertip guess.
- `move_to(arm, xyz, yaw=deg, pitch=deg)`: pitch 90 points the gripper straight down; the fingers then close
  along the line at the yaw's heading. yaw and yaw + 180 close along the same line.
- A held object turns with the wrist: turning the wrist by d degrees between grasp and release turns the object
  by d about the vertical.

## Measuring before you grasp

`locate(text, camera, pixel)` returns `object_pose`: `center_xyz` (footprint centre at half height), `top_z`,
`height_m`, `length_m`, `width_m`, `long_axis_deg` (long side, degrees from +x, -90..90) and `grasp_yaw` (the
move_to yaw that closes the fingers across the narrow side). For small or precise parts, measure from a wrist
camera above the object: `look_down(arm, xyz, yaw=grasp_yaw)` holds the gripper in the pick posture with the
wrist camera about 20 cm above `xyz` (from a head-camera locate), then `locate(text, camera=result['camera'])`.
From there `object_pose` is within about a millimetre and a degree, and the pick only re-pitches and descends.

## Functions

`pick(arm, pose, yaw=None, depth=None, hover=0.05, lift=0.06, opening=1.0, close_steps=8, reserve_native=None,
label='pick')`: open, move above the object at the grasp yaw (of yaw and yaw + 180: the reachable one with
less joint travel, then the smaller wrist turn; a tilted pitch from `check_reach` when straight down is not
certainly reachable), descend straight and precisely to `depth` below the top (default half the height, at most
2 cm; fingertips kept off the table), stop on contact, close until the fingers stop, lift. Result: every stage,
`yaw`, `pitch`, `grasp_z`, `holding` (measured), and `status`: `lifted` / `missed` (closed on nothing; nothing
lifted: re-measure and pick again) / `hover_failed` / `descend_failed` / `lift_failed`.

`place(arm, xyz, yaw=None, clearance=0.05, press=0.0, open_steps=6, retreat=0.06, period=None,
reserve_native=None, label='place')`: raise, carry over `xyz` while turning to `yaw` (None keeps the current
heading), descend precisely and stop on contact, optionally press `press` metres further, open, withdraw up.
`xyz` is the grasp point at release: support top + (held object's bottom below the grasp point) + 2-3 mm.
Pass `period` (180 for parts that look the same end to end, 90 for square ones) whenever the object is
symmetric: any yaw + k * period is accepted and the one with the least joint travel is used. Status:
`released` / `carry_failed` / `lower_failed`.

`look_down(arm, xyz, height=0.20, yaw=None, max_steps=60)`: wrist camera `height` above `xyz` with its axis
through it. The camera sits tilted on the gripper, so the gripper is pitched (about 60 degrees on the ARX arms)
to make it look straight down; if that is not reachable it stays at pitch 90 and looks obliquely
(`camera_tilt_deg`; oblique views measure 5-16 mm instead of 1-3 mm). Returns `camera` ('wrist_l' /
'wrist_r') for the next locate, and the `yaw` / `pitch` used.

`press_at(arm, xyz, depth=0.01, yaw=0.0, hover=0.05, grip=0.0)`: hover, push to `depth` below xyz (stops on
contact), back up.

`handover(giver, receiver, xyz, yaw, pitch=0.0, approach=0.06, opening=0.7, close_steps=8, open_steps=6,
retreat=0.06, min_separation=0.025)`: pass the part `giver` holds to `receiver`. `xyz` is the receiver's grasp
point on the exposed part (measure it while the giver holds still: two wrist views); `yaw` / `pitch` its wrist
(pitch 0 = horizontal). The receiver approaches along its own tool axis (`gripper_rotation(yaw, pitch) @
TOOL_AXIS`) from `approach` back, the last part precise and stopping on contact, closes, and only when its
fingers hold does the giver open; the hold is checked again and the giver withdraws up and toward its side.
Close across faces the giver's fingers leave free (closing lines about perpendicular), at least
`min_separation` from the giver's grasp point. Status: `handed_over`; nothing moved: `too_close` /
`unreachable`; giver still holds: `approach_failed` / `receiver_missed`; `dropped`; `giver_retreat_failed`
(receiver holds). Prefer it to a hand-written handover or to `interact`: those knocked the part out or let go
early in every recorded try.

`screw_down(arm, xyz, yaw, strokes=3, stroke_deg=-90, segment_deg=15, sweep=0.006, regrasp_lift=0.008)`: screw a
held nut or cap down its thread in ratchet strokes. Start with the part grasped and seated on the thread start
(grasp point at xyz, wrist at yaw). Each stroke turns `stroke_deg` (negative = clockwise from above =
tightening) while following the thread down by `sweep`, releases, turns the wrist back, regrasps lower and
repeats, so the wrist never reaches its joint limit. Read `descended_m` and the images: the depth reached, not
the number of turns, is what scores.

`rim_transfer(arm=, center=, destination=, radius=, height=, envelope=, neck_radius=, neck_depth=, angle=,
axis=, geometry_ref=, attachment_ref=, tcp_offset_local=, calibration_ref=, max_total_steps=, ...)`: pour from
a rigidly held upright cylinder: lift, move the low rim edge over the receiver opening, tilt about a horizontal
axis ('x' or 'y', perpendicular to the finger line) in `increment`-degree steps clearing the receiver envelope,
and (`restore=True`) tilt back. `release_retreat(arm=, tcp_offset_local=, calibration_ref=, support_ref=,
max_total_steps=, distance=0.10, lift=0.10)`: open, withdraw along the tool axis, lift. Both return tracking
only; pouring and release stay 'unknown' until you look.

Helpers: `grasp_point(arm)` -> (current grasp-point xyz, wrist yaw), `release_yaw(grasp_yaw, heading_at_grasp,
target_heading, period=180)` (wrist yaw that turns the held object to a target heading), `grasp_height(pose,
depth)`, `equivalent_yaws(yaw)`, `best_yaw(arm, xyz, yaws)`, `choose_yaw(arm, xyz, yaw)`,
`gripper_rotation(yaw, pitch)`, `rotation_matrix(q)`, `rotation_quaternion(R)`.

## Placing neatly

1. Decide the object's final heading before grasping (from the scoring rule, or from the reference you place
   it on or next to: measure that reference's `long_axis_deg` too).
2. Grasp across the narrow side (`grasp_yaw`), then compute the release yaw with `release_yaw`.
3. After the lift the object may have shifted in the fingers: before the final descent, measure the held object
   (the other arm's wrist camera via `observe`, or the head camera) and shift the release point by
   (wanted object position - measured object position).
4. Release a few millimetres above the support; do not drop from height unless the task wants it.
5. After release, measure the object again (position and `long_axis_deg`) and correct it if the budget allows.

## Two objects, two arms

When two objects go to two places, let each arm take the object on its side and run the transits together:
`both(left=dict(xyz=..., yaw=..., frame='grasp'), right=dict(...))`, or start one arm with `wait=False`, work
with the other, then `wait()`. Motion planning does not check one arm against the other: keep their paths
apart (each arm on its own side, crossing only one at a time).

## When something goes wrong

- `missed`: the fingers closed on nothing. Re-measure (wrist view) and pick again; do not lift to test.
- `stalled` / `hover_failed`: the arm could not follow that pose. Try the other arm, a `check_reach` variant or
  a lower pitch; never repeat it unchanged (an identical stalled move is refused: `repeat_refused`).
- `path_blocked` / `model_unreachable`: nothing moved; change pose, pitch or arm.
- An object knocked out of reach or tipped over: finish the objects that can still score first, then come back
  to it if actions remain.
