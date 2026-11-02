# Program API

You act by submitting Python programs with the `exec` tool: `exec(code, include=[...])`. One program can run a
whole subgoal (approach, grasp, lift, carry, place) guarded by checkpoints. The function names, arguments and
result fields are those of the H4/H5 harness, so `skills/` and `lessons/` written for it run unchanged.

## The program namespace

- Python with `import numpy as np` / `import numpy` / `import math` / `from math import ...` /
  `from numpy import ...` only. `np` and `math` are already defined.
- Not available: other imports, names or attributes starting with `_`, classes, `open`/files, `eval`/`exec`,
  `getattr`/`setattr`, `globals`/`vars`/`dir`, numpy file I/O (`np.load`, `np.save`, `a.tofile`, ...).
  Such a program is rejected before anything runs (`error.type = 'SandboxError'`, with the line).
- `print(...)` goes to the receipt's `stdout` (at most 8000 characters: head and tail kept). Print the numbers
  you will need for the next decision.
- The namespace persists across `exec` calls: functions and variables defined by one program are there in the
  next. Measurements do not stay valid: re-measure after anything moved.
- Preloaded, no include needed: everything below plus the manipulation library (`pick`, `place`, `look_down`,
  `press_at`, `screw_down`, `grasp_point`, `best_yaw`, `choose_yaw`, `release_yaw`, `grasp_height`,
  `equivalent_yaws`; see api_reference topic `manipulation`), the checks and guarded primitives below
  (`hold_test`, `insert_along_axis`, `choose_arm`, `release_check`, `scene_snapshot` / `displaced`,
  `path_clearance`, `jet_landing` / `jet_speed`), the pour tools (`rim_transfer`, `release_retreat`) and
  `guarded_move` / `bounded_recovery`.
- Constants: `ARMS`, `PHASE` ('practice' | 'test'), `GRASP_OFFSET` (grasp point in the flange frame),
  `FINGERTIP_BEYOND_GRASP_M`, `TABLE_Z`, `TOOL_AXIS`, `CLOSING_AXIS`, `WRIST_CAMERAS`.
- Wall-clock deadline per program (default 20 min). Every robot call and every loop iteration checks it;
  past it the program stops with `stop_reason='timeout'`.

## Includes

`include` lists files run, in order, before the program in the same namespace:
`skills/<name>.py` (task skills) and `skills/shared/<name>.py` (read-only shared library). Anything else
(lessons/, scratch/, paths leaving the workspace) is refused. The H4/H5 names `tools/high_level.py`,
`tools/manipulation.py`, `tools/pour.py` and `primitives/closed_loop/motion.py` are accepted and do nothing
(their functions are preloaded). An include file must not move the robot at top level; define functions.

## Frames, angles, units

World frame in metres, +z up; quaternions are `[w, x, y, z]`.
- `frame='ee'` (default of `move_to`): the flange (link6) origin; `get_observation()['state'][arm+'_ee_pose']`.
- `frame='grasp'`: the grasp point between the finger pads (`GRASP_OFFSET` along the pointing axis; on the
  RoboDojo ARX arms 0.145 m, fingertips 0.013 m beyond). Use it for grasps and placements.
- `frame='tcp'`: another contact point, `tcp_offset_local=[x, y, z]` in the flange frame plus a
  `calibration_ref` naming where the offset came from.
- `frame='tip'`: the tip of a held object measured with `inspect('axis_tip', ..., arm=...)`.
- Orientation: `quaternion`, or `yaw`/`pitch`/`roll` degrees. pitch 90 points the gripper straight down (the
  default when only yaw is given; yaw defaults to 0, roll to 0); then the fingers close along the line at the
  yaw's heading. yaw and yaw + 180 close along the same line. Without any orientation the current one is kept.
- Gripper commands: 0 closed .. 1 open (`'open'`/`'close'` also accepted by `set_gripper`).

## Motion

`move_to(arm, xyz, quaternion=None, *, yaw, pitch, roll, grip=None, max_steps=None, position_tolerance=None,
angle_tolerance=None, translation_step=None, rotation_step=None, frame='ee', tcp_offset_local=None,
calibration_ref=None, label=None, reserve_native=None, precision=False, stop_on_contact=False,
on_stall='interrupt', path='auto', final=False, wait=True, queue=False)`

The move is planned before anything moves (joint waypoints from the robot model, each solved from the previous
one; `path='auto'|'line'|'joint'|'ee'`; `'ee'` is planned as a straight line when the model allows it). `xyz` may
also be a reference id from `locate`/`grasp_candidates` (a grasp reference supplies yaw and pitch). `grip=None`
keeps the last gripper command (a held part stays squeezed). Steps: 4 cm per native action with an open gripper,
3 cm when closed, 0.15 rad of rotation, and at most 0.2 rad per joint; the arm moves on to the next waypoint
only once it has arrived near the current one, so a slowed arm is waited for, not left behind. While the gripper
is closed on a part, the move stops with `'dropped'` as soon as the fingers close on nothing.
Final tolerance 3 mm / 0.03 rad unless given. `precision=True`: the last 2 cm in 3 mm steps, 1 mm / 0.015 rad.
`stop_on_contact=True`: stopping short against something returns `'blocked'` (not a stall) and holds there.
`max_steps` caps the native actions of this move (default: what the plan needs plus a stall margin).
`label` names the stage in `budget_panel.labeled_stages`; `reserve_native=N` refuses to start, or stops, the
move once only N native actions would remain.

Result: `status`, `arm`, `frame`, `reachable` (True / False / 'unknown'), `requested_pose`, `actual_xyz`,
`position_error_m`, `position_error_mm`, `orientation_error_rad`, `calls` (native actions), `path`,
`planned_actions`, `track`.

| status | meaning | moved? |
| --- | --- | --- |
| `tracking_reached` | within tolerance | yes |
| `blocked` | stopped against something (only with `stop_on_contact`) | yes |
| `stalled` / `timeout` | stopped making progress / ran out of `max_steps` | yes |
| `dropped` | the part held when the move started slipped out of the fingers (`opening` says how far they closed) | yes |
| `model_unreachable` | the robot model has no solution for that pose | no |
| `path_blocked` | no collision-free path (`reason` says why) | no |
| `over_budget` | the move plus the way back to the start pose and a 10-action reserve exceed what is left | no |
| `repeat_refused` | the same move just stalled or was blocked; change arm, pitch, height, path or approach | no |
| `episode_over` | the episode ended during the move | - |

`on_stall='interrupt'` (default): `stalled`, `dropped`, `timeout` and `over_budget` stop the whole program
(`stop_reason='motion_interrupted'` or `'over_budget_refused'`, with the program line). Catching the exception
does not help: every later robot call in that program raises it again. `on_stall='return'` returns these
statuses so your code can switch arm, pitch or height. `final=True`: the last scoring motion, allowed to use
the actions reserved for the way back.

`set_gripper(arm, command, *, settle_steps=None, until_still=True, label=None, reserve_native=None,
prepare_lift_ref=None, verify_lift_ref=None)`: holds the arm, commands the gripper until the measured fingers
stop (at most `settle_steps`, default 8 for a close and 3 for an open, which only has to clear the part; an
already open gripper costs nothing). Result: `status`
('command_sent' | 'episode_over'), `commanded`, `width` (measured opening, 1 open, about 0 closed on nothing),
`holding` after a close ('yes' when the fingers stopped on something, 'no' when they closed on nothing), `calls`.

`check_reach(xyz, quaternion=None, *, yaw, pitch, roll, frame='grasp', arms=('left', 'right'))`: no motion.
`{arm: {'reachable': True | False | 'unknown', 'variants': [...], 'joint_travel_deg'}}`. Variants when not
certainly reachable: `{'pitch': p, 'yaw', 'roll'}` (a tilted approach) or `{'xyz': [...], 'note': '2 cm higher'}`.
`unknown` moves stall more often; prefer the other arm or a variant. Of two reachable poses prefer less
`joint_travel_deg`.

## Two arms at once

- `move_to(..., wait=False)` (and `set_gripper(..., wait=False)`) returns `{'status': 'running', 'track': id}` at
  once; the arm moves while your program continues. Commands to the other arm run in the same ticks.
  A call on a busy arm is refused unless `queue=True`.
- `wait(tracks=None, ticks=None, seconds=None)`: waits for the given tracks (default: every asynchronous move
  this program started), or advances time. Returns `{'waited_ticks', 'tracks': [result, ...]}` with the same
  result fields as the synchronous call; `on_stall='interrupt'` applies when the result arrives.
- `both(left=dict(...), right=dict(...))`: one command per arm, run concurrently, then waits for both. Each dict
  holds `move_to` keyword arguments (with `xyz=`) or `set_gripper` arguments (with `command=`). Returns
  `{'left': result, 'right': result, 'ticks': n}`. Two 20 cm moves cost about 5 native actions together
  instead of 10 one after the other.
- `tracks()`: running/queued tracks per arm lane; `cancel_track(track_id)`.
- Tracks still running when a program ends are waited for (completed program) or cancelled (stopped program).

## Perception and skills (no native actions except `observe`)

- `locate(text, camera='head', pixel=None, max_objects=1)`: `{'status': 'located' | 'not_found', 'ref', 'xyz'
  (top-face centre), 'object_pose': {'center_xyz' (footprint centre at half height), 'top_z', 'height_m',
  'length_m', 'width_m', 'long_axis_deg', 'grasp_yaw' (move_to yaw closing across the narrow side),
  'closing_heading_deg'}, 'objects': [...], 'warnings'}`. Cameras: 'head', 'left_wrist' / 'wrist_l',
  'right_wrist' / 'wrist_r'. Pass `pixel=[u, v]` on the instance you mean when similar objects exist: a name
  or number alone often matches several similar objects. `pixel` is in the camera's native resolution; the
  images you see may be downscaled (each image's label gives both sizes and the factor). Head
  camera: about 1 cm XY, heading unreliable; a wrist camera 15-25 cm straight above: about 1 mm.
- `inspect(mode, target, camera='head', arm=None)`: `shape`, `axis_tip` (held or elongated object's axis and
  tip; pass `arm` when held, then `frame='tip'` works), `support_plane`, `opening` (rim/hole centre, radius).
  `status` is `'measured'`, or `'not_found'` (with `reason`) when no mask or fit was found: the program goes on,
  so check the status and fall back (another camera, a wrist view from above, `locate` with a pixel).
- `observe(arm, target, view='down', distance_m=0.22)`: moves that arm's wrist camera to look at a target (xyz
  or reference) from 'down', 'front', 'back', 'left_side', 'right_side', 'oblique' or a direction vector. The
  free arm can watch the working arm. Returns the move's `status` ('tracking_reached', ...), `camera` and
  `target_pixel`; the view comes back with the receipt.
- `grasp_candidates(target_ref, arm='auto')`: ranked top grasps (`ref`, `grasp_xyz`, `yaw_deg`, `arm`,
  `reachable`). `move_to(c['arm'], c['ref'], frame='grasp')` takes yaw and pitch from the candidate.
- `run_skill(skill_id, **bindings)`: a library skill run to its endpoint as one track; returns its receipt.

## Checks before irreversible steps, and guarded primitives (preloaded)

Each returns a `status` and the numbers behind it; none opens the gripper unless its description says so, and
a failed check moves nothing further. Finger openings are measurements: confirm a hold that matters with an
image.

- `choose_arm(stages, arms=None, yaws=None, pitch=90)`: no motion. Which arm and yaw reach every stage of a
  plan (grasp, lift, carry, entry, release, turned pose...). A stage is `[x, y, z]` or a dict with `xyz` and
  `quaternion`, or `yaw` / `yaw_offset` (added to the candidate yaw) / `pitch` / `roll`. `'reachable'` with
  `arm`, `yaw`, `joint_travel_deg` (least summed travel), or `'unreachable'` with `best_partial` (first failing
  stage). Check the whole plan before the grasp: a pose that fails halfway leaves the part in the wrong hand.
- `hold_test(arm, lift=0.02, hold_ticks=3, jog=0.0, max_opening_drop=0.1)`: right after a grasp, lift a little,
  hold still, optionally jog sideways and back (across the closing line), and compare the measured opening.
  `held` / `slipping` / `empty` / `not_holding` / `dropped` / `lift_failed` / `jog_failed`. Also
  `pick(..., hold_ticks=3, hold_jog=0.02)`: then pick's status is `'hold_<status>'` unless the hold is `held`.
  About 10 native actions; it moves a weak grasp's failure from mid-carry to just above the source.
- `insert_along_axis(arm, entry_xyz, axis, depth, step=0.008, search=0.0, max_searches=8, twist_deg=0.0)`: the
  held part (orientation kept) advances from the entry along `axis` (`[0, 0, -1]` into a hole below; the measured
  shaft direction to thread a handle or loop over a peg) in precise steps that stop on contact; when stuck,
  `search` metres of sideways offsets are tried; the finger opening is checked after every step; `twist_deg`
  turns the part about the axis at the end (a key). Preflights entry, seated and twisted poses. Never opens:
  `inserted_unverified` / `twisted_unverified` / `blocked` / `grasp_changed` / `dropped` / `entry_failed` /
  `reach_unconfirmed` ..., with `depth_m` (grasp-point travel, not proof of insertion) and `searches`.
- `release_check(arm, region, below_grasp, support_z, max_drop=0.02, offset_xy=(0, 0), margin=0.0)`: before
  opening, where the held part lands: region `{'center', 'radius'}` or `{'polygon': [[x, y], ...]}`;
  `ok` / `outside` / `too_high` / `below_support`, with `inside_m` and `drop_m`. `place(..., max_drop_m=0.015)`
  withholds the release (`'release_too_high'`, still holding) when the descent stopped that far above `xyz`.
- `scene_snapshot(names)` then, after a contact-rich move, `displaced(snapshot, tolerance=0.015)`: re-locates each
  object (no native actions) and lists the ones that moved or vanished (`unchanged` / `displaced` / `lost`). A
  moved container, rack or receiver invalidates every target measured before the move.
- `path_clearance(points, poses, clearance=0.03)`: distance from a grasp-point route (e.g. an observer move) to
  measured object boxes (`object_pose`s); `clear` / `conflict`. Only the grasp point is checked, not the hand.
- `jet_landing(mouth_xyz, outflow, receiver_xyz, receiver_radius, exit_speed=0.3, margin=0.01)`: where a
  stream from the mouth lands at rim height (it is thrown forward, more so from higher up): `inside` / `outside`
  with `miss_m`, `throw_m` and `aim_mouth_xyz` (the mouth position that lands on the centre). After the first
  visible stream, `jet_speed(mouth_xyz, outflow, landing_xyz)` fits the speed; its `lateral_m` is mouth error.

## Raw observation and actions

- `get_instruction()`; `get_observation()`: `{'instruction', 'vision': {'cam_head': {'color'},
  'cam_left_wrist': ..., 'cam_right_wrist': ...}, 'state': {'left_ee_pose': [x, y, z, qw, qx, qy, qz],
  'left_arm_joint_state': [6], 'left_ee_joint_state': [measured opening], right_...}, 'camera_calibration':
  {'cam_head': {'intrinsics': K, 'extrinsics_world': 4x4 camera-to-world, 'size': [w, h]}, ...}}`.
  `ee_joint_state` is the measured finger opening, not the last command: sending it back as a command can
  release a held part.
- `step(action, stage=None)`: exactly one native action for both arms, end-effector mode
  (`left_ee_pose`, `right_ee_pose`, `*_ee_joint_state`) or joint mode (`*_arm_joint_state`,
  `*_ee_joint_state`). A missing `*_ee_joint_state` keeps the last command. Both arms must be idle.
  Returns `(observation, 0.0, terminated, truncated, info)`; `info['execution_receipt']` has the native
  counts. `stage={'label', 'reserve_native', 'arm', 'goal_pose', 'position_tolerance',
  'orientation_tolerance'}` is optional. Prefer `move_to`: raw steps get no planning, reach or budget check.
- `reset(force=False)`: practice only. Restarts the same scene; the native-action budget starts over (the
  program count does not); references and marks are dropped. Refused with fewer than 10 counted programs left
  unless `force=True` (the scene can no longer be completed).

## Checkpoints, sub-goals and the budget panel

`checkpoint(name, kind, **params)` is evaluated by the host mid-program and costs no native action. A failed
or undecidable checkpoint stops the rest of the program (`stop_reason='checkpoint_failed'`, with the line).
Names: 1-40 of `[A-Za-z0-9_-]`. Marks live only within one program.
- `mark`: remember the current poses under `name`.
- `native_reserve`, `min_remaining=N`: pass while at least N native actions remain.
- `pose`, `arm`, `xyz`, `tolerance` (metres, at most 0.1), optional `frame` ('ee' default, or 'grasp'): pass
  when that point is within tolerance of xyz.
- `lift_retained`, `arm`, `before` (a mark), `source`, `support_z`, `allow_unknown=False`: pass when the gripper
  is closed on something (measured opening) and moved at least 2.5 cm (or 10 degrees) since the mark. It reads
  only the finger opening and the motion, not the image: a part caught by one finger, or left on its pad, can
  pass; confirm a lift that matters with a view or a `locate`.
- A condition your program computed: `checkpoint(name, ok, **evidence)` with `ok` True or False (the evidence is
  logged with it).

`subgoal_done(name)` marks a finished sub-goal (an object placed, a screw down). `budget_panel()` (also in
every receipt): `native_steps_used`, `native_steps_remaining`, `return_to_origin_estimate`, `safety_reserve`,
`usable_native_steps`, `at_origin_for_scoring`, `origin_distance`, `subgoals` (`native_steps_each`,
`median_native_steps`, `more_affordable`), `labeled_stages`, `advice`. `status()` adds arm states and tracks.

```python
red = locate('red block')['object_pose']
checkpoint('before_lift', 'mark')
p = pick('left', red, label='red')
checkpoint('held', 'lift_retained', arm='left', before='before_lift', source=red['center_xyz'], support_z=TABLE_Z)
bottom_below_grasp = p['grasp_z'] - (red['top_z'] - red['height_m'])
place('left', [0.15, 0.12, 0.79 + bottom_below_grasp + 0.003], period=180, label='red')
subgoal_done('red_on_blue')
print(budget_panel()['usable_native_steps'])
```
If `held` fails, nothing after it moves: look at the images and regrasp in the next program.

## The receipt

`ok` (no error), `stop_reason` ('completed' | 'exception' | 'checkpoint_failed' | 'motion_interrupted' |
'over_budget_refused' | 'episode_over' | 'timeout'), `error` (`type`, `message`, `file`, `line`,
`program_line`: the top-level statement that was running), `interruption` (reason, line, detail), `stdout`,
`native_steps` (`before`, `after`, `used`), `tracks` (motions finished during the program), `raw_steps`,
`calls` (per function), `checkpoints`, `includes`, `budget_panel`, `unwaited_tracks`, and in practice
`score_feedback`. Views taken with `observe` come with it; your next context shows the current head and wrist
images. Receipts report what the robot did, never task success: judge that from the images.

## What things cost (native actions)

| motion | native actions |
| --- | --- |
| `move_to` in free space | about 1 per 4 cm (3 cm when holding); `planned_actions` in the result |
| `pick` (hover 5 cm, precise descent, close, lift) | about 20 |
| `place` over a short carry (raise, carry, precise lower, open, retreat) | about 20, plus 1 per 3 cm carried |
| `set_gripper` | until the fingers stop: up to 6 for a full close or open, 2-4 onto a part (0 when already open) |
| `handover` (approach 6 cm, close, giver opens and withdraws) | about 25 |
| `look_down` + wrist locate | 8-15 |
| `hold_test` (2 cm lift, 3 ticks, 2 cm jog and back) | about 10 |
| `insert_along_axis` | about 3 per 8 mm step, plus 4-6 per sideways search and 2 per 15 degrees of twist |
| `choose_arm`, `release_check`, `scene_snapshot`, `displaced`, `path_clearance`, `jet_landing` | 0 |
| `precision=True` | about 7 for the last 2 cm, so only for the final approach |
| two arms with `both()` / `wait=False` | the longer of the two moves, not the sum |
| `locate`, `inspect`, `check_reach`, `checkpoint`, `budget_panel` | 0 |

The wrist view pays for itself only when millimetres matter (seating, inserting, stacking on a small part);
for a large object the head camera's 1 cm is enough. Mark each finished sub-goal with `subgoal_done(name)`;
the budget panel then shows what a sub-goal costs and how many more fit before the way back.
