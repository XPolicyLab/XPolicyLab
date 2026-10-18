# Tool library for the generic agent

A small set of task-agnostic `robo` commands for the generic (open) profile, taken from the tools that the
optimizer agent wrote for the 34 non-Open RoboDojo tasks. Nothing here was taken from, or checked against, the 8
Open tasks (align_blocks, general_pickup, stack_blocks_by_language, solve_equation, classify_objects_by_language,
pick_from_conveyor_by_image, store_tools_in_toolbox, pour_by_language).

Each `library/<name>/` is an ordinary tool directory (`tool.py` + `interface.md` + helper files), loaded by
`roboshell/server/tools.py` when `<name>` is listed in the open profile's `enabled_tools.txt`. All eight names:

```
rgb_point
rgb_quad
boundary_candidates
rim_geometry
guarded_grasp
guarded_press
bounded_rotate
timed_pick
```

Together they add 12 commands (6 free, 6 budgeted). No command name clashes with the base commands (move, rotate,
point, gripper, home, wait, obs, status, done), the open-profile commands (goto, reach, pixels, py), the shared tools
(`table_plane`, `pick`) or the names kept free for other new tools (segment, locate, scene).

Selection rules: no object sizes, colours, positions, templates or layout priors in the code; robot geometry, the
workspace box and generic tolerances are allowed. Every chosen file was read in full. Each `interface.md` was
rewritten for an agent that has never seen the source task; command syntax is unchanged.

## Chosen tools

| Library name | Source (task / original path) | Purpose | Commands | Dependencies |
|---|---|---|---|---|
| `rgb_point` | swap_blocks / `tasks/swap_blocks/tools/rgb_point/` | 3D position of an image point: automatic height by matching a pixel patch (or a plain region's outline) across the other cameras, or a ray on a given plane, or two-view triangulation, or triangulation with a ray saved earlier | `rgb_point`, `rgb_ray` (free) | `auto_surface.py`, `silhouette.py` (same directory); numpy, OpenCV |
| `rgb_quad` | store_laptop_and_headphones / `tasks/store_laptop_and_headphones/tools/rgb_quad/` | Corners, centre, side lengths and tilt of a flat four-sided face from two rough image rectangles | `rgb_quad` (free) | `rgb_measure.py` = copy of `tasks/store_laptop_and_headphones/tools/rgb_measure/tool.py`, used as a helper module only (its own command is not exposed); numpy, OpenCV (SIFT/AKAZE code in the helper is not used by `rgb_quad`) |
| `boundary_candidates` | pour_liquid_into_cup / `tasks/pour_liquid_into_cup/tools/boundary_candidates/` | Closed elliptical edges in an image rectangle, with exact edge pixels (input for `fit-rim`) | `boundary_candidates` (free) | numpy, OpenCV, Pillow |
| `rim_geometry` | pour_balls_into_vase / `tasks/pour_balls_into_vase/tools/rim_geometry/` | Centre, radius and height of a level circle from edge pixels in 2–3 views (no point-to-point matching needed; optional snapping to image edges) | `fit-rim` (free) | numpy, OpenCV |
| `guarded_grasp` | make_toast / `tasks/make_toast/tools/guarded_grasp/` | Top-down or tilted grasp with any finger yaw; grasp along any approach direction (e.g. from the side); put down at a point and release | `grasp_point`, `grasp_pose`, `place_point` (budget) | numpy |
| `guarded_press` | make_toast / `tasks/make_toast/tools/guarded_press/` | Contact stroke: approach a point with a chosen gripper direction, then move a bounded stroke (press down, push sideways), withdraw; or the same with a point of a held object | `press_pose`, `press_feature` (budget) | numpy |
| `bounded_rotate` | plug_in_charger / `tasks/plug_in_charger/tools/bounded_rotate/` | Large rotation (up to 180° per axis) about the TCP in checked steps, with reachability preflight (tilting, pouring, turning over) | `bounded-rotate` (budget) | `rotation_steps.py` = the function `orientation_steps`, copied verbatim from `tasks/plug_in_charger/tools/mate_pair/tool.py`; numpy; `api.geometry`, `api.estimate_tcp_chain` |
| `timed_pick` | match_and_pick_from_conveyor / `tasks/match_and_pick_from_conveyor/tools/timed_pick/` | Grasp of a point moving at constant velocity, timed with a reachability and timing preflight (zero velocity: timed top-down grasp) | `timed_pick` (budget) | numpy; `roboshell.server.core` (`tool_rotation`, `GRIPPER_STEPS`, `SETTLE_MAX_STEPS`), `roboshell.server.motion.CONTROL_HZ` |

The free perception tools work from the RGB images and camera matrices of `api.observe()` only (no depth). None of the
chosen tools needs `tools/table_plane`; the helper `rgb_quad/rgb_measure.py` can call it (plane mode, not used by
`rgb_quad`) through `Path(__file__).parents[2] / "tools/table_plane/tool.py"`.

Typical use: `rgb_point` / `rgb_quad` / `boundary_candidates` + `fit-rim` to measure, `grasp_point` or `grasp_pose`
to pick, `place_point` to put down, `press_pose` to press or push, `bounded-rotate` to tilt or pour,
`timed_pick` for moving objects.

## Changes against the source code

Only what the move needed, plus three small fixes; each changed line is commented or listed here.

- `rgb_quad/tool.py`: loads the helper from `rgb_measure.py` next to it (was `../rgb_measure/tool.py`).
  `rgb_quad/rgb_measure.py`: its scan mode loads `tool.py` next to it (was `../rgb_quad/tool.py`); its plane mode
  finds table_plane at `parents[2]` (was `parents[4]`, which is wrong outside `tasks/<task>/tools/<name>/`).
- `boundary_candidates/tool.py`: the source loaded `pour_liquid_into_cup/tools/region_geometry/tool.py` only for its
  camera-name table; that table is now inline (identical values).
- `bounded_rotate/tool.py`: loads `rotation_steps.py` (the one function it used) instead of the whole 839-line
  `mate_pair` tool of the source task.
- Fix: `guarded_grasp`, `guarded_press` and `timed_pick` returned exit code 1 (`EXIT_BAD_ARGS`, "budget not
  consumed") on every failure, although these commands are budgeted and the command was already charged. They now
  return 2 (`EXIT_EXEC_FAILED`), as the agent manual and `roboshell/contract/v0.py` define. Feedback is unchanged.
- Fix: `rgb_point` accepted only the observation names `cam_head`, `cam_left_wrist`, `cam_right_wrist`, while the
  agent manual and the other tools use `head`, `wrist_l`, `wrist_r`; it now maps those names too (old names still work).

Known limits kept as they are (documented in the manuals): `guarded_press` stops in place, without withdrawing,
when a stroke is resisted; `guarded_grasp` opens to 0.5 (4.4 cm) by default; the free tools report failures with
exit code 1 or 2 but never use budget; no motion tool checks collisions with objects.

## Load test

The development harness (not part of this directory) loads every `library/<name>/tool.py` the way the server does,
checks the `TOOL` schema, name clashes, argument validation and the robo client's sub-commands, and runs each command
once on a synthetic ray-traced scene through a stand-in EpisodeAPI (CPU only). Result on 2026-10-18: 8/8 tools
loaded, 12 commands, 0 schema problems, smoke tests 20/20.

## Overlap with other commands

- `goto` / `reach` (open profile) cover single absolute moves and reachability checks, so no IK-query tool
  (`transfer_reach`, `transfer_check`, `probe-transfer`) and no plain carry tool was taken.
- `tools/pick` (shared) is a simpler top-down pick; `grasp_point` adds finger yaw, tilt and opening, and
  `grasp_pose` adds any approach direction.
- `tools/scene` (`segment`, `locate`, `pixel_world`) overlaps with `rgb_point` for pixel-to-3D; `rgb_point` stays
  for its automatic height and two-view modes.

## Rejected candidates

One line per enabled tool of the 34 non-Open tasks (`table_plane` is shared and not repeated). Common reasons:
*task routine* = the commands encode the source task's sequence or object structure; *overlap* = covered by a
chosen command; *heavy* = large file with sibling-tool dependencies for little extra generality; *depth* = needs a
depth image that the RGB-only setup does not have.

- arrange_largest_number / `precision_transfer`: heavy (1020 lines + 2 helpers), its `locate` command clashes with a reserved name; overlap with rgb_point + guarded_grasp.
- arrange_largest_number / `metric_patch`, `transfer_patch`: depend on precision_transfer; overlap with rgb_point.
- arrange_largest_number / `transfer_reach`: IK route query, covered by `reach`.
- build_tower / `precise_transfer`: heavy (1047 lines, needs color_region, layered_geometry, rgb_geometry); many assembly-specific options.
- build_tower / `carry`: needs precise_transfer; covered by place_point and goto.
- build_tower / `rgb_geometry`: `rgb_assembly` encodes the source structure (supports, spans, cap, crest); `rgb_measure` overlaps rgb_point / rgb_quad.
- classify_objects / `pixel_grasp`: heavy (767 lines, 5 commands); overlap with rgb_point + guarded_grasp.
- classify_objects / `stereo_point`: good correspondence refinement, but overlaps rgb_point's automatic mode.
- classify_objects / `line_fit`: narrow (one straight feature), needs pixel_grasp helper.
- cover_blocks / `precision_transfer`: finds faces by a given hue; batch and return-transfer commands follow the source routine.
- deposit_coin / `rgb_geometry`: overlap with rgb_point (its dark-segment refinement is narrow).
- deposit_coin / `rgb_insert`: task routine (thin round part into a slot), 651 lines, SciPy.
- fasten_screws / `pick_part`, `insert_part`, `boundary_pose`: shaped for a ring-like part on a post; boundary_pose overlaps fit-rim.
- fill_egg_holder / `carton_items`: object-specific model (holder with four recesses, lid, hinge).
- fill_pen_holder / `segment_*`, `upright_transfer`, `aperture_*`, `side_grasp`, `release_retract`: one chain for long thin items into an upright holder (shared segkit, "vivid receptacle" colour heuristic); side_grasp is covered by grasp_pose, release_retract by place_point.
- fold_clothes / `panel_cycle`, `crease_transfer`, `cloth_view`, `surface_outline`, `view_reacquire`: deformable-sheet specific, heavy.
- hang_mugs / `precision`, `pegs`: peg and loop specific commands, 1300–1600 lines each.
- imitate_sorting_sequence / `vision_checks`: `pixel-world` needs depth, `point-still` is an odd combination; only `wait-still` is generic.
- imitate_sorting_sequence / `checked_transfer`, `checked_pick`, `landing_search`: depth.
- imitate_sorting_sequence / `rgb_watch`: watching a demonstration, complex (660 lines).
- imitate_sorting_sequence / `rgb_transfer`: its `locate` clashes with a reserved name; container heuristics; 1172 lines.
- insert_key / `key_tools`: object-specific.
- insert_tubes / `rigid_place`: task routine (1599 lines). `axis_grasp`: covered by grasp_point yaw. `return_home`: same as `home`. `stereo_feature`: overlap with rgb_point / rgb_quad. `surface_circles`: narrow (holes in one coloured face).
- make_kong / `row_extension`: task routine. `rgb_geometry`: overlap. `region_geometry`: assumes pale regions and colour patterns.
- make_toast / `rgb_geometry` (`locate_pixel`, good stereo diagnostics), `rgb_patch`, `rgb_corners`, `rgb_edges`: overlap with rgb_point. `frame_place`: full 6-DoF placement, powerful but many arguments. `edge_grasp`, `place_between`: narrow.
- match_and_pick_from_conveyor / `rgb_geometry`: four commands overlapping rgb_point (its `rgb_track` velocity tracker is useful but bundled). `lift_hold`: same as `move --dz`. `planar_ring`: one-view circle on a given height, covered by fit-rim.
- organize_table / `precision`: heavy (1261 lines, 6 commands, `checked_pull` needs depth). `stereo_region`, `patch_plane`, `vertical_span`: overlap with rgb_point. `planar_align`, `region_candidates`, `support_place`: narrow.
- pack_objects_into_box / `guarded_transfer`: heavy (871 lines), rim-routing logic of the source. `transfer_check`, `relay_check`: IK queries (covered by reach). `surface_region`, `stereo_points`, `stereo_edges`: overlap.
- play_stacking_toy / `rgb_geometry` (`rgb_point`, `rgb_circle`, `rgb_rim`): good, but `rgb_point` clashes with the chosen rgb_point and the circle fits overlap fit-rim. `precision_transfer`: held-feature place, many arguments. `rgb_silhouette`: sphere fit, narrow.
- play_tic_tac_toe / `vertical_transfer`: requires both arms home and waits for another actor (source flow). `wait_view`: waits for a region to change and come back, tied to that flow. `surface_points`: grid on a rectangle, narrow; triangulation overlaps.
- play_Xylophone / `rgb_sequence`: object-specific (sphere-and-stem tool, strike series).
- plug_in_charger / `surface_points`: 840 lines, `plane-pair` finds a dark pair (object-specific); stereo modes overlap. `stereo_guides`: helpful epipolar hints but needs that file. `temporal_points`: overlap with rgb_ray. `mate_pair`: pair insertion. `carry_offset`, `axial_pick`: covered by place_point / grasp_pose. `plane_height`: overlap with rgb_point.
- pour_balls_into_vase / `controlled_transfer`: 970 lines, 5 commands; its pour command is tied to source-shaped rim/neck/envelope arguments. `planned_transfer`: full pour routine. `release_retreat`: simple and generic, left out to keep the set small (place_point releases and retreats; goto + gripper cover the rest).
- pour_liquid_into_cup / `settled_cycle`: full pour routine with a source-tuned aiming rule and a colour check. `planar_ring`: 576 lines, overlaps fit-rim. `upright_geometry`: narrow.
- press_by_number / `surface_press`: `ordered_press` encodes the source press sequence and arm assignment; `surface_point` overlaps rgb_point.
- push_T / `planar_contact`: registration of two equal coloured regions (source concept), 1014 lines.
- put_bottles_into_dustbin / `axis_grasp`, `span_grasp`, `vertical_grasp`: upright/lying cylinder grasps with fixed entry directions; covered by grasp_pose. `supported_transfer`, `opposed_transfer`: hand-over routines. `plane_project`: same as `rgb_point --z`. `vertical_span`, `horizontal_span`, `stereo_span`: cylinder axis fits, narrower.
- sort_nesting_dolls_by_size / `rgb_geometry`, `rgb_transfer`: fixed colour list (yellow, red, green, blue, orange). `circle_geometry`: one-view circle, covered by fit-rim. `tcp_route`: generic waypoint idea, but its automatic checks import rgb_transfer's colour segmentation and can refuse routes.
- stack_blocks / `top_pick`: robust, but fingers only along x or y; grasp_point has free yaw. `top_place`: assumes an upright held item centred on the TCP. `settled_pick`: retry wrapper. `plane_localize`, `stereo_localize`: overlap with rgb_point. `prism_localize`: square-prism shape prior.
- stack_bowls / `rim_measure`: needs ordered point-to-point rim matches (fit-rim does not). `rim_grasp`: rim pinch, narrow.
- store_laptop_and_headphones / `rgb_measure`: good general tool, kept only as rgb_quad's helper because its modes overlap rgb_point. `guarded_grip`, `wrist_guard`, `arc_push`: narrow (closure check, feature-tracked moves, short arcs about a hinge).
- swap_blocks / `cartesian_actions`: heavy (1514 lines), legacy depth paths. `press_surface`: needs cartesian_actions and surface_patch, legacy depth path; guarded_press covers pressing.
- swap_T / `guarded_transfer`: overlap with guarded_grasp, paired-arm commands of the source. `surface_pair`: two matching regions (source concept).
- sweep_blocks / `rgb_manipulation`, `rgb_place`, `rgb_regrasp`, `rgb_align`: overlap with rgb_point + guarded_grasp, many tracking arguments. `rgb_entry_path`, `rgb_entry_stroke`: pushing through an opening, narrow.
