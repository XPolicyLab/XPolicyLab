# Shared public tool extensions

Use these as robo methods. A successful preview is not a collision or support guarantee.

## robo.tool_doc(*names)

Signature and complete contract of any `robo` tool: the robot's own methods (including
the ones the reference only lists by name) and the shared extensions on this page.
No actions.

## robo.preview_move(xyz, *, arm, pitch=None, yaw=None, roll=None, path='auto')

Robot helper; inspect current observations before motion.

## robo.preview_candidates(coarse, *, refine=(), charge=None)

Ordered, bounded first-feasible search from one locked joint state.

Candidates use preview() arguments: xyz in world metres; pitch/yaw/roll
in the existing tool convention, arm left/right, path auto/line. Refinements
are explicit alternate poses, tried only if coarse search fails. No action,
scene/object collision guarantee, waypoint FK or automatic execution.

## robo.move_both(left, right, *, clearance_evidence, max_actions=40, own_robot_margin_m=None, preview_only=False, approved_plan=None)

Plan or execute bounded synchronized motion, preserving both grippers.

preview_only=True returns zero-action planned joint/TCP samples and a
single-use plan_id. Inspect that actual route against current RGB and held
material, then call again with approved_plan=plan_id and identical poses,
max_actions and margin. Dispatch consumes the stored route without replanning.
Changed robot state/status or another preview invalidates the ticket.
Errors return to the caller before motion for stale/mismatched plans; partial
execution errors require observation/reconciliation, never blind retry.
TCP/optional sampled robot separation is not scene/cloth collision checking.
Calls without a ticket plan automatically.

## robo.set_grippers(*, left, right, min_actions=2, max_actions=10)

Robot helper; inspect current observations before motion.

## robo.observation_bundle(*, cameras=('cam_head', 'cam_left_wrist', 'cam_right_wrist'))

No actions. Reject views spanning an observation step.

Returns state and views={camera:{rgb,calibration,rgb_sha256,step}}. Original
pixels and calibration are preserved; images are not automatically displayed.
The caller must explicitly display/review the views relevant to its decision.

## robo.geometry_capture(text_prompt, camera='cam_head')

Segment CURRENT RGB and retain its calibration/robot mask.

Returns an opaque handle, step, mask score/hash and limitations. Full mask
evidence is archived once per explicit call, not on physics steps. No
motion, object identity proof or success feedback is used.

## robo.geometry_measure(first, second, *, stationarity_basis, planar_table=False)

Fit stored views of the same described object; never certify stability.

Handles belong only to this episode and expire after four captures.
Invalid/missing handles or different descriptions are rejected. A review
string describes an assumption, not a stationarity oracle.
planar_table=True fits a flat marking on the existing calibrated
table plane. It is invalid for raised objects; consistency is not accuracy.

## Library and state

`robo.skill_catalog(capability=None)` retrieves module/function signatures. `robo.skill_doc("capability/module.py", "function")` returns its complete docstring. `robo.skill_read("relative/path")` reads text or image bytes. Use `display(robo.skill_read("path.png"))` to actually see an example. `robo.record_progress(dict)` retains a compact agent-authored checklist. `robo.action_count()` returns actions executed. do not waste actions.
