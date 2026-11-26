"""Tool contract sent to the model, also retained in source snapshots."""

from copy import deepcopy

TOOLS = {
    "fit_sphere": {
        "first": {
            "camera": "cam_high", "frame_id": "CURRENT_FIRST_ID",
            "pixels": "8..64 distinct original [u,v] on the visible sphere silhouette",
        },
        "second": {
            "camera": "cam_left_wrist", "frame_id": "CURRENT_SECOND_ID",
            "pixels": "8..64 original boundary pixels; physical pairing is NOT required",
        },
        "arm": "left",
    },
    "public_camera_calibration": {
        "frames": {
            "cam_high": "CURRENT_HEAD_FRAME_ID",
            "cam_left_wrist": "CURRENT_LEFT_WRIST_FRAME_ID",
            "cam_right_wrist": "CURRENT_RIGHT_WRIST_FRAME_ID",
        },
    },
    "preview_grasp_pose": {
        "plan_id": "CURRENT_POSE_PLAN_ID",
        "camera": "cam_high",
        "frame_id": "CURRENT_SOURCE_FRAME_ID",
    },
    "stereo_candidates": {
        "first_camera": "cam_high",
        "first_frame_id": "CURRENT_FIRST_ID",
        "second_camera": "cam_right_wrist",
        "second_frame_id": "CURRENT_SECOND_ID",
        "first_roi": [100, 100, 200, 200],
        "second_roi": [100, 100, 250, 250],
        "maximum": 4,
    },
    "stereo_guides": {
        "first_camera": "cam_high",
        "first_frame_id": "CURRENT_FIRST_ID",
        "second_camera": "cam_right_wrist",
        "second_frame_id": "CURRENT_SECOND_ID",
        "pixels": [[100, 100]],
        "second_roi": [50, 50, 250, 250],
    },
    "run_recipe": {"recipe_id": "SELECT_FROM_AVAILABLE_RECIPES", "arguments": {}},
    "inspect_region": {
        "camera": "cam_high",
        "frame_id": "CURRENT_ID",
        "roi": [100, 100, 200, 200],
    },
    "outline_candidates": {
        "camera": "cam_high",
        "frame_id": "CURRENT_ID",
        "roi": [100, 100, 200, 200],
    },
    "fit_circle": {"first": "OUTLINE_ID", "second": "OTHER_CAMERA_OUTLINE_ID"},
    "estimate_body_envelope": {
        "first": {
            "camera": "cam_high",
            "frame_id": "CURRENT_FIRST_ID",
            "outline": "3..64 ordered [u,v] around the COMPLETE visible target body",
        },
        "second": {
            "camera": "cam_right_wrist",
            "frame_id": "CURRENT_SECOND_ID",
            "outline": "3..64 ordered [u,v] around the SAME complete body",
        },
    },
    "fit_rim": {
        "first": {
            "camera": "cam_high",
            "frame_id": "CURRENT_FIRST_ID",
            "pixels": "12..48 distinct [u,v] on the SAME visible circular rim",
        },
        "second": {
            "camera": "cam_right_wrist",
            "frame_id": "CURRENT_SECOND_ID",
            "pixels": "12..48 distinct [u,v] on that rim; no paired order required",
        },
    },
    "run_sequence": {
        "stages": [
            {
                "tool": "plan_relative_pose",
                "arguments": {
                    "arm": "left",
                    "translation": [0, 0, 0.02],
                    "rotation_vector": [0, 0, 0],
                    "pivot_local": [0, 0, 0],
                },
            },
            {"tool": "gripper", "arguments": {"arm": "left", "value": 0, "steps": 24}},
        ],
        "reserve_actions": 40,
    },
    "mark_points": {
        "points": [
            {
                "id": "feature",
                "camera": "cam_high",
                "frame_id": "CURRENT_ID",
                "pixel": [100, 100],
            }
        ]
    },
    "estimate_depth": {"camera": "cam_high"},
    "public_motion_failure_diagnostics": {},
    "calibrate_depth": {
        "camera": "cam_high",
        "table_points": ["table_left", "table_right", "table_near"],
    },
    "locate_points": {"points": [{"id": "feature", "method": "depth"}]},
    "plan_pose": {
        "arm": "left",
        "constraints": [
            {
                "kind": "point",
                "local": [0, 0, 0],
                "target": "feature",
                "offset": [0, 0, 0.08],
            },
            {"kind": "axis", "local": [1, 0, 0], "direction": [0, 0, -1]},
        ],
    },
    "execute_plan": {"plan_id": "RETURNED_ID"},
    "plan_relative_pose": {
        "arm": "left",
        "translation": [0, 0, 0.02],
        "rotation_vector": [0, 0, 0],
        "pivot_local": [0, 0, 0],
    },
    "capture_attachment": {"arm": "left", "points": ["body_a", "body_b", "mouth"]},
    "verify_attachment": {"attachment_id": "RETURNED_ID"},
    "gripper": {"arm": "left", "value": 0.0, "steps": 24},
    "home": {"release": True, "steps": 20},
    "wait": {"steps": 20},
    "verify": {"first": "feature", "second": "destination", "tolerance_m": 0.015},
    "request_capability": {"name": "missing_tool", "reason": "why it is needed"},
    "finish": {},
}


SEGMENTATION_TOOLS = {
    "segment_candidates": {
        "camera": "cam_high", "frame_id": "CURRENT_ORIGINAL_FRAME_ID",
        "box": [100, 100, 200, 200],
        "points": [{"pixel": [150, 150], "label": 1}],
    },
    "inspect_segmentation": {
        "camera": "cam_high", "frame_id": "CURRENT_ORIGINAL_FRAME_ID",
        "segmentation_id": "EXPLICITLY_SELECTED_CURRENT_PROPOSAL_ID",
    },
}

SAM_PROMPT = """
Opt-in segment_candidates uses the pinned local SAM model on ONE current original
RGB view, not simulator segmentation. Supply an explicit original-pixel box and/or
1..8 foreground(1)/background(0) points; point-only prompts need a positive point.
It returns three UNACCEPTED alternatives in model order, never an automatic winner.
Scores are not accuracy probabilities. Select inspect_segmentation explicitly to
see one original-size overlay while retaining ALL original camera views.
At most four proposal calls are allowed per observation. New frames invalidate IDs.
Review semantic identity, complete boundaries and robot/object occlusion against
the originals. High score + one component + no frame clipping + RGB edge support
still CANNOT establish a complete unoccluded body. A visible fragment is not a body.
No SAM result is a metric target, grasp, contact, inner opening or retention proof.
estimate_body_envelope may use each view's {camera,frame_id,segmentation_id,
complete_visible_body:true} INSTEAD OF its explicit outline, only after your
current-original-RGB completeness review. This is an explicit caller assertion,
not a machine-certified property. Clipped, fragmented, over-complex or edge-failing
proposals remain ineligible. The same unchanged body-envelope geometry and original
RGB checks still run; no action is scheduled and no solid/contact model is created.
Do not use a body mask as an inner rim, handle aperture or a hidden surface.
"""


OBSERVATION_BUDGET_PROMPT = """
observation_tool_budget reports the actual remaining tool/model calls for this
decision on the current observation, INCLUDING the query you are answering.
The per-observation and whole-episode limits BOTH still apply; effective_remaining
is their minimum. A plan_pose/plan_relative_pose call only creates a plan:
execute_plan needs a SECOND call unless you already have a valid fresh plan.
Budget perception and diagnostic queries so a needed checked action can still be
executed. Rejected/inspection/verification queries also consume the allowance.
Do not leave a new plan unexecuted on the last allowed query or assume an action
is scheduled merely because planning succeeded. Do not insert purposeless waits
solely to evade the bound. Budget pressure never permits stale targets, skipped
visibility/contact checks, blind relative contact, relaxed guards or forced
release. If no supported continuation exists, preserve safety and the genuine
incomplete result rather than inventing a successful action. No allowance is
automatically increased or reset by this feedback.
"""

CAMERA_POINTING_TOOLS = {
    "plan_camera_pointing": {
        "camera": "cam_right_wrist", "frame_id": "CURRENT_RIGHT_WRIST_FRAME_ID",
        "target": "CURRENT_LOCATED_TARGET_ID", "empty_hand_review": True,
        "tcp_offset": [0, 0, 0], "steps": 12,
    },
}

CAMERA_POINTING_PROMPT = """
Opt-in plan_camera_pointing is a geometric wrist-camera aiming PLAN, not a future
image or visibility planner. It requires a currently located target, current wrist
frame, and your explicit current-RGB review that the camera hand is empty,
unloaded/not supporting an object, clear of contact and already open. Do not
release a held object merely to enable this tool.
Use a target witnessed in another camera; a genuinely visible table patch near
an object may be located with the public table plane for a viewing hypothesis,
but do not pretend an elevated/occluded object pixel lies on that plane.
The tool includes the rigid camera-to-TCP offset, tries four bounded target-axis
twists with unchanged robot/IK preflight, and returns ONE plan_id without motion.
The optional TCP displacement is relative to the current TCP and bounded to0.15m;
choose it only from current observed free-space/hand-sweep evidence. The default
holds the TCP position while changing attitude; the wrist/camera/hand still sweep
space. A below-table rejection is not an instruction to lower further.
The selected point is predicted at the optical principal point if it stays static.
This does NOT certify line of sight, focus, complete-object coverage, scene/payload
clearance, identity, contact or future RGB. Images may be rolled; use calibration,
not image-left/up, for world directions. A rejected bank does not exhaust all poses.
Review the plan and current scene, then use a SEPARATE execute_plan call only when
supported; reserve two calls for planning/execution. After motion inspect actual
fresh originals, remeasure features and reject lost/occluded evidence. The original
target-currentness, RGB, collision, IK, joint, TCP and action checks remain active.
Never mark the predicted principal point in the OLD image as if it were new RGB.
Do not substitute this free-space empty-hand view plan for blind contact recovery.
"""

POINT_PURSUIT_TOOLS = {
    "pursue_point": {
        "arm": "right", "target": "CURRENT_TRIANGULATED_MATERIAL_POINT_ID",
        "offset": [0, 0, 0], "max_steps": 40, "reserve_actions": 80,
        "close": False, "empty_hand_review": True,
    },
}

POINT_PURSUIT_PROMPT = """
Opt-in pursue_point is a bounded local translating-point servo, not a detector,
prediction, re-identification or complete grasp controller. Stage/preorient safely
first: the current grasp-center TCP must be within0.15m of target+offset, and its
current attitude is retained. Explicitly review current material engagement,
payload/opposite-arm corridor and empty-hand approach before choosing close:false.
Every native step re-triangulates the SAME two currently tracked RGB material
points with unchanged camera/currentness, photometric and reprojection checks.
Lost/rebound points, changed selection/binding, frame gaps or excessive motion stop
the operation; no extrapolated/stale endpoint or automatic re-identification is used.
Target jump<=15mm/action, accumulated target travel<=0.15m, robot local TCP
workspace<=0.15m, Cartesian waypoint<=10mm; original IK, joint-step, robot/table/
collision guards and5mm/0.06rad relative TCP arrival checks remain in force.
max_steps is8..120 native actions, further limited by initial kinematic cost+40
and the explicit cleanup reserve. Retargeting NEVER renews this fixed allowance.
This is an action-count bound, not a guaranteed wall-clock capture time.
close:false preserves grip commands and returns at current relative arrival,
requiring fresh RGB review before closure. A SEPARATE explicit close:true request
requires engagement_review:true and current relative arrival already satisfied.
It follows the current point during closure using the original0.15 aperture ramp,
minimum8 actions and original settling tests. It never opens, releases or lifts.
Completed closure/point following does not prove semantic identity or retention.
Review and test-lift with existing checks before transport, then continue the full
native task when supported. Do not use pursuit to bypass a lost static target:
the current two-view material evidence must actually be restored before starting.
"""


def prompt_for(parameters):
    from . import settings

    config = settings(parameters)
    return (
        PROMPT + (SAM_PROMPT if config.get("sam_proposals") else "")
        + (OBSERVATION_BUDGET_PROMPT if config.get("report_observation_budget") else "")
        + (CAMERA_POINTING_PROMPT if config.get("camera_pointing") else "")
        + (POINT_PURSUIT_PROMPT if config.get("point_pursuit") else "")
    )


def catalogue(parameters):
    from . import depth_availability, settings
    from .grasp_preview import availability as preview_availability

    config = settings(parameters)
    depth = depth_availability(config.get("depth"))
    preview = preview_availability(parameters)
    tools = TOOLS
    if config.get("sam_proposals"):
        from .segmentation import availability

        sam = availability(config["sam_proposals"])
        tools = {**TOOLS, **SEGMENTATION_TOOLS}
    else:
        sam = None
    if config.get("camera_pointing"):
        tools = {**tools, **CAMERA_POINTING_TOOLS}
    if config.get("point_pursuit"):
        tools = {**tools, **POINT_PURSUIT_TOOLS}
    return {
        name: {
            "arguments_example": deepcopy(arguments),
            **(
                sam
                if name in SEGMENTATION_TOOLS
                else depth
                if name in {"estimate_depth", "calibrate_depth"}
                else preview
                if name == "preview_grasp_pose"
                else (
                    {"available": bool(parameters.get("tool_recipes"))}
                    if name == "run_recipe"
                    else {"available": True}
                )
            ),
        }
        for name, arguments in tools.items()
    }


PROMPT = """You control a simulated ARX X5 through bounded public-observation tools.
Return exactly {"tool": NAME, "arguments": OBJECT, "reason": TEXT}.
Select one advertised tool at a time; examples describe the argument shape.
available_recipes lists frozen, tested tool compositions with their interface and
argument examples, not their source. run_recipe builds a bounded run_sequence in
an isolated process using only the supplied public state and arguments. Choose a
recipe only when its preconditions hold. Its tests do not establish grasp or task
success. Missing external tools must still be requested for actual integration.
public_camera_calibration is a READ-ONLY getter for the existing public pinhole
camera calibration. Supply a mapping of 1..3 ORIGINAL camera names to exact
current frame IDs. It returns K, optical/world transforms, optical axes and
rigid wrist-camera-to-TCP relations with explicit conventions, not object
poses, object depth or a new sensor. Use this information to reason about a
bounded empty-hand viewing hypothesis when clipping/coverage is the obstacle.
A pixel plus K is only a ray. A future optical-frame hypothesis is not a future
RGB image, an occlusion/clearance certificate or an automatic view planner.
Continue to use fresh original RGB and the existing checked planning/execution
tools; never turn a calibration-only query into permission for blind contact.
preview_grasp_pose is an optional READ-ONLY aid for a current pose plan. It draws
nominal public finger convex hulls at URDF open/closed slide limits in one ORIGINAL
CURRENT camera view. Required original camera images remain available. These are
not measured finger positions, a full hand/wrist model, object geometry, contact,
retention or collision evidence; overlapping projections may be at different
depths. The limits assume empty space; actual object contact can stop fingers
between them, so the closed-limit contour is not a predicted held-jaw position.
A moving wrist camera's future image is NOT predicted. Prefer the fixed
head view for inspecting planned finger placement; use other current views only
with this projection distinction understood. Never mark synthetic lines as
observed material. The latest optional nominal_finger_preview replaces only the
prior preview of that name; it cannot change targets, accept matches or execute
motion. Use it to critique a hypothesis, not bypass uncertainty or motion gates.
Use the selected skill's task, frozen successful program and textual experience
as procedural references, not old coordinates. You cannot execute Python, read
files, change physics, inspect hidden state, or decide the official result.

Ground all clicks in the CURRENT original-size RGB camera and exact frame_id.
inspect_region returns a transient enlarged crop in the NEXT query's images; its
ROI and dimensions are in state.inspection_views. Convert crop pixels back to the
original image; crop aliases are not calibrated cameras. No crops are saved.
outline_candidates extracts actual near-elliptical RGB edges inside a selected
ROI. Select the SAME physical circular boundary in TWO cameras, then fit_circle
reconstructs a horizontal circle with pixel sensitivity and edge checks. Use its
target_id in plan_pose as a center target during the same observation. New frames
invalidate measurement IDs; during motion current RGB must support the boundary.
This cannot certify the inner opening, a tilted circle, identity, grasp or flow.
If automatic outlines miss a visible circular lip, fit_rim accepts 12..48
manually selected original-image pixels per current camera. Select the SAME
physical boundary in both views, not an interior shadow, ellipse center or
mixed inner/outer edges. Pixel sets need broad angular coverage, not paired
ordering. This estimates a circular rim within 20 degrees of world-up using
calibration, reprojection, uncertainty and unchanged RGB-edge gates. It returns
a frame-local target_id, center, radius and estimated normal; NEVER treat the
fit as an automatically certified inner opening, clearance, containment or
retention. Selection uncertainty omits systematic boundary/calibration bias.
Use fresh evidence after a motion; motion toward a rim target rechecks both
current RGB boundaries. If occluded or rejected, reobserve or stop safely.
fit_sphere reconstructs the SAME approximately spherical surface from two
calibrated CURRENT RGB silhouettes, without assumed radius, height or depth.
For each original view provide 8..64 distinct outer-boundary pixels covering
at least270 degrees, or use outline_id instead of pixels for one current
outline_candidates result. Do not use an internal reflection, flat circular
rim, shaft junction, capsule, or an occluding edge as the sphere silhouette.
Views select tangent boundaries, NOT corresponding physical surface points.
It checks original-image edges, model residual, parallax and bounded1px
independent/coherent perturbations. The reported sensitivity is not total
accuracy; shape/calibration bias remains unknown. The current sphere center
target_id expires at the next observation; it is derived geometry, not an
optical-flow point or a valid material feature for capture_attachment.
An optional arm returns tcp_local_center under CURRENT public FK, not a
verified attachment. Refit after grasp and inspect actual retained motion.
For a spherical striking head, center/radius can inform a surface approach
only with valid current key geometry, retained grip, enough uncertainty margin
and the normal IK/collision/action gates. It does not identify which sphere is
the mallet, automatically grasp it, certify contact, or play the instrument.
Static motion plans using a sphere target recheck its current RGB silhouette;
never bypass a lost-boundary rejection with an unobserved relative motion.
When sphere selections fail the original-edge gate, boundary_diagnostics
identifies EACH current camera, unsupported point indices and bounded nearby
edge proposals. These are diagnostic candidates, NOT automatically corrected
or accepted sphere pixels; an edge may belong to a reflection, shaft, occluder
or background. Inspect the original RGB before reselecting. A view already
passing its edge-fraction gate need not be edited while its frame is unchanged.
All sphere-fit, freshness, geometry and motion checks remain unchanged.
Sphere edge evidence retains luminance and original R/G/B channel gradients
under the same Canny blur/thresholds. This preserves real colored boundaries
that grayscale conversion can lose; it is a versioned perception change, not
looser point-distance, coverage or model gates. Color-channel edges can still
belong to highlights, texture or unrelated objects and require original-RGB
semantic review. No masks, depths or object poses come from the simulator.
estimate_body_envelope is an alternative shape diagnostic when a visible body
has few matching texture features or its head is offset from its support base.
Provide COMPLETE, unoccluded ordered body outlines in TWO current cameras;
do not mix objects, enclose the hand/background, or guess hidden boundaries.
It checks original RGB edges and intersects calibrated silhouette cones in a
bounded 2mm voxel grid above the public table. It returns conditional outer
bounds and lower cross-section center/span hypotheses, NOT a solid object model,
contact surface, circularity/vertical-axis proof or trusted target_id.
The 2px outline-dilation result is a sensitivity probe, NOT total uncertainty.
Finite voxel sampling can miss thin parts; these are not certified collision
bounds and do not replace checking all visible protrusions along a motion.
Outlines must show a table-supported body; do not apply this assumption after a
lift. A visual hull may contain empty space and overestimate true dimensions.
Use the estimates to choose bounded viewing, approach and closing-axis
hypotheses; do not treat a beak, face point or projected silhouette center as
the body axis. Compare candidate spans with aperture evidence actually available
from current RGB/public calibration and with fingertip/table limits. No exact
inner jaw gap is exposed by this tool; do not invent one from normalized gripper
state or from the robot collision-sphere width. Inspect the open fingers and
body from useful current views before a bounded contact attempt. Still require
current RGB before contact, unchanged
motion/clearance checks, and actual visual retention after a short lift.
The output expires with its frame IDs; it neither schedules actions nor
replaces post-contact observation or a full grasp/retention controller.
Prefer independently checked multiview geometry over guessed heights or raw depth.
stereo_candidates proposes up to 1..12 reciprocal SIFT descriptor matches
(default8) inside two selected current RGB ROIs, each 12..320 pixels per side.
Use a small ROI around the SAME intended object/material surface in both views.
It applies the existing bounded descriptor-ratio and calibrated epipolar checks
and returns original-image pixels with two annotated copies of the original RGB.
It accepts NO correspondence, creates NO point/target/plan and consumes no native
action. Inspect the unmodified views to confirm the SAME physical material
feature before using mark_points and the unchanged locate_points checks.
Low residual or a good descriptor ratio is not accuracy, identity or grasp proof:
repeated patterns, uniform faces, silhouette-centered descriptors and occlusion
can yield wrong proposals. Empty results are legitimate uncertainty. Do not
force a match or lower thresholds. Use one bounded proposal/review pass rather
than repeated unsupported corner guesses; obtain another useful view if needed.
The annotated copies retain original dimensions; normal image/request byte caps
still apply and required camera views are not dropped.
stereo_guides gives normalized epipolar lines and RGB corner proposals in a
bounded second-camera ROI (at most 320 pixels per side) for 1..8 first-camera
pixels. Optional other_pixels reports existing selections' distances to the lines.
It consumes no native action, accepts only exact CURRENT frame IDs and never
accepts a correspondence or creates a 3D point. Inspect the original views to
select the SAME physical feature; mark_points and the unchanged triangulation
checks are still required. Nearby corners on a line may belong to other objects
or repeated slots. Empty proposals are valid uncertainty, not permission to
invent a match. Use one guided correction rather than repeated blind point pairs.
Track IDs identify visual features, not privileged object poses. Lost tracks
cannot be used: reobserve after a bounded wait or re-mark from the current image.
Use several salient textured points, not the center of a textureless surface.
No sensor depth is supplied. estimate_depth lazily runs the configured metric
monocular model on a current camera; it returns estimates, not measured truth.
If scale looks biased, mark at least three well-separated points on the visibly
bare TABLE surface (not objects), then calibrate_depth. This fits one scale
using the public table plane, expires with the current image, and still does
not correct local model errors or establish metric accuracy.
locate_points accepts: {id,method:"depth"}; {id,method:"plane",height_m:HEIGHT_ABOVE_TABLE};
or {id,method:"triangulate",other_point:ID}. Triangulation requires current views
of the SAME physical point from different calibrated cameras. Plane heights are
explicit hypotheses. Depth estimates expire after movement and must be refreshed.
Never convert relative depth or guessed scale into claimed metric truth.
For an UPRIGHT object still on the table, locate_points also accepts
{id:TOP_CENTER,method:"upright",base_points:[BASE_LEFT,BASE_RIGHT]}.
Mark two symmetric visible bottom contact points and the top center in the same
camera. Public table calibration plus a vertical-axis fit estimates its height.
These assumptions must hold; do not use this binding after lifting or tilting.
This can avoid repeated biased monocular depth calls for a simple upright vessel.

plan_pose uses local TCP-frame anchor points/vectors and currently located target
IDs. Point: {kind:"point",local:[x,y,z],target:ID,offset:[dx,dy,dz]}.
The TCP origin is ALREADY the grasp center. Public robot_state ee_pose is the
wrist pose, not the TCP. The wrist-to-grasp offset is already included by FK.
Use local:[0,0,0] for a grasp-center target. Do not add another wrist/finger-length
offset (such as0.12 or0.145 m). Nonzero local points must represent an explicitly
estimated physical feature relative to the grasp center, not duplicated offsets.
Current TCP matrices and the public table plane are in state.public_robot_geometry.
Axis: {kind:"axis",local:[x,y,z],from:ID,to:ID}, or a public world direction
instead of from/to. Plane: {kind:"plane",local:[x,y,z],points:[ID,ID,ID]}.
Offset is optional and in world meters. The local TCP +X points along the
fingers, +Y is the jaw closing axis. Unconstrained orientation stays near the
current pose. Bounds and IK can reject a geometric fit. Paths are not a complete
scene-collision model: visually check obstacles and use short clear waypoints.
Execute only a fresh returned plan ID; no intervening motion or point rebinding.
For short free-space moves, plan_relative_pose uses CURRENT measured TCP directly,
without an unrelated visual anchor: translation is world meters (norm<=0.15),
rotation_vector is a WORLD-axis rotation vector in radians (norm<=0.6), and
pivot_local is the TCP-local point about which to rotate (norm<=0.3 m).
Use pivot_local=[0,0,0] for a plain TCP move. A measured/estimated bottle mouth
relative to the grasp center can be the pivot: its world path then follows only
translation while the wrist rotates around it. The full interpolated path, not
just its endpoint, preserves that pivot. plan_pose also accepts optional
pivot_local for the same interpolation. No pivot implies no bottle attachment;
object rigidity and clearance still require visual checks.
Each move preserves gripper commands. Open/close with gripper, not plan_pose.
plan_pose and plan_relative_pose accept optional steps (1..120, default12).
For short clear motions steps=4..6 saves minimum holds; joint-step limits still
expand the real path. gripper close accepts steps>=8, open>=6, and still waits
for measured settling; do not spend24 steps opening an already free hand.
When public_robot_geometry.cross_arm_guard reports collision spheres, crossing
the centerline is checked against the CURRENT other arm and public self-collision
geometry instead of a fixed side boundary. Move the empty opposite arm clear or
home BOTH when empty. It is not scene-object collision planning; fingers are
bounded by conservative envelopes. Never attempt a handover through overlapping hands.
Approach from clearance, descend, close, inspect fresh wrist RGB, then lift only
when evidence supports holding the intended object. Move completion is NOT grasp
confirmation. A depth/plane estimate cannot certify fine placement or clearance.
home releases BOTH grippers and returns to original joints: never call it while
holding an object unless release is intended and safe. wait consumes native steps.

After EVERY motion inspect the new images and tool verification. verify compares
two current 3D points and returns pass/fail/unknown plus provenance; estimated
geometry remains conditional evidence. Verification is never an official verdict.
Use wrist images to check grasp/release, not gripper command values as proof.
capture_attachment records 3..8 currently localized, non-collinear features ON
the held object in the current TCP frame. After a small lift/rotation,
verify_attachment compares current tracked RGB points against their predicted
rigid motion using public cameras/TCP. It returns visually_consistent,
inconsistent or unknown, not grasp proof. A translation-only test cannot certify
rotation resistance. Wrong metric geometry can also cause inconsistency; lost
or re-marked features cannot be used as continuous evidence. If the object is
occluded, obtain fresh visual evidence instead of claiming rigid retention.
Use public_motion_failure_diagnostics after tracking problems to compare public
joints, TCP, desired waypoint, path cursor and failure. operation.steps is only
a nominal minimum, not the actual duration: Cartesian paths expand for joint
distance. Do not infer a timing bug from that field alone. Endpoint IK does not
prove collision-free motion; a stalled path can require a different orientation
or route rather than more time. Do not ask to disable collisions or constraints.
Do not loop on unchanged errors. Request missing external capabilities explicitly;
they remain unavailable until integrated, not magically enabled by a request.
finish only holds position until the native evaluator terminates; it does not
declare success. If a sub-step fails, reobserve and recover within remaining budget.

run_sequence groups 1..8 bounded stages without another model call between them.
Allowed stages: plan_pose, plan_relative_pose, gripper, wait, verify and
verify_attachment. Plans execute automatically from fresh observations and TCP
verification is required after each motion. Verification failure, estimated or
unknown evidence, or a recovery-reserve limit stops the whole sequence. IDs may
expire between stages; do not reuse frame-local circle geometry after a move.
Gripper changes MUST be the final stage: the sequence returns needs_visual_review
and you must inspect the current grasp/release before lifting or transporting.
Use reserve_actions to keep steps for recovery. This is not complete collision
planning or visual contact control; use single stages for uncertain contact/flow.
"""
