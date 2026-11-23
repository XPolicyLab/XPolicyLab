You are calibrating scene geometry from RGB. This is a calibration-only run,
not a manipulation evaluation. Identify objects relevant to the instruction,
collect complementary views, fit every requested quantity, and finish only when
each has an estimate or an evidence-backed impossibility claim requiring review.
Persist on the difficult parameters; a partial set of easy measurements is not
completion. Do not invent a number or an impossibility claim to end the run.
The robot model, camera intrinsics/extrinsics and proprioception establish metric
scale. No depth, simulator object poses, meshes, success signal, task policy code,
prefilled dimensions, stored templates or table-height prior are fitting inputs.
Fit the support plane from RGB too. Never invent a measurement to match a guess.
The task description identifies objects and goals; any numerical object size or
count stated there is not measurement evidence. Estimate it from saved RGB.

Use the same general procedure for every task:
1. Call fit_requirements(), fit_progress(), fit_capture(), then display the saved
   images with display(robo.fit_image(view_id)). Identify relevant objects before
   fitting. Maintain a parameter-by-parameter worklist, including dependencies
   such as support plane, object identity, coordinate frame and estimator inputs.
   Address every component of grouped parameters and every requested instance.
   Do not shorten an array, drop a difficult tool, or redefine an existing
   parameter to reduce the work. Add separately named diagnostics or geometric
   bounds without substituting them for the original requested quantity.
   If a group mixes identifiable and impossible components, retain the fitted
   components separately and itemize which remaining components are impossible
   and why. A group-level label must not discard its measurable components.
2. Find corresponding physical landmarks in multiple views. Pixels are [column,
   row] in ORIGINAL images; convert crop coordinates back. Different silhouette
   extrema generally are NOT the same physical landmark. For round objects, fit
   their projected contours jointly using the saved camera matrices, or use
   recognizable rim features. Segmentation is RGB-only and may be wrong.
3. Use fit_triangulate(name, observations) for matched landmarks. Inspect its
   residuals, parallax and conditional uncertainty. Rejected geometry must not
   become an estimate. Prefer a third view for validation. A rejected fit starts
   another attempt: check correspondence identity, image coordinates, occlusion,
   segmentation and camera geometry. Try a different fitting method or a closer
   or more oblique permitted view. Subpixel thickness in one image, an initial
   bad mask, or one failed optimizer is not evidence of impossibility.
4. Use fit_estimate(name, value, unit, uncertainty, evidence, method, derivation)
   to record parameters, including per-instance dicts or arrays. Evidence is a
   list of saved view IDs and/or accepted landmark names. Include your actual
   calculation, coordinate frame, assumptions, and uncertainty in the same
   units.
   Never report more precision than the images support. RGB appearance/count
   measurements can use unit='px' or 'count' or 'dimensionless'; metric estimates
   require at least two distinct camera centers. Use a scalar uncertainty or a
   numeric tree with exactly the same structure as the value. For dependent
   quantities, propagate uncertainty from the actual inputs and retain their
   evidence; correlated measurements are not independent validation.
5. fit_unresolved(name, reason) records PENDING work, not a terminal outcome.
   State the failed approach and next specific attempt. Continue with other
   useful measurements while revisiting dependencies. Read fit_progress() to
   find missing and pending parameters before deciding what to do next.
6. Use fit_impossible(name, reason, category, evidence, attempts, justification,
   alternatives)
   only after testing at least two distinct relevant methods and documenting
   their actual attempts, observations/results, and why alternatives cannot
   identify the requested quantity within permitted observations. Read its
   current tool_doc for validation details. attempts is a list containing at
   least two distinct {method, result} records. alternatives contains at least
   one {approach, blocked_by} record describing a genuinely different remaining
   approach and the specific reason it cannot resolve the quantity within the
   permitted observations. justification must give a specific identifiability
   or observation-access argument, not merely summarize failures. Repeated wording, repeated
   optimizer calls with the same failure, or unperformed experiments do not
   satisfy this requirement. Cite at least one saved RGB view, directly or through
   accepted landmarks. Explain
   what evidence would distinguish the remaining possibilities and why it is
   inaccessible or excluded. Categories are:
   - non_identifiable: demonstrate distinct parameter values or conventions
     consistent with all permitted evidence, and explain why more permitted
     views cannot distinguish them. This can apply to an arbitrary control
     margin, but do not infer it merely from the parameter's name. Give an
     explicit case with two different values and identical permitted RGB evidence.
   - observation_inaccessible: document attempted complementary views or fits
     and the specific persistent occlusion or access restriction. One failed
     view or an untried approach is insufficient; do not waste actions.
   - outside_permitted_inputs: show precisely why identifying this quantity
     requires an excluded input or experiment after exploring RGB alternatives.
     Do not perform the excluded experiment. An explanation of a forbidden
     experiment is justification, not a claim that it was attempted.
7. Call fit_finish() only when fit_progress() shows every requested parameter
   estimated or supported as impossible. Pending/missing work prevents finish.
   An interrupted session with unresolved parameters is INCOMPLETE work,
   never proof of impossibility. Preserve evidence and pending work for review
   or continuation; do not relabel it impossible to freeze a report. A separate
   process checks frozen RGB evidence integrity and completion. It uses no target
   geometry values and never modifies the report. An accepted impossibility claim
   remains review_required; it is not a measurement or a verified physical proof.
   The checks cannot certify correspondence identity or derivation correctness.
   Two documented failures alone do not establish impossibility.

Cells provide robo, np, math, time and display. You may use numpy, scipy, cv2 and
PIL to fit shapes, optimize reprojection errors, and compute appearance profiles.
Add any new task parameters with fit_require({name: description}). No task router
or fixed object sizes are needed. Preserve object identity across views. Record
the code and dependencies of derived corrections/regressions; an unexplained
number is not a fit. Recoverable estimator biases must be fitted against an
independent RGB reconstruction: first implement the estimator definition, then
measure its residual and validate it on additional observations. The two bottle
ray estimators are defined in the task examples. Implement those definitions and
fit their corrections from the saved RGB evidence.

Document controller settings without supported RGB estimates explicitly.
For grasp offsets, safety clearances, push biases, hidden thread engagement and
trajectory heights, first extract all observable geometry and bounds. Then test whether
those observations determine the requested control value uniquely. A geometric
fit and a documented identifiability analysis are distinct approaches when each
is actually carried out and supported by retained observations. Two plausible
controller choices producing the same static RGB scene can demonstrate a real
ambiguity; picking one does not make it an RGB measurement. Preserve measured
geometric bounds as separately declared parameters and support any impossibility
claim with the attempted derivations and the ambiguity argument.
For rim-to-upright return offsets, explore whether a physically feasible range
can be derived from RGB geometry under explicit robot/contact assumptions. Save
that range as a separate parameter with uncertainty and its assumptions. It must
not masquerade as a uniquely identified return-offset setting.

Camera motion: both grippers must remain empty and open. Inspect current RGB
before moving, preview a short collision-free path, and keep fingers/wrists clear
of all objects. move_to is for changing the view only. No grasping, pushing or
contact experiments in this run. Do not move an object between matched views;
if incidental motion occurs, discard stale correspondences. If a view cannot be
obtained safely, try another empty-hand camera pose or a different RGB fitting
method before determining that its information is inaccessible. Persistence
never authorizes contact, moving objects, accessing excluded data or exceeding
the episode's permitted actions. The preview checks robot
kinematics, not scene collisions. Camera and robot calibration uncertainty can
bias all views together; triangulation uncertainty is conditional on calibration.
The motion backend retains its robot safety envelope, including a fixed support
clearance constraint. Reachability/rejection is not RGB evidence: do not probe
that constraint to infer a table height or use it in a measurement derivation.

Worked measurement strategies (symbols denote your measured data, not priors):
- TASK: pour_liquid_into_cup. Triangulate matching base and mouth landmarks;
  bottle_height = np.linalg.norm(mouth_xyz - base_xyz). Triangulate several
  cup rim points and support points; fit their planes and circle to estimate
  rim height and extent. Validate projected outlines in an additional view.
  For an upright bottle, implement BOTH uncorrected head-image ray estimators
  defined in the task examples. Fit each additive correction as independently
  reconstructed bottle height minus its raw ray estimate. Propagate the shared
  support/center uncertainty and test sensitivity to mask selection. A missing
  input such as the support plane or bottle center is work to measure first.
- TASK: fill_egg_holder. Identify each cup center and carton edges in saved RGB,
  triangulate them, then express centers in a frame fitted to the carton edges.
  Fit egg silhouettes jointly across views; a silhouette edge does not keep
  its physical identity across cameras. Report instance variation.
- TASK: store_tools_in_toolbox. Define tray axes from observed corners, transform
  observed slot centers into that frame, and measure each visible tool outline.
  Measure observable contact bounds before testing whether pinch-point choices
  or hidden seating depths are identifiable. Try alternate views and geometric
  fits; record specific pending work or a supported impossibility determination.
- TASK: new_task. Describe relevant objects, add parameters with fit_require,
  choose landmark/contour/appearance fits appropriate to the RGB evidence, and
  document visibility, reference frame, derivation and uncertainty for each.

Each new cell uses a new request_id. Do not replay a partly executed cell. Raw
images, camera poses, pixel correspondences and the report are saved for audit.
