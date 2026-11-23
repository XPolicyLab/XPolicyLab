"""Value-free measurement requests.

These are measurement requests and value-free estimator definitions. Unknown
tasks use the same tools and declare parameters with fit_require(). Missing
measurements remain pending until estimated or supported as impossible.
"""

# Insertion order is also the batch launch priority: heavy, moderate, then light.
TASKS = {
    "pour_liquid_into_cup": ("heavy", {
        "bottle_height": "Bottle base to mouth along its axis (m).",
        "rim_height": "Cup support to rim (m).",
        "cup_extent": "Cup outer footprint diameter including its visible extent (m).",
        "ray_corrections": "[located-center-ray additive correction, base-inferred-center-ray additive correction] (m). Implement the exact uncorrected estimators in POUR_RAY_ESTIMATORS below; each correction is independent multi-view upright bottle height minus its raw estimate. Fit missing support/center inputs first, then validate bias and uncertainty.",
        "spout_radius": "Bottle mouth/spout radius (m).",
        "return_offset": "[signed-x magnitude, y magnitude] of rim-to-upright TCP return offsets (m). Derive observable placement/contact bounds, then test whether these fix the selected control offsets; justify any ambiguity using distinct choices consistent with RGB. Do not perform a grasp/return experiment.",
    }),
    "pour_by_language": ("heavy", {
        "bottle_height": "Height of each bottle (m); report variation across bottles.",
        "placement_z": "Bottle bottom placement target world z including clearance (m); a control choice, not simply support height.",
        "mouth_z": "Standing bottle mouth world z (m).",
        "bowl_rim_radius": "Bowl rim circle radius (m).",
        "bottle_mid_z": "Bottle localization reference-plane world z (m); geometric center and estimator reference need not agree.",
        "bowl_mid_z": "Bowl localization reference-plane world z (m); geometric center and estimator reference need not agree.",
        "pour_heights": "Pour trajectory heights; control choices require dynamic RGB evidence, not static shape.",
    }),
    "fasten_screws": ("heavy", {
        "threading_heights": "Successive pre-thread approach nut-reference world z values (m). Fit visible bolt/nut geometry using complementary views, then test whether geometry determines each selected approach stage and hidden engagement; document alternatives before any impossibility claim.",
        "bolt_cap_z": (
            "World z at the center of the exposed upper shaft-end planar face (m); "
            "return one numeric scalar per visible bolt keyed by identity. "
            "Record hex-head, bevel and shoulder planes in separate diagnostic parameters. "
            "Identify the planar top-face rim in RGB and distinguish it from the outer bevel, "
            "thread shoulder and shading edges before selecting samples. Fit a consistent "
            "physical boundary across views; if a bevel is visible, model its distinct surfaces "
            "or exclude its boundary from the planar-face fit. Inspect projected samples on "
            "the original RGB, validate the face and its height in an independent view, and "
            "include boundary-selection and view sensitivity in uncertainty."
        ),
        "bolt_height": (
            "Full bolt axial height (m), including the hex head: distance along the measured "
            "shaft axis from its intersection with the lower supporting face of the hex head "
            "to the center of the exposed upper shaft-end planar face defined by bolt_cap_z. "
            "Reconstruct the lower supporting plane from the visible bottom perimeter and "
            "support contact, and fit the shaft axis and upper face from RGB. For an upright "
            "bolt this equals upper-face world z minus lower-support world z; for a tilted "
            "bolt retain the axial distance and record its orientation separately. Return one "
            "numeric scalar per visible bolt keyed by identity. Record threaded-shaft length "
            "above the hex head separately. Validate both endpoint surfaces in independent "
            "views and propagate their shared uncertainty."
        ),
        "nut_root_z": "Nut model-reference center world z (m). Fit visible top/bottom faces and test candidate reference definitions; an arbitrary unseen model origin requires an evidence-supported identifiability argument, not a guessed center.",
        "nut_radius": "Nut outer circumradius (m).",
    }),
    "fill_egg_holder": ("heavy", {
        "egg_size": "[axial length,transverse diameter] (m), aggregate visible eggs; record per-egg variation separately.",
        "carton_cup_layout": "Cup centers [[x,y],...] in carton frame (m), ordered back-left/back-right/front-left/front-right; measure count too.",
        "carton_half_depth": "Half the carton base depth (m).",
        "hinge_lid_geometry": "[hinge y offset,hinge height above support,lid length] (m).",
        "grasp_z": "Grasp contact world z; geometry constrains a range, not a unique control target.",
        "basket_radius": "Basket detection search-region radius prior (m); operational ROI, not a unique physical outer radius.",
    }),
    "fill_pen_holder": ("heavy", {
        "holder_height": "Holder base-to-rim height (m).",
        "tool_sizes": "Aggregate observed physical {length,width,thick} (m) keyed marker/pen; report the aggregation rule, instance count and separate per-instance variation. Do not alter an RGB measurement to match a round number.",
        "kind_offsets": "Per-kind additive [length,width] corrections (m), keyed purple_marker/blue_marker/black_pencil/white_pen/yellow_black_pen, AFTER the measure_blob and view-stretch stages defined in PEN_MEASUREMENT_DEFINITIONS. Fit the unknown shared view-stretch coefficients jointly with kind offsets against independent multiview RGB dimensions; a residual of a newly chosen PCA estimator is a different diagnostic. Retain fitted present-kind components separately if an absent kind prevents the complete group.",
        "appearance_templates": "Per-kind 20-by-6 longitudinal RGB profiles using the exact projection, bins, 0.035 m width normalization and HSV classes in PEN_MEASUREMENT_DEFINITIONS. The normalization is an algorithmic scale, not a tool size prior. Record endpoint orientation, masks, view variation and uncertainty. Never reuse stored templates or fabricate an absent-kind profile.",
        "insertion_depth": "Insertion depth below rim; interior floor may be occluded and operational margin is a control choice.",
    }),
    "store_laptop_and_headphones": ("heavy", {
        "lid_top_z": "Open laptop lid top world z (m).",
        "laptop_mid_z": "Closed laptop front-lip pinch-reference world z (m), used after closing the lid on its riser. Reconstruct base/lid thickness and closure geometry, then distinguish the physical closed-body midpoint from the selected pinch plane. The open assembly's bounding midpoint is a different quantity.",
        "laptop_size": "Laptop planar [width,depth] (m).",
        "hinge_local": "Hinge [x,y,z] relative to laptop top_center in basis(axis,closing,+z) (m); define visible origin and axes.",
        "stand_z": "Headphone stand coordinate-frame origin world z (m), used to express controller offsets; no physical defining landmark is specified by that use. Measure stand bounds and plausible origins, and test whether RGB identifies the selected origin. The saddle contact crest is a separate quantity.",
        "cradle_outline_z": "Headphone stand upper wooden saddle ('cradle') silhouette projection-reference plane world z (m), not the separate laptop rack. Reconstruct the saddle surface/bounds and assess the effective horizontal plane used to back-project its outline; do not substitute the maximum contact-crest height. See LAPTOP_MEASUREMENT_DEFINITIONS.",
        "cradle_dims": "Headphone stand wooden BASE plate footprint [length,width] (m). The base rectangle is the quantity, despite the cradle name. Measure its outer outline on the fitted base-top plane; this is not the small laptop rack or the upper saddle.",
        "stand_base_height": "Headphone stand wooden base top above local support (m); report bevel/relief and the surface convention used for outline projection.",
        "pixel_model": "Numeric regression dict edge_centre_col (px), edge_mm_per_col (mm/px), focal_px (px), left_mm_per_px (mm/px), right_px_per_mm (px/mm). These describe the held-laptop hinge-strip/face-width estimator in LAPTOP_MEASUREMENT_DEFINITIONS, not arbitrary depth/f scales or left/right cameras. Fit its stated conditions or substantiate non-identifiability under empty-gripper RGB restrictions.",
    }),
    "store_tools_in_toolbox": ("heavy", {
        "place": (
            "Dict of seated tool-outline reference points [tray x,tray y] (m), keyed "
            "hammer/wrench/pliers/tape_measure. The tray origin is the midpoint across the "
            "side edges on the front wall's foot line where the wall meets the support; "
            "+x runs along that front line with nonnegative world-x component, +y runs into the tray, "
            "and +z is upward. With yaw theta counter-clockwise viewed from above, "
            "world_xy = origin_xy + R(theta) @ [x,y], where R is the planar rotation from "
            "world +x to tray +x. The requested point is the tool outline-origin defined "
            "under pinch_points after registration in its receiving geometry. Measure "
            "recess boundaries and support relief, then register the reconstructed tool "
            "outline and orientation; retain fitted tool yaw and alternative registrations "
            "separately. A receiving-region bounding-box center alone does not define this "
            "point. Validate registration in independent oblique RGB views and preserve "
            "unresolved pose ambiguity rather than substituting another center convention."
        ),
        "tool_sizes": (
            "Dict of recognition-outline [length,width] (m), keyed "
            "hammer/wrench/pliers/tape_measure. Length is longitudinal y extent and width "
            "is transverse x extent in the outline frame defined under pinch_points. "
            "The recognition estimator backprojects the head-view filled RGB silhouette "
            "onto a declared horizontal plane through the grasp-reference height. Measure "
            "the silhouette and relevant height from RGB, and record the plane, its "
            "uncertainty, and resulting outline extents. Save independently reconstructed "
            "physical extents separately because a thick object's projected outline on "
            "one plane need not equal its physical dimensions. If the selected reference "
            "height is not identifiable, address that estimator dependency explicitly "
            "while retaining the observable physical measurements."
        ),
        "tool_widths": (
            "Dict of transverse contact widths (m), keyed hammer/wrench/pliers/tape_measure, "
            "at the selected pinch_points and contact height. Measure opposing physical "
            "surfaces in complementary RGB views; keep separate handles and intervening "
            "gaps distinct. Declare the selected section and finger-closing direction. "
            "When the controller's contact choice is not identifiable, retain widths at "
            "explicit candidate sections separately and support the remaining ambiguity."
        ),
        "pinch_points": (
            "Dict of selected pinch offsets [local x,local y] (m), keyed "
            "hammer/wrench/pliers/tape_measure. Local x is transverse and local y is "
            "longitudinal. Define the outline frame from a filled RGB silhouette "
            "backprojected onto the declared grasp-reference plane: compute its planar-area "
            "weighted centroid and covariance, choose the major principal axis with "
            "nonnegative world-y component as +local y, and choose +local x so the basis "
            "is a counter-clockwise planar rotation R(theta) of world axes. If the "
            "principal-axis spread ratio sqrt(largest/smallest covariance eigenvalue) is "
            "at least 1.6 (a dimensionless estimator rule), "
            "shift the centroid along local y to the midpoint of the two longitudinal "
            "extrema; keep its transverse area-centroid coordinate. Otherwise use world "
            "axes and the area centroid. This is the tool outline-origin: "
            "pinch_world_xy = outline_origin_xy + R(theta) @ [local x,local y]. "
            "Measure candidate contacts and record their geometry in this frame, then "
            "test whether RGB identifies the selected controller contact. Multiple "
            "geometrically valid choices require an evidence-supported ambiguity argument."
        ),
        "seat_heights": (
            "Dict of seated-minus-table-resting vertical offsets (m), keyed "
            "hammer/wrench/pliers/tape_measure, applied to the same tool grasp reference; "
            "positive raises that reference. The controller applies this offset to the "
            "TCP z at grasp closure while maintaining the grasp. Define the tool reference "
            "and placement orientation, fit visible tray/slot/tool contact geometry, and "
            "derive the reference's vertical transfer conditional on that registration. "
            "For the same rigid reference and unchanged orientation, a constant tool-to-TCP "
            "vertical offset cancels in seated-minus-resting translation; changing that "
            "reference alone is not an ambiguity witness. If orientation changes, derive "
            "the reference transfer using the measured rotation instead. "
            "Try complementary oblique views and support-contact bounds. Keep physical "
            "transfer, unseen underside/contact ambiguity, and any selected controller "
            "margin distinct; support residual ambiguity rather than guessing a seat offset."
        ),
        "tray_depth": (
            "Tray front-to-back depth (m): distance along tray +y from the front wall's "
            "foot line on the support to the far edge of the lit upper rim. Use the "
            "front-foot origin and axes defined under place; reconstruct both visible "
            "edges at their own fitted heights. Record upper-rim rectangle depth separately "
            "if the molded wall offsets its front edge from the foot line."
        ),
        "rim_z": "World z of the lit upper tray rim (m); fit its visible top edges across views and separate wall-foot/support height.",
    }),
    "play_xylophone": ("heavy", {
        "mallet_head_z": "Resting mallet head center world z (m).",
        "key_plane_z": "Horizontal world-z plane used to backproject xylophone key RGB features (m). Reconstruct actual key surfaces and ridge variation separately, then document how the effective projection plane is selected; a curved key need not have one surface height.",
        "strike_height": "Controller strike target above estimated base (m); distinguish contact geometry from impact choice.",
        "shaft_length": "Mallet shaft-axis extent from the reconstructed mallet-head center to the visible handle butt (m), used for choosing a grasp distance from that head origin. Define and fit those endpoints; any concealed material endpoint inside the head is a different quantity and must be recorded separately.",
        "strike_heights": "[hover,dip] TCP world-z offsets (m) from the intermediate strike target, which is estimated xylophone-base world z plus strike_height. Fit visible mallet/key contact geometry and declare the hand-to-mallet relation, then demonstrate whether these controller offsets are uniquely determined; measured head-center or tabletop-relative contact bounds are separate diagnostics.",
        "key_pitch": "Adjacent key center spacing (m).",
    }),
    "deposit_coin": ("heavy", {
        "coin_radius": "Coin radius (m); obtain a close face view.",
        "coin_thickness": "Coin axial thickness (m); needs a close oblique/edge view.",
        "holder_size": "Holder length, width, height (m).",
        "holder_top": "Holder top above support (m).",
        "pickup_offset": "Selected fingertip world-z offset above the reconstructed coin center (m) during upright holder pickup. The target is fingertip_z = max(coin_center_z + pickup_offset, holder_top_z + clearance). Measure exposed coin/holder/gripper geometry and feasible contacts, then distinguish geometric bounds from the controller's selected offset.",
        "clearance": "Selected minimum fingertip-to-holder-top vertical clearance (m) during coin pickup, used in the target formula under pickup_offset. This is distinct from coin-to-bank-slot insertion clearance. Reconstruct pickup contact bounds and support any residual control-choice ambiguity with the same-RGB/different-target argument.",
    }),
    "make_kong": ("moderate", {
        "tile_thickness": "Mahjong tile thickness (m).",
        "tile_height": "Mahjong tile long dimension (m).",
        "rack_count": "Count visible rack tiles; do not assume a fixed count.",
        "pair_indices": "Identify matching tiles and their zero-based, left-to-right rack indices from RGB; report the integer index array with unit='count' (the index/count representation expected by this parameter).",
        "rack_rows": "[start,stop_exclusive] head-image row detector ROI (px); an operational search window, not a unique visible object bound.",
    }),
    "organize_table": ("moderate", {
        "overpush_bias": "Keyboard over-push bias is a control choice, not a static dimension.",
        "contact_height": "Keyboard-push TCP world-z offset (m) above the estimated lower support plane center_z-height/2. It is not the fingertip's contact height. Reconstruct keyboard/gripper contact geometry and its TCP relation, and distinguish contact bounds from the selected controller offset.",
        "contact_dx": "Positive half-separation (m) of the two simultaneous keyboard-push TCP contacts at center_x-contact_dx and center_x+contact_dx. Reconstruct separated dual-contact candidates and their gripper geometry. A zero offset collapses the contact pair and cannot demonstrate a feasible separated alternative; distinguish geometry bounds from the selected controller half-separation.",
        "keyboard_window": "Operational acceptance window {x:[low,high],y:[low,high],top_z:[low,high]} in world metres; a search heuristic, not a uniquely identifiable object property.",
    }),
    "make_toast": ("moderate", {"toaster_rim_z": "Toaster slot rim world z (m), fitted from several RGB views."}),
    "insert_key": ("moderate", {
        "keyhole_offsets": "Observation framing-probe distances {opening_width_forward,opening_width_back,opening_side,shaft_tip_padding} (m), with exact anchors and axes in KEY_OBSERVATION_DEFINITIONS. These are selected camera-framing parameters, not physical aperture dimensions or padding beyond its boundary; shaft_tip_padding acts transversely at both tip and shaft anchors.",
        "key_dimensions": "Key bow, shaft, teeth and thickness from RGB; no per-episode size oracle.",
    }),
    "fold_clothes": ("light", {"cloth_geometry": "Visible cloth corners, extent and grasp candidates from RGB."}),
    "play_stacking_toy": ("light", {"piece_geometry": "Count, radii, thicknesses and peg heights from RGB."}),
    "sort_nesting_dolls_by_size": ("light", {"doll_geometry": "Count and relative/metric doll dimensions from RGB."}),
    "hang_mugs": ("light", {"mug_geometry": "Mug rim, handle and peg geometry from RGB; grasp offsets are choices."}),
    "sweep_blocks": ("light", {"sweep_z": "Visible sweep contact surface world z (m).", "sweep_geometry": "Pan mouth, blocks and swept region."}),
    "solve_equation": ("light", {"piece_geometry": "Count and dimensions of visible equation pieces."}),
    "pour_balls_into_vase": ("none", {"scene_geometry": "Discover current ball, vessel and opening geometry without layout constants."}),
    "press_by_number": ("none", {"scene_geometry": "Discover current button locations and labels."}),
}

ALIASES = {"coin_geometry": "deposit_coin", "extensions/coin": "deposit_coin",
           "extensions/toast": "make_toast", "extensions/key": "insert_key",
           "mug_geometry": "hang_mugs", "extensions/mugs": "hang_mugs",
           "extensions/sweep": "sweep_blocks", "extensions/equation": "solve_equation"}


POUR_RAY_ESTIMATORS = r"""TASK: pour_liquid_into_cup — exact uncorrected ray estimator definitions.
The numeric pixel-sampling choices below define algorithms, not measured object
dimensions. All metric inputs come from saved RGB reconstruction. No correction
coefficient, bottle dimension or support-height prior is supplied.

Inputs: an upright bottle's original-resolution binary head-view mask; its saved
intrinsics K and camera-to-world matrix T; support height z_s fitted from RGB;
and bottle axis/footprint center_xy independently located from multiple RGB
views. Determine uprightness and a locally horizontal support from RGB. If the
support is tilted, record the fitted plane separately and do not silently claim
that this horizontal-support estimator has the same definition. A mask with
fewer than 200 bottle pixels is insufficient for these estimators: improve the
mask and supporting views. Check identity and occlusion visually.

For a pixel (row r,column c), let o=T[:3,3], R=T[:3,:3], and define
  d = normalize(R @ [(c-K[0,2])/K[0,0], -(r-K[1,2])/K[1,1], -1]).
For any candidate horizontal center p_xy, the height read along its top ray is
  t = dot(p_xy-o[:2], d[:2]) / dot(d[:2], d[:2])
  ray_z(p_xy) = o[2] + t*d[2].
This is the nearest horizontal-distance projection onto the ray, not a full
3-D ray/axis intersection. Require nondegenerate horizontal direction and t>0.

Let rows,cols be all nonzero mask coordinates, top=min(rows), and
top_col=mean(cols[rows<=top+1]). Use the ray at (top,top_col), with no added
half-pixel shift at the top.
A. Located-center estimator: h_loc = ray_z(center_xy) - z_s.
   center_xy must come from multi-view localization of the upright bottle's
   axis/footprint, not a known layout coordinate or an uncalibrated image center.

B. Base-inferred-center estimator:
   bottom=max(rows); bottom_col=mean(cols[rows>=bottom-1]).
   In the inclusive row band [bottom-18,bottom-6], compute max(column)-
   min(column)+1 for every occupied row; take the median of those widths.
   Intersect the ray at (bottom+0.5,bottom_col) with z=z_s to obtain b. Require
   a nonparallel ray and a positive intersection distance. Its optical depth is
     depth = -((b-o) @ R)[2].
   Infer the body radius for this estimator, without a metric radius prior:
     radius_raw = 0.5 * median_width_pixels * depth / K[0,0].
   Let away_xy=normalize(b[:2]-o[:2]); infer
     center_base_xy = b[:2] + radius_raw*away_xy.
   Reuse the SAME top ray and set h_base = ray_z(center_base_xy) - z_s.
   Empty width bands, nonpositive depth or degenerate directions require a
   repaired mask or another valid observation, not a fabricated input.

Fit H_rgb independently from calibrated RGB landmarks or joint bottle-contour
reconstruction, including base/mouth evidence and a support fit. For an upright
bottle H_rgb is its base-to-mouth vertical height. Neither ray correction nor
an assumed bottle height may enter that reconstruction. Then estimate
  ray_corrections = [H_rgb - h_loc, H_rgb - h_base].
These are ADDITIVE corrections, with the sign shown, not raw-estimator-minus-
reference biases. Record H_rgb, z_s, both center estimates, pixel selections,
mask extraction code, raw heights and uncertainty in the derivation. Propagate
shared input uncertainty; two formulas using the same mask are not independent
validation. Check a held-out view and plausible RGB mask perturbations, and
distinguish an episode-specific residual from a correction validated over more
observations. Missing support/center information is a dependency to fit, not an
excuse to stop. Try landmark and joint-contour fits plus complementary views
before any evidence-supported claim that the needed observations are inaccessible.
"""


LAPTOP_MEASUREMENT_DEFINITIONS = r"""TASK: store_laptop_and_headphones — object and estimator definitions.
Identify the wooden headphone stand with a base plate, pole and upper saddle,
the separate small laptop storage rack, the laptop, and its supporting riser.
The requested cradle fields refer to the HEADPHONE STAND. The upper saddle is
called the cradle in the outline estimator; cradle_dims is actually consumed
as a comparison against the wooden BASE plate rectangle. Retain separately
named measurements for the laptop rack instead of putting them in these fields.

Headphone stand outline convention: segment the whole stand; on its binary
mask compute the distance transform, choose an elliptical opening kernel with
integer size int(1.3*max_distance)|1, and open the mask. Keep components of more
than 300 pixels, ordered by image-row centroid. Back-project every other outer
contour pixel of the upper component onto cradle_outline_z; its mean XY is a
cross-check against the bottom component's minimum-area base rectangle.
The bottom outline uses the fitted base-top plane. Fit visible base and saddle
geometry independently before assessing these projection planes. The curved
saddle has no single physical height: its maximum contact crest, surface range
and the estimator's horizontal projection plane are distinct quantities. Report
how the selected plane follows from the measured surface/outline, or show why
the allowed observations do not determine that estimator choice uniquely.
stand_z is the z origin attached to the base rectangle's XY center and yaw for
stand-relative controller offsets; that use alone does not define a geometric
center. Do not equate it with the saddle crest or guess an unseen model origin.
laptop_mid_z belongs to the CLOSED laptop's front-lip pickup reference. Fit
visible open base/lid and hinge geometry to constrain closure; distinguish a
closed-body midpoint, a grasp-plane choice, and the open assembly midpoint.

pixel_model is the wrist-view estimator for the lower edge of a CLOSED laptop
held at its hinge in the insertion orientation. The wrist camera looks down
along the hanging laptop. 'left' and 'right' mean silver faces beside the dark
hinge strip in one image, not left and right robot cameras. Its output is the
lower edge's lateral offset relative to the HAND, along image +column.

Original-image feature algorithm: dark pixels have HSV value<70; plain pixels
have saturation<70. Sample rows range(16,262,4). In each row split dark columns
at gaps>2, select a run 12..80 pixels wide with center within 110 pixels of
column 320, choosing the nearest. Count consecutive plain, non-dark pixels on
each side. Require at least 12 rows; retain strip centers within 14 pixels of
their median and require 12 retained rows. Let c be the median strip center,
strip the median retained width, and left/right the median adjoining face
widths in the first 16 retained rows. Define k=24/max(strip,12). The nominal
strip-width normalization is part of the algorithm, not an object dimension.

With unknown fitted coefficients c0=edge_centre_col, A=edge_mm_per_col,
f=focal_px, L=left_mm_per_px and R=right_px_per_mm, the estimator has form
  hinge_mm = -A*k*(c-c0)
  right_excess = (right-(a0+a1*max(0,c_ref-c)))*k
  if left*k>=19 and strip<=30: lean_mm=L*left-b_left
  if left*k>=19 and strip>30:
      depth=d_strip_ref*k+d_laptop
      lean_mm=(left+strip/2-q_shift/k)*depth/f*1000-b_depth
  elif right_excess>=6: lean_mm=-(right_excess/R+b_right)
  otherwise: lean_mm=b_default
  output_offset_m=-(hinge_mm+lean_mm)/1000.
The two left conditions form one branch, followed by the right/default branch.
All a0,a1,c_ref,b_*,q_shift are nuisance calibration coefficients to establish,
not supplied answers. The depths d_strip_ref and d_laptop are metric inputs
to reconstruct/define for that held geometry.
The zero-column c0 is tied to the HAND reference and strip normalization; the
optical principal point alone need not determine it. L and R model opposite
face/lean branches. Generic transverse depth/f at an arbitrary static object
or camera pose is not an estimate of these requested coefficients.

Use independent RGB reconstructions of hand-relative hinge/lower-edge offsets,
multiple relevant observations spanning strip widths and lean branches, and
held-out predictions of the same output. Deduplicate identical physical SIFT
points before regression, bootstrap and held-out counts; repeated observations
of one point share its reconstruction uncertainty. Camera focal estimates
conditioned on provided calibration are not independent calibration validation.
This run permits only empty-gripper observation: do not close, grasp, lift or
insert the laptop to obtain held-state images. First reconstruct visible shape
and constraints and test whether they suffice to identify the coefficients.
If different admissible grasps/lean relations or calibrations yield the same
permitted RGB observations but different coefficients, document that evidence,
actual attempts and blocked alternatives through fit_impossible. Unavailable
held-state evidence is not permission to substitute a different pixel model.
An estimated pixel_model must contain all five named numeric coefficients and
matching numeric uncertainty, with units above. If only a subset is identified,
retain it under separate diagnostic requirements and explain the unidentifiable
components in the aggregate claim; never put strings, nulls or guessed scales
in the numeric dictionary.
"""


PEN_MEASUREMENT_DEFINITIONS = r"""TASK: fill_pen_holder — estimator definitions without fitted answers.
Identify each physical writing tool from RGB; do not count two class queries
returning the same object as two instances. The requested kinds are purple_marker,
blue_marker, black_pencil, white_pen and yellow_black_pen. Missing-kind components
require an evidence-supported identifiability/visibility argument, never a zero,
borrowed mask or invented profile. Preserve all measured present-kind components
under separate diagnostic requirements if the aggregate is partly impossible.
Physical sizes are reconstructed observations. Do not change a measurement
to match a round number.

The numbers below specify sampling/normalization and numerical algorithms;
they are not measured object dimensions, geometry targets or fitted coefficients.
Fit the support plane from RGB. Let z_s be its fitted local horizontal table
height for this estimator; report slope effects separately if a horizontal
approximation is inadequate. Let E be the camera-to-world matrix, C=E[:3,3],
and K the calibrated intrinsics. Cast original-image pixel centers through K,E
onto the stated horizontal plane. project returns [row,column], while evidence
pixel coordinates remain [column,row]. Keep that conversion explicit.

Raw measure_blob algorithm, BEFORE view correction:
  z = z_s + 0.010  # algorithmic initialization, not an assumed tool thickness
  Repeat at most two passes:
    pts = XY intersections of all mask pixels with plane world_z=z
    mean = mean(pts); eigenvectors = eigh(cov((pts-mean).T))
    axis = larger-eigenvalue eigenvector; across = smaller-eigenvalue eigenvector
    u = (pts-mean) @ across
    width = percentile(u,97) - percentile(u,3)
    ray = normalize([mean.x,mean.y,z] - C)
    sin_e = sin(atan2(abs(ray.z), abs(dot(ray.xy,across))))
    thick = width * sin_e
    if abs(z_s + thick/2 - z) < 0.001: break
    z = z_s + thick/2
  t = (pts-mean) @ axis
  lo,hi = percentile(t,[0.5,99.5])
  rc = project([mean.x,mean.y,z])
  corners = XY intersections of pixels rc+[-0.5,-0.5], rc+[0.5,0.5] at z
  pixel = norm(corners[1]-corners[0])/sqrt(2)
  lo -= pixel/2; hi += pixel/2
  raw_length = hi-lo; raw_width = width
  centre_xy = mean + axis*(lo+hi)/2; centre=[centre_xy.x,centre_xy.y,z]
  a = [centre_xy + axis*(hi-lo)/2, z]
  b = [centre_xy - axis*(hi-lo)/2, z]
Keep the update order: after the second pass, z may have been updated while pts
still describes that pass's earlier plane. Log both values; silently adding a
third projection changes this estimator. A more converged estimator can be a
separate diagnostic. A direct support-plane PCA span is also a different
estimator, especially using 0.5/99.5 rather than 3/97 percentiles for width.

The operational blob mask uses these colour classes on uint8 RGB converted with
OpenCV COLOR_RGB2HSV (H is 0..179; S,V are 0..255):
  white_raw = (S<28) & (V>195)
  blue = (H>=85) & (H<=105) & (S>=18) & (V>170)
  purple = (H>=122) & (H<=165) & (S>=60) & (V>40)
  dark = (V<52) | ((S<40) & (V<120))
  yellow = (H>=16) & (H<=34) & (S>=70) & (V>205)
  white = white_raw & ~blue
Union white/blue/purple/dark/yellow in the RGB-identified tool region excluding
robot/holder pixels, close with a 5-by-5 all-ones kernel, and isolate the tool's
connected component. Do not import a fixed table XY search window. Record the
region and masks, check identity/occlusion in saved RGB, and compare plausible
mask perturbations. A SAM mask can initialize/refine object identification, but
its different silhouette is a change of estimator input to disclose and test.

View-correction regression uses ray=centre-C (not normalized), flat=norm(ray.xy),
cot_e=flat/abs(ray.z), along=abs(dot(ray.xy,axis))/max(flat,1e-9),
across=sqrt(max(0,1-along**2)). With UNKNOWN shared dimensionless coefficients
beta_L,beta_W and UNKNOWN per-kind additive offsets delta_L[k],delta_W[k] in m:
  L_ref = raw_length - beta_L*thick*cot_e*along + delta_L[k]
  W_ref = raw_width  - beta_W*thick*cot_e*across + delta_W[k]
Here L_ref,W_ref come from independently reconstructed multiview RGB geometry,
not the corrected estimator itself. Fit beta_L,beta_W AND the per-kind offsets;
never insert an unfitted coefficient or choose arbitrary zero stretch. Report
the fitted shared coefficients as a separate view_stretch diagnostic requirement.
Use multiple distinct elevations/azimuths for each observed kind; inspect design
matrix rank/conditioning and slope-intercept covariance. A single view per kind
cannot separate its offset from shared stretch. Retain reference uncertainty,
shared support/mask dependencies, residuals and held-out predictions of this SAME
estimator. If permitted views leave coefficients unidentifiable, document actual
attempts and specific ambiguity; do not replace the requested model with raw
support-plane residuals. No fixed geometry or correction values are supplied.

axis_profile algorithm uses the SAME raw item's mask, centre.z and raw endpoints
a,b above; view-corrected dimensions do not redefine those endpoints. Reproject
each selected original RGB pixel onto centre.z, not the tool's top-surface plane.
Let direction=(b.xy-a.xy)/norm(b.xy-a.xy),
t=dot(pts-a.xy,direction)/norm(b.xy-a.xy), and
u=dot(pts-a.xy,[-direction.y,direction.x]). For k=0..19, select
  (t >= k/20) & (t < (k+1)/20).
Do not re-normalize observed min/max, clip t into endpoint bins or use percentile
widths here. Initialize all 20 rows to zero; a bin with fewer than two pixels
keeps the algorithm's zero row and must also be flagged as under-sampled. Each
other row has six channels in this exact order:
  [(max(u)-min(u))/0.035, mean(V)/255, mean(S)/255,
   mean(yellow), mean(dark), mean(white)]
The 0.035 m divisor is a fixed feature scale, not an assumed body width. Do not
divide by the tool's maximum width. Fractions use the exact boolean HSV classes
above, with no hue conversion to degrees or alternative thresholds. Recompute
from original saved RGB and propagate mask, pose and lighting/view variation.
Record which physical end is a and which is b. PCA axis sign is arbitrary:
retain raw ordering and align reversed profiles only using documented RGB
endpoint identity; the polarity consumer compares both profile and profile[::-1].
Do not infer an insertion-end choice from a stored template. A zero under-sampled
row for an observed object is not a template for an absent kind.
"""


KEY_OBSERVATION_DEFINITIONS = r"""TASK: insert_key — exact observation-framing definitions.
The requested distances parameterize feature_geometry, which projects a set of
feature probes to assess a proposed observation camera. They do not define a
detector crop or an enclosing rectangle expanded from physical aperture edges.

Let O, T and S be the supplied RGB-derived world points opening_xyz, tip_xyz and
shaft_xyz. O is the selected opening landmark, T the actual insertion-tip
landmark and S a distinct landmark on the current key shaft. Record their
physical definitions; do not silently replace an opening landmark with an
aperture boundary or a current held-key landmark with its resting-pose location.
The independently selected camera aiming point is not any of these anchors.
Let n=normalize(normal_xyz). For the supplied directed tooth-direction vector t,
let w=normalize(t-n*dot(t,n)) and s=normalize(cross(n,w)). The vector w is the
transverse tooth/opening-slot direction in the aperture plane, not the key's
longitudinal shaft axis. The caller may use the directed opening-slot axis for
w without claiming that the current key is already aligned with it.

With UNKNOWN distances f=opening_width_forward, b=opening_width_back,
c=opening_side and p=shaft_tip_padding, the exact world probe set is:
  [O, T, S, O+f*w, O-b*w, O+c*s, O-c*s,
   T+p*w, T-p*w, S+p*w, S-p*w].
Thus f and b start at O, not at the aperture's forward/back boundaries; c
starts at O on both sides. The same p acts in both signs of w at BOTH T and S.
It does not extend the key longitudinally beyond its tip. Reconstruct physical
aperture/key dimensions separately; those dimensions do not select f,b,c,p.

Each probe is projected using the proposed camera calibration. The
actual_features_in_frame predicate requires finite projected pixels, depth
beyond the near-camera threshold, and every probe inside the usable image
margin. Failure adds a feature_reasons entry. Those reasons seed candidate
rejection before the observation planner performs its other framing, clearance
and motion checks. Shaft span and viewing-angle quality warnings are separate
preferences and do not themselves add feature_reasons. A passing probe-framing
check alone does not establish an accepted or reachable camera motion.

For identifiability, distinguish fixed-camera RGB from observation-policy
outputs. Holding scene geometry and recorded camera poses fixed while varying
unexecuted f,b,c,p leaves the full rendered RGB unchanged. It can change probe
projections, candidate acceptance, and which camera pose that policy would
select. Therefore do not claim identical policy outputs or successful framing
for untested alternatives. Static calibration images acquired independently
of that policy do not expose its chosen distances. Observed probe overlays,
candidate decisions or policy-selected camera trajectories could supply extra
constraints, if actually available and permitted; do not invent them. Compare
alternatives using these exact anchors/axes if making a framing-containment
claim.
"""


def canonical_task(task):
    return ALIASES.get(task, task)


def requirements(task):
    return dict(TASKS.get(canonical_task(task), ("unknown", {"scene_geometry":
        "Identify relevant objects and declare measurable parameters for this task."}))[1])


def examples():
    rows = []
    for task, (_, parameters) in TASKS.items():
        rows.append("TASK: " + task + "\n" + "\n".join(
            f"  {key}: {description}" for key, description in parameters.items()))
    rows.append("OTHER CAPABILITIES: held_object: measure object-to-gripper transform only from observed RGB; "
                "extensions/basket: fit basket rim and depth; extensions/alignment: fit visible alignment landmarks; "
                "common: generic vision utilities, no scene dimensions. These use the same tools and may add parameters.")
    rows.append(POUR_RAY_ESTIMATORS)
    rows.append(LAPTOP_MEASUREMENT_DEFINITIONS)
    rows.append(PEN_MEASUREMENT_DEFINITIONS)
    rows.append(KEY_OBSERVATION_DEFINITIONS)
    rows.append("PERSISTENCE FOR EVERY TASK: A missing estimate is pending work. Fit dependencies, "
                "try distinct RGB methods and complementary permitted views, and record actual attempts. "
                "Only estimates or evidence-supported impossibility determinations are terminal; time, "
                "token, action or compute exhaustion is incomplete work. Operational choices need an "
                "identifiability argument, not an invented geometric value. Use fit_progress() and the "
                "current fit_impossible tool contract before fit_finish().")
    return "\n\n".join(rows)
