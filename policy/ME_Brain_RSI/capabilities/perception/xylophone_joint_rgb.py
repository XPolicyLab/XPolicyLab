"""Joint original-RGB red-head/wood-shaft detector.

Public sphere (13 mm) and shaft (7.5 mm x 200.6 mm) only. No task IDs,
layouts, reward, depth, or simulator state. XY is conditioned on the caller's
unchanged resting-plane prior: this is NOT measured depth or grasp success.
Head shape is scored with the shaft, not gated by rectified circularity.
"""
import cv2
import numpy as np

HEAD_RADIUS = .013
SHAFT_RADIUS = .00375
SHAFT_END = .2006
REST_PLANE = .7956


def _sample(image, uv, outside=0.):
    uv = np.asarray(uv, np.float32)
    shape = uv.shape[:-1]
    values = cv2.remap(image.astype(np.float32), uv[..., 0].reshape(1, -1),
                       uv[..., 1].reshape(1, -1), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=float(outside))
    return values.reshape(shape)


def sphere_outline(camera, center, count=48):
    center = np.asarray(center, float)
    direction = camera.c2w[:3, 3]-center
    distance = np.linalg.norm(direction)
    if distance <= HEAD_RADIUS:
        raise ValueError("Camera inside sphere")
    direction /= distance
    first = np.cross(direction, [0., 0., 1.])
    if np.linalg.norm(first) < 1e-8:
        first = np.cross(direction, [0., 1., 0.])
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    angles = np.arange(count)*2*np.pi/count
    tangent_center = center+direction*HEAD_RADIUS**2/distance
    radius = HEAD_RADIUS*np.sqrt(1-(HEAD_RADIUS/distance)**2)
    return camera.project(tangent_center+radius*(
        np.cos(angles)[:, None]*first+np.sin(angles)[:, None]*second))


def _head_proposals(rgb, camera, z):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = [np.uint8(((h <= 7) | (h >= 160)) & (s >= sat) & (v >= val))*255
             for sat, val in ((80, 30), (110, 35), (140, 40), (170, 45))]
    proposals = []
    for threshold, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 6 or len(contour) < 5:
                continue
            moment = cv2.moments(contour)
            uv = np.array([moment['m10']/area, moment['m01']/area])
            try:
                world = camera.pixel_to_plane(uv, z)
            except ValueError:
                continue
            # Public observable workspace, not scene-specific coordinates.
            if not (-.67 <= world[0] <= .67 and -.50 <= world[1] <= .25):
                continue
            outline = sphere_outline(camera, world)
            radius = np.mean(np.linalg.norm(outline-uv, axis=1))
            if not (.20*np.pi*radius**2 <= area <= 3.*np.pi*radius**2):
                continue
            # Broad proposal bound only: no circularity/axis-ratio rejection.
            if cv2.minEnclosingCircle(contour)[1] > 2.5*radius:
                continue
            proposals.append(dict(uv=uv, threshold=threshold, contour=contour))
    groups = []
    for item in proposals:
        matches = [g for g in groups if np.linalg.norm(g[0]['uv']-item['uv']) < 2.]
        if matches:
            matches[0].append(item)
        else:
            groups.append([item])
    return masks, [dict(uv=np.median([p['uv'] for p in g], axis=0),
                        threshold_count=len(set(p['threshold'] for p in g)))
                   for g in groups if len(set(p['threshold'] for p in g)) >= 2]


def _head_evidence(mask, camera, world):
    center = camera.project([world])[0]
    ring = sphere_outline(camera, world)
    # Red interior and non-red surround jointly reject red bars / large blobs.
    inner = center+(ring-center)*np.array([.25, .50, .75])[:, None, None]
    outer = center+(ring-center)*1.55
    fill = float(np.mean(_sample(mask, inner) > .5))
    surround = float(np.mean(_sample(mask, outer) > .5))
    boundary = .5*(ring+np.roll(ring, 1, axis=0))
    supported = _sample(mask, center+.85*(boundary-center)) > .5
    sectors = supported.reshape(8, -1).mean(1)
    return dict(red_fill=fill, red_surround=surround,
                red_sectors=int(np.sum(sectors >= .5)),
                head_score=fill-.5*surround)


def _shaft_evidence(rgb, camera, world, angles, distance):
    angles = np.asarray(angles)
    axes = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))]
    normals = np.c_[-np.sin(angles), np.cos(angles), np.zeros(len(angles))]
    lengths = np.linspace(.025, .195, 52)
    centers = world+axes[:, None, :]*lengths[None, :, None]
    uv = camera.project(centers)
    left = camera.project(centers+normals[:, None, :]*SHAFT_RADIUS)
    right = camera.project(centers-normals[:, None, :]*SHAFT_RADIUS)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    interior = _sample(gray, uv)
    # Sample external strips in image space; never classify unsupported
    # samples as invisible using privileged masks or delete inconvenient data.
    delta = left-right
    delta /= np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-6)
    left_outer = _sample(gray, left+delta*2.)
    right_outer = _sample(gray, right-delta*2.)
    bright = (interior > left_outer+5.) & (interior > right_outer+5.)
    ldist = _sample(distance, left, 99.)
    rdist = _sample(distance, right, 99.)
    edge = (ldist <= 1.4) & (rdist <= 1.4)
    inside = ((left[..., 0] >= 2) & (right[..., 0] >= 2) &
              (left[..., 0] < rgb.shape[1]-2) & (right[..., 0] < rgb.shape[1]-2) &
              (left[..., 1] >= 2) & (right[..., 1] >= 2) &
              (left[..., 1] < rgb.shape[0]-2) & (right[..., 1] < rgb.shape[0]-2))
    support = bright & edge & inside
    coverage = support.mean(1)
    near = support[:, lengths < .075].mean(1)
    far = support[:, lengths > .145].mean(1)
    # White stand can interrupt the middle, but near AND far must be supported.
    # Unknown gaps remain recorded. A short fragment is not a whole tool.
    accepted = inside.all(1) & (coverage >= .60) & (near >= .50) & (far >= .50)
    score = coverage+.2*bright.mean(1)-.1*np.minimum(ldist+rdist, 4.).mean(1)
    return dict(angles=angles, accepted=accepted, score=score, coverage=coverage,
                near=near, far=far, support=support, uv=uv, left=left, right=right,
                brightness=bright.mean(1), edge_support=edge.mean(1))


def detect_joint_mallet(rgb, camera, plane_z=REST_PLANE, diagnostics=False):
    """Return a directed plane-conditioned fit, or None with explicit reasons.

    Independently detected global line segments propose shaft axes; a bounded
    dense angular search also permits broken/occluded lines. Neither proposal
    alone constitutes a detection. There is no per-case coordinate/threshold.
    """
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or
            rgb.shape[:2] != camera.resolution[::-1] or not np.isfinite(plane_z)):
        raise ValueError("Expected calibrated uint8 original RGB and finite plane")
    masks, heads = _head_proposals(rgb, camera, plane_z)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, 12,
                           minLineLength=15, maxLineGap=8)
    lines = [] if lines is None else lines[:, 0]
    fits, rejected = [], []
    for proposal in heads:
        uv = proposal['uv']
        world = camera.pixel_to_plane(uv, plane_z)
        evidence = _head_evidence(masks[1]/255., camera, world)
        # Keep a broad head candidate for a joint comparison; no roundness test.
        if (evidence['red_fill'] < .40 or evidence['red_surround'] > .40
                or evidence['red_sectors'] < 4):
            rejected.append(dict(uv=uv.tolist(), reason='head_color_extent', **evidence))
            continue
        angles = list(np.arange(0., 2*np.pi, np.deg2rad(2.)))
        line_axes = 0
        for line in lines:
            a, b = np.asarray(line, float).reshape(2, 2)
            vector = b-a
            length = np.linalg.norm(vector)
            if length < 1:
                continue
            direction = vector/length
            normal = np.array([-direction[1], direction[0]])
            if abs((uv-a)@normal) > 10:
                continue
            try:
                w0 = camera.pixel_to_plane(a, plane_z)
                w1 = camera.pixel_to_plane(b, plane_z)
            except ValueError:
                continue
            angle = np.arctan2(*(w1-w0)[1::-1])
            angles.extend([angle, angle+np.pi]);line_axes += 1
        coarse = _shaft_evidence(rgb, camera, world, angles, distance)
        ranked = np.argsort(-coarse['score'])
        seeds = []
        for index in ranked:
            angle = coarse['angles'][index]
            if all(np.cos(angle-other) < np.cos(np.deg2rad(5)) for other in seeds):
                seeds.append(angle)
            if len(seeds) == 3:
                break
        fine = np.concatenate([angle+np.deg2rad(np.arange(-3., 3.01, .5)) for angle in seeds])
        trials = []
        # Color centroid is only a seed. Jointly refine image head center and
        # directed shaft; a 2 px contour uncertainty must not rotate the whole
        # tool about a biased center. This bound is generic, never case-specific.
        def evaluate_position(trial_uv, trial_angles):
            trial_world = camera.pixel_to_plane(trial_uv, plane_z)
            ev = _head_evidence(masks[1]/255., camera, trial_world)
            if ev['red_fill'] < .40 or ev['red_surround'] > .40 or ev['red_sectors'] < 4:
                return
            dat = _shaft_evidence(rgb, camera, trial_world, trial_angles, distance)
            total = dat['score']+ev['head_score']
            j = int(np.argmax(total))
            trials.append((float(total[j]), trial_uv, trial_world, ev, dat, j))
        for dx in (-2., 0., 2.):
            for dy in (-2., 0., 2.):
                evaluate_position(uv+[dx,dy], fine)
        if not trials:
            rejected.append(dict(uv=uv.tolist(), reason='joint_refinement_no_evidence'))
            continue
        top=max(trials,key=lambda t:t[0])
        for dx in (-1.,-.5,0.,.5,1.):
            for dy in (-1.,-.5,0.,.5,1.):
                candidate_uv=top[1]+[dx,dy]
                if np.max(np.abs(candidate_uv-uv)) <= 2.:
                    evaluate_position(candidate_uv, top[4]['angles'][top[5]]+
                                      np.deg2rad(np.arange(-1.,1.01,.25)))
        _,uv,world,evidence,data,best=max(trials,key=lambda t:t[0])
        record = dict(head_uv=uv.tolist(), head_xy=world[:2].tolist(),
                      angle=float(data['angles'][best]), **evidence,
                      shaft_support=float(data['coverage'][best]),
                      near_support=float(data['near'][best]), far_support=float(data['far'][best]),
                      brightness_support=float(data['brightness'][best]),
                      bilateral_edge_support=float(data['edge_support'][best]),
                      threshold_count=proposal['threshold_count'], line_axis_proposals=line_axes,
                      score=float(data['score'][best]+evidence['head_score']),
                      support=data['support'][best].tolist())
        competing_axes = []
        for _,_,_,ev,dat,_ in trials:
            for index in np.flatnonzero(dat['accepted']):
                score = float(dat['score'][index]+ev['head_score'])
                if (score >= record['score']-.15 and
                        np.cos(dat['angles'][index]-record['angle']) < np.cos(np.deg2rad(8))):
                    competing_axes.append(float(dat['angles'][index]))
        if competing_axes:
            record['reason']='ambiguous_shaft_axes'
            record['competing_angles']=competing_axes
            rejected.append(record)
            continue
        if not data['accepted'][best] or record['score'] < 1.45:
            record['reason']='insufficient_connected_shaft'
            rejected.append(record)
            continue
        record['reason']='joint_red_head_and_long_shaft'
        fits.append(record)
    fits.sort(key=lambda x: -x['score'])
    reason = 'no_joint_candidate'
    result = None
    if fits:
        best = fits[0]
        rivals = [f for f in fits[1:] if (
            np.linalg.norm(np.array(f['head_uv'])-best['head_uv']) > 2. or
            np.cos(f['angle']-best['angle']) < np.cos(np.deg2rad(5)))]
        # A weak red-bar/edge combination must lose by an explicit margin.
        # Two strongly supported complete tools stay ambiguous even if unequal.
        strong_rivals = [r for r in rivals if r['head_score'] >= .65 and r['shaft_support'] >= .80]
        if strong_rivals or (rivals and best['score']-max(r['score'] for r in rivals) < .30):
            reason = 'ambiguous_joint_candidates'
        else:
            reason = 'accepted'
            axis = np.array([np.cos(best['angle']), np.sin(best['angle'])])
            result = dict(head_xy=np.array(best['head_xy']), axis_xy=axis,
                          head_uv=np.array(best['head_uv']),
                          handle_end_xy=np.array(best['head_xy'])+axis*SHAFT_END,
                          head_fit='original_rgb_joint_model', evidence=best,
                          observability='plane_conditioned_head_and_axis_only',
                          depth_observed=False, roll_observed=False, grasp_verified=False)
    diagnostic = dict(reason=reason, candidates=fits, rejected=rejected,
                      head_proposals=len(heads), line_proposals=len(lines),
                      plane_z_prior=float(plane_z))
    return (result, diagnostic) if diagnostics else result
