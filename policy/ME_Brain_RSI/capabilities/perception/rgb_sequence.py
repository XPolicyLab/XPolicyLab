"""RGB-only calibrated geometry and bounded, ordered feature-point motion.

All positions are supplied by the caller or reconstructed from current cameras.
The 23 mm endpoint gap, 40 mm rise and 35 mm TCP clearance are motion
design values, not scene sizes or a collision certificate.
"""
import json
import cv2
import numpy as np

CAMERAS = {'head': 'cam_head', 'wrist_l': 'cam_left_wrist', 'wrist_r': 'cam_right_wrist'}


def arg(name, type='float', **kw):
    return dict(name=name, type=type, **kw)


ARM = dict(name='arm', positional=True, choices=['left', 'right'])
TOOL = dict(name='rgb_sequence', commands=[
    dict(name='stem_pick', budget=True, help='stereo-measure a sphere and an adjacent exposed span, grasp the span and remeasure', args=[ARM,
        arg('pixels', 'str', required=True), arg('segments', 'str', required=True),
        arg('camera', 'str', default='head', choices=list(CAMERAS)),
        arg('other', 'str', default='wrist_l', choices=list(CAMERAS)), arg('radius', default=None),
        arg('inset', default=.002), arg('clearance', default=.05), arg('lift', default=.07)]),
    dict(name='sphere_pick', budget=True, help='stereo-measure and grasp an upper spherical section', args=[ARM,
        arg('pixels', 'str', required=True), arg('camera', 'str', default='head', choices=list(CAMERAS)),
        arg('other', 'str', default='wrist_l', choices=list(CAMERAS)), arg('radius', default=None),
        arg('clearance', default=.05), arg('lift', default=.07)]),
    dict(name='rgb_span', budget=False, help='reconstruct overlapping visible straight segments from two RGB views', args=[
        arg('segments', 'str', required=True),
        arg('camera', 'str', default='head', choices=list(CAMERAS)),
        arg('other', 'str', default='wrist_l', choices=list(CAMERAS))]),
    dict(name='rgb_sphere', budget=False, help='fit a coloured spherical centre from RGB silhouettes', args=[
        arg('pixels', 'str', required=True), arg('camera', 'str', default='head', choices=list(CAMERAS)),
        arg('other', 'str', default='wrist_l', choices=list(CAMERAS)),
        arg('radius', default=None), arg('temporal', 'int', default=0, choices=[0, 1]), arg('arm', 'str', default='left', choices=['left', 'right'])]),
    dict(name='rgb_points', budget=False, help='triangulate matched pixels or intersect a measured horizontal plane', args=[
        arg('pairs', 'str', default='[]'), arg('pixels', 'str', default='[]'),
        arg('camera', 'str', default='head', choices=list(CAMERAS)),
        arg('other', 'str', default='wrist_l', choices=list(CAMERAS)),
        arg('z', default=None), arg('arm', 'str', default='left', choices=['left', 'right'])]),
    dict(name='span_pick', budget=True, help='grasp the midpoint of two measured 3D endpoints', args=[ARM,
        arg('points', 'str', required=True), arg('inset', default=0.002),
        arg('clearance', default=0.05), arg('lift', default=0.07)]),
    dict(name='point_series', budget=True, help='ordered descend and rise using a TCP-local feature offset', args=[ARM,
        arg('points', 'str', required=True), arg('offset', 'str', required=True),
        arg('gap', default=0.023), arg('rise', default=0.04),
        arg('tcp_clearance', default=.035),
        arg('resume', 'int', default=0, choices=[0, 1])])])


def array(value, shape):
    a = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('invalid finite array; expected ' + str(shape))
    return a


def points(value):
    a = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not 1 <= len(a) <= 32 or not np.isfinite(a).all():
        raise ValueError('points must be a JSON list of 1..32 finite world XYZ triples')
    if np.any(np.abs(a[:, :2]) > 1.2) or np.any((a[:, 2] < 0.3) | (a[:, 2] > 1.6)):
        raise ValueError('world points outside geometry envelope')
    return a


def bounded(args, name, default, lo, hi):
    x = float(args.get(name, default))
    if not np.isfinite(x) or not lo <= x <= hi:
        raise ValueError(name + ' outside bounds')
    return x


def view(obs, name):
    source = CAMERAS[name]
    c = obs['cameras'][source]
    k = array(c['intrinsics'], (3, 3))
    t = array(c['extrinsics_world'], (4, 4))
    im = cv2.imdecode(np.frombuffer(obs['png'][source], np.uint8), cv2.IMREAD_COLOR)
    if im is None or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError('invalid camera calibration or image')
    return k, t, im.shape[:2]


def ray(v, pixel):
    k, t, (h, w) = v
    u, y = array(pixel, (2,))
    if not 0 <= u < w or not 0 <= y < h:
        raise ValueError('pixel outside current image')
    d = t[:3, :3] @ np.linalg.solve(k, [u, y, 1.])
    return t[:3, 3], d / np.linalg.norm(d)


def project(v, p):
    k, t, _ = v
    q = t[:3, :3].T @ (p-t[:3, 3])
    if q[2] <= 0:
        raise ValueError('point behind camera')
    q = k @ q
    return q[:2] / q[2]


def triangulate(a, b, pa, pb):
    o, d = ray(a, pa)
    e, f = ray(b, pb)
    sine = np.linalg.norm(np.cross(d, f))
    if sine < 0.10:
        raise ValueError('insufficient stereo baseline')
    lengths = np.linalg.lstsq(np.column_stack((d, -f)), e-o, rcond=None)[0]
    if np.min(lengths) <= 0:
        raise ValueError('correspondence behind camera')
    x, y = o+lengths[0]*d, e+lengths[1]*f
    p = (x+y)*0.5
    gap = float(np.linalg.norm(x-y))
    reproj = max(float(np.linalg.norm(project(a, p)-pa)), float(np.linalg.norm(project(b, p)-pb)))
    if gap > 0.004 or reproj > 1.5:
        raise ValueError('inconsistent correspondence: ray gap %.4f m, reprojection %.2f px' % (gap, reproj))
    return p, dict(ray_gap_m=gap, reprojection_px=reproj, baseline_sine=float(sine))


def intersect(v, px, z):
    o, d = ray(v, px)
    if abs(d[2]) < 0.10 or (z-o[2])/d[2] <= 0:
        raise ValueError('plane intersection behind camera or grazing')
    return o+d*((z-o[2])/d[2])


def stereo_span(a, b, segments):
    """Intersect interpretation planes; endpoints need not correspond.

    Inputs identify the same exposed straight surface ridge, not the whole
    silhouette or an assumed cylinder axis. No height or thickness is assumed.
    """
    segments = array(segments, (2, 2, 2))
    normals, origins = [], []
    for v, seg in zip((a, b), segments):
        if np.linalg.norm(seg[1]-seg[0]) < 12:
            raise ValueError('visible segments must span at least 12 pixels')
        o, d = ray(v, seg[0])
        _, e = ray(v, seg[1])
        n = np.cross(d, e)
        normals.append(n/np.linalg.norm(n))
        origins.append(o)
    direction = np.cross(*normals)
    sine = float(np.linalg.norm(direction))
    if sine < .15 or np.linalg.norm(origins[0]-origins[1]) < .02:
        raise ValueError('insufficient line stereo baseline; segments near epipolar direction')
    direction /= sine
    anchor = np.linalg.solve(np.vstack((normals, direction)),
        [np.dot(n, o) for n, o in zip(normals, origins)]+[0.])
    intervals = []
    for v, seg in zip((a, b), segments):
        values = []
        for px in seg:
            o, d = ray(v, px)
            t, distance = np.linalg.lstsq(np.column_stack((direction, -d)), o-anchor, rcond=None)[0]
            if distance <= 0:
                raise ValueError('span behind camera')
            values.append(t)
        intervals.append(sorted(values))
    lo, hi = max(i[0] for i in intervals), min(i[1] for i in intervals)
    if not .02 <= hi-lo <= .20:
        raise ValueError('common exposed span must be 20..200 mm')
    # Trim the common interval away from supplied occlusion/support boundaries.
    lo, hi = lo+.1*(hi-lo), hi-.1*(hi-lo)
    xyz = points([anchor+lo*direction, anchor+hi*direction])
    for v in (a, b):
        for p in xyz:
            ray(v, project(v, p))
    return xyz, sine


def measure_span(api, args):
    name, other = args.get('camera', 'head'), args.get('other', 'wrist_l')
    if name == other:
        raise ValueError('two distinct cameras required')
    obs = api.observe()
    a, b = view(obs, name), view(obs, other)
    segments = array(args['segments'], (2, 2, 2))
    xyz, sine = stereo_span(a, b, segments)
    # A small pixel error can move a nearly degenerate line a long way. Report
    # and gate one-pixel normal shifts of either observed line independently.
    midpoint = xyz.mean(0)
    sensitivity = 0.
    for i in range(2):
        delta = segments[i, 1]-segments[i, 0]
        normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
        for sign in (-1, 1):
            shifted = segments.copy()
            shifted[i] += sign*normal
            perturbed, _ = stereo_span(a, b, shifted)
            sensitivity = max(sensitivity, float(np.linalg.norm(perturbed.mean(0)-midpoint)))
    if sensitivity > .008:
        raise ValueError('line depth too sensitive to one-pixel error')
    return dict(plan_ok=True, plan_fail_reason=None, points_world=xyz.tolist(),
        midpoint_world=midpoint.tolist(), plane_angle_sine=sine,
        one_pixel_sensitivity_m=sensitivity, source='RGB stereo interpretation planes',
        note='Same exposed straight surface ridge in both images; no inferred axis, thickness or grasp proof.'), 0


def robust_cone(rays, k, seed):
    """Fit a tangent cone to a mostly visible round boundary with a thin tail.

    Hypotheses and support are in calibrated angular coordinates. No radius,
    colour, camera pose or scene position is assumed.
    """
    focal = min(k[0, 0], k[1, 1])
    seed_ray = np.linalg.solve(k, [*seed, 1.])
    seed_ray /= np.linalg.norm(seed_ray)
    rng = np.random.default_rng(0)
    candidates = []
    for _ in range(384):
        a, b, c = rays[rng.choice(len(rays), 3, replace=False)]
        axis = np.cross(b-a, c-a)
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis /= norm
        if axis[2] < 0:
            axis = -axis
        alpha = np.arccos(np.clip(a@axis, -1, 1))
        if not .002 < alpha < .25:
            continue
        keep = np.abs(np.arccos(np.clip(rays@axis, -1, 1))-alpha)*focal < .65
        if keep.sum() < max(18, .55*len(rays)):
            continue
        # Refit the angular plane on the consensus, twice.
        for _ in range(2):
            selected = rays[keep]
            axis = np.linalg.svd(selected-selected.mean(0), full_matrices=False)[2][-1]
            if axis[2] < 0:
                axis = -axis
            theta = np.arccos(np.clip(rays@axis, -1, 1))
            alpha = float(np.median(theta[keep]))
            keep = np.abs(theta-alpha)*focal < .65
            if keep.sum() < 18:
                break
        if keep.sum() < max(18, .55*len(rays)) or not .002 < alpha < .25:
            continue
        if np.arccos(np.clip(seed_ray@axis, -1, 1)) > alpha+2/focal:
            continue
        radial = rays[keep]-(rays[keep]@axis)[:, None]*axis
        basis = np.linalg.svd(radial, full_matrices=False)[2][:2]
        angles = np.sort(np.arctan2(radial@basis[1], radial@basis[0]))
        max_gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
        # At least 270 degrees of supported boundary; reject half-discs and
        # straight edges that happen to admit a large-radius circular fit.
        if max_gap > np.pi/2:
            continue
        residual = float(np.sqrt(np.mean((theta[keep]-alpha)**2))*focal)
        if residual/(alpha*focal) > .10:
            continue
        candidates.append((int(keep.sum()), axis.copy(), alpha, residual, max_gap))
    if not candidates:
        raise ValueError('silhouette has no well-supported round boundary')
    candidates.sort(key=lambda c: (-c[0], c[3]))
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0] < .9*best[0]:
            continue
        separation = np.arccos(np.clip(candidate[1]@best[1], -1, 1))*focal
        if separation > 1.5 or abs(candidate[2]-best[2])*focal > 1.5:
            raise ValueError('ambiguous round boundary')
    return best[1], best[2], best[3], best[0]/len(rays), float(np.rad2deg(best[4]))


def silhouette(image, k, seed):
    h, w = image.shape[:2]
    seed = np.asarray(seed, dtype=float)
    if seed.shape != (2,) or not np.isfinite(seed).all():
        raise ValueError('seed must be a finite pixel pair')
    x, y = np.rint(seed).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('seed outside image')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Local saturated pixels tolerate a seed on the white specular highlight.
    patch = hsv[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)].reshape(-1,3)
    patch = patch[(patch[:,1] > 65) & (patch[:,2] > 40)]
    if len(patch) < 3:
        raise ValueError('seed needs a saturated colour region')
    angles = patch[:,0].astype(float)*np.pi/90
    hue = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())*90/np.pi % 180
    delta = np.abs(hsv[:,:,0].astype(float)-hue)
    mask = ((np.minimum(delta,180-delta) < 14) & (hsv[:,:,1] > 55) & (hsv[:,:,2] > 30)).astype('uint8')*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    near = [c for c in contours if cv2.pointPolygonTest(c, (float(x),float(y)), True) >= -3]
    if len(near) != 1:
        raise ValueError('seed does not identify one colour silhouette')
    c = near[0].reshape(-1,2).astype(float)
    area = cv2.contourArea(c.astype('float32'))
    if len(c) < 16 or area < 20 or np.any(c.min(0) <= 1) or np.any(c.max(0) >= [w-2,h-2]):
        raise ValueError('silhouette too small or clipped')
    # A sphere's tangent rays form a circular cone: axis dot ray = cos(alpha).
    rays = np.linalg.solve(k, np.column_stack((c, np.ones(len(c)))).T).T
    rays /= np.linalg.norm(rays,axis=1)[:,None]
    _, _, vh = np.linalg.svd(rays-rays.mean(0), full_matrices=False)
    axis = vh[-1]
    if axis[2] < 0:
        axis = -axis
    theta = np.arccos(np.clip(rays@axis,-1,1))
    alpha = float(np.median(theta))
    residual = float(np.sqrt(np.mean((theta-alpha)**2))*min(k[0,0],k[1,1]))
    radial = (rays-(rays@axis)[:,None]*axis)
    _, s, _ = np.linalg.svd(radial,full_matrices=False)
    mode, support, missing_arc = 'complete', 1., 0.
    if not .002 < alpha < .25 or residual > .8 or residual/(alpha*min(k[0,0],k[1,1])) > .16 or s[1]/s[0] < .75:
        axis, alpha, residual, support, missing_arc = robust_cone(rays, k, seed)
        mode = 'robust_arc'
    # Raster contours run through boundary pixel centres; half-pixel correction
    # estimates the actual colour boundary, avoiding systematic radius shrinkage.
    alpha += .5 / np.sqrt(k[0,0]*k[1,1])
    pixel = k@axis
    return dict(axis=axis, alpha=alpha, pixel=pixel[:2]/pixel[2],
                residual_px=residual, hue=hue, fit_mode=mode,
                boundary_support_fraction=support, max_missing_arc_deg=missing_arc)


def stereo_sphere(a, b, ta, tb):
    o, e = ta[:3,3], tb[:3,3]
    d, f = ta[:3,:3]@a['axis'], tb[:3,:3]@b['axis']
    if np.linalg.norm(np.cross(d,f)) < .1:
        raise ValueError('insufficient sphere stereo baseline')
    lengths = np.linalg.lstsq(np.column_stack((d,-f)), e-o,rcond=None)[0]
    if np.min(lengths) <= 0:
        raise ValueError('sphere behind camera')
    p, q = o+lengths[0]*d, e+lengths[1]*f
    radii = lengths*np.sin([a['alpha'],b['alpha']])
    gap = float(np.linalg.norm(p-q))
    if gap > .004 or abs(radii[0]-radii[1]) > max(.0015,.15*radii.mean()):
        raise ValueError('inconsistent sphere silhouettes')
    return (p+q)/2, float(radii.mean()), gap


def match_sphere(obs, name, a, ta, radius):
    """Use a measured radius for correspondence, never monocular range.

    A subpixel boundary error produces centimetres of monocular range error
    for small distant images. Stereo centre rays do not have that dependency.
    """
    matches = []
    for other in CAMERAS:
        if other == name or CAMERAS[other] not in obs.get('png', {}):
            continue
        v = view(obs, other)
        im = cv2.imdecode(np.frombuffer(obs['png'][CAMERAS[other]], np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        delta = np.abs(hsv[:, :, 0].astype(float)-a['hue'])
        mask = ((np.minimum(delta, 180-delta) < 14) &
                (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 30)).astype('uint8')
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
        if len(candidates) > 24:
            raise ValueError('too many same-colour stereo candidates')
        for i in candidates:
            component = (labels == i).astype('uint8')
            distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            try:
                b = silhouette(im, v[0], [x, y])
                centre, measured, gap = stereo_sphere(a, b, ta, v[1])
                if abs(measured/radius-1.) > .2:
                    continue
                d, f = ta[:3, :3]@a['axis'], v[1][:3, :3]@b['axis']
                baseline = float(np.linalg.norm(np.cross(d, f)))
                matches.append((baseline, other, b, centre, measured, gap))
            except ValueError:
                continue
    if not matches:
        raise ValueError('sphere stereo match unavailable; supply two visible boundary seeds in distinct calibrated views')
    matches.sort(key=lambda row: -row[0])
    best = matches[0]
    if any(np.linalg.norm(row[3]-best[3]) > .006 for row in matches[1:]):
        raise ValueError('ambiguous sphere stereo matches; supply explicit paired seeds')
    return best[1:]


def refresh_offset(api, arm, offset, measured):
    """Current stereo only; the predicted centre associates, never sets range."""
    obs = api.observe()
    pose = arm.tcp()
    predicted = pose[:3, 3] + pose[:3, :3] @ offset
    matches = []
    for name, source in CAMERAS.items():
        if source not in obs.get('png', {}):
            continue
        try:
            v = view(obs, name)
            seed = project(v, predicted)
            im = cv2.imdecode(np.frombuffer(obs['png'][source], np.uint8), cv2.IMREAD_COLOR)
            a = silhouette(im, v[0], seed)
            hue_delta = abs(a['hue'] - measured['hue'])
            if min(hue_delta, 180-hue_delta) >= 14:
                continue
            other, b, centre, radius, gap = match_sphere(obs, name, a, v[1], measured['radius'])
            if np.linalg.norm(centre-predicted) > .020:
                continue
            matches.append((centre, radius, gap, name, other))
        except (ValueError, KeyError):
            continue
    if not matches:
        raise ValueError('current stereo unavailable near predicted feature')
    if any(np.linalg.norm(row[0]-matches[0][0]) > .004 for row in matches[1:]):
        raise ValueError('ambiguous current feature centres')
    centre, radius, gap, name, other = min(matches, key=lambda row: row[2])
    updated = (centre-pose[:3, 3]) @ pose[:3, :3]
    return updated, dict(status='measured', centre_world=centre.tolist(),
                         offset_local=updated.tolist(), radius_m=radius, ray_gap_m=gap,
                         cameras=[name, other], innovation_m=float(np.linalg.norm(updated-offset)))


def temporal_sphere(arm, camera, fit, transform, radius):
    """Intersect camera rays in TCP coordinates under a rigid attachment hypothesis.

    Radius is only a consistency check. Range comes from pose-separated rays.
    No robot motion is issued here, and no rigid holding certificate is inferred.
    """
    if arm.gripper() > .5:
        arm._rgb_temporal = []
        raise ValueError('temporal measurement requires closed fingers')
    pose = arm.tcp()
    local_camera = np.linalg.inv(pose) @ transform
    history = getattr(arm, '_rgb_temporal', [])
    current = (camera, fit, local_camera, radius)
    matches = []
    for old_camera, old_fit, old_transform, old_radius in history:
        hue_delta = abs(old_fit['hue']-fit['hue'])
        if min(hue_delta, 180-hue_delta) >= 14 or abs(radius/old_radius-1) > .05:
            continue
        try:
            offset, measured, gap = stereo_sphere(old_fit, fit, old_transform, local_camera)
            # A held feature must remain close to the TCP. This is a workspace
            # design bound, not an assumed object dimension.
            if np.linalg.norm(offset) > .30 or abs(measured/radius-1) > .10:
                continue
            # Bound sensitivity to a one-pixel centre-ray error in either view.
            sensitivity = 0.
            for which in (0, 1):
                for axis in (0, 1):
                    for sign in (-1, 1):
                        fits = [dict(old_fit), dict(fit)]
                        changed = fits[which]
                        pixel = changed['pixel'].copy()
                        pixel[axis] += sign
                        direction = np.linalg.solve(changed['intrinsics'], [*pixel, 1.])
                        changed['axis'] = direction/np.linalg.norm(direction)
                        shifted, _, _ = stereo_sphere(*fits, old_transform, local_camera)
                        sensitivity = max(sensitivity, float(np.linalg.norm(shifted-offset)))
            if sensitivity > .006:
                continue
            matches.append((offset, measured, gap, sensitivity, old_camera))
        except ValueError:
            continue
    # Keep bounded recent observations; identical poses add no information.
    if not history or np.linalg.norm(history[-1][2]-local_camera) > .01:
        arm._rgb_temporal = (history+[current])[-4:]
    if not matches:
        raise ValueError('temporal sample retained; need a matching visible boundary at a different TCP pose with sufficient parallax')
    best = min(matches, key=lambda row: row[3])
    if any(np.linalg.norm(row[0]-best[0]) > .006 for row in matches):
        arm._rgb_temporal = [current]
        raise ValueError('inconsistent temporal offsets; attachment or correspondence changed')
    offset, measured, gap, sensitivity, old_camera = best
    return pose[:3, 3]+pose[:3, :3]@offset, measured, gap, sensitivity


def measure_sphere(api, args):
    obs = api.observe()
    name, other = args.get('camera', 'head'), args.get('other', 'wrist_l')
    seeds = json.loads(args['pixels']) if isinstance(args['pixels'], str) else args['pixels']
    radius = args.get('radius')
    seeds = np.asarray(seeds, dtype=float)
    if seeds.shape not in ((1, 2), (2, 2)) or not np.isfinite(seeds).all():
        raise ValueError('provide one or two finite pixel seeds')
    prior_radius = None if radius is None else bounded(args, 'radius', 0, .002, .1)
    if len(seeds) == 1 and prior_radius is None:
        raise ValueError('one seed requires a previously measured radius')
    def fit(camera, seed):
        v = view(obs, camera)
        im = cv2.imdecode(np.frombuffer(obs['png'][CAMERAS[camera]], np.uint8), cv2.IMREAD_COLOR)
        try:
            return silhouette(im, v[0], seed), v[1]
        except ValueError as exc:
            raise ValueError(camera + ': ' + str(exc)) from exc
    a, ta = fit(name, seeds[0])
    a['intrinsics'] = view(obs, name)[0]
    temporal = args.get('temporal', 0)
    if temporal not in (0, 1):
        raise ValueError('temporal must be 0 or 1')
    if temporal and (len(seeds) != 1 or prior_radius is None):
        raise ValueError('temporal mode requires one seed and a measured radius')
    def diagnostic(camera, fit):
        return dict(camera=camera, centre_pixel=fit['pixel'].tolist(),
                    **{key: fit[key] for key in ('residual_px', 'fit_mode',
                        'boundary_support_fraction', 'max_missing_arc_deg')})
    diagnostics = [diagnostic(name, a)]
    sensitivity = None
    if temporal:
        centre, radius, gap, sensitivity = temporal_sphere(
            api.arm(args.get('arm', 'left')), name, a, ta, prior_radius)
        source = 'RGB temporal tangent cones in TCP frame; assumes rigid attachment'
    elif len(seeds) == 2:
        if name == other:
            raise ValueError('two distinct cameras required')
        b, tb = fit(other, seeds[1])
        centre, radius, gap = stereo_sphere(a, b, ta, tb)
        if prior_radius is not None and abs(radius/prior_radius-1.) > .2:
            raise ValueError('stereo radius differs from measured prior')
        diagnostics.append(diagnostic(other, b))
        source = 'RGB stereo tangent cones'
    else:
        other, b, centre, radius, gap = match_sphere(obs, name, a, ta, prior_radius)
        diagnostics.append(diagnostic(other, b))
        source = 'RGB automatic stereo tangent cones; caller-measured radius used only for association'
    points([centre])
    arm = api.arm(args.get('arm','left'))
    pose = arm.tcp()
    offset = (centre-pose[:3,3])@pose[:3,:3]
    lift_evidence = dict(status='no_grasp_baseline', grasp_verified=False)
    baseline = getattr(arm, '_rgb_pick_feature', None)
    if baseline is not None:
        hue_delta = abs(a['hue']-baseline['hue'])
        same_appearance = (min(hue_delta, 180-hue_delta) < 14
                           and abs(radius/baseline['radius']-1.) < .2)
        gain = float(centre[2]-baseline['centre'][2])
        lift_evidence.update(status=('lift_observed' if gain >= .025 else 'feature_not_lifted')
                             if same_appearance else 'feature_identity_unresolved',
                             height_gain_m=gain)
        # Height change is necessary evidence, never a rigid-holding certificate.
        baseline['status'] = lift_evidence['status']
        baseline['offset'] = offset.copy()
    arm._rgb_sphere_measurement = dict(centre=centre.copy(), radius=radius, hue=a['hue'])
    return dict(plan_ok=True, plan_fail_reason=None, centre_world=centre.tolist(),
                offset_local=offset.tolist(), radius_m=radius, lift_evidence=lift_evidence,
                silhouettes=diagnostics, ray_gap_m=gap, source=source,
                temporal_rigidity_assumed=bool(temporal), one_pixel_sensitivity_m=sensitivity), 0


def measure(api, args):
    obs = api.observe()
    name, other = args.get('camera', 'head'), args.get('other', 'wrist_l')
    a = view(obs, name)
    pairs = json.loads(args.get('pairs', '[]'))
    pixels = json.loads(args.get('pixels', '[]'))
    if not isinstance(pairs, list) or not isinstance(pixels, list) or len(pairs) > 32 or len(pixels) > 32:
        raise ValueError('at most 32 correspondences/pixels')
    found, diagnostics = [], []
    if pairs:
        if name == other:
            raise ValueError('two distinct cameras required')
        b = view(obs, other)
        for pair in pairs:
            pair = array(pair, (2, 2))
            p, receipt = triangulate(a, b, pair[0], pair[1])
            found.append(p)
            diagnostics.append(receipt)
    z = args.get('z')
    plane = None
    if z is not None:
        z = bounded(args, 'z', 0, 0.3, 1.6)
        plane = dict(z=z, source='caller measured height')
    elif pixels:
        if len(found) < 3:
            raise ValueError('plane projection requires --z or at least three coplanar stereo correspondences')
        cloud = np.asarray(found)
        if np.ptp(cloud[:, 2]) > 0.006 or np.linalg.svd(cloud[:, :2]-cloud[:, :2].mean(0), compute_uv=False)[1] < 0.01:
            raise ValueError('plane anchors must span two axes and agree in height within 6 mm')
        z = float(np.median(cloud[:, 2]))
        plane = dict(z=z, height_spread_m=float(np.ptp(cloud[:, 2])), source='RGB stereo anchors')
    result = [intersect(a, px, z) for px in pixels] if pixels else found
    if not result:
        raise ValueError('provide matched pairs or pixels with a measured plane')
    xyz = points(result)
    pose = api.arm(args.get('arm', 'left')).tcp()
    offsets = (xyz-pose[:3, 3]) @ pose[:3, :3]
    return dict(plan_ok=True, plan_fail_reason=None, points_world=xyz.tolist(),
                offsets_local=offsets.tolist(), stereo=diagnostics, plane=plane,
                anchors_world=[p.tolist() for p in found], source='current RGB camera geometry'), 0


def moved(api, arm, target, name, result, offset=None):
    if api.over:
        raise ValueError('episode_over')
    fb = {}
    code = api.move_tcp(arm, target.copy(), fb)
    pose = arm.tcp()
    offset = np.zeros(3) if offset is None else offset
    error = float(np.linalg.norm(pose[:3, 3]+pose[:3, :3] @ offset-target[:3, 3]-target[:3, :3] @ offset))
    result['stages'].append(dict(stage=name, plan_ok=fb.get('plan_ok', code == 0),
                                 feature_error_m=error, reason=fb.get('plan_fail_reason')))
    if code or not fb.get('plan_ok', True) or error > 0.004 or api.over:
        raise ValueError(fb.get('plan_fail_reason') or 'tracking_error_or_episode_over')


def preflight(api, arm, chain, closed_series=False):
    receipt = api.estimate_tcp_chain(arm, chain)
    if not receipt.get('estimate_ok'):
        return False, receipt
    required = receipt['total_action_steps']
    if closed_series:
        # The shared estimator bundles two gripper operations and a final home.
        # This command executes neither. Use the complete per-stage estimate
        # (including its settling allowance), plus 0.4 s of timing margin.
        rows = receipt.get('stage_costs')
        if (not isinstance(rows, list) or len(rows) != len(chain)
                or [r.get('stage') for r in rows] != [n for n, _ in chain]):
            return False, dict(receipt, reason='incomplete_stage_cost_estimate')
        costs = np.asarray([r['action_steps'] for r in rows], dtype=float)
        if not np.isfinite(costs).all() or np.any(costs < 0) or np.any(costs != np.floor(costs)):
            return False, dict(receipt, reason='invalid_stage_cost_estimate')
        motion_steps = int(costs.sum())
        required = motion_steps + 10
        receipt = dict(receipt, sequence_action_steps=motion_steps,
                       timing_margin_steps=10, required_action_steps=required,
                       home_reserved=False,
                       home_fits_after_sequence=(required + receipt.get('home_action_steps', 0)
                                                 <= receipt['remaining_action_steps']))
    enough = required <= receipt['remaining_action_steps']
    if not enough:
        receipt = dict(receipt, reason='insufficient_action_budget')
    return enough, receipt


def grasp(api, args, result):
    p = points(args['points'])
    if len(p) != 2:
        raise ValueError('exactly two exposed span endpoints required')
    inset = bounded(args, 'inset', .002, 0, .006)
    clearance = bounded(args, 'clearance', .05, .03, .12)
    lift = bounded(args, 'lift', .07, .04, .12)
    axis = p[1]-p[0]
    length = np.linalg.norm(axis)
    if not .015 <= length <= .20 or abs(axis[2]) > .25*length:
        raise ValueError('span must be 15..200 mm and approximately horizontal')
    arm = api.arm(args['arm'])
    pose = arm.tcp().copy()
    down = np.array([0., 0., -1.])
    across = np.cross(down, axis)
    across /= np.linalg.norm(across)
    rotations = [np.column_stack((down, s*across, np.cross(down, s*across))) for s in (1, -1)]
    rotation = max(rotations, key=lambda r: np.trace(pose[:3, :3].T @ r))
    goal = p.mean(0)-[0, 0, inset]
    execute_grasp(api, arm, goal, rotation, clearance, lift, result)


def sphere_grasp(api, args, result):
    clearance = bounded(args, 'clearance', .05, .03, .12)
    lift = bounded(args, 'lift', .07, .04, .12)
    # Always remeasure current simultaneous views: never execute a cached pose
    # or use pose-separated geometry that assumes an already rigid attachment.
    measurement, _ = measure_sphere(api, dict(args, temporal=0))
    result['measurement'] = measurement
    radius = measurement['radius_m']
    if not .006 <= radius <= .025:
        raise ValueError('diameter outside 12..50 mm grasp design envelope')
    arm = api.arm(args['arm'])
    across = arm.tcp()[:3, 1].copy()
    across[2] = 0
    if np.linalg.norm(across) < .1:
        across = np.array([1., 0., 0.])
    across /= np.linalg.norm(across)
    down = np.array([0., 0., -1.])
    rotation = np.column_stack((down, across, np.cross(down, across)))
    # Dimensionless design fraction: raises the TCP while retaining 94% of
    # the equatorial cross-section width. No surface height is assumed.
    goal = np.array(measurement['centre_world']) + [0, 0, .35*radius]
    result['upper_section_fraction'] = .35
    execute_grasp(api, arm, goal, rotation, clearance, lift, result)


def adjacent_span(span, centre, radius):
    """Select only observed support, leaving a visible remote round feature.

    Distance bounds are grasp design limits, not assumed scene dimensions.
    The extrapolated ridge must intersect the measured spherical envelope.
    """
    span = points(span)
    centre = array(centre, (3,))
    radius = float(radius)
    if len(span) != 2 or not np.isfinite(radius) or not .002 <= radius <= .1:
        raise ValueError('invalid adjacent geometry')
    axis = span[1]-span[0]
    length = np.linalg.norm(axis)
    if not .015 <= length <= .20 or abs(axis[2]) > .25*length:
        raise ValueError('adjacent span must be horizontal and 15..200 mm long')
    axis /= length
    distance = (span-centre) @ axis
    radial = np.linalg.norm(span[0]-centre-distance[0]*axis)
    if radial > radius+.004 or distance[0]*distance[1] <= 0:
        raise ValueError('exposed ridge does not extend toward the spherical envelope')
    sign = np.sign(distance.mean())
    # At least 15 mm between the spherical boundary and the selected span;
    # 40..80 mm remote reach keeps fingers away without a long lever arm.
    lower = max(.040, radius+.015, float(np.min(np.abs(distance))))
    upper = min(.080, float(np.max(np.abs(distance))))
    if upper-lower < .015:
        raise ValueError('need >=15 mm exposed overlap 40..80 mm from spherical centre')
    origin = span[0]-distance[0]*axis
    return np.array([origin+sign*lower*axis, origin+sign*upper*axis])


def stem_grasp(api, args, result):
    # Validate scalar inputs before perception or motion. Both measurements
    # precede motion, so their current-view geometry shares a physical pose.
    for name, default, lo, hi in [('inset', .002, 0, .006),
                                 ('clearance', .05, .03, .12), ('lift', .07, .04, .12)]:
        bounded(args, name, default, lo, hi)
    measured, _ = measure_sphere(api, dict(args, temporal=0))
    span, _ = measure_span(api, args)
    selected = adjacent_span(span['points_world'], measured['centre_world'], measured['radius_m'])
    result.update(measurement=measured, span_measurement=span, selected_span_world=selected.tolist())
    grasp(api, dict(args, points=selected), result)
    arm = api.arm(args['arm'])
    baseline = arm._rgb_pick_feature
    # Translation-only lift prediction associates the current image feature;
    # never use it as a returned centre or as evidence of holding.
    predicted = np.asarray(measured['centre_world'])+[0, 0, float(args.get('lift', .07))]
    pose = arm.tcp()
    expected_offset = (predicted-pose[:3, 3]) @ pose[:3, :3]
    try:
        offset, diagnostic = refresh_offset(api, arm, expected_offset, baseline)
    except ValueError as exc:
        result['post_lift_measurement'] = dict(status='unavailable', reason=str(exc))
        result['lift_evidence'] = dict(status='awaiting_post_grasp_measurement', grasp_verified=False)
        return
    centre = np.asarray(diagnostic['centre_world'])
    gain = float(centre[2]-baseline['centre'][2])
    status = 'lift_observed' if gain >= .025 else 'feature_not_lifted'
    baseline.update(status=status, offset=offset.copy())
    arm._rgb_sphere_measurement = dict(centre=centre, radius=diagnostic['radius_m'], hue=baseline['hue'])
    result.update(post_lift_measurement=diagnostic, offset_local=offset.tolist(),
                  lift_evidence=dict(status=status, height_gain_m=gain, grasp_verified=False),
                  note='Current stereo offset measured after lift; rigid holding remains unverified.')
    if status != 'lift_observed':
        raise ValueError('feature_lift_unconfirmed: '+status)


def execute_grasp(api, arm, goal, rotation, clearance, lift, result):
    pose = arm.tcp().copy()
    chain = []
    pose[:3, :3] = rotation
    chain.append(('orient', pose.copy()))
    for name, dz in [('above', clearance), ('descend', 0), ('lift', lift)]:
        pose[:3, 3] = goal+[0, 0, dz]
        chain.append((name, pose.copy()))
    ok, receipt = preflight(api, arm, chain)
    result.update(preflight=receipt, grasp_verified=False)
    if not ok:
        raise ValueError(receipt.get('reason', 'path_unreachable'))
    arm._rgb_temporal = []
    measured = getattr(arm, '_rgb_sphere_measurement', None)
    if measured is not None:
        arm._rgb_pick_feature = dict(measured, status='awaiting_post_grasp_measurement')
    if arm.gripper() < .95:
        api.set_gripper(arm, 1.)
    for name, target in chain:
        if name == 'lift':
            if api.over:
                raise ValueError('episode_over')
            api.set_gripper(arm, 0.)
        moved(api, arm, target, name, result)
    result.update(grasp_world=goal.tolist(), gripper=float(arm.gripper()),
                  note='Closure and TCP lift only; remeasure the held feature to establish lift and local offset.')


def make_chain(pose, rotation, offset, targets, gap, rise, start, pending, compact=False):
    def at(feature):
        t = np.eye(4)
        t[:3, :3] = rotation
        t[:3, 3] = feature-rotation @ offset
        return t
    current = pose[:3, 3]+pose[:3, :3] @ offset
    chain = []
    # Rotate at the TCP, not about the remote feature. A remote pivot adds a
    # large sideways wrist excursion, often outside the reachable workspace.
    relative = rotation @ pose[:3, :3].T
    rotvec = cv2.Rodrigues(relative)[0].reshape(3)
    swept_offsets = [cv2.Rodrigues(rotvec*t)[0] @ pose[:3, :3] @ offset
                     for t in np.linspace(0., 1., 33)]
    # Bound between-sample arc sag by arc length (not just sampled heights).
    arc_margin = np.linalg.norm(rotvec)*np.linalg.norm(offset)/64.
    safe_z = max(float(current[2]), float(np.max(targets[:, 2])+gap+rise))
    if pending is not None:
        safe_z = max(safe_z, float(pending[2]+rise))
    up = pose.copy()
    up[2, 3] = max(pose[2, 3], safe_z-min(v[2] for v in swept_offsets)+arc_margin)
    if up[2, 3]-pose[2, 3] > .001 or pending is not None:
        chain.append(('initial_rise', up))
    if not np.allclose(rotation, pose[:3, :3], atol=1e-5):
        turned = up.copy()
        turned[:3, :3] = rotation
        chain.append(('orient', turned))
    for i in range(start, len(targets)):
        low = targets[i]+[0, 0, gap]
        high = low+[0, 0, rise]
        # After a completed vertical rise, a diagonal approach removes one
        # stop, not any lift or endpoint. The first approach stays elevated.
        if not compact or i == start:
            chain.append((f'above:{i}', at(high)))
        chain.extend([(f'descend:{i}', at(low)), (f'rise:{i}', at(high))])
    return chain


def orientation_candidates(rotation, offset, gap=.023, tcp_clearance=.035):
    # Bounded yaw search first: preserves the measured vertical offset and
    # avoids imposing a downward tool tilt on an otherwise valid level grip.
    yaws = (0, 30, -30, 60, -60, 90, -90, 120, -120)
    for tilt in (0, 15, -15, 30, -30):
        for yaw in yaws:
            r = cv2.Rodrigues(np.array([0., 0., np.deg2rad(yaw)]))[0] @ rotation
            axis = np.cross(r @ offset, [0., 0., 1.])
            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0., 1., 0.])
            axis /= np.linalg.norm(axis)
            r = cv2.Rodrigues(axis*np.deg2rad(tilt))[0] @ r
            yield r, dict(yaw_deg=yaw, tilt_deg=tilt)

    # An upward feature can require more than 30 degrees just to become level.
    # Derive the fallback tilt from its measured vector and clearance margin;
    # also vary tool roll, which changes the IK family without changing grip.
    length = float(np.linalg.norm(offset))
    desired_z = gap-tcp_clearance-.002  # 2 mm slack above the clearance gate
    if length < abs(desired_z):
        return
    desired_elevation = np.arcsin(desired_z/length)
    for roll in (0, 90, -90):
        rolled = rotation @ cv2.Rodrigues(np.array([np.deg2rad(roll), 0., 0.]))[0]
        for yaw in yaws:
            r = cv2.Rodrigues(np.array([0., 0., np.deg2rad(yaw)]))[0] @ rolled
            v = r @ offset
            elevation = np.arctan2(v[2], np.linalg.norm(v[:2]))
            tilt = min(0., float(desired_elevation-elevation))
            if abs(tilt) > np.deg2rad(120):
                continue
            axis = np.cross(v, [0., 0., 1.])
            if np.linalg.norm(axis) < 1e-8:
                axis = np.array([0., 1., 0.])
            axis /= np.linalg.norm(axis)
            r = cv2.Rodrigues(axis*tilt)[0] @ r
            yield r, dict(yaw_deg=yaw, tilt_deg=float(np.rad2deg(tilt)),
                          roll_deg=roll, geometry_derived=True)


def series(api, args, result):
    targets = points(args['points'])
    offset = array(args['offset'], (3,))
    if np.linalg.norm(offset) > .35:
        raise ValueError('offset too large')
    # Leave margin for RGB calibration, rigid-offset and tracking errors.
    # A boundary endpoint is brittle even when the planner reaches it exactly.
    # Centre the nominal endpoint in the stated 10..36 mm interval. The
    # previous 18 mm lower bound left only 8 mm for combined surface, feature
    # and tracking errors; 22..26 mm leaves at least 10 mm on either side.
    gap = bounded(args, 'gap', .023, .022, .026)
    rise = bounded(args, 'rise', .04, .035, .10)
    minimum_clearance = bounded(args, 'tcp_clearance', .035, .020, .060)
    arm = api.arm(args['arm'])
    key = json.dumps([targets.tolist(), gap, rise, minimum_clearance])
    state = getattr(arm, '_rgb_series', None)
    if state is not None:
        if not args.get('resume', 0) or state['key'] != key:
            raise ValueError('existing series: resume=1 and unchanged points/gap/rise required')
    else:
        if args.get('resume', 0):
            raise ValueError('no series to resume')
        state = dict(key=key, next_index=0, completed=[], pending=None, uncertain_index=None)
    result.update(completed_indices=list(state['completed']), next_index=state['next_index'],
                  endpoint_count=len(state['completed']), checker_verified=False)
    if state['uncertain_index'] is not None:
        raise ValueError('uncertain descent at index %d; automatic replay blocked' % state['uncertain_index'])
    if state['next_index'] == len(targets) and state['pending'] is None:
        result['note'] = 'Geometric sequence already complete; no replay.'
        return
    if arm.gripper() > .9:
        raise ValueError('gripper is open')
    evidence = getattr(arm, '_rgb_pick_feature', None)
    if evidence is not None:
        status = evidence['status']
        if status != 'lift_observed':
            raise ValueError('feature_lift_unconfirmed: '+status)
        if not np.allclose(offset, evidence['offset'], atol=.003, rtol=0):
            raise ValueError('offset differs from post-grasp feature measurement')
    pose = arm.tcp().copy()
    rotation = pose[:3, :3]
    attempts = []
    budget_candidates = []
    transition_candidates = []
    chosen = None
    result['minimum_tcp_clearance_m'] = minimum_clearance
    for r, angles in orientation_candidates(rotation, offset, gap, minimum_clearance):
        # IK does not check contact. A TCP only 5--10 mm above the surface
        # permits the fingers to interfere even though the remote point fits.
        # This conservative stand-off is a design margin, not measured finger
        # geometry. Tilting the remote feature downward raises the TCP without
        # changing any feature endpoint or shrinking the inter-endpoint rise.
        tcp_clearance = gap-float((r @ offset)[2])
        if tcp_clearance < minimum_clearance:
            continue
        chain = make_chain(pose, r, offset, targets, gap, rise, state['next_index'], state['pending'])
        ok, receipt = preflight(api, arm, chain, closed_series=True)
        attempts.append(dict(receipt, **angles, tcp_surface_clearance_m=tcp_clearance))
        failed = str(receipt.get('failed_stage', ''))
        if receipt.get('reason') == 'ik_unreachable' and failed.startswith('above:'):
            split = int(failed.split(':')[1])
            if split > state['next_index']:
                transition_candidates.append((split, r.copy(), angles, tcp_clearance, chain))
        if receipt.get('reason') == 'insufficient_action_budget':
            if not any(np.allclose(r, old[1]) for old in budget_candidates):
                budget_candidates.append((receipt['required_action_steps'], r.copy(), angles, tcp_clearance))
        if ok:
            chosen = chain
            result['selected_orientation'] = dict(angles, tcp_surface_clearance_m=tcp_clearance)
            result['selected_budget'] = receipt
            result['selected_path'] = 'vertical_approaches'
            break
    # Bounded fallback only for a reachable but over-budget path. Nearby,
    # nearly level endpoints keep the diagonal above the supplied surfaces;
    # this is still geometric clearance, not a scene collision certificate.
    if (chosen is None and len(targets) > 1 and np.ptp(targets[:, 2]) <= .006
            and np.max(np.linalg.norm(np.diff(targets[:, :2], axis=0), axis=1)) <= .08):
        for _, r, angles, clearance in sorted(budget_candidates, key=lambda row: row[0])[:3]:
            chain = make_chain(pose, r, offset, targets, gap, rise,
                               state['next_index'], state['pending'], compact=True)
            ok, receipt = preflight(api, arm, chain, closed_series=True)
            attempts.append(dict(receipt, **angles, tcp_surface_clearance_m=clearance,
                                 path='diagonal_approaches'))
            if ok:
                chosen = chain
                result['selected_orientation'] = dict(angles, tcp_surface_clearance_m=clearance)
                result['selected_budget'] = receipt
                result['selected_path'] = 'diagonal_approaches'
                break
    # A single orientation may cover only a prefix. Try one reorientation
    # after its last full rise, twisting about the measured feature vector.
    # This preserves the feature offset in world coordinates and TCP clearance;
    # make_chain still bounds the interpolated rotation's swept height. Every
    # candidate includes the original prefix in the full IK/budget estimate.
    if chosen is None and np.linalg.norm(offset) > 1e-6:
        for split, r, angles, clearance, original in sorted(
                transition_candidates, key=lambda row: -row[0])[:3]:
            stop = next(i for i, (name, _) in enumerate(original) if name == f'above:{split}')
            prefix = original[:stop]
            raised = prefix[-1][1]
            axis = r @ offset / np.linalg.norm(offset)
            for twist in (30, -30, 60, -60):
                turned = cv2.Rodrigues(axis*np.deg2rad(twist))[0] @ r
                chain = prefix + make_chain(raised, turned, offset, targets,
                                             gap, rise, split, None)
                ok, receipt = preflight(api, arm, chain, closed_series=True)
                attempts.append(dict(receipt, path='raised_reorientation',
                                     transition_index=split, axial_twist_deg=twist,
                                     **angles, tcp_surface_clearance_m=clearance))
                if ok:
                    chosen = chain
                    result['selected_orientation'] = dict(angles,
                        tcp_surface_clearance_m=clearance,
                        transition_index=split, axial_twist_deg=twist)
                    result['selected_budget'] = receipt
                    result['selected_path'] = 'raised_reorientation'
                    break
            if chosen is not None:
                break
    result['preflight_attempts'] = attempts
    if chosen is None:
        raise ValueError('no full reachable path with TCP clearance and action reserve')
    setattr(arm, '_rgb_series', state)
    refresh_due = True
    for stage_number, (name, target) in enumerate(chosen):
        descending = name.startswith('descend:')
        index = int(name.split(':')[1]) if ':' in name else None
        if descending:
            state['uncertain_index'] = index
        moved(api, arm, target, name, result, offset)
        if name == 'orient':
            refresh_due = True
        # An orientation transition may change the grip again. Refresh at
        # its next elevated waypoint, before any new descent, not just once
        # per command. At most two observations for the bounded joined path.
        measured = getattr(arm, '_rgb_sphere_measurement', None)
        if name.startswith('above:') and refresh_due and measured is not None:
            refresh_due = False
            try:
                updated, diagnostic = refresh_offset(api, arm, offset, measured)
            except ValueError as exc:
                diagnostic = dict(status='unavailable', reason=str(exc),
                                  rigid_offset_assumed=True, stage=name)
                result['offset_refresh'] = diagnostic
                result.setdefault('offset_refreshes', []).append(diagnostic)
            else:
                diagnostic = dict(diagnostic, stage=name)
                result['offset_refresh'] = diagnostic
                result.setdefault('offset_refreshes', []).append(diagnostic)
                remaining = []
                for later, old_target in chosen[stage_number+1:]:
                    corrected = old_target.copy()
                    corrected[:3, 3] += corrected[:3, :3] @ (offset-updated)
                    if later.startswith('descend:') and gap-float((corrected[:3, :3] @ updated)[2]) < minimum_clearance:
                        raise ValueError('refreshed offset violates TCP clearance')
                    remaining.append((later, corrected))
                ok, receipt = preflight(api, arm, remaining, closed_series=True)
                result['refreshed_budget'] = receipt
                if not ok:
                    raise ValueError('refreshed path: '+str(receipt.get('reason', 'unreachable')))
                chosen[stage_number+1:] = remaining
                offset = updated
                if evidence is not None:
                    evidence['offset'] = updated.copy()
                arm._rgb_temporal = []
        reached = arm.tcp()
        feature = reached[:3, 3]+reached[:3, :3] @ offset
        if descending:
            state['completed'].append(index)
            state['next_index'] = index+1
            state['pending'] = feature.tolist()
            state['uncertain_index'] = None
        if name.startswith('rise:') or name == 'initial_rise':
            if state['pending'] is not None:
                if feature[2]-state['pending'][2] < .025:
                    raise ValueError('insufficient_measured_rise')
                state['pending'] = None
        result.update(completed_indices=list(state['completed']), endpoint_count=len(state['completed']),
                      next_index=state['next_index'], lift_pending=state['pending'] is not None)
    result['note'] = 'Ordered geometric endpoints and lifts completed; grip rigidity assumed, checker state unavailable.'


def run(api, command, args):
    result = dict(plan_ok=False, plan_fail_reason=None, stages=[])
    try:
        if command == 'rgb_sphere':
            return measure_sphere(api, args)
        if command == 'rgb_points':
            return measure(api, args)
        if command == 'rgb_span':
            return measure_span(api, args)
        if command == 'span_pick':
            grasp(api, args, result)
        elif command == 'sphere_pick':
            sphere_grasp(api, args, result)
        elif command == 'stem_pick':
            stem_grasp(api, args, result)
        elif command == 'point_series':
            series(api, args, result)
        else:
            raise ValueError('unknown command')
        result.update(plan_ok=True, plan_fail_reason=None)
        return result, 0
    except Exception as exc:
        result.update(plan_ok=False, plan_fail_reason=str(exc) or type(exc).__name__)
        if command == 'point_series':
            try:
                state = getattr(api.arm(args['arm']), '_rgb_series', None)
                if state:
                    result.update(next_index=state['next_index'], completed_indices=list(state['completed']),
                                  endpoint_count=len(state['completed']), uncertain_index=state['uncertain_index'],
                                  lift_pending=state['pending'] is not None, checker_verified=False)
            except Exception:
                pass
        return result, 2
