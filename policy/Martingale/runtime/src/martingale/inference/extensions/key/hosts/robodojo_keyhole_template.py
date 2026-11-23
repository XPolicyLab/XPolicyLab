# Copyright © 2026 Martingale. All Rights Reserved.
import hashlib
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
try:
    from martingale.inference.extensions.key.hosts import robodojo_keyhole as geometry
    from martingale.inference.extensions.key.hosts import robodojo_keyhole_uncertainty as uncertainty
except ImportError:
    import keyhole as geometry
    import keyhole_uncertainty as uncertainty

SHAPE_KEYS = ('radius_m', 'slot_half_width_m', 'slot_extent_m')
HEIGHT_KEY = 'height_above_table_m'
NO_GEOMETRY = 'No opening geometry has been given in this episode. Pass opening_geometry once: a range [low, high] in metres for each of ' + ', '.join((HEIGHT_KEY, *SHAPE_KEYS)) + '.'

def _interval(name, value, spread):
    """(low, high) in metres of a length given as [low, high], or as one number +/- spread."""
    array = np.asarray(value, float).ravel()
    if array.size == 1:
        array = np.array([array[0] - spread, array[0] + spread])
    if array.size != 2 or not np.isfinite(array).all() or not 0 < array[0] <= array[1] < 0.5:
        raise ValueError(name + ' must be a range [low, high] in metres (or one measured length), positive and below 0.5 m.')
    return (float(array[0]), float(array[1]))

def episode_geometry(primitives, opening_geometry=None):
    """Opening shape and rim height as ranges supplied by the caller in the CURRENT episode.

    Nothing is stored with the package or carried between episodes.
    opening_geometry gives a range [low, high] in metres for each of:
    height_above_table_m (the opening's rim above the table top), radius_m
    (round shaft entry), slot_half_width_m, and slot_extent_m (from the round
    entry's centre to the slot's far end). Each is a general range, or one
    narrowed by measuring this episode's images. A single number is read as
    that value +/- uncertainty_m (default 0.5 mm, allowed 0.1..3 mm). The pose
    is fitted at the middle of every range and both ends of each range enter
    the reported uncertainty, so wide ranges give a wide opening uncertainty.
    The ranges are kept on this episode's primitives until others replace
    them. Returns None when none have been supplied yet.
    """
    from martingale.hosts import robodojo_tools as tools
    if opening_geometry is None:
        return getattr(primitives, '_keyhole_geometry', None)
    if not isinstance(opening_geometry, dict):
        raise ValueError('opening_geometry must be a dictionary of ranges [low, high] in metres.')
    missing = [name for name in (HEIGHT_KEY, *SHAPE_KEYS) if name not in opening_geometry]
    if missing:
        raise ValueError('opening_geometry lacks: ' + ', '.join(missing))
    spread = float(opening_geometry.get('uncertainty_m', 0.0005))
    if not np.isfinite(spread) or not 0.0001 <= spread <= 0.003:
        raise ValueError('uncertainty_m must be between 0.1 and 3 mm.')
    ranges = {name: _interval(name, opening_geometry[name], spread) for name in (HEIGHT_KEY, *SHAPE_KEYS)}
    middle = {name: (low + high) / 2 for name, (low, high) in ranges.items()}
    half = {name: (high - low) / 2 for name, (low, high) in ranges.items()}
    if middle['slot_half_width_m'] >= middle['radius_m']:
        raise ValueError('The slot must be narrower than the round entry: slot_half_width_m below radius_m.')
    ident = hashlib.sha256(json.dumps([[round(end, 6) for end in ranges[name]] for name in (HEIGHT_KEY, *SHAPE_KEYS)]).encode()).hexdigest()[:16]
    model = dict({name: middle[name] for name in SHAPE_KEYS}, rim_z_m=tools.kin.TABLE_Z + middle[HEIGHT_KEY], height_uncertainty_m=half[HEIGHT_KEY], shape_uncertainty_m={name: half[name] for name in SHAPE_KEYS}, ranges_m={name: list(ends) for name, ends in ranges.items()}, geometry_id=ident, source='ranges supplied by the caller in this episode')
    primitives._keyhole_geometry = model
    return model

def boundary(radius, half_width, extent):
    angle = np.arcsin(np.clip(half_width / radius, 0.01, 0.99))
    theta = np.linspace(angle, 2 * np.pi - angle, 96)
    arc = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    a = arc[-1]
    b = np.array([extent, -half_width])
    c = np.array([extent, half_width])
    d = arc[0]
    return np.vstack([arc, *[x + (y - x) * np.linspace(0, 1, 40)[:, None] for x, y in [(a, b), (b, c), (c, d)]]])

def world_boundary(x, y, z, yaw, shape):
    local = boundary(*shape)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return np.c_[local @ rotation.T + [x, y], np.full(len(local), z)]

def detect(image, seed, radius_px=80):
    image = np.asarray(image)[..., :3]
    h, w = image.shape[:2]
    seed = np.asarray(seed, float)
    if seed.shape != (2,) or not np.isfinite(seed).all():
        raise ValueError('Need a finite image seed.')
    x, y = np.rint(seed).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Opening is outside this image.')
    x0, x1 = (max(0, x - radius_px), min(w, x + radius_px + 1))
    y0, y1 = (max(0, y - radius_px), min(h, y + radius_px + 1))
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    threshold = min(75.0, max(18.0, 0.45 * float(np.percentile(gray, 70))))
    mask = (gray < threshold).astype('uint8')
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, len(stats)):
        bx, by, bw, bh, area = stats[label]
        if area < 6 or area > 8000 or bx == 0 or (by == 0) or (bx + bw >= mask.shape[1]) or (by + bh >= mask.shape[0]):
            continue
        component = (labels == label).astype('uint8')
        contour = max(cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0].astype(float) + [x0, y0]
        distance = float(np.linalg.norm(contour - seed, axis=1).min())
        if distance > 30:
            continue
        center = contour.mean(0)
        delta = contour - center
        contour += 0.5 * delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1.0)
        candidates.append((distance, dict(contour=contour, area=int(area), threshold=threshold)))
    if not candidates:
        raise ValueError('No isolated aperture near the selected lock center.')
    return min(candidates, key=lambda x: x[0])[1]

def residual(params, detections, views, shape=None, fixed_z=None):
    if shape is None:
        x, y, z, yaw, r, b, l = params
        shape = np.array([r, b, l]) * 0.001
    else:
        x, y, yaw = params
        z = fixed_z * 1000
    points = world_boundary(x * 0.001, y * 0.001, z * 0.001, yaw, shape)
    values = []
    for name, d in detections.items():
        predicted, depth = geometry.project(views[name], points)
        observed = d['contour']
        tree = cKDTree(predicted)
        distance, idx = tree.query(observed)
        reverse, ri = cKDTree(observed).query(predicted)
        values.extend(((observed - predicted[idx]) / np.sqrt(len(observed))).ravel())
        values.extend(((predicted - observed[ri]) / np.sqrt(len(predicted))).ravel())
    return np.asarray(values)

def polygon_center(points):
    origin = np.mean(points, axis=0)
    a = points - origin
    b = np.roll(a, -1, axis=0)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if abs(cross.sum()) < 1e-15:
        return origin
    return origin + ((a + b) * cross[:, None]).sum(0) / (3 * cross.sum())

def solve_pose(detection, view, shape, z, initial=None, diagnostics=None):
    contour = detection['contour']
    world = geometry.on_plane(view, contour, z)
    center = world[:, :2].mean(0)
    _, _, axes = np.linalg.svd(world[:, :2] - center, full_matrices=False)
    yaw = np.arctan2(axes[0, 1], axes[0, 0])
    seeds = []
    if initial is not None:
        seeds = [initial]
    else:
        center = polygon_center(world[:, :2])
        local_center = polygon_center(boundary(*shape))
        for angle in [yaw + offset for offset in (-0.3, 0, 0.3, np.pi - 0.3, np.pi, np.pi + 0.3)]:
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            seeds.append(np.r_[(center - rotation @ local_center) * 1000, angle])
    best = None
    solutions = []
    for seed in seeds:
        fit = least_squares(residual, seed, args=({'view': detection}, {'view': view}, shape, z), x_scale=[10.0, 10.0, 1.0], diff_step=0.0001, max_nfev=150)
        solutions.append(fit)
        if best is None or fit.cost < best.cost:
            best = fit
    if diagnostics is not None:
        opposite = [s for s in solutions if np.cos(s.x[2] - best.x[2]) < 0]
        gap = min((np.sqrt(s.cost) - np.sqrt(best.cost) for s in opposite), default=float('inf'))
        diagnostics['heading_gap_px'] = float(gap)
        diagnostics['heading_resolved'] = bool(gap > 0.5)
    points = world_boundary(best.x[0] * 0.001, best.x[1] * 0.001, z, best.x[2], shape)
    projected, _ = geometry.project(view, points)
    d = cKDTree(projected).query(contour)[0]
    back = cKDTree(contour).query(projected)[0]
    rms = float(np.sqrt((np.mean(d * d) + np.mean(back * back)) / 2))
    return (best.x, rms, points)

def fit(images, views, entry_pixels, axis_pixels=None, *, template=None):
    """Pose of the opening from the shape and rim-height ranges given in this episode.

    template is episode_geometry's result. One useful view suffices;
    low-resolution support never vetoes it.
    """
    model = template
    if model is None:
        return dict(accepted=False, status='needs_observation', reasons=[NO_GEOMETRY])
    shape = np.array([model[name] for name in SHAPE_KEYS])
    z = model['rim_z_m']
    attempts = []
    if not entry_pixels:
        return dict(accepted=False, status='needs_observation', reasons=['Select the lock center in one current image.'])
    for name, seed in entry_pixels.items():
        if name not in images or name not in views:
            continue
        try:
            detection = detect(images[name], seed)
            heading = {}
            p, rms, outline = solve_pose(detection, views[name], shape, z, diagnostics=heading)
            span = float(np.linalg.norm(np.ptp(detection['contour'], axis=0)))
            attempts.append(dict(camera=name, parameters=p, rms_px=rms, span_px=span, detection=detection, outline=outline, **heading))
        except ValueError:
            continue
    if not attempts:
        return dict(accepted=False, status='needs_observation', reasons=['Opening not visible in supplied images.'], next_observation='One close overhead wrist view of the unoccupied opening.')
    viable = [a for a in attempts if a['rms_px'] <= 2.5 and a['heading_resolved'] and (a['span_px'] >= 25)]
    chosen = max(viable or attempts, key=lambda a: a['span_px'] / max(1.0, a['rms_px']))
    name = chosen['camera']
    p = chosen['parameters']
    view = views[name]
    det = chosen['detection']
    entry = np.r_[p[:2] * 0.001, z]
    direction = np.array([np.cos(p[2]), np.sin(p[2]), 0.0])
    entries = [entry.tolist()]
    directions = [direction.tolist()]
    cases = []
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        cases.append((det['contour'] + [dx, dy], shape, z))
    for sign in [-1, 1]:
        cases.append((det['contour'], shape, z + sign * model['height_uncertainty_m']))
        for j, key in enumerate(['radius_m', 'slot_half_width_m', 'slot_extent_m']):
            varied = shape.copy()
            varied[j] += sign * model['shape_uncertainty_m'][key]
            cases.append((det['contour'], varied, z))
    for contour, ss, zz in cases:
        pp, _, _ = solve_pose(dict(det, contour=contour), view, ss, zz, initial=p)
        entries.append([pp[0] * 0.001, pp[1] * 0.001, zz])
        directions.append([np.cos(pp[2]), np.sin(pp[2]), 0.0])
    samples = dict(entry_samples_xyz=entries, direction_samples_xyz=directions)
    accuracy = uncertainty.opening_accuracy(samples, entry, direction)
    reasons = []
    if chosen['rms_px'] > 2.5:
        reasons.append('Image does not match the middle of the opening shape ranges; check the view, or measure the opening in current images and pass narrower ranges.')
    if chosen['span_px'] < 25:
        reasons.append('Need one closer view to resolve the opening pose.')
    if not chosen['heading_resolved']:
        reasons.append('Round entry and slot end are ambiguous in this view; use one closer overhead view.')
    if accuracy['irreducible_limits_exceeded']:
        reasons.extend(accuracy['reasons'])
    accepted = not reasons
    return dict(accepted=accepted, status='ready' if accepted else 'needs_observation', reasons=reasons, entry_xyz=entry.tolist(), observation_seed_only=not accepted, rim_z=z, normal_xyz=[0.0, 0.0, 1.0], tooth_direction_xyz=direction.tolist(), slot_axis_xyz=direction.tolist(), long_axis_xyz=direction.tolist(), orientation_resolved=accepted, orientation_reasons=[], outline_xyz=chosen['outline'].tolist(), reference_camera=name, supporting_views=[name], contour_rms_px=chosen['rms_px'], span_px=chosen['span_px'], height_sensitivity_m=model['height_uncertainty_m'], uncertainty=samples, opening_accuracy=accuracy, geometry_id=model['geometry_id'], measurement_method='measured_shape_pose', view_diagnostics=[dict(camera=a['camera'], rms_px=a['rms_px'], span_px=a['span_px'], heading_gap_px=a['heading_gap_px']) for a in attempts], next_observation=None if accepted else 'One close overhead wrist view; keep the opening unoccupied.', note='Shape and rim height are the middles of the ranges supplied in this episode, whose ends enter the uncertainty; the current image estimates entry and directed slot pose. Low-resolution secondary contours never veto a precise primary view.')

def quick_calibrate(primitives, refresh=False, opening_geometry=None):
    """Bounded pre-pickup observation: cached result, coarse RGB seed, close view."""
    from martingale.hosts import robodojo_tools as tools
    try:
        model = episode_geometry(primitives, opening_geometry)
    except (ValueError, TypeError) as exc:
        return dict(accepted=False, status='needs_observation', steps_used=0, reasons=[str(exc)])
    if model is None:
        return dict(accepted=False, status='needs_observation', steps_used=0, reasons=[NO_GEOMETRY])
    cached = getattr(primitives, '_keyhole_opening_cache', None)
    if cached and (not refresh) and (cached.get('geometry_id') == model['geometry_id']):
        return dict(cached, steps_used=0, reused=True)
    if getattr(primitives, '_key_observed_grips', ()):
        return dict(accepted=False, status='needs_observation', steps_used=0, reasons=['Automatic camera calibration is for empty arms before pickup; use measure_keyhole on current views or the cached opening.'])
    sight = tools.locate(primitives, 'the brown cylindrical lock')
    if not sight.get('found') or not sight.get('centroid_rc'):
        return dict(accepted=False, status='needs_observation', steps_used=0, reasons=['Locate the lock in the current head image.'])
    obs, images, views = geometry.current(primitives)
    head_seed = list(sight['centroid_rc'])[::-1]
    first = fit(images, views, {'cam_head': head_seed}, template=model)
    if first.get('accepted'):
        primitives._keyhole_opening_cache = first
        return dict(first, steps_used=0, reused=False)
    if first.get('entry_xyz') is None:
        return dict(first, steps_used=0, reused=False)
    aim = np.array(first['entry_xyz'])
    arm = 'left' if aim[0] < 0 else 'right'
    camera = 'cam_' + arm + '_wrist'
    starts = {a: tools._joints(obs['state'], a) for a in tools.ARMS}
    moves = []
    steps = 0
    for pitch, distance in [(30.0, 0.13), (45.0, 0.12)]:
        state = tools._obs(primitives)['state']
        view = tools._view(camera, state)
        current = tools.kin.rot_from_quat(state[arm + '_ee_pose'][3:])
        tcp = tools.kin.tcp(tools.kin.fk(arm, tools._joints(state, arm)))
        desired = tools.kin.orientation(pitch, 0.0, 0.0)
        mount_r = current.T @ view.pose[:3, :3]
        mount_t = current.T @ (view.pos - tcp)
        ray = desired @ mount_r @ np.array([0.0, 0.25, -1.0])
        ray /= np.linalg.norm(ray)
        target = aim - distance * ray - desired @ mount_t
        target_pose, _ = tools._target(arm, tools._joints(state, arm), target, pitch, 0.0, 0.0)
        ways, _, error, note = tools._plan(arm, tools._joints(state, arm), target_pose, 'line', starts)
        if error or note:
            moves.append(dict(reached=False, reason=error or note, steps_used=0))
            continue
        active = getattr(primitives, '_key_interarm_guard', False)
        try:
            primitives._key_interarm_guard = True
            motion = tools.move_pose(primitives, starts, target.tolist(), arm=arm, pitch=pitch, yaw=0.0, roll=0.0, gripper=1.0, path='line')
        finally:
            primitives._key_interarm_guard = active
        moves.append(motion)
        steps += motion.get('steps_used', 0)
        if not motion.get('reached'):
            continue
        _, images, views = geometry.current(primitives)
        pixel, _ = geometry.project(views[camera], aim[None])
        measured = fit(images, views, {camera: pixel[0].tolist(), 'cam_head': head_seed}, template=model)
        first = measured
        if measured.get('accepted'):
            primitives._keyhole_opening_cache = measured
            return dict(measured, steps_used=steps, reused=False, camera_moves=moves)
    return dict(first, accepted=False, status='needs_observation', steps_used=steps, reused=False, camera_moves=moves, next_observation='Inspect the latest close wrist image and select its aperture; do not repeat a camera tour.')
