import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ANGLES = np.linspace(0, 2 * np.pi, 128, endpoint=False)
RING = np.c_[np.cos(ANGLES), np.sin(ANGLES)]
DIRECTIONS = RING[::2]

def project(view, points):
    row, col, depth = view.project(points)
    return (np.c_[col, row], depth)

def basis(yaw, tilt):
    normal = np.array([np.cos(tilt) * np.cos(yaw), np.cos(tilt) * np.sin(yaw), np.sin(tilt)])
    u = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    v = np.cross(normal, u)
    return np.column_stack([u, v, normal])

def disk_vertices(parameters):
    center = np.asarray(parameters[:3]) * 0.001
    r, t = np.asarray(parameters[3:5]) * 0.001
    frame = basis(*parameters[5:])
    circle = RING @ frame[:, :2].T * r
    return np.vstack([center + circle + sign * t / 2 * frame[:, 2] for sign in (-1, 1)])

def observation(name, view, mask, occluder=None):
    mask = np.asarray(mask, bool)
    unknown = np.zeros_like(mask) if occluder is None else np.asarray(occluder, bool)
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No coin pixels in ' + name)
    contour = max(contours, key=cv2.contourArea)[:, 0].astype(float)
    if len(contour) < 12 or mask.sum() < 25:
        raise ValueError('Coin too small in ' + name)
    values = contour @ DIRECTIONS.T
    indices = values.argmax(0)
    support = values[indices, np.arange(len(DIRECTIONS))] + 0.5
    rim = contour[indices].astype(int)
    near = cv2.dilate((unknown & ~mask).astype('uint8'), np.ones((5, 5), np.uint8)) > 0
    complete = ~near[rim[:, 1], rim[:, 0]]
    h, w = mask.shape
    complete &= (rim[:, 0] > 1) & (rim[:, 0] < w - 2) & (rim[:, 1] > 1) & (rim[:, 1] < h - 2)
    return dict(name=name, view=view, mask=mask, unknown=unknown, support=support, complete=complete)

def disk_residual(p, observations):
    xyz = disk_vertices(p)
    values = []
    for ob in observations:
        pixels, depth = project(ob['view'], xyz)
        delta = np.max(pixels @ DIRECTIONS.T, axis=0) - ob['support']
        values.extend(np.where(ob['complete'], delta, np.minimum(delta, 0.0)))
        values.append(1000.0 * min(float(depth.min()) - 0.01, 0.0))
    return np.asarray(values)

def silhouette_iou(p, ob):
    pixels, depth = project(ob['view'], disk_vertices(p))
    predicted = np.zeros(ob['mask'].shape, np.uint8)
    if depth.min() <= 0:
        return 0.0
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    cv2.fillConvexPoly(predicted, hull, 1)
    valid = ~ob['unknown']
    a = predicted.astype(bool) & valid
    b = ob['mask'] & valid
    return float((a & b).sum() / max(1, (a | b).sum()))

def fit_coin(observations, center_seed_xyz):
    if len(observations) < 2:
        return dict(accepted=False, reasons=['Need two calibrated views of the same coin.'])
    center = np.asarray(center_seed_xyz, float) * 1000
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite world XYZ seed required.')
    radii = []
    for ob in observations:
        r, c = np.nonzero(ob['mask'])
        radius_px = 0.25 * (np.ptp(r) + np.ptp(c))
        radii.append(radius_px * np.linalg.norm(center * 0.001 - ob['view'].pos) / ob['view'].fx * 1000)
    radius = np.clip(np.median(radii), 3, 50)
    low = np.r_[center - [65, 65, 80], 2, 0.2, -2 * np.pi, -np.pi / 2]
    high = np.r_[center + [65, 65, 80], 60, 15, 2 * np.pi, np.pi / 2]
    fits = []
    for yaw, tilt in itertools.product(np.linspace(-np.pi, np.pi, 8, endpoint=False), [0.0, 0.6]):
        seed = np.r_[center, radius, 2.0, yaw, tilt]
        fit = least_squares(disk_residual, seed, args=(observations,), bounds=(low, high), x_scale=[20, 20, 20, 15, 2, 1, 1], loss='soft_l1', f_scale=1.0, max_nfev=140, diff_step=1e-05)
        fits.append(fit)
    fits.sort(key=lambda f: np.mean(disk_residual(f.x, observations) ** 2))
    best = fits[0]
    p = best.x
    errors = disk_residual(p, observations).reshape(len(observations), -1)[:, :-1]
    rms = np.sqrt(np.mean(errors ** 2, axis=1))
    ious = {ob['name']: silhouette_iou(p, ob) for ob in observations}
    reasons = []
    rays = [p[:3] * 0.001 - ob['view'].pos for ob in observations]
    rays = [r / np.linalg.norm(r) for r in rays]
    parallax = max((np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in itertools.combinations(rays, 2)))
    if parallax < 10:
        reasons.append('Insufficient camera separation.')
    if max(rms) > 2.0:
        reasons.append('Visible contours disagree with a circular coin.')
    if min(ious.values()) < 0.65:
        reasons.append('Coin silhouette agreement is insufficient.')
    if p[4] <= 0.21 or p[4] >= 14.9:
        reasons.append('Thickness is at a fitting bound.')
    if p[4] > p[3]:
        reasons.append('Not a thin disk.')
    if sum((np.count_nonzero(ob['complete']) for ob in observations)) < 32:
        reasons.append('Too little unoccluded rim evidence.')
    samples = [p]
    for j in range(len(observations)):
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            changed = [dict(ob) for ob in observations]
            changed[j]['support'] = changed[j]['support'] + DIRECTIONS @ shift
            f = least_squares(disk_residual, p, args=(changed,), bounds=(low, high), x_scale=[20, 20, 20, 15, 2, 1, 1], max_nfev=80)
            samples.append(f.x)
    for f in fits[1:]:
        if np.sqrt(np.mean(disk_residual(f.x, observations) ** 2)) <= np.sqrt(np.mean(errors ** 2)) + 0.5:
            samples.append(f.x)
    normals = np.array([basis(*s[5:])[:, 2] for s in samples])
    normal = basis(*p[5:])[:, 2]
    center_spread = float(np.max(np.linalg.norm(np.asarray(samples)[:, :3] - p[:3], axis=1)) * 0.001)
    normal_spread = float(np.max(np.degrees(np.arccos(np.clip(np.abs(normals @ normal), -1, 1)))))
    thickness_spread = float(np.max(np.abs(np.asarray(samples)[:, 4] - p[4])) * 0.001)
    if center_spread > 0.005 or normal_spread > 10 or thickness_spread > 0.003:
        reasons.append('Coin geometry remains ambiguous at pixel resolution.')
    return dict(accepted=not reasons, reasons=reasons, kind='oriented_disk', parameters=p.tolist(), center_xyz=(p[:3] * 0.001).tolist(), normal_xyz=normal.tolist(), radius_m=float(p[3] * 0.001), thickness_m=float(p[4] * 0.001), view_errors_px={ob['name']: float(e) for ob, e in zip(observations, rms)}, view_iou=ious, view_separation_deg=float(parallax), center_sensitivity_m=center_spread, normal_sensitivity_deg=normal_spread, thickness_sensitivity_m=thickness_spread, parameter_samples=[s.tolist() for s in samples], normal_sign_ambiguous=True, note='Rotation about a circular coin normal is unobservable and unnecessary. No grasp, clearance or insertion success is certified.')

def triangulate(views, pixels):
    matrices = []
    origins = []
    for view, pixel in zip(views, pixels):
        ray = view.rays([pixel[1]], [pixel[0]])[0]
        ray /= np.linalg.norm(ray)
        a = np.eye(3) - np.outer(ray, ray)
        matrices.append(a)
        origins.append(a @ view.pos)
    return np.linalg.solve(sum(matrices), sum(origins))

def slot_corners(parameters):
    rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
    local = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]) * np.r_[parameters[6:8] / 2, 0] * 0.001
    return local @ rotation.T + parameters[:3] * 0.001

def fit_slot(views, annotations):
    """Fit an oriented rectangular entrance; correspondence is physical corner ID."""
    names = list(annotations)
    if len(names) < 2:
        return dict(accepted=False, reasons=['Need corresponding rim corners in two views.'])
    values = {n: np.asarray(annotations[n], float) for n in names}
    if any((p.shape != (4, 2) or not np.isfinite(p).all() for p in values.values())):
        raise ValueError('Four finite (column,row) corners per view are required.')
    initial = np.array([triangulate([views[n] for n in names], [values[n][i] for n in names]) for i in range(4)])
    u = (initial[2] + initial[3] - initial[0] - initial[1]) / 2
    length = np.linalg.norm(u)
    u /= length
    v = (initial[1] + initial[2] - initial[0] - initial[3]) / 2
    width = np.linalg.norm(v)
    v -= u * (v @ u)
    v /= np.linalg.norm(v)
    rotation = np.column_stack([u, v, np.cross(u, v)])
    seed = np.r_[initial.mean(0) * 1000, Rotation.from_matrix(rotation).as_rotvec(), length * 1000, width * 1000]

    def residual(p, obs):
        return np.concatenate([(project(views[n], slot_corners(p))[0] - obs[n]).ravel() for n in names])
    low = np.r_[seed[:3] - 50, [-10] * 3, 2, 0.2]
    high = np.r_[seed[:3] + 50, [10] * 3, 200, 100]
    best = least_squares(residual, np.clip(seed, low + 1e-06, high - 1e-06), args=(values,), bounds=(low, high), x_scale=[20] * 3 + [1] * 3 + [30, 5], loss='soft_l1', f_scale=1.0, max_nfev=200)
    p = best.x
    errors = residual(p, values).reshape(len(names), 4, 2)
    rms = np.sqrt(np.mean(errors ** 2))
    reasons = []
    if max(np.linalg.norm(errors, axis=2).ravel()) > 2.0:
        reasons.append('Rim corner identities or planar model disagree across views.')
    rays = [p[:3] * 0.001 - views[n].pos for n in names]
    rays = [r / np.linalg.norm(r) for r in rays]
    parallax = max((np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in itertools.combinations(rays, 2)))
    if parallax < 10:
        reasons.append('Slot views lack parallax.')
    if np.linalg.matrix_rank(best.jac) < 8:
        reasons.append('Slot dimensions or plane are unresolved.')
    samples = [p]
    for n in names:
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            changed = dict(values)
            changed[n] = values[n] + shift
            fit = least_squares(residual, p, args=(changed,), bounds=(low, high), x_scale=[20] * 3 + [1] * 3 + [30, 5], max_nfev=100)
            samples.append(fit.x)
    rotation = Rotation.from_rotvec(p[3:6]).as_matrix()
    normal = rotation[:, 2]
    if normal[2] < 0:
        normal = -normal
    spread = float(np.max(np.linalg.norm(np.asarray(samples)[:, :3] - p[:3], axis=1)) * 0.001)
    if spread > 0.003:
        reasons.append('Slot center sensitivity exceeds 3 mm.')
    return dict(accepted=not reasons, reasons=reasons, kind='oriented_slot_rim', center_xyz=(p[:3] * 0.001).tolist(), normal_xyz=normal.tolist(), long_axis_xyz=rotation[:, 0].tolist(), short_axis_xyz=rotation[:, 1].tolist(), length_m=float(p[6] * 0.001), width_m=float(p[7] * 0.001), corners_xyz=slot_corners(p).tolist(), reprojection_rms_px=float(rms), view_separation_deg=float(parallax), center_sensitivity_m=spread, parameter_samples=[s.tolist() for s in samples], note='Visible entrance rim only; internal channel and collision clearance are not measured.')

def coin_plane_half_extent(radius, thickness, normal, direction):
    """Exact support of a circular cylinder along a unit world direction."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    cosine = float(np.clip(n @ d, -1, 1))
    return float(radius * np.sqrt(max(0.0, 1 - cosine * cosine)) + thickness / 2 * abs(cosine))

def rounded_slot_vertices(p):
    """Planar rounded rectangular rim; p[8] is corner radius / short width."""
    length, width = p[6:8] * 0.001
    radius = width * p[8]
    points = []
    for x, y, start in [(1, 1, 0), (-1, 1, np.pi / 2), (-1, -1, np.pi), (1, -1, 3 * np.pi / 2)]:
        angles = np.linspace(start, start + np.pi / 2, 33)
        points.extend(np.c_[x * (length / 2 - radius) + radius * np.cos(angles), y * (width / 2 - radius) + radius * np.sin(angles), np.zeros(len(angles))])
    return np.asarray(points) @ Rotation.from_rotvec(p[3:6]).as_matrix().T + p[:3] * 0.001

def fit_rounded_slot(observations, rectangle_seed):
    """Fit the observed rim, without treating rounded extrema as physical corners."""
    seed = np.asarray(rectangle_seed['parameter_samples'][0])
    seed = np.r_[seed, 0.2]
    low = np.r_[seed[:3] - 30, [-10] * 3, 2, 0.2, 0.0]
    high = np.r_[seed[:3] + 30, [10] * 3, 200, 100, 0.5]

    def residual(p, obs):
        xyz = rounded_slot_vertices(p)
        return np.concatenate([np.max(project(o['view'], xyz)[0] @ DIRECTIONS.T, axis=0) - o['support'] for o in obs])
    fits = []
    for fraction in [0.05, 0.2, 0.45]:
        start = seed.copy()
        start[8] = fraction
        fits.append(least_squares(residual, start, args=(observations,), bounds=(low, high), x_scale=[20] * 3 + [1] * 3 + [30, 5, 0.2], loss='soft_l1', f_scale=1.0, max_nfev=300))
    best = min(fits, key=lambda f: np.mean(residual(f.x, observations) ** 2))
    p = best.x
    rms = np.sqrt(np.mean(residual(p, observations).reshape(len(observations), -1) ** 2, axis=1))
    samples = [p]
    for i in range(len(observations)):
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            changed = [dict(o) for o in observations]
            changed[i]['support'] = changed[i]['support'] + DIRECTIONS @ shift
            trial = least_squares(residual, p, args=(changed,), bounds=(low, high), x_scale=[20] * 3 + [1] * 3 + [30, 5, 0.2], max_nfev=150)
            samples.append(trial.x)
    rotation = Rotation.from_rotvec(p[3:6]).as_matrix()
    normal = rotation[:, 2]
    if normal[2] < 0:
        normal = -normal
    spread = float(np.max(np.linalg.norm(np.asarray(samples)[:, :3] - p[:3], axis=1)) * 0.001)
    normal_spread = max((float(np.degrees(np.arccos(np.clip(abs(Rotation.from_rotvec(s[3:6]).as_matrix()[:, 2] @ normal), -1, 1)))) for s in samples))
    reasons = []
    if len(observations) < 3:
        reasons.append('Need three distinct views to check narrow-rim orientation.')
    if max(rms) > 2:
        reasons.append('Rounded slot contours disagree across views.')
    if spread > 0.003 or normal_spread > 10:
        reasons.append('Slot geometry is ambiguous at pixel resolution.')
    if rectangle_seed['view_separation_deg'] < 10:
        reasons.append('Slot views lack parallax.')
    return dict(accepted=not reasons, reasons=reasons, kind='rounded_slot_rim', center_xyz=(p[:3] * 0.001).tolist(), normal_xyz=normal.tolist(), long_axis_xyz=rotation[:, 0].tolist(), short_axis_xyz=rotation[:, 1].tolist(), length_m=float(p[6] * 0.001), width_m=float(p[7] * 0.001), corner_radius_m=float(p[7] * p[8] * 0.001), view_errors_px={o['name']: float(e) for o, e in zip(observations, rms)}, view_separation_deg=rectangle_seed['view_separation_deg'], center_sensitivity_m=spread, normal_sensitivity_deg=normal_spread, parameter_samples=[s.tolist() for s in samples], note='RGB silhouette fit of visible rounded rim; internal channel is not estimated.')

def disk_ray_depth(parameters, view, rows, cols):
    """Visible ray intersections with both faces and the circular edge wall."""
    p = np.asarray(parameters, float)
    frame = basis(*p[5:])
    origin = (view.pos - p[:3] * 0.001) @ frame
    rays = view.rays(rows, cols) @ frame
    r, t = p[3:5] * 0.001
    count = len(rays)
    possible = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for side in (-1, 1):
            depth = (side * t / 2 - origin[2]) / rays[:, 2]
            xy = origin[:2] + rays[:, :2] * depth[:, None]
            valid = (depth > 0) & (np.sum(xy * xy, axis=1) <= r * r + 1e-12)
            possible.append(np.where(valid, depth, np.inf))
        a = np.sum(rays[:, :2] ** 2, axis=1)
        b = 2 * rays[:, :2] @ origin[:2]
        c = origin[:2] @ origin[:2] - r * r
        disc = b * b - 4 * a * c
        for side in (-1, 1):
            depth = (-b + side * np.sqrt(np.maximum(disc, 0))) / (2 * a)
            z = origin[2] + rays[:, 2] * depth
            valid = (a > 1e-16) & (disc >= 0) & (depth > 0) & (abs(z) <= t / 2 + 1e-12)
            possible.append(np.where(valid, depth, np.inf))
    out = np.min(np.asarray(possible), axis=0)
    out[~np.isfinite(out)] = np.nan
    return out

def fit_coin_grasp(observations, center_seed_xyz, radius_m, thickness_m):
    """Fit only XYZ and face yaw of a known upright coin before pickup.

    Acceptance is for a holder-aware grasp, not measured dimensions or slot
    entry. Multiple yaw starts and pixel perturbations guard pose ambiguity.
    """
    if len(observations) < 2:
        return dict(accepted=False, grasp_accepted=False, reasons=['Need two distinct views for the grasp pose.'])
    center = np.asarray(center_seed_xyz, float) * 1000
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite world XYZ seed required.')
    if not 0 < thickness_m < radius_m:
        raise ValueError('Invalid fixed coin dimensions.')

    def expand(q):
        return np.r_[q[:3], radius_m * 1000, thickness_m * 1000, q[3], 0.0]

    def residual(q, obs):
        return disk_residual(expand(q), obs)
    low = np.r_[center - [65, 65, 80], -2 * np.pi]
    high = np.r_[center + [65, 65, 80], 2 * np.pi]

    def solve(seed, obs):
        return least_squares(residual, seed, args=(obs,), bounds=(low, high), x_scale=[20, 20, 20, 1], loss='soft_l1', f_scale=1.0, max_nfev=100, diff_step=1e-05)
    fits = [solve(np.r_[center, yaw], observations) for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    fits.sort(key=lambda f: np.mean(residual(f.x, observations) ** 2))
    q = fits[0].x
    p = expand(q)
    errors = residual(q, observations).reshape(len(observations), -1)[:, :-1]
    rms = np.sqrt(np.mean(errors ** 2, axis=1))
    ious = {ob['name']: silhouette_iou(p, ob) for ob in observations}
    samples = [q]
    for j in range(len(observations)):
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            changed = [dict(ob) for ob in observations]
            changed[j]['support'] = changed[j]['support'] + DIRECTIONS @ shift
            samples.append(solve(q, changed).x)
    for fit in fits[1:]:
        if np.sqrt(np.mean(residual(fit.x, observations) ** 2)) <= np.sqrt(np.mean(errors ** 2)) + 0.5:
            samples.append(fit.x)
    normals = np.array([basis(s[3], 0.0)[:, 2] for s in samples])
    normal = basis(q[3], 0.0)[:, 2]
    center_spread = float(np.max(np.linalg.norm(np.asarray(samples)[:, :3] - q[:3], axis=1)) * 0.001)
    normal_spread = float(np.max(np.degrees(np.arccos(np.clip(np.abs(normals @ normal), -1, 1)))))
    rays = [p[:3] * 0.001 - ob['view'].pos for ob in observations]
    rays = [r / np.linalg.norm(r) for r in rays]
    parallax = max((np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in itertools.combinations(rays, 2)))
    reasons = []
    if parallax < 10:
        reasons.append('Need a separated view for the grasp pose.')
    if sum((np.count_nonzero(ob['complete']) for ob in observations)) < 32:
        reasons.append('Insufficient visible rim to locate the grasp.')
    if max(rms) > 2.0 or min(ious.values()) < 0.65:
        reasons.append('Selected views disagree with the known upright coin.')
    if center_spread > 0.003:
        reasons.append('Grasp position sensitivity exceeds 3 mm.')
    if normal_spread > 10:
        reasons.append('Face direction sensitivity exceeds 10 degrees.')
    return dict(accepted=not reasons, grasp_accepted=not reasons, reasons=reasons, acceptance_scope='holder_pickup_only', full_geometry_measured=False, kind='known_upright_coin_pose', parameters=p.tolist(), center_xyz=(p[:3] * 0.001).tolist(), normal_xyz=normal.tolist(), radius_m=radius_m, thickness_m=thickness_m, upright_assumed=True, view_errors_px={ob['name']: float(e) for ob, e in zip(observations, rms)}, view_iou=ious, view_separation_deg=float(parallax), center_sensitivity_m=center_spread, normal_sensitivity_deg=normal_spread, parameter_samples=[expand(s).tolist() for s in samples], normal_sign_ambiguous=True, note='XYZ and face yaw from RGB; fixed dimensions and upright holder assumption. Suitable only for pickup planning; verify retention and slot alignment separately.')
