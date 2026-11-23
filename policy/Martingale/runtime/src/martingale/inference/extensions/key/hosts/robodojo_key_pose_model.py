# Copyright © 2026 Martingale. All Rights Reserved.
import hashlib
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

GEOMETRY_KEYS = ('tip_extension_m', 'shaft_width_m', 'tooth_depth_m', 'tooth_width_m', 'tooth_gap_m', 'shaft_thickness_m')
NO_GEOMETRY = 'No key geometry has been given in this episode. Pass key_geometry once: a range [low, high] in metres for each of ' + ', '.join(GEOMETRY_KEYS) + '.'

def _interval(name, value, spread):
    """(low, high) in metres of a length given as [low, high], or as one number +/- spread."""
    array = np.asarray(value, float).ravel()
    if array.size == 1:
        array = np.array([array[0] - spread, array[0] + spread])
    if array.size != 2 or not np.isfinite(array).all() or not 0 < array[0] <= array[1] < 0.1:
        raise ValueError(name + ' must be a range [low, high] in metres (or one measured length), positive and below 0.1 m.')
    return (float(array[0]), float(array[1]))

def episode_geometry(primitives, key_geometry=None):
    """Key dimensions as ranges supplied by the caller in the CURRENT episode.

    Nothing is stored with the package or carried between episodes. key_geometry
    gives, for each length of GEOMETRY_KEYS, a range [low, high] in metres: a
    general range, or one narrowed by measuring this episode's images. A single
    number is read as that value +/- uncertainty_m (default 0.5 mm, allowed
    0.1..3 mm). The pose is solved at the middle of every range; both ends of
    each range, and all low / all high ends together, enter the reported
    uncertainty, so wide ranges give a wide pose uncertainty. The ranges are
    kept on this episode's primitives until others replace them. Returns None
    when none have been supplied yet.
    """
    if key_geometry is None:
        return getattr(primitives, '_key_geometry', None)
    if not isinstance(key_geometry, dict):
        raise ValueError('key_geometry must be a dictionary of ranges [low, high] in metres.')
    missing = [name for name in GEOMETRY_KEYS if name not in key_geometry]
    if missing:
        raise ValueError('key_geometry lacks: ' + ', '.join(missing))
    spread = float(key_geometry.get('uncertainty_m', 0.0005))
    if not np.isfinite(spread) or not 0.0001 <= spread <= 0.003:
        raise ValueError('uncertainty_m must be between 0.1 and 3 mm.')
    ranges = {name: _interval(name, key_geometry[name], spread) for name in GEOMETRY_KEYS}
    values = {name: (low + high) / 2 for name, (low, high) in ranges.items()}
    samples = [dict(values, **{name: end}) for name, ends in ranges.items() for end in ends]
    samples += [{name: ends[side] for name, ends in ranges.items()} for side in (0, 1)]
    ident = hashlib.sha256(json.dumps([[round(end, 6) for end in ranges[name]] for name in GEOMETRY_KEYS]).encode()).hexdigest()[:16]
    model = dict(values, ranges_m={name: list(ends) for name, ends in ranges.items()}, geometry_samples=samples, geometry_id=ident, source='ranges supplied by the caller in this episode')
    primitives._key_geometry = model
    return model

def face_points(model):
    """x from tip toward bow, y toward teeth, z=x cross y; metres."""
    e, w, d, t, g = (model['tip_extension_m'], model['shaft_width_m'], model['tooth_depth_m'], model['tooth_width_m'], model['tooth_gap_m'])
    return np.array([[x, y, 0.0] for a in [e, e + t + g] for x, y in [(a, w / 2), (a, w / 2 + d), (a + t, w / 2 + d), (a + t, w / 2)]])

def project(view, points):
    row, col, depth = view.project(points)
    return (np.c_[col, row], depth)

def unpack(value):
    if isinstance(value, dict):
        value = [*value['distal'], *value['proximal'], *value.get('back_tip_outer', [])]
    array = np.asarray(value, float)
    if array.shape not in ((8, 2), (10, 2)) or not np.isfinite(array).all():
        raise ValueError('Mark the four corners of each tooth broad face, distal tooth first: tip-root, tip-outer, bow-outer, bow-root.')
    return array

def pose_matrix(p):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    result[:3, 3] = p[:3] * 0.001
    return result

def parameters(matrix):
    return np.r_[matrix[:3, 3] * 1000, Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]

def visible_points(matrix, view, model, count=8):
    rotation = matrix[:3, :3]
    tip = matrix[:3, 3]
    normal = rotation[:, 2]
    side = 1.0 if (view.pos - tip) @ normal >= 0 else -1.0
    points = face_points(model)
    points[:, 2] = side * model['shaft_thickness_m'] / 2
    if count == 10:
        back = points[[1, 5]].copy()
        back[:, 2] *= -1
        points = np.vstack([points, back])
    return (points @ rotation.T + tip, side)

def shaft_edges(matrix, view, model):
    """Two silhouette edges of the rectangular shaft, in original pixels."""
    points = np.array([[[x, y, z] for x in (0.005, 0.03)] for y in (-model['shaft_width_m'] / 2, model['shaft_width_m'] / 2) for z in (-model['shaft_thickness_m'] / 2, model['shaft_thickness_m'] / 2)])
    pixels = project(view, points.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])[0].reshape(4, 2, 2)
    axis = np.mean(pixels[:, 1] - pixels[:, 0], axis=0)
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    distances = pixels.mean(1) @ normal
    return (pixels[[np.argmin(distances), np.argmax(distances)]], normal)

def residual(p, observations, views, model, shaft_lines=None):
    matrix = pose_matrix(p)
    values = []
    for name, pixels in observations.items():
        xyz, _ = visible_points(matrix, views[name], model, len(pixels))
        predicted, _ = project(views[name], xyz)
        values.extend((predicted - pixels).ravel())
    for name, lines in (shaft_lines or {}).items():
        expected, normal = shaft_edges(matrix, views[name], model)
        order = np.argsort(lines.mean(1) @ normal)
        for predicted, line in zip(expected, lines[order]):
            delta = line[1] - line[0]
            n = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            values.extend((predicted - line[0]) @ n)
    return np.asarray(values)

def solve(observations, views, model, initial=None, shaft_lines=None):
    seeds = []
    if initial is not None:
        seeds = [parameters(initial)]
    else:
        obj = face_points(model).astype(np.float64)
        for name, pixels in observations.items():
            view = views[name]
            K = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1.0]])
            answer = cv2.solvePnPGeneric(obj, pixels[:8].astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE)
            if not answer[0]:
                continue
            for r, t in zip(answer[1], answer[2]):
                matrix = np.eye(4)
                frame = view.rot @ np.diag([1.0, -1.0, -1.0])
                matrix[:3, :3] = frame @ cv2.Rodrigues(r)[0]
                surface = view.pos + frame @ t.ravel()
                side = 1.0 if (view.pos - surface) @ matrix[:3, 2] >= 0 else -1.0
                matrix[:3, 3] = surface - side * model['shaft_thickness_m'] / 2 * matrix[:3, 2]
                seeds.append(parameters(matrix))
    fits = []
    for seed in seeds:
        result = least_squares(residual, seed, args=(observations, views, model, shaft_lines), x_scale=[10, 10, 10, 1, 1, 1], max_nfev=100, diff_step=1e-05, ftol=1e-08, xtol=1e-08, gtol=1e-08)
        matrix = pose_matrix(result.x)
        if all((np.min(project(views[n], visible_points(matrix, views[n], model)[0])[1]) > 0 for n in observations)):
            fits.append(result)
    if not fits:
        raise ValueError('No key pose with the observed faces in front of their cameras.')
    fits.sort(key=lambda x: x.cost)
    return (fits[0], fits)

def estimate(views, tooth_pixels, model=None, shaft_lines=None):
    """Fresh 6D pose from the tooth-face geometry ranges given in this episode; no tip annotation."""
    if model is None:
        return dict(accepted=False, reasons=[NO_GEOMETRY], method='calibrated_two_tooth_faces')
    try:
        if not tooth_pixels:
            raise ValueError('Mark two broad tooth faces in one clear wrist view.')
        observations = {n: unpack(v) for n, v in tooth_pixels.items()}
        shaft_lines = {n: np.asarray(v, float) for n, v in (shaft_lines or {}).items()}
        for n, lines in shaft_lines.items():
            if n not in views or lines.shape != (2, 2, 2) or (not np.isfinite(lines).all()):
                raise ValueError('Two finite longitudinal shaft-edge lines are required per supporting camera.')
            h, w = views[n].shape[:2]
            if np.any(lines < 0) or np.any(lines > [w - 1, h - 1]):
                raise ValueError('Shaft boundaries must lie inside the original image.')
            if np.min(np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)) < 25:
                raise ValueError('Supporting shaft edges must span at least 25 pixels.')
        for name, pixels in observations.items():
            if name not in views:
                raise ValueError('Unavailable tooth-face camera: ' + name)
            h, w = views[name].shape[:2]
            if np.any(pixels < 0) or np.any(pixels > [w - 1, h - 1]):
                raise ValueError('Tooth corners must be inside the original image.')
            if np.linalg.norm(pixels[0] - pixels[4]) < 15 or min(np.linalg.norm(pixels[1] - pixels[0]), np.linalg.norm(pixels[5] - pixels[4])) < 12:
                raise ValueError('Tooth faces are too small or foreshortened; use a closer view of both broad faces.')
        best, alternatives = solve(observations, views, model, shaft_lines=shaft_lines)
        matrix = pose_matrix(best.x)
        errors = {n: np.linalg.norm(project(views[n], visible_points(matrix, views[n], model, len(pixels))[0])[0] - pixels, axis=1).tolist() for n, pixels in observations.items()}
        if max((max(v) for v in errors.values())) > 2.5:
            raise ValueError('Tooth-face corners do not agree with the middle of the key geometry ranges; check face/edge identities, or measure the key in current images and pass narrower ranges.')
        if shaft_lines and max(abs(best.fun[sum((p.size for p in observations.values())):])) > 2.5:
            raise ValueError('Supporting shaft boundaries disagree with the measured tooth faces.')
        inputs = [('teeth', n, p) for n, p in observations.items()] + [('shaft', n, p) for n, p in shaft_lines.items()]
        derivatives = []
        for kind, name, pixels in inputs:
            for i in range(pixels.size):
                changed = pixels.copy()
                changed.flat[i] += 0.001
                obs = dict(observations)
                lines = dict(shaft_lines)
                (obs if kind == 'teeth' else lines)[name] = changed
                derivatives.append((residual(best.x, obs, views, model, lines) - best.fun) / 0.001)
        inverse = -np.linalg.pinv(best.jac) @ np.asarray(derivatives).T
        cases = []
        for row in inverse:
            for sign in (-1, 1):
                offset = sign * np.sign(row)
                i = 0
                sample = {}
                sample_lines = {}
                for kind, n, pixels in inputs:
                    (sample if kind == 'teeth' else sample_lines)[n] = pixels + offset[i:i + pixels.size].reshape(pixels.shape)
                    i += pixels.size
                cases.append((sample, sample_lines, model))
        for name in observations:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cases.append((dict(observations, **{name: observations[name] + [dx, dy]}), shaft_lines, model))
        for variant in model.get('geometry_samples', []):
            cases.append((observations, shaft_lines, dict(model, **variant)))
        samples = [matrix]
        for observed, lines, geometry in cases:
            result, _ = solve(observed, views, geometry, initial=matrix, shaft_lines=lines)
            samples.append(pose_matrix(result.x))
        rms = np.sqrt(2 * best.cost / best.fun.size)
        for fit in alternatives[1:]:
            if np.sqrt(2 * fit.cost / fit.fun.size) <= rms + 1.0:
                samples.append(pose_matrix(fit.x))
        samples = np.asarray(samples)
        tip = matrix[:3, 3]
        axis = matrix[:3, 0]
        width = matrix[:3, 1]
        tips = samples[:, :3, 3]
        axes = samples[:, :3, 0]
        widths = samples[:, :3, 1]
        return dict(accepted=True, reasons=[], tip_xyz=tip.tolist(), shaft_axis_xyz=axis.tolist(), width_axis_xyz=width.tolist(), key_to_world=matrix.tolist(), pose_samples=samples.tolist(), tip_samples_xyz=tips.tolist(), shaft_samples_xyz=axes.tolist(), width_samples_xyz=widths.tolist(), pixel_sensitivity_m=float(np.linalg.norm(tips - tip, axis=1).max()), shaft_spread_deg=float(np.degrees(np.arccos(np.clip(axes @ axis, -1, 1))).max()), width_spread_deg=float(np.degrees(np.arccos(np.clip(widths @ width, -1, 1))).max()), reprojection_errors_px=errors, rms_px=float(rms), geometry_id=model['geometry_id'], method='calibrated_two_tooth_faces', pixel_error_px=1.0, supporting_shaft_cameras=list(shaft_lines), uncertainty_note='Nonlinear sensitivity samples include independent pixel extrema, coherent shifts, both ends of every supplied key-dimension range and plausible planar branches; not a statistical confidence bound.')
    except (ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        return dict(accepted=False, reasons=[str(exc)], method='calibrated_two_tooth_faces')

def hand_matrix(state, arm):
    from martingale.hosts import robodojo_tools as tools
    pose = np.asarray(state[arm + '_ee_pose'])
    matrix = np.eye(4)
    matrix[:3, :3] = tools.kin.rot_from_quat(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix

def refine_corners(image, predicted):
    """Image evidence is mandatory: flat patches never count as observations."""
    gray = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2GRAY).astype(np.uint8)
    response = cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)
    points = []
    h, w = gray.shape
    for point in predicted:
        x, y = np.rint(point).astype(int)
        if x < 7 or x >= w - 7 or y < 7 or (y >= h - 7):
            raise ValueError('Predicted tooth corner is clipped.')
        patch = response[y - 5:y + 6, x - 5:x + 6]
        j, i = np.unravel_index(patch.argmax(), patch.shape)
        if patch[j, i] < 0.0007:
            raise ValueError('A predicted tooth corner has no current image evidence.')
        seed = np.array([[[x + i - 5.0, y + j - 5.0]]], np.float32)
        refined = cv2.cornerSubPix(gray, seed, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))[0, 0]
        if np.linalg.norm(refined - point) > 6.0:
            raise ValueError('Key may have slipped; mark its current tooth faces again.')
        points.append(refined)
    if min((np.linalg.norm(a - b) for i, a in enumerate(points) for b in points[i + 1:])) < 3:
        raise ValueError('Two predicted features converged to the same image corner.')
    return np.asarray(points)

def track_corners(previous, image, old_pixels, predicted):
    """Track the same annotated patches; do not jump to a stronger nearby corner."""
    gray = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2GRAY)
    before = np.asarray(old_pixels, np.float32).reshape(-1, 1, 2)
    guess = np.asarray(predicted, np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=0.0001)
    after, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, before, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
    if after is None or not np.all(status):
        raise ValueError('Fresh tooth features are unresolved.')
    back, status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, after, None, **options)
    if back is None or not np.all(status) or np.max(np.linalg.norm(back - before, axis=2)) > 1.0:
        raise ValueError('Tooth feature tracking is inconsistent.')
    points = after[:, 0]
    h, w = gray.shape
    if np.any(points < 7) or np.any(points > [w - 8, h - 8]) or np.max(np.linalg.norm(points - predicted, axis=1)) > 8.0:
        raise ValueError('Key may have slipped or left the field of view.')
    return points

def refine_shaft(image, edges):
    gray = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    result = []
    h, w = gray.shape
    for edge in edges:
        direction = edge[1] - edge[0]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        points = []
        for p in edge:
            offsets = np.linspace(-5, 5, 41)
            xy = p + offsets[:, None] * normal
            if np.any(xy < 2) or np.any(xy > [w - 3, h - 3]):
                raise ValueError('Predicted shaft edge is clipped.')
            values = cv2.remap(gray.astype(np.float32), xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            gradient = np.abs(np.gradient(values))
            index = int(gradient.argmax())
            if gradient[index] < 4.0:
                raise ValueError('No fresh shaft-edge contrast.')
            points.append(xy[index])
        result.append(points)
    return np.asarray(result)

def validate_faces(image, pixels):
    """Reject annotations on the background, including stale image pixels."""
    rgb = np.asarray(image)[..., :3].astype(float)
    for face in (pixels[:4], pixels[4:8]):
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(face).astype(np.int32), 1)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
        samples = rgb[mask]
        if len(samples) < 20 or np.mean((samples.min(1) > 140) & (np.ptp(samples, axis=1) < 55)) < 0.65:
            raise ValueError('Tooth annotations must cover the current white broad faces, not background or a thickness face.')

def measure(primitives, tooth_pixels, arm, near_edges=None, far_edges=None, key_geometry=None):
    try:
        from martingale.inference.extensions.key.hosts import robodojo_keyhole as geometry
    except ImportError:
        import keyhole as geometry
    from martingale.hosts import robodojo_tools as tools
    if arm not in ('left', 'right'):
        return dict(accepted=False, reasons=['Choose the arm holding the key.'])
    try:
        model = episode_geometry(primitives, key_geometry)
    except (ValueError, TypeError) as exc:
        return dict(accepted=False, reasons=[str(exc)])
    if model is None:
        return dict(accepted=False, reasons=[NO_GEOMETRY])
    obs, images, views = geometry.current(primitives)
    state = tools._shown(obs)
    cache = getattr(primitives, '_key_model_grasp', None)
    automatic = not bool(tooth_pixels)
    try:
        if set(near_edges or {}) != set(far_edges or {}):
            raise ValueError('Provide both near and far cross-sections for each supporting shaft camera.')
        shaft_lines = {n: np.stack([np.asarray(near_edges[n], float), np.asarray(far_edges[n], float)], axis=1) for n in near_edges or {}}
    except (ValueError, TypeError) as exc:
        return dict(accepted=False, reasons=[str(exc)])
    if automatic:
        if not cache or cache['arm'] != arm or cache['geometry_id'] != model['geometry_id']:
            return dict(accepted=False, reasons=['Establish the receiving-hand key pose from both broad tooth faces after handoff.'])
        predicted = hand_matrix(state, arm) @ np.asarray(cache['key_to_hand'])
        tooth_pixels = {}
        for name in cache['cameras']:
            if name not in images:
                continue
            xyz, _ = visible_points(predicted, views[name], model, cache.get('feature_counts', {}).get(name, 8))
            pixels, depth = project(views[name], xyz)
            if np.min(depth) <= 0:
                continue
            try:
                tooth_pixels[name] = track_corners(cache['images'][name], images[name], cache['pixels'][name], pixels).tolist()
            except ValueError:
                continue
        if not tooth_pixels:
            return dict(accepted=False, reasons=['Cached grasp predicts the key, but fresh tooth features are unresolved. Reannotate the broad faces in a current clear image.'], predicted_tip_xyz=predicted[:3, 3].tolist(), prediction_only=True)
        for name in cache.get('shaft_cameras', []):
            try:
                shaft_lines[name] = refine_shaft(images[name], shaft_edges(predicted, views[name], model)[0])
            except (ValueError, KeyError):
                return dict(accepted=False, reasons=['Fresh supporting shaft edges are unresolved; mark two cross-sections on the visible straight shaft again.'])
    try:
        for name, pixels in tooth_pixels.items():
            validate_faces(images[name], unpack(pixels))
    except (ValueError, KeyError) as exc:
        return dict(accepted=False, reasons=[str(exc)])
    result = estimate(views, tooth_pixels, model, shaft_lines=shaft_lines)
    if result['accepted']:
        matrix = np.asarray(result['key_to_world'])
        if automatic:
            displacement = float(np.linalg.norm(matrix[:3, 3] - predicted[:3, 3]))
            rotation = float(np.degrees(Rotation.from_matrix(predicted[:3, :3].T @ matrix[:3, :3]).magnitude()))
            result['grasp_prediction_residual'] = dict(tip_m=displacement, rotation_deg=rotation)
            if displacement > 0.003 or rotation > 5:
                return dict(result, accepted=False, reasons=['Fresh key pose disagrees with the cached grasp; explicitly remeasure after possible slip.'])
        primitives._key_model_grasp = dict(arm=arm, geometry_id=model['geometry_id'], key_to_hand=(np.linalg.inv(hand_matrix(state, arm)) @ matrix).tolist(), cameras=list(tooth_pixels), shaft_cameras=list(shaft_lines), feature_counts={n: len(unpack(v)) for n, v in tooth_pixels.items()}, images={n: cv2.cvtColor(images[n][..., :3], cv2.COLOR_RGB2GRAY) for n in tooth_pixels}, pixels={n: unpack(v).tolist() for n, v in tooth_pixels.items()})
    return dict(result, automatic_reobservation=automatic, observed_tooth_pixels=tooth_pixels, observed_shaft_lines={n: lines.tolist() for n, lines in shaft_lines.items()})
