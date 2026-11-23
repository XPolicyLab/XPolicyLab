# Copyright © 2026 Martingale. All Rights Reserved.
import json
import time
import numpy as np
from martingale.hosts.robodojo_sight import View
from martingale.inference.extensions.toast.hosts.robodojo_slot_fit import fit

def slots_from_rectangles(rectangles):
    if 'outer' not in rectangles:
        return {k: np.asarray(v, float) for k, v in rectangles.items()}
    outer, divider = (np.asarray(rectangles['outer']), np.asarray(rectangles['divider']))
    u = (outer[2] + outer[3] - outer[0] - outer[1]) / 2
    u[2] = 0
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0], 0.0])
    if v @ (outer[1] - outer[0]) < 0:
        v = -v
    lo, hi = (min(outer @ u), max(outer @ u))
    cuts = [min(outer @ v), min(divider @ v), max(divider @ v), max(outer @ v)]
    z = float(outer[:, 2].mean())
    return {name: np.array([a * u + b * v + [0, 0, z] for a, b in [(lo, left), (lo, right), (hi, right), (hi, left)]]) for name, left, right in [('slot_left', cuts[0], cuts[1]), ('slot_right', cuts[2], cuts[3])]}

def measure(robot, features, step=-1):
    features = json.loads(json.dumps(features))
    rectangles = features.get('rectangles', {})
    if set(rectangles) == {'outer'}:
        if not features.get('divider_edges'):
            raise ValueError('A shared opening needs visible divider_edges to identify both slots')
    elif set(rectangles) != {'slot_left', 'slot_right'}:
        raise ValueError('Use outer plus divider_edges, or slot_left and slot_right rectangles')
    features['coplanar'] = True
    cameras = set()
    for name, observations in rectangles.items():
        if 'cam_head' not in observations or len(observations) < 2:
            raise ValueError('Each rim needs head and at least one wrist observation')
        for camera, pixels in observations.items():
            p = np.asarray(pixels, float)
            if camera not in robot.CAMERAS or p.shape != (4, 2) or (not np.isfinite(p).all()) or np.any(p < 0) or np.any(p[:, 0] >= 640) or np.any(p[:, 1] >= 480):
                raise ValueError('Supply four visible corners in original-image (column,row) coordinates')
            cameras.add(camera)
    for camera, edges in features.get('divider_edges', {}).items():
        if camera not in cameras:
            raise ValueError('Divider camera must also observe the rim')
        if set(edges) != {'left', 'right'}:
            raise ValueError('Divider needs left and right long edges')
        for pixels in edges.values():
            p = np.asarray(pixels, float)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or (not np.isfinite(p).all()):
                raise ValueError('Each divider edge needs at least two visible (column,row) points')
    if 'outer' in rectangles and set(features['divider_edges']) != cameras:
        raise ValueError('Observe both divider long edges in every fitted camera')
    state = robot.state(step)
    actual_step = state.step
    metadata = {c: robot.camera_meta(c, step=actual_step) for c in sorted(cameras)}
    views = {c: View(m['intrinsic_matrix'], m['extrinsic_matrix']) for c, m in metadata.items()}
    started = time.perf_counter()
    fitted = fit(features, views)
    polygons = slots_from_rectangles(fitted['corners'])
    reasons = []
    if not fitted['success'] or fitted['jacobian_rank'] < fitted['parameter_count']:
        reasons.append('fit did not converge with fully constrained parameters')
    if fitted['reprojection_rms_px'] > 2 or max(fitted['reprojection_per_view'].values()) > 3:
        reasons.append('image disagreement; inspect corresponding features or obtain a clearer view')
    center = np.mean(np.concatenate(list(polygons.values())), axis=0)
    directions = {c: (v.pos - center) / np.linalg.norm(v.pos - center) for c, v in views.items()}
    parallax = max((np.rad2deg(np.arccos(np.clip(directions['cam_head'] @ d, -1, 1))) for c, d in directions.items() if c != 'cam_head'))
    if parallax < 10:
        reasons.append('insufficient viewpoint separation')
    slots = {}
    for name, p in sorted(polygons.items(), key=lambda item: item[1][:, 0].mean()):
        along = (p[2] + p[3] - p[0] - p[1]) / 2
        across = (p[1] + p[2] - p[0] - p[3]) / 2
        length = float(np.linalg.norm(along))
        width = float(np.linalg.norm(across))
        if not 0.003 < width < min(length, 0.09):
            reasons.append('slot dimensions implausible')
        angle = float(np.rad2deg(np.arctan2(along[1], along[0])) % 180)
        slots[name] = {'center_xyz': p.mean(axis=0).tolist(), 'rim_z': float(p[:, 2].mean()), 'corners_xyz': p.tolist(), 'length_m': length, 'width_m': width, 'long_axis_deg': angle, 'cross_axis_deg': (angle + 90) % 180}
    result = {'accepted': not reasons, 'reasons': reasons, 'step': actual_step, 'slots': slots, 'rim_geometry': fitted['corners'], 'reprojection_rms_px': fitted['reprojection_rms_px'], 'reprojection_per_view': fitted['reprojection_per_view'], 'view_separation_deg': float(parallax), 'elapsed_s': time.perf_counter() - started, 'note': 'Rim geometry only; slot bottom, held-bread pose and insertion success are not measured.'}
    return result
