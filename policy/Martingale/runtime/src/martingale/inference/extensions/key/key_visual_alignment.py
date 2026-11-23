# Copyright © 2026 Martingale. All Rights Reserved.
import itertools
import math
import numpy as np
from martingale.inference.extensions.key.key_alignment_report import build_report
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

def _vector(value, size, name):
    result = np.asarray(value, float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(name + ' must contain ' + str(size) + ' finite numbers')
    return result.copy()

def _rotation(quat):
    q = _vector(quat, 4, 'eef_quat')
    if np.linalg.norm(q) < 1e-08:
        raise ValueError('Invalid hand quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

def _ray(meta, pixel):
    k = np.asarray(meta['intrinsic_matrix'], float)
    pose = np.asarray(meta['extrinsic_matrix'], float)
    if k.shape != (3, 3) or pose.shape != (4, 4) or (not np.isfinite(k).all()) or (not np.isfinite(pose).all()):
        raise ValueError('Invalid camera calibration')
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid focal length')
    local = np.array([(pixel[0] - k[0, 2]) / k[0, 0], -(pixel[1] - k[1, 2]) / k[1, 1], -1.0])
    ray = pose[:3, :3] @ local
    return (pose[:3, 3], ray / np.linalg.norm(ray))

def triangulate(metas, pixels):
    rays = [_ray(meta, pixel) for meta, pixel in zip(metas, pixels)]
    matrices = [np.eye(3) - np.outer(d, d) for _, d in rays]
    system = sum(matrices)
    if np.linalg.cond(system) > 1000000.0:
        raise ValueError('Views have insufficient parallax')
    point = np.linalg.solve(system, sum((m @ o for m, (o, _) in zip(matrices, rays))))
    if any(((point - o) @ d <= 0 for o, d in rays)):
        raise ValueError('Annotated point is behind a camera')
    residual = max((float(np.linalg.norm(m @ (point - o))) for m, (o, _) in zip(matrices, rays)))
    parallax = min((min(_angle(a[1], b[1]), 180 - _angle(a[1], b[1])) for a, b in itertools.combinations(rays, 2)))
    return (point, residual, parallax)

def measure(robot, opening, tip_pixels, shaft_pixels, *, arm, observed_step, shaft_landmark):
    """Measure alignment from the given pixels and calibrated views; no motion."""
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    state = robot.state()
    if type(observed_step) is not int or observed_step != state.step:
        return dict(accepted=False, reasons=['stale_observation'])
    if not isinstance(shaft_landmark, str) or not shaft_landmark.strip():
        raise ValueError('Name the same physical shaft station in both images')
    names = sorted(tip_pixels)
    observer = 'cam_' + ('right' if arm == 'left' else 'left') + '_wrist'
    if len(names) not in (1, 2) or set(names) != set(shaft_pixels) or (not set(names) <= set(CAMERAS)) or (len(names) == 2 and observer not in names):
        raise ValueError('Use two synchronized cameras including the empty observer wrist, or one visible camera for an unresolved axis overlay only')
    entry = _vector(opening['entry_xyz'], 3, 'entry_xyz')
    normal = _vector(opening['normal_xyz'], 3, 'normal_xyz')
    if np.linalg.norm(normal) < 1e-08:
        raise ValueError('normal_xyz must be nonzero')
    normal /= np.linalg.norm(normal)
    metas = [robot.camera_meta(name, step=observed_step) for name in names]
    tips = np.array([_vector(tip_pixels[name], 2, 'tip pixel') for name in names])
    shafts = np.array([_vector(shaft_pixels[name], 2, 'shaft pixel') for name in names])
    for meta, tip, shaft in zip(metas, tips, shafts):
        h, w = meta['shape'][:2]
        if np.any(np.r_[tip, shaft] < 0) or np.any(np.array([tip, shaft]) >= [w, h]):
            raise ValueError('Landmark pixel outside image')

    def finish(result):
        result.setdefault('observed_step', observed_step)
        return build_report(robot, result, opening, names, metas, tips, shafts, arm, shaft_landmark)
    if len(names) == 1:
        return finish(dict(accepted=False, reasons=['Only one annotated view: axis overlay available, but 3D tip offset and tilt are unresolved. Do not invent a hidden second-camera feature.']))
    spans = np.linalg.norm(tips - shafts, axis=1)
    if np.min(spans) < 12:
        return finish(dict(accepted=False, reasons=['Shaft station spans fewer than 12 pixels in one view; inspect existing images or use the single-view overlay. Do not reposition cameras.'], shaft_span_pixels=dict(zip(names, spans.tolist()))))
    try:
        tip, tip_residual, tip_parallax = triangulate(metas, tips)
        shaft, shaft_residual, shaft_parallax = triangulate(metas, shafts)
    except ValueError as exc:
        return finish(dict(accepted=False, reasons=[str(exc)]))
    length = float(np.linalg.norm(shaft - tip))
    reasons = []
    if min(tip_parallax, shaft_parallax) < 8:
        reasons.append('Insufficient stereo angle (<8 degrees); use existing images or the single-view overlay without repositioning cameras.')
    if max(tip_residual, shaft_residual) > 0.002:
        reasons.append('Corresponding rays disagree by more than 2 mm; reidentify physical landmarks.')
    if not 0.01 <= length <= 0.075:
        reasons.append('Tip/shaft separation is outside 10–75 mm; check the physical correspondence.')
    if length < 1e-08:
        return finish(dict(accepted=False, reasons=reasons or ['Degenerate shaft station']))
    axis = (shaft - tip) / length
    if axis @ normal <= 0:
        reasons.append('Shaft station must be above the distal terminal tip.')
    tip_sensitivity = tilt_sensitivity = 0.0
    tilt = _angle(axis, normal)
    for which in ('tip', 'shaft'):
        for camera in range(2):
            for coordinate in range(2):
                displacements, angles = ([], [])
                for sign in (-1, 1):
                    perturbed = (tips if which == 'tip' else shafts).copy()
                    perturbed[camera, coordinate] += sign
                    try:
                        changed, _, _ = triangulate(metas, perturbed)
                    except ValueError:
                        displacements.append(math.inf)
                        angles.append(math.inf)
                        continue
                    t, s = (changed, shaft) if which == 'tip' else (tip, changed)
                    displacements.append(float(np.linalg.norm(t - tip)))
                    angles.append(abs(_angle(s - t, normal) - tilt))
                tip_sensitivity += max(displacements)
                tilt_sensitivity += max(angles)
    offset = tip - entry
    height = float(offset @ normal)
    lateral = offset - height * normal
    result = dict(accepted=not reasons, reasons=reasons, observed_step=observed_step, tip_xyz=tip.tolist(), shaft_xyz=shaft.tolist(), shaft_axis_xyz=axis.tolist(), tip_above_rim_m=height, lateral_error_m=float(np.linalg.norm(lateral)), lateral_correction_xyz=(-lateral).tolist(), tilt_deg=tilt, shaft_span_pixels=dict(zip(names, spans.tolist())), ray_residual_m=max(tip_residual, shaft_residual), parallax_deg=min(tip_parallax, shaft_parallax), tip_pixel_sensitivity_m=tip_sensitivity, tilt_pixel_sensitivity_deg=tilt_sensitivity, tip_pixels=tip_pixels, shaft_pixels=shaft_pixels, shaft_landmark=shaft_landmark, entry_complete=False, note='Accepted means geometrically consistent annotated rays, not correct landmark identity, clearance, insertion, or a command to move. Identify the actual terminal tip in both views; a tooth edge or reflected highlight is not the tip. For large sensitivity inspect existing images/crops; do not reposition cameras. Clip corrections to small checked moves while clear and inspect again.')
    hand = state.arm(arm)
    r = _rotation(hand.eef_quat)
    local_tip = r.T @ (tip - np.asarray(hand.tcp_pos))
    observer_pose = np.asarray(metas[names.index(observer)]['extrinsic_matrix'])
    previous = getattr(robot, '_stable_visual_key_measurement', None)
    result['tracking_comparable'] = False
    if previous and previous['step'] < observed_step and (previous['arm'] == arm) and (previous['landmark'] == shaft_landmark) and (previous['cameras'] == names) and (np.linalg.norm(previous['entry'] - entry) < 1e-06) and (np.linalg.norm(previous['normal'] - normal) < 1e-06):
        old = previous['observer_pose']
        if np.linalg.norm(old[:3, 3] - observer_pose[:3, 3]) < 0.0005 and np.linalg.norm(old[:3, :3] - observer_pose[:3, :3]) < 0.005:
            result.update(tracking_comparable=True, tip_feed_since_previous_m=float((previous['tip'] - tip) @ normal), tip_in_hand_change_m=float(np.linalg.norm(local_tip - previous['local_tip'])), tracking_note='Changes also include annotation uncertainty; they are not automatic contact/slip verdicts.')
    if result['accepted']:
        robot._stable_visual_key_measurement = dict(step=observed_step, arm=arm, landmark=shaft_landmark, tip=tip, local_tip=local_tip, observer_pose=observer_pose.copy(), cameras=names, entry=entry.copy(), normal=normal.copy())
    return finish(result)
