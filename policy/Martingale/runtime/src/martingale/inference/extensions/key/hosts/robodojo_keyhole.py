# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree
try:
    from martingale.inference.extensions.key.hosts import robodojo_keyhole_uncertainty as uncertainty
    from martingale.inference.extensions.key.hosts import robodojo_keyhole_tooth as tooth_geometry
    from martingale.inference.extensions.key.hosts import robodojo_keyhole_shaft as shaft_geometry
    from martingale.inference.extensions.key.hosts import robodojo_keyhole_tip as tip_geometry
except ImportError:
    import keyhole_uncertainty as uncertainty
    import keyhole_tooth as tooth_geometry
    import keyhole_shaft as shaft_geometry
    import keyhole_tip as tip_geometry

def detect(image, seed):
    image = np.asarray(image)[..., :3]
    pixel = np.asarray(seed, float)
    h, w = image.shape[:2]
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or (not (0 <= pixel[0] < w and 0 <= pixel[1] < h)):
        raise ValueError('Entry seeds must be finite original-image (column,row) pixels.')
    cx, cy = np.rint(pixel).astype(int)
    x0, x1 = (max(0, cx - 48), min(w, cx + 49))
    y0, y1 = (max(0, cy - 48), min(h, cy + 49))
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    threshold = min(75.0, max(18.0, 0.45 * float(np.percentile(gray, 70))))
    mask = (gray < threshold).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    label = labels[cy - y0, cx - x0]
    if label == 0:
        rr, cc = np.nonzero(mask)
        if not len(rr):
            raise ValueError('No dark aperture at the selected entry point.')
        d = np.hypot(cc + x0 - pixel[0], rr + y0 - pixel[1])
        i = int(d.argmin())
        if d[i] > 3:
            raise ValueError('Entry seed is not inside a visible dark opening.')
        label = labels[rr[i], cc[i]]
    component = (labels == label).astype('uint8')
    if stats[label, cv2.CC_STAT_AREA] < 6:
        raise ValueError('Opening is too small in this view; obtain a closer view.')
    if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
        raise ValueError('Opening contour is clipped or joins an occluder.')
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0].astype(float) + [x0, y0]
    if len(contour) < 6:
        raise ValueError('Opening contour is unresolved.')
    center = contour.mean(0)
    direction = contour - center
    contour += 0.5 * direction / np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1.0)
    return dict(contour=contour, area=int(component.sum()), threshold=threshold)

def on_plane(view, pixels, z):
    p = np.asarray(pixels, float)
    rays = view.rays(p[:, 1], p[:, 0])
    return view.pos + rays * ((z - view.pos[2]) / rays[:, 2])[:, None]

def project(view, xyz):
    row, col, depth = view.project(xyz)
    return (np.c_[col, row], depth)

def height_sensitivity_interval(cost, fitted_z, bounds, increment=1.0):
    """Continuous local interval where contour cost is within +1 of its fit.

    The coarse search grid locates the minimum only; its spacing is neither
    an uncertainty floor nor an uncertainty estimate. Bracket both local
    level crossings and solve them on the continuous cost function. A flat
    or boundary-clipped interval does not establish a bounded sensitivity.
    """
    lower, upper = map(float, bounds)
    fitted_z, increment = (float(fitted_z), float(increment))
    if not np.isfinite([lower, fitted_z, upper, increment]).all() or not lower < fitted_z < upper or increment <= 0:
        raise ValueError('Height sensitivity requires an interior fit and finite positive cost increment.')
    fitted_cost = float(cost(fitted_z))
    if not np.isfinite(fitted_cost):
        raise ValueError('Opening height cost is nonfinite at its fitted height.')
    level = fitted_cost + increment

    def residual(height):
        value = float(cost(height)) - level
        if not np.isfinite(value):
            raise ValueError('Opening height sensitivity contains a nonfinite contour cost.')
        return value
    crossings = []
    for sign, limit in ((-1.0, lower), (1.0, upper)):
        inside = fitted_z
        step = min(1e-05, abs(limit - fitted_z))
        while True:
            outside = max(lower, min(upper, fitted_z + sign * step))
            value = residual(outside)
            if value > 0:
                a, b = sorted((inside, outside))
                crossings.append(float(brentq(residual, a, b, xtol=1e-10)))
                break
            if outside == limit:
                raise ValueError('Opening height sensitivity is flat or unbounded within the search range; obtain a clearer supporting view.')
            inside = outside
            step *= 2
    return crossings

def fit(images, views, entry_pixels, axis_pixels=None, table_z=0.765):
    """Fit the opening contour in a shared horizontal plane; no motion."""
    if not isinstance(entry_pixels, dict) or len(entry_pixels) < 2:
        return dict(accepted=False, reasons=['Select the same entry point in at least two current views.'])
    detections = {}
    try:
        for name, pixel in entry_pixels.items():
            if name not in images or name not in views:
                raise ValueError('Unknown or unavailable camera: ' + name)
            detections[name] = detect(images[name], pixel)
    except ValueError as exc:
        return dict(accepted=False, reasons=[str(exc)])
    reference = max(detections, key=lambda name: detections[name]['area'])
    contours = {name: d['contour'] for name, d in detections.items()}
    trees = {name: cKDTree(p) for name, p in contours.items()}

    def error(z, details=False):
        losses = {}
        for name in contours:
            if name == reference:
                continue
            forward, depth = project(views[name], on_plane(views[reference], contours[reference], z))
            backward, other_depth = project(views[reference], on_plane(views[name], contours[name], z))
            if min(depth.min(), other_depth.min()) <= 0:
                losses[name] = 1000000.0
                continue
            a = trees[name].query(forward)[0]
            b = trees[reference].query(backward)[0]
            scale = np.sqrt(detections[reference]['area'] / detections[name]['area'])
            losses[name] = float((np.mean(np.minimum(a * scale, 12) ** 2) + np.mean(np.minimum(b, 12) ** 2)) / 2)
        return losses if details else float(np.mean(list(losses.values())))
    heights = np.linspace(table_z + 0.003, table_z + 0.25, 180)
    costs = np.array([error(z) for z in heights])
    best = int(costs.argmin())
    if best in (0, len(heights) - 1):
        return dict(accepted=False, reasons=['No bounded shared opening-plane fit.'])
    result = minimize_scalar(error, bounds=(heights[best - 1], heights[best + 1]), method='bounded', options={'xatol': 1e-07})
    z = float(result.x)
    try:
        height_interval = height_sensitivity_interval(error, z, (heights[0], heights[-1]))
    except ValueError as exc:
        return dict(accepted=False, reasons=[str(exc)])
    height_span = float(max(z - height_interval[0], height_interval[1] - z))
    outline = on_plane(views[reference], contours[reference], z)
    entries = np.array([on_plane(views[c], np.array([entry_pixels[c]]), z)[0] for c in contours])
    weights = np.array([detections[c]['area'] for c in contours], float)
    entry = np.average(entries, axis=0, weights=weights)
    disagreement = float(np.max(np.linalg.norm(entries - entry, axis=1)))
    rays = [(views[c].pos - entry) / np.linalg.norm(views[c].pos - entry) for c in contours]
    angle = max((float(np.rad2deg(np.arccos(np.clip(abs(a @ b), 0, 1)))) for a in rays for b in rays))
    _, _, axes = np.linalg.svd(outline[:, :2] - outline[:, :2].mean(0), full_matrices=False)
    long_axis = np.r_[axes[0], 0.0]
    cross_axis = np.r_[-axes[0, 1], axes[0, 0], 0.0]
    reasons = []
    if angle < 10:
        reasons.append('Insufficient view separation; acquire a supporting wrist view.')
    if np.sqrt(result.fun) > 2.5:
        reasons.append('Opening contours disagree across views.')
    if disagreement > 0.003:
        reasons.append('Selected entry points disagree; check the same shaft-entry feature.')
    if height_span > 0.006:
        reasons.append('Opening height is poorly resolved; obtain a closer supporting view.')
    slot_axis = None
    axis_errors = []
    if axis_pixels:
        directions = []
        for name, pixels in axis_pixels.items():
            p = np.asarray(pixels, float)
            if name not in views or p.shape != (2, 2) or (not np.isfinite(p).all()):
                raise ValueError('Opening axis must give two finite (column,row) points per camera.')
            if np.linalg.norm(p[1] - p[0]) < 5:
                axis_errors.append('Shaft-entry edge is too short in pixels to resolve its direction.')
                continue
            endpoints = on_plane(views[name], p, z)
            direction = endpoints[1] - endpoints[0]
            direction /= np.linalg.norm(direction)
            if directions and direction @ directions[0] < 0:
                direction = -direction
            directions.append(direction)
        if directions:
            slot_axis = np.mean(directions, axis=0)
            slot_axis /= np.linalg.norm(slot_axis)
            if any((np.rad2deg(np.arccos(np.clip(d @ slot_axis, -1, 1))) > 10 for d in directions)):
                axis_errors.append('Opening-axis annotations disagree between views.')
        if axis_errors:
            slot_axis = None
    tooth_direction = None
    if slot_axis is not None:
        distances = (outline - entry) @ slot_axis
        asymmetry = float(distances.max() + distances.min())
        if abs(asymmetry) > 0.002:
            tooth_direction = slot_axis if asymmetry > 0 else -slot_axis
    samples = accuracy = None
    if tooth_direction is not None:
        try:
            samples = uncertainty.opening_samples(views, entry_pixels, axis_pixels, z, height_span, weights, tooth_direction)
            accuracy = uncertainty.opening_accuracy(samples, entry, tooth_direction)
        except ValueError as exc:
            axis_errors.append('Opening uncertainty is unresolved: ' + str(exc))
    return dict(accepted=not reasons, reasons=reasons, entry_xyz=entry.tolist() if not reasons else None, rim_z=z, normal_xyz=[0.0, 0.0, 1.0], long_axis_xyz=long_axis.tolist(), slot_axis_xyz=slot_axis.tolist() if slot_axis is not None else None, orientation_resolved=slot_axis is not None, orientation_reasons=axis_errors, tooth_direction_xyz=tooth_direction.tolist() if tooth_direction is not None else None, outline_xyz=outline.tolist(), length_m=float(np.ptp(outline @ long_axis)), width_m=float(np.ptp(outline @ cross_axis)), reference_camera=reference, supporting_views=list(contours), contour_rms_px=float(np.sqrt(result.fun)), entry_disagreement_m=disagreement, view_separation_deg=angle, height_sensitivity_m=height_span, height_sensitivity_interval_m=height_interval, uncertainty=samples, opening_accuracy=accuracy, detected_contours={c: p.tolist() for c, p in contours.items()}, note='Stationary horizontal opening only. Outline extents are not guaranteed clearances; long axis has 180-degree ambiguity. Sensitivity is not an accuracy guarantee.')

def current(primitives):
    from martingale.hosts import robodojo_tools as tools
    obs = tools._obs(primitives)
    state = tools._shown(obs)
    images = {c: np.asarray(v['color']) for c, v in obs['vision'].items() if c in ('cam_head', 'cam_left_wrist', 'cam_right_wrist') and v.get('color') is not None}
    return (obs, images, {c: tools._view(c, state) for c in images})

def measure(primitives, entry_pixels, axis_pixels=None, opening_geometry=None):
    """Without an opening shape measured in this episode: the shared-plane contour fit of
    two or more views. With one: its pose in a single useful view."""
    _, images, views = current(primitives)
    try:
        from martingale.inference.extensions.key.hosts import robodojo_keyhole_template as template
    except ImportError:
        import keyhole_template as template
    try:
        model = template.episode_geometry(primitives, opening_geometry)
    except (ValueError, TypeError) as exc:
        return dict(accepted=False, status='needs_observation', reasons=[str(exc)])
    if model is None:
        return fit(images, views, entry_pixels, axis_pixels)
    return template.fit(images, views, entry_pixels, axis_pixels, template=model)

def centered_pixels(pixels, views):
    """Use the midpoint of two opposite shaft edges, never one visible edge."""
    centers = {}
    paired = len(pixels) >= 2
    for name, value in pixels.items():
        value = np.asarray(value, float)
        if name not in views or views[name] is None:
            raise ValueError('Unavailable landmark camera: ' + name)
        if value.shape not in ((2,), (2, 2)) or not np.isfinite(value).all():
            raise ValueError('Use finite [column,row] or two opposite edge pixels [[column,row],[column,row]].')
        h, w = views[name].shape[:2]
        if np.any(value < 0) or np.any(value > [w - 1, h - 1]):
            raise ValueError('Every shaft edge must be inside the original image.')
        if value.shape == (2, 2):
            if not np.isfinite(value).all() or np.linalg.norm(value[1] - value[0]) < 2:
                raise ValueError('Opposite shaft edges must be finite and at least 2 pixels apart; obtain a closer view.')
            centers[name] = value.mean(axis=0).tolist()
        elif value.shape == (2,) and np.isfinite(value).all():
            centers[name] = value.tolist()
            paired = False
        else:
            raise ValueError('Use [column,row] or two opposite edge pixels [[column,row],[column,row]].')
    return (centers, paired)

def landmark(pixels, views):
    """Reuse the existing checked RGB triangulator; its input order is row,col."""
    from martingale.inference.extensions.key.triangulation_math import triangulate_checked
    observations = []
    for name, pixel in pixels.items():
        if name not in views:
            raise ValueError('Unavailable landmark camera: ' + name)
        p = np.asarray(pixel, float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError('Landmarks must use finite (column,row) coordinates.')
        v = views[name]
        observations.append(dict(camera=name, pixel_rc=p[::-1].tolist(), shape=[*v.shape, 3], intrinsic_matrix=[[v.fx, 0, v.cx], [0, v.fy, v.cy], [0, 0, 1]], extrinsic_matrix=v.pose.tolist()))
    result = triangulate_checked(observations, pixel_error_px=1.0)
    if result['accepted']:
        result['samples_xyz'] = uncertainty.triangulated_samples(pixels, views)
    return result

def alignment(primitives, opening, tip_pixels, shaft_pixels, tooth_pixels, approach_clearance_m=0.01, arm=None, *, tip_model='end_face_center', key_geometry=None):
    """Current key pose relative to the measured aperture; does not command motion."""
    from martingale.hosts import robodojo_tools as tools
    if arm is not None and arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    if tip_model not in ('end_face_center', 'rectangular_end_rims', 'calibrated_key'):
        raise ValueError('tip_model must be end_face_center, rectangular_end_rims or calibrated_key')
    if not opening.get('accepted') or opening.get('entry_xyz') is None:
        return dict(accepted=False, ready_to_insert=False, reasons=['An accepted opening measurement is required.'])
    clearance = float(approach_clearance_m)
    if not np.isfinite(clearance) or not 0.002 <= clearance <= 0.05:
        raise ValueError('Approach clearance must be between 2 and 50 mm.')
    obs, _, views = current(primitives)
    shown = tools._shown(obs)
    for observed_arm in ('left', 'right'):
        a = np.asarray(shown[observed_arm + '_ee_pose'], float)
        b = np.asarray(obs['state'][observed_arm + '_ee_pose'], float)
        if np.linalg.norm(a[:3] - b[:3]) > 0.0005 or tools.kin.rotation_error(tools.kin.rot_from_quat(a[3:]), tools.kin.rot_from_quat(b[3:])) > 0.5:
            return dict(accepted=False, ready_to_insert=False, reasons=['Images predate the latest motion; settle and reobserve.'])
    model_mode = tip_model == 'calibrated_key'
    key_pose = tooth_edge = None
    if model_mode:
        try:
            from martingale.inference.extensions.key.hosts import robodojo_key_pose_model as pose_model
        except ImportError:
            import key_pose_model as pose_model
        key_pose = pose_model.measure(primitives, tooth_pixels, arm, tip_pixels, shaft_pixels, key_geometry=key_geometry)
        if not key_pose['accepted']:
            return dict(accepted=False, ready_to_insert=False, reasons=key_pose['reasons'], key_pose=key_pose)
        tip_paired = shaft_paired = True
        edge_mode = rim_mode = True
        axis = np.asarray(key_pose['shaft_axis_xyz'])
        axes = np.asarray(key_pose['shaft_samples_xyz'])
        tip = np.asarray(key_pose['tip_xyz'])
        measurements = {'tip': dict(accepted=True, xyz=tip.tolist(), samples_xyz=key_pose['tip_samples_xyz'], pixel_sensitivity_m=key_pose['pixel_sensitivity_m'], model='calibrated_key')}
        tooth_edge = dict(accepted=True, width_axis_xyz=key_pose['width_axis_xyz'], width_samples_xyz=key_pose['width_samples_xyz'])
        shaft_measurement = dict(accepted=True, method='calibrated_two_tooth_faces', shaft_axis_xyz=axis.tolist())
    else:
        tip_centers, tip_paired = centered_pixels(tip_pixels, views)
        shaft_centers, shaft_paired = centered_pixels(shaft_pixels, views)
        edge_mode = bool(tooth_pixels) and all((np.asarray(p).shape == (2, 2) for p in tooth_pixels.values()))
        rim_mode = tip_model == 'rectangular_end_rims'
        shaft_measurement = None
    if rim_mode and (not model_mode):
        if not (tip_paired and shaft_paired):
            return dict(accepted=False, ready_to_insert=False, reasons=['Rectangular end rims require endpoint pairs in both views.'])
        if tooth_pixels and (not edge_mode):
            return dict(accepted=False, ready_to_insert=False, reasons=['Use a directed transverse tooth edge with rectangular end-rim geometry.'])
        shaft_measurement = shaft_geometry.estimate(views, tip_pixels, shaft_pixels)
        if not shaft_measurement['accepted']:
            return dict(accepted=False, ready_to_insert=False, reasons=shaft_measurement['reasons'], shaft_measurement=shaft_measurement)
        axis = np.asarray(shaft_measurement['shaft_axis_xyz'])
        axes = np.asarray(shaft_measurement['shaft_samples_xyz'])
        measured_tip = tip_geometry.estimate(views, tip_pixels, axis, axes)
        if not measured_tip['accepted']:
            seed = landmark(tip_centers, views) if not tooth_pixels else None
            return dict(accepted=False, ready_to_insert=False, reasons=measured_tip['reasons'], tip_xyz=seed['xyz'] if seed and seed['accepted'] else None, tip_is_observation_seed_only=True, shaft_axis_xyz=axis.tolist(), tip_model=tip_model, landmarks={'tip': measured_tip})
        measurements = {'tip': measured_tip}
        tip = np.asarray(measured_tip['xyz'])
    elif not model_mode:
        inputs = [('tip', tip_centers), ('shaft', shaft_centers)]
        if tooth_pixels and (not edge_mode):
            inputs.append(('tooth', tooth_pixels))
        measurements = {name: landmark(pixels, views) for name, pixels in inputs}
        failed = [name for name, m in measurements.items() if not m['accepted']]
        if failed:
            return dict(accepted=False, ready_to_insert=False, reasons=['Unresolved key landmarks: ' + ', '.join(failed)], landmarks=measurements)
        tip, shaft = [np.asarray(measurements[n]['xyz']) for n in ('tip', 'shaft')]
        axis = shaft - tip
        if np.linalg.norm(axis) < 0.02:
            return dict(accepted=False, ready_to_insert=False, reasons=['Choose a shaft landmark at least 20 mm behind the tip.'])
        axis /= np.linalg.norm(axis)
    if not tooth_pixels and (not model_mode):
        return dict(accepted=False, ready_to_insert=False, reasons=['Shaft measured. Mark a directed transverse tooth edge in a close elevated view.'], tip_xyz=tip.tolist(), shaft_axis_xyz=axis.tolist(), tip_model=tip_model, paired_centerline=tip_paired and shaft_paired, landmarks={name: {k: v for k, v in value.items() if k != 'samples_xyz'} for name, value in measurements.items()})
    if edge_mode:
        if not rim_mode:
            tip_samples = np.asarray(measurements['tip']['samples_xyz'])
            shaft_samples = np.asarray(measurements['shaft']['samples_xyz'])
            axes = (shaft_samples[None, :, :] - tip_samples[:, None, :]).reshape(-1, 3)
            axes /= np.linalg.norm(axes, axis=1)[:, None]
        if not model_mode:
            tooth_edge = tooth_geometry.estimate(views, tooth_pixels, axis, axes)
        if not tooth_edge['accepted']:
            return dict(accepted=False, ready_to_insert=False, reasons=tooth_edge['reasons'], tooth_edge=tooth_edge, tip_xyz=tip.tolist(), shaft_axis_xyz=axis.tolist(), landmarks={name: {k: v for k, v in value.items() if k != 'samples_xyz'} for name, value in measurements.items()})
        width_axis = np.asarray(tooth_edge['width_axis_xyz'])
    else:
        tooth = np.asarray(measurements['tooth']['xyz'])
        width_axis = tooth - tip - axis * ((tooth - tip) @ axis)
        if np.linalg.norm(width_axis) < 0.003:
            return dict(accepted=False, ready_to_insert=False, reasons=['Choose an outer tooth midpoint off the shaft centerline.'])
        width_axis /= np.linalg.norm(width_axis)
    normal = np.asarray(opening['normal_xyz'], float)
    entry = np.asarray(opening['entry_xyz'], float)
    tilt = float(np.rad2deg(np.arccos(np.clip(axis @ normal, -1, 1))))
    lateral = tip - entry - normal * ((tip - entry) @ normal)
    above = float((tip - entry) @ normal)
    desired_width = opening.get('tooth_direction_xyz')
    twist = None
    if desired_width is not None:
        desired_width = np.asarray(desired_width, float)
        projected_width = width_axis - normal * (width_axis @ normal)
        if np.linalg.norm(projected_width) > 0.2:
            projected_width /= np.linalg.norm(projected_width)
            twist = float(np.rad2deg(np.arccos(np.clip(projected_width @ desired_width, -1, 1))))
    reasons = []
    if not (tip_paired and shaft_paired):
        reasons.append('Mark both opposite edges of the shaft end face and the shaft cross-section in each view; a single visible edge is not its center.')
    if tilt > 5:
        reasons.append('Rotate the measured shaft axis toward the opening normal while clear, then remeasure.')
    if twist is None:
        reasons.append('Remeasure the opening with a visible tooth-slot extension; its direction must be resolved, including which side.')
    elif twist > 8:
        reasons.append('Rotate the blade width to match the measured shaft-entry direction, then remeasure.')
    if np.linalg.norm(lateral) > 0.002:
        reasons.append('Center the actual tip over the measured shaft entry while clear.')
    if above < -0.002:
        reasons.append('Tip is below the shallow entry band; do not infer alignment from an occluded tip.')
    if above > 0.05:
        reasons.append('Move to a clear preinsertion pose within 50 mm above the rim, then remeasure.')
    sensitivity = max((m['pixel_sensitivity_m'] for m in measurements.values()))
    if sensitivity > 0.003:
        reasons.append('Landmark sensitivity exceeds 3 mm; obtain a closer supporting view.')
    envelope = None
    opening_uncertainty = opening.get('uncertainty')
    if not opening_uncertainty:
        reasons.append('Remeasure the opening: entry and slot-direction uncertainty are required.')
    else:
        try:
            summarize = uncertainty.summarize_widths if edge_mode else uncertainty.summarize
            envelope = summarize(measurements['tip']['samples_xyz'], measurements['shaft']['samples_xyz'] if not rim_mode else None, tooth_edge['width_samples_xyz'] if edge_mode else measurements['tooth']['samples_xyz'], opening_uncertainty['entry_samples_xyz'], opening_uncertainty['direction_samples_xyz'], normal, **{'shaft_axes': axes} if rim_mode else {})
        except (ValueError, KeyError) as exc:
            reasons.append('Alignment uncertainty is unresolved: ' + str(exc))
        if envelope is not None:
            if envelope['max_tilt_deg'] > 5:
                reasons.append('Shaft tilt can exceed 5 degrees within one-pixel annotation error; use clearer close views.')
            if envelope['max_twist_deg'] > 8:
                reasons.append('Tooth twist can exceed 8 degrees within one-pixel annotation error; obtain a closer elevated view of a straight transverse tooth edge.')
            if envelope['max_lateral_m'] > 0.002:
                reasons.append('Tip centering can exceed 2 mm including key and opening uncertainty; improve the entry and tip observations.')
    correction_clear = above >= 0.003
    if envelope is not None:
        correction_clear = envelope['min_above_m'] >= 0.003
    target = entry + normal * clearance
    preinsert_pose = rotation_pose = None
    if arm is not None and desired_width is not None and (sensitivity <= 0.003) and correction_clear:
        source = np.column_stack([width_axis, np.cross(axis, width_axis), axis])
        destination = np.column_stack([desired_width, np.cross(normal, desired_width), normal])
        correction = destination @ source.T
        ee = np.asarray(obs['state'][arm + '_ee_pose'], float)
        hand_rotation = tools.kin.rot_from_quat(ee[3:])
        tcp = ee[:3] + hand_rotation[:, 0] * tools.kin.TCP_OFFSET
        corrected_rotation = correction @ hand_rotation
        pitch, yaw, roll = tools.kin.angles(corrected_rotation)
        actual_correction = tools.kin.orientation(pitch, yaw, roll) @ hand_rotation.T
        corrected_tcp = target - actual_correction @ (tip - tcp)
        pose = dict(arm=arm, pitch=float(pitch), yaw=float(yaw), roll=float(roll))
        rotation_pose = dict(pose, xyz=tcp.tolist())
        preinsert_pose = dict(pose, xyz=corrected_tcp.tolist())
    return dict(accepted=True, ready_to_insert=not reasons, alignment_within_tolerance=not reasons, reasons=reasons, tip_xyz=tip.tolist(), shaft_axis_xyz=axis.tolist(), blade_width_axis_xyz=width_axis.tolist(), target_shaft_axis_xyz=normal.tolist(), target_blade_width_axis_xyz=opening.get('tooth_direction_xyz'), tilt_deg=tilt, twist_error_deg=twist, lateral_error_m=float(np.linalg.norm(lateral)), tip_above_rim_m=above, translation_to_preinsert_m=(target - tip).tolist(), rotation_only_pose=rotation_pose, preinsert_pose=preinsert_pose, target_tip_xyz=target.tolist(), landmark_sensitivity_m=sensitivity, uncertainty=envelope, correction_clear=correction_clear, paired_centerline=tip_paired and shaft_paired, tip_model=tip_model, shaft_measurement={k: v for k, v in shaft_measurement.items() if k != 'shaft_samples_xyz'} if shaft_measurement else None, centerline_pixels=dict(tip=tip_pixels, shaft=shaft_pixels), tooth_measurement_mode='calibrated_two_tooth_faces' if model_mode else 'transverse_edge' if edge_mode else 'matched_midpoint', key_pose={k: v for k, v in key_pose.items() if k not in ('pose_samples', 'tip_samples_xyz', 'shaft_samples_xyz', 'width_samples_xyz')} if key_pose else None, tooth_edge={k: v for k, v in tooth_edge.items() if k != 'width_samples_xyz'} if tooth_edge else None, landmarks={name: {k: v for k, v in value.items() if k != 'samples_xyz'} for name, value in measurements.items()}, note='accepted means measurable, not aligned. Require ready_to_insert before descent. Pose targets assume a rigid grip and do not check reachability or the swept collision path. Rotate only while clear, then remeasure; do not reuse the old translation after rotation. Readiness includes sampled one-pixel key/slot uncertainty, not a calibrated confidence bound. Incorrect feature identities can still pass; use the same shaft-center and tooth feature in each view. Near the rim only straight shallow entry is allowed; retreat before correction. This checks observed entry alignment, not internal obstruction or successful insertion/turning.')
