# Copyright © 2026 Martingale. All Rights Reserved.
import math
import textwrap
import numpy as np

def project(meta, point):
    """Camera-to-world pose, camera looking down -z with +y up."""
    pose = np.asarray(meta['extrinsic_matrix'], float)
    k = np.asarray(meta['intrinsic_matrix'], float)
    x, y, z = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
    if z >= -1e-06:
        return None
    return [float(k[0, 2] + k[0, 0] * x / -z), float(k[1, 2] - k[1, 1] * y / -z)]

def slot_basis(opening, normal):
    tooth = opening.get('tooth_direction_xyz')
    along = np.asarray(tooth if tooth is not None else np.eye(3)[np.argmin(abs(normal))], float)
    if along.shape != (3,) or not np.isfinite(along).all():
        raise ValueError('Invalid tooth_direction_xyz')
    along = along - along @ normal * normal
    if np.linalg.norm(along) < 1e-08:
        raise ValueError('tooth_direction_xyz is parallel to the opening normal')
    along /= np.linalg.norm(along)
    return (along, np.cross(normal, along), 'slot' if tooth is not None else 'arbitrary_transverse')

def _context_equal(a, b):
    if any((a[k] != b[k] for k in ('arm', 'names', 'landmark'))):
        return False
    return all((np.allclose(a[k], b[k], atol=1e-08, rtol=0) for k in ('entry', 'normal', 'along')))

def build_report(robot, result, opening, names, metas, tips, shafts, arm, landmark):
    entry = np.asarray(opening['entry_xyz'], float)
    normal = np.asarray(opening['normal_xyz'], float)
    normal /= np.linalg.norm(normal)
    along, across, basis_name = slot_basis(opening, normal)
    step = result['observed_step']
    valid = bool(result.get('accepted'))
    views = {}
    for name, meta, tip_pixel, shaft_pixel in zip(names, metas, tips, shafts):
        axis = [project(meta, entry), project(meta, entry + 0.03 * normal)]
        view = dict(camera=name, meta=meta, tip_pixel=tip_pixel.tolist(), shaft_pixel=shaft_pixel.tolist(), expected_axis_pixels=axis, target_tip_pixel=None, expected_shaft_pixel=None, projected_shaft_angle_deg=None, tip_distance_to_axis_px=None)
        if all((p is not None for p in axis)):
            d = np.asarray(axis[1]) - axis[0]
            if np.linalg.norm(d) >= 2.0:
                error = tip_pixel - axis[0]
                view['tip_distance_to_axis_px'] = float(abs(d[0] * error[1] - d[1] * error[0]) / np.linalg.norm(d))
        if valid:
            tip = np.asarray(result['tip_xyz'])
            length = np.linalg.norm(np.asarray(result['shaft_xyz']) - tip)
            view['target_tip_pixel'] = project(meta, entry + result['tip_above_rim_m'] * normal)
            expected = project(meta, tip + length * normal)
            projected_tip = project(meta, tip)
            view['expected_shaft_pixel'] = expected
            if expected is not None and projected_tip is not None:
                a = shaft_pixel - tip_pixel
                b = np.asarray(expected) - projected_tip
                if min(np.linalg.norm(a), np.linalg.norm(b)) >= 2.0:
                    view['projected_shaft_angle_deg'] = float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        views[name] = view
    result['alignment_views'] = views
    context = dict(arm=arm, names=names, landmark=landmark, entry=entry.tolist(), normal=normal.tolist(), along=along.tolist())
    observer = 'cam_' + ('right' if arm == 'left' else 'left') + '_wrist'
    observer_pose = np.asarray(metas[names.index(observer)]['extrinsic_matrix']) if observer in names else None
    previous = getattr(robot, '_key_alignment_report_previous', None)
    comparable = False
    reason = 'No accepted preceding measurement.'
    if not valid:
        reason = 'Current measurement is unresolved; reidentify landmarks in existing images/crops or use the single-view overlay. Do not reposition cameras.'
    elif previous:
        if previous['step'] >= step:
            reason = 'No intervening recorded movement; this is not a correction comparison.'
        elif not _context_equal(previous['context'], context):
            reason = 'Camera names, landmark identity, or opening calibration changed.'
        elif np.linalg.norm(previous['observer_pose'][:3, 3] - observer_pose[:3, 3]) >= 0.0005 or np.linalg.norm(previous['observer_pose'][:3, :3] - observer_pose[:3, :3]) >= 0.005:
            reason = 'Empty-wrist camera moved; establish a new comparison baseline.'
        else:
            comparable, reason = (True, 'Same opening, landmarks and stationary empty wrist.')
    rows, values, sensitivity = ([], {}, {})
    if valid:
        offset = np.asarray(result['tip_xyz']) - entry
        values = dict(along_mm=float(offset @ along * 1000), across_mm=float(offset @ across * 1000), lateral_mm=result['lateral_error_m'] * 1000, tilt_deg=result['tilt_deg'], height_mm=result['tip_above_rim_m'] * 1000)
        sensitivity = {k: result['tilt_pixel_sensitivity_deg'] if k == 'tilt_deg' else result['tip_pixel_sensitivity_m'] * 1000 for k in values}
        for key, value in values.items():
            before = previous['values'][key] if comparable else None
            reduction = abs(before) - abs(value) if comparable and key != 'height_mm' else None
            noise = sensitivity[key] + previous['sensitivity'][key] if comparable else None
            status = 'baseline'
            if comparable:
                status = 'height_change_only' if reduction is None else 'improved' if reduction > noise else 'worsened' if reduction < -noise else 'within_pixel_sensitivity'
            rows.append(dict(metric=key, before=before, after=value, absolute_error_reduction=reduction, comparison_sensitivity=noise, current_pixel_sensitivity=sensitivity[key], status=status))
    lines = ['metric       before      after    abs-error reduction   pixel sensitivity   result']
    for row in rows:
        before = '--' if row['before'] is None else f"{row['before']:.3f}"
        delta = '--' if row['absolute_error_reduction'] is None else f"{row['absolute_error_reduction']:+.3f}"
        lines.append(f"{row['metric']:12} {before:>8} {row['after']:10.3f} {delta:>14} {row['current_pixel_sensitivity']:16.3f}   {row['status']}")
    if not valid:
        lines.append('UNRESOLVED: ' + '; '.join(result.get('reasons', [])))
    elif not comparable:
        lines.append('Baseline only: ' + reason)
    lines.append('Pixel sensitivity only; hole calibration error and wrong landmark identity are NOT included.')
    result['alignment_report'] = dict(status='measured' if valid else 'unresolved', comparison_valid=comparable, comparison_reason=reason, basis=basis_name, rows=rows, table_text='\n'.join(lines), correction_xyz=result.get('lateral_correction_xyz'), uncertainty_note='One-pixel perturbation sensitivity is not a confidence interval. Hole pose/normal uncertainty and systematic landmark mistakes are unquantified.', descent_certified=False)
    if valid:
        robot._key_alignment_report_previous = dict(step=step, context=context, observer_pose=observer_pose.copy(), values=values, sensitivity=sensitivity)
    else:
        robot._key_alignment_report_previous = None
    print(result['alignment_report']['table_text'])
    return result

def overlay(robot, measurement):
    """Return a PIL montage; caller attaches it with display(...). No motion."""
    from PIL import Image, ImageDraw, ImageFont
    if measurement.get('observed_step') != robot.state().step:
        raise ValueError('Alignment overlay requires the current measurement step')
    views = measurement.get('alignment_views', {})
    if not views:
        raise ValueError('No alignment views available; obtain a fresh annotated measurement')
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    panels = []
    for name, view in views.items():
        raw = Image.fromarray(np.asarray(robot.image(name, step=measurement['observed_step']), dtype=np.uint8)).convert('RGB')
        annotated = raw.copy()
        draw = ImageDraw.Draw(annotated)

        def mark(p, color, label):
            if p is not None and all((math.isfinite(x) and abs(x) < 1000000.0 for x in p)):
                x, y = p
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                draw.text((x + 6, y + 3), label, fill=color, font=font)

        def line(points, color, width=2):
            if all((p is not None and all((math.isfinite(x) and abs(x) < 1000000.0 for x in p)) for p in points)):
                draw.line([tuple(p) for p in points], fill=color, width=width)
        axis = view['expected_axis_pixels']
        line(axis, '#4dff88', 3)
        mark(axis[0], '#4dff88', 'hole')
        line([view['tip_pixel'], view['shaft_pixel']], '#ff5757')
        mark(view['tip_pixel'], '#ff5757', 'tip')
        mark(view['shaft_pixel'], '#ff5757', 'shaft')
        line([view['tip_pixel'], view['expected_shaft_pixel']], '#ffd54d')
        line([view['tip_pixel'], view['target_tip_pixel']], '#55dfff')
        mark(view['target_tip_pixel'], '#55dfff', 'target at tip height')
        points = [view['tip_pixel'], view['shaft_pixel']]
        if axis[0] is not None and 0 <= axis[0][0] < raw.width and (0 <= axis[0][1] < raw.height):
            points.append(axis[0])
        coords = np.asarray(points)
        size = min(max(96, int(np.max(np.ptp(coords, axis=0))) + 40), min(raw.size))
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2
        x = int(np.clip(center[0] - size / 2, 0, raw.width - size))
        y = int(np.clip(center[1] - size / 2, 0, raw.height - size))
        box = (x, y, x + size, y + size)
        panel = Image.new('RGB', (640, 868), '#151b24')
        ImageDraw.Draw(panel).text((10, 6), f"{name} | step {measurement['observed_step']}", fill='white', font=font)
        panel.paste(annotated.resize((640, 480)), (0, 30))
        panel.paste(raw.crop(box).resize((320, 320), Image.Resampling.NEAREST), (0, 540))
        panel.paste(annotated.crop(box).resize((320, 320), Image.Resampling.NEAREST), (320, 540))
        pd = ImageDraw.Draw(panel)
        pd.text((5, 516), 'Raw crop (no added detail)', fill='white', font=font)
        pd.text((325, 516), 'Annotated crop', fill='white', font=font)
        panels.append(panel)
    width = 640 * len(panels)
    legend = 'Green: expected axis | Red: annotated key | Cyan: lateral target at tip height | Yellow: desired shaft direction'
    table = [part for line in [legend, *measurement['alignment_report']['table_text'].splitlines()] for part in textwrap.wrap(line, width=max(50, width // 9), replace_whitespace=False)]
    out = Image.new('RGB', (width, 890 + 20 * (len(table) + 2)), '#151b24')
    for i, panel in enumerate(panels):
        out.paste(panel, (640 * i, 0))
    d = ImageDraw.Draw(out)
    for i, text in enumerate(table):
        d.text((10, 875 + 20 * i), text, fill='white', font=font)
    return out
