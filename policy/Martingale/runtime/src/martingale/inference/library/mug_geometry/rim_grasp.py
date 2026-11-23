"""Local rim pinch candidates from RGB correspondences; no motion.

Two calibrated views mark the SAME inner/outer surface points at the lip and
at the proposed deeper contact. This is a local straight-wall approximation,
not an inner-cup reconstruction or collision certificate.
"""
import itertools
import math
import numpy as np
POINTS = ('outer_top', 'inner_top', 'outer_contact', 'inner_contact')

def _ray(meta, rc):
    k = np.asarray(meta['intrinsic_matrix'], float)
    pose = np.asarray(meta['extrinsic_matrix'], float)
    rc = np.asarray(rc, float)
    h, w = meta['shape'][:2]
    if k.shape != (3, 3) or pose.shape != (4, 4) or rc.shape != (2,) or (not all((np.isfinite(x).all() for x in (k, pose, rc)))) or (min(k[0, 0], k[1, 1]) <= 0) or (not (0 <= rc[0] < h and 0 <= rc[1] < w)):
        raise ValueError('Invalid calibration or out-of-image row/column.')
    direction = pose[:3, :3] @ np.array([(rc[1] - k[0, 2]) / k[0, 0], -(rc[0] - k[1, 2]) / k[1, 1], -1.0])
    return (pose[:3, 3], direction / np.linalg.norm(direction))

def _triangulate(metas, pixels):
    rays = [_ray(m, p) for m, p in zip(metas, pixels)]
    origins, directions = map(np.array, zip(*rays))
    angle = math.degrees(math.acos(np.clip(abs(directions[0] @ directions[1]), 0, 1)))
    if angle < 8:
        raise ValueError('Views have insufficient triangulation angle; obtain a supporting view.')
    distances = np.linalg.lstsq(np.column_stack([directions[0], -directions[1]]), origins[1] - origins[0], rcond=None)[0]
    if min(distances) <= 0:
        raise ValueError('Correspondence triangulates behind a camera.')
    ends = origins + distances[:, None] * directions
    return (ends.mean(0), float(np.linalg.norm(ends[0] - ends[1])))

def rim_grasp_plan(observations, *, clearance_m=0.008):
    """Return an abstaining candidate planner for an upright, stationary mug.

    Exactly two observations: {camera, meta, outer_top_rc, inner_top_rc,
    outer_contact_rc, inner_contact_rc}. All coordinates are [row,column] in
    original images, on the SAME local wall section in both views. Contact
    points must be observed below the lip, never extrapolated from an outer
    silhouette. The expected pad contact midpoint is returned; inspect actual
    finger/palm geometry before treating it as a TCP motion target.
    """
    result = dict(valid=False, part='rim', motion_authorized=False, reasons=[])
    try:
        if len(observations) != 2 or len({o['camera'] for o in observations}) != 2:
            raise ValueError('Provide two distinct current camera views of the same wall section.')
        if not math.isfinite(clearance_m) or clearance_m < 0.008:
            raise ValueError('Use at least 8 mm total approach clearance.')
        metas = [o['meta'] for o in observations]
        points, spreads = ({}, {})
        perturbations = np.array(list(itertools.product((-1.0, 1.0), repeat=4))).reshape(-1, 2, 2)
        for name in POINTS:
            pixels = np.array([o[name + '_rc'] for o in observations], float)
            if pixels.shape != (2, 2):
                raise ValueError('Each feature needs row/column in both views.')
            point, residual = _triangulate(metas, pixels)
            if residual > 0.002:
                raise ValueError('Surface correspondences disagree by over 2 mm.')
            variants = np.array([_triangulate(metas, pixels + delta)[0] for delta in perturbations])
            spread = float(np.linalg.norm(variants - point, axis=1).max())
            if spread > 0.002:
                raise ValueError('Local surface position is too sensitive to pixel errors; get a closer supporting view.')
            points[name], spreads[name] = (point, spread)
        outer, inner = (points['outer_contact'], points['inner_contact'])
        delta = outer - inner
        width = np.linalg.norm(delta[:2])
        uncertainty = max(spreads.values())
        if width <= spreads['outer_contact'] + spreads['inner_contact']:
            raise ValueError('Inner and outer walls are not resolved separately at contact depth.')
        if not 0.001 <= width <= 0.02:
            raise ValueError('Local wall width is implausible; check surface identity.')
        closing = np.r_[delta[:2] / width, 0.0]
        tangent = np.array([-closing[1], closing[0], 0.0])
        target = (outer + inner) / 2
        for level in ('top', 'contact'):
            a, b = (points['outer_' + level], points['inner_' + level])
            if abs(a[2] - b[2]) > 0.002 or np.dot(a - b, closing) <= 0:
                raise ValueError('Wall endpoints must be paired at the same height and consistent inner/outer order.')
        top = (points['outer_top'] + points['inner_top']) / 2
        depth = float(top[2] - target[2])
        if not 0.006 <= depth <= 0.035:
            raise ValueError('Observed pad contact must be 6–35 mm below the lip; do not extrapolate.')
        vertices = np.array(list(points.values()))
        if np.ptp(vertices @ tangent) > 0.003:
            raise ValueError('Selected features do not describe one local wall cross-section.')
        half_corridor = float(np.abs((vertices - target) @ closing).max())
        gap = 2 * (half_corridor + uncertainty) + clearance_m
        result.update(surface_points_xyz={k: v.tolist() for k, v in points.items()}, contact_xyz=target.tolist(), closing_axis_xyz=closing.tolist(), local_width_m=float(width), depth_below_lip_m=depth, approach_width_m=2 * half_corridor, pixel_sensitivity_m=uncertainty, gap_m=gap, clearance_m=clearance_m, pitch_deg=90.0, yaw_deg=math.degrees(math.atan2(closing[1], closing[0])), roll_deg=0.0, note='Local pad-contact candidate only. Verify same wall identity, straight surfaces between samples, pad overlap, fingertip-to-floor and palm-to-lip clearance, nearby handle, and reachability in current views. No motion is authorized by valid=True. Reobserve after the mug moves.')
        if gap > 0.0888:
            raise ValueError('Wall corridor plus clearance exceeds the physical gripper opening.')
        result['valid'] = True
    except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        result['reasons'] = [str(exc)]
    return result
