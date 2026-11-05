"""Mallet head/shaft RGB geometry for a supplied rectification plane.

Public mesh: red spherical head radius 13 mm, wooden handle diameter 7.5 mm,
local +Y from head towards handle end at 200.6 mm. Uses red circular support
plus TWO parallel shaft boundaries, never color alone. It observes planar
head position/directed axis, NOT depth, roll or a verified rigid grasp.
"""
from .xylophone_measurement import round_heads, dense_shaft
from itertools import combinations
import cv2
import numpy as np


def _refined_shaft_candidates(heads, lines, distance, gray, scale):
    """Refine fragmented Hough seeds against the complete bilateral edges.

    Two-pixel endpoint uncertainty over the 40 mm minimum seed gives the
    angular search bound. Half-pixel endpoint spacing sets the angle grid.
    Hough segments only propose directions; the unchanged metric width,
    center offset, 30--190 mm extent and 80% BOTH-side support decide acceptance.
    """
    candidates = []
    step = np.arctan2(.5, .19*scale)
    bins = int(np.ceil(2*np.pi/step))
    step = 2*np.pi/bins
    reach = int(np.ceil(np.arctan2(2., .04*scale)/step))
    offsets = np.unique(np.r_[np.arange(-.009*scale, .009*scale, .5),
                              0., .009*scale])
    for head in heads:
        indices = set()
        for line in lines[:, 0]:
            points = np.asarray(line, float).reshape(2, 2)
            near, far = sorted(points, key=lambda p: np.linalg.norm(p-head))
            axis = far-near
            norm = np.linalg.norm(axis)
            if norm == 0:
                continue
            axis /= norm
            normal = np.array([-axis[1], axis[0]])
            along = (points-head)@axis
            offset = float((points-head).mean(0)@normal)
            if not (abs(offset) <= .009*scale
                    and -.005*scale <= min(along) <= .14*scale
                    and .07*scale <= max(along) <= .22*scale):
                continue
            center = int(round(np.arctan2(axis[1], axis[0])/step))
            indices.update((center+d) % bins for d in range(-reach, reach+1))
        if not indices:
            continue
        angles = np.array(sorted(indices))*step
        axes = np.c_[np.cos(angles), np.sin(angles)]
        normals = np.c_[-axes[:, 1], axes[:, 0]]
        base = head+axes[:, None, :]*np.linspace(.03, .19, 40)[None, :, None]*scale
        support, errors = [], []
        for offset in offsets:
            samples = np.rint(base+normals[:, None, :]*offset).astype(int)
            inside = ((samples[:, :, 0] >= 0) & (samples[:, :, 0] < distance.shape[1])
                      & (samples[:, :, 1] >= 0) & (samples[:, :, 1] < distance.shape[0]))
            values = distance[np.clip(samples[:, :, 1], 0, distance.shape[0]-1),
                              np.clip(samples[:, :, 0], 0, distance.shape[1]-1)]
            support.append(inside.all(1) & ((values <= 1.5).mean(1) >= .8))
            errors.append(values.mean(1))
        support, errors = np.array(support), np.array(errors)
        fits = []
        for j, left in enumerate(offsets):
            for k, right in enumerate(offsets):
                if (left >= 0 or right <= 0 or not .004*scale <= right-left <= .012*scale
                        or abs(left+right)/2 > .003*scale):
                    continue
                for index in np.flatnonzero(support[j] & support[k]):
                    fits.append((errors[j, index]+errors[k, index], index, right-left))
        if not fits:
            continue
        _, index, width = min(fits)
        # Do not select a favorable fit if full-length evidence has two axes.
        if any(axes[index]@axes[other] < .98 for _, other, _ in fits):
            return []
        axis = axes[index]
        normal = normals[index]
        centerline = head+np.linspace(.03, .19, 40)[:, None]*scale*axis
        # Additional bright wooden-shaft evidence: the inside must be brighter
        # than BOTH external strips along 80% of the same verified extent.
        # This rejects a table/shadow edge that happens to form two Hough seeds.
        # Never interpret this appearance prior as measured depth or attachment.
        pixels = []
        for offset in (0., -width/2-2., width/2+2.):
            xy = np.rint(centerline+normal*offset).astype(int)
            if not ((xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1])
                    & (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])).all():
                break
            pixels.append(gray[xy[:, 1], xy[:, 0]].astype(float))
        if len(pixels) != 3 or any(np.mean(pixels[0] > side) < .8 for side in pixels[1:]):
            continue
        candidates.append(dict(head_px=head.copy(), axis_px=axis,
                               handle_end_estimated_px=head+axis*.2006*scale,
                               shaft_width_m=width/scale,
                               observability="planar_head_and_axis_only",
                               roll_observed=False, depth_observed=False,
                               shaft_fit="bilateral_raster_refinement"))
    return candidates


def _original_planar_mallet(rgb, scale=600):
    if (not np.isfinite(scale) or scale <= 0 or rgb.ndim != 3
            or rgb.shape[2] != 3 or rgb.dtype != np.uint8):
        raise ValueError("Expected uint8 RGB and positive metric image scale")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # Rendered rose-red still needs round-head and full bilateral shaft support.
    # Color alone never establishes tool geometry or a successful grasp.
    red = np.uint8(((h <= 7) | (h >= 160)) & (s >= 120) & (v >= 40))*255
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or 4*np.pi*area/perimeter**2 < .65:
            continue
        center, radius = cv2.minEnclosingCircle(contour)
        if not .009*scale <= radius <= .018*scale:
            continue
        if area/(np.pi*radius**2) < .68:
            continue
        heads.append(np.array(center))
    stable=round_heads(rgb,(.009*scale,.018*scale))
    dense=dense_shaft(rgb,[x['center'] for x in stable],scale) if stable else None
    if dense is not None:
        return dense
    if not heads:
        return None
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 30, 90)
    # Short raster fragments only propose axes; acceptance still verifies
    # the full 30--190 mm length on BOTH handle sides below.
    lines = cv2.HoughLinesP(edges, 1, np.pi/720, threshold=15,
                           minLineLength=.04*scale, maxLineGap=.01*scale)
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    if lines is None:
        return None
    candidates = []
    for head in heads:
        sides = []
        for line in lines[:, 0]:
            points = np.asarray(line, float).reshape(2, 2)
            near, far = sorted(points, key=lambda p: np.linalg.norm(p-head))
            axis = far-near
            axis /= np.linalg.norm(axis)
            normal = np.array([-axis[1], axis[0]])
            longitudinal = (points-head)@axis
            offset = float((points-head).mean(0)@normal)
            if (abs(offset) <= .009*scale and
                    -.005*scale <= min(longitudinal) <= .14*scale and
                    .07*scale <= max(longitudinal) <= .22*scale):
                sides.append((axis, offset, far))
        for a, b in combinations(sides, 2):
            if np.dot(a[0], b[0]) < .995 or a[1]*b[1] >= 0:
                continue
            width = abs(a[1]-b[1])/scale
            if not .004 <= width <= .012:
                continue
            axis = a[0]+b[0]
            axis /= np.linalg.norm(axis)
            if abs(a[1]+b[1])/2 > .003*scale:
                continue
            # A Hough seed can be fragmented: verify both full-length edges.
            supports = []
            for side in (a, b):
                normal = np.array([-side[0][1], side[0][0]])
                samples = (head+np.linspace(.03,.19,40)[:,None]*scale*side[0]
                           +normal*side[1]).round().astype(int)
                inside = ((samples[:,0]>=0)&(samples[:,0]<edges.shape[1])&
                          (samples[:,1]>=0)&(samples[:,1]<edges.shape[0]))
                supports.append(bool(inside.all()) and
                                np.mean(distance[samples[:,1],samples[:,0]]<=1.5)>=.8)
            if not all(supports):
                continue
            candidates.append(dict(head_px=head.copy(), axis_px=axis,
                                   handle_end_estimated_px=head+axis*.2006*scale,
                                   shaft_width_m=width,
                                   observability="planar_head_and_axis_only",
                                   roll_observed=False, depth_observed=False))
    if not candidates:
        candidates = _refined_shaft_candidates(heads, lines, distance,
                                               cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), scale)
    if not candidates:
        return None
    first = candidates[0]
    if any(np.linalg.norm(c["head_px"]-first["head_px"]) > .006*scale
           or np.dot(c["axis_px"], first["axis_px"]) < .98 for c in candidates[1:]):
        return None
    return first


def detect_planar_mallet(rgb, scale=600):
    """Preserve every accepted original output; ellipse fallback only on failure."""
    original=_original_planar_mallet(rgb,scale)
    if original is not None:return original
    from .xylophone_measurement import elliptical_heads
    heads=elliptical_heads(rgb,(.009*scale,.018*scale))
    fallback=dense_shaft(rgb,[h['center'] for h in heads],scale) if heads else None
    if fallback is not None:
        fallback['head_fit']='three_threshold_filled_ellipse_fallback'
        return fallback
    from .xylophone_measurement import saturation_core_heads
    heads=saturation_core_heads(rgb,(.009*scale,.018*scale))
    fallback=dense_shaft(rgb,[h['center'] for h in heads],scale) if heads else None
    if fallback is not None:
        fallback['head_fit']='bounded_saturation_core_full_shaft'
        fallback['measurement_is_depth']=False
        fallback['grasp_verified']=False
    return fallback
