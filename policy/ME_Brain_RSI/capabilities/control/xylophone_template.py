"""Whole fixed-size keyboard pose from RGB; no per-scene coordinates.

X21's successful rigid fits remain preferred. A failure-only dense color-template
fit does not require separately connected full bar contours. Eight targets are
always one rigid transform of the same public template, never independent fits.
"""
from functools import lru_cache
import cv2
import numpy as np
from .xylophone_rgb import (BAR_CENTERS, BAR_LENGTHS, BAR_WIDTH, COLOR_GROUPS,
                            HUE_RANGES, fit_painted_keys)
from .xylophone_keys import KEY_HITS


@lru_cache(maxsize=512)
def masks(degrees, scale):
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    polygons = []
    for center, length in zip(BAR_CENTERS, BAR_LENGTHS):
        polygon = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        polygon = polygon * [BAR_WIDTH / 2, length / 2] + center
        polygons.append(polygon @ rotation.T * scale)
    low = np.floor(np.vstack(polygons).min(0)).astype(int) - 2
    high = np.ceil(np.vstack(polygons).max(0)).astype(int) + 2
    shape = tuple((high-low+1)[::-1])
    bars = []
    for polygon in polygons:
        mask = np.zeros(shape, np.float32)
        cv2.fillConvexPoly(mask, np.rint(polygon-low).astype(np.int32), 1.)
        bars.append(mask)
    weighted = [sum((bars[i] / bars[i].sum() / 8 for i in range(8)
                     if COLOR_GROUPS[i] == group), np.zeros(shape, np.float32))
                for group in range(6)]
    return rotation, low, bars, weighted


def color_maps(rgb):
    hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    maps = []
    for group, (lo, hi) in enumerate(HUE_RANGES):
        color = (h >= lo) & (h <= hi)
        if group == 0: color |= h >= 160
        # Preserve discrimination against brown wood. Soft saturation avoids
        # breaking a whole painted bar at a highlight or adjacent bolt.
        confidence = np.clip((s.astype(float)-90)/60, 0, 1)
        confidence *= np.clip((v.astype(float)-(100 if group < 3 else 25))/40, 0, 1)
        maps.append((color * confidence).astype(np.float32))
    return maps


def _score_angle(maps_, angle, scale):
    rotation, low, bars, weighted = masks(float(angle), float(scale))
    if any(a < b for a, b in zip(maps_[0].shape, bars[0].shape)):
        return None
    score = sum(cv2.matchTemplate(image, stencil, cv2.TM_CCORR)
                for image, stencil in zip(maps_, weighted))
    y, x = np.unravel_index(score.argmax(), score.shape)
    value = float(score[y, x])
    yy,xx=np.ogrid[:score.shape[0],:score.shape[1]]
    separated=(xx-x)**2+(yy-y)**2>(.04*scale)**2
    spatial_rival=float(np.max(score[separated])) if np.any(separated) else 0.
    # Quadratic interpolation of the global rigid translation only.
    offset = np.zeros(2)
    for axis in range(2):
        index, limit = (x, score.shape[1]) if axis == 0 else (y, score.shape[0])
        if 0 < index < limit-1:
            a, b, c = (score[y,x-1],score[y,x],score[y,x+1]) if axis == 0 else (score[y-1,x],score[y,x],score[y+1,x])
            denom = a-2*b+c
            if denom < -1e-6: offset[axis] = np.clip(.5*(a-c)/denom, -.5, .5)
    origin = np.array([x,y], float) + offset - low
    support = [float((maps_[group][y:y+bar.shape[0],x:x+bar.shape[1]]*bar).sum()/bar.sum())
               for group, bar in zip(COLOR_GROUPS, bars)]
    return dict(origin_px=origin, rotation=rotation, dense_score=value,
                paint_support=support, angle_deg=float(angle), spatial_rival_score=spatial_rival,
                key_pixels=KEY_HITS @ rotation.T * scale + origin)


def fit_fixed_keyboard(rgb, scale=600, legacy_fit=None):
    old = (fit_painted_keys if legacy_fit is None else legacy_fit)(rgb, scale)
    if old is not None:
        return dict(old, pose_method='X21_rigid_constellation', fixed_scale=True)
    maps_ = color_maps(rgb)
    if sum(np.count_nonzero(m > .5) for m in maps_) < .012*scale*scale:
        return None
    # No scene heading prior: search a full turn at fixed scale, then refine
    # both distinct directional modes. All eight colors/order jointly score.
    coarse = [_score_angle(maps_, a, scale) for a in range(-180,180,5)]
    coarse = sorted((c for c in coarse if c is not None), key=lambda c:-c['dense_score'])
    if not coarse: return None
    starts = []
    for c in coarse:
        if all(abs((c['angle_deg']-p['angle_deg']+180)%360-180) > 15 for p in starts):
            starts.append(c)
        if len(starts) == 3: break
    refined = [_score_angle(maps_, p['angle_deg']+d, scale)
               for p in starts for d in np.arange(-4.5,5,.5)]
    refined = sorted((c for c in refined if c is not None), key=lambda c:-c['dense_score'])
    best = refined[0]
    support = np.array(best['paint_support'])
    # Full rectangles include unpainted fasteners/bevels; require broad area
    # coverage on ALL eight bars, not a handful of favorable center pixels.
    if (best['dense_score'] < .60 or min(support) < .50 or sum(support >= .60) < 6
            or sum(support[:4] >= .60) < 3):
        return None
    rivals = [c for c in refined if np.max(np.linalg.norm(c['key_pixels']-best['key_pixels'],axis=1)) > .04*scale]
    runner_up=max([c['dense_score'] for c in rivals]+[c['spatial_rival_score'] for c in refined])
    margin = best['dense_score']-runner_up
    if margin < .08: return None
    return dict(best, pose_method='X28_whole_fixed_color_template', fixed_scale=True,
                ambiguity_margin=float(margin), contour_keys=[],
                matched_keys=np.flatnonzero(support >= .60).tolist())
