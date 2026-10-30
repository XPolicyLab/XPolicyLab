"""Conservative upright numeral support from observed counters and contour.

No equation, layout ID, expected answer, or inventory is an input. Only prune
already supported labels; a weak, clipped, or conflicting cue leaves them alone.
"""
import cv2
import numpy as np
from .interior import depressions


def refine_upright(rgb, glyph, support):
    audit = {'applied': False, 'cue': None}
    if len(support) < 2:
        return support, audit
    ys, xs = np.nonzero(glyph)
    if not len(xs):
        return support, audit
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    m = glyph[y0:y1, x0:x1] > 0
    h, w = m.shape
    if min(h, w) < 20 or not .48 < w/h < .95:
        return support, audit
    structure = depressions(rgb, glyph)
    strong = [c for c in structure['components']
              if c['area']/(w*h) >= .015 and c['clearance']/w >= .14
              and .2 <= c['center'][0]/w <= .8]
    # Open 5 may contain a shallow self-shadow. Reject that alternative only
    # for a larger enclosed counter, not the weaker 6/8/9 discrimination cue.
    if '5' in {c['label'] for c in support}:
        strong = [c for c in strong if c['area']/(w*h) >= .03]
    upper = [c for c in strong if .16 < c['center'][1]/h < .40]
    lower = [c for c in strong if .50 < c['center'][1]/h < .76]
    labels = {c['label'] for c in support}
    keep = set(labels)
    # A single well enclosed counter distinguishes upright 6 from 9. This
    # narrow rule abstains for 0/4 and tiny shadow speckles; an enclosed
    # lower counter also rules out a visually competing open digit 5.
    if labels <= {'5', '6', '8', '9'} and len(strong) == 1:
        if upper and '9' in labels:
            keep = {'9'}; audit['cue'] = 'enclosed_upper_counter'
        elif lower and '6' in labels:
            keep = {'6'}; audit['cue'] = 'enclosed_lower_counter'
    # Open mid-left contour supports 2/3, while an enclosed 8/9 has no such
    # opening. Retain both 2 and 3 when the silhouette cannot separate them.
    left = np.array([np.flatnonzero(row)[0]/w if row.any() else 1. for row in m])
    top = float(np.median(left[int(.2*h):int(.35*h)]))
    waist = float(np.median(left[int(.4*h):int(.6*h)]))
    bottom = float(np.median(left[int(.65*h):int(.8*h)]))
    if not strong and waist-max(top, bottom) >= .12 and waist >= .18 and labels & {'2', '3'}:
        keep = labels - {'8', '9'}; audit['cue'] = 'open_left_waist'
    # Never remove all candidates or override a sharply better silhouette.
    chosen = [dict(c) for c in support if c['label'] in keep]
    if chosen and len(chosen) < len(support) and min(c['cost'] for c in chosen) <= min(c['cost'] for c in support) + .06:
        audit.update(applied=True, before=sorted(labels), after=sorted(keep),
                     strong_counters=strong, left_profile=[top, waist, bottom])
        return chosen, audit
    return support, audit
