"""Unconnected RGB prototype fitted to public painted key geometry, not reward boxes.

Independently implemented. Dimensions are /World/visual material-subset bounds
from Geometry/xylophone/00000/object.usdz. No layout/state/reward input.
Rectification must use the visible key surface plane; real-RGB validation is
required before policy integration. Missing/ambiguous evidence returns None.
"""
from itertools import combinations, product
import cv2
import numpy as np
from .xylophone_keys import KEY_HITS

# x/y centers and visible bar lengths in metres, extracted from public mesh.
BAR_CENTERS = np.array([
    [.14663138, -.01484262], [.10415124, -.01484262],
    [.06167108, -.00858272], [.01919091, -.00399213],
    [-.02328924, -.00274015], [-.06576940, .00059848],
    [-.10824955, .00143313], [-.15072970, .00518906]])
BAR_LENGTHS = np.array([.20894937, .19604786, .18561724, .18019819,
                        .17044276, .15778573, .14625192, .14240152])
BAR_WIDTH = .03861445
COLOR_GROUPS = (0, 1, 2, 3, 4, 4, 5, 5)
# Real fixed-panel RGB includes rose-red and yellow-orange painted bars.
HUE_RANGES = ((0, 7), (8, 25), (26, 38), (39, 90), (91, 130), (131, 159))


def bar_candidates(rgb, scale=600):
    """Conservative elongated colored components in a metric rectified image."""
    if scale <= 0 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image and positive pixels/metre")
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    ranges = HUE_RANGES
    candidates = []
    for group, (lo, hi) in enumerate(ranges):
        color = (hue >= lo) & (hue <= hi)
        if group == 0:
            color |= hue >= 160
        # Painted saturated surfaces only; brown wood shares orange hue.
        # Initial prototype threshold, not validated against actual scenes.
        mask = np.uint8(color & (sat >= 160) & (val >= 35)) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width, length = sorted(rect[1])
            if not (.025*scale <= width <= .055*scale and
                    .10*scale <= length <= .24*scale and length/width >= 2.5):
                continue
            if cv2.contourArea(contour) < .65*width*length:
                continue
            box = cv2.boxPoints(rect)
            edges = np.roll(box, -1, axis=0)-box
            long_axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
            long_axis /= np.linalg.norm(long_axis)
            candidates.append(dict(group=group, center=np.array(rect[0]),
                                   length=length/scale, axis=long_axis))
    return candidates


def _paint_support(hsv, rotation, origin, scale):
    """Verify interiors independently of connected contour completeness.

    Fifteen points per public bar avoid reliance on a single center pixel.
    Hue alone is never a pose proposal; several metric contours must agree.
    """
    supports = []
    for i, group in enumerate(COLOR_GROUPS):
        grid = np.array([[x, y] for x in (-.2, 0, .2)
                         for y in (-.35, -.2, 0, .2, .35)])
        grid = grid*np.array([BAR_WIDTH, BAR_LENGTHS[i]])+BAR_CENTERS[i]
        pixels = np.rint(grid@rotation.T*scale+origin).astype(int)
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= hsv.shape[1])
                or np.any(pixels[:, 1] >= hsv.shape[0])):
            supports.append(0.0)
            continue
        hue, sat, val = hsv[pixels[:, 1], pixels[:, 0]].T
        lo, hi = HUE_RANGES[group]
        color = (hue >= lo) & (hue <= hi)
        if group == 0:
            color |= hue >= 160
        # Dark brown table hue must not count as a missing orange/yellow bar.
        minimum_value = 150 if group < 3 else 35
        supports.append(float(np.mean(color & (sat >= 140) & (val >= minimum_value))))
    return supports


def _bar_endpoint_error(matches, observed, rotation, origin, scale):
    """Max residual of BOTH public long-axis endpoints of every matched bar.

    Sign-align the unoriented measured axis. This is extra contour evidence
    for ambiguity resolution, not a relaxed pixel/color/3D acceptance gate.
    """
    errors=[]
    for index,item in zip(matches,observed):
        direction=item['axis']*np.sign(item['axis']@rotation[:,1])
        for sign in (-1,1):
            expected=(BAR_CENTERS[index]+[0.,sign*BAR_LENGTHS[index]/2])@rotation.T*scale+origin
            measured=item['center']+sign*direction*item['length']*scale/2
            errors.append(float(np.linalg.norm(expected-measured)))
    return max(errors) if errors else float('inf')


def _endpoint_excludes_rival(best, other, limit):
    """Only a supported best pose may exclude a geometrically unsupported rival."""
    first=best.get('endpoint_max_px',float('inf'))
    second=other.get('endpoint_max_px',float('inf'))
    return bool(np.isfinite(first) and np.isfinite(second) and first<=limit<second)


def fit_painted_keys(rgb, scale=600, max_error_px=4.0):
    """Fit ordered public bar constellation; never infer order from image x alone.

    Metric elongated contours propose a pose; painted interiors verify it.
    At least three contour matches, six painted bars and three of the first
    four distinct colors are required. This remains a plane-conditioned fit,
    not an observed 3D pose or measured functional-point accuracy.
    """
    candidates = bar_candidates(rgb, scale)
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hypotheses = []
    for a, b in combinations(range(8), 2):
        for ca, cb in product([c for c in candidates if c["group"] == COLOR_GROUPS[a]],
                              [c for c in candidates if c["group"] == COLOR_GROUPS[b]]):
            local = (BAR_CENTERS[b]-BAR_CENTERS[a])*scale
            observed = cb["center"]-ca["center"]
            if abs(np.linalg.norm(local)-np.linalg.norm(observed)) > 2*max_error_px:
                continue
            angle = np.arctan2(observed[1], observed[0])-np.arctan2(local[1], local[0])
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            origin = (ca["center"]+cb["center"]-
                      (BAR_CENTERS[a]+BAR_CENTERS[b])@rotation.T*scale)/2
            predicted = BAR_CENTERS@rotation.T*scale+origin
            matches, used, errors, observed_centers, observed_bars = [], set(), [], [], []
            for i, xy in enumerate(predicted):
                eligible = [(np.linalg.norm(item["center"]-xy), j)
                            for j, item in enumerate(candidates)
                            if j not in used and item["group"] == COLOR_GROUPS[i]
                            and abs(item["length"]-BAR_LENGTHS[i]) < .025
                            and abs(np.dot(item["axis"], rotation[:, 1])) > .97]
                if not eligible:
                    continue
                error, j = min(eligible)
                if error <= max_error_px:
                    matches.append(i)
                    used.add(j)
                    errors.append(error)
                    observed_centers.append(candidates[j]['center'])
                    observed_bars.append(candidates[j])
            if len(matches) < 3:
                continue
            # Refit ALL matched bar centers, not the seed pair. The seed merely
            # supplies correspondence; reflection/scale changes are forbidden.
            local=BAR_CENTERS[matches]*scale
            observed=np.asarray(observed_centers)
            u,_,vt=np.linalg.svd((local-local.mean(0)).T@(observed-observed.mean(0)))
            correction=np.eye(2);correction[-1,-1]=np.linalg.det(vt.T@u.T)
            rotation=vt.T@correction@u.T
            origin=observed.mean(0)-local.mean(0)@rotation.T
            errors=np.linalg.norm(local@rotation.T+origin-observed,axis=1)
            if max(errors)>max_error_px:continue
            if any(abs(np.dot(candidates[j]['axis'],rotation[:,1]))<=.97 for j in used):
                continue
            support = _paint_support(hsv, rotation, origin, scale)
            painted = [i for i, value in enumerate(support) if value >= .65]
            if len(painted) < 6 or sum(i < 4 for i in painted) < 3:
                continue
            hypotheses.append(dict(origin_px=origin, rotation=rotation,
                                   endpoint_max_px=_bar_endpoint_error(matches,observed_bars,rotation,origin,scale),
                                   matched_keys=painted, contour_keys=matches,
                                   paint_support=support,
                                   rms_px=float(np.sqrt(np.mean(np.square(errors)))),
                                   key_pixels=KEY_HITS@rotation.T*scale+origin))
    if not hypotheses:
        return None
    hypotheses.sort(key=lambda h: (-len(h["matched_keys"]), -len(h["contour_keys"]), h["rms_px"]))
    best = hypotheses[0]
    for other in hypotheses[1:]:
        same_quality = (len(other["matched_keys"]) == len(best["matched_keys"])
                        and len(other["contour_keys"]) == len(best["contour_keys"])
                        and other["rms_px"] <= best["rms_px"]+.5)
        separated = np.max(np.linalg.norm(other["key_pixels"]-best["key_pixels"],axis=1)) > 3.
        if same_quality and separated:
            # A stretched/merged colored component can mimic a second pose
            # by center alone. Do not let its inconsistent endpoints veto a
            # fully endpoint-supported pose. Two supported poses stay unknown.
            if not _endpoint_excludes_rival(best,other,max_error_px):
                return None
    return best
