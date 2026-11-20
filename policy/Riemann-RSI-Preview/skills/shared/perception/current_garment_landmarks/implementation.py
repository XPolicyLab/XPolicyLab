from rsi.image_files import copy_rgb
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from skills.shared.perception.current_rgb_segmentation.implementation import prompted
from skills.shared.geometry.garment_camera_projection.implementation import points_on_plane, project_points as project, pixel_rays
from skills.shared.perception.public_support_geometry_context.implementation import run as retained_support
from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel



LANDMARKS = ('left_cuff', 'right_cuff', 'left_hem', 'right_hem', 'left_chest', 'right_chest',
             'left_shoulder', 'right_shoulder')
# The retained cuff names carry the public sleeve landmark; the shoulder pair
# carries the public shoulder landmark the packaged fold_body() landed on.
SHAPE_MANIFEST = 'assets/perception/garments/metadata.json'
SHAPE_ARRAYS = 'assets/perception/garments/geometry.npz'


def public_outline(points, counts, indices):
    """Rasterize one public garment surface into its outline in garment metres."""
    resolution = .001
    lower = np.min(points[:, :2], axis=0) - .003
    upper = np.max(points[:, :2], axis=0) + .003
    dimensions = np.ceil((upper - lower) / resolution).astype(int) + 1
    raster = np.zeros((dimensions[1], dimensions[0]), dtype=np.uint8)
    pixels = np.rint((points[:, :2] - lower) / resolution).astype(np.int32)
    cursor = 0
    for count in counts:
        face = pixels[indices[cursor:cursor + count]]
        cv2.fillConvexPoly(raster, face, 1)
        cursor += count
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 1., True).reshape(-1, 2)
    return contour.astype(float) * resolution + lower


SHAPES = None


def public_shapes():
    """The shipped public garment outlines and their public local landmarks."""
    global SHAPES
    if SHAPES is None:
        root = Path(__file__).resolve().parents[4]
        manifest = json.loads((root / SHAPE_MANIFEST).read_text())
        arrays = np.load(root / SHAPE_ARRAYS)
        SHAPES = {key: {'outline': public_outline(arrays[key + '_points'], arrays[key + '_counts'],
                                                  arrays[key + '_indices']),
                        'landmarks': manifest[key]['landmarks_public_local_m']}
                  for key in manifest}
    return SHAPES


def template_mask(api, key, parameters, height):
    angle = float(parameters[2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = public_shapes()[key]['outline'] @ rotation.T + np.asarray(parameters[:2], dtype=float)
    pixels = np.rint(project(api, 'cam_head',
                             np.column_stack([xy, np.full(len(xy), height + .005)]))).astype(np.int32)
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    return mask


def _observed_center(api, observed, height):
    rows, columns = np.where(observed > 0)
    # The public plane tool accepts a bounded sample, so stride the observed
    # silhouette down to it instead of back-projecting every pixel.  The stride
    # has to round up: with 1200..1799 observed pixels the floor kept stride 1
    # and handed the tool 1200+ samples, which raised out of a plain scene view
    # and ended the episode with no further robot action.
    stride = max(1, -(-len(rows) // 900))
    xy = points_on_plane(api, 'cam_head',
                         np.column_stack([columns, rows])[::stride], height + .005)[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    if np.count_nonzero(valid) < 300:
        return None
    return xy[valid].mean(0)


def template_landmarks(api, observed, height):
    """Fit a public garment template and return its own public local landmarks.

    The packaged locate(phase="flat") never derived landmark positions from the
    silhouette: it fitted each public garment outline to the current frame with
    one translation and one rotation, then transformed that template's
    landmarks_public_local_m through the same transform.  Reading the landmarks
    off cross-section fractions instead put left_hem on the collar (r81) and
    left_chest on the sleeve tip (r82), so the body fold grasped the collar and
    landed on the armpit line, and the sleeve fold stopped short of the
    opposite chest.
    """
    shapes = public_shapes()
    target = int(np.count_nonzero(observed))
    if target < 1200:
        return None
    center = _observed_center(api, observed, height)
    if center is None:
        return None
    best = None
    for key, shape in shapes.items():
        outline = shape['outline']
        midpoint = (outline.min(0) + outline.max(0)) / 2.

        def objective(parameters, key=key):
            prediction = template_mask(api, key, parameters, height)
            intersection = np.count_nonzero(prediction & observed)
            union = np.count_nonzero(prediction) + target - intersection
            return 1. - intersection / max(1, union)

        starts = []
        for angle in np.radians(np.arange(-40, 41, 10)):
            start = np.r_[center - midpoint, angle]
            prediction = template_mask(api, key, start, height)
            rows, columns = np.where(prediction > 0)
            if len(rows):
                stride = max(1, len(rows) // 900)
                predicted = points_on_plane(api, 'cam_head',
                                            np.column_stack([columns, rows])[::stride],
                                            height + .005)[:, :2]
                valid = np.isfinite(predicted).all(axis=1)
                if np.count_nonzero(valid):
                    start[:2] += center - predicted[valid].mean(0)
            starts.append((objective(start), start))
        for loss, start in sorted(starts, key=lambda item: item[0])[:2]:
            simplex = np.tile(start, (4, 1))
            simplex[1, 0] += .008
            simplex[2, 1] += .008
            simplex[3, 2] += .06
            fit = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 140, 'xatol': .0003, 'fatol': .0001,
                                    'initial_simplex': simplex})
            if best is None or float(fit.fun) < best[0]:
                best = (float(fit.fun), key, np.asarray(fit.x, dtype=float))
    if best is None or 1. - best[0] < .62:
        return None
    loss, key, parameters = best
    angle = float(parameters[2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = {name: (rotation @ np.asarray(point[:2], dtype=float) + parameters[:2]).tolist()
              for name, point in shapes[key]['landmarks'].items()}
    return {'template': key, 'parameters': parameters, 'rotation': rotation,
            'iou': float(1. - loss), 'landmarks': points}


def stable_candidates(candidates, previous):
    """Apply the old stable-current-mask contract to current public RGB."""
    if previous is None:
        return candidates
    old_axis = np.asarray(previous.get('longitudinal_axis'), dtype=float)
    old_outline = previous.get('head_outline')
    if old_axis.shape != (2,) or old_outline is None:
        return candidates
    kept = []
    for candidate in candidates:
        axis = np.asarray(candidate['longitudinal_axis'], dtype=float)
        if abs(float(np.dot(axis, old_axis))) < .80:
            continue
        # A fresh outline fit carries landmarks only; read() attaches the
        # current outline afterwards.  Indexing the field directly crashed the
        # whole episode (r74, KeyError: 'head_outline') the first time a lift
        # failed on part 0 and forced a fresh view.
        current_outline = candidate.get('head_outline')
        if current_outline is not None and outline_iou(current_outline, old_outline) < .20:
            continue
        kept.append(candidate)
    return kept


def current_mask(image, outline):
    mask = np.zeros((480, 640), dtype=np.uint8)
    contour = np.asarray(outline, dtype=float)
    if image.shape != (480, 640, 3) or contour.shape != (96, 2) or not np.isfinite(contour).all():
        raise ValueError('A garment mask belongs only to the current native RGB frame')
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255)
    return mask



def largest(mask, minimum):
    """The packaged largest() component selector, with its own area gate."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8), 8)
    if count <= 1:
        raise ValueError('no current public component')
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < minimum:
        raise ValueError('no current public component above the minimum area')
    return (labels == index).astype(np.uint8)


def segment(image, seed=None):
    """The packaged current-RGB garment segmentation, seeded or unseeded.

    Ported from the packaged perception.segment().  It learns the cloth's own
    colour from a current core -- a fixed central torso window when unseeded,
    the eroded seed interior otherwise -- and competes that colour against the
    current tabletop, so material randomisation needs no stored colour lookup.
    GrabCut only runs when the colour test cannot produce a large enough
    component.  The packaged rule used this for the torso stage, where the
    prompted region service described the whole workspace instead of the cloth.
    """
    image = np.asarray(image)
    if image.shape != (480, 640, 3):
        raise ValueError('Garment perception requires a current 640x480 RGB frame')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    if seed is None:
        core_rgb = image[240:310, 300:350].reshape(-1, 3)
        core_hsv = hsv[240:310, 300:350].reshape(-1, 3)
        core_lab = lab[240:310, 300:350].reshape(-1, 3)
    else:
        core_mask = cv2.erode(np.asarray(seed, np.uint8), np.ones((17, 17), np.uint8)) > 0
        core_rgb, core_hsv, core_lab = image[core_mask], hsv[core_mask], lab[core_mask]
    if len(core_rgb) < 100:
        raise ValueError('Insufficient current torso appearance evidence')
    # Dense fabric patterns keep their texture where their mean colour
    # resembles the table, so a bounded textured component is used first.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    texture = cv2.blur((np.abs(cv2.Laplacian(gray, cv2.CV_32F)) > 10).astype(float), (9, 9))
    core_texture = texture[240:310, 300:350] if seed is None else texture[core_mask]
    texture_level = float(np.median(core_texture))
    if texture_level > .6:
        patterned = (texture > texture_level * .6).astype(np.uint8)
        patterned[:175] = 0
        patterned[425:] = 0
        patterned[:, :125] = 0
        patterned[:, 515:] = 0
        patterned = cv2.morphologyEx(patterned, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        try:
            patterned = largest(patterned, 2500)
            contours = cv2.findContours(patterned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(patterned, contours, -1, 1, -1)
            if patterned.sum() < 45000 and patterned[240:310, 300:350].mean() > .7:
                return patterned
        except ValueError:
            pass
    centre_hsv = np.median(core_hsv, axis=0)
    centre_lab = np.median(core_lab, axis=0)
    # Hue wraps at red, so a scalar median of hues near 0 and 179 invents a
    # blue foreground; derive the hue from the median current RGB instead.
    centre_rgb = np.median(core_rgb, axis=0).astype(np.uint8).reshape(1, 1, 3)
    anchor = float(cv2.cvtColor(centre_rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])
    coloured = core_hsv[core_hsv[:, 1] > 35, 0]
    centre_hsv[0] = ((anchor + np.median((coloured - anchor + 90) % 180 - 90)) % 180
                     if len(coloured) > 100 else anchor)
    hue_distance = np.abs((hsv[:, :, 0] - centre_hsv[0] + 90) % 180 - 90)
    chroma = np.linalg.norm(lab[:, :, 1:] - centre_lab[1:], axis=2)
    background = np.concatenate([lab[185:220, 180:250].reshape(-1, 3),
                                 lab[185:220, 390:460].reshape(-1, 3)])
    table_lab = np.median(background, axis=0)
    color_separation = float(np.linalg.norm(centre_lab[1:] - table_lab[1:]))
    if centre_hsv[1] > 55 or (centre_hsv[1] > 40 and color_separation > 10):
        table_rgb = cv2.cvtColor(np.rint(table_lab).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)
        table_hsv = cv2.cvtColor(table_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        separation = abs((centre_hsv[0] - float(table_hsv[0]) + 90) % 180 - 90)
        hue_tolerance = 12 if centre_hsv[1] < 70 else 7
        if table_hsv[1] > 40 and separation < 15:
            hue_tolerance = min(hue_tolerance, max(3., separation * .55))
        appearance = ((hue_distance < hue_tolerance)
                      & (hsv[:, :, 1] > max(12, centre_hsv[1] - 60))
                      & (hsv[:, :, 2] > max(35, centre_hsv[2] - 65)) & (chroma < 30))
    elif centre_hsv[2] > 165:
        appearance = ((hsv[:, :, 1] < min(90, max(45, centre_hsv[1] + 25)))
                      & (hsv[:, :, 2] > centre_hsv[2] - 55) & (chroma < 23))
    else:
        appearance = (hsv[:, :, 2] < centre_hsv[2] + 45) & (chroma < 22)
    # Peach cloth and the wood can share a hue and a chroma, so the competing
    # table brightness is learned from this current frame as well.
    if np.linalg.norm((centre_lab - table_lab) * [.8, 1., 1.]) > 12:
        foreground_distance = np.linalg.norm((lab - centre_lab) * [.8, 1., 1.], axis=2)
        background_distance = np.linalg.norm((lab - table_lab) * [.8, 1., 1.], axis=2)
        appearance &= foreground_distance < background_distance
    region = np.zeros((480, 640), np.uint8)
    region[175:425, 125:515] = 1
    appearance = appearance.astype(np.uint8) & region
    kernel = 11 if centre_hsv[1] > 40 and color_separation > 10 else 5
    appearance = cv2.morphologyEx(appearance, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    try:
        component = largest(appearance, 2500)
        if component.sum() < 45000:
            contours = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(component, contours, -1, 1, -1)
            return component
    except ValueError:
        pass
    labels = np.full((480, 640), cv2.GC_BGD, np.uint8)
    labels[180:430, 110:530] = cv2.GC_PR_BGD
    if seed is None:
        labels[220:402, 170:470] = cv2.GC_PR_FGD
        labels[255:350, 280:360] = cv2.GC_FGD
    else:
        grown = cv2.dilate(np.asarray(seed, np.uint8), np.ones((25, 25), np.uint8))
        labels[grown > 0] = cv2.GC_PR_FGD
        core = cv2.erode(np.asarray(seed, np.uint8), np.ones((13, 13), np.uint8))
        labels[core > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    # GrabCut initialises its mixture models from OpenCV's RNG; reset it so an
    # unrelated earlier perception call cannot change this same observation.
    cv2.setRNGSeed(0)
    cv2.grabCut(copy_rgb(image), labels, None,
                background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = largest(mask, 1300)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(mask, contours, -1, 1, -1)
    return mask


def seeded_masks(image):
    """The packaged retry_flat_appearance() seed windows, deduplicated.

    The packaged locate() re-estimated the cloth from six bounded current-RGB
    seed windows whenever the unseeded colour test could not explain it, and
    kept the candidate whose own fit explained the most of its own mask.
    """
    seen = set()
    for cx, cy in ((320, 255), (320, 275), (320, 295), (340, 295), (300, 315), (320, 315)):
        seed = np.zeros((480, 640), np.uint8)
        seed[cy - 20:cy + 20, cx - 20:cx + 20] = 1
        try:
            mask = segment(image, seed)
        except ValueError:
            continue
        fingerprint = mask.tobytes()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield mask


def outline_supported(api, previous, outline, image):
    """Judge a current silhouette against the retained numeric estimate.

    A previously seen landmark is only a same-episode hypothesis, so a new
    outline replaces the numeric garment estimate only while it still holds
    every retained landmark.  Otherwise the last qualified silhouette stands:
    a reading that cannot explain the cloth we already measured is the table
    or an arm, not a new garment.
    """
    if previous is None:
        return True
    try:
        mask = current_mask(image, outline)
    except ValueError:
        return False
    support = cv2.dilate(mask, np.ones((17, 17), np.uint8)) > 0
    height = previous.get('height_m')
    if not isinstance(height, (int, float)) or not np.isfinite(height):
        return True
    for name in LANDMARKS:
        point = np.asarray(previous.get(name), dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            continue
        pixel = project(api, 'cam_head', np.array([[*point, float(height)]]))[0]
        if not np.isfinite(pixel).all():
            return False
        column, row = np.rint(pixel).astype(int)
        if not (0 <= column < 640 and 0 <= row < 480) or not support[row, column]:
            return False
    return True


def plane_points(api, camera, pixels, height):
    """Back-project any number of current pixels onto the measured support."""
    pixels = np.asarray(pixels, dtype=float)
    origin, rays = pixel_rays(api, camera, pixels)
    valid = np.abs(rays[:, 2]) > 1e-8
    distance = np.full(len(pixels), np.nan)
    distance[valid] = (height - origin[2]) / rays[valid, 2]
    valid &= distance > 0
    points = np.full((len(pixels), 3), np.nan)
    points[valid] = origin + rays[valid] * distance[valid, None]
    return points


def _component_outline(component):
    """Sample one current RGB component into the retained 96-point outline."""
    contours = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(contour) < 8:
        return None
    closed = np.vstack([contour, contour[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-6:
        return None
    targets = np.linspace(0., perimeter, 96, endpoint=False)
    cumulative = np.r_[0., np.cumsum(lengths)]
    outline = []
    for target in targets:
        index = min(int(np.searchsorted(cumulative, target, side='right') - 1), len(contour) - 1)
        fraction = (target - cumulative[index]) / max(lengths[index], 1e-6)
        outline.append((contour[index] * (1. - fraction)
                        + contour[(index + 1) % len(contour)] * fraction).tolist())
    return np.asarray(outline, dtype=float)


def numeric_outline(previous):
    """The retained numeric silhouette as a mask, or None."""
    numeric = np.asarray(previous.get('head_outline'), dtype=float)
    if numeric.shape != (96, 2) or not np.isfinite(numeric).all():
        return None
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [np.rint(numeric).astype(np.int32)], 1)
    return mask


def retained_appearance_mask(image, previous):
    """Segment the cloth from its own retained colour inside the retained footprint.

    The packaged locate() closed an incoherent torso segmentation with
    stable_current_body_mask(), a consensus over the current wrist images.  This
    module has no wrist-view consensus, so the same role is played by the one
    colour the episode has already measured on this cloth: flat cloth is the
    least saturated region here (S 33 against the table's 80), and the retained
    appearance keeps describing the garment after the fixed core window has
    fallen on bare table.
    """
    numeric = np.asarray(previous.get('head_outline'), dtype=float)
    appearance = np.asarray(previous.get('appearance_hsv'), dtype=float)
    if (numeric.shape != (96, 2) or not np.isfinite(numeric).all()
            or appearance.shape != (3,) or not np.isfinite(appearance).all()):
        return None
    reference = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(reference, [np.rint(numeric).astype(np.int32)], 255)
    outline_area = int(np.count_nonzero(reference))
    if outline_area < 400:
        return None
    # A sleeve folded across the body drags the whole garment a few centimetres,
    # so search the seed's immediate neighbourhood, not its exact footprint.
    region = cv2.dilate(reference, np.ones((41, 41), np.uint8))
    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV).astype(float)
    saturation = float(np.clip(appearance[1], 0., 255.)) + 25.
    value = float(np.clip(appearance[2], 0., 255.)) - 45.
    mask = ((hsv[..., 1] <= saturation) & (hsv[..., 2] >= value) & (region > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    try:
        component = largest(mask, 2000)
    except ValueError:
        return None
    # Folding both sleeves inboard removes roughly half the flat silhouette; a
    # region far outside that band is the table or an arm, not this garment.
    area = int(np.count_nonzero(component))
    if not max(1200., .20 * outline_area) <= area <= 1.35 * outline_area:
        return None
    # The pads occlude the cloth they hold and the folded stack is thicker than
    # one layer, so grow the measured component before sampling its outline.
    return cv2.dilate(component, np.ones((9, 9), np.uint8))


def torso_observed(image, previous):
    """The packaged locate(phase="sleeves_folded") current silhouette.

    The packaged rule segmented the torso frame once from the unseeded colour
    test and then -- whenever that mask was much larger than the numeric
    garment neighbourhood and mostly lay outside it -- again from the numeric
    outline itself, accepting the seeded retry only inside the packaged area
    and support band.  r81 ran neither: the prompted region described the whole
    workspace (91100 px), the flat silhouette was retained, and the hem read
    that followed put the right hem eleven centimetres off the cloth, so the
    right pad closed on bare table and only the left half was ever folded.
    """
    if previous is None:
        return None
    outline = numeric_outline(previous)
    if outline is None:
        return None
    outline_area = int(np.count_nonzero(outline))
    if outline_area < 400:
        return None
    support = cv2.dilate(outline, np.ones((17, 17), np.uint8)) > 0
    try:
        observed = segment(image)
    except ValueError:
        observed = None
    if observed is None:
        return None
    area = int(np.count_nonzero(observed))
    supported = float(observed[support].sum() / max(1, area))
    # A silhouette that is much larger than the numeric garment neighbourhood,
    # or that mostly lies outside it, is not the cloth already measured.  The
    # packaged rule re-read the appearance from the numeric outline first and
    # then fell back to its current-body consensus.
    if area > 1.5 * outline_area or supported < .70:
        accepted = False
        try:
            candidate = segment(image, outline)
        except ValueError:
            candidate = None
        if candidate is not None:
            candidate_area = int(np.count_nonzero(candidate))
            candidate_supported = float(candidate[support].sum() / max(1, candidate_area))
            if (max(1300., .35 * outline_area) <= candidate_area <= 1.25 * outline_area
                    and candidate_supported >= .85):
                observed = candidate
                accepted = True
        if not accepted:
            observed = retained_appearance_mask(image, previous)
        if observed is None:
            # The packaged rule raised here and the caller kept the last
            # qualified estimate; an incoherent silhouette must not move the
            # hem onto bare table.
            return None
    if int(np.count_nonzero(observed)) < 300:
        return None
    return observed


def _retained_torso(previous, plane, support, empty, reason):
    """Keep the last qualified estimate through an incoherent reading."""
    anchor = deepcopy(previous)
    anchor.update(support_memory=support['memory'], height_m=float(plane['distance_m']),
                  support_uncertainty_m=float(plane['height_uncertainty_m']))
    return dict(empty, anchor=anchor, alternatives=[anchor], status='observed',
                torso_read_fallback=reason)


def torso_read(api, previous, plane, support, image, empty, retry=True):
    """Replace only the two hem landmarks from the current silhouette.

    The packaged locate(phase="sleeves_folded") kept the whole flat estimate and
    re-read nothing but the two hem corners, because its template fit had
    already followed the cloth: the shoulders, chests and cuffs stayed where the
    flat reading put them, and only the hem edge moved when the sleeves were
    folded inboard.  Re-deriving a grasp end and a landing end from the current
    silhouette instead put the grasp on the collar and the landing on the
    sleeve tip (r81, r82), so the torso fold never folded the garment in half.
    """
    if previous is None:
        return empty
    retained = _retained_torso(previous, plane, support, empty, 'no_current_torso_reading')
    lateral = np.asarray(previous.get('lateral_axis'), dtype=float)
    longitudinal = np.asarray(previous.get('longitudinal_axis'), dtype=float)
    origin = np.asarray(previous.get('center'), dtype=float)
    if (lateral.shape != (2,) or longitudinal.shape != (2,) or origin.shape != (2,)
            or not (np.isfinite(lateral).all() and np.isfinite(longitudinal).all()
                    and np.isfinite(origin).all())):
        return retained
    lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-9)
    longitudinal = longitudinal / max(float(np.linalg.norm(longitudinal)), 1e-9)
    # The retained axes are the fitted template rotation, so they carry the
    # garment's own frame across the sleeve stage without a second fit.
    rotation = np.column_stack([lateral, longitudinal])
    height = float(plane['distance_m'])
    observed = torso_observed(image, previous)
    if observed is None:
        return retained
    mask = observed
    rows, columns = np.where(mask > 0)
    if len(rows) < 300:
        return retained
    xy = plane_points(api, 'cam_head', np.column_stack([columns, rows]), height + .005)[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    if np.count_nonzero(valid) < 300:
        return retained
    xy = xy[valid]
    local = (xy - origin) @ rotation
    # A sleeve that stayed outside the torso must not turn into an imaginary
    # wide hem: restrict the edge search to the retained public hem
    # neighbourhood, with the packaged 35 mm deformation margin.
    public_hem = np.asarray([np.asarray(previous[side + '_hem'], dtype=float)
                             for side in ('left', 'right')])
    if public_hem.shape != (2, 2) or not np.isfinite(public_hem).all():
        return retained
    hem_local = (public_hem - origin) @ rotation
    lower, upper = float(hem_local[:, 0].min()) - .035, float(hem_local[:, 0].max()) + .035
    hem_seed = float(np.mean(hem_local[:, 1]))
    hem_region = local[(local[:, 0] > lower) & (local[:, 0] < upper)
                       & (local[:, 1] < hem_seed + .065) & (local[:, 1] > hem_seed - .065)]
    if len(hem_region) < 100:
        if retry:
            # The packaged rule re-segmented once from the numeric outline seed
            # before it gave up on the current frame.
            seed = numeric_outline(previous)
            if seed is not None:
                try:
                    resegmented = segment(image, seed)
                except ValueError:
                    resegmented = None
                if resegmented is not None and int(np.count_nonzero(resegmented)) >= 300:
                    rows, columns = np.where(resegmented > 0)
                    xy = plane_points(api, 'cam_head', np.column_stack([columns, rows]),
                                      height + .005)[:, :2]
                    valid = np.isfinite(xy).all(axis=1)
                    if np.count_nonzero(valid) >= 300:
                        local = (xy[valid] - origin) @ rotation
                        hem_region = local[(local[:, 0] > lower) & (local[:, 0] < upper)
                                           & (local[:, 1] < hem_seed + .065)
                                           & (local[:, 1] > hem_seed - .065)]
                        if len(hem_region) >= 100:
                            mask = resegmented
        if len(hem_region) < 100:
            return retained
    left_x, right_x = (float(value) for value in np.quantile(hem_region[:, 0], [.04, .96]))
    points = {}
    for side, x in (('left', left_x + .018), ('right', right_x - .018)):
        corner = hem_region[np.abs(hem_region[:, 0] - x) < .020]
        if len(corner) < 20:
            return retained
        corner_y = float(np.quantile(corner[:, 1], .03))
        points[side + '_hem'] = (rotation @ np.array([x, corner_y + .028]) + origin).tolist()
    # supported_contact() reads the current mask, so the retained outline must
    # be this frame's silhouette rather than the flat one.
    outline = _component_outline(mask)
    if outline is None:
        return retained
    samples = image[mask > 0]
    if len(samples) < 200:
        return retained
    anchor = dict(previous, **points)
    anchor.update(source='current_public_rgb_garment_outline', support_memory=support['memory'],
                  height_m=height, support_uncertainty_m=float(plane['height_uncertainty_m']),
                  head_outline=outline.tolist(),
                  appearance_hsv=np.asarray(previous['appearance_hsv'], dtype=float).tolist())
    return dict(empty, anchor=anchor, alternatives=[anchor], status='observed',
                current_torso_bounds_local_m=[left_x, right_x, hem_seed])


def read(api, previous=None, reuse=False, phase='flat'):
    empty = {'anchor': None, 'support_memory': previous['support_memory'] if previous else None,
             'status': 'not_observed', 'rgb_geometry_is_hidden_surface': False}
    support = retained_support(api, empty['support_memory'], passive_view=True)
    empty['support_memory'] = support.get('memory')
    if support['support']['status'] != 'observed':
        return empty
    plane = support['support']['plane']
    image = api.image('cam_head')
    if phase != 'flat':
        return torso_read(api, previous, plane, support, image, empty)
    height = float(plane['distance_m'])
    regions = prompted(api, ['shirt', 'clothing', 'garment'])
    heads = [row for row in regions['cam_head'] if not row['clipped']]
    outline = None
    if len(heads) == 1:
        candidate = np.asarray(heads[0]['outline'], dtype=float)
        if outline_supported(api, previous, candidate, image):
            outline = candidate
    # The packaged rule reuses the first flat estimate for the second sleeve:
    # fold_sleeve() refines only the contact point against the current head
    # image and never re-fits a flat-shirt outline once a sleeve is folded.
    # A fresh fit on a half-folded garment is what previously produced an empty
    # anchor and a zero-action abort, so reuse keeps the retained axes and
    # snaps every landmark onto the current public outline instead.
    if reuse and previous is not None:
        if outline is None:
            outline = np.asarray(previous['head_outline'], dtype=float)
        landmarks = deepcopy({name: previous[name] for name in
                              ('center', 'longitudinal_axis', 'lateral_axis', 'confidence', *LANDMARKS)})
        projected = points_on_plane(api, 'cam_head', outline, height)[:, :2]
        for name in ('left_cuff', 'right_cuff', 'left_hem', 'right_hem',
                     'left_chest', 'right_chest'):
            point = np.asarray(landmarks[name])
            distances = np.linalg.norm(projected - point, axis=1)
            if distances.min() < .055:
                landmarks[name] = projected[np.argmin(distances)].tolist()
        # supported_contact() reads head_outline, so a reused anchor must carry
        # the current frame's public outline.  The shoulders are not snapped:
        # they are the body fold's landing point and come from the fitted
        # public template, not from the current silhouette edge.
        landmarks['head_outline'] = outline.tolist()
        landmark_candidates = [landmarks]
    else:
        # The packaged flat reading fitted a public garment outline and
        # transformed that template's own public landmarks.
        fit = None
        if outline is not None:
            try:
                fit = template_landmarks(api, current_mask(image, outline), height)
            except ValueError:
                # A retained outline that cannot be rasterized into this frame
                # is a missing reading, exactly like the unusable segmentation
                # below it, not a reason to abandon the episode with no further
                # robot action.
                fit = None
        if fit is None:
            # The packaged retry_flat_appearance(): six bounded current-RGB
            # seed windows were segmented independently and the fit that
            # explained the most of its own mask was kept.
            best = None
            candidates = []
            try:
                candidates.append(segment(image))
            except ValueError:
                pass
            candidates.extend(seeded_masks(image))
            for mask in candidates:
                candidate = template_landmarks(api, mask, height)
                if candidate is None:
                    continue
                if best is None or candidate['iou'] > best[0]['iou']:
                    best = (candidate, mask)
            if best is not None:
                fit = best[0]
                retry_outline = _component_outline(best[1])
                if retry_outline is not None:
                    outline = retry_outline
        if fit is None:
            if previous is not None:
                landmarks = deepcopy(previous)
                if outline is not None:
                    landmarks['head_outline'] = outline.tolist()
                landmark_candidates = [landmarks]
            else:
                return empty
        else:
            if outline is None:
                outline = np.asarray(previous['head_outline'], dtype=float) if previous else None
                if outline is None:
                    return empty
            marks = fit['landmarks']
            rotation = np.asarray(fit['rotation'], dtype=float)
            landmark_candidates = [{
                'center': np.asarray(fit['parameters'][:2], dtype=float).tolist(),
                'longitudinal_axis': (rotation @ np.array([0., 1.])).tolist(),
                'lateral_axis': (rotation @ np.array([1., 0.])).tolist(),
                'confidence': float(min(1., max(0., fit['iou']))),
                'left_cuff': marks['left_sleeve'], 'right_cuff': marks['right_sleeve'],
                'left_hem': marks['left_hem'], 'right_hem': marks['right_hem'],
                'left_chest': marks['left_chest'], 'right_chest': marks['right_chest'],
                'left_shoulder': marks['left_shoulder'], 'right_shoulder': marks['right_shoulder'],
                'head_outline': outline.tolist()}]
    if not landmark_candidates:
        return empty
    mask = current_mask(image, outline)
    samples = image[mask > 0]
    if len(samples) < 200:
        return empty
    appearance = np.median(cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3), axis=0)
    if previous is not None:
        # The cloth keeps one colour for the whole episode.  Re-measuring it
        # from a silhouette that already includes table let r81 drift the
        # retained appearance from S=33 to S=76, the table's own saturation,
        # and every later colour test then described the table instead of the
        # garment.
        appearance = np.asarray(previous['appearance_hsv'], dtype=float)
    anchors = [dict(landmarks, source='current_public_rgb_garment_outline',
                    support_memory=support['memory'], height_m=float(plane['distance_m']),
                    support_uncertainty_m=float(plane['height_uncertainty_m']),
                    head_outline=outline.tolist(), appearance_hsv=appearance.tolist())
               for landmarks in landmark_candidates]
    return dict(empty, anchor=anchors[0], alternatives=anchors, status='observed')


def supported_contact(api, anchor, name):
    if name not in LANDMARKS or anchor['source'] != 'current_public_rgb_garment_outline':
        raise ValueError('A garment contact names an actual current outline landmark')
    # Original v45 head cuff contour refinement uses M + 8 mm.  This
    # is a garment reference convention, not a measured support height.
    height = anchor['height_m'] + (.008 if name.endswith('_cuff') else 0.)
    point = np.r_[anchor[name], height]
    pixel = np.asarray(project(api, 'cam_head', [point])[0], dtype=float)
    mask = current_mask(api.image('cam_head'), anchor['head_outline'])
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    coordinate = np.rint(pixel).astype(int)
    if not 3 <= coordinate[0] < 637 or not 3 <= coordinate[1] < 477:
        return None
    candidates = []
    for horizontal in range(max(2, coordinate[0] - 18), min(638, coordinate[0] + 19), 2):
        for vertical in range(max(2, coordinate[1] - 18), min(478, coordinate[1] + 19), 2):
            threshold = 1 if name.endswith('_hem') else 2
            if distance[vertical, horizontal] >= threshold:
                candidates.append((np.linalg.norm([horizontal - pixel[0], vertical - pixel[1]]), horizontal, vertical))
    if not candidates:
        return None
    unused_distance, horizontal, vertical = min(candidates)
    observed = points_on_plane(api, 'cam_head', [[horizontal, vertical]], height)[0]
    return observed


def interior_contact(api, anchor, point, direction, reach=.04, step=.002):
    """Slide a cuff contact to the deepest current-mask interior along the sleeve.

    The packaged rule pinches "across the narrow sleeve, inside its current RGB
    edge" because the outermost silhouette vertex alone leaves one finger
    outside flat cloth.  The retained anchor landmark is exactly such a vertex,
    so after the original 2.2 cm inward step the contact is moved along the
    sleeve to the point of largest current-mask clearance inside a bounded
    probe.  Only the current public head image and outline enter the choice.
    """
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(direction))
    if length < 1e-8 or point.shape not in ((2,), (3,)):
        return point
    direction = direction / length
    try:
        mask = current_mask(api.image('cam_head'), anchor['head_outline'])
        clearance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        offset = np.zeros_like(point)
        offset[:2] = direction
        steps = max(1, int(round(reach / step)))
        candidates = [point + offset * (step * index) for index in range(steps + 1)]
        pixels = np.asarray(project(api, 'cam_head', candidates), dtype=float)
    except (ValueError, KeyError, TypeError, IndexError, cv2.error):
        return point
    best, selected = -1., point
    for candidate, pixel in zip(candidates, pixels):
        if not np.isfinite(pixel).all():
            continue
        horizontal, vertical = int(round(pixel[0])), int(round(pixel[1]))
        if not 0 <= horizontal < 640 or not 0 <= vertical < 480:
            continue
        value = float(clearance[vertical, horizontal])
        if value > best + 1e-9:
            best, selected = value, candidate
    return selected


def wrist_contact(api, anchor, side, nominal, interior_axis=None):
    """Backproject the current wrist RGB onto the measured support plane.

    Ported from the packaged vision.retry_contact(): the head silhouette can
    put a nominal hem pad on a curled edge, and the wrist camera looking down
    at the cloth is the only current view that resolves the surface under that
    pad.  The packaged call also rasterized its own finger hulls into this
    image to drop self-occlusion; this bundle carries no own-tool mesh, and the
    appearance mask already rejects the dark pads, so the annulus and the thin
    interior band are what bound the sample instead.
    """
    nominal = np.asarray(nominal, dtype=float)
    if side not in ('left', 'right') or nominal.shape != (3,) or not np.isfinite(nominal).all():
        raise ValueError('A wrist contact names a declared own arm and a finite nominal point')
    camera = 'cam_' + side + '_wrist'
    height = float(anchor['height_m']) + .003
    hsv = cv2.cvtColor(api.image(camera), cv2.COLOR_RGB2HSV).astype(float)
    hint = np.asarray(anchor['appearance_hsv'], dtype=float)
    hue = np.abs((hsv[:, :, 0] - hint[0] + 90) % 180 - 90)
    if hint[1] > 55:
        mask = (hue < 9) & (hsv[:, :, 1] > max(35., hint[1] - 70.)) & (hsv[:, :, 2] > 65.)
    else:
        mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 115.)
    row, column = np.where(mask)
    if len(row) < 20:
        raise ValueError('No current wrist RGB appearance support for a bounded regrasp')
    world = plane_points(api, camera, np.column_stack([column, row]), height)
    delta = world[:, :2] - nominal[:2]
    distance = np.linalg.norm(delta, axis=1)
    valid = np.isfinite(world).all(axis=1) & (distance < .035) & (distance > .014)
    valid &= (row > 130) & (row < 420) & (column > 180) & (column < 460)
    if interior_axis is None:
        # The packaged radial strategy took its inward side from the garment
        # centre rather than from the fitted axis.
        inward = np.asarray(anchor['center'], dtype=float)[:2] - nominal[:2]
        inward /= max(float(np.linalg.norm(inward)), 1e-8)
        valid &= delta @ inward > .008
    else:
        # A 4 mm band on the inward side of the pad: the packaged rule asked
        # for cloth beyond the nominal point along the garment's own axis, not
        # for any garment-coloured pixel in the annulus.
        inward = np.asarray(interior_axis, dtype=float)[:2]
        inward /= max(float(np.linalg.norm(inward)), 1e-8)
        valid &= (delta @ inward > .014) & (np.abs(delta @ np.array([inward[1], -inward[0]])) < .004)
    if np.count_nonzero(valid) < 20:
        raise ValueError('No current wrist RGB contact for a bounded regrasp')
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(distance[indices])[:60]]
    point = np.median(world[indices], axis=0)
    point[2] = height
    return point, {'source': 'current_wrist_RGB_plane_backprojection', 'camera': camera,
                   'samples': int(np.count_nonzero(valid)), 'contact': point.tolist()}


def lifted_surface_feedback(api, anchor, sides):
    """Verify lifted cloth with current RGB and public wrist pose projection.

    The head and wrist cameras see the same cloth under different illumination:
    the wrist view is often darker and more saturated because the fingers cast a
    shadow.  Use the old rule's one-sided HSV evidence (lower bounds rather than
    an exact colour distance), while projecting a small public TCP/pad patch from
    the measured end-effector pose.  No private finger mesh or template asset is
    used here.
    """
    result = {'status': 'uncertain', 'head_colour_fraction': 0.,
              'wrist_colour_fraction': 0., 'gripper_measurement_is_contact': False}
    if anchor is None:
        return result
    appearance = np.asarray(anchor['appearance_hsv'], dtype=float)
    if appearance.shape != (3,) or not np.isfinite(appearance).all():
        return result

    def match_fraction(image, pixel, radius):
        image = np.asarray(image)
        point = np.rint(pixel).astype(int)
        if not radius <= point[0] < 640 - radius or not radius <= point[1] < 480 - radius:
            return 0.
        patch = cv2.cvtColor(image[point[1] - radius:point[1] + radius + 1,
                                  point[0] - radius:point[0] + radius + 1],
                             cv2.COLOR_RGB2HSV).astype(float)
        hue = np.minimum(np.abs(patch[..., 0] - appearance[0]),
                         180. - np.abs(patch[..., 0] - appearance[0]))
        if appearance[1] > 30.:
            # Keep the garment through the wrist-camera shadow and renderer
            # colour shift.  The head view supplies the hue family; saturation
            # and value are only lower bounds, as in the old Rule verifier.
            matched = ((hue < 26.)
                       & (patch[..., 1] > max(12., appearance[1] - 90.))
                       & (patch[..., 2] > 25.))
        elif appearance[2] > 165.:
            matched = ((patch[..., 1] < 100.)
                       & (patch[..., 2] > 28.))
        else:
            matched = patch[..., 2] < appearance[2] + 45.
        return float(matched.mean())

    # The measured pad is the acknowledged TCP, root @ FK @ declared offset.
    # Stopping at the FK origin put the sample 14.5 cm behind the real pad and
    # behind the wrist camera, so every wrist projection returned no branch and
    # the review silently fell back to the head camera alone.
    model = GarmentOwnModel()
    roots = model.roots(api)
    head_fractions, head_tips = [], []
    wrist_fractions, lateral_fractions, robot_occlusions = [], [], []
    for side in sides:
        try:
            pad_pose = model.pad(np.asarray(api.own_joint_states()[side], dtype=float), roots[side])
            pad = pad_pose[:3, 3]
        except (KeyError, TypeError, ValueError, IndexError):
            head_fractions.append(0.)
            head_tips.append(0.)
            wrist_fractions.append(0.)
            lateral_fractions.append(0.)
            robot_occlusions.append(0.)
            continue
        if pad[2] - float(anchor['height_m']) < .025:
            return result
        head, tip, occlusion = 0., 0., 0.
        try:
            head_image = api.image('cam_head')
            head_pixel = np.asarray(project(api, 'cam_head', [pad])[0], dtype=float)
            head = match_fraction(head_image, head_pixel, 16)
            tip = match_fraction(head_image, head_pixel, 6)
            point = np.rint(head_pixel).astype(int)
            # The fixed X5 housing can cover the lifted pad in head RGB.  Record
            # that current-view occlusion instead of reading it as missing cloth.
            patch = cv2.cvtColor(head_image[max(0, point[1] - 16):point[1] + 17,
                                            max(0, point[0] - 16):point[0] + 17],
                                 cv2.COLOR_RGB2HSV).astype(float)
            occlusion = float(((patch[..., 1] < 35)
                               | ((patch[..., 2] < 60) & (patch[..., 1] < 100))).mean())
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        # The packaged rule measured the wrist view at radius 20 as its primary
        # evidence and the two fixed pad faces at radius 6 as the shadow-proof
        # fallback.  Keep that ordering: the head view is only the secondary
        # witness, because the wrist camera looks straight at the pinched cloth.
        wrist, lateral = 0., 0.
        try:
            wrist_image = api.image('cam_' + side + '_wrist')
            wrist = match_fraction(wrist_image, np.asarray(
                project(api, 'cam_' + side + '_wrist', [pad])[0], dtype=float), 20)
            lateral = max(match_fraction(wrist_image, np.asarray(project(
                api, 'cam_' + side + '_wrist',
                [(pad_pose @ np.asarray([0., offset, 0., 1.]))[:3]])[0], dtype=float), 6)
                for offset in (-.008, .008))
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        head_fractions.append(head)
        head_tips.append(tip)
        wrist_fractions.append(wrist)
        lateral_fractions.append(lateral)
        robot_occlusions.append(occlusion)

    if head_fractions and wrist_fractions:
        # The closed public review receipt carries exactly these two fractions;
        # the tip, lateral and occlusion terms are internal evidence for the
        # same packaged decision and must not widen that receipt.
        result.update(head_colour_fraction=min(head_fractions),
                      wrist_colour_fraction=min(wrist_fractions))
        if all((wrist > .18 and tip > .06) or (lateral > .5 and occlusion > .65)
               for wrist, tip, lateral, occlusion in zip(wrist_fractions, head_tips,
                                                         lateral_fractions, robot_occlusions)):
            result['status'] = 'current_surface_follows_own_pad'
    return result
