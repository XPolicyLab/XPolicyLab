"""Measure classification scenes from RGB and parse their public instruction."""
from itertools import combinations, permutations, product
from copy import deepcopy
import hashlib
import math
import re

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from coap.perception import toy_face_pattern
from coap.motion import quat_matrix


NORMAL_CATEGORIES = ('toy_car', 'action_camera', 'pen', 'watch', 'garage', 'phone')
LANGUAGE_CATEGORIES = ('watch', 'car', 'wooden_toy', 'pepper', 'chocolate_bar')
GRASP_HEIGHTS = {'toy_car': .021, 'car': .021, 'garage': .039,
                 'action_camera': .030, 'pen': .012, 'watch': .027,
                 'wooden_toy': .025, 'pepper': .025, 'chocolate_bar': .012, 'phone': .020}
SURFACE_HEIGHTS = {'toy_car': .033, 'car': .036, 'garage': .055,
                   'action_camera': .044, 'pen': .017, 'watch': .035,
                   'wooden_toy': .038, 'pepper': .045, 'chocolate_bar': .022, 'phone': .028}


def instruction_mapping(instruction):
    """Read the category-to-basket mapping without consulting scene metadata."""
    if isinstance(instruction, (list, tuple)):
        instruction = ' '.join(str(value) for value in instruction)
    text = str(instruction).lower()
    matches = re.findall(r'([a-z_ ]+?)\s+objects?\s+into\s+the\s+(left|middle|right)\s+basket', text)
    result = {}
    for name, side in matches:
        name = re.sub(r'^(?:put|and)\s+', '', name.strip()).strip().replace(' ', '_')
        name = {'cars': 'car', 'watches': 'watch', 'peppers': 'pepper',
                'wooden_toys': 'wooden_toy', 'chocolate_bars': 'chocolate_bar'}.get(name, name)
        result[name] = ('left', 'middle', 'right').index(side)
    return result


def _head(ctx):
    camera = next((name for name in ctx.obs['vision'] if 'head' in name), next(iter(ctx.obs['vision'])))
    image = np.asarray(ctx.obs['vision'][camera]['color'], np.uint8)[..., :3]
    return image, camera


def _project(ctx, pixels, z, camera, shape):
    return np.array([ctx.perception.project(point, z, camera, ctx.state, shape)
                     for point in np.asarray(pixels).reshape(-1, 2)])


def _geometry(ctx, contour, surface_z, camera, image):
    contour = np.asarray(contour, np.float32).reshape(-1, 2)
    moments = cv2.moments(contour)
    if moments['m00']:
        pixel = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
    else:
        pixel = np.mean(contour, axis=0)
    world = _project(ctx, contour, surface_z, camera, image.shape)
    rectangle = cv2.minAreaRect(world[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rectangle)
    edges = np.roll(corners, -1, axis=0)-corners
    axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    axis /= max(np.linalg.norm(axis), 1e-8)
    cross = np.array([-axis[1], axis[0]])
    length = np.ptp(np.einsum('ni,i->n', world[:, :2], axis))
    short = np.ptp(np.einsum('ni,i->n', world[:, :2], cross))
    position = _project(ctx, [pixel], surface_z, camera, image.shape)[0]
    return pixel, position, float(math.atan2(axis[1], axis[0]) % math.pi), (float(length), float(short)), world


def measure_baskets(ctx, image=None, camera=None):
    """Measure every pale basket, including the central basket omitted upstream."""
    if image is None:
        image, camera = _head(ctx)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red, green, blue = cv2.split(image.astype(float))
    layers = [(saturation < 35) & (value > 155),
              (hue > 90) & (hue < 135) & (saturation > 125) & (value > 75),
              ((hue < 5) | (hue > 175)) & (saturation > 150) & (value > 110)
              & (red > 2.5*green) & (red > 2.5*blue)]
    contours = []
    for selected in layers:
        mask = selected.astype(np.uint8)*255
        mask &= ctx.perception.workspace_mask(image.shape)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(outlines)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 900:
            continue
        pixel, position, yaw, size, world = _geometry(ctx, contour, ctx.table_z+.07, camera, image)
        if not (.13 < size[0] < .48 and .09 < size[1] < .38):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        # The upper portion is the visible opening rather than the near wall.
        pixel = np.array([x+width*.50, y+height*.35])
        position = _project(ctx, [pixel], ctx.table_z+.07, camera, image.shape)[0]
        position[2] = ctx.table_z+.035
        candidates.append(dict(position=position, pixel=tuple(pixel), bbox=(x, y, width, height),
                               camera=camera, contour=contour[:, 0, :], size=(*size, .07),
                               rim_z=ctx.table_z+.07, yaw=yaw, area=area, confidence=.75))
    candidates.sort(key=lambda item: -item['area'])
    baskets = []
    for item in candidates:
        if any(np.linalg.norm(item['position'][:2]-other['position'][:2]) < .11 for other in baskets):
            continue
        baskets.append(item)
        if len(baskets) == 3:
            break
    return sorted(baskets, key=lambda item: item['position'][0])


def _object_mask(ctx, image, baskets):
    """Remove the measured baskets before separating colored object material."""
    workspace = ctx.perception.workspace_mask(image.shape)
    for basket in baskets:
        contour = np.asarray(basket['contour'], np.int32).reshape(-1, 1, 2)
        basket_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(basket_mask, [contour], -1, 255, -1)
        basket_mask = cv2.dilate(basket_mask, np.ones((5, 5), np.uint8))
        workspace[basket_mask > 0] = 0
        x, y, width, height = basket['bbox']
        workspace[max(0, y-5):min(image.shape[0], y+height+8),
                  max(0, x-5):min(image.shape[1], x+width+5)] = 0
    return workspace


def _on_robot_material(ctx, pixel, image):
    """Project the two open fingers separately instead of masking their empty gap."""
    for arm in ('left', 'right'):
        pose = np.asarray(ctx.state[f'{arm}_ee_pose'], float)
        rotation = quat_matrix(pose[3:])
        grip = float(np.asarray(ctx.state.get(f'{arm}_ee_joint_state', [1.])).reshape(-1)[0])
        offsets = [([0., 0., 0.], 35.), ([.06, 0., .025], 29.)]
        offsets += [([.14, side*.044*grip, 0.], 11.) for side in (-1., 1.)]
        for offset, radius in offsets:
            point = pose[:3]+rotation@np.asarray(offset)
            projected = ctx.perception.world_to_pixel(point, image_shape=image.shape)
            if np.linalg.norm(np.asarray(pixel)-projected) < radius:
                return True
    return False


def _merge_regions(contours):
    """Combine touching color fragments while retaining separate nearby objects."""
    regions = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, width, height = cv2.boundingRect(contour)
        for index, existing in enumerate(regions):
            a, b, w, h = cv2.boundingRect(existing)
            inside = a-2 <= x and b-2 <= y and x+width <= a+w+2 and y+height <= b+h+2
            distance = float(np.min(np.linalg.norm(contour[:, None, 0, :].astype(float)
                                                     -existing[None, :, 0, :].astype(float), axis=2)))
            overlap = max(0, min(x+width, a+w)-max(x, a))*max(0, min(y+height, b+h)-max(y, b))
            if inside or distance < 3.2 or overlap > .35*min(width*height, w*h):
                regions[index] = cv2.convexHull(np.r_[existing, contour])
                break
        else:
            regions.append(contour)
    changed = True
    while changed:
        changed = False
        for index, contour in enumerate(regions):
            x, y, width, height = cv2.boundingRect(contour)
            for other_index in range(index+1, len(regions)):
                other = regions[other_index]
                a, b, w, h = cv2.boundingRect(other)
                overlap = max(0, min(x+width, a+w)-max(x, a))*max(0, min(y+height, b+h)-max(y, b))
                if overlap > .35*min(width*height, w*h):
                    regions[index] = cv2.convexHull(np.r_[contour, other])
                    del regions[other_index]
                    changed = True
                    break
            if changed:
                break
    return regions


def _contours(ctx, image, workspace, language):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r, g, b = cv2.split(image.astype(float))
    masks = [(h > 85) & (h < 155) & (s > 30) & (v > 45),
             (h >= 24) & (h < 42) & (s > 95) & (v > 100),
             (h >= 42) & (h <= 90) & (s > 45) & (v > 40),
             (s < 55) & (v > 175),
             ((h < 7) | (h > 170)) & (s > 100) & (v > 65) & (r > g*2.1) & (r > b*1.6),
             (v < 95) & (s < 130)]
    if language:
        masks.append((h > 9) & (h < 30) & (s > 45) & (s < 165) & (v > 190) & (g > .72*r))
    contours = []
    for selected in masks:
        mask = (selected & (workspace > 0)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in outlines:
            area = cv2.contourArea(contour)
            if 12 < area < 7500:
                contours.append(contour)
    return _merge_regions(contours)


def _features(image, contour, item):
    x, y, width, height = item['bbox']
    crop = image[y:y+height, x:x+width]
    local = np.asarray(contour, np.int32).reshape(-1, 1, 2)-[x, y]
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [local], -1, 1, -1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    selected = mask > 0
    h, s, v = hsv[selected].T
    pixels = crop[selected].astype(float)
    red, green, blue = pixels.T
    rect = cv2.minAreaRect(np.asarray(contour, np.float32))
    rectangle_area = max(np.prod(rect[1]), 1.)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 45, 110)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, thickness, _, solid_pixel = cv2.minMaxLoc(distance)
    local_h, local_s, local_v = cv2.split(hsv)
    wood = (local_h > 4) & (local_h < 27) & (local_s > 65) & (local_v > 65)
    metal = ((local_s < 80) | (local_v < 70)).astype(np.uint8)
    outlines, hierarchy = cv2.findContours(metal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ring_score = 0.
    if hierarchy is not None:
        for outline, node in zip(outlines, hierarchy[0]):
            if node[3] < 0:
                continue
            area = cv2.contourArea(outline)
            if not 6 < area < .65*width*height:
                continue
            hole = np.zeros((height, width), np.uint8)
            cv2.drawContours(hole, [outline], -1, 1, -1)
            inner = cv2.erode(hole, np.ones((2, 2), np.uint8)) > 0
            boundary = (cv2.dilate(hole, np.ones((3, 3), np.uint8)) > 0) & (hole == 0)
            if inner.any() and boundary.any():
                ring_score = max(ring_score, float(wood[inner].mean()*metal[boundary].mean()))
    return dict(dark=float(np.mean(v < 95)), pale=float(np.mean((s < 70) & (v > 165))),
                green=float(np.mean((h > 35) & (h < 90) & (s > 75))),
                red=float(np.mean(((h < 9) | (h > 170)) & (s > 95))),
                yellow=float(np.mean((h > 18) & (h < 38) & (s > 90))),
                blue=float(np.mean((h > 90) & (h < 145) & (s > 65))),
                brown=float(np.mean((h > 5) & (h < 28) & (s > 60) & (v < 175))),
                saturation=float(np.mean(s)/255), brightness=float(np.mean(v)/255),
                color_std=float(np.mean(np.std(pixels, axis=0))/80),
                rectangularity=float(cv2.contourArea(np.asarray(contour, np.float32))/rectangle_area),
                aspect=float(item['size'][0]/max(item['size'][1], .002)),
                edge=float(np.mean(edge[selected] > 0)),
                solid_pixel=np.asarray(solid_pixel)+[x, y], thickness=float(thickness),
                toy_face=toy_face_pattern(crop), ring_score=ring_score)


def _normal_scores(features, item):
    f = features
    aspect = f['aspect']
    slender = float(aspect > 4.2 and item['size'][0] > .075)
    phone = f['blue'] > .25 and aspect > 2.5
    return {'pen': 5*slender+1.2*float(aspect > 3)-2*f['pale'],
            'watch': 2*f['dark']+1.2*f['green']+.5*float(aspect < 2.4)-3*slender,
            'action_camera': 2.2*f['dark']+.9*f['blue']+float(aspect > 1.7)-3*slender,
            'toy_car': 1.8*f['red']+1.2*f['yellow']+.6*f['blue']+.6*float(1.5 < aspect < 4.5)-2*slender,
            'garage': 2*f['pale']+1.8*float(f['toy_face'] is not None)+.6*f['brown']-2*slender-.5*float(phone),
            'phone': 2*f['blue']+.5*f['pale']-2*slender}


def _language_scores(features, item):
    f = features
    aspect, rectangularity = f['aspect'], f['rectangularity']
    chromatic = f['red']+f['green']+f['yellow']
    return {'watch': 2.2*f['dark']+.3*f['green']-.4*f['saturation']+5*f['ring_score'],
            'car': .6*f['red']+.6*f['blue']+1.3*f['green']+1.0*f['dark']+.7*float(1.5 < aspect < 4.5)+.3*f['color_std'],
            'wooden_toy': 1.3*f['pale']+1.2*f['brown']+.4*f['color_std']+.3*float(aspect < 2.7),
            'pepper': 1.8*chromatic+f['saturation']+.7*(1-rectangularity)-.8*f['dark']+.4*float(aspect < 2.8),
            'chocolate_bar': 1.2*rectangularity+.8*float(aspect > 2.5)+.7*f['brown']+.4*f['edge']+.3*f['color_std']}


def _dino_scores(ctx, image, objects):
    """Reuse the retained v286 RGB prototypes for their existing object families."""
    from coap.tasks.sorting_paired_car_semantics import PROTOTYPES, PROTOTYPE_LABELS
    from coap.tasks.sorting_r03_dino import _encode
    from coap.tasks.memory import crop_of
    cache = getattr(ctx, '_classify_dino_cache', None)
    if cache is None:
        cache = ctx._classify_dino_cache = {}
    crops = [crop_of(image, item, margin=3) for item in objects]
    keys = [hashlib.sha256(crop.tobytes()+str(crop.shape).encode()).hexdigest() for crop in crops]
    pending = [index for index, key in enumerate(keys) if key not in cache]
    if pending:
        vectors = _encode([crops[index] for index in pending])
        for index, vector in zip(pending, vectors):
            cache[keys[index]] = vector
    aliases = {'toy_car': 'car', 'garage': 'toy', 'action_camera': 'camera', 'watch': 'watch', 'phone': 'phone'}
    for item, key in zip(objects, keys):
        similarities = np.einsum('ij,j->i', PROTOTYPES, cache[key])
        semantic = {category: float(max(value for value, label in zip(similarities, PROTOTYPE_LABELS)
                                        if label == prototype)) for category, prototype in aliases.items()}
        item['dino_class_cosines'] = semantic
        for category, value in semantic.items():
            item['category_scores'][category] = 10.*(value-.4)
        features = item['rgb_features']
        slender = features['aspect'] > 4.2 and item['size'][0] > .075
        item['category_scores']['pen'] = 7. if slender else -1.
        if 'hollow_ring_score' in item:
            item['category_scores']['watch'] += 5.
        item['category_scores']['watch'] += 3.*features['ring_score']


def _assign_categories(objects, categories, language, initial=True):
    if not objects:
        return
    scores = np.array([[item['category_scores'].get(category, -5.) for category in categories]
                       for item in objects])
    # A normal language episode contains two or three visible instances per named category.
    # This is a visual grouping prior, not an episode layout or object-state lookup.
    best = None
    if language and initial and 2*len(categories) <= len(objects) <= 3*len(categories):
        for counts in product((2, 3), repeat=len(categories)):
            if sum(counts) != len(objects):
                continue
            slots = np.repeat(np.arange(len(categories)), counts)
            rows, columns = linear_sum_assignment(-scores[:, slots])
            value = float(scores[rows, slots[columns]].sum())
            if best is None or value > best[0]:
                best = value, rows, slots[columns]
    if not language and initial and 3 <= len(objects) <= 9:
        for chosen in combinations(range(len(categories)), 3):
            for counts in product((1, 2, 3), repeat=3):
                if sum(counts) != len(objects):
                    continue
                slots = np.repeat(chosen, counts)
                rows, columns = linear_sum_assignment(-scores[:, slots])
                value = float(scores[rows, slots[columns]].sum())
                if best is None or value > best[0]:
                    best = value, rows, slots[columns]
    if best is None:
        labels = np.argmax(scores, axis=1)
        best = 0., np.arange(len(objects)), labels
    for row, column in zip(best[1], best[2]):
        category = categories[int(column)]
        objects[int(row)]['category'] = category
        objects[int(row)]['semantic_score'] = float(scores[int(row), int(column)])


def _decorate_grasp(ctx, image, camera, item):
    category = item['category']
    height = GRASP_HEIGHTS[category]
    surface_height = SURFACE_HEIGHTS[category]
    old_surface = float(item['surface_z'])
    pixel = np.asarray(item.get('case_pixel', item['pixel']), float)
    if category in ('pepper', 'wooden_toy'):
        pixel = np.asarray(item['rgb_features']['solid_pixel'], float)
    if category == 'garage' and item['rgb_features']['toy_face'] is not None:
        pixel = np.asarray(item['rgb_features']['toy_face'], float)+item['bbox'][:2]
    position = _project(ctx, [pixel], ctx.table_z+surface_height, camera, image.shape)[0]
    position[2] = ctx.table_z+height
    item['position'] = position.copy()
    item['grasp_position'] = position.copy()
    item['grasp_height'] = height
    item['surface_z'] = ctx.table_z+surface_height
    item['grasp_pixel'] = tuple(pixel)
    item['grasp_yaw'] = float(item['yaw'])
    item['size'] = (*item['size'][:2], surface_height)
    item['appearance_family'] = {'car': 'toy_car', 'wooden_toy': 'garage'}.get(category, category)
    item['surface_height_source'] = 'public_family_size'
    item['initial_projection_surface_z'] = old_surface


def _free_mapping(objects, baskets):
    categories = sorted({item['category'] for item in objects})
    if len(categories) > len(baskets):
        counts = {category: sum(item['category'] == category for item in objects) for category in categories}
        categories = sorted(categories, key=lambda category: -counts[category])[:len(baskets)]
    best = None
    for slots in permutations(range(len(baskets)), len(categories)):
        mapping = dict(zip(categories, slots))
        cost = sum(abs(item['position'][0]-baskets[mapping[item['category']]]['position'][0])
                   for item in objects if item['category'] in mapping)
        if best is None or cost < best[0]:
            best = cost, mapping
    return best[1] if best else {}


def observe(ctx):
    """Return visually measured sources and baskets with instruction grounding."""
    image, camera = _head(ctx)
    mapping = instruction_mapping(ctx.obs.get('instruction', ''))
    language = bool(mapping) or 'by_language' in str(ctx.task_name)
    categories = tuple(mapping) if mapping else (LANGUAGE_CATEGORIES if language else NORMAL_CATEGORIES)
    cached_baskets = getattr(ctx, '_classify_basket_cache', None)
    if cached_baskets is not None:
        baskets = deepcopy(cached_baskets)
    else:
        baskets = measure_baskets(ctx, image, camera)
        if len(baskets) == 3:
            ctx._classify_basket_cache = deepcopy(baskets)
    workspace = _object_mask(ctx, image, baskets)
    objects = []
    for contour in _contours(ctx, image, workspace, language):
        pixel, position, yaw, size, world = _geometry(ctx, contour, ctx.table_z+.030, camera, image)
        if not (.025 < size[0] < .23 and (.010 if language else .003) < size[1] < .15):
            continue
        if abs(position[0]) > .56 or not -.285 < position[1] < .40:
            continue
        if baskets:
            # Source selection follows the measured fronts of the basket row.
            front_y = min(basket['position'][1] for basket in baskets)
            if position[1] > front_y-.045:
                continue
        if _on_robot_material(ctx, pixel, image):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if width < 3 or height < 3 or area < 18:
            continue
        object_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 1, -1)
        crop = image[y:y+height, x:x+width].copy()
        item = dict(name='object', kind='object', position=position, yaw=yaw, size=(*size, .030),
                    pixel=tuple(pixel), bbox=(x, y, width, height), head_bbox=(x, y, width, height),
                    camera=camera, contour=contour[:, 0, :], crop=crop, area=area,
                    mean_rgb=image[object_mask > 0].mean(axis=0), confidence=.65,
                    surface_z=ctx.table_z+.030,
                    endpoints=np.array([world[np.argmin(world[:, :2]@[math.cos(yaw), math.sin(yaw)])],
                                        world[np.argmax(world[:, :2]@[math.cos(yaw), math.sin(yaw)])]]))
        item['views'] = {camera: dict(bbox=item['bbox'], pixel=item['pixel'], crop=crop, confidence=.65)}
        features = _features(image, contour[:, 0, :], item)
        item['rgb_features'] = features
        item['category_scores'] = (_language_scores if language else _normal_scores)(features, item)
        objects.append(item)
    if 'watch' in categories:
        rings, _ = ctx.perception.hollow_ring_objects(ctx.obs)
        for ring in rings:
            if baskets and ring['position'][1] > min(b['position'][1] for b in baskets)-.045:
                continue
            def ring_distance(item):
                x, y, width, height = item['bbox']
                a, b, w, h = ring['bbox']
                overlap = max(0, min(x+width, a+w)-max(x, a))*max(0, min(y+height, b+h)-max(y, b))
                if overlap > .15*min(width*height, w*h):
                    return 0.
                return np.linalg.norm(item['position'][:2]-ring['position'][:2])
            nearest = min(objects, key=ring_distance, default=None)
            if nearest is not None and ring_distance(nearest) < .055:
                nearest['category_scores']['watch'] += 5.
                nearest['case_pixel'] = ring['case_pixel']
                nearest['ring_pixel'] = ring['ring_pixel']
                nearest['hollow_ring_score'] = ring['hollow_ring_score']
            else:
                ring['surface_z'] = ctx.table_z+.030
                ring['rgb_features'] = _features(image, ring['contour'], ring)
                ring['category_scores'] = {category: -3. for category in categories}
                ring['category_scores']['watch'] = 6.
                objects.append(ring)
    classifier = getattr(ctx, 'classify_rgb', None)
    if callable(classifier) and objects:
        semantic_scores = classifier([item['crop'] for item in objects], categories)
        for item, values in zip(objects, semantic_scores):
            item['category_scores'] = {category: float(value) for category, value in zip(categories, values)}
    elif objects and not language:
        _dino_scores(ctx, image, objects)
    initial = not bool(getattr(ctx, '_classify_observed_initial', False))
    if not language and not initial:
        categories = getattr(ctx, '_classify_normal_categories', categories)
    count_prior = initial and not bool(getattr(ctx, '_classify_skip_count_prior', False))
    _assign_categories(objects, categories, language, initial=count_prior)
    if objects:
        ctx._classify_observed_initial = True
        if not language and initial:
            ctx._classify_normal_categories = tuple(sorted({item['category'] for item in objects}))
    for item in objects:
        _decorate_grasp(ctx, image, camera, item)
    objects.sort(key=lambda item: (item['category'], item['position'][0], item['position'][1]))
    if not language and baskets:
        mapping = getattr(ctx, '_classify_free_mapping', None)
        if mapping is None:
            mapping = _free_mapping(objects, baskets)
            ctx._classify_free_mapping = dict(mapping)
    return dict(objects=objects, baskets=baskets, instruction_mapping=mapping, mapping=mapping,
                evidence=dict(method='rgb_regions_and_public_projection', language=language,
                              category_source='instruction' if language else 'rgb_family_features',
                              camera=camera, basket_count=len(baskets), object_count=len(objects)))


classify_scene = observe
observe_scene = observe
