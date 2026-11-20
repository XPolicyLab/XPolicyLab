from itertools import combinations

import cv2
import numpy as np

from skills.shared.perception.current_vessel_regions.implementation import boundary


def intersection(first, second):
    direction = first[1] - first[0]
    other = second[1] - second[0]
    system = np.column_stack((direction, -other))
    if abs(np.linalg.det(system)) < 1e-5:
        return None
    fraction = np.linalg.solve(system, second[0] - first[0])[0]
    return first[0] + direction * fraction


def run(image, region):
    image = np.asarray(image)
    outline = np.asarray(region.get('outline'), dtype=float)
    if (image.shape != (480, 640, 3) or image.dtype != np.uint8 or outline.shape != (96, 2)
            or not np.isfinite(outline).all() or region.get('clipped') is not False
            or np.any(outline <= 1) or np.any(outline[:, 0] >= 638) or np.any(outline[:, 1] >= 478)):
        return []
    contained = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(contained, [np.rint(outline).astype(np.int32)], 1)
    parent_area = float(np.count_nonzero(contained))
    upper_level = np.min(outline[:, 1]) + .25 * np.ptp(outline[:, 1])
    upper_mask = np.zeros_like(contained)
    for start, end in zip(outline, np.roll(outline, -1, axis=0)):
        if max(start[1], end[1]) <= upper_level:
            cv2.line(upper_mask, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), 1, 1)
    upper_distance = cv2.distanceTransform((upper_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 6, 18)
    for channel in (gray, *cv2.split(image)):
        edges |= cv2.Canny(channel, 18, 58)
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    bounds = cv2.boundingRect(outline.astype(np.float32))
    length = max(8, int(min(bounds[2:]) * .25))
    detected = cv2.HoughLinesP(edges * contained, 1., np.pi / 180., length,
        minLineLength=length, maxLineGap=3)
    if detected is None:
        return []
    segments = []
    for candidate in sorted(detected.reshape(-1, 2, 2), key=lambda value: np.linalg.norm(value[1] - value[0]), reverse=True):
        candidate = candidate.astype(float)
        direction = candidate[1] - candidate[0]
        direction /= np.linalg.norm(direction)
        normal = np.asarray([-direction[1], direction[0]])
        if any(abs(direction @ previous[1]) > .995 and abs(normal @ (candidate[0] - previous[0][0])) < 2.
               for previous in segments):
            continue
        segments.append((candidate, direction))
        if len(segments) == 24:
            break
    pairs = []
    for first, second in combinations(range(len(segments)), 2):
        line, direction = segments[first]
        other, axis = segments[second]
        normal = np.asarray([-direction[1], direction[0]])
        if abs(direction @ axis) >= .86 and abs(normal @ (other[0] - line[0])) >= 4.:
            pairs.append((first, second))
    pairs.sort(key=lambda pair: sum(np.linalg.norm(segments[index][0][1] - segments[index][0][0]) for index in pair), reverse=True)
    found = []
    fractions = np.linspace(0., 1., 24)
    for first, second in combinations(pairs[:96], 2):
        if len(set(first + second)) != 4 or abs(segments[first[0]][1] @ segments[second[0]][1]) > .8:
            continue
        vertices = [intersection(segments[first[0]][0], segments[second[0]][0]),
            intersection(segments[first[0]][0], segments[second[1]][0]),
            intersection(segments[first[1]][0], segments[second[1]][0]),
            intersection(segments[first[1]][0], segments[second[0]][0])]
        if any(point is None for point in vertices):
            continue
        vertices = np.asarray(vertices)
        if (not np.isfinite(vertices).all() or np.any(vertices <= 1)
                or np.any(vertices[:, 0] >= 638) or np.any(vertices[:, 1] >= 478)
                or not cv2.isContourConvex(vertices.astype(np.float32))):
            continue
        area = float(cv2.contourArea(vertices.astype(np.float32)))
        if not max(40., parent_area * .08) <= area <= parent_area * .92:
            continue
        samples = np.asarray([start + fractions[:, None] * (end - start)
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0))])
        pixels = np.rint(samples).astype(np.int32)
        if np.mean(contained[pixels[..., 1], pixels[..., 0]]) < .99:
            continue
        distances = edge_distance[pixels[..., 1], pixels[..., 0]]
        support = np.mean(distances <= 1.5, axis=1)
        if np.min(support) < .85 or np.mean(distances) > .8:
            continue
        center = intersection(vertices[[0, 2]], vertices[[1, 3]])
        if center is None or cv2.pointPolygonTest(vertices.astype(np.float32), tuple(center), False) <= 0:
            continue
        score = area * float(np.mean(support))
        upper_agreement = bool(np.any(np.mean(upper_distance[pixels[..., 1], pixels[..., 0]] <= 2.5, axis=1) >= .85))
        previous_index = next((index for index, item in enumerate(found) if np.linalg.norm(center - item['projective_center']) < 3.
            and abs(area - item['area_pixels']) < .15 * area), None)
        if previous_index is not None:
            if score <= found[previous_index]['edge_support_score']:
                continue
            found.pop(previous_index)
        found.append({'outline': boundary(vertices), 'centroid': center.tolist(), 'projective_center': center.tolist(),
            'bbox': list(cv2.boundingRect(vertices.astype(np.float32))), 'area_pixels': area,
            'edge_support_score': score, 'edge_support_per_side': support.tolist(),
            'observed_upper_body_edge': bool(upper_agreement),
            'source': 'current_public_closed_mask_line_boundary', 'clipped': False,
            'boundary_is_current_image_line_fit': True, 'physical_asset_geometry_used': False})
    return sorted(found, key=lambda item: item['edge_support_score'], reverse=True)[:4]
