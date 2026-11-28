"""Reconstruct a painted keyboard and a wooden mallet from calibrated RGB."""

from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import Camera, pose_matrix


# Color order describes what the visible keyboard encodes, including repeated hues.
KEY_COLORS = np.array([0, 1, 2, 3, 4, 4, 5, 5])
GROUP_INDEX = np.array([0., 1., 2., 3., 4.5, 6.5])
UP = np.array([0., 0., 1.])


def wrist_mount():
    # Fixed transforms from the public X5A URDF and wrist camera calibration.
    def frame(position, angles, order='xyz', degrees=False):
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = Rotation.from_euler(order, angles, degrees=degrees).as_matrix()
        return transform

    return (frame([.057, 0, 0], [0, 0, -3.1416])
            @ frame([-.0275, 0, .05], [0, .3491, np.pi])
            @ frame([0, 0, .001], [0, -80, -90], 'XYZ', True))


WRIST_MOUNT = wrist_mount()


def views_for_observation(head, observation):
    views = {}
    for name, image in observation['vision'].items():
        rgb = np.asarray(image['color'])
        if name == 'cam_head':
            views[name] = (head, rgb)
        elif name in ('cam_left_wrist', 'cam_right_wrist'):
            side = name.split('_')[1]
            height, width = rgb.shape[:2]
            focal = width * 13 / 20.955
            camera = Camera(dict(K=[[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]],
                                 camera_to_world_opengl=pose_matrix(observation['state'][side + '_ee_pose']) @ WRIST_MOUNT,
                                 resolution=[width, height]))
            views[name] = (camera, rgb)
    return views


def rays(camera, pixels):
    pixels = np.asarray(pixels, float)
    homogeneous = np.concatenate((pixels, np.ones_like(pixels[..., :1])), axis=-1)
    return homogeneous @ np.linalg.inv(camera.k).T @ camera.cv_to_gl.T @ camera.c2w[:3, :3].T


def on_plane(camera, pixels, z):
    ray = rays(camera, pixels)
    origin = camera.c2w[:3, 3]
    return origin + ray * ((z-origin[2])/ray[..., 2])[..., None]


def sample(image, pixels):
    shape = np.shape(pixels)[:-1]
    pixels = np.float32(pixels).reshape(-1, 2)
    values = cv2.remap(image.astype(np.float32),
                       pixels[:, 0].reshape(-1, 1), pixels[:, 1].reshape(-1, 1),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return values.reshape(shape)


def key_masks(rgb):
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    limits = ((0, 7, 170), (8, 25, 173), (26, 38, 150), (39, 90, 120),
              (91, 130, 90), (131, 159, 90))
    masks = []
    for group, (lo, hi, minimum) in enumerate(limits):
        hue = (h >= lo) & (h <= hi)
        if group == 0:
            hue |= h >= 160
        masks.append(np.uint8(hue & (s >= 150) & (v >= minimum)))
    return np.asarray(masks)


def keyboard(camera, rgb, support_z):
    """Group painted contours in a metric proxy image, retaining raw pixel rays."""
    masks = key_masks(rgb)
    groups = []
    for group, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:7]:
            if cv2.contourArea(contour) < 45:
                continue
            xy = on_plane(camera, contour[:, 0], support_z)[:, :2]
            center, dimensions, _ = cv2.minAreaRect(xy.astype(np.float32))
            span = sorted(dimensions)
            if span[0] <= 0 or span[1] > 10*span[0]:
                continue
            candidates.append((np.array(center), span, group))
        groups.append(candidates)
    candidates = [value for group in groups for value in group]
    best = None
    for a, b in combinations(candidates, 2):
        i, j = a[2], b[2]
        if i == j:
            continue
        step = (b[0]-a[0])/(GROUP_INDEX[j]-GROUP_INDEX[i])
        pitch = np.linalg.norm(step)
        if pitch < min(a[1][0], b[1][0])/3 or pitch > max(a[1][0], b[1][0])*2:
            continue
        origin = a[0]-GROUP_INDEX[i]*step
        axis = step/pitch
        along = np.array([-axis[1], axis[0]])
        matched = []
        for group, choices in enumerate(groups):
            predicted = origin + GROUP_INDEX[group]*step
            acceptable = [(abs((point-predicted)@axis) + .35*abs((point-predicted)@along), point, span)
                          for point, span, _ in choices
                          if abs((point-predicted)@axis) < .62*pitch
                          and abs((point-predicted)@along) < pitch
                          and span[1] > 1.2*pitch and span[0] < (1.4 if group < 4 else 2.8)*pitch]
            if acceptable:
                cost, point, span = min(acceptable, key=lambda row: row[0])
                matched.append((group, point, cost))
        if len(matched) < 3:
            continue
        across = origin + np.arange(len(KEY_COLORS))[:, None] * step
        scan = across[:, None, :] + np.linspace(-1.2, 1.2, 9)[None, :, None]*pitch*along
        projected = camera.project(np.concatenate((scan, np.full((*scan.shape[:2], 1), support_z)), axis=-1))
        paint = np.array([sample(masks[group], pixels).mean() for group, pixels in zip(KEY_COLORS, projected)])
        score = (paint > .40).sum() + .2*len(matched) + .12*paint.sum() - sum(v[2] for v in matched)/pitch*.10
        if best is None or score > best[0]:
            best = (score, origin, step, matched, paint)
    if best is None:
        return None

    score, origin, step, matched, paint = best
    pitch = np.linalg.norm(step)
    axis = step/pitch
    along = np.array([-axis[1], axis[0]])
    across = origin+np.arange(len(KEY_COLORS))[:, None]*step
    ray_patches = []
    key_points = []
    supports = []
    for center, group in zip(across, KEY_COLORS):
        # Find the current surface interval inside each observed lateral stripe.
        stripe = center + pitch*(np.linspace(-.46, .46, 17)[None, :, None]*axis
                                 + np.linspace(-3.4, 3.4, 87)[:, None, None]*along)
        pix = camera.project(np.concatenate((stripe, np.full((*stripe.shape[:2], 1), support_z)), axis=-1))
        labeled = sample(masks[group], pix) > .60
        rows = np.flatnonzero(labeled.mean(1) > .35)
        cols = np.flatnonzero(labeled.mean(0) > .17)
        if len(rows) >= 6 and len(cols) >= 3:
            r0, r1 = np.quantile(rows, [.07, .93])
            c0, c1 = np.quantile(cols, [.06, .94])
            point = (stripe[int(round(r0)), int(round(c0))]+stripe[int(round(r1)), int(round(c1))])/2
            supports.append(True)
        else:
            # Use the spacing measured from the other painted bars when a bar is occluded.
            point = center
            supports.append(False)
        key_points.append(point)
        # Use measured color boundaries for the multi-view plane fit.
        stride = 4
        a = labeled[::stride, ::stride]
        ray_patches.append((int(group), pix[::stride, ::stride].reshape(-1, 2), a.ravel()))
    if sum(supports) < 5:
        return None
    # Pool the visible centers into one ordered row. Single-bar masks can
    # include the painted sidewall or part of a neighbor with the same hue.
    key_points = np.asarray(key_points)
    seen = np.asarray(supports)
    row_basis = np.c_[np.ones(len(KEY_COLORS)), np.arange(len(KEY_COLORS))]
    initial = np.linalg.lstsq(row_basis[seen], key_points[seen], rcond=None)[0]
    row = least_squares(lambda p: (row_basis[seen]@p.reshape(2, 2)-key_points[seen]).ravel(),
                        initial.ravel(), loss='soft_l1', f_scale=.10*pitch, max_nfev=20)
    row_centers = row_basis@row.x.reshape(2, 2)
    offset = np.clip(key_points-row_centers, -.12*pitch, .12*pitch)
    key_points = row_centers + .20*offset*seen[:, None]
    pixels = camera.project(np.c_[key_points, np.full(len(KEY_COLORS), support_z)])
    return dict(pixels=pixels, ray_patches=ray_patches, pitch=float(pitch),
                score=float(score), observed_bars=sum(supports))


def board_depth(head, board, wrist_views, support_z):
    """Choose the plane aligning observed paint edges across calibrated views."""
    offsets = np.linspace(0., 1.7*board['pitch'], 53)
    heights = support_z + offsets
    scores = []
    counts = []
    for camera, rgb in wrist_views:
        masks = key_masks(rgb)
        numerator = np.zeros(len(heights))
        denominator = np.zeros(len(heights))
        for group, pixels, label in board['ray_patches']:
            if np.count_nonzero(label) < 8:
                continue
            world = on_plane(head, pixels[None, :, :], heights[:, None])
            uv = camera.project(world)
            visible = ((uv[..., 0] > 2) & (uv[..., 0] < rgb.shape[1]-3)
                       & (uv[..., 1] > 2) & (uv[..., 1] < rgb.shape[0]-3))
            # Positive and negative samples are balanced independently.
            value = sample(masks[group], uv)
            pos = np.sum(visible & label, axis=1)
            neg = np.sum(visible & ~label, axis=1)
            score = (np.sum(visible*label*value, axis=1)/np.maximum(1, pos)
                     - np.sum(visible*~label*value, axis=1)/np.maximum(1, neg))
            weights = np.minimum(pos, neg) * (pos >= 8) * (neg >= 8)
            numerator += score*weights
            denominator += weights
        valid = denominator > 12
        profile = np.where(valid, numerator/np.maximum(1, denominator), 0.)
        if profile.max() > .18:
            scores.append(profile)
            counts.append(denominator)
    if not scores:
        return None
    weights = np.asarray(counts)**.5
    score = np.sum(np.asarray(scores)*weights, axis=0)/np.maximum(1., weights.sum(0))
    index = int(np.argmax(score))
    if score[index] < .18 or not 0 < index < len(heights)-1:
        return None
    return float(heights[index]), float(score[index])


def red_heads(rgb):
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    groups = []
    for saturation in (105, 135, 165, 190):
        red = np.uint8(((h < 8) | (h >= 160)) & (s >= saturation) & (v >= 35))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 9 or cv2.contourArea(contour) < 12:
                continue
            center, axes, _ = cv2.fitEllipse(contour)
            radius = .5*np.sqrt(np.prod(axes))
            if (min(axes) < .58*max(axes) or not 2.7 < radius < min(rgb.shape[:2])*.19
                    or cv2.contourArea(contour) < .60*np.pi*radius**2):
                continue
            center = np.asarray(center)
            match = next((points for points in groups if np.linalg.norm(points[0][0]-center) < 1.6), None)
            row = (center, float(radius))
            if match is None:
                groups.append([row])
            else:
                match.append(row)
    return [(np.median([v[0] for v in group], axis=0), np.median([v[1] for v in group]))
            for group in groups if len(group) > 1]


def shaft(rgb, center, radius):
    """Find the directed thin brightness ridge and measure its visible extent."""
    angle = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    axis = np.c_[np.cos(angle), np.sin(angle)]
    normal = np.c_[-axis[:, 1], axis[:, 0]]
    distance = np.linspace(1.5, 18.5, 76)*radius
    ribbon = center+axis[:, None, :]*distance[None, :, None]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scores = []
    for width in (.27, .42, .60):
        inside = sample(gray, ribbon)
        left = sample(gray, ribbon + normal[:, None, :]*(width*radius+1.))
        right = sample(gray, ribbon - normal[:, None, :]*(width*radius+1.))
        contrast = np.minimum(inside-left, inside-right)
        support = (contrast > 5.) & (inside > 90.)
        # Require a sustained near segment as well as evidence farther along.
        near = support[:, :28].mean(1)
        far = support[:, 28:].mean(1)
        rank = near + far + .006*np.maximum(0., np.minimum(contrast[:, :45], 45.)).mean(1)
        index = int(np.argmax(rank))
        hit = np.flatnonzero(support[index])
        if len(hit) >= 14 and near[index] >= .30 and far[index] >= .12:
            extent = float(distance[int(np.quantile(hit, .94))])
            scores.append((float(rank[index]), index, extent, hit))
    if not scores:
        return None
    score, index, extent, hit = max(scores, key=lambda p: p[0])
    offsets = np.linspace(-1.4, 1.4, 37)*radius
    strips = ribbon[index, hit, None, :] + offsets[None, :, None]*normal[index]
    profile = np.median(sample(gray, strips), axis=0)
    middle = len(profile)//2
    dark = np.percentile(profile, 20)
    levels = profile > .5*(dark+profile[middle])
    bounds = np.flatnonzero(~levels)
    left = bounds[bounds < middle]
    right = bounds[bounds > middle]
    half_width = .5*(offsets[right[0]]-offsets[left[-1]]) if len(left) and len(right) else np.std(offsets)
    return dict(axis=axis[index], extent=extent, half_width_px=float(half_width), score=score)


def mallet(rgb):
    candidates = []
    for center, radius in red_heads(rgb):
        handle = shaft(rgb, center, radius)
        if handle is not None:
            candidates.append(dict(center=center, radius_px=float(radius), **handle))
    return max(candidates, key=lambda x: x['score']) if candidates else None


def triangulate(records):
    """Fit one point from camera/pixel pairs in a common frame."""
    lhs, rhs = [], []
    for camera, pixel in records:
        ray = rays(camera, pixel)
        ray /= np.linalg.norm(ray)
        normal = np.eye(3)-np.outer(ray, ray)
        lhs.append(normal)
        rhs.append(normal@camera.c2w[:3, 3])
    return np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)[0]


def mallet_depth(head, detection, wrist_views, support_z, pitch):
    """Match the observed spherical head along its epipolar line."""
    target = detection['center']
    endpoints = on_plane(head, target, np.array([support_z, support_z + 2*pitch]))
    matches = []
    for camera, rgb in wrist_views:
        line = camera.project(endpoints)
        direction = line[1]-line[0]
        norm2 = direction@direction
        for center, radius in red_heads(rgb):
            ratio = np.clip((center-line[0])@direction/max(norm2, 1.e-6), 0., 1.)
            error = np.linalg.norm(line[0]+ratio*direction-center)
            if error > max(4., .4*radius):
                continue
            world = triangulate(((head, target), (camera, center)))
            distance = np.linalg.norm(world-camera.c2w[:3, 3])
            head_distance = np.linalg.norm(world-head.c2w[:3, 3])
            apparent = radius/camera.k[0, 0]*distance
            radius_prior = detection['radius_px']/head.k[0, 0]*head_distance
            discrepancy = abs(np.log(apparent/radius_prior))
            if support_z-pitch*.15 < world[2] < support_z+2*pitch and discrepancy < .55:
                matches.append((error+8*discrepancy, world, (apparent+radius_prior)/2))
    if not matches:
        return None
    _, center, radius = min(matches, key=lambda p: p[0])
    return center, float(radius)


def associated_head(camera, rgb, expected, search_fraction=.04):
    target = camera.project(expected)
    diameter = min(rgb.shape[:2])
    candidates = [(np.linalg.norm(center-target), center, radius)
                  for center, radius in red_heads(rgb)]
    if not candidates:
        return None
    distance, center, radius = min(candidates, key=lambda p: p[0])
    return (center, radius) if distance < max(4*radius, search_fraction*diameter) else None


def held_head(records):
    """Fit an attachment and radius in the moving link6 frame."""
    if len(records) < 2:
        return None
    origins = np.array([camera.c2w[:3, 3] for camera, _, _ in records])
    if np.max(np.linalg.norm(origins[:, None]-origins[None, :], axis=-1)) < .024:
        return None
    start = triangulate([(camera, pixel) for camera, pixel, _ in records])

    def residual(point):
        return np.concatenate([camera.project(point)-pixel for camera, pixel, _ in records])

    fit = least_squares(residual, start, loss='soft_l1', f_scale=1.5, max_nfev=28)
    point = fit.x
    errors = np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)
    used = errors <= max(2.5, np.median(errors)*1.8)
    if sum(used) < 2 or np.median(errors) > 4.:
        return None
    radii = [radius*np.linalg.norm(point-camera.c2w[:3, 3])/camera.k[0, 0]
             for i, (camera, _, radius) in enumerate(records) if used[i]]
    return point, float(np.median(radii)), float(np.median(errors))
