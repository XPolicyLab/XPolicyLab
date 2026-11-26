"""Shared perception primitives for skills executed by the episode runner."""

import cv2
import numpy as np

from . import instructions


def prompt_results(query, prompts, *, on_error=None, **query_options):
    """Yield facade results in prompt order; never query beyond the caller's cutoff."""
    for prompt in prompts:
        try:
            result = query(prompt, **query_options)
        except Exception as error:
            if on_error is None:
                raise
            on_error(prompt, error)
            continue
        yield prompt, result


def find(robo, prompts, camera='cam_head'):
    for prompt, result in prompt_results(robo.locate, prompts, camera=camera):
        if result.found and result.center_xyz is not None:
            return prompt, result
    return None, None


def camera_matrices(robo, camera, *, dtype=float, step=None):
    meta = robo.camera_meta(camera) if step is None else robo.camera_meta(camera, step=step)
    return np.array(meta['intrinsic_matrix'], dtype), np.array(meta['extrinsic_matrix'], dtype)


def fit_circular_pad_row(gradient, grid, spacing):
    """Fit a centered row using the sum of rim support penalized for uneven support.

    A rim sample must be stronger than its immediate radial neighbors. Scaling the
    harmonic mean by the row length lets supported outer pads beat a brighter subset
    without letting strong inner rims make up for a missing outer pad.
    """
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    radial_offsets = np.array([-0.004, 0.0, 0.004])[:, None, None]
    height, width = gradient.shape

    def score(count, y, radius, dx):
        centers = (np.arange(count) - (count - 1) / 2.0) * spacing + dx
        rings = radius + radial_offsets
        rows, cols = grid.rc(centers[None, :, None] + rings * cosines, y + rings * sines)
        rows, cols = np.rint(rows).astype(int), np.rint(cols).astype(int)
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        samples = np.where(in_bounds, gradient[rows.clip(0, height - 1), cols.clip(0, width - 1)], np.nan)
        visible = np.isfinite(samples).all(axis=0)
        rims = np.where(visible, samples[1] - np.maximum(samples[0], samples[2]), 0.0).mean(axis=1)
        if not (rims > 0).all():
            return 0.0
        return count ** 2 / np.reciprocal(rims).sum()

    best = None
    for count in range(2, 1 + int(0.60 / spacing)):
        coarse = [(score(count, y, radius, 0.0), y, radius)
                  for y in np.arange(-0.118, -0.032, 0.002) for radius in (0.038, 0.040, 0.042)]
        for _, y, radius in sorted(coarse, reverse=True)[:3]:
            for yy in np.arange(y - 0.0015, y + 0.0016, 0.0005):
                for rr in (radius - 0.001, radius, radius + 0.001):
                    for dx in (-0.002, -0.001, 0.0, 0.001, 0.002):
                        support = score(count, yy, rr, dx)
                        if support > 0 and (best is None or support > best[0]):
                            best = (support, count, float(yy), float(rr), dx)
    return best


def locate_baskets(robo):
    """Measure three distinct baskets and order them by observed x coordinate.

    Uses several SAM3 wordings and, if needed, connected components on a generic
    basket mask. Does not fall back to a fixed layout table.
    """
    import math

    candidates = []  # (x, y, score)

    def _add(x, y, score=0.0):
        x, y, score = float(x), float(y), float(score)
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        if not (-0.55 <= x <= 0.55 and -0.35 <= y <= 0.35):
            return
        for index, (cx, cy, prior) in enumerate(candidates):
            if math.hypot(x - cx, y - cy) < 0.12:
                if score > prior:
                    candidates[index] = (x, y, score)
                return
        candidates.append((x, y, score))

    prompts = (
        'basket',
        'white basket',
        'plastic basket',
        'left basket',
        'middle basket',
        'center basket',
        'centre basket',
        'right basket',
        'leftmost basket',
        'rightmost basket',
    )
    for prompt in prompts:
        for min_score in (0.4, 0.25, 0.15):
            try:
                sight = robo.locate(prompt, camera='cam_head', min_score=min_score)
            except Exception:
                continue
            if sight.found and sight.center_xyz is not None:
                _add(sight.center_xyz[0], sight.center_xyz[1], sight.score or min_score)
                break

    if len(candidates) < 3:
        sam3 = getattr(robo, '_sam3', None)
        if sam3 is not None:
            try:
                image = robo.image('cam_head')
                intrinsics, extrinsics = camera_matrices(robo, 'cam_head')
                for prompt in ('basket', 'white basket', 'baskets'):
                    result = sam3.segment(image, text_prompt=prompt, min_score=0.12)
                    mask = getattr(result, 'mask', None)
                    if not getattr(result, 'found', False) or mask is None:
                        continue
                    for _mask, area, centroid in component_masks(mask, min_px=400):
                        row, col = centroid
                        point = ray_plane(intrinsics, extrinsics, float(row), float(col), 0.78)
                        _add(point[0], point[1], area / 1.0e4)
            except Exception:
                pass

    if len(candidates) < 3:
        raise ValueError(f'could not locate three baskets; found {len(candidates)}')

    candidates.sort(key=lambda item: item[0])
    if len(candidates) == 3:
        chosen = candidates
    else:
        left, right = candidates[0], candidates[-1]
        mid = min(candidates[1:-1], key=lambda item: abs(item[0] - 0.5 * (left[0] + right[0])))
        chosen = [left, mid, right]
    return tuple(item[0] for item in chosen), tuple(item[1] for item in chosen)


def pixel_ray(intrinsics, extrinsics, row, col):
    camera_direction = np.array([
        (col - intrinsics[0, 2]) / intrinsics[0, 0],
        -(row - intrinsics[1, 2]) / intrinsics[1, 1], -1.0,
    ])
    return extrinsics[:3, :3] @ camera_direction


def ray_plane(intrinsics, extrinsics, row, col, z):
    direction = pixel_ray(intrinsics, extrinsics, row, col)
    origin = extrinsics[:3, 3]
    return origin + ((z - origin[2]) / direction[2]) * direction


def plane_points(intrinsics, extrinsics, rows, cols, z, *, pixel_center=False, xy_only=False):
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    if pixel_center:
        rows = rows + 0.5
        cols = cols + 0.5
    directions = np.stack([
        (cols - intrinsics[0, 2]) / intrinsics[0, 0],
        -(rows - intrinsics[1, 2]) / intrinsics[1, 1], -np.ones(len(rows)),
    ], 1) @ extrinsics[:3, :3].T
    scale = (z - extrinsics[2, 3]) / directions[:, 2]
    axes = slice(2 if xy_only else 3)
    return extrinsics[axes, 3][None, :] + directions[:, axes] * scale[:, None]


def colour_components(mask, *, as_uint8=True, connectivity=None):
    """Return OpenCV labels, statistics and centroids without task-specific filtering."""
    if as_uint8:
        return cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if connectivity is None:
        return cv2.connectedComponentsWithStats(mask)
    return cv2.connectedComponentsWithStats(mask, connectivity=connectivity)


def component_masks(mask, min_px, *, with_centroid=True):
    count, labels, stats, centroids = colour_components(mask)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_px:
            component = (labels == index, area)
            if with_centroid:
                component += (tuple(centroids[index, ::-1]),)
            components.append(component)
    return components
