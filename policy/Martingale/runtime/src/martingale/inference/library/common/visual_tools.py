"""Image geometry from camera calibration."""
import numpy as np

def closest_point_on_line(point, start, end, *, segment=False):
    p, a, b = map(lambda x: np.asarray(x, float), (point, start, end))
    if p.shape != a.shape or a.shape != b.shape or (not all((np.isfinite(x).all() for x in (p, a, b)))):
        raise ValueError('Invalid points')
    d = b - a
    den = float(d @ d)
    if den <= 1e-12:
        raise ValueError('Degenerate line')
    t = float((p - a) @ d / den)
    if segment:
        t = float(np.clip(t, 0, 1))
    return {'point': a + t * d, 'fraction': t}

def crop(image, box, scale=1):
    """Crop and optionally resize RGB; return image AND original-coordinate mapping.

    box is (left, top, right, bottom) in XY pixels of the input image, with
    exclusive right/bottom. For a 640x480 image: right<=640 and bottom<=480.
    scale must be positive. Returns a DICT with exactly image (PIL.Image),
    original_xy_offset ([left,top]), scale, mapping (text); there is NO box key.
    Display result['image']. Convert any displayed (row,col) back as
    (row/scale+top, col/scale+left) before using the original camera calibration.
    Raises ValueError for bounds/scale errors. No robot motion or object detection.
    """
    from PIL import Image
    array = np.asarray(image)
    left, top, right, bottom = map(int, box)
    if not (0 <= left < right <= array.shape[1] and 0 <= top < bottom <= array.shape[0]) or scale <= 0:
        raise ValueError('Invalid crop')
    result = Image.fromarray(array[top:bottom, left:right]).resize((round((right - left) * scale), round((bottom - top) * scale)))
    return {'image': result, 'original_xy_offset': [left, top], 'scale': scale, 'mapping': 'original row=display row/scale+top; col=display col/scale+left'}

def annotate(image, points=(), lines=()):
    from PIL import Image, ImageDraw
    result = Image.fromarray(np.asarray(image).copy())
    draw = ImageDraw.Draw(result)
    for index, (row, col) in enumerate(points):
        draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill='red')
        draw.text((col + 5, row), str(index), fill='red')
    for a, b in lines:
        draw.line((a[1], a[0], b[1], b[0]), fill='lime', width=2)
    return result

def path_pixels(points, shape, *, coordinates):
    """Convert an ordered path to pixel (row, col); convention is mandatory.

    Supported: pixel_rc, normalized_1000_rc (Google example), normalized_01_xy
    (HAMSTER example). Normalized endpoints map to the first/last pixel centers.
    """
    h, w = shape[:2]
    p = np.asarray(points, float)
    if h < 2 or w < 2 or p.ndim != 2 or (p.shape[1] != 2) or (len(p) < 2) or (not np.isfinite(p).all()):
        raise ValueError('Need at least two finite points and a valid image shape')
    if coordinates == 'pixel_rc':
        result = p.copy()
    elif coordinates in ('normalized_1000_rc', 'normalized_01_xy'):
        scale = 1000.0 if coordinates == 'normalized_1000_rc' else 1.0
        if np.any(p < 0) or np.any(p > scale):
            raise ValueError('Normalized point outside image')
        result = (p if scale == 1000 else p[:, ::-1]) / scale * np.array([h - 1, w - 1])
    else:
        raise ValueError('Explicit supported coordinate convention required')
    if np.any(result < 0) or np.any(result > np.array([h - 1, w - 1])):
        raise ValueError('Pixel point outside image')
    return result

def trajectory_overlay(image, paths, *, coordinates):
    """Draw named ordered candidate polylines; does not execute or certify paths.

    paths maps a candidate name to points. Original RGB is never modified.
    Inspect this derived image alongside the original and another camera view.
    """
    from PIL import Image, ImageDraw
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')
    if not paths:
        raise ValueError('No candidate paths')
    result = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(result)
    converted = {}
    palette = ('red', 'cyan', 'yellow', 'magenta', 'lime', 'orange')
    for index, (name, points) in enumerate(paths.items()):
        rc = path_pixels(points, array.shape, coordinates=coordinates)
        converted[name] = rc.tolist()
        color = palette[index % len(palette)]
        draw.line([(float(c), float(r)) for r, c in rc], fill=color, width=3)
        for step, (row, col) in enumerate(rc):
            draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill=color)
            draw.text((col + 5, row + 3), str(name) + ':' + str(step), fill=color)
    return {'image': result, 'paths_pixel_rc': converted, 'collision_checked': False, 'limitation': 'Image-space proposals only; depth, arm, grasp orientation and swept geometry are not checked.'}

def path_on_plane(meta, points, *, coordinates, plane_point, plane_normal):
    """Intersect image rays with an explicit caller-specified world plane; no motion.

    points follow the mandatory coordinates convention accepted by path_pixels.
    meta is synchronized original-image camera-to-world OpenGL calibration.
    plane_point and plane_normal are world-frame 3-vectors; point units are meters.
    Returns dict(xyz=list of world XYZ lists, plane_point=list, plane_normal=list,
    collision_checked=False, depth_source=text). There is NO world_xyz key.
    This is conditional on the supplied plane, NOT measured depth or a collision
    check. ValueError rejects malformed planes/pixels, parallel rays or intersections
    behind the camera. Cropped/scaled pixels must first map to original-image pixels.
    """
    from ..common.assistive_vision import _ray
    origin = np.asarray(plane_point, float)
    normal = np.asarray(plane_normal, float)
    if origin.shape != (3,) or normal.shape != (3,) or (not np.isfinite(origin).all()) or (not np.isfinite(normal).all()) or (np.linalg.norm(normal) < 1e-09):
        raise ValueError('Invalid explicit plane')
    normal = normal / np.linalg.norm(normal)
    result = []
    for point in path_pixels(points, meta['shape'], coordinates=coordinates):
        camera, ray = _ray(meta, point)
        den = float(normal @ ray)
        if abs(den) < 1e-06:
            raise ValueError('Ray nearly parallel to plane')
        distance = float(normal @ (origin - camera) / den)
        if distance <= 0:
            raise ValueError('Plane intersection is not in front of camera')
        result.append((camera + distance * ray).tolist())
    return {'xyz': result, 'plane_point': origin.tolist(), 'plane_normal': normal.tolist(), 'collision_checked': False, 'depth_source': 'caller-specified plane; not measured scene depth'}

def mask_axis(mask, *, minimum_pixels=40, minimum_ratio=1.4):
    """2D visible-mask PCA, NOT a verified 3D object pose. Occlusion biases it."""
    pixels = np.argwhere(np.asarray(mask, dtype=bool))
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < minimum_pixels:
        raise ValueError('Insufficient mask pixels')
    center = pixels.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((pixels - center).T))
    ratio = float(values[-1] / max(values[0], 1e-12))
    if ratio < minimum_ratio:
        raise ValueError('Near-circular silhouette; seek another view')
    axis = vectors[:, -1]
    projection = (pixels - center) @ axis
    return {'center_rc': center, 'endpoints_rc': [center + axis * np.quantile(projection, q) for q in (0.05, 0.95)], 'eigenvalue_ratio': ratio, 'pose_verified': False, 'limitation': 'Visible silhouette only; identity and occlusion must be checked'}

def axis_from_views(metas, lines, *, maximum_plane_residual_degrees=8, minimum_second_singular_value=0.2):
    from ..common.assistive_vision import _ray, _unit
    if len(metas) != len(lines) or len(lines) < 3:
        raise ValueError('At least three independent views required')
    normals = []
    for meta, line in zip(metas, lines):
        if len(line) != 2:
            raise ValueError('Two row/col points per axis line')
        _, a = _ray(meta, line[0])
        _, b = _ray(meta, line[1])
        normals.append(_unit(np.cross(a, b)))
    _, singular, vh = np.linalg.svd(normals)
    axis = vh[-1]
    residual = np.degrees(np.arcsin(np.clip(np.abs(np.asarray(normals) @ axis), 0, 1)))
    if singular[1] < minimum_second_singular_value or max(residual) > maximum_plane_residual_degrees:
        raise ValueError('Inconsistent or degenerate views')
    return {'axis': axis, 'plane_residual_degrees': residual, 'pose_verified': False, 'limitation': 'Consistent lines may still mark wrong features; verify held-object identity and post-move behavior'}
