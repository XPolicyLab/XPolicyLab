import cv2
import numpy as np

from skills.shared.perception.public_camera_projection.implementation import points_on_plane, project_points
from skills.shared.perception.rgb_surface_geometry.implementation import triangulate


def coverage(api, camera, region, head_region, height, *, destination_camera='cam_head'):
    pixels = np.asarray(region['pixels'], dtype=float)
    indices = np.linspace(0, len(pixels) - 1, min(512, len(pixels))).astype(int)
    world = points_on_plane(api, camera, pixels[indices], height)
    valid = np.isfinite(world).all(axis=1)
    if valid.sum() < 16:
        return 0.0
    projected = project_points(api, destination_camera, world[valid])
    projected = projected[np.isfinite(projected).all(axis=1)]
    if not len(projected):
        return 0.0
    projected = np.rint(projected).astype(int)
    mask = cv2.dilate(head_region['mask'].astype(np.uint8), np.ones((5, 5), np.uint8))
    height_pixels, width_pixels = mask.shape
    inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width_pixels)
              & (projected[:, 1] >= 0) & (projected[:, 1] < height_pixels))
    points = projected[inside]
    return float(mask[points[:, 1], points[:, 0]].sum() / len(projected)) if inside.any() else 0.0


def centroid_surface(api, head_region, regions):
    candidates = []
    for camera in ('cam_left_wrist', 'cam_right_wrist'):
        for region in regions(api.image(camera)):
            stereo = triangulate(api, 'cam_head', head_region['centroid'], camera, region['centroid'])
            if stereo is None:
                continue
            fraction = coverage(api, camera, region, head_region, float(stereo['point'][2]))
            if fraction >= .84:
                candidates.append((dict(stereo, surface_coverage=fraction), camera, region))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0]['surface_coverage'], item[0]['ray_separation_m']))
    chosen = candidates[0][0]
    if any(abs(item[0]['point'][2] - chosen['point'][2]) > .008 for item in candidates[1:]):
        mutual = [item for item in candidates if coverage(api, 'cam_head', head_region, item[2],
            float(item[0]['point'][2]), destination_camera=item[1]) >= .84]
        if not mutual:
            return None
        chosen = mutual[0][0]
        if any(abs(item[0]['point'][2] - chosen['point'][2]) > .008 for item in mutual[1:]):
            return None
    return chosen
