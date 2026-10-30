"""Unchanged selected Hexa RGB evidence functions; privately adapted camera input."""
import cv2
import numpy as np

SOURCES = {"head": "cam_head", "wrist_l": "cam_left_wrist", "wrist_r": "cam_right_wrist"}

def camera_data(obs, name):
    source = SOURCES[name]
    image = cv2.imdecode(np.frombuffer(obs['png'][source], np.uint8), cv2.IMREAD_COLOR)
    cam = obs['cameras'][source]
    k = np.asarray(cam['intrinsics'], float)
    t = np.asarray(cam['extrinsics_world'], float)
    if (image is None or k.shape != (3, 3) or t.shape != (4, 4)
            or not np.isfinite(k).all() or not np.isfinite(t).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-3)):
        raise ValueError('invalid RGB image or calibration')
    return image, k, t

def rays(obs, name, pixels):
    image, k, t = camera_data(obs, name)
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    if (not np.isfinite(pixels).all() or np.any(pixels < 0)
            or np.any(pixels[:, 0] >= image.shape[1]) or np.any(pixels[:, 1] >= image.shape[0])):
        raise ValueError('pixel outside image')
    directions = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(k).T @ t[:3, :3].T
    return t[:3, 3], directions

def triangulate(obs, name, u, v, other, mu, mv):
    a, da = rays(obs, name, [[u, v]])
    b, db = rays(obs, other, [[mu, mv]])
    da, db = da[0] / np.linalg.norm(da), db[0] / np.linalg.norm(db)
    if np.linalg.norm(np.cross(da, db)) < .025:
        raise ValueError('insufficient triangulation angle')
    scales = np.linalg.lstsq(np.column_stack([da, -db]), b-a, rcond=None)[0]
    p, q = a + scales[0]*da, b + scales[1]*db
    if min(scales) <= 0 or np.linalg.norm(p-q) > .004:
        raise ValueError('inconsistent point correspondence')
    return (p+q)/2

def sift_source_evidence(before, after, point, floor):
    """Negative retention evidence: tracked material still low in two live views.

    No evidence of absence is inferred from failed matching or occlusion.
    Thresholds are conservative design tolerances, not scene dimensions.
    """
    unknown = dict(status='unknown', retention_verified=False)
    try:
        point = np.asarray(point, float)
        if point.shape != (3,) or point[2] < floor + .010:
            return unknown
        old, k, t = camera_data(before, 'head')
        local = t[:3, :3].T @ (point - t[:3, 3])
        if local[2] <= 0:
            return unknown
        seed = (k @ local)[:2] / local[2]
        sift = cv2.SIFT_create()
        ka, da = sift.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
        current, _, _ = camera_data(after, 'head')
        kb, db = sift.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
        if da is None or db is None:
            return unknown

        def mutual(a, b):
            matcher = cv2.BFMatcher()
            def pairs(x, y):
                return {row[0].queryIdx: row[0].trainIdx
                        for row in matcher.knnMatch(x, y, k=2)
                        if len(row) == 2 and row[0].distance < .65 * row[1].distance}
            forward, reverse = pairs(a, b), pairs(b, a)
            return {i: j for i, j in forward.items() if reverse.get(j) == i}

        tracks = {i: j for i, j in mutual(da, db).items()
                  if np.linalg.norm(np.asarray(ka[i].pt) - seed) <= 12}
        if len(tracks) < 3:
            return unknown
        confirmed = {}
        for camera in ('wrist_l', 'wrist_r'):
            try:
                image, _, _ = camera_data(after, camera)
                kc, dc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
                if dc is None:
                    continue
                matches = mutual(db, dc)
                for i, j in tracks.items():
                    if j not in matches:
                        continue
                    uv, other = kb[j].pt, kc[matches[j]].pt
                    try:
                        world = triangulate(after, 'head', *uv, camera, *other)
                    except ValueError:
                        continue
                    if (world[2] > floor + .006 and abs(world[2] - point[2]) < .015
                            and np.linalg.norm(world[:2] - point[:2]) <= .05):
                        confirmed[i] = world
            except (KeyError, ValueError):
                continue
        # SIFT may give multiple orientations at one physical location.
        unique = []
        for i, world in confirmed.items():
            if all(np.linalg.norm(np.asarray(ka[i].pt)-ka[j].pt) >= 3 for j in unique):
                unique.append(i)
        if len(unique) < 3:
            return unknown
        pixels = np.array([ka[i].pt for i in unique])
        if np.linalg.norm(pixels.mean(axis=0)-seed) > 6:
            return unknown
        return dict(status='source_material_remains', retention_verified=False,
                    matched_points=len(unique),
                    source_world=np.median([confirmed[i] for i in unique], axis=0).tolist())
    except Exception:
        return unknown
