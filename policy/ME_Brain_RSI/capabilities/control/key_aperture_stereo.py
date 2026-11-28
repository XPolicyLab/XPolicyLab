"""Cross-check a current aperture using two public calibrated RGB views."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def measured_stereo_aperture(ctx, arm, prior):
    """Use a fresh body anchor and require a unique independent two-view fit."""
    from coap.tasks import precision as p
    cameras = ['cam_head', f'cam_{arm}_wrist']
    if any(camera not in ctx.vision for camera in cameras):
        return None
    slot = p._best(ctx, 'key_slot')
    if slot is None:
        return None
    expected = np.r_[np.asarray(slot['position'], float)[:2], ctx.table_z+.055]
    prior = np.asarray(prior, float).reshape(3)
    if (not np.isfinite(expected).all() or not np.isfinite(prior).all()
            or np.linalg.norm(expected[:2]-prior[:2])>.060):
        return None
    asset = json.loads((Path(__file__).resolve().parents[1]/
                        'data/key_slot_aperture.json').read_text())
    vertices = np.asarray(asset['polygon_xy'])-asset['circle_center_xy']
    polygon = []
    for point, delta in zip(vertices, np.roll(vertices, -1, axis=0)-vertices):
        samples = max(2, int(np.linalg.norm(delta)/.00015))
        polygon.extend(point+fraction*delta
                       for fraction in np.linspace(0., 1., samples, endpoint=False))
    polygon = np.asarray(polygon)

    def project(parameters, camera):
        x, y, z, yaw = parameters
        cosine, sine = np.cos(yaw), np.sin(yaw)
        world = np.c_[cosine*polygon[:, 0]-sine*polygon[:, 1]+x,
                      sine*polygon[:, 0]+cosine*polygon[:, 1]+y,
                      np.full(len(polygon), z)]
        image = ctx.vision[camera]['color']
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        local = np.einsum('ij,nj->ni', rotation.T, world-origin)
        if not np.isfinite(local).all() or np.any(local[:, 2]>=-.005):
            return None
        height, width = image.shape[:2]
        return np.c_[width/2-fx*local[:, 0]/local[:, 2],
                     height/2+fy*local[:, 1]/local[:, 2]]

    fits = []
    for threshold in (40, 60, 80, 100, 120, 140):
        views = []
        for camera in cameras:
            image = ctx.vision[camera]['color']
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            predicted = ctx.perception.world_to_pixel(expected, camera, ctx.state, image.shape)
            rows, columns = np.indices(image.shape[:2])
            search = (columns-predicted[0])**2+(rows-predicted[1])**2<70**2
            mask = ((hsv[..., 2]<threshold)&search).astype(np.uint8)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            candidates = []
            for contour in contours:
                if not 3<cv2.contourArea(contour)<2500:
                    continue
                moments = cv2.moments(contour)
                center = np.array([moments['m10']/moments['m00'],
                                   moments['m01']/moments['m00']])
                distance = np.linalg.norm(center-predicted)
                if distance<70:
                    candidates.append((distance, contour[:, 0].astype(float)))
            if not candidates:
                break
            _, points = min(candidates, key=lambda candidate: candidate[0])
            views.append((camera, points, cKDTree(points)))
        if len(views)!=2:
            continue

        def parts(parameters):
            errors = []
            for camera, points, tree in views:
                pixels = project(parameters, camera)
                if pixels is None:
                    errors.append(np.full(len(polygon)+len(points), 1e4))
                else:
                    errors.append(np.r_[tree.query(pixels)[0],
                                        cKDTree(pixels).query(points)[0]])
            return errors

        def residual(parameters):
            # Give each independent camera equal weight despite different pixel scales.
            return np.concatenate([values/np.sqrt(len(values)) for values in parts(parameters)])

        for height in (ctx.table_z+.049, ctx.table_z+.055):
            for yaw in np.linspace(-math.pi, math.pi, 9)[:-1]:
                fit = least_squares(residual, [*expected[:2], height, yaw],
                    bounds=(np.r_[expected[:2]-.025, ctx.table_z+.030, yaw-1.],
                            np.r_[expected[:2]+.025, ctx.table_z+.065, yaw+1.]),
                    max_nfev=250, diff_step=1e-4)
                errors = [float(np.sqrt(np.mean(values**2))) for values in parts(fit.x)]
                if not np.isfinite(fit.x).all() or not np.isfinite(errors).all():
                    continue
                fits.append({'xyz_yaw':fit.x.tolist(), 'per_camera_rms':errors,
                             'rms':float(np.sqrt(np.mean(np.asarray(errors)**2))),
                             'threshold':threshold,
                             'contour_points':[len(view[1]) for view in views]})
    if not fits:
        return None
    fits.sort(key=lambda row: row['rms'])
    best = fits[0]
    parameters = np.asarray(best['xyz_yaw'])
    competitors = [row for row in fits
                   if abs(math.atan2(math.sin(row['xyz_yaw'][3]-parameters[3]),
                                     math.cos(row['xyz_yaw'][3]-parameters[3])))>.20
                   or np.linalg.norm(np.asarray(row['xyz_yaw'])[:2]-parameters[:2])>.002]
    gap = min((row['rms']-best['rms'] for row in competitors), default=float('inf'))
    agreeing = [row for row in fits if row['rms']<=best['rms']+.5]
    centers = np.asarray([row['xyz_yaw'][:3] for row in agreeing])
    spread = float(np.max(np.linalg.norm(centers-centers.mean(axis=0), axis=1)))
    ctx.memory['keyhole_stereo_fit_review'] = {
        'camera_views':cameras, 'fresh_anchor':expected, 'prior_anchor':np.asarray(prior),
        'best':best, 'runner_gap_pixels':gap, 'center_spread_m':spread,
        'candidates':fits[:12]}
    if (max(best['per_camera_rms'])>=1. or gap<.5 or spread>.0015
            or len(agreeing)<2 or np.linalg.norm(parameters[:2]-expected[:2])>.025):
        return None
    return {'position':parameters[:3].copy(), 'slit_yaw':float(parameters[3]),
            'camera':cameras[1], 'camera_views':cameras,
            'measurement_step':int(ctx.steps), 'geometry_source':asset['source'],
            'rms_pixels':best['rms'], 'per_camera_rms_pixels':best['per_camera_rms'],
            'runner_gap_pixels':gap, 'center_spread_m':spread,
            'threshold':best['threshold'], 'contour_points':best['contour_points'],
            'public_circle_radius_m':asset['circle_radius_m'], 'fresh_outer_anchor':True}
