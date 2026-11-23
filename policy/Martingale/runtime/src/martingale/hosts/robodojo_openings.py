# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

from . import robodojo_kinematics as kin


def detect(image):
    """Closed hole contours inside the blue top plate; clipped holes are omitted."""
    hsv = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([85, 55, 45]), np.array([130, 255, 255]))
    if np.count_nonzero(blue) > 100:
        cut, _ = cv2.threshold(hsv[:, :, 2][blue > 0], 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blue[hsv[:, :, 2] <= cut] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if count <= 1:
        return [], blue.astype(bool)
    plate = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype('uint8')
    contours, hierarchy = cv2.findContours(plate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = []
    if hierarchy is None:
        return holes, plate.astype(bool)
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] < 0 or len(contour) < 6 or cv2.contourArea(contour) < 9:
            continue
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        if min(axes) < 2 or max(axes) > 180 or min(axes)/max(axes) < .12:
            continue
        pixels = contour[:, 0].astype(float)
        delta = pixels - center
        pixels -= .5 * delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1)
        holes.append(dict(pixels=pixels, center=np.asarray(center), axes=np.asarray(axes)))
    holes.sort(key=lambda h: (h['center'][0], h['center'][1]))
    return holes, plate.astype(bool)


def on_plane(view, pixels, z):
    rays = view.rays(pixels[:, 1], pixels[:, 0])
    return view.pos + rays * ((z-view.pos[2])/rays[:, 2])[:, None]


def circle_on_plane(view, pixels, z):
    points = on_plane(view, pixels, z)[:, :2]
    origin = points.mean(0)
    xy = points-origin
    solution = np.linalg.lstsq(np.c_[2*xy, np.ones(len(xy))], (xy*xy).sum(1), rcond=None)[0]
    center = origin + solution[:2]
    radius = np.sqrt(max(0, solution[2]+solution[:2]@solution[:2]))
    residual = np.sqrt(np.mean((np.linalg.norm(points-center, axis=1)-radius)**2))
    return np.r_[center,z], float(radius), float(residual)


def fit(images, views):
    detections = {c:detect(image)[0] for c,image in images.items()}
    head = detections.get('cam_head', [])
    wrists = [c for c in detections if c != 'cam_head' and len(detections[c]) >= 2]
    if len(head) < 3 or not wrists:
        return dict(accepted=False, reasons=['Need clear closed hole rims in head and a wrist view.'])
    reference = {c:[circle_on_plane(views[c], h['pixels'], kin.TABLE_Z) for h in holes]
                 for c,holes in detections.items()}

    def at_height(camera,z):
        view = views[camera]
        scale = (z-view.pos[2])/(kin.TABLE_Z-view.pos[2])
        return [(np.r_[view.pos[:2]+(p[:2]-view.pos[:2])*scale,z],r*abs(scale),e*abs(scale))
                for p,r,e in reference[camera]]

    def evaluate(z, details=False):
        circles = at_height('cam_head',z)
        losses, matches = [], []
        for c in wrists:
            measured = at_height(c,z)
            if np.any(views[c].project(np.array([m[0] for m in measured]))[2] <= 0):
                losses.extend([.008]*len(measured))
                continue
            distance = np.array([[np.linalg.norm(a[0][:2]-b[0][:2]) for a in circles] for b in measured])
            radii = np.array([[abs(a[1]-b[1]) for a in circles] for b in measured])
            cost = distance + radii
            ii,jj = linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                losses.append(min(float(cost[i,j]), .008))
                matches.append((c, int(i), int(j), float(cost[i,j])))
        value = float(np.mean(np.square(losses))) if losses else 1.
        return (value,circles,matches) if details else value

    heights = np.linspace(kin.TABLE_Z+.015, kin.TABLE_Z+.30, 191)
    costs = np.asarray([evaluate(z) for z in heights])
    candidates = [i for i in range(1,len(heights)-1)
                  if costs[i] <= min(costs[i-1],costs[i+1]) and costs[i] < max(costs[i-1],costs[i+1])]
    if not candidates:
        return dict(accepted=False,reasons=['No bounded shared-plane solution.'])
    solutions = [minimize_scalar(evaluate,bounds=(heights[i-1],heights[i+1]),method='bounded',
                                options={'xatol':1e-6}) for i in candidates]
    solution = min(solutions,key=lambda s:s.fun)
    z = float(solution.x)
    error,circles,matches = evaluate(z,True)
    good = [m for m in matches if m[3] < .006]
    reasons = []
    if len(good) < 3:
        reasons.append('Fewer than three consistent cross-view hole matches.')
    if len(good) < .6 * len(matches):
        reasons.append('Too many hole rims disagree; obtain a clearer wrist view.')
    if np.sqrt(error) > .006:
        reasons.append('Hole correspondences disagree across views.')
    alternatives = [s for s in solutions if abs(s.x-z) > .01 and s.fun < error*2 + 1e-6]
    if alternatives:
        reasons.append('Repeated hole pattern leaves height ambiguous; get another view.')
    center = np.mean([c[0] for c in circles],axis=0)
    directions = {c:(view.pos-center)/np.linalg.norm(view.pos-center) for c,view in views.items()}
    parallax = max([float(np.rad2deg(np.arccos(np.clip(directions['cam_head']@directions[c],-1,1))))
                    for c in {m[0] for m in good}] or [0.])
    if parallax < 10:
        reasons.append('Insufficient viewpoint separation; move the wrist sideways.')
    holes = []
    for j,(center,radius,circular_error) in enumerate(circles):
        support = [m for m in good if m[2] == j]
        holes.append(dict(id=j,center_xyz=center.tolist(),radius_m=radius,
                          circular_residual_m=circular_error,head_pixel_xy=head[j]['center'].tolist(),
                          supporting_views=['cam_head']+[m[0] for m in support],
                          directly_matched=bool(support)))
    return dict(accepted=not reasons,reasons=reasons,rim_z=z,normal_xyz=[0.,0.,1.],
                view_separation_deg=parallax,
                holes=holes,fit_error_m=float(np.sqrt(error)),matched_holes=len(good),
                matches=[dict(camera=c,wrist_hole=i,head_hole=j,error_m=e) for c,i,j,e in matches],
                detections_per_view={c:len(h) for c,h in detections.items()},
                note='Horizontal rim plane and visible empty openings only; no tube pose or seating depth measured.')


def current_observation(primitives):
    from . import robodojo_tools as tools
    obs = tools._obs(primitives)
    state = tools._shown(obs)
    images = {c:np.asarray(cam['color']) for c,cam in obs['vision'].items()
              if c in ('cam_head','cam_left_wrist','cam_right_wrist') and cam.get('color') is not None}
    return images, {c:tools._view(c,state) for c in images}


def measure(primitives):
    images, views = current_observation(primitives)
    result = fit(images,views)
    result['episode_action_step'] = int(primitives.env.get_status().get('step',0))
    return result


def triangulate(pixels, views):
    """A named physical landmark, observed at (column,row) in several cameras."""
    rows, rhs, cameras = [], [], []
    for camera, pixel in pixels.items():
        if camera not in views:
            raise ValueError('Unknown camera: '+camera)
        p = np.asarray(pixel,float)
        if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or p[0]>=640 or p[1]>=480:
            raise ValueError('Landmarks must be finite original-image (column,row) pixels.')
        view = views[camera]
        ray = view.rays(np.array([p[1]]),np.array([p[0]]))[0]
        ray /= np.linalg.norm(ray)
        projection = np.eye(3)-np.outer(ray,ray)
        rows.append(projection); rhs.append(projection@view.pos); cameras.append(camera)
    if len(cameras) < 2:
        raise ValueError('Each tube landmark needs two camera observations.')
    xyz = np.linalg.lstsq(np.concatenate(rows),np.concatenate(rhs),rcond=None)[0]
    errors, directions = [], []
    for c in cameras:
        r,col,depth = views[c].project(xyz[None])
        if depth[0] <= 0:
            raise ValueError('Triangulated point is behind a camera.')
        errors.extend(np.array([col[0],r[0]])-pixels[c])
        direction = views[c].pos-xyz
        directions.append(direction/np.linalg.norm(direction))
    angle = max(float(np.rad2deg(np.arccos(np.clip(a@b,-1,1)))) for a in directions for b in directions)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    if angle < 10 or rms > 2:
        raise ValueError('Tube landmarks disagree or lack parallax; get a clearer supporting view.')
    return xyz,rms


def alignment(primitives, openings, hole_id, tip_pixels, shaft_pixels, tube_radius_m,
              clearance_m=.002, insertion_m=.03):
    """Read-only alignment check for the actual held tube, not its gripper TCP."""
    if not openings.get('accepted'):
        return dict(ready_to_lower=False,reasons=['Opening fit was not accepted.'])
    radius,clearance,depth = float(tube_radius_m),float(clearance_m),float(insertion_m)
    if not (0 < radius < .1 and 0 <= clearance <= .02 and 0 <= depth <= .2):
        raise ValueError('Invalid tube radius, clearance or planned insertion distance.')
    hole = next((h for h in openings['holes'] if h['id']==hole_id),None)
    if hole is None:
        raise ValueError('hole_id is absent from this opening fit.')
    from . import robodojo_tools as tools
    obs = tools._obs(primitives)
    shown = tools._shown(obs)
    for arm in ('left','right'):
        before = np.asarray(shown[f'{arm}_ee_pose'],float)
        now = np.asarray(obs['state'][f'{arm}_ee_pose'],float)
        if (np.linalg.norm(before[:3]-now[:3]) > .002 or
            kin.rotation_error(kin.rot_from_quat(before[3:]),kin.rot_from_quat(now[3:])) > 1):
            return dict(ready_to_lower=False,reasons=[
                'Image predates the final hand motion. Hold the current pose to settle, then reobserve.'])
    _,views = current_observation(primitives)
    try:
        tip,tip_error = triangulate(tip_pixels,views)
        shaft,shaft_error = triangulate(shaft_pixels,views)
    except ValueError as exc:
        return dict(ready_to_lower=False,reasons=[str(exc)])
    axis = shaft-tip
    if np.linalg.norm(axis) < .025:
        return dict(ready_to_lower=False,reasons=['Choose shaft landmark at least 25 mm above the tip.'])
    axis /= np.linalg.norm(axis)
    normal = np.array(openings['normal_xyz'],float)
    cosine = float(axis@normal)
    tilt = float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
    center = np.asarray(hole['center_xyz'])
    offset = float(np.linalg.norm(tip[:2]-center[:2]))
    occupied = radius/max(.01,cosine) + depth*np.tan(np.deg2rad(min(tilt,89.)))
    margin = float(hole['radius_m']-clearance-offset-occupied)
    above = float(tip[2]-center[2])
    reasons = []
    if tilt > 5:
        reasons.append('Rotate the held tube upright while clear, then remeasure its tip.')
    if margin < 0:
        reasons.append('Tube envelope does not clear this hole; recenter or choose a larger hole.')
    if above < .005:
        reasons.append('Tip is already at/below the entry zone; do not correct sideways through the rim.')
    target = center+normal*.015
    return dict(ready_to_lower=not reasons,reasons=reasons,tip_xyz=tip.tolist(),
                tube_axis_xyz=axis.tolist(),tilt_deg=tilt,lateral_error_m=offset,
                remaining_clearance_m=margin,tip_above_rim_m=above,
                target_tip_xyz=target.tolist(),translation_to_entry_m=(target-tip).tolist(),
                landmark_reprojection_rms_px=max(tip_error,shaft_error),
                note='Translation is valid only while holding the same orientation. Remeasure after rotation. '
                     'This checks entry geometry, not contact, obstruction or successful seating.')
