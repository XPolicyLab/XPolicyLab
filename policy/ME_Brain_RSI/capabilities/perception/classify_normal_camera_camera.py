"""Refine only camera XY contact points toward visible wrist-RGB body support."""
import cv2
import numpy as np

def refine_camera_point(ctx, arm, point, quaternion):
    """Soft-rank material interior points while retaining contact height and orientation."""
    point = np.asarray(point, float).copy()
    camera = 'cam_' + arm + '_wrist'
    image = np.asarray(ctx.vision[camera]['color'], np.uint8)[..., :3]
    height, width = image.shape[:2]
    perception = ctx.perception
    nominal = perception.world_to_pixel(point, camera, ctx.state, image.shape)
    corners = np.array([point + [x, y, 0.] for x, y in
                        [(-.075, -.075), (.075, -.075), (.075, .075), (-.075, .075)]])
    pixels = np.array([perception.world_to_pixel(corner, camera, ctx.state, image.shape)
                       for corner in corners])
    region = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(region, np.round(np.clip(pixels, -3000, 3000)).astype(np.int32), 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    material = ((hsv[..., 1] < 110) & (hsv[..., 2] < 135) & (region > 0)).astype(np.uint8)
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(material, cv2.DIST_L2, 5)
    rows, columns = np.nonzero(material)
    evidence = dict(method='wrist_camera_material_interior_soft_point_ranking',
                    camera=camera, nominal_point=point.tolist(), nominal_pixel=nominal.tolist(),
                    contact_height_unchanged=True, grasp_orientation_unchanged=True)
    if not len(rows):
        evidence.update(selected_point=point.tolist(), selected_pixel=nominal.tolist(),
                        material_pixels=0, displacement_m=0., nominal_clearance_m=0.,
                        selected_clearance_m=0.)
        return point, evidence
    origin, rotation, fx, fy = perception.camera_geometry(camera, ctx.state, image.shape)
    rays = np.einsum('ni,ji->nj', np.column_stack(((columns - width / 2.) / fx,
                     -(rows - height / 2.) / fy, -np.ones(len(rows)))), rotation)
    scales = (point[2] - origin[2]) / rays[:, 2]
    positions = origin + scales[:, None] * rays
    first = perception.project(nominal + [1., 0.], point[2], camera, ctx.state, image.shape)
    second = perception.project(nominal + [0., 1.], point[2], camera, ctx.state, image.shape)
    pixel_scale = .5 * (np.linalg.norm(first[:2] - point[:2])
                       + np.linalg.norm(second[:2] - point[:2]))
    clearances = distance[rows, columns] * pixel_scale
    moves = np.linalg.norm(positions[:, :2] - point[:2], axis=1)
    scores = moves ** 2 + .000064 * np.logaddexp(0., (.008 - clearances) / .002) ** 2
    local = np.rint(nominal).astype(int)
    nominal_clearance = float(distance[np.clip(local[1], 0, height - 1),
                                         np.clip(local[0], 0, width - 1)] * pixel_scale)
    nominal_score = float(.000064 * np.logaddexp(0., (.008 - nominal_clearance) / .002) ** 2)
    chosen = int(np.argmin(scores))
    # The unchanged point participates in the same soft ranking as RGB measurements.
    if float(scores[chosen]) < nominal_score:
        selected = positions[chosen].copy()
        selected[2] = point[2]
        selected_pixel = [int(columns[chosen]), int(rows[chosen])]
        selected_clearance = float(clearances[chosen])
        selected_score = float(scores[chosen])
    else:
        selected = point.copy()
        selected_pixel = nominal.tolist()
        selected_clearance = nominal_clearance
        selected_score = nominal_score
    evidence.update(material_pixels=int(len(rows)), selected_point=selected.tolist(),
        selected_pixel=selected_pixel, nominal_clearance_m=nominal_clearance,
        selected_clearance_m=selected_clearance, displacement_m=float(np.linalg.norm(selected - point)),
        nominal_soft_cost=nominal_score, selected_soft_cost=selected_score)
    return selected, evidence
