"""Acquire another RGB view using the observed wrist mount, without object dimensions."""
import numpy as np
from scipy.spatial.transform import Rotation
from . import Camera, GeometryUnavailable


def wrist_cover_ray(observation, side, pixel, robot_base, robot_reach):
    """Cover the head ray inside a public robot reach bound, without assigning depth.

    The reach sphere is a loose search bound, not an assertion of IK feasibility.
    The returned segment must never be used as an object pose.
    """
    head = Camera.from_observation(observation)
    base = np.asarray(robot_base, float)
    if (base.shape != (3,) or not np.isfinite(base).all()
            or not np.isfinite(robot_reach) or robot_reach <= 0):
        raise GeometryUnavailable('Observed robot base and public reach bound required')
    origin = head.transform[:3, 3]
    ray = head.unproject([pixel], [1.])[0] - origin
    ray /= np.linalg.norm(ray)
    delta = base - origin
    middle = ray @ delta
    discriminant = robot_reach**2 - (delta @ delta - middle**2)
    if discriminant <= 0:
        raise GeometryUnavailable('Head feature ray misses robot search volume')
    lower, upper = max(1e-6, middle - np.sqrt(discriminant)), middle + np.sqrt(discriminant)
    if upper <= lower:
        raise GeometryUnavailable('Robot search volume is behind head camera')
    points = origin + np.linspace(lower, upper, 25)[:, None] * ray
    wrist = Camera.from_observation(observation, f'cam_{side}_wrist')
    choices = []
    for fraction in (.25, .5, .75):
        target = origin + (lower + fraction * (upper - lower)) * ray
        try:
            result = wrist_look_at(observation, side, target)
            predicted = Camera(wrist.image, wrist.k, result['camera_transform'])
            camera_points = (points-predicted.transform[:3, 3]) @ predicted.transform[:3, :3]
            front = camera_points[:, 2] < -1e-6
            uv = np.full((len(points), 2), np.inf)
            if front.any():
                uv[front], _ = predicted.project(points[front])
        except GeometryUnavailable:
            continue
        visible = ((uv[:, 0] >= .05*wrist.width) & (uv[:, 0] <= .95*wrist.width)
                   & (uv[:, 1] >= .05*wrist.height) & (uv[:, 1] <= .95*wrist.height))
        coverage = float(visible.mean())
        result.update(ray_coverage=coverage, search_segment=[points[0].tolist(), points[-1].tolist()],
                      source='head_feature_ray_and_public_robot_reach')
        choices.append((coverage, -result['rotation_radians'], result))
    if not choices:
        raise GeometryUnavailable('No bounded inspection orientation covers feature ray')
    return max(choices, key=lambda item: item[:2])[2]


def wrist_look_at(observation, side, proposal, *, maximum_rotation=1.2):
    """Aim a wrist camera at a *proposal*, keeping the observed EE position fixed.

    This is an observation pose, not a grasp estimate or a collision certificate.
    The caller must check reachability/collisions and measure the object again.
    Camera offset from the EE is included as it rotates with the hand.
    """
    if side not in ('left', 'right'):
        raise GeometryUnavailable('Unknown wrist')
    target = np.asarray(proposal, float)
    ee = np.asarray(observation['state'][f'{side}_ee_pose'], float)
    if (target.shape != (3,) or ee.shape != (7,)
            or not np.isfinite(np.r_[target, ee, maximum_rotation]).all()
            or not 0 < maximum_rotation < np.pi or np.linalg.norm(ee[3:]) < 1e-8):
        raise GeometryUnavailable('Finite target, observed EE, and rotation bound required')
    wrist = Camera.from_observation(observation, f'cam_{side}_wrist')
    initial = Rotation.from_quat(ee[[4, 5, 6, 3]]).as_matrix()
    mount = initial.T @ wrist.transform[:3, :3]
    offset = initial.T @ (wrist.transform[:3, 3] - ee[:3])
    rotation = initial.copy()
    for _ in range(32):
        origin = ee[:3] + rotation @ offset
        direction = target - origin
        distance = np.linalg.norm(direction)
        if distance <= np.linalg.norm(offset) + 1e-6:
            raise GeometryUnavailable('Proposal too close to rotating wrist')
        direction /= distance
        forward = -(rotation @ mount)[:, 2]
        cross = np.cross(forward, direction)
        sine = np.linalg.norm(cross)
        cosine = float(np.clip(forward @ direction, -1, 1))
        if sine < 1e-10:
            if cosine < 0:
                raise GeometryUnavailable('Ambiguous opposite viewing direction')
            break
        correction = Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()
        rotation = correction @ rotation
    angle = float(Rotation.from_matrix(rotation @ initial.T).magnitude())
    if angle > maximum_rotation:
        raise GeometryUnavailable('Required wrist rotation exceeds inspection bound')
    transform = wrist.transform.copy()
    transform[:3, :3] = rotation @ mount
    transform[:3, 3] = ee[:3] + rotation @ offset
    predicted = Camera(wrist.image, wrist.k, transform)
    pixel = predicted.project([target])[0][0]
    principal = wrist.k[:2, 2]
    if np.linalg.norm(pixel - principal) > 1:
        raise GeometryUnavailable('Wrist aim did not converge')
    return {'pose': np.r_[ee[:3], Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]].tolist(),
            'camera_transform': transform, 'proposal_pixel': pixel.tolist(),
            'rotation_radians': angle, 'source': 'observed_wrist_mount_and_visual_proposal'}
