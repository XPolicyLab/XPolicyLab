"""Agent calculation helpers (world frame, metres); not a robot controller.

The harness configures current observation calibration automatically. Use camera_ray,
project_camera, pixel_scale and plane_pixel_scale with a camera NAME string. The
at_height and triangulate accept the NumPy origin/direction arrays returned by
camera_ray.
"""
import numpy as np
from scipy.optimize import least_squares


def at_height(origin, direction, z):
    """World point where a forward ray reaches z; reject parallel/backward intersections."""
    origin, direction = _vec3(origin, 'origin'), _vec3(direction, 'direction')
    z = _scalar(z, 'z (world metres)')
    if abs(direction[2]) < 1e-10:
        raise ValueError('Ray is parallel to the requested height plane')
    distance = (z - origin[2]) / direction[2]
    if distance < 0:
        raise ValueError('Height plane intersects behind the ray origin')
    return origin + distance * direction


def triangulate(p1, d1, p2, d2):
    """Midpoint of the closest approach of two rays and their miss distance."""
    p1, d1, p2, d2 = [_vec3(v, name) for v, name in zip((p1,d1,p2,d2), ('p1','d1','p2','d2'))]
    if np.linalg.norm(np.cross(d1, d2)) <= 1e-10*np.linalg.norm(d1)*np.linalg.norm(d2):
        raise ValueError('Triangulation needs two nonzero, nonparallel camera rays')
    s = np.linalg.lstsq(np.c_[d1, -d2], p2 - p1, rcond=None)[0]
    q1, q2 = p1 + s[0] * d1, p2 + s[1] * d2
    return (q1 + q2) / 2, float(np.linalg.norm(q1 - q2))


def fit_circle_px(xs, ys):
    """Least-squares circle (uc, vc, r) through boundary pixels, and its rms residual in px."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.ndim != 1 or xs.shape != ys.shape or len(xs) < 3 or not np.isfinite([xs, ys]).all():
        raise ValueError('fit_circle_px needs matching 1D x/y arrays with at least three finite pixels')
    x0 = [xs.mean(), ys.mean(), 0.25 * (np.ptp(xs) + np.ptp(ys))]
    res = least_squares(lambda q: np.hypot(xs - q[0], ys - q[1]) - q[2], x0)
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


# Camera helpers use the calibration attached to the observation.
import json
from scipy.spatial.transform import Rotation
_CALIBRATION = {}


def _numeric_array(value, shape, name):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError):
        result = None
    description = '3-vector [x,y,z]' if shape == (3,) else 'array of shape '+str(shape)
    if (result is None or result.shape != shape or result.dtype.kind not in 'iuf'
            or not np.isfinite(result).all()):
        raise ValueError(f'{name} must be a finite numeric {description}')
    return result.astype(float)


def _vec3(value, name):
    return _numeric_array(value, (3,), name)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f'{name} must be one finite number')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be one finite number') from None
    if not np.isfinite(result):
        raise ValueError(f'{name} must be one finite number')
    return result


def _camera(camera, observation_id=None):
    calibration = _CALIBRATION
    if observation_id is not None:
        # Keep the current calibration untouched when inspecting a historical frame.
        try:
            from . import observation_io as obs
        except ImportError:  # Helpers are also copied as standalone sandbox modules.
            import observation_io as obs
        calibration = obs.state(observation_id)['calibration']
    if not isinstance(camera, str) or camera not in calibration:
        raise ValueError('camera must be a configured name: '+', '.join(calibration))
    return calibration[camera]


def configure(observation_path):
    """Load observation.json, not its directory or manifest; normally done by the harness."""
    global _CALIBRATION
    with open(observation_path) as stream:
        _CALIBRATION = json.load(stream)['calibration']


def camera_ray(camera, u, v, *, observation_id=None):
    """Camera name + pixel -> (world origin, unnormalized world direction), two NumPy vectors.

    Uses USD/OpenGL calibration (+X right,+Y up,-Z forward). Omit observation_id for
    the latest observation, or pass the same ID used to load a historical image.
    """
    c = _camera(camera, observation_id)
    u, v = _scalar(u, 'u (pixel x)'), _scalar(v, 'v (pixel y)')
    K, T = np.asarray(c['K']), np.asarray(c['camera_to_world'])
    optical = np.linalg.solve(K, [u, v, 1.])
    gl = optical * [1, -1, -1]
    return T[:3, 3].copy(), T[:3, :3] @ gl


def project_camera(camera, world_point, *, observation_id=None):
    """Camera name + world point -> NumPy [u,v], or None for a point behind the camera."""
    c = _camera(camera, observation_id)
    K, T = np.asarray(c['K']), np.asarray(c['camera_to_world'])
    gl = T[:3, :3].T @ (_vec3(world_point, 'world_point')-T[:3, 3])
    optical = gl*[1, -1, -1]
    if optical[2] <= 0:
        return None
    pixel = K @ optical
    return pixel[:2]/pixel[2]


def pixel_scale(camera, world_point, *, observation_id=None):
    """Local frontoparallel mm/pixel at an estimated point, not a global constant."""
    c = _camera(camera, observation_id)
    K, T = np.asarray(c['K']), np.asarray(c['camera_to_world'])
    depth = -(T[:3, :3].T @ (_vec3(world_point, 'world_point')-T[:3, 3]))[2]
    if depth <= 0:
        raise ValueError('Point is behind the camera')
    return {'depth_m': float(depth), 'horizontal_mm_per_pixel': float(1000*depth/K[0,0]),
            'vertical_mm_per_pixel': float(1000*depth/K[1,1])}


def plane_pixel_scale(camera, u, v, surface_point, surface_normal, *, observation_id=None):
    """Metres per pixel on an observed plane, accounting for view foreshortening."""
    p, n = _vec3(surface_point, 'surface_point'), _vec3(surface_normal, 'surface_normal')
    if np.linalg.norm(n) < 1e-10:
        raise ValueError('surface_normal must be nonzero')
    u, v = _scalar(u, 'u (pixel x)'), _scalar(v, 'v (pixel y)')
    def hit(x, y):
        origin, direction = camera_ray(camera, x, y, observation_id=observation_id)
        denom = float(n @ direction)
        if abs(denom) < 1e-8:
            raise ValueError('Ray is parallel to plane')
        distance = float(n @ (p-origin))/denom
        if distance <= 0:
            raise ValueError('Plane is behind camera')
        return origin + distance*direction
    center = hit(u, v)
    return {'point': center, 'du_world_m': hit(u+1, v)-center, 'dv_world_m': hit(u, v+1)-center}


def _target(target):
    from copy import deepcopy
    if not isinstance(target, dict):
        raise ValueError('target must be a single-arm action dict with target_points and orientation_point')
    required = {'target_points', 'orientation_point'}
    if not required <= target.keys() or target.keys() - required - {'contact_constraint'}:
        raise ValueError('target needs target_points and orientation_point, with optional contact_constraint; '
                         'convert an observed arm with geom.target_from_state(arm_state)')
    result = deepcopy(target)
    result['target_points'] = _numeric_array(target['target_points'], (2, 3), 'target_points').tolist()
    result['orientation_point'] = _vec3(target['orientation_point'], 'orientation_point').tolist()
    return result


def target_from_state(arm_state):
    """Single observed arm -> action dict. Preserve point order; put o 0.1 m behind midpoint.

    Pass obs.state()['state']['left'] or ['right'], not the complete observation.
    Does not infer gripper commands or contact constraints.
    """
    if not isinstance(arm_state, dict) or not {'gripper_pose_points', 'tool_axis'} <= arm_state.keys():
        raise ValueError("arm_state must be obs.state()['state']['left'] or ['right']")
    points = _numeric_array(arm_state['gripper_pose_points'], (2, 3), 'gripper_pose_points')
    tool = _vec3(arm_state['tool_axis'], 'tool_axis')
    if np.linalg.norm(tool) < 1e-10:
        raise ValueError('tool_axis must be nonzero')
    return {'target_points': points.tolist(),
            'orientation_point': (points.mean(axis=0)-.1*tool/np.linalg.norm(tool)).tolist()}


def translate_points(target, displacement):
    """Action dict -> new dict; translate three points in world metres, keep other fields."""
    target = _target(target)
    d = _vec3(displacement, 'displacement')
    target['target_points'] = [(np.asarray(p)+d).tolist() for p in target['target_points']]
    target['orientation_point'] = (np.asarray(target['orientation_point'])+d).tolist()
    return target


def rotate_points(target, pivot, rotation_vector):
    """New action dict rotated about a world pivot; rotation_vector is axis-angle radians.

    Vector direction is the world rotation axis; its norm is the angle, not Euler angles.
    Point order and other fields, including any fixed-world contact plane, are retained.
    """
    target = _target(target)
    r, p = Rotation.from_rotvec(_vec3(rotation_vector, 'rotation_vector')), _vec3(pivot, 'pivot')
    target['target_points'] = [(p+r.apply(np.asarray(q)-p)).tolist() for q in target['target_points']]
    target['orientation_point'] = (p+r.apply(np.asarray(target['orientation_point'])-p)).tolist()
    return target


def grasp_reference_for(working_point, tool_axis, jaw_axis, region='front_tip'):
    """Measured robot offsets; visual object/working-point estimation remains the caller's job."""
    t=_vec3(tool_axis,'tool_axis')
    if np.linalg.norm(t) < 1e-10:
        raise ValueError('tool_axis must be nonzero')
    t=t/np.linalg.norm(t)
    j=_vec3(jaw_axis,'jaw_axis');j=j-t*np.dot(t,j)
    if np.linalg.norm(j) < 1e-10:
        raise ValueError('jaw_axis must have a component perpendicular to tool_axis')
    j=j/np.linalg.norm(j)
    lateral=np.cross(t,j)
    offsets={'front_tip':(.01257,-.000002,-.00014012),
             'distal_pad_bbox_center':(.00507,-.000002,-.00144044),
             'grasp_reference':(0.,0.,0.)}
    if region not in offsets: raise ValueError('Unknown robot working region')
    x,y,z=offsets[region]
    return _vec3(working_point,'working_point')-x*t-y*j-z*lateral


def projection_jacobian(camera, world_point, epsilon_m=.0001, *, observation_id=None):
    """2x3 local pixels/metre Jacobian at a supplied point, with the current camera fixed."""
    p=_vec3(world_point, 'world_point');columns=[]
    epsilon_m = _scalar(epsilon_m, 'epsilon_m')
    if epsilon_m <= 0:
        raise ValueError('epsilon_m must be positive')
    for axis in np.eye(3):
        hi=project_camera(camera,p+epsilon_m*axis,observation_id=observation_id)
        lo=project_camera(camera,p-epsilon_m*axis,observation_id=observation_id)
        if hi is None or lo is None: raise ValueError('Point behind camera')
        columns.append((hi-lo)/(2*epsilon_m))
    return np.column_stack(columns)
