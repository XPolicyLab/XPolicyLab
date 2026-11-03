"""Select live wrist calibration by the matching observed end-effector pose."""
import numpy as np
from . import Camera, GeometryUnavailable


def current_wrist_camera(observation, ee_pose):
    pose = np.asarray(ee_pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise GeometryUnavailable('A finite observed end-effector pose is required')
    matches = []
    for arm in ('left', 'right'):
        try:
            current = np.asarray(observation['state'][arm + '_ee_pose'], float)
        except (KeyError, TypeError) as error:
            raise GeometryUnavailable('Current wrist proprioception is required') from error
        if current.shape != (7,) or not np.isfinite(current).all() or np.linalg.norm(current[3:]) < 1e-8:
            raise GeometryUnavailable('Invalid current wrist proprioception')
        dot = abs(np.dot(pose[3:], current[3:]) / (np.linalg.norm(pose[3:]) * np.linalg.norm(current[3:])))
        if np.linalg.norm(pose[:3]-current[:3]) < 1e-5 and 1-min(dot, 1.) < 1e-8:
            matches.append(arm)
    if len(matches) != 1:
        raise GeometryUnavailable('Wrist pose is stale or does not uniquely identify a current camera')
    return Camera.from_observation(observation, 'cam_' + matches[0] + '_wrist')


def ee_relative_camera(observation, ee_pose):
    """Return current USD camera pose in the observed EE frame and live K."""
    from scipy.spatial.transform import Rotation
    camera = current_wrist_camera(observation, ee_pose)
    pose = np.asarray(ee_pose, float)
    ee = np.eye(4)
    ee[:3, :3] = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
    ee[:3, 3] = pose[:3]
    return np.linalg.solve(ee, camera.transform), camera.k.copy()
