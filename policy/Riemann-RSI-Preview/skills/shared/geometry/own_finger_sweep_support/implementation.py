"""Pure support of declared own finger geometry, without measured jaw state."""
import numpy as np

from skills.shared.geometry.native_rigid_pose.implementation import quat_to_mat_wxyz


def minimum_z(fingers, wrist_pose):
    wrist = np.asarray(wrist_pose, dtype=float)
    if (wrist.shape != (7,) or not np.isfinite(wrist).all()
            or abs(float(np.linalg.norm(wrist[3:])) - 1.) > 1e-3):
        raise ValueError('A finite unit-quaternion public own wrist pose is required')
    if len(fingers) != 2 or {f['link'] for f in fingers} != {'link7', 'link8'}:
        raise ValueError('Declare exactly both own X5 finger links')
    rotation = quat_to_mat_wxyz(wrist[3:])
    result = float('inf')
    for finger in fingers:
        origin = np.asarray(finger['origin'], dtype=float)
        axis = np.asarray(finger['axis'], dtype=float)
        limits = np.asarray(finger['limits'], dtype=float)
        vertices = np.asarray(finger['vertices'], dtype=float)
        if (origin.shape != (4, 4) or axis.shape != (3,) or limits.shape != (2,)
                or vertices.ndim != 2 or vertices.shape[1] != 3 or not 4 <= len(vertices) <= 512
                or not all(np.isfinite(x).all() for x in (origin, axis, limits, vertices))
                or not np.allclose(origin[3], [0., 0., 0., 1.])
                or not np.allclose(origin[:3, :3].T @ origin[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(origin[:3, :3]), 1., atol=1e-6)
                or not np.isclose(np.linalg.norm(axis), 1.) or limits[0] >= limits[1]):
            raise ValueError('Invalid bounded static prismatic finger geometry')
        for opening in limits:
            local = (vertices + axis * opening) @ origin[:3, :3].T + origin[:3, 3]
            world = local @ rotation.T + wrist[:3]
            result = min(result, float(world[:, 2].min()))
    return result
