"""Project only public robot mesh and housing geometry into observed RGB."""
import json
from pathlib import Path
import cv2
import numpy as np
from coap.perception import WRIST_HOUSING_BOUNDS
_ROBOT_GEOMETRY=None


def robot_mask(ctx, image, camera):
    """Project the public arm meshes, separate fingers, and camera housings."""
    global _ROBOT_GEOMETRY
    if _ROBOT_GEOMETRY is None:
        path = Path(__file__).with_name('data')/'classify_r03_robot_geometry.json'
        _ROBOT_GEOMETRY = {name: np.asarray(vertices, float)
                           for name, vertices in json.loads(path.read_text()).items()}
    kinematics = getattr(ctx, 'kinematics', None)
    if kinematics is None:
        from coap.kinematics import X5Kinematics
        kinematics = X5Kinematics(ctx.initial_state)
    origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
    mask = np.zeros(image.shape[:2], np.uint8)

    def fill(points):
        local = np.einsum('ni,ij->nj', points-origin, rotation)
        uv = np.column_stack((image.shape[1]/2-fx*local[:, 0]/local[:, 2],
                              image.shape[0]/2+fy*local[:, 1]/local[:, 2]))
        hull = cv2.convexHull(np.round(np.clip(uv, -3000, 3000)).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)

    for arm in ('left', 'right'):
        frames = kinematics.forward_links(arm, ctx.state[f'{arm}_arm_joint_state'])
        grip = float(np.asarray(ctx.state.get(f'{arm}_ee_joint_state', [1.])).reshape(-1)[0])
        for name, vertices in _ROBOT_GEOMETRY.items():
            frame = frames[0 if name == 'base_link' else min(int(name[4:]), 6)]
            local = vertices.copy()
            if name in ('link7', 'link8'):
                side = 1. if name == 'link7' else -1.
                local += [.08657, side*(.0249+.044*grip), -.0002436]
            points = np.einsum('ni,ji->nj', local, frame[:3, :3])+frame[:3, 3]
            fill(points)
        frame = frames[6]
        for lower, upper in WRIST_HOUSING_BOUNDS:
            corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                                for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
            fill(np.einsum('ni,ji->nj', corners, frame[:3, :3])+frame[:3, 3])
    return mask
