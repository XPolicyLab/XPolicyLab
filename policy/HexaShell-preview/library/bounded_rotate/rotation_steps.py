"""orientation_steps, copied verbatim from tasks/plug_in_charger/tools/mate_pair/tool.py (the only function
bounded_rotate uses from that file), so the library does not carry the rest of that task tool."""
import numpy as np


def orientation_steps(start, goal, calibration, max_degrees, tolerance):
    """Bound TCP bow caused by straight end-link translation plus SLERP.

    For an offset radius r and angular increment a, departure from the ideal
    fixed-TCP rotation is <= r*(1-cos(a/2)). This is not a collision bound.
    """
    if max_degrees == 0:
        return [goal.copy()]
    relative = goal[:3, :3] @ start[:3, :3].T
    angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    if angle < 1e-8:
        return [goal.copy()]
    radius = float(np.linalg.norm(calibration[:3, 3]))
    limit = np.deg2rad(max_degrees)
    if radius > tolerance:
        limit = min(limit, 2 * np.arccos(1 - tolerance / radius))
    count = int(np.ceil(angle / limit))
    # The unit eigenvector with eigenvalue 1 is stable also at 180 degrees.
    _, vectors = np.linalg.eigh((relative + relative.T) / 2)
    axis = vectors[:, -1]
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    if axis @ skew < 0:
        axis = -axis
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    poses = []
    for fraction in np.arange(1, count + 1) / count:
        a = angle * fraction
        rotation = np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * (cross @ cross)
        pose = start.copy()
        pose[:3, :3] = rotation @ start[:3, :3]
        poses.append(pose)
    poses[-1] = goal.copy()
    return poses
