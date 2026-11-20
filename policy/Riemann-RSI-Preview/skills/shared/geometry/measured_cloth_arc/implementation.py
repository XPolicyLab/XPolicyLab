import math

import numpy as np


def fold_arc_wrist(start, displacement, angle):
    start = np.asarray(start, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if (start.shape != (4, 4) or displacement.shape not in ((2,), (3,))
            or not np.isfinite(start).all() or not np.isfinite(displacement).all()
            or isinstance(angle, bool) or not math.isfinite(angle) or not 0 <= angle <= math.pi
            or not np.allclose(start[3], [0., 0., 0., 1.])
            or not np.allclose(start[:3, :3].T @ start[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(start[:3, :3])), 1., abs_tol=1e-6)):
        raise ValueError('A cloth arc needs a finite own wrist frame and current visible displacement')
    displacement = displacement[:2]
    radius = .5 * float(np.linalg.norm(displacement))
    target = start.copy()
    target[:2, 3] += .5 * (1. - np.cos(angle)) * displacement
    target[2, 3] += .012 + radius * np.sin(angle)
    return target


def chunk_angles(start_angle, end_angle, chunk_index, chunk_count):
    if (any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= math.pi
            for value in (start_angle, end_angle))
            or type(chunk_count) is not int or not 1 <= chunk_count <= 8
            or type(chunk_index) is not int or not 0 <= chunk_index < chunk_count):
        raise ValueError('A fold chunk has a finite declared angle range and bounded own-command index')
    angles = []
    for index in range(16):
        fraction = (index + 1) / 16
        smooth = fraction * fraction * (3 - 2 * fraction)
        angles.append(start_angle + (end_angle - start_angle) * (chunk_index + smooth) / chunk_count)
    return angles


def run(request):
    expected = {'start_wrist', 'displacement_xy', 'start_angle', 'end_angle', 'chunk_index', 'chunk_count'}
    if not isinstance(request, dict) or set(request) != expected:
        raise ValueError('Only a current own-wrist frame, public displacement and declared fold chunk are accepted')
    angles = chunk_angles(request['start_angle'], request['end_angle'], request['chunk_index'], request['chunk_count'])
    return {'own_wrist_targets': [fold_arc_wrist(request['start_wrist'], request['displacement_xy'], angle).tolist()
                                  for angle in angles],
            'robot_actions_issued': 0, 'public_cloth_lift_confirmed': False, 'official_success_claimed': False}
