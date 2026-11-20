import math

import numpy as np


def mat_to_quat_wxyz(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = np.trace(matrix)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = np.array([scale / 4, (matrix[2, 1] - matrix[1, 2]) / scale,
                               (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, scale / 4,
                               (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        quaternion = np.array([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                               scale / 4, (matrix[1, 2] + matrix[2, 1]) / scale])
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        quaternion = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, scale / 4])
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)
