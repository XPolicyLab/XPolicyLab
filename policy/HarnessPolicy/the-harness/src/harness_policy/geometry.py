"""Geometry shared by skills, workers and baselines: link6 and fingertip poses,
quaternions and rotation matrices, and camera geometry (OpenCV convention,
camera_to_world 4x4, intrinsics 3x3). Imports numpy; nothing else."""

import math

import numpy as np


# link6 orientations (wxyz); the approach axis is link6 +x. Measured with
# scripts/robodojo_reach_probe.py.
FORWARD = (0.707, 0.0, 0.0, 0.707)
DOWN = (0.5, -0.5, 0.5, 0.5)
# "Fingertips" = centre of the gripping face, 0.1501 m from link6 along its +x (X5A.urdf:
# the face of link7 spans 0.1426-0.1576 m; the same grasp point as Zhang et al. 2026b).
# The jaw ends reach 0.0075 m beyond it.
FINGERTIP_OFFSET = 0.150

# Top face of the table: the table box is centred at z = 0.74 and 0.05 thick. With DOWN
# orientation the open jaws touch it when link6 is at 0.923 (measured).
TABLE_Z = 0.765


def down_with_yaw(yaw):
    """DOWN turned by `yaw` radians about world z. The fingers slide along link6
    local y (URDF joint7/joint8), which for DOWN is world x; after the turn they
    close along (cos yaw, sin yaw, 0)."""
    half = 0.5 * float(yaw)
    rz = (math.cos(half), 0.0, 0.0, math.sin(half))
    w1, x1, y1, z1 = rz
    w2, x2, y2, z2 = DOWN
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def fingertip_of(pose, offset=None):
    """Fingertip position [x, y, z] of a link6 pose [x, y, z, qw, qx, qy, qz]."""
    offset = FINGERTIP_OFFSET if offset is None else float(offset)
    w, x, y, z = (float(v) for v in pose[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    axis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    return [float(p) + offset * a for p, a in zip(pose[:3], axis)]


def pose_matrix(pose):
    import numpy as np

    x, y, z, w, i, j, k = (float(v) for v in pose)
    norm = math.sqrt(w * w + i * i + j * j + k * k)
    w, i, j, k = w / norm, i / norm, j / norm, k / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (j * j + k * k), 2 * (i * j - w * k), 2 * (i * k + w * j)],
                      [2 * (i * j + w * k), 1 - 2 * (i * i + k * k), 2 * (j * k - w * i)],
                      [2 * (i * k - w * j), 2 * (j * k + w * i), 1 - 2 * (i * i + j * j)]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def matrix_quaternion(rotation):
    """wxyz quaternion (w >= 0) of a 3x3 rotation matrix."""
    r = rotation
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        q = [(r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s]
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        q = [(r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s]
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        q = [(r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s]
    if q[0] < 0:
        q = [-v for v in q]
    return [float(v) for v in q]


def link6_for_tip(tip, orientation, offset=None):
    """link6 pose [x, y, z, qw, qx, qy, qz] placing the fingertips at `tip`."""
    offset = FINGERTIP_OFFSET if offset is None else float(offset)
    w, x, y, z = (float(v) for v in orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # First column of the rotation matrix = link6 +x in the world frame.
    axis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    position = [float(t) - offset * a for t, a in zip(tip, axis)]
    return position + [w, x, y, z]


def tilted_toward(heading, pitch_deg):
    """link6 orientation whose approach axis (+x) points along `heading` (radians in
    the table plane) tilted down by `pitch_deg`, the fingers closing horizontally."""
    import numpy as np

    pitch = math.radians(pitch_deg)
    axis = np.array([math.cos(pitch) * math.cos(heading), math.cos(pitch) * math.sin(heading), -math.sin(pitch)])
    closing = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return matrix_quaternion(np.column_stack([axis, closing, np.cross(axis, closing)]).tolist())


def at_border(mask):
    """Whether a mask touches the image border (the object may extend beyond it)."""
    array = mask.array
    return bool(array[0].any() or array[-1].any() or array[:, 0].any() or array[:, -1].any())


def project(observation, camera, points):
    """World points (N x 3) -> pixel coordinates (N x 2) and depth along the camera axis."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    transform = observation.camera_to_world[camera]
    rotation, origin = transform[:3, :3], transform[:3, 3]
    in_camera = (points - origin) @ rotation
    depth = in_camera[:, 2]
    pixels = (observation.intrinsics[camera] @ (in_camera / depth[:, None]).T).T[:, :2]
    return pixels, depth


def pixel_ray(observation, camera, u, v):
    """Unit ray (origin, direction) in the world frame through pixel (u, v)."""
    transform = observation.camera_to_world[camera]
    direction_camera = np.linalg.solve(observation.intrinsics[camera], np.array([u, v, 1.0]))
    direction = transform[:3, :3] @ direction_camera
    return transform[:3, 3].copy(), direction / np.linalg.norm(direction)


def pixel_to_plane(observation, camera, u, v, z=TABLE_Z):
    """Intersect the ray through (u, v) with the horizontal plane at height z."""
    origin, direction = pixel_ray(observation, camera, u, v)
    if abs(direction[2]) < 1e-9:
        return None
    distance = (z - origin[2]) / direction[2]
    return None if distance <= 0 else origin + distance * direction


def triangulate(observation, pixels):
    """Least-squares 3D point from {camera: (u, v)} seen in two or more cameras."""
    if len(pixels) < 2:
        raise ValueError("triangulate needs pixels from at least two cameras")
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for camera, (u, v) in pixels.items():
        origin, direction = pixel_ray(observation, camera, u, v)
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
    return np.linalg.solve(a, b)


class Mask:
    """One segmented instance: boolean HxW array, box [x0, y0, x1, y1], score."""

    def __init__(self, array, box, score):
        self.array = np.asarray(array, dtype=bool)
        self.box = [float(v) for v in box]
        self.score = float(score)

    @property
    def area(self):
        return int(self.array.sum())

    @property
    def centroid(self):
        rows, cols = np.nonzero(self.array)
        return (float(cols.mean()), float(rows.mean())) if len(rows) else None

    def pixels(self, step=1):
        """(u, v) pixel coordinates inside the mask, every `step`-th."""
        rows, cols = np.nonzero(self.array)
        return np.stack([cols, rows], axis=1)[::max(1, int(step))].astype(float)

    def __repr__(self):
        return f"Mask(score={self.score:.2f}, area={self.area}, box={[round(v) for v in self.box]})"


def mask_to_plane(observation, camera, mask, z=TABLE_Z, step=4):
    """World points where the rays through the mask's pixels meet the plane at height z."""
    if not isinstance(mask, Mask):
        raise TypeError(f"mask must be a Mask from segment_text, got {type(mask).__name__}; "
                        "segment_text returns [] when nothing matches: check the list before indexing "
                        "and try a more descriptive prompt (e.g. 'white bowl' or a plural such as 'bowls')")
    points = [pixel_to_plane(observation, camera, u, v, z) for u, v in mask.pixels(step)]
    points = [p for p in points if p is not None]
    return np.array(points) if points else np.zeros((0, 3))
