"""implementation capabilities."""
from __future__ import annotations
import numpy as np

class CalibratedRgbGeometry:

    def __init__(self, center, rotation, focal, principal):
        self.center = np.asarray(center)
        self.rotation = np.asarray(rotation)
        self.focal = focal
        self.principal = principal
        if self.center.shape != (3,) or self.rotation.shape != (3, 3) or len(principal) != 2:
            raise ValueError('Calibrated RGB geometry requires explicit camera calibration')

    def project(self, points):
        p = (np.asarray(points) - self.center) @ self.rotation
        return np.c_[self.principal[0] + self.focal * p[:, 0] / -p[:, 2], self.principal[1] - self.focal * p[:, 1] / -p[:, 2]]

    def unproject(self, uv, z):
        uv = np.asarray(uv).reshape(-1, 2)
        r = np.c_[(uv[:, 0] - self.principal[0]) / self.focal, -(uv[:, 1] - self.principal[1]) / self.focal, -np.ones(len(uv))] @ self.rotation.T
        return self.center + r * ((z - self.center[2]) / r[:, 2])[:, None]

def sample_closed_contour(hull):
    points = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a)) + 1)
        points.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.concatenate(points)
