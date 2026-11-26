
import numpy as np

from .transforms import rot_x, rot_y, rot_z


class HeadCamera:

    W, H = 640, 480
    FX = 640 * 10.0 / 22.212
    CX, CY = 320.0, 240.0
    POS = np.array([0.0, -0.41, 1.308])
    R = rot_x(np.deg2rad(30.0))

    @classmethod
    def ray(cls, u, v):
        d = cls.R @ np.array([(u - cls.CX) / cls.FX, -(v - cls.CY) / cls.FX, -1.0])
        return d / np.linalg.norm(d)

    @classmethod
    def to_plane(cls, u, v, z):
        d = cls.ray(u, v)
        t = (z - cls.POS[2]) / d[2]
        return cls.POS + t * d

    @classmethod
    def project(cls, p):
        c = cls.R.T @ (np.asarray(p, float) - cls.POS)
        if c[2] >= 0:
            return None
        return np.array([cls.CX + cls.FX * c[0] / -c[2], cls.CY - cls.FX * c[1] / -c[2]])
def euler_xyz_deg(rx, ry, rz):
    return rot_x(np.deg2rad(rx)) @ rot_y(np.deg2rad(ry)) @ rot_z(np.deg2rad(rz))
class Pinhole:

    def __init__(self, fx, w=640, h=480):
        self.FX, self.W, self.H = fx, w, h
        self.CX, self.CY = w / 2.0, h / 2.0

    def project(self, T_world_cam, p):
        c = T_world_cam[:3, :3].T @ (np.asarray(p, float) - T_world_cam[:3, 3])
        if c[2] >= -1e-6:
            return None
        return np.array([self.CX + self.FX * c[0] / -c[2], self.CY - self.FX * c[1] / -c[2]])

    def ray(self, T_world_cam, u, v):
        d = T_world_cam[:3, :3] @ np.array([(u - self.CX) / self.FX, -(v - self.CY) / self.FX, -1.0])
        return T_world_cam[:3, 3], d / np.linalg.norm(d)

    def to_plane(self, T_world_cam, u, v, z):
        o, d = self.ray(T_world_cam, u, v)
        t = (z - o[2]) / d[2]
        return o + t * d
WRIST = Pinhole(640 * 13.0 / 20.955)


def _project_head(P, C=HeadCamera):
    c = (np.asarray(P, float) - C.POS) @ C.R
    z = -c[:, 2]
    return np.stack([C.CX + C.FX * c[:, 0] / z, C.CY - C.FX * c[:, 1] / z], 1)
