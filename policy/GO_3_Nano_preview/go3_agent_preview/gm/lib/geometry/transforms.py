"""Pure rotation and pose algebra. No body, no camera, no task.

Every function here is a function of its arguments.
"""
from __future__ import annotations

import math

import numpy as np


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def rpy(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
def axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T
def rot_err(R_target, R_cur):
    Re = R_target @ R_cur.T
    c = np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        w, v = np.linalg.eigh((Re + Re.T) / 2)
        axis = v[:, np.argmax(w)]
        return axis * angle
    axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * np.sin(angle))
    return axis * angle
def rot_angle(Ra, Rb):
    return float(np.linalg.norm(rot_err(Ra, Rb)))
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


TABLE_Z = 0.765


def _unwrap(q, ref):
    from ..kinematics import chain as kin

    q = np.array(q, float)
    for j in (0, 5):
        alt = ref[j] + ((q[j] - ref[j] + np.pi) % (2 * np.pi) - np.pi)
        if kin.JOINT_LIMITS[j, 0] <= alt <= kin.JOINT_LIMITS[j, 1]:
            q[j] = alt
    return q


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(i, c, s_):
    if i == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
    if i == 1:
        return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _rot3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

