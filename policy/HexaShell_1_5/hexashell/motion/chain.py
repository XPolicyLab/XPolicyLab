"""Forward kinematics of one X5 arm, read from the arm's URDF.

The chain is base_link -> joint1 .. joint6 -> link6 of the rig's URDF (RoboDojo: ``assets/X5A.urdf``,
a copy of RoboDojo's ``Assets/Robots/x5/X5A.urdf``).  RoboDojo reports ``<arm>_ee_pose`` as link6's
world pose, so the grasp point is found by sliding along link6's x axis (wrist -> fingertips); the
fingers close along link6's y axis.  Where the arms stand and how far the grasp point is from link6
come from the :class:`~hexashell.motion.rig.Rig`.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..geometry import homogeneous, rot_z, rpy
from .rig import ROBODOJO, Rig

URDF_PATH = ROBODOJO.urdf


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray  # 4x4, parent link -> joint frame at zero angle
    axis: np.ndarray  # unit axis in the joint frame


def read_urdf_joints(path: Path = URDF_PATH) -> dict[str, Joint]:
    """All joints of a URDF, keyed by child link."""
    root = ET.parse(path).getroot()
    out: dict[str, Joint] = {}
    for el in root.findall("joint"):
        origin = el.find("origin")
        xyz = [0.0, 0.0, 0.0]
        angles = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            angles = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis_el = el.find("axis")
        axis = np.array([1.0, 0.0, 0.0])
        if axis_el is not None:
            axis = np.array([float(v) for v in axis_el.get("xyz").split()])
            axis = axis / np.linalg.norm(axis)
        child = el.find("child").get("link")
        out[child] = Joint(
            name=el.get("name"),
            parent=el.find("parent").get("link"),
            child=child,
            kind=el.get("type"),
            origin=homogeneous(rpy(*angles), xyz),
            axis=axis,
        )
    return out


def _turn(axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation by ``angle`` about a unit ``axis`` (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    t = 1.0 - c
    m = np.eye(4)
    m[:3, :3] = [
        [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
        [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
        [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
    ]
    return m


class ArmChain:
    """One arm: its base pose on the table and the revolute chain up to link6."""

    def __init__(self, name: str, base: np.ndarray, joints: list[Joint], rig: Rig = ROBODOJO):
        self.name = name
        self.base = base
        self.joints = joints
        self.rig = rig
        self.low = np.array([lo for lo, _ in rig.joint_ranges])
        self.high = np.array([hi for _, hi in rig.joint_ranges])

    @classmethod
    def for_arm(cls, name: str, rig: Rig = ROBODOJO) -> "ArmChain":
        xyz, heading = rig.bases[name]
        return cls(name, homogeneous(rot_z(heading), xyz), _revolute_chain(rig.tips[name], rig.urdf), rig)

    # -- forward kinematics ----------------------------------------------------------------------
    def frames(self, q) -> list[np.ndarray]:
        """World frames of link1 .. link6 for joint angles ``q``."""
        t = self.base
        out = []
        for joint, angle in zip(self.joints, q):
            t = t @ joint.origin @ _turn(joint.axis, float(angle))
            out.append(t)
        return out

    def flange(self, q) -> np.ndarray:
        """World pose of link6 (what RoboDojo reports as the end-effector pose)."""
        return self.frames(q)[-1]

    def grasp_pose(self, q, reach: float | None = None) -> np.ndarray:
        return slide(self.flange(q), self.rig.grasp_reach if reach is None else reach)

    def jacobian(self, q, reach: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(6x6 world Jacobian of the point ``reach`` along link6 x (default: the grasp point), that
        point's pose)."""
        reach = self.rig.grasp_reach if reach is None else reach
        frames = self.frames(q)
        tip = slide(frames[-1], reach)
        p = tip[:3, 3]
        jac = np.zeros((6, 6))
        for i, (joint, frame) in enumerate(zip(self.joints, frames)):
            # the joint's axis and origin in the world: the joint rotates frame i about its axis
            a = frame[:3, :3] @ joint.axis
            o = frame[:3, 3]
            jac[:3, i] = np.cross(a, p - o)
            jac[3:, i] = a
        return jac, tip

    def clip(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.low, self.high)

    def within_range(self, q, slack: float = 1e-6) -> bool:
        q = np.asarray(q, float)
        return bool(np.all(q >= self.low - slack) and np.all(q <= self.high + slack))


def slide(pose: np.ndarray, reach: float) -> np.ndarray:
    """``pose`` moved ``reach`` metres along its own x axis."""
    out = pose.copy()
    out[:3, 3] = pose[:3, 3] + reach * pose[:3, 0]
    return out


def link_points(chain: ArmChain, q, spacing: float = 0.02) -> np.ndarray:
    """Points about ``spacing`` apart along the arm's centre line, from joint 1 to the fingertips.
    The same chain always gives the same number of points, in the same order."""
    frames = chain.frames(q)
    corners = [f[:3, 3] for f in frames] + [slide(frames[-1], chain.rig.tip_reach)[:3, 3]]
    out = [corners[0]]
    for a, b in zip(corners, corners[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def arm_gap(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """The closest distance between two arms' centre lines (from :func:`link_points`).  The links
    and grippers are thick: arms whose centre lines are 6-8 cm apart touch."""
    return float(np.min(np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)))


@lru_cache(maxsize=4)
def _all_joints(path: Path = URDF_PATH) -> dict[str, Joint]:
    return read_urdf_joints(path)


def _revolute_chain(tip: str = "link6", path: Path = URDF_PATH) -> list[Joint]:
    """The six revolute joints from the arm's base link up to ``tip``, base first.  The base link is
    where the walk meets the first fixed joint after them (the rig gives that link's world pose)."""
    joints = _all_joints(path)
    chain = []
    link = tip
    while link in joints:
        joint = joints[link]
        if joint.kind not in ("revolute", "continuous"):
            break
        chain.append(joint)
        link = joint.parent
    chain.reverse()
    if len(chain) != 6:
        raise RuntimeError(f"expected 6 revolute joints up to {tip} in {path.name}, found {len(chain)}")
    return chain


@lru_cache(maxsize=8)
def arm_chain(name: str, rig: Rig = ROBODOJO) -> ArmChain:
    return ArmChain.for_arm(name, rig)
