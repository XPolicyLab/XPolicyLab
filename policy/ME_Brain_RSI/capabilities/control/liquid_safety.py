"""Nominal public gripper envelope, not scene/contact or liquid-state sensing."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix

# Outward-rounded bounds from released x5_v2 meshes and joint7/8 origins.
# Finger boxes include the whole public prismatic travel 0..44 mm; this
# deliberately overbounds the grasp, without assuming observed bottle rigidity.
BOXES = (
    ((-.0001, -.0871, -.0301), (.0841, .0871, .0304)),
    ((.0715, .0003, -.0307), (.1577, .0819, .0305)),
    ((.0715, -.0819, -.0307), (.1577, -.0003, .0305)),
)
VERTICES = np.array([p for low, high in BOXES for p in product(*zip(low, high))])


def mouth_clearance_floor(upright, axis, mouth_in_ee, cup_height):
    """A common mouth height clears the whole end gripper over 0..120 deg.

    One-degree samples plus a 3 mm interpolation allowance exceed the maximum
    change of the public <=0.22 m envelope over half a degree. Additional
    20 mm clearance is nominal tolerance, not collision-free certification.
    """
    rotations = Rotation.from_rotvec(np.deg2rad(np.arange(121))[:, None]*axis).as_matrix() @ upright
    relative = VERTICES - mouth_in_ee
    lowest = np.einsum('aij,vj->avi', rotations, relative)[:, :, 2].min()
    return float(cup_height + .023 - lowest)


def stalled(poses, position_error, orientation_error, age):
    if age < 40 or len(poses) < 20:
        return False
    points = np.array(poses[-20:])
    return bool(position_error >= .009 and orientation_error < .15
                and np.max(np.linalg.norm(points-points[0], axis=1)) < .001)


def contact_stalled(poses, position_error, orientation_error, age):
    """Robot tracking stagnation, NOT a force/contact or task-success sensor."""
    if age < 20 or len(poses) < 12:
        return False
    points = np.array(poses[-12:])
    return bool(position_error >= .009 and orientation_error < .15
                and np.max(np.linalg.norm(points-points[0], axis=1)) < .001)


def near_cup(pose, mouth_in_ee, table_z, cup_height):
    """Speed zone from the nominal grasp anchor and RGB-inferred cup height."""
    mouth = pose[:3] + quat_matrix(pose[3:]) @ mouth_in_ee
    return bool(mouth[2] <= table_z + cup_height + .12)
