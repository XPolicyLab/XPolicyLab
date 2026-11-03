"""Exact nonadjacent arm-mesh checks using the robot's public URDF.

This supplements approximate IK collision spheres. It does not model task
objects, inter-arm collisions, or the swept path between joint targets.
"""
from pathlib import Path
import numpy as np


class RobotArmMeshCollision:
    def __init__(self, urdf, active_joint_names):
        import pinocchio as pin
        self.pin = pin
        urdf = Path(urdf)
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.joint_ids = [self.model.getJointId(name) for name in active_joint_names]
        if (len(set(self.joint_ids)) != len(self.joint_ids)
                or any(joint == 0 or joint >= self.model.njoints for joint in self.joint_ids)
                or any(self.model.joints[joint].nq != 1 for joint in self.joint_ids)):
            raise ValueError('Distinct single-coordinate arm joints required')
        self.geometry = pin.buildGeomFromUrdf(
            self.model, str(urdf), pin.GeometryType.COLLISION,
            package_dirs=[str(urdf.parent)])
        active = {0, *self.joint_ids}
        for first, a in enumerate(self.geometry.geometryObjects):
            for second in range(first + 1, len(self.geometry.geometryObjects)):
                b = self.geometry.geometryObjects[second]
                if a.parentJoint not in active or b.parentJoint not in active:
                    continue
                if (a.parentJoint == b.parentJoint
                        or self.model.parents[a.parentJoint] == b.parentJoint
                        or self.model.parents[b.parentJoint] == a.parentJoint):
                    continue
                self.geometry.addCollisionPair(pin.CollisionPair(first, second))
        if not len(self.geometry.collisionPairs):
            raise ValueError('Public robot arm collision meshes are required')
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry)

    def is_collision_free(self, joints):
        values = np.asarray(joints, float)
        if values.shape != (len(self.joint_ids),) or not np.isfinite(values).all():
            raise ValueError('Finite active arm joint coordinates required')
        q = self.pin.neutral(self.model)
        for joint, value in zip(self.joint_ids, values):
            q[self.model.joints[joint].idx_q] = value
        return not bool(self.pin.computeCollisions(
            self.model, self.data, self.geometry, self.geometry_data, q, True))
