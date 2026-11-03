"""World-frame primitive targets to a named-joint robot-base planner.

Calibration and gripper affine maps are explicit frozen configuration. The
planner collision world must already be expressed in this robot-base frame.
"""

import math
import time
from copy import deepcopy

from .primitives import vector


def quaternion(value):
    value = vector(value, 4, "wxyz quaternion")
    norm = math.sqrt(sum(v * v for v in value))
    if abs(norm - 1) > 1e-3:
        raise ValueError("Expected a unit quaternion (rounding tolerance 1e-3)")
    return [v / norm for v in value]


def product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    ]


def to_base(world_base_pose, position_m, quaternion_wxyz):
    base = vector(world_base_pose, 7, "world base pose")
    q = quaternion(base[3:])
    inverse = [q[0], -q[1], -q[2], -q[3]]
    position = vector(position_m, 3, "world target position")
    offset = [position[i] - base[i] for i in range(3)]
    rotated = product(product(inverse, [0, *offset]), q)[1:]
    return rotated, product(inverse, quaternion(quaternion_wxyz))


class WorldArmPlanner:
    def __init__(
        self,
        backend,
        *,
        arm,
        world_base_pose,
        arm_joint_names,
        gripper_mapping,
        finger_tolerance=1e-5,
        scene_provider=None,
    ):
        if arm not in ("left", "right"):
            raise ValueError("Arm must be left or right")
        self.backend, self.arm = backend, arm
        if scene_provider is not None and not callable(scene_provider):
            raise ValueError("Scene provider must be callable")
        self.scene_provider = scene_provider
        self.base_pose = vector(world_base_pose, 7, "world base pose")
        self.base_pose[3:] = quaternion(self.base_pose[3:])
        self.arm_names = list(arm_joint_names)
        self.gripper_mapping = deepcopy(gripper_mapping)
        names = self.arm_names + list(self.gripper_mapping)
        if (
            not self.arm_names
            or len(names) != len(set(names))
            or set(names) != set(backend.joint_names)
        ):
            raise ValueError(
                "Every planner joint must have exactly one observed source"
            )
        for mapping in self.gripper_mapping.values():
            if (
                set(mapping) != {"index", "scale", "offset"}
                or type(mapping["index"]) is not int
                or mapping["index"] < 0
            ):
                raise ValueError("Explicit gripper index, scale and offset required")
            vector([mapping["scale"], mapping["offset"]], 2, "gripper affine map")
        if not math.isfinite(finger_tolerance) or finger_tolerance < 0:
            raise ValueError("Finite nonnegative finger tolerance required")
        self.finger_tolerance = finger_tolerance

    def observed_joints(self, state):
        """Named full model state, shared by planning and inactive-arm FK."""
        start = vector(
            state[self.arm + "_arm_joint_state"], len(self.arm_names), "arm state"
        )
        gripper = state[self.arm + "_ee_joint_state"]
        gripper = vector(gripper, len(gripper), "normalized observed gripper")
        if any(v < 0 or v > 1 for v in gripper):
            raise ValueError("Observed gripper must be normalized to [0, 1]")
        joints = dict(zip(self.arm_names, start))
        for name, mapping in self.gripper_mapping.items():
            if mapping["index"] >= len(gripper):
                raise ValueError("Observed gripper dimension differs from calibration")
            joints[name] = (
                mapping["offset"] + mapping["scale"] * gripper[mapping["index"]]
            )
        return joints

    def collision_spheres(self, *, state, deadline):
        joints = self.observed_joints(state)
        return self.backend.collision_spheres(
            [joints[name] for name in self.backend.joint_names], deadline=deadline
        )

    def __call__(self, *, arm, start, position_m, quaternion_wxyz, state, deadline):
        if arm != self.arm:
            raise ValueError("Primitive addressed another arm planner")
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Arm planner deadline reached")
        start = vector(start, len(self.arm_names), "observed arm joints")
        if start != vector(
            state[arm + "_arm_joint_state"], len(self.arm_names), "arm state"
        ):
            raise ValueError("Planner start differs from observed state")
        joints = self.observed_joints(state)
        position, orientation = to_base(self.base_pose, position_m, quaternion_wxyz)
        names = self.backend.joint_names
        scene_arguments = {}
        if self.scene_provider is not None:
            # Never fall back to the initial world when fresh perception fails.
            scene = self.scene_provider(
                arm=arm, state=deepcopy(state), deadline=deadline
            )
            if scene is None:
                raise ValueError("Scene provider returned no current collision world")
            if time.monotonic() >= deadline:
                raise TimeoutError("Scene acquisition returned after deadline")
            scene_arguments["scene"] = scene
        path = self.backend.plan(
            start=[joints[name] for name in names],
            position_m=position,
            quaternion_wxyz=orientation,
            deadline=deadline,
            **scene_arguments,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("Arm planner returned after deadline")
        if not isinstance(path, (list, tuple)) or not path:
            raise ValueError("Planner returned no trajectory")
        result = []
        for row in path:
            values = dict(zip(names, vector(row, len(names), "planned joints")))
            if any(
                abs(values[name] - joints[name]) > self.finger_tolerance
                for name in self.gripper_mapping
            ):
                raise ValueError(
                    "Trajectory changes held gripper joints; cannot discard those changes"
                )
            result.append([values[name] for name in self.arm_names])
        return result
