"""Construct a base-frame collision world from explicit observable geometry.

Inactive-arm FK spheres use enclosing cuboids (conservative, potentially less
permissive than the sphere model). Callers supply calibrated public fixtures and
perception estimates, never hidden simulator object poses or benchmark layouts.
"""

import math

from .planner_bridge import product, quaternion, to_base
from .primitives import vector


def validate_scene(scene, max_obstacles):
    if type(max_obstacles) is not int or max_obstacles < 1:
        raise ValueError("Positive collision capacity required")
    if (
        not isinstance(scene, dict)
        or set(scene) != {"cuboid"}
        or not isinstance(scene["cuboid"], dict)
        or not 1 <= len(scene["cuboid"]) <= max_obstacles
    ):
        raise ValueError("Expected a nonempty cuboid scene within configured capacity")
    result = {}
    for name, box in scene["cuboid"].items():
        if not isinstance(name, str) or not name or set(box) != {"pose", "dims"}:
            raise ValueError("Named cuboid pose and dimensions required")
        pose = vector(box["pose"], 7, "base-frame cuboid pose")
        pose[3:] = quaternion(pose[3:])
        dims = vector(box["dims"], 3, "cuboid dimensions")
        if any(d <= 0 for d in dims):
            raise ValueError("Positive cuboid dimensions required")
        result[name] = dict(pose=pose, dims=dims)
    return {"cuboid": result}


def world_point(base_pose, point):
    base = vector(base_pose, 7, "world base pose")
    q = quaternion(base[3:])
    qi = [q[0], -q[1], -q[2], -q[3]]
    point = vector(point, 3, "base-frame point")
    rotated = product(product(q, [0, *point]), qi)[1:]
    return [a + b for a, b in zip(base[:3], rotated)]


def collision_scene(
    *,
    active_base_pose,
    obstacles,
    inactive_base_pose,
    inactive_spheres,
    padding_m=0.005,
):
    if not math.isfinite(padding_m) or padding_m < 0:
        raise ValueError("Finite nonnegative collision padding required")
    if not inactive_spheres:
        raise ValueError("Inactive arm collision geometry is required")
    cuboids = {}
    for name, obstacle in obstacles.items():
        if not isinstance(name, str) or not name or set(obstacle) != {"pose", "dims"}:
            raise ValueError("Named obstacle pose and dimensions required")
        pose = vector(obstacle["pose"], 7, "world obstacle pose")
        dims = vector(obstacle["dims"], 3, "obstacle dimensions")
        if any(d <= 0 for d in dims):
            raise ValueError("Positive obstacle dimensions required")
        position, rotation = to_base(active_base_pose, pose[:3], pose[3:])
        cuboids["observed_" + name] = dict(
            pose=position + rotation, dims=[d + 2 * padding_m for d in dims]
        )
    for index, sphere in enumerate(inactive_spheres):
        x, y, z, radius = vector(sphere, 4, "inactive-arm collision sphere")
        if radius <= 0:
            raise ValueError("Disabled or invalid inactive-arm collision sphere")
        center = world_point(inactive_base_pose, [x, y, z])
        position, _ = to_base(active_base_pose, center, [1, 0, 0, 0])
        cuboids["inactive_" + str(index)] = dict(
            pose=position + [1, 0, 0, 0], dims=[2 * (radius + padding_m)] * 3
        )
    return {"cuboid": cuboids}


class ObservedCollisionScene:
    """Rebuild the world for each plan from observable obstacles and other-arm FK.

    Providers run in the trusted runtime, must honor the absolute deadline, and
    must not consult hidden simulator poses. Obstacles are world-frame cuboids;
    spheres are in the inactive arm's calibrated base frame. No cache/fallback.
    """

    def __init__(
        self,
        *,
        base_poses,
        obstacles,
        arm_spheres,
        padding_m=0.005,
        max_obstacles=256,
    ):
        from copy import deepcopy

        if set(base_poses) != {"left", "right"}:
            raise ValueError("Both calibrated arm base poses required")
        self.base_poses = deepcopy(base_poses)
        for pose in self.base_poses.values():
            vector(pose, 7, "world base pose")
            quaternion(pose[3:])
        if not callable(obstacles) or not callable(arm_spheres):
            raise ValueError("Observable obstacle and arm FK providers required")
        self.obstacles, self.arm_spheres = obstacles, arm_spheres
        self.padding_m, self.max_obstacles = padding_m, max_obstacles

    def __call__(self, *, arm, state, deadline):
        import time
        from copy import deepcopy

        if arm not in self.base_poses:
            raise ValueError("Arm must be left or right")

        def check():
            if not math.isfinite(deadline) or time.monotonic() >= deadline:
                raise TimeoutError("Collision scene deadline reached")

        check()
        obstacles = self.obstacles(state=deepcopy(state), deadline=deadline)
        check()
        inactive = "right" if arm == "left" else "left"
        spheres = self.arm_spheres(
            arm=inactive, state=deepcopy(state), deadline=deadline
        )
        check()
        scene = collision_scene(
            active_base_pose=self.base_poses[arm],
            obstacles=obstacles,
            inactive_base_pose=self.base_poses[inactive],
            inactive_spheres=spheres,
            padding_m=self.padding_m,
        )
        return validate_scene(scene, self.max_obstacles)
