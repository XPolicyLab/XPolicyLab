"""Lazy CuRobo v2 pose planner in the robot-base frame.

All modeled joints are supplied and returned in explicit order. The caller owns
world/base transforms, observed gripper configuration, collision geometry and
process supervision: a Python deadline cannot interrupt a running GPU kernel.
"""

import math
import time

from .primitives import vector
from .collision_world import validate_scene


class NoPathFound(RuntimeError):
    """Completed planning attempt without a feasible trajectory; service is reusable."""


def reorder(values, source_names, target_names):
    if (
        len(set(source_names)) != len(source_names)
        or len(set(target_names)) != len(target_names)
        or set(source_names) != set(target_names)
    ):
        raise ValueError("Planner joint sets differ or contain duplicates")
    values = vector(values, len(source_names), "named joint vector")
    return [values[source_names.index(name)] for name in target_names]


class CuroboV2Planner:
    def __init__(
        self,
        *,
        robot,
        scene,
        joint_names,
        tool_frame,
        device="cuda:0",
        max_attempts=3,
        max_obstacles=256,
    ):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo GPU unavailable")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Explicit unique planner joint order required")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Positive planner attempt budget required")
        self.device_cfg = DeviceCfg(device=torch.device(device))
        self.max_obstacles = max_obstacles
        self.initial_scene = validate_scene(scene, max_obstacles)
        config = MotionPlannerCfg.create(
            robot=robot,
            scene_model=self.initial_scene,
            collision_cache={"cuboid": max_obstacles},
            device_cfg=self.device_cfg,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            self_collision_check=True,
            use_cuda_graph=False,
            max_batch_size=1,
            multi_env=False,
            max_goalset=1,
        )
        self.planner = MotionPlanner(config)
        self.joint_names = list(joint_names)
        self.tool_frame = tool_frame
        self.max_attempts = max_attempts
        actual_joints = list(self.planner.joint_names)
        self._engine_names = actual_joints
        actual_tools = list(self.planner.tool_frames)
        if set(actual_joints) != set(self.joint_names) or list(
            self.planner.tool_frames
        ) != [tool_frame]:
            self.close()
            raise ValueError(
                f"Planner identity differs: joints={actual_joints}, tools={actual_tools}; "
                f"expected joints={self.joint_names}, tool={tool_frame}"
            )

    def _kinematics(self, joints):
        import torch
        from curobo.types import JointState

        ordered = reorder(joints, self.joint_names, self._engine_names)
        state = JointState.from_position(
            torch.tensor([ordered], dtype=torch.float32, device=self.device_cfg.device),
            joint_names=self._engine_names,
        )
        return self.planner.compute_kinematics(state)

    def collision_spheres(self, joints):
        spheres = self._kinematics(joints).robot_spheres.detach().cpu()
        while spheres.ndim > 2 and spheres.shape[0] == 1:
            spheres = spheres[0]
        if spheres.ndim != 2 or spheres.shape[1] != 4 or not len(spheres):
            raise ValueError("Unexpected robot collision sphere shape")
        return [vector(row, 4, "robot collision sphere") for row in spheres.tolist()]

    def tool_pose(self, joints):
        pose = self._kinematics(joints).tool_poses.to_dict()[self.tool_frame]
        return (
            pose.position.reshape(-1, 3)[0].tolist(),
            pose.quaternion.reshape(-1, 4)[0].tolist(),
        )

    def plan(self, *, start, position_m, quaternion_wxyz, deadline, scene=None):
        import torch
        from curobo.types import GoalToolPose, JointState, Pose
        from curobo.scene import Scene

        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Planner deadline reached")
        start = vector(start, len(self.joint_names), "planner start")
        position = vector(position_m, 3, "robot-base position in metres")
        quaternion = vector(quaternion_wxyz, 4, "robot-base quaternion wxyz")
        if abs(sum(x * x for x in quaternion) - 1) > 1e-4:
            raise ValueError("Unit quaternion required")
        # Every call selects its complete world. Omission restores the frozen
        # initial scene, rather than inheriting a previous request's obstacles.
        selected = validate_scene(
            self.initial_scene if scene is None else scene, self.max_obstacles
        )
        self.planner.update_world(Scene.create(selected))

        def tensor(values):
            return torch.tensor(
                [values], dtype=torch.float32, device=self.device_cfg.device
            )

        current = JointState.from_position(
            tensor(reorder(start, self.joint_names, self._engine_names)),
            joint_names=self._engine_names,
        )
        goal = GoalToolPose.from_poses(
            {
                self.tool_frame: Pose(
                    position=tensor(position), quaternion=tensor(quaternion)
                )
            },
            ordered_tool_frames=[self.tool_frame],
            num_goalset=1,
        )
        result = self.planner.plan_pose(
            goal_tool_poses=goal, current_state=current, max_attempts=self.max_attempts
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("Planner returned after deadline")
        if result is None or not bool(result.success.all().item()):
            raise NoPathFound("CuRobo did not find a successful trajectory")
        trajectory = result.get_interpolated_plan()
        if set(trajectory.joint_names) != set(self.joint_names):
            raise ValueError("CuRobo trajectory joint order changed")
        positions = trajectory.position.detach().cpu()
        while positions.ndim > 2 and positions.shape[0] == 1:
            positions = positions[0]
        if (
            positions.ndim != 2
            or positions.shape[1] != len(self.joint_names)
            or not len(positions)
        ):
            raise ValueError(
                f"Unexpected CuRobo trajectory shape: {tuple(positions.shape)}"
            )
        return [
            reorder(row, list(trajectory.joint_names), self.joint_names)
            for row in positions.tolist()
        ]

    def close(self):
        self.planner.destroy()
