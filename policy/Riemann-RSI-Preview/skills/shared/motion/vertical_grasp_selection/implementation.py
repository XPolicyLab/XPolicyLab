"""Original bounded vertical-grasp selection over public geometry and own FK."""
from copy import deepcopy
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from harness.execution import ExecutionError
from skills.shared.geometry.orientation_quaternion.implementation import mat_to_quat_wxyz
from skills.shared.motion.public_phase_execution.implementation import PublicRobotAPI
from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix



def quat_mul_wxyz(first, second):
    first_scalar, first_vector = float(first[0]), np.asarray(first[1:])
    second_scalar, second_vector = float(second[0]), np.asarray(second[1:])
    return np.r_[first_scalar * second_scalar - np.dot(first_vector, second_vector),
                 first_scalar * second_vector + second_scalar * first_vector
                 + np.cross(first_vector, second_vector)].astype(np.float32)


def yaw_quat(angle):
    half = 0.5 * float(angle)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


quat_to_mat_wxyz = rotation_matrix
base = SimpleNamespace(quat_mul_wxyz=quat_mul_wxyz, yaw_quat=yaw_quat)


class VisionSwapBlocksTeacher:
    @staticmethod
    def _grasp_quat(robot: Any | None=None, cube_quat: np.ndarray | None=None) -> np.ndarray:
        frame = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
        base_quat = mat_to_quat_wxyz(frame)
        if cube_quat is None:
            return base_quat
        cube_rot = quat_to_mat_wxyz(np.asarray(cube_quat, dtype=np.float64).reshape(4))
        cube_x = cube_rot[:2, 0]
        if float(np.linalg.norm(cube_x)) < 1e-08:
            return base_quat
        cube_yaw = math.atan2(float(cube_x[1]), float(cube_x[0]))
        return quat_mul_wxyz(yaw_quat(cube_yaw), base_quat)


class PublicVerticalGraspSelector:
    def __init__(self, api, request, surface, support_plane_m):
        if type(api) is not PublicRobotAPI:
            raise ExecutionError('Original grasp selection receives only the restricted public robot API')
        point = np.asarray(surface['point'], dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not math.isfinite(float(surface['edge_m'])) or float(surface['edge_m']) <= 0
                or not math.isfinite(float(surface['yaw_radians']))
                or request['side'] not in ('auto', 'left', 'right')):
            raise ExecutionError('Original grasp selection requires finite current public surface geometry')
        self.api, self.request, self.support_plane_m = api, request, support_plane_m
        self.bases = api.own_robot_bases()
        targets = api.targets()
        sides = ('left', 'right') if request['side'] == 'auto' else (request['side'],)
        self.robots = {side: SimpleNamespace(arm_name=side + '_arm') for side in sides
                       if targets[side + '_ee_joint_state'][0] >= 0.9}
        self._events = []
        self._selected_grasp_quat = None
        self.approach_height = None
        self.successful_plans = []
        self.engagement = float(surface['edge_m']) * 0.5 - 0.015
        reference = np.asarray(surface['point'], dtype=float).copy()
        reference[2] -= float(surface['edge_m']) * 0.5
        self.source = np.r_[reference, yaw_quat(surface['yaw_radians'])]

    def _robot_base_pose(self, robot):
        return self.bases[robot.arm_name.removesuffix('_arm')]

    def _wrist_pose(self, center, height, quaternion):
        point = np.asarray(center[:3], dtype=float).copy()
        point[2] = height
        return np.r_[point - rotation_matrix(quaternion)[:, 0] * 0.155, quaternion]

    def _plan_pose_for(self, robot, pose):
        side = robot.arm_name.removesuffix('_arm')
        response = self.api.plan_trajectory(side, pose, self.support_plane_m, self.request['steps'], 1.8)
        if response['status'] == 'Budget':
            raise ExecutionError('Original grasp preflight exceeds the declared action allowance')
        if response['status'] != 'Success':
            raise RuntimeError('cuRobo plan_path failed for public grasp preflight')
        self.successful_plans.append((side, np.asarray(pose).copy(), deepcopy(response)))
        return response['positions']

    def _choose_vertical_arm(self, source):
        robots = sorted(self.robots.values(), key=lambda robot: float(np.linalg.norm(self._robot_base_pose(robot)[:2] - source[:2])))
        failures = []
        for height in [0.04, 0.02]:
            for robot in robots:
                nominal = VisionSwapBlocksTeacher._grasp_quat(robot, source[3:])
                for yaw in [0.0, math.pi / 2, -math.pi / 2, math.pi]:
                    quat = base.quat_mul_wxyz(base.yaw_quat(yaw), nominal).astype(np.float32)
                    tip = float(source[2] + 0.015)
                    approach = self._wrist_pose(source, tip + height, quat)
                    grasp = self._wrist_pose(source, tip, quat)
                    try:
                        self._plan_pose_for(robot, approach)
                        self._plan_pose_for(robot, grasp)
                    except RuntimeError as error:
                        if 'cuRobo plan_path failed' not in str(error):
                            raise
                        failures.append({'arm': robot.arm_name, 'yaw': yaw, 'height': height})
                        continue
                    self._selected_grasp_quat = quat
                    self.approach_height = height
                    self._events.append({'event': 'reachable_grasp', 'arm': robot.arm_name, 'yaw_offset': yaw, 'approach_height': height, 'rejected_candidates': failures})
                    return robot
        raise RuntimeError(f'No reachable vertical grasp among {len(failures)} candidates')

    def select(self):
        robot = self._choose_vertical_arm(self.source)
        side = robot.arm_name.removesuffix('_arm')
        tip = float(self.source[2] + 0.015)
        approach = self._wrist_pose(self.source, tip + self.approach_height, self._selected_grasp_quat)
        grasp = self._wrist_pose(self.source, tip, self._selected_grasp_quat)
        approach_plan = self.successful_plans[-2]
        if approach_plan[0] != side or not np.allclose(approach_plan[1], approach):
            raise ExecutionError('A selected grasp requires the matching original approach preflight')
        return {'side': side, 'approach': approach, 'grasp': grasp, 'engagement': self.engagement,
                'approach_plan': deepcopy(approach_plan[2]), 'event': deepcopy(self._events[-1])}
