"""Validated block-grasp and closed-carry phases, without layout constants."""
import numpy as np
from types import SimpleNamespace

from skills.shared.motion.public_phase_execution.implementation import pose_phase, translated
from skills.shared.motion.upright_transfer.implementation import axis_rotation
from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix


def wrist_for_surface(surface, orientation=None):
    quaternion = (axis_rotation(np.asarray([0.5, -0.5, 0.5, 0.5]), [0, 0, 1], surface['yaw_radians'])
                  if orientation is None else np.asarray(orientation, dtype=float))
    point = surface['point'].copy()
    point[2] -= float(surface['edge_m']) * 0.5 - 0.015
    wrist = point - rotation_matrix(quaternion)[:, 0] * 0.155
    return np.asarray([*wrist, *quaternion]), float(surface['point'][2] - point[2])


def grasp_phases(initial, surface, selection=None, *, lift_clearance=0.055):
    if selection is None:
        grasp, engagement = wrist_for_surface(surface)
        hover = translated(grasp, [0, 0, 0.04])
    else:
        grasp, engagement = selection['grasp'].copy(), selection['engagement']
        hover = selection['approach'].copy()
    phases = [pose_phase('open_at_current_pose', initial, 1, 1, False),
              pose_phase('approach_above_visible_target', hover, 1),
              pose_phase('descend_to_visible_grasp', grasp, 1),
              pose_phase('close_command', grasp, 0, 2, False),
              pose_phase('settle_command', grasp, 0, 1, False),
              pose_phase('test_lift_keep_orientation', translated(grasp, [0, 0, lift_clearance]), 0)]
    return phases, engagement


class CarryRecipe:
    def __init__(self, api, request, plane):
        self.api, self.request, self.plane = api, request, plane
        self.args = SimpleNamespace(transfer_clearance=0.055)
        self.phases = []
        self.preflight = None

    def _wrist_pose(self, center, height, quaternion):
        point = np.asarray(center[:3], dtype=float).copy()
        point[2] = height
        wrist = point - rotation_matrix(quaternion)[:, 0] * 0.155
        return np.asarray([*wrist, *quaternion])

    def _plan_pose_for(self, robot, pose):
        if self.api is None or not self.api.trajectory_available():
            return []
        self.preflight = self.api.plan_trajectory(self.request['side'], pose, self.plane,
            self.request['steps'], 1.4)
        if self.preflight['status'] != 'Success':
            raise RuntimeError('cuRobo plan_path failed for public carry preflight')
        return self.preflight['positions']

    def _run_arm_pose(self, robot, pose, phase, grip):
        self.phases.append(pose_phase(phase, pose, grip))

    def _carry_closed(self, robot, source, destination, source_tip_z, destination_tip_z,
                      source_quat, destination_quat, phase_prefix, clearance=None):
        clearance = self.args.transfer_clearance if clearance is None else clearance
        height = max(source_tip_z, destination_tip_z) + clearance
        source_high = self._wrist_pose(source, height, source_quat)
        try:
            self._plan_pose_for(robot, source_high)
        except RuntimeError as error:
            if 'cuRobo plan_path failed' not in str(error):
                raise
            low = self._wrist_pose(source, source_tip_z + 0.025, source_quat)
            midpoint = (np.asarray(source[:3]) + np.asarray(destination[:3])) / 2
            middle = self._wrist_pose(midpoint, height, source_quat)
            high = self._wrist_pose(destination, height, destination_quat)
            self._run_arm_pose(robot, low, f'{phase_prefix}_carry_low', grip=0.0)
            self._run_arm_pose(robot, middle, f'{phase_prefix}_carry_midpoint', grip=0.0)
            self._run_arm_pose(robot, high, f'{phase_prefix}_carry_destination', grip=0.0)
            return high
        return self._carry_upright(robot, source, destination, source_tip_z, destination_tip_z,
                                   source_quat, destination_quat, phase_prefix, clearance)

    def _carry_upright(self, robot, source, destination, source_tip_z, destination_tip_z,
                       source_quat, destination_quat, phase_prefix, clearance=None):
        if abs(float(np.dot(source_quat, destination_quat))) < 0.9999:
            raise RuntimeError('Central carry must preserve the grasp orientation')
        clearance = self.args.transfer_clearance if clearance is None else clearance
        height = max(source_tip_z, destination_tip_z) + clearance
        source_high = self._wrist_pose(source, height, source_quat)
        destination_high = self._wrist_pose(destination, height, destination_quat)
        self._run_arm_pose(robot, source_high, f'{phase_prefix}_carry_source', grip=0.0)
        self._run_arm_pose(robot, destination_high, f'{phase_prefix}_carry_destination', grip=0.0)
        return destination_high


def placement_phases(initial, source, support, engagement, api=None, request=None, plane=None,
                     *, carry_clearance=None):
    landing = initial.copy()
    point = support['point'].copy()
    point[2] += source['edge_m'] - engagement
    landing[:3] = point - rotation_matrix(initial[3:])[:, 0] * 0.155
    clearance = ((0.035 if support['point'][2] > source['point'][2] + source['edge_m'] * 0.5 else 0.055)
                 if carry_clearance is None else carry_clearance)
    recipe = CarryRecipe(api, request, plane)
    recipe._carry_closed(None, source['point'], point, float(source['point'][2] - engagement),
                         float(point[2]), initial[3:], initial[3:], 'visible_pair', clearance)
    phases = recipe.phases + [pose_phase('lower_to_visible_support', landing, 0, 2, False),
              pose_phase('release_after_planned_contact_placement', landing, 1, 2, False),
              pose_phase('retreat_from_stack', translated(landing, [0, 0, 0.055]), 1)]
    return phases, float(support['point'][2] + source['edge_m']), recipe.preflight
