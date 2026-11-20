import math
from copy import deepcopy
import numpy as np

from skills.shared.geometry.measured_cloth_arc.implementation import fold_arc_wrist
from skills.shared.motion.garment_joint_sampling.implementation import CHUNK, GarmentOwnModel, current_snapshot, sampled_wrist_path
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class ReleasedClothSettle(GarmentProgram):
    """Acknowledge the original hold; this is not another endpoint/contact gate."""

    def __init__(self, api, request, current, controls):
        super().__init__(api, request, current, controls, following='home')
        self.phases[0]['name'] = 'released_cloth_settle_before_origin'

    def endpoint_reached(self):
        # The original hold repeats its binding's measured own joint targets.
        # It checks command execution, not an additional pose/contact predicate.
        return True


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if (current['phase'] == 'prehome_settle' and current['part'] == 3
            and current['route'] is None):
        # v45 hold() binds the current public six-joint states once, rather
        # than reusing the withdrawal's nominal endpoint or replanning midway.
        binding = current_snapshot(api)
        for side in ('left', 'right'):
            binding[side + '_ee_joint_state'] = [1.]
        controls = [deepcopy(binding) for unused in range(CHUNK)]
        return ReleasedClothSettle(api, request, current, controls)
    if not allowed(current, 'lower'):
        return unavailable(api, request, current)
    goals = {}
    for side in current['route']['arms']:
        if current['part'] == 2:
            displacement = np.asarray(current['route']['destinations'][side])[:2] - np.asarray(current['route']['contacts'][side])[:2]
            # The landing pose is the planned arc endpoint, so it has to be
            # measured from the planned contact frame rather than from the pose
            # the cloth left the wrist in.
            goal = fold_arc_wrist(pose_matrix(current['route']['contact_poses'][side]), displacement, math.pi)
            goal[2, 3] -= .003
        else:
            goal = pose_matrix(api.targets()[side + '_ee_pose'])
            goal[2, 3] -= .063
        goals[side] = goal
    try:
        controls = sampled_wrist_path(api, GarmentOwnModel(), goals, contact=True)
    except ValueError:
        current['phase'] = 'recover'
        return unavailable(api, request, current)
    return GarmentProgram(api, request, current, controls, following='release')
