import math
import numpy as np

from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel, sampled_wrist_path, sampled_body_arc
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.perception.current_garment_landmarks.implementation import lifted_surface_feedback
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class LiftGarment(GarmentProgram):
    def review_completion(self):
        self.visual = lifted_surface_feedback(self.api, self.current['anchor'], self.current['route']['arms'])
        if self.visual['status'] == 'current_surface_follows_own_pad':
            self.current['phase'] = 'carry'
            return
        if self.continue_after_shortfall():
            # The official budget cannot fund a regrasp and the fold that has to
            # follow it, so a negative verdict would trade a finished fold for
            # one that stops mid-arc with the jaw closed.  Carry on instead: a
            # false negative still has the cloth pinched and completes the fold,
            # and a true negative has already left the cloth unfolded, so the
            # tier the earlier parts banked cannot be lost either way.
            self.current.update(phase='carry', attempts=0)
            return
        self.current['phase'] = 'recover'
        self.status = 'uncertain'


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'lift') or current['start_wrists'] is None:
        return unavailable(api, request, current)
    model = GarmentOwnModel()
    # The packaged fold_body() anchored the torso arc on the wrist frames that
    # plan_body_contacts() returned.  start_wrists is where the arm actually
    # settled, and the paired-hem lower stage deliberately stops short against
    # the cloth (its own bound is 6 cm for part 2), so anchoring there moved the
    # whole arc 2.7-3.9 cm outward: r90's sample at 1.96 rad then sat at 0.524 m
    # from its own base against a live seed at 0.505 m, past the arm's reach, and
    # the inverse solve refused it.  Use the planned frame the packaged rule used.
    anchor = current['route']['contact_poses'] if current['part'] == 2 else current['start_wrists']
    starts = {side: pose_matrix(value) for side, value in anchor.items()}
    try:
        if current['part'] == 2:
            displacements = {side: np.asarray(current['route']['destinations'][side])[:2]
                             - np.asarray(current['route']['contacts'][side])[:2] for side in starts}
            controls = sampled_body_arc(api, model, starts, displacements, 0., math.pi / 2., 0, 1)
        else:
            goals = {side: pose_matrix(api.targets()[side + '_ee_pose']) for side in starts}
            for goal in goals.values():
                goal[2, 3] += .075
            controls = sampled_wrist_path(api, model, goals)
    except ValueError:
        current['phase'] = 'recover'
        return unavailable(api, request, current)
    return LiftGarment(api, request, current, controls)
