from skills.shared.motion.garment_joint_sampling.implementation import (GarmentOwnModel, sampled_joint_path,
                                                                     sampled_wrist_path)
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.verification.observed_garment_mount_state.implementation import recorded



class RecoverGarment(GarmentProgram):
    def observe(self, observation):
        self._last_observation = observation
        super().observe(observation)


    def review_completion(self):
        # Recovery is a state boundary for the route, not for the retained
        # estimate.  The packaged rule keeps one estimate across both sleeves
        # and only re-reads the torso silhouette for the paired hems, so
        # dropping the anchor here forced a fresh flat-shirt fit on an already
        # folded garment: r68 and r70 both then failed three consecutive views
        # and ended the episode with zero further actions.  Keep the anchor,
        # clear the route, and let the next view re-snap the landmarks onto the
        # current public outline.  A completed recovery is itself the bounded
        # retry, so the part also restarts with a fresh view budget.
        self.current.update(phase='view', route=None, start_wrists=None, chunk=0, attempts=0)


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != 'recover' or current['route'] is None or current['attempts'] >= 3:
        return unavailable(api, request, current)
    model = GarmentOwnModel()
    starts = {side: pose_matrix(api.targets()[side + '_ee_pose']) for side in current['route']['arms']}
    # A failed carry can leave the wrist at the top of its reach, where the
    # original 8 cm Cartesian retreat has no IK branch and the stage would
    # answer precondition-unmet three times.  Keep the same bounded retreat
    # order, then fall back to the qualified approach joints, which the rule
    # already solved for this part, through the native sixteen-command clock.
    for lift in (.08, .04, .02):
        goals = {side: start.copy() for side, start in starts.items()}
        for goal in goals.values():
            goal[2, 3] += lift
        try:
            controls = sampled_wrist_path(api, model, goals, {side: 1. for side in goals})
            return RecoverGarment(api, request, current, controls)
        except ValueError:
            continue
    try:
        joints = {side: current['route']['approach_joints'][side] for side in starts}
        controls = sampled_joint_path(api, joints)
    except ValueError:
        return unavailable(api, request, current)
    return RecoverGarment(api, request, current, controls)
