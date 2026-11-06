from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel, sampled_gripper_clock
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class CloseGarment(GarmentProgram):
    def review_completion(self):
        self.current['start_wrists'] = {side: self.api.targets()[side + '_ee_pose'].tolist()
                                        for side in self.current['route']['arms']}
        self.current['phase'], self.current['chunk'] = 'lift', 0


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'close'):
        return unavailable(api, request, current)
    try:
        controls = sampled_gripper_clock(api, GarmentOwnModel(), tuple(current['route']['arms']))
    except ValueError:
        current['phase'] = 'recover'
        return unavailable(api, request, current)
    return CloseGarment(api, request, current, controls)
