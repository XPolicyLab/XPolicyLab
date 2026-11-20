from skills.shared.motion.garment_joint_sampling.implementation import sampled_joint_path
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, unavailable
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != 'home' or current['part'] != 3:
        return unavailable(api, request, current)
    controls = sampled_joint_path(api, api.home_joint_states())
    for command in controls:
        for side in ('left', 'right'):
            command[side + '_ee_joint_state'] = [1.]
    return GarmentProgram(api, request, current, controls, following='hold')
