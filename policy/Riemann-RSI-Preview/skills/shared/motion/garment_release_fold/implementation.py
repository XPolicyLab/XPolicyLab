from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel, sampled_gripper_clock
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'release'):
        return unavailable(api, request, current)
    try:
        controls = sampled_gripper_clock(api, GarmentOwnModel(), tuple(current['route']['arms']),
                                         release=True, cuff=current['part'] != 2)
    except ValueError:
        return unavailable(api, request, current)
    return GarmentProgram(api, request, current, controls, following='clear')
