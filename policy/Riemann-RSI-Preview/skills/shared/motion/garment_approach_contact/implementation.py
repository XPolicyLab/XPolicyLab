from skills.shared.motion.garment_joint_sampling.implementation import sampled_joint_path
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'approach'):
        return unavailable(api, request, current)
    controls = sampled_joint_path(api, current['route']['approach_joints'])
    return GarmentProgram(api, request, current, controls, following='contact')
