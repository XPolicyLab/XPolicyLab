import math

from skills.shared.motion.public_phase_execution.implementation import PhasedProgram, pose_phase, relative_rotation


def run(api, request):
    initial = api.targets()[request['side'] + '_ee_pose']
    count = max(1, math.ceil(abs(request['angle']) / (math.pi / 12)))
    phases = []
    for index in range(count):
        pose = relative_rotation(initial, request['axis'], request['angle'] * (index + 1) / count)
        phases.append(pose_phase('incremental_tilt_' + str(index + 1), pose, 0))
        phases.append(pose_phase('inspect_during_tilt_' + str(index + 1), pose, 0, request['dwell'], False))
    if request['return_upright']:
        phases.append(pose_phase('restore_starting_grasp_orientation', initial, 0))
    applicable = 'pour' in api.instruction().lower()
    return PhasedProgram(api, request, phases, precondition=applicable)
