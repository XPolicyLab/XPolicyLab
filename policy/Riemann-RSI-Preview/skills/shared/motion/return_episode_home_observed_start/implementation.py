from skills.shared.motion.observed_start_phase_execution.implementation import PhasedProgram, pose_phase

def run(api, request):
    pose = api.home_targets()[request['side'] + '_ee_pose']
    return PhasedProgram(api, request, [pose_phase('return_to_public_episode_home', pose, 1, 4)])
