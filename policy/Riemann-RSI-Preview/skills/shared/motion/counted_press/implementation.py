from skills.shared.motion.public_phase_execution.implementation import PhasedProgram, downward_tool_pose, pose_phase, translated


def run(api, request):
    initial = api.targets()[request['side'] + '_ee_pose']
    hover = downward_tool_pose(translated(initial, request['delta']))
    pressed = translated(hover, [0, 0, -request['depth']])
    phases = [pose_phase('approach_visible_button', hover, 0)]
    for cycle in range(request['count']):
        phases.extend([
            pose_phase('press_down_' + str(cycle + 1), pressed, 0),
            pose_phase('press_hold_' + str(cycle + 1), pressed, 0, request['dwell'], False),
            pose_phase('press_release_' + str(cycle + 1), hover, 0),
        ])
    text = api.instruction().lower()
    applicable = 'press' in text or 'button' in text
    return PhasedProgram(api, request, phases, precondition=applicable)
