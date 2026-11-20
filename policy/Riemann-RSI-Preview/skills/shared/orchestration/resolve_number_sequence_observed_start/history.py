from copy import deepcopy
import math
import numpy as np
from skills.shared.planning.descending_number_assignment.implementation import run as assign_numbers
from skills.shared.planning.observed_number_contract_observed_start.implementation import advance, empty_progress, validate_anchor, validate_progress
TOOLS = ('number_scene_view', 'number_home', 'number_empty_approach', 'grasp_number', 'lift_number', 'carry_number', 'place_number', 'verify_number_placement', 'recover_number_grasp', 'relay_number')
STAGES = ('grasp_number', 'lift_number', 'carry_number', 'place_number', 'verify_number_placement')

def recorded(memory):
    for event in reversed(memory):
        if event.get('skill') not in TOOLS:
            continue
        feedback = event.get('execution_feedback', {})
        measurements = feedback.get('image_measurements', {})
        if feedback.get('actions_executed', 0) <= 0 or measurements.get('command_phases_completed') is not True:
            continue
        if 'public_number_anchor' not in measurements:
            continue
        try:
            anchor = validate_anchor(measurements['public_number_anchor'])
            progress = validate_progress(measurements['public_number_progress'], len(anchor['digits']))
            return (anchor, progress, deepcopy(measurements.get('public_number_route')))
        except (ValueError, KeyError, TypeError):
            return (None, empty_progress(), None)
    return (None, empty_progress(), None)

def current(api):
    """Return a qualified own-command anchor/progress/route, rejecting inventories larger than a retained anchor.

    Anchor surfaces have shape (N, 3); inventory counts/labels contain no geometry and cannot
    restore an old anchor or authorize contact after a missing current measurement.
    """
    from skills.shared.perception.read_number_scene_observed_start.implementation import measure, public_inventory
    anchor, progress, route = recorded(api.command_memory())
    inventory = public_inventory(api.command_memory())
    if anchor is not None and len(anchor['digits']) >= inventory['minimum_count'] and (set(inventory['observed_labels']) <= {source['digit'] for source in anchor['digits']}):
        return (anchor, progress, route)
    progress = empty_progress()
    scene = measure(api)
    anchor = scene['anchor']
    return (validate_anchor(anchor), empty_progress(len(anchor['digits'])), None) if anchor else (None, progress, None)

def view_completed(memory):
    return any((event.get('skill') == 'number_scene_view' and event.get('execution_feedback', {}).get('actions_executed', 0) > 0 and any((report.get('phase') == 'settle_public_view_after_robot_motion' and report.get('actions', 0) > 0 for report in event.get('execution_feedback', {}).get('phase_reports', []))) for event in memory))

def assignment(anchor, progress):
    if progress['cursor'] >= len(anchor['digits']):
        return None
    digits = [{'digit': item['digit'], 'surface_xyz': item['surface_xyz']} for item in anchor['digits']]
    return assign_numbers(digits, anchor['targets'])[progress['cursor']]

def matches(request, anchor, progress, stage):
    item = assignment(anchor, progress)
    return item is not None and progress['stage'] == stage and (request['digit'] == item['digit']) and (request['target_index'] == item['target_index'])

def uncertain_attempts(memory, request):
    return sum((event.get('skill') == request['skill'] and event.get('outcome') == 'public_tool_uncertain' and (event.get('request', {}).get('digit') == request.get('digit')) and (event.get('request', {}).get('target_index') == request.get('target_index')) and (event.get('execution_feedback', {}).get('actions_executed', 0) > 0) for event in memory))

def home_error(api, side):
    actual, home = (api.targets()[side + '_ee_pose'], api.home_targets()[side + '_ee_pose'])
    translation = float(np.linalg.norm(np.asarray(actual[:3]) - home[:3]))
    rotation = 2 * math.acos(float(np.clip(abs(np.dot(actual[3:], home[3:])), 0, 1)))
    return (translation, rotation)

def approach_error(api, route):
    actual = api.targets()[route['side'] + '_ee_pose']
    expected = np.asarray(route['approach'])
    return (float(np.linalg.norm(actual[:3] - expected[:3])), 2 * math.acos(float(np.clip(abs(np.dot(actual[3:], expected[3:])), 0, 1))))

def empty_approach(memory):
    from skills.shared.orchestration.resolve_number_sequence_observed_start.contract import validate_empty_approach_feedback
    homes = []
    for event in reversed(memory):
        feedback = event.get('execution_feedback', {})
        if event.get('skill') == 'number_home':
            homes.append(event)
            continue
        if event.get('skill') != 'number_empty_approach':
            return None
        try:
            validate_empty_approach_feedback(event['request'], feedback)
            if feedback['status'] != 'completed' or event.get('outcome') != 'public_tool_completed' or event.get('actions') != feedback['actions_executed'] or (event.get('official_success_claimed') is not False) or (type(event.get('public_action_epoch')) is not int):
                return None
            measurements = feedback['image_measurements']
            for home in homes:
                returned = home['execution_feedback']
                actual = returned['image_measurements']
                if home['request']['side'] == feedback['side'] or home['request']['purpose'] != 'inspect' or home.get('outcome') != 'public_tool_completed' or (returned['status'] != 'completed') or (returned['actions_executed'] <= 0) or (actual.get('command_phases_completed') is not True) or any((actual.get(key) != measurements[key] for key in ('public_number_anchor', 'public_number_progress', 'public_number_route'))):
                    return None
            return deepcopy(event)
        except (ValueError, KeyError, TypeError):
            return None
    return None
