from skills.shared.orchestration.instruction_semantics.implementation import resolver_contract
from skills.shared.perception.read_number_outline.implementation import run as read_outlines
from skills.shared.perception.read_number_scene_observed_start.implementation import applicable, fresh_source, pad_regions, target_row
from skills.shared.orchestration.resolve_number_sequence_observed_start.history import STAGES, assignment, current, home_error, view_completed

def operation(name, parameters, progress, stage, reason):
    return {'applicable': True, 'operations': [{'skill': name, 'parameters': {key: [value] for key, value in parameters.items()}}], 'relation': {'kind': 'public_number_sorting', 'cursor': progress['cursor'], 'stage': stage}, 'reason': reason}

def matching_attempts(memory, skill, progress, item=None):
    """Read bounded own history; return matching stage attempts since its acknowledged recovery."""
    attempts = []
    for event in memory:
        request = event.get('request', {})
        if skill == 'verify_number_placement' and event.get('skill') == 'number_scene_view' and (request.get('cursor') == progress['cursor']):
            feedback = event.get('execution_feedback', {})
            if feedback.get('status') == 'completed' and feedback.get('actions_executed', 0) > 0 and any((phase.get('phase') == 'settle_public_view_after_robot_motion' and phase.get('actions', 0) > 0 for phase in feedback.get('phase_reports', []))):
                attempts = []
            continue
        matches = item is None and request.get('cursor') == progress['cursor'] or (item is not None and request.get('digit') == item['digit'] and (request.get('target_index') == item['target_index']))
        if not matches:
            continue
        if event.get('skill') in ('recover_number_grasp', 'relay_number') and skill != event.get('skill'):
            feedback = event.get('execution_feedback', {})
            if event.get('skill') == 'relay_number' and feedback.get('image_measurements', {}).get('public_number_relay', {}).get('relocalized_after_release') is not True:
                continue
            if feedback.get('status') == 'completed' and feedback.get('actions_executed', 0) > 0 and (feedback.get('image_measurements', {}).get('command_phases_completed') is True):
                attempts = []
        elif event.get('skill') == skill:
            attempts.append(event)
    return attempts

def actual_views(memory, progress):
    return [event for event in matching_attempts(memory, 'number_scene_view', progress) if any((report.get('phase') == 'settle_public_view_after_robot_motion' and report.get('actions', 0) > 0 for report in event.get('execution_feedback', {}).get('phase_reports', [])))]

def view_request(api, memory, progress, maximum_attempts):
    """Select a bounded current camera view from public API/history; return an operation or None."""
    attempts = matching_attempts(memory, 'number_scene_view', progress)
    views = actual_views(memory, progress)
    initial = progress['cursor'] == 0 and progress['stage'] == 0
    maximum_views = 5 if initial and maximum_attempts > 0 else maximum_attempts
    if len(views) >= maximum_views or len(attempts) >= 2 * maximum_views:
        return None
    family = 'targets' if initial and (not views) else 'numerals'
    if not read_outlines(api, 'cam_head')['digits'] and target_row(pad_regions(api.image('cam_head'))) is None:
        return None
    side = 'left' if not views or (initial and len(views) % 2 == 1) else 'right'
    return operation('number_scene_view', {'side': side, 'cursor': progress['cursor'], 'view_family': family}, progress, 'view', 'Acquire independent visible target and numeral views; opening alone is not camera qualification.')

@resolver_contract(applicable)
def run(api, memory, maximum_uncertain_attempts=2):
    """Select an operation from public history/state; own EE poses have shape (7,) and openings (1,)."""
    empty = {'applicable': applicable(api.instruction()), 'operations': [], 'relation': None, 'reason': 'No publicly qualified next number sorting skill.'}
    if not empty['applicable']:
        return empty
    maximum_attempts = max(0, min(2, maximum_uncertain_attempts))
    anchor, progress, route = current(api)
    if progress['cursor'] == 0 and progress['stage'] == 0 and (not view_completed(memory)):
        return view_request(api, memory, progress, maximum_attempts) or dict(empty, reason='The bounded actual initial view has not qualified current number geometry.')
    if anchor is None:
        return view_request(api, memory, progress, maximum_attempts) or empty
    if progress['cursor'] == progress['count']:
        if progress['final_stage'] >= 2:
            return dict(empty, reason='Own final home commands are complete; only the official evaluator determines success.')
        side = 'left' if progress['final_stage'] == 0 else 'right'
        return operation('number_home', {'side': side, 'purpose': 'finish'}, progress, 'home_' + side, 'Complete both original own-joint homes after the last current visual placement check.')
    item = assignment(anchor, progress)
    name = STAGES[progress['stage']]
    parameters = {'side': 'auto', 'digit': item['digit'], 'target_index': item['target_index']}
    attempts = matching_attempts(memory, name, progress, item)
    failed = [event for event in attempts if event.get('outcome') == 'public_tool_uncertain' or event.get('execution_feedback', {}).get('status') in ('uncertain', 'precondition_unmet', 'budget')]
    physical_failures = [event for event in failed if event.get('execution_feedback', {}).get('actions_executed', 0) > 0]
    if progress['stage'] in (0, 1) and physical_failures and (progress['retry'] == 0):
        recoveries = matching_attempts(memory, 'recover_number_grasp', progress, item)
        if len(recoveries) < maximum_attempts:
            return operation('recover_number_grasp', parameters, progress, 'recover', 'Open and reset once before selecting an orthogonal mature grasp axis from current RGB.')
        return dict(empty, reason='The bounded open-and-reset recovery did not complete.')
    if maximum_attempts > 0 and progress['stage'] == 0 and any((event.get('execution_feedback', {}).get('image_measurements', {}).get('public_number_grasp_precondition') == 'no_single_arm_route' for event in failed)):
        relay_attempts = matching_attempts(memory, 'relay_number', progress, item)
        if relay_attempts:
            return dict(empty, reason='The single measured intermediary relay did not qualify; no blind or repeated relay is permitted.')
        if fresh_source(api, anchor, item['digit']) is not None:
            return operation('relay_number', parameters, progress, 'relay', 'Restore the mature checked first-arm relay and fresh-source receiver handoff before learned fallback.')
    if len(failed) >= maximum_attempts:
        return dict(empty, reason='This physical stage exhausted its bounded attempts; permit learned recovery, not endless rule replay.')
    if progress['stage'] == 0 and all((float(api.targets()[side + '_ee_joint_state'][0]) >= 0.9 for side in ('left', 'right'))):
        for side in ('left', 'right'):
            translation, rotation = home_error(api, side)
            if translation <= 0.02 and rotation <= 0.12:
                continue
            homes = [event for event in memory if event.get('skill') == 'number_home' and event.get('request', {}).get('side') == side and (event.get('request', {}).get('purpose') == 'inspect') and (not (event.get('execution_feedback', {}).get('status') == 'completed' and event.get('execution_feedback', {}).get('actions_executed', 0) > 0))]
            if len(homes) >= maximum_attempts:
                return dict(empty, reason='The bounded open-arm inspection home did not clear the transfer workspace.')
            return operation('number_home', {'side': side, 'purpose': 'inspect'}, progress, 'home_' + side, 'Return the empty observing arm to its own recorded home before a loaded cross-table transfer.')
    if progress['stage'] == 0 and fresh_source(api, anchor, item['digit']) is None:
        return view_request(api, memory, progress, maximum_attempts) or dict(empty, reason='No current public camera independently relocalizes the source; stale anchors do not authorize grasp.')
    if progress['stage'] == 4:
        if failed:
            return view_request(api, memory, progress, maximum_attempts) or dict(empty, reason='Current independent outline placement remains uncertain after bounded public views.')
    return operation(name, parameters, progress, name, 'Compose mature sampled grasp, lift, carry, release and current visual verification using own command handoffs.')
