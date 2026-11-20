from functools import lru_cache
import json
from pathlib import Path
from skills.shared.orchestration.resolve_number_sequence_observed_start.history import STAGES, assignment

def validate_empty_approach_feedback(request, feedback):
    from skills.shared.planning.observed_number_contract_observed_start.implementation import validate_anchor, validate_progress, validate_route
    if feedback['status'] != 'completed':
        return
    measurements = feedback['image_measurements']
    anchor = validate_anchor(measurements['public_number_anchor'])
    progress = validate_progress(measurements['public_number_progress'], len(anchor['digits']))
    route = validate_route(measurements['public_number_route'])
    selected = assignment(anchor, progress)
    reports = feedback['phase_reports']
    if progress['stage'] != 0 or selected is None or route['side'] != request['side'] or (feedback['side'] != route['side']) or (route['digit'] != request['digit']) or (selected['digit'] != request['digit']) or (selected['target_index'] != request['target_index']) or (measurements.get('current_source_identity_before_grasp') != request['digit']) or (measurements.get('command_phases_completed') is not True) or (feedback['actions_executed'] < 4) or (feedback['required_minimum_actions'] != 4) or (feedback['budget_sufficient'] is not True) or (feedback['phase_count'] != 2) or (feedback['last_gripper_command'] != 1.0) or (feedback['public_gripper_value'] < 0.9) or (len(reports) != 2) or (reports[0]['phase'] != 'number_open') or (reports[0]['status'] != 'command_dwell') or (reports[0]['actions'] != 2) or (reports[1]['phase'] != 'number_approach') or (reports[1]['status'] != 'pose_reached') or (reports[1]['actions'] < 2) or (reports[1]['position_error_m'] > 0.025) or (reports[1]['rotation_error_radians'] > 0.12) or (sum((report['actions'] for report in reports)) != feedback['actions_executed']):
        raise ValueError('An empty approach needs its whole original open/approach receipt without contact or progress advancement')

def validate_workflow(value):
    expected = [{'id': name.removesuffix('_number').removeprefix('verify_number_'), 'skill': name} for name in STAGES]
    expected += [{'id': 'recover', 'skill': 'recover_number_grasp'}]
    expected += [{'id': 'relay', 'skill': 'relay_number'}]
    expected += [{'id': stage, 'skill': 'number_home'} for stage in ('home_left', 'home_right')]
    if value.get('schema') != 'riemann-public-command-workflow-v1' or value.get('stages') != expected or value.get('view_skill') != 'number_scene_view' or (value.get('initial_actual_wrist_view_required') is not True) or (value.get('initial_view_families') != ['targets', 'left_numerals', 'right_numerals']) or (value.get('maximum_uncertain_attempts') != 2) or (value.get('maximum_orthogonal_retries') != 1) or (value.get('maximum_relay_attempts') != 1) or (value.get('observed_row_count_only') is not True) or (value.get('robot_mount_reference_is_object_measurement') is not False) or (value.get('completion_is_official_success') is not False):
        raise ValueError('Number sorting retains independent current measurements and bounded mature commands')
    return value

@lru_cache(maxsize=1)
def workflow():
    path = Path(__file__).resolve().parent / 'workflow.json'
    if path.is_symlink():
        raise ValueError('Use the contained frozen shared number workflow')
    return validate_workflow(json.loads(path.read_text()))

def resolve(api, memory, maximum_uncertain_attempts=2):
    from skills.shared.orchestration.resolve_number_sequence_observed_start.implementation import run
    return run(api, memory, min(maximum_uncertain_attempts, workflow()['maximum_uncertain_attempts']))
