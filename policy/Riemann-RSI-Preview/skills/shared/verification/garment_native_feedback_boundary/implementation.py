"""Closed Fold feedback boundary with its explicitly isolated datum contract.

Specialized from the generic public phase validator on the garment-only domain.
No packet is rewritten, no field is stripped and no callback selects a validator.
"""
from copy import deepcopy
import math

from harness.execution import ExecutionError
from harness.runtime.manipulation import ACTION_PARAMETERS, validate_action_parameters
from skills.shared.motion.public_phase_execution.implementation import PROGRAM_STATUSES, REPORT_STATUSES
from skills.shared.verification.observed_garment_mount_state.implementation import ACTIONS


def _validate_garment_phase_feedback(request, value, automatic_side=False):
    if request.get('skill') not in ACTIONS:
        raise ExecutionError('The isolated garment feedback boundary rejects another tool domain')
    fields = {'kind', 'side', 'status', 'phase', 'phase_reports', 'issued_cycles', 'pose_qualified_cycles', 'public_gripper_value', 'last_gripper_command', 'visual_review_required', 'object_grasp_confirmed', 'official_success_claimed', 'image_measurements', 'phase_count', 'actions_executed', 'required_minimum_actions', 'budget_sufficient', 'visual_lift_feedback'}
    if type(automatic_side) is not bool:
        raise ExecutionError('Automatic own-arm selection requires a trusted declared tool contract')
    if not isinstance(value, dict) or set(value) != fields or value['kind'] != request['skill'] or (value['status'] not in PROGRAM_STATUSES) or (not isinstance(value['phase'], str)) or (not 1 <= len(value['phase']) <= 64):
        raise ExecutionError('Task-tool feedback does not match its declared phase contract')
    if value['side'] not in ('left', 'right') or (value['side'] != request['side'] and (not (automatic_side and request['side'] == 'auto'))):
        raise ExecutionError('Task-tool feedback must preserve the declared own-arm selection')
    if value['visual_review_required'] is not True or value['object_grasp_confirmed'] is not False or value['official_success_claimed'] is not False:
        raise ExecutionError('Task-tool feedback cannot claim object contact, grasp or task success')
    minimum = 0 if request['skill'] in ('liquid_scene_view', 'liquid_observe_spout') else 1
    if type(value['required_minimum_actions']) is not int or not minimum <= value['required_minimum_actions'] <= 640 or type(value['budget_sufficient']) is not bool or (value['budget_sufficient'] != (value['required_minimum_actions'] <= request['steps'])):
        raise ExecutionError('Phase feedback must distinguish insufficient action allowance from visual uncertainty')
    for name in ('public_gripper_value', 'last_gripper_command'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or (not 0 <= value[name] <= 1):
            raise ExecutionError('Gripper feedback is a normalized public value, not measured fingers')
    (issued, qualified) = (value['issued_cycles'], value['pose_qualified_cycles'])
    if type(issued) is not int or type(qualified) is not int or (not 0 <= qualified <= issued <= request.get('count', 0)):
        raise ExecutionError('Robot press cycles cannot be reported as official button counts')
    reports = value['phase_reports']
    if not isinstance(reports, list) or len(reports) > 8 or type(value['phase_count']) is not int or (not len(reports) <= value['phase_count'] <= 40) or (type(value['actions_executed']) is not int) or (not 0 <= value['actions_executed'] <= request['steps']):
        raise ExecutionError('Task-tool phase reports exceed their bound')
    actions = 0
    for row in reports:
        if not isinstance(row, dict) or set(row) != {'phase', 'status', 'actions', 'position_error_m', 'rotation_error_radians'} or (not isinstance(row['phase'], str)) or (not 1 <= len(row['phase']) <= 64) or (row['status'] not in REPORT_STATUSES) or (type(row['actions']) is not int) or (row['actions'] < 0):
            raise ExecutionError('Task-tool phase feedback has an undeclared field')
        for name in ('position_error_m', 'rotation_error_radians'):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]) or row[name] < 0:
                raise ExecutionError('Task-tool pose feedback must be finite and public')
        if row['rotation_error_radians'] > math.pi:
            raise ExecutionError('Rotation feedback exceeds its public quaternion range')
        actions += row['actions']
    if actions > value['actions_executed']:
        raise ExecutionError('Task-tool reports exceed the shared action budget')
    measurements = value['image_measurements']
    flags = {'source_visible_before', 'source_visible_after', 'support_visible_before', 'support_visible_after', 'measurement_ambiguous'}
    angles = {'axis_error_before_radians', 'axis_error_after_radians'}
    extra = set()
    if isinstance(measurements, dict):
        from skills.shared.verification.observed_garment_mount_state.implementation import measurements as validate_garment_measurements
        try:
            extra = validate_garment_measurements(measurements, value['status'], value['actions_executed'])
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError('Garment feedback must contain only current image evidence and own acknowledged commands') from error
    if not isinstance(measurements, dict) or set(measurements) != flags | angles | extra:
        raise ExecutionError('Task-tool image evidence contains an undeclared measurement')
    if any((type(measurements[name]) is not bool for name in flags)):
        raise ExecutionError('Image visibility measurements must be explicit booleans')
    for name in angles:
        angle = measurements[name]
        if angle is not None and (type(angle) not in (int, float) or not math.isfinite(angle) or (not 0 <= angle <= math.pi / 2)):
            raise ExecutionError('An image-axis error is not a calibrated world angle')
    lift = value['visual_lift_feedback']
    if not isinstance(lift, dict) or set(lift) != {'status', 'head_displacement_fraction', 'head_area_ratio', 'robot_translation_m'} or lift['status'] not in ('unavailable', 'ambiguous', 'no_visible_follow', 'visible_change'):
        raise ExecutionError('Visual lift feedback is an uncertain public-image hypothesis')
    for name in ('head_displacement_fraction', 'head_area_ratio', 'robot_translation_m'):
        number = lift[name]
        if number is None and name != 'robot_translation_m':
            continue
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ExecutionError('Visual lift measurements must be finite public image/robot differences')
    if lift['head_displacement_fraction'] is not None and lift['head_displacement_fraction'] > math.sqrt(2) or lift['robot_translation_m'] > 0.3:
        raise ExecutionError('Visual lift measurements exceed the declared relative movement')
    return deepcopy(value)


def validate_feedback(request, feedback):
    request = validate_action_parameters(request)
    if request['skill'] not in ACTIONS:
        raise ExecutionError('The isolated garment feedback boundary rejects another tool domain')
    declared_sides = ACTION_PARAMETERS[request['skill']]['side']['enum']
    checked = _validate_garment_phase_feedback(request, feedback, automatic_side='auto' in declared_sides)
    if checked['required_minimum_actions'] not in (2, 16):
        raise ExecutionError('A garment tool must retain one whole original sixteen-command segment or an explicit unmet precondition')
    return checked
