from copy import deepcopy
import hashlib
import re

import numpy as np


SEMANTICS = 'previous_delivered_head_RGB_during_open_camera_only_commands_not_hidden_state'


def validate(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'digits', 'target_centres', 'image_sha256', 'semantics'}
            or value['schema'] != 'riemann-public-number-head-reference-v1' or value['semantics'] != SEMANTICS
            or not isinstance(value['image_sha256'], str) or re.fullmatch('[0-9a-f]{64}', value['image_sha256']) is None):
        raise ValueError('A head reference contains only previously delivered public pixels and independent numeral reads')
    digits = value['digits']
    if not isinstance(digits, list) or not 2 <= len(digits) <= 8:
        raise ValueError('A head reference requires a bounded independently complete numeral family')
    labels = []
    for digit in digits:
        if (not isinstance(digit, dict) or set(digit) != {'digit', 'camera', 'anchor_image', 'yaw_radians'}
                or type(digit['digit']) is not int or not 0 <= digit['digit'] <= 9
                or digit['camera'] != 'cam_head' or type(digit['yaw_radians']) not in (int, float)
                or not np.isfinite(digit['yaw_radians']) or abs(digit['yaw_radians']) > np.pi):
            raise ValueError('Only independent head-image label, pixel anchor and visual angle may be retained')
        pixels = np.asarray(digit['anchor_image'], dtype=float)
        if pixels.shape != (2,) or not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels >= [640, 480]):
            raise ValueError('A reference pixel is inside the delivered native head frame')
        labels.append(digit['digit'])
    targets = np.asarray(value['target_centres'], dtype=float)
    if (len(set(labels)) != len(labels) or targets.shape != (len(digits), 2) or not np.isfinite(targets).all()
            or np.any(targets < 0) or np.any(targets >= [640, 480])):
        raise ValueError('Target count must independently agree; no missing object or forced label is filled')
    return deepcopy(value)


def capture(report, row, image):
    if row is None or len(report['digits']) != len(row) or not 2 <= len(row) <= 8:
        return None
    value = {'schema': 'riemann-public-number-head-reference-v1', 'semantics': SEMANTICS,
             'digits': [{name: deepcopy(digit[name]) for name in ('digit', 'camera', 'anchor_image', 'yaw_radians')}
                        for digit in report['digits']],
             'target_centres': [np.asarray(region['centroid']).tolist() for region in row],
             'image_sha256': hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}
    try:
        return validate(value)
    except (ValueError, TypeError, KeyError):
        return None


def retained(memory):
    for event in reversed(memory):
        feedback = event.get('execution_feedback', {})
        if feedback.get('actions_executed', 0) <= 0:
            continue
        if event.get('skill') not in ('number_scene_view', 'number_home'):
            return None
        if event.get('skill') == 'number_home' and event.get('request', {}).get('purpose') != 'inspect':
            return None
        value = feedback.get('image_measurements', {}).get('public_number_head_reference')
        if value is not None and any(phase.get('phase') == 'settle_public_view_after_robot_motion'
                                     and phase.get('actions', 0) > 0 for phase in feedback.get('phase_reports', [])):
            try:
                return validate(value)
            except (ValueError, TypeError, KeyError):
                return None
    return None


def compatible(value, current, row):
    if value is None or row is None:
        return False
    value = validate(value)
    centres = np.asarray([region['centroid'] for region in row])
    previous = np.asarray(value['target_centres'])
    if centres.shape != previous.shape or np.max(np.linalg.norm(centres - previous, axis=1)) > 4:
        return False
    labels = {digit['digit']: digit for digit in value['digits']}
    if not current['digits']:
        return False
    for digit in current['digits']:
        old = labels.get(digit['digit'])
        if old is None or np.linalg.norm(np.asarray(old['anchor_image']) - digit['anchor_image']) > 4:
            return False
    return True
