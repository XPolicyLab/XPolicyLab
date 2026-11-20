from copy import deepcopy
import numpy as np
from skills.shared.perception.public_dense_plane_alignment_observed_start.implementation import qualified_alignments, run as current_support
SOURCE = 'independent_public_rgb_support_before_contact'
CURRENT_SOURCE = 'current_independent_public_rgb_support_geometry'

def validate(value):
    return validated_support(value, SOURCE)

def validate_geometry(value):
    if not isinstance(value, dict) or value.get('source') not in (SOURCE, CURRENT_SOURCE):
        raise ValueError('Support geometry must identify its actual public observation context')
    return validated_support(value, value['source'])

def validated_support(value, expected_source):
    fields = {'height_estimate_m', 'height_uncertainty_m', 'camera_alignments', 'own_base_poses', 'source'}
    if not isinstance(value, dict) or set(value) != fields or value['source'] != expected_source:
        raise ValueError('View support memory contains only previously qualified public evidence')
    for name, lower, upper in (('height_estimate_m', -3.0, 3.0), ('height_uncertainty_m', 0.0, 0.006)):
        number = value[name]
        if type(number) not in (int, float) or not np.isfinite(number) or (not lower <= number <= upper):
            raise ValueError('View support memory has bounded observed height and uncertainty')
    records = value['camera_alignments']
    fields = {'camera', 'height_estimate_m', 'correlation', 'current_matching_pixels', 'height_uncertainty_m', 'peak_unique', 'qualified'}
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError('A retained support must retain both independent camera alignment receipts')
    for item in records:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError('Alignment receipts cannot carry saved layout or ground-truth state')
        if type(item['qualified']) is not bool or type(item['peak_unique']) is not bool or type(item['current_matching_pixels']) is not int or (not 0 <= item['current_matching_pixels'] <= 76800) or (type(item['correlation']) not in (int, float)) or (not -1.0 <= item['correlation'] <= 1.0) or (type(item['height_estimate_m']) not in (int, float)) or (not -3.0 <= item['height_estimate_m'] <= 3.0) or (type(item['height_uncertainty_m']) not in (int, float)) or (not 0.0 <= item['height_uncertainty_m'] <= 0.01):
            raise ValueError('Retained alignments use only finite public matching metrics')
    if sorted((item['camera'] for item in records)) != ['cam_left_wrist', 'cam_right_wrist']:
        raise ValueError('A retained plane cannot replace an independent camera with a duplicate')
    accepted, qualification = qualified_alignments(records)
    if not accepted or qualification is None or abs(np.mean([item['height_estimate_m'] for item in accepted]) - value['height_estimate_m']) > 0.003:
        raise ValueError('A retained support must have met the unchanged original independent qualification')
    bases = value['own_base_poses']
    if not isinstance(bases, dict) or set(bases) != {'left', 'right'}:
        raise ValueError('View support is anchored only to its two observed own robot bases')
    for pose in bases.values():
        values = np.asarray(pose, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all() or np.max(np.abs(values[:3])) > 3.0 or (not 0.99 <= np.linalg.norm(values[3:]) <= 1.01):
            raise ValueError('Support anchors contain only bounded own robot poses')
    return deepcopy(value)

def run(api, previous=None, passive_view=False):
    if type(passive_view) is not bool:
        raise ValueError('Before-contact view retention is an explicit boolean execution condition')
    observed = current_support(api)
    old = validate(previous) if previous is not None else None
    available = passive_view and api.own_robot_bases_available()
    opened = available and all((float(api.targets()[side + '_ee_joint_state'][0]) >= 0.9 for side in ('left', 'right')))
    bases = {side: pose.tolist() for side, pose in api.own_robot_bases().items()} if available else None
    if old is not None and (not opened or any((np.linalg.norm(np.asarray(bases[side][:3]) - old['own_base_poses'][side][:3]) > 0.004 or abs(np.dot(bases[side][3:], old['own_base_poses'][side][3:])) < 0.999 for side in ('left', 'right')))):
        old = None
    safe = opened
    if observed['status'] == 'observed':
        plane = observed['plane']
        if old is not None and abs(plane['distance_m'] - old['height_estimate_m']) > 0.006:
            return {'support': {'status': 'uncertain', 'plane': None}, 'memory': None, 'reason': 'current_support_contradicts_previous_public_plane'}
        record = {'height_estimate_m': plane['distance_m'], 'height_uncertainty_m': plane['height_uncertainty_m'], 'camera_alignments': observed['camera_alignment'], 'own_base_poses': bases, 'source': SOURCE} if safe else None
        return {'support': observed, 'memory': validate(record) if record is not None else None, 'reason': 'current_independent_public_support'}
    if old is None:
        return {'support': observed, 'memory': None, 'reason': 'no_qualified_before_contact_support'}
    conflicting = [item for item in observed.get('camera_alignment', []) if item.get('qualified') is True and item.get('peak_unique') is True and (abs(item['height_estimate_m'] - old['height_estimate_m']) > 0.006)]
    if conflicting:
        return {'support': {'status': 'uncertain', 'plane': None}, 'memory': None, 'reason': 'current_qualified_camera_contradicts_previous_public_plane'}
    plane = {'normal': [0.0, 0.0, 1.0], 'distance_m': old['height_estimate_m'], 'height_uncertainty_m': old['height_uncertainty_m'], 'semantics': 'previous_independently_observed_support_retained_only_during_passive_public_view'}
    return {'support': {'status': 'observed', 'plane': plane, 'official_success_claimed': False}, 'memory': old, 'reason': 'qualified_public_plane_retained_before_any_contact'}
