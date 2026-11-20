from copy import deepcopy
import math

import numpy as np


STAGES = ('grasp', 'lift', 'upright', 'seat')


def vector(value, size, limit=3.):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or np.max(np.abs(result)) > limit:
        raise ValueError('Insertion receipts contain bounded public measurements or own commands')
    return result.tolist()


def validate_source(value):
    if not isinstance(value, dict) or set(value) != {'grasp', 'axis', 'cap'}:
        raise ValueError('A source is a current visible cap and horizontal body hypothesis')
    result = {name: vector(value[name], 3) for name in value}
    if abs(result['axis'][2]) > 1e-6 or not .99 <= np.linalg.norm(result['axis']) <= 1.01:
        raise ValueError('A source axis requires current horizontal unit geometry')
    return result


def validate_anchor(value):
    fields = {'sources', 'targets', 'source_surface_m', 'target_surface_m', 'support_plane_m', 'target_radius_m'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('An insertion anchor contains only initial public scene hypotheses')
    result = deepcopy(value)
    if not isinstance(value['sources'], list) or not 2 <= len(value['sources']) <= 8:
        raise ValueError('Use a bounded currently observed source family')
    result['sources'] = [validate_source(source) for source in value['sources']]
    if not isinstance(value['targets'], list) or len(value['targets']) != len(value['sources']):
        raise ValueError('Public receiving goals must match the bounded source slots')
    result['targets'] = [vector(target, 3) for target in value['targets']]
    for name in ('source_surface_m', 'target_surface_m', 'support_plane_m', 'target_radius_m'):
        if type(value[name]) not in (float, int) or not math.isfinite(value[name]) or abs(value[name]) > 3:
            raise ValueError('Scene hypotheses must be finite public geometry')
    if not .005 <= value['target_radius_m'] <= .12:
        raise ValueError('Receiving size must be a bounded current visual measurement')
    return result


def validate_route(value):
    if not isinstance(value, dict) or set(value) != {'fingertip_target', 'quaternion', 'joint_target'}:
        raise ValueError('A route contains only public receiving geometry and an own-joint command')
    result = {'fingertip_target': vector(value['fingertip_target'], 3),
              'quaternion': vector(value['quaternion'], 4),
              'joint_target': vector(value['joint_target'], 6, 64.)}
    if not .99 <= np.linalg.norm(result['quaternion']) <= 1.01:
        raise ValueError('An upright route requires a unit own-tool orientation')
    return result


def empty_progress(order):
    if not isinstance(order, (list, tuple)) or not order:
        raise ValueError('Initial progress requires a bounded observed-slot order')
    value = {'order': list(order), 'cursor': 0, 'stage': 0, 'source': None, 'route': None,
             'target_index': order[0], 'inspection_done': False, 'recovery_used': False,
             'final_stage': 0, 'settled': False}
    return validate_progress(value)


def validate_progress(value):
    fields = {'order', 'cursor', 'stage', 'source', 'route', 'target_index',
              'inspection_done', 'recovery_used', 'final_stage', 'settled'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Insertion progress records only a bounded own-command sequence')
    order = value['order']
    if (not isinstance(order, list) or not 2 <= len(order) <= 8
            or any(type(slot) is not int for slot in order) or sorted(order) != list(range(len(order)))):
        raise ValueError('Own source order must be a bounded observed-slot permutation')
    for name, maximum in (('cursor', len(order) + 1), ('stage', 3), ('target_index', len(order) - 1), ('final_stage', 3)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('A command cursor is finite and cannot skip outside its workflow')
    if any(type(value[name]) is not bool for name in ('inspection_done', 'recovery_used', 'settled')):
        raise ValueError('Own command flags are explicit booleans, not contact or success')
    if value['cursor'] > len(order) and not value['recovery_used']:
        raise ValueError('Only one explicitly recorded recovery can exceed the initial source count')
    if value['cursor'] >= len(order) and value['stage'] > 0 and not value['recovery_used']:
        raise ValueError('Extra insertion stages require an actual recorded recovery command')
    result = deepcopy(value)
    if value['source'] is not None:
        result['source'] = validate_source(value['source'])
    if value['route'] is not None:
        result['route'] = validate_route(value['route'])
    if (value['stage'] == 0 and (value['source'] is not None or value['route'] is not None)
            or value['stage'] > 0 and value['source'] is None
            or (value['stage'] == 3) != (value['route'] is not None)
            or value['final_stage'] and (value['cursor'] < len(order) or value['stage'] != 0)
            or value['settled'] != (value['final_stage'] == 3)):
        raise ValueError('Own insertion handoffs must agree with the actually completed stage')
    return result


def advance(value, stage, source=None, route=None, target_index=None):
    result = validate_progress(value)
    if result['final_stage'] or result['cursor'] > len(result['order']) or stage != STAGES[result['stage']]:
        raise ValueError('Advance only the current actually executed insertion tool')
    if stage == 'grasp':
        result['source'] = validate_source(source)
        if target_index is not None:
            result['target_index'] = target_index
        if result['cursor'] == len(result['order']):
            if result['recovery_used']:
                raise ValueError('The declared single recovery cannot be repeated')
            result['recovery_used'], result['inspection_done'] = True, True
    if stage == 'upright':
        result['route'] = validate_route(route)
    if stage == 'seat':
        result.update(cursor=result['cursor'] + 1, stage=0, source=None, route=None)
        result['target_index'] = result['order'][result['cursor']] if result['cursor'] < len(result['order']) else 0
    else:
        result['stage'] += 1
    return validate_progress(result)
