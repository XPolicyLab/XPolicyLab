from copy import deepcopy

import numpy as np


MOTION_TOOLS = {'reference_approach': 'approach', 'reference_contact': 'grasp_adjust',
    'reference_lift': 'lift', 'reference_carry': 'transfer', 'reference_align': 'approach',
    'reference_insert': 'preplace', 'reference_rotate': 'grasp_adjust', 'reference_press': 'preplace',
    'reference_retract': 'retreat', 'reference_handover': 'transfer'}
CLOCK_TOOLS = ('reference_close', 'reference_release', 'reference_hold')
SUBSTAGES = ('ready', 'acquire_receiver', 'fill', 'release_receiver', 'handover', 'insert', 'turn', 'thread',
    'press', 'close_lid', 'observe_demo', 'observe_reference', 'return', 'finish', 'delivered')


def initial_job():
    return {'substage': 'ready', 'held_side': None, 'receiver_anchor': None,
        'receiver_contact_pose': None, 'receiver_initial_pose': None, 'receiver_raised_command': None,
        'own_openings': {'left': None, 'right': None}, 'labels': [], 'identities': [],
        'observation_count': 0, 'empty_observations': 0, 'reference_left_view': False,
        'demonstration_started': False, 'logical_completed': [], 'thread_last_hole_pose': None,
        'thread_last_command_epoch': 0, 'thread_rotation_degrees': 0.}


def checked_pose(value):
    target = np.asarray(value, dtype=float)
    if (target.shape != (7,) or not np.isfinite(target).all() or np.max(abs(target)) > 4.
            or not .99 <= np.linalg.norm(target[3:]) <= 1.01):
        raise ValueError('A recipe target is a bounded current reference or own command')
    return target


def descriptor(value):
    if (not isinstance(value, dict) or set(value) != {'source', 'values'}
            or value['source'] != 'current_public_rgb_shape_colour_descriptor_v1'):
        raise ValueError('Identity memory contains only a current public image descriptor')
    values = np.asarray(value['values'], dtype=float)
    if values.shape != (28,) or not np.isfinite(values).all() or np.max(abs(values)) > 50:
        raise ValueError('Public image descriptors have a bounded explicit numeric shape')


def job(value):
    if (not isinstance(value, dict) or set(value) != set(initial_job()) or value['substage'] not in SUBSTAGES
            or value['held_side'] not in (None, 'left', 'right')
            or type(value['observation_count']) is not int or not 0 <= value['observation_count'] <= 192
            or type(value['empty_observations']) is not int or not 0 <= value['empty_observations'] <= 3
            or type(value['reference_left_view']) is not bool or type(value['demonstration_started']) is not bool):
        raise ValueError('Task progress is bounded same-episode public evidence, not an evaluator ledger')
    if (not isinstance(value['own_openings'], dict) or set(value['own_openings']) != {'left', 'right'}
            or any(opening is not None and (type(opening) not in (int, float) or not np.isfinite(opening)
                or not 0 <= opening <= 1) for opening in value['own_openings'].values())):
        raise ValueError('Peer holds retain only issued normalized own commands, never fingers or contact')
    for name in ('receiver_contact_pose', 'receiver_initial_pose', 'receiver_raised_command', 'thread_last_hole_pose'):
        if value[name] is not None:
            checked_pose(value[name])
    if value['receiver_anchor'] is not None:
        from skills.shared.verification.observed_rigid_transfer.implementation import anchor
        anchor(value['receiver_anchor'])
    if (type(value['thread_last_command_epoch']) is not int or value['thread_last_command_epoch'] < 0
            or type(value['thread_rotation_degrees']) not in (int, float)
            or not np.isfinite(value['thread_rotation_degrees']) or not 0 <= value['thread_rotation_degrees'] <= 1440):
        raise ValueError('Thread progress contains only acknowledged own rotation commands')
    for name, maximum in (('labels', 8), ('logical_completed', 32)):
        if (not isinstance(value[name], list) or len(value[name]) > maximum
                or any(not isinstance(item, str) or not 1 <= len(item) <= 100 for item in value[name])):
            raise ValueError('Semantic progress is a bounded list, not an external task identity')
    if not isinstance(value['identities'], list) or len(value['identities']) > 16:
        raise ValueError('Demonstration and reference memory remain bounded')
    for item in value['identities']:
        if (not isinstance(item, dict) or set(item) != {'description', 'descriptor', 'observed_command_count'}
                or not isinstance(item['description'], str) or not 1 <= len(item['description']) <= 100
                or type(item['observed_command_count']) is not int or item['observed_command_count'] < 0):
            raise ValueError('Reference identity comes only from an actual previous public frame')
        descriptor(item['descriptor'])
    return deepcopy(value)


def step(value):
    fields = {'tool', 'side', 'target', 'opening', 'commands', 'maximum_actions', 'visual_check'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['tool'] not in (*MOTION_TOOLS, *CLOCK_TOOLS) or value['side'] not in ('left', 'right')
            or type(value['opening']) not in (int, float) or not np.isfinite(value['opening'])
            or not 0 <= value['opening'] <= 1 or type(value['maximum_actions']) is not int
            or not 1 <= value['maximum_actions'] <= 240 or value['visual_check'] not in ('none', 'following', 'placement')):
        raise ValueError('An independent recipe tool declares a real bounded own motion or command clock')
    if value['tool'] in MOTION_TOOLS:
        checked_pose(value['target'])
        if value['commands'] is not None:
            raise ValueError('A motion tool cannot silently resample a gripper clock')
    elif (value['target'] is not None or not isinstance(value['commands'], list)
            or not 1 <= len(value['commands']) <= 40
            or any(type(opening) not in (int, float) or not np.isfinite(opening) or not 0 <= opening <= 1
                   for opening in value['commands']) or value['maximum_actions'] != len(value['commands'])):
        raise ValueError('A command tool contains an explicit already-sampled native clock')
    return deepcopy(value)


def recipe(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'source', 'steps', 'after', 'substage', 'complete_item', 'held_side_after',
            'thread_target_after', 'thread_rotation_after', 'reset_thread_after'}
            or value['source'] != 'current_public_reference_recipe'
            or not isinstance(value['steps'], list) or not 1 <= len(value['steps']) <= 48
            or value['after'] not in ('view', 'home') or value['substage'] not in SUBSTAGES
            or type(value['complete_item']) is not bool or value['held_side_after'] not in ('unchanged', None, 'left', 'right')
            or type(value['reset_thread_after']) is not bool):
        raise ValueError('A composition contains actual independent stages grounded in current public references')
    if value['thread_target_after'] is not None:
        checked_pose(value['thread_target_after'])
    if (value['thread_rotation_after'] is not None and (type(value['thread_rotation_after']) not in (int, float)
            or not np.isfinite(value['thread_rotation_after']) or not 0 <= value['thread_rotation_after'] <= 1440)):
        raise ValueError('The next own rotation accumulator is bounded')
    for item in value['steps']:
        step(item)
    return deepcopy(value)


def movement(tool, side, target, opening, maximum=160, visual_check='none'):
    return step({'tool': tool, 'side': side, 'target': np.asarray(target, dtype=float).tolist(),
        'opening': float(opening), 'commands': None, 'maximum_actions': maximum, 'visual_check': visual_check})


def command(tool, side, values, visual_check='none'):
    values = np.asarray(values, dtype=float).tolist()
    return step({'tool': tool, 'side': side, 'target': None, 'opening': float(values[-1]),
        'commands': values, 'maximum_actions': len(values), 'visual_check': visual_check})


def compose(steps, substage='ready', complete_item=True, after='view', held_side_after='unchanged',
        thread_target_after=None, thread_rotation_after=None, reset_thread_after=False):
    return recipe({'source': 'current_public_reference_recipe', 'steps': steps, 'after': after,
        'substage': substage, 'complete_item': complete_item, 'held_side_after': held_side_after,
        'thread_target_after': thread_target_after, 'thread_rotation_after': thread_rotation_after,
        'reset_thread_after': reset_thread_after})
