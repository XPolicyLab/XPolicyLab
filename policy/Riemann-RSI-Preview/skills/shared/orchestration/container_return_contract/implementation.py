from copy import deepcopy

import numpy as np


MINIMUM = {'cover_first': 23, 'cover_second': 18, 'cover_third': 18,
           'uncover_red': 18, 'uncover_green': 18, 'uncover_blue': 18,
           'home_left': 2, 'home_right': 2, 'settle': 13}


def validate_anchor(value, labels=('red', 'green', 'blue')):
    if (not isinstance(value, dict) or set(value) != {'cups', 'blocks', 'order', 'support_plane_m', 'cap_geometry'}
            or not isinstance(value['cups'], list) or len(value['cups']) != len(labels)
            or not isinstance(value['blocks'], dict) or set(value['blocks']) != set(labels)
            or not isinstance(value['order'], list) or sorted(value['order']) != sorted(labels)):
        raise ValueError('A container anchor contains only initially public return homes and labelled goals')
    plane = float(value['support_plane_m'])
    if not np.isfinite(plane):
        raise ValueError('A container support plane must be a finite public estimate')
    geometry = value['cap_geometry']
    if (not isinstance(geometry, dict) or set(geometry) != {'height_m', 'size_m', 'observed_head_indices', 'wrist_views'}
            or type(geometry['height_m']) not in (int, float) or not np.isfinite(geometry['height_m'])
            or not .005 < geometry['height_m'] - plane <= .3
            or type(geometry['size_m']) not in (int, float) or not .005 <= geometry['size_m'] <= .12
            or not isinstance(geometry['observed_head_indices'], list)
            or not 2 <= len(geometry['observed_head_indices']) <= len(labels)
            or geometry['observed_head_indices'] != sorted(set(geometry['observed_head_indices']))
            or any(type(index) is not int or not 0 <= index < len(labels) for index in geometry['observed_head_indices'])
            or not isinstance(geometry['wrist_views'], list)
            or geometry['wrist_views'] not in [['cam_left_wrist'], ['cam_right_wrist'], ['cam_left_wrist', 'cam_right_wrist']]
            or len(geometry['observed_head_indices']) < len(labels) and len(geometry['wrist_views']) < 2):
        raise ValueError('Container cap geometry requires bounded same-episode public multi-member evidence')
    for point in value['cups'] + [value['blocks'][label] for label in labels]:
        coordinates = np.asarray(point, dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all() or np.max(np.abs(coordinates)) > 3:
            raise ValueError('A container goal is a bounded public observation, not an object record')
    return deepcopy(value)


def validate_release(value):
    if not isinstance(value, dict) or value.get('status') not in ('ambiguous', 'visible_near_commanded_goal', 'not_at_visible_goal'):
        raise ValueError('Release evidence must describe only current public visibility')
    fields = {'status', 'contact_claimed', 'official_success_claimed'}
    if value['status'] != 'ambiguous':
        fields.add('head_distance_fraction')
        distance = value.get('head_distance_fraction')
        if type(distance) not in (int, float) or not np.isfinite(distance) or not 0 <= distance <= 2:
            raise ValueError('Release proximity is a bounded image measurement')
    if set(value) != fields or value['contact_claimed'] is not False or value['official_success_claimed'] is not False:
        raise ValueError('Release imagery cannot claim real contact or official success')
    return value
