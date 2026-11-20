from copy import deepcopy
import math

import numpy as np

from skills.shared.geometry.planar_rigid_contact.implementation import checked_pose


ANCHOR_FIELDS = {'version', 'source', 'colour', 'initial_pose', 'destination_pose', 'support_plane_m',
                 'top_plane_m', 'maximum_ray_separation_m', 'initial_head_area'}
PROGRESS_FIELDS = {'version', 'stage', 'side', 'updates', 'route', 'shape_in_wrist', 'predicted_pose',
                   'last_observed_wrist', 'pose_source', 'independent_visual_confirmed', 'recontacts', 'issued_actions'}
ROUTE_FIELDS = {'side', 'turn', 'contact_wrist', 'high_wrist', 'low_joints', 'high_joints', 'joint_margin_rad'}


def anchor(value):
    if not isinstance(value, dict) or set(value) != ANCHOR_FIELDS or value['version'] != 1:
        raise ValueError('The T anchor has only declared public observation fields')
    if (value['source'] not in ('current_public_t_silhouettes_and_independent_stereo_planes',
                              'current_public_t_silhouette_on_same_episode_qualified_planes')
            or value['colour'] not in ('blue', 'red') or type(value['initial_head_area']) is not int
            or not 40 <= value['initial_head_area'] <= 46080):
        raise ValueError('A T anchor cannot contain hidden identity, template or private layout evidence')
    initial, destination = checked_pose(value['initial_pose']), checked_pose(value['destination_pose'])
    for name in ('support_plane_m', 'top_plane_m', 'maximum_ray_separation_m'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or abs(value[name]) > 3:
            raise ValueError('Planar surface hypotheses must be finite public measurements')
    if (not .003 <= value['top_plane_m'] - value['support_plane_m'] <= .04
            or not 0 <= value['maximum_ray_separation_m'] <= .004
            or abs(initial[2] - (value['top_plane_m'] + value['support_plane_m']) * .5) > 1e-6
            or abs(destination[2] - initial[2]) > 1e-6):
        raise ValueError('T height and command Z must remain tied to actual independent public planes')
    return deepcopy(value)


def route(value):
    if (not isinstance(value, dict) or set(value) not in (ROUTE_FIELDS, ROUTE_FIELDS | {'staging_destination_pose'})
            or value['side'] not in ('left', 'right') or value['turn'] not in (-1, 1)):
        raise ValueError('A contact route names one own arm and a declared closing direction')
    low, high = checked_pose(value['contact_wrist']), checked_pose(value['high_wrist'])
    if (float(np.linalg.norm(high[:2] - low[:2])) > 1e-8 or abs(high[2] - low[2] - .10) > 1e-8
            or not np.allclose(high[3:], low[3:], atol=1e-8)):
        raise ValueError('The mature approach is the declared relative vertical clearance')
    for name in ('low_joints', 'high_joints'):
        joints = np.asarray(value[name], dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all() or np.max(np.abs(joints)) > math.pi:
            raise ValueError('A contact route contains only finite own-arm joint solutions')
    if type(value['joint_margin_rad']) not in (int, float) or not math.isfinite(value['joint_margin_rad']) or not .08 <= value['joint_margin_rad'] <= math.pi:
        raise ValueError('Contact keeps the mature own-joint rotational reserve')
    if 'staging_destination_pose' in value:
        staged = checked_pose(value['staging_destination_pose'])
        if abs(staged[2] - (low[2] - .154)) > 1e-6:
            raise ValueError('An intermediate command stays on the independently qualified source plane')
    return deepcopy(value)


def active_destination(observed, current):
    observed, current = anchor(observed), progress(current)
    selected = (current['route'] or {}).get('staging_destination_pose')
    return checked_pose(observed['destination_pose'] if selected is None else selected)


def initial_progress():
    return {'version': 1, 'stage': 0, 'side': 'none', 'updates': 0, 'route': None,
            'shape_in_wrist': None, 'predicted_pose': None, 'last_observed_wrist': None,
            'pose_source': 'unavailable', 'independent_visual_confirmed': False, 'recontacts': 0, 'issued_actions': 0}


def progress(value):
    if not isinstance(value, dict) or set(value) != PROGRESS_FIELDS or value['version'] != 1:
        raise ValueError('Contact progress is an explicit bounded own-command receipt')
    if (type(value['stage']) is not int or not 0 <= value['stage'] <= 6
            or value['side'] not in ('none', 'left', 'right') or type(value['updates']) is not int or not 0 <= value['updates'] <= 100
            or type(value['recontacts']) is not int or not 0 <= value['recontacts'] <= 1
            or type(value['issued_actions']) is not int or not 0 <= value['issued_actions'] <= 640
            or type(value['independent_visual_confirmed']) is not bool
            or value['pose_source'] not in ('unavailable', 'current_public_planar_observation', 'own_wrist_contact_hypothesis_not_object_measurement')):
        raise ValueError('An own contact tracker cannot claim true pose, touch or official success')
    if value['route'] is not None:
        route(value['route'])
        if value['route']['side'] != value['side']:
            raise ValueError('The own arm must match its observed route')
    for name in ('shape_in_wrist', 'predicted_pose', 'last_observed_wrist'):
        if value[name] is not None:
            checked_pose(value[name])
    if value['stage'] >= 2 and (value['side'] == 'none' or value['route'] is None):
        raise ValueError('Physical contact stages require an actual qualified own route')
    if value['stage'] >= 4 and any(value[name] is None for name in ('shape_in_wrist', 'predicted_pose', 'last_observed_wrist')):
        raise ValueError('A sliding stage requires its own previously acknowledged attachment hypothesis')
    return deepcopy(value)
