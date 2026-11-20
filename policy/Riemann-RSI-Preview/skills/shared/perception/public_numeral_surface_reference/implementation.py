"""Retain independently measured numeral surfaces only across open, camera-only commands."""

from copy import deepcopy
import math

import numpy as np

from skills.shared.perception.public_camera_projection.implementation import points_on_plane


SEMANTICS = 'same_episode_independent_public_numeral_surfaces_before_contact'
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def validate(value):
    """Validate a bounded historical public measurement dictionary; return a detached copy.

    Surface XYZ and appearance-free geometry are (3,), row pixels are (N, 2), and each
    own robot base pose is (7,) in XYZ/WXYZ order. No target or missing numeral is inferred.
    """
    fields = {'schema', 'semantics', 'surfaces', 'support_reference_m', 'target_centres', 'robot_bases'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'riemann-public-numeral-surface-reference-v1'
            or value['semantics'] != SEMANTICS):
        raise ValueError('Numeral memory contains only bounded independent public measurements')
    reference = value['support_reference_m']
    row = np.asarray(value['target_centres'], dtype=float)
    surfaces = value['surfaces']
    if (type(reference) not in (int, float) or not math.isfinite(reference) or abs(reference) > 3
            or row.ndim != 2 or row.shape[1:] != (2,) or not 2 <= len(row) <= 8
            or not np.isfinite(row).all() or np.any(row < 0) or np.any(row >= [640, 480])
            or not isinstance(surfaces, list) or not 1 <= len(surfaces) <= len(row)):
        raise ValueError('Retain only a partial observed numeral family and its actual public row')
    bases = value['robot_bases']
    if not isinstance(bases, dict) or set(bases) != {'left', 'right'}:
        raise ValueError('Surface memory requires both delivered own-robot base poses')
    for pose in bases.values():
        pose = np.asarray(pose, dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all() or np.max(np.abs(pose[:3])) > 3
                or not .99 <= np.linalg.norm(pose[3:]) <= 1.01):
            raise ValueError('Own base poses are finite public XYZ/WXYZ vectors')
    required = {'digit', 'surface_xyz', 'estimated_grasp_span_m', 'yaw_radians',
                'public_feature_height_interval_m', 'maximum_ray_separation_m', 'cameras', 'height_semantics'}
    labels = []
    for surface in surfaces:
        if (not isinstance(surface, dict) or not required <= set(surface)
                or not set(surface) <= required | {'current_relocalization_source', 'current_relocalization_cameras'}):
            raise ValueError('A saved numeral has only its independently observed public surface fields')
        label = surface['digit']
        point = np.asarray(surface['surface_xyz'], dtype=float)
        interval = np.asarray(surface['public_feature_height_interval_m'], dtype=float)
        cameras = surface['cameras']
        if (type(label) is not int or not 0 <= label <= 9 or label in labels
                or point.shape != (3,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 3
                or type(surface['estimated_grasp_span_m']) not in (int, float)
                or not 0 < surface['estimated_grasp_span_m'] <= .08
                or abs(point[2] - reference - surface['estimated_grasp_span_m']) > 1e-6
                or not math.isfinite(surface['yaw_radians']) or abs(surface['yaw_radians']) > math.pi
                or interval.shape != (2,) or not np.isfinite(interval).all()
                or not reference < interval[0] <= point[2] <= interval[1] <= reference + .08
                or not 0 <= surface['maximum_ray_separation_m'] <= .004
                or not isinstance(cameras, list) or not 2 <= len(cameras) == len(set(cameras)) <= 3
                or not set(cameras) <= set(CAMERAS)
                or surface['height_semantics'] not in (
                    'current_visual_feature_above_public_mount_reference_not_asset_height',
                    'public_static_scene_multi_view_correspondence_not_asset_height')):
            raise ValueError('Retained surfaces preserve their independently measured height and cameras')
        if 'current_relocalization_source' in surface and surface['current_relocalization_source'] != (
                'independent_current_public_camera_label_and_remembered_public_surface'):
            raise ValueError('Relocalization can refer only to an actual current public camera label')
        if 'current_relocalization_cameras' in surface:
            current = surface['current_relocalization_cameras']
            if (not isinstance(current, list) or not 1 <= len(current) == len(set(current)) <= 3
                    or not set(current) <= set(CAMERAS)):
                raise ValueError('Relocalization names only the currently corroborating cameras')
        labels.append(label)
    if np.ptp([surface['surface_xyz'][2] for surface in surfaces]) > .012:
        raise ValueError('Retained numeral features must remain on the same measured surface family')
    return deepcopy(value)


def capture(api, surfaces, row, reference):
    """Capture qualified surface dictionaries and row pixels (N, 2) from public API; return memory or None."""
    if not surfaces or row is None or not api.own_robot_bases_available():
        return None
    value = {'schema': 'riemann-public-numeral-surface-reference-v1', 'semantics': SEMANTICS,
             'surfaces': deepcopy(surfaces), 'support_reference_m': float(reference),
             'target_centres': [np.asarray(region['centroid']).tolist() for region in row],
             'robot_bases': {side: np.asarray(pose).tolist() for side, pose in api.own_robot_bases().items()}}
    try:
        return validate(value)
    except (ValueError, TypeError, KeyError):
        return None


def retained(api, row, reference):
    """Read own command history and current geometry; return compatible pre-contact memory or None.

    Current row pixels are (N, 2). Current base poses (7,) and open grippers must agree;
    any acknowledged non-camera action ends reuse, including learned recovery or contact.
    """
    if (row is None or not api.own_robot_bases_available()
            or any(float(api.targets()[side + '_ee_joint_state'][0]) < .9 for side in ('left', 'right'))):
        return None
    for event in reversed(api.command_memory()):
        feedback = event.get('execution_feedback', {})
        if feedback.get('actions_executed', 0) <= 0:
            continue
        if (event.get('skill') not in ('number_scene_view', 'number_home')
                or event.get('skill') == 'number_home' and event.get('request', {}).get('purpose') != 'inspect'):
            return None
        value = feedback.get('image_measurements', {}).get('public_number_surface_reference')
        if value is None:
            continue
        if not any(phase.get('phase') == 'settle_public_view_after_robot_motion' and phase.get('actions', 0) > 0
                   for phase in feedback.get('phase_reports', [])):
            return None
        value = validate(value)
        pixels = np.asarray([region['centroid'] for region in row])
        previous = np.asarray(value['target_centres'])
        if (pixels.shape != previous.shape or np.max(np.linalg.norm(pixels - previous, axis=1)) > 4
                or abs(value['support_reference_m'] - reference) > .005):
            return None
        for side, pose in api.own_robot_bases().items():
            pose, old = np.asarray(pose), np.asarray(value['robot_bases'][side])
            if np.linalg.norm(pose[:3] - old[:3]) > .001 or abs(np.dot(pose[3:], old[3:])) < 1 - 1e-6:
                return None
        return value
    return None


def combine(api, reports, fresh, previous):
    """Return (currently qualified surfaces, retained surfaces) from public RGB reads and memory.

    Each current pixel anchor is (2,); projected points are (K, 3). A remembered surface
    requires a current matching label within 8 mm at its independently measured height.
    Unseen measurements may remain dormant in memory, but never fill a current anchor.
    """
    current = {surface['digit']: deepcopy(surface) for surface in fresh}
    saved = {} if previous is None else {surface['digit']: surface for surface in validate(previous)['surfaces']}
    if any(label in saved and np.linalg.norm(np.asarray(surface['surface_xyz']) - saved[label]['surface_xyz']) > .008
           for label, surface in current.items()):
        saved = {}
    memory = dict(saved)
    memory.update(current)
    if memory and np.ptp([surface['surface_xyz'][2] for surface in memory.values()]) > .012:
        saved = {}
        memory = dict(current)
    for label, surface in saved.items():
        if label in current:
            continue
        evidence = []
        contradicted = False
        for camera in CAMERAS:
            records = [record for record in reports[camera]['digits'] if record['digit'] == label]
            if not records:
                continue
            if len(records) != 1:
                contradicted = True
                break
            points = points_on_plane(api, camera, [records[0]['anchor_image']], surface['surface_xyz'][2])
            if not np.isfinite(points).all() or np.linalg.norm(points[0] - surface['surface_xyz']) > .008:
                contradicted = True
                break
            evidence.append(camera)
        if contradicted:
            return list(fresh), list(fresh)
        if evidence:
            candidate = deepcopy(surface)
            candidate.update(height_semantics='public_static_scene_multi_view_correspondence_not_asset_height',
                             current_relocalization_source='independent_current_public_camera_label_and_remembered_public_surface',
                             current_relocalization_cameras=evidence)
            current[label] = candidate
    return sorted(current.values(), key=lambda value: -value['digit']), sorted(memory.values(), key=lambda value: -value['digit'])
