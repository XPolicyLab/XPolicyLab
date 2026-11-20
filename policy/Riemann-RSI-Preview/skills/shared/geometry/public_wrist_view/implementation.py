from functools import lru_cache
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

from skills.shared.motion.urdf_pose_kinematics.implementation import OwnURDFKinematics, matrix_pose, pose_matrix
from skills.shared.perception.public_camera_projection.implementation import points_on_plane
from skills.shared.perception.rgb_surface_geometry.implementation import camera_model


@lru_cache(maxsize=1)
def own_kinematics():
    return OwnURDFKinematics(Path(__file__).resolve().parents[4] / 'assets/robots/x5/X5A.urdf')


def joint_origins(kinematics, base, joints):
    joints = np.asarray(joints, dtype=float)
    transform = np.asarray(base, dtype=float).copy()
    if joints.shape != (6,) or transform.shape != (4, 4) or not np.isfinite(joints).all() or not np.isfinite(transform).all():
        raise ValueError('Visibility preflight uses only finite observed own joints and the declared own URDF')
    origins = [transform[:3, 3].copy()]
    for (origin, axis, unused_lower, unused_upper), angle in zip(kinematics.joints, joints):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        transform = transform @ origin @ rotation
        origins.append(transform[:3, 3].copy())
    return np.asarray(origins)


@lru_cache(maxsize=1)
def gripper_joint_envelope():
    path = Path(__file__).resolve().parents[4] / 'assets/robots/x5/X5A.urdf'
    document = ElementTree.parse(path).getroot()
    declared = {joint.attrib['name']: joint for joint in document.findall('joint')}
    envelope = []
    for name in ('joint7', 'joint8'):
        joint = declared[name]
        if joint.attrib['type'] != 'prismatic' or joint.find('parent').attrib['link'] != 'link6':
            raise ValueError('Use only the declared own gripper mount and complete joint-limit envelope')
        origin = np.fromstring(joint.find('origin').attrib['xyz'], sep=' ')
        axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
        limit = joint.find('limit')
        limits = [float(limit.attrib[key]) for key in ('lower', 'upper')]
        if (origin.shape != (3,) or axis.shape != (3,) or not np.isfinite(origin).all()
                or not np.isfinite(axis).all() or not np.isfinite(limits).all() or limits[0] > limits[1]):
            raise ValueError('The declared own gripper view envelope is incomplete')
        envelope.extend(origin + axis * value for value in limits)
    return np.asarray(envelope)


def gripper_segments(kinematics, base, joints):
    wrist = np.asarray(base) @ kinematics.forward(joints)
    points = gripper_joint_envelope() @ wrist[:3, :3].T + wrist[:3, 3]
    return [np.asarray([wrist[:3, 3], point]) for point in points]


def projected_overlap(api, chains, protected_pixels, plane_height, margin_m=.07):
    pixels = np.asarray(protected_pixels, dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= 48
            or not np.isfinite(pixels).all() or np.any(pixels < 0)
            or np.any(pixels[:, 0] >= 640) or np.any(pixels[:, 1] >= 480)
            or not math.isfinite(plane_height) or not .01 <= margin_m <= .08):
        raise ValueError('Protect bounded current public image anchors with an explicit own-view clearance margin')
    camera, focal = camera_model(api, 'cam_head')
    inverse = np.linalg.inv(camera)
    projected_surface = points_on_plane(api, 'cam_head', pixels, plane_height)
    surface_depth = -((projected_surface - camera[:3, 3]) @ camera[:3, :3])[:, 2]
    overlapped = np.zeros(len(pixels), dtype=bool)
    for chain in chains:
        chain = np.asarray(chain, dtype=float)
        if chain.ndim != 2 or chain.shape[1] != 3 or not 2 <= len(chain) <= 7 or not np.isfinite(chain).all():
            raise ValueError('Project bounded declared own-arm origins and the full own-gripper envelope')
        local = chain @ inverse[:3, :3].T + inverse[:3, 3]
        depths = -local[:, 2]
        for index in range(len(chain) - 1):
            if min(depths[index:index + 2]) <= .02:
                continue
            segment = local[index:index + 2]
            coordinates = np.column_stack((320 + focal * segment[:, 0] / depths[index:index + 2],
                                           240 - focal * segment[:, 1] / depths[index:index + 2]))
            direction = coordinates[1] - coordinates[0]
            fractions = np.clip((pixels - coordinates[0]) @ direction / max(float(direction @ direction), 1e-9), 0., 1.)
            distances = np.linalg.norm(pixels - coordinates[0] - fractions[:, None] * direction, axis=1)
            nearest_depth = 1. / ((1. - fractions) / depths[index] + fractions / depths[index + 1])
            overlapped |= (distances <= margin_m * focal / nearest_depth) & (nearest_depth < surface_depth - .01)
    return int(np.count_nonzero(overlapped))


def camera_target(focus, distance, offset, yaw):
    focus = np.asarray(focus, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if (focus.shape != (3,) or offset.shape != (2,) or not np.isfinite(focus).all() or not np.isfinite(offset).all()
            or np.max(np.abs(focus)) > 3 or not .2 <= distance <= .45 or np.linalg.norm(offset) > .5
            or not math.isfinite(yaw)):
        raise ValueError('A wrist view aims at a measured public image family, not a stored task layout')
    transform = np.eye(4)
    transform[:3, 3] = focus + distance * np.r_[offset, 1.]
    backwards = transform[:3, 3] - focus
    backwards /= np.linalg.norm(backwards)
    right = np.asarray([math.cos(yaw), math.sin(yaw), 0.])
    right -= backwards * float(right @ backwards)
    right /= np.linalg.norm(right)
    transform[:3, :3] = np.column_stack((right, np.cross(backwards, right), backwards))
    return transform


def validate_evidence(value):
    required = {'status', 'candidate_count', 'reachable_candidates', 'semantics',
                'collision_validation_bypassed', 'scene_state_used'}
    profiles = ('outward_oblique', 'vertical', 'outward_quarter_turn', 'front_oblique')
    if (not isinstance(value, dict) or not required.issubset(value)
            or value['status'] not in ('uncertain', 'view_candidate_selected')
            or value['candidate_count'] != 4 or type(value['candidate_count']) is not int
            or type(value['reachable_candidates']) is not int or not 0 <= value['reachable_candidates'] <= 4
            or value['semantics'] != 'own_URDF_view_preflight_not_a_contact_or_true_object_visibility_sensor'
            or value['collision_validation_bypassed'] is not False or value['scene_state_used'] is not False):
        raise ValueError('View preflight cannot inject environment state or pretend to be a visibility sensor')
    if value['status'] == 'uncertain':
        if (set(value) != required | {'reason'} or value['reachable_candidates'] != 0
                or value['reason'] != 'no_declared_camera_profile_is_own_IK_reachable'):
            raise ValueError('Unreachable view evidence has a closed non-action outcome')
    elif (set(value) != required | {'profile', 'projected_overlap_count'}
            or value['reachable_candidates'] < 1 or value['profile'] not in profiles
            or type(value['projected_overlap_count']) is not int or not 0 <= value['projected_overlap_count'] <= 48):
        raise ValueError('A selected view is one bounded declared own-robot hypothesis')
    return value


def run(api, side, focus, distance, protected_pixels, plane_height, preferred_profile=None):
    if side not in ('left', 'right') or not api.own_robot_bases_available():
        raise ValueError('A public wrist view needs the observed own-arm FK capability')
    kinematics = own_kinematics()
    bases = {arm: pose_matrix(value) for arm, value in api.own_robot_bases().items()}
    joints = api.own_joint_states()
    camera, unused_focal = camera_model(api, 'cam_' + side + '_wrist')
    camera_from_wrist = np.linalg.inv(pose_matrix(api.targets()[side + '_ee_pose'])) @ camera
    outward = -.32 if side == 'left' else .32
    profiles = [('outward_oblique', [outward, 0.], 0.), ('vertical', [0., 0.], 0.),
                ('outward_quarter_turn', [outward, 0.], -math.pi / 2 if side == 'left' else math.pi / 2),
                ('front_oblique', [.875 * outward, -.25], 0.)]
    if preferred_profile is not None and preferred_profile not in {profile[0] for profile in profiles}:
        raise ValueError('Prefer only a declared own-camera profile, not an invented view')
    candidates = []
    for index, (name, offset, yaw) in enumerate(profiles):
        target = matrix_pose(camera_target(focus, distance, offset, yaw) @ np.linalg.inv(camera_from_wrist))
        solution = kinematics.solve(bases[side], target, joints[side])
        if solution is None:
            continue
        chains = [joint_origins(kinematics, bases[arm], solution['joints'] if arm == side else joints[arm])
                  for arm in ('left', 'right')]
        for arm in ('left', 'right'):
            chains.extend(gripper_segments(kinematics, bases[arm], solution['joints'] if arm == side else joints[arm]))
        overlap = projected_overlap(api, chains, protected_pixels, plane_height)
        travel = float(np.linalg.norm(solution['joints'] - joints[side]))
        candidates.append({'profile': name, 'index': index, 'pose': target,
                           'projected_overlap_count': overlap, 'joint_travel_radians': travel})
    evidence = {'status': 'uncertain', 'candidate_count': len(profiles), 'reachable_candidates': len(candidates),
                'semantics': 'own_URDF_view_preflight_not_a_contact_or_true_object_visibility_sensor',
                'collision_validation_bypassed': False, 'scene_state_used': False}
    if not candidates:
        return {'pose': None, 'evidence': dict(evidence, reason='no_declared_camera_profile_is_own_IK_reachable')}
    selected = min(candidates, key=lambda value: (value['projected_overlap_count'],
        int(preferred_profile is not None and value['profile'] != preferred_profile),
        value['joint_travel_radians'], value['index']))
    evidence.update(status='view_candidate_selected', profile=selected['profile'],
                    projected_overlap_count=selected['projected_overlap_count'])
    return {'pose': selected['pose'], 'evidence': evidence}
