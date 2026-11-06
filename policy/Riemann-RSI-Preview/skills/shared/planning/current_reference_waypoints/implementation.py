import math

import numpy as np
from scipy.spatial.transform import Rotation

from skills.shared.geometry.closing_direction_orientation.implementation import closing_orientation
from skills.shared.geometry.current_reference_frame_transfer.implementation import run as transfer
from skills.shared.geometry.wrist_contact_pose.implementation import run as wrist_pose
from skills.shared.planning.surface_grasp_sections.implementation import candidate_sections
from skills.shared.verification.observed_rigid_transfer.implementation import scene


def run(api, source, destination, target_outline, observed_support, forward=.155, attempt=0, side='auto',
        tilt_degrees=None, approach_height=.055, lift_height=.15):
    if (not math.isfinite(approach_height) or not 0. < approach_height <= .2
            or not math.isfinite(lift_height) or not 0. < lift_height <= .2):
        raise ValueError('Own approach and lift clearances use bounded declared motion parameters')
    surface, anchor = source['surface'], source['anchor']
    plane = observed_support['height_estimate_m']
    if side == 'auto':
        side = min(('left', 'right'), key=lambda arm: np.linalg.norm(api.targets()[arm + '_ee_pose'][:3] - anchor['center_m']))
    reference = np.asarray(anchor['reference_pose'], dtype=float)
    rotation = Rotation.from_quat(np.roll(reference[3:], -1)).as_matrix()
    sections = candidate_sections(surface['points_xyz'], surface['rgb_samples'], plane, surface['height_estimate_m'])
    if not sections:
        return None
    quality, fraction, band, point, width, minimum_z, maximum_z, level = sections[attempt % min(3, len(sections))]
    closing_yaw = math.atan2(rotation[1, 1], rotation[0, 1])
    tilt = (25 if attempt else 15) if tilt_degrees is None else tilt_degrees
    grasp_rotation = closing_orientation(closing_yaw, tilt, side)
    contact = np.r_[point, max(plane + .002, (minimum_z + maximum_z) * .5), np.roll(Rotation.from_matrix(grasp_rotation).as_quat(), 1)]
    contact_wrist = wrist_pose(contact[:3], contact[2], forward, contact[3:])
    destination = np.asarray(destination, dtype=float)
    mapped = np.asarray(transfer(reference, destination, contact_wrist)['pose'], dtype=float)
    pregrasp, lift = contact_wrist.copy(), contact_wrist.copy()
    pregrasp[2] += approach_height
    lift[2] += lift_height
    carry, align, place, retract = mapped.copy(), mapped.copy(), mapped.copy(), mapped.copy()
    carry[2] += .135
    carry[3:] = contact_wrist[3:]
    align[2] += .135
    place[2] += .005
    retract[2] += .07
    return scene({'source_anchor': anchor, 'source_reference': reference.tolist(), 'target_reference': destination.tolist(),
        'target_outline': np.asarray(target_outline).tolist(), 'support_memory': observed_support,
        'side': side, 'matching_confidence': min(anchor['coverage']), 'observation_epoch': api.issued_action_count(),
        'route': {name: target.tolist() for name, target in (('pregrasp', pregrasp), ('contact', contact_wrist),
            ('lift', lift), ('carry', carry), ('align', align), ('place', place), ('retract', retract))}})
