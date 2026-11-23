from martingale import self_play_constants as sp
# Copyright © 2026 Martingale. All Rights Reserved.
import math
import numpy as np

def check_insertion(face_corners_xyz, thickness_m, slot, *, measured_step, current_step, clearance_m=0.002, center_tolerance_m=0.005, max_release_height_m=0.01):
    """Allow tilted bread when centered over, and projected inside, the slot.

    ready_to_lower checks the whole extruded face footprint and its center.
    ready_to_release additionally requires its lowest bottom point no more than
    10 mm above the rim. Top-edge tilt alone is not a reason to reject a pose.
    This checks a vertical drop corridor, not friction or guaranteed seating.
    """
    face = np.asarray(face_corners_xyz, dtype=float)
    center = np.asarray(slot['center_xyz'], dtype=float)
    scalars = [thickness_m, clearance_m, center_tolerance_m, max_release_height_m, slot['width_m'], slot['length_m'], slot['rim_z'], slot['long_axis_deg']]
    if face.shape != (4, 3) or center.shape != (3,) or (not np.isfinite(face).all()) or (not np.isfinite(center).all()) or (not np.isfinite(scalars).all()):
        raise ValueError('Expected four finite, ordered 3D face corners and a finite slot fit')
    if min(thickness_m, slot['width_m'], slot['length_m']) <= 0 or min(clearance_m, center_tolerance_m, max_release_height_m) < 0:
        raise ValueError('Invalid dimensions, clearance or release tolerance')
    width = (face[1] - face[0] + (face[2] - face[3])) / 2
    up = (face[0] + face[1] - face[2] - face[3]) / 2
    normal = np.cross(width, up)
    if np.linalg.norm(width) < 1e-08 or np.linalg.norm(normal) < 1e-08 or up[2] <= 0:
        raise ValueError('Face corners must define a nondegenerate upright slice with top corners first')
    width /= np.linalg.norm(width)
    normal /= np.linalg.norm(normal)
    theta = math.radians(slot['long_axis_deg'])
    along = np.array([math.cos(theta), math.sin(theta)])
    across = np.array([-along[1], along[0]])
    flat_width = width[:2]
    yaw = math.degrees(math.acos(np.clip(abs(flat_width @ along) / max(np.linalg.norm(flat_width), 1e-12), 0, 1)))
    top_tilt = math.degrees(math.asin(np.clip(abs(width[2]), 0, 1)))
    face_lean = math.degrees(math.asin(np.clip(abs(normal[2]), 0, 1)))
    corners = np.concatenate([face + sign * normal * thickness_m / 2 for sign in [-1, 1]])
    xy = corners[:, :2] - center[:2]
    along_points, across_points = (xy @ along, xy @ across)
    end_margin = float(slot['length_m'] / 2 - np.max(np.abs(along_points)))
    side_margin = float(slot['width_m'] / 2 - np.max(np.abs(across_points)))
    center_offset = [float((p.min() + p.max()) / 2) for p in (along_points, across_points)]
    lowest_bottom_z = float(np.min(face[2:, 2]) - abs(normal[2]) * thickness_m / 2)
    drop_height = float(lowest_bottom_z - slot['rim_z'])
    reasons = []
    if measured_step != current_step:
        reasons.append('stale_bread_observation')
    if max((abs(v) for v in center_offset)) > center_tolerance_m:
        reasons.append('bread_not_centered_over_slot')
    if min(end_margin, side_margin) < clearance_m:
        reasons.append('insufficient_slot_clearance')
    ready_to_lower = not reasons
    if drop_height > max_release_height_m:
        reasons.append('bread_too_high_above_slot')
    return {'ready_to_lower': ready_to_lower, 'ready_to_release': not reasons, 'reasons': reasons, 'top_edge_tilt_deg': top_tilt, 'face_lean_deg': face_lean, 'slot_yaw_error_deg': yaw, 'end_margin_m': end_margin, 'side_margin_m': side_margin, 'center_offset_along_across_m': center_offset, 'lowest_bottom_height_above_rim_m': drop_height}

def guarded_release(robo, face_corners_xyz, thickness_m, slot, *, measured_step, arm='left', opening=sp.BREAD_RELEASE_OPENING):
    """Episode-script release gate; failed checks issue no motion/gripper action."""
    check = check_insertion(face_corners_xyz, thickness_m, slot, measured_step=measured_step, current_step=robo.state().step)
    release = robo.set_gripper(arm, robo.OPEN, opening=opening) if check['ready_to_release'] else None
    return {'check': check, 'release': release}
POLICY_GUIDANCE = "Bread may enter at an angle; a level top edge is not required.\nKeep the current grasping hand when it can place the bread over the slot. One\nunreachable correction pose does not establish that a handoff is needed. Center\nthe BREAD's projected outline over the slot, not just the gripper TCP. Use fresh\nobserved bread geometry and fit its projected outline inside the opening.\nCall check_insertion with corresponding 3D face corners, thickness and the fitted slot.\nLower only when ready_to_lower is true. Use guarded_release for the final opening.\nRelease close to the rim; both bottom corners need not already be inside. If a\ncheck fails, retain the grasp, adjust its position or obtain a better observation.\nDo not rotate the bread or hand it off merely to make its top edge level.\nRe-measure after motion; do not certify orientation using a plane\nassumed from the gripper's commanded angles. The release guard applies only to\ncalls routed through guarded_release; the ordinary set_gripper API is unchanged."
