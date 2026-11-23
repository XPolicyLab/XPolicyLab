from martingale import self_play_constants as sp
# Copyright © 2026 Martingale. All Rights Reserved.
import math
import numpy as np
from martingale.hosts import robodojo_kinematics as kin

def preview_body_pickup(plan, state, pregrasp_height=sp.MUG_PREGRASP_HEIGHT_M):
    if not plan.get('valid') or plan.get('part') != 'body':
        raise ValueError('An accepted body grasp plan is required')
    height = float(pregrasp_height)
    if not math.isfinite(height) or not 0.04 <= height <= 0.12:
        raise ValueError('pregrasp_height must be between 0.04 and 0.12 m')
    contact = np.asarray(plan['contact_xyz'], float)
    closing = np.asarray(plan['closing_axis_xyz'], float)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        raise ValueError('Invalid contact position')
    if closing.shape != (3,) or not np.isfinite(closing).all() or (not np.isclose(np.linalg.norm(closing), 1)):
        raise ValueError('Invalid closing direction')
    pregrasp = contact + [0, 0, height]
    accepted, rejected = ([], [])
    for arm in ('left', 'right'):
        current = np.asarray(state.arm(arm).joints, float)
        for pitch in (90.0, 75.0, 60.0):
            for yaw in (float(plan['yaw_deg']), float(plan['yaw_deg']) + 180.0):
                rotation = kin.orientation(pitch, yaw, 0.0)
                assert abs(np.dot(rotation[:, 1], closing)) > 1 - 1e-06
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = (rotation, pregrasp)
                solution = kin.best_solution(arm, pose, current, extra_seeds=[np.zeros(6)])
                row = dict(arm=arm, pitch=pitch, yaw=yaw, roll=0.0, pregrasp_xyz=pregrasp.tolist(), contact_xyz=contact.tolist(), gap_m=float(plan['gap_m']))
                if not solution.ok or solution.collides:
                    rejected.append({**row, 'reason': 'pregrasp not accurately reachable', 'position_error_m': solution.pos_err_m, 'orientation_error_deg': solution.rot_err_deg})
                    continue
                pose[:3, 3] = contact
                descent, why = kin.line_path(arm, solution.q, pose)
                if why:
                    rejected.append({**row, 'reason': why})
                    continue
                travel = float(np.max(np.abs(np.asarray(solution.q) - current)))
                accepted.append({**row, 'max_joint_travel_rad': travel, 'descent_waypoints': len(descent), 'scene_clearance_verified': False, 'tilted_approach': pitch != 90.0})
    accepted.sort(key=lambda row: (row['tilted_approach'], row['max_joint_travel_rad'], row['descent_waypoints']))
    return dict(candidates=accepted, rejected=rejected, episode_actions_used=0, note='Reachability preview only. Inspect the complete finger/palm corridor and other arm before executing. A tilted pose is not an approved grasp merely because IK succeeds. Keep the original physical gap, guarded descent and separate test-lift checks.')
