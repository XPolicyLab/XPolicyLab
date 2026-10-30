"""Soft-rank measured wrist headings at a fixed measured grasp point."""
import math
import numpy as np
from coap.motion import down_quat, quat_matrix, quat_multiply, tcp_to_link


def _heading(quaternion):
    """Recover the horizontal closing heading used by down_quat."""
    rotation = quat_matrix(quaternion)
    return float(math.atan2(-rotation[0, 1], rotation[1, 1]))


def _tilted_quat(yaw, arm, point, degrees):
    """Keep the measured heading and reuse the ordinary source tilt."""
    base = np.array([-.30 if arm == 'left' else .30, -.45])
    direction = np.asarray(point[:2], float)-base
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    angle = math.radians(float(degrees))
    tilt = np.r_[math.cos(angle/2.),
                 np.array([direction[1], -direction[0], 0.])*math.sin(angle/2.)]
    return quat_multiply(tilt, down_quat(yaw))


def select_refined_pose(ctx, obj, arm, source_point, source_quat,
                        measured_point, measured_quat, lift_height):
    """Preserve the measured point and soft-rank contact/lift pose candidates."""
    point = np.asarray(measured_point, float).copy()
    refined_quat = np.asarray(measured_quat, float).copy()
    if not ctx.numeric_ik:
        return point, refined_quat, dict(method='measured_pose_without_numeric_ik')
    from coap.tasks.classify_r07_periodic_ik import PeriodicPostureKinematics
    solver = getattr(ctx, '_classify_source_periodic_ik', None)
    if solver is None:
        solver = PeriodicPostureKinematics(ctx.kinematics)
        ctx._classify_source_periodic_ik = solver
    seed = np.asarray(ctx.state[arm+'_arm_joint_state'], float)
    category = str(obj.get('category', obj.get('appearance_family', 'unknown')))
    degrees = 22. if category in ('garage', 'wooden_toy', 'action_camera') else 16.
    yaw = _heading(refined_quat)
    candidates = [('measured_quaternion', refined_quat, 0.),
                  ('measured_heading_source_tilt', _tilted_quat(yaw, arm, point, degrees), .05),
                  ('source_quaternion_at_measured_point', np.asarray(source_quat, float), .25)]
    records = []
    for name, quaternion, prior in candidates:
        heading_error = abs(math.atan2(math.sin(_heading(quaternion)-yaw),
                                      math.cos(_heading(quaternion)-yaw)))
        contact = solver.solve(arm, tcp_to_link(point, quaternion),
            current_joints=seed, position_tol=.002, rotation_tol=.020,
            max_starts=8, max_nfev=140)
        contact_joints = np.asarray(contact['joint_value'], float)
        lift = solver.solve(arm, tcp_to_link(point+[0., 0., lift_height], quaternion),
            current_joints=contact_joints, position_tol=.002, rotation_tol=.020,
            max_starts=8, max_nfev=140)
        cost = float(contact['objective']+.5*lift['objective']+prior+.4*heading_error)
        records.append(dict(name=name, cost=cost, quaternion=quaternion.tolist(),
            measured_heading_error_rad=heading_error,
            contact_position_error_m=float(contact['position_error']),
            contact_rotation_error_rad=float(contact['rotation_error']),
            contact_joint_value=contact['joint_value'],
            contact_path_cost=float(contact['posture_cost']),
            lift_position_error_m=float(lift['position_error']),
            lift_rotation_error_rad=float(lift['rotation_error']),
            lift_joint_value=lift['joint_value'],
            lift_path_cost=float(lift['posture_cost'])))
    selected = min(records, key=lambda item: item['cost'])
    evidence = dict(method='measured_point_contact_lift_pose_soft_ranking',
                    selected=selected['name'], measured_point=point.tolist(),
                    source_point=np.asarray(source_point, float).tolist(),
                    measured_point_changed=False, lift_height=float(lift_height),
                    candidates=records, runtime_rejection_added=False)
    return point, np.asarray(selected['quaternion'], float), evidence
